"""Full-information constructive RL environment for the benchmark.

The environment uses the common evaluator and finite action decoder with
normalized route/contention features. No action is truncated or schedule repaired.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from adaptive_feedback_llm.evaluator.core import evaluate_schedule
from adaptive_feedback_llm.models.domain import EvaluatorScenario, Schedule, TrafficClass
from adaptive_feedback_llm.physics.core import rate_per_sru_bps
from adaptive_feedback_llm.rl.decoder import (
    build_options,
    decision_for_option,
    evaluation_weight,
    increment,
    rejected_schedule,
)


PROFILE_ID = "full-scheduling"
MAX_NODES = 32
MAX_FLOWS = 18
MAX_CAPACITY_SRUS = 8
PATHS_PER_FLOW = 5
MAX_COMPLETE_ACTIONS = 4096
EDGE_FEATURES = 8
FLOW_FEATURES = 11
ACTION_FEATURES = 11
GLOBAL_FEATURES = 6


def validate_scenario_envelope(scenario: EvaluatorScenario) -> None:
    if len(scenario.node_ids) > MAX_NODES or len(scenario.flows) > MAX_FLOWS:
        raise ValueError("scenario exceeds scheduling 32-node/18-flow envelope")
    if any(link.capacity_srus > MAX_CAPACITY_SRUS for link in scenario.links):
        raise ValueError("scenario exceeds scheduling 8-SRU edge envelope")
    if any(len(catalog.paths) != PATHS_PER_FLOW for catalog in scenario.path_catalogs):
        raise ValueError("scheduling RL requires exactly five catalog paths per flow")
    if not scenario.links or not scenario.flows:
        raise ValueError("scheduling RL requires nonempty links and flows")
    maximum_hops = max(path.hop_count for catalog in scenario.path_catalogs for path in catalog.paths)
    if maximum_hops > len(scenario.node_ids) - 1:
        raise ValueError("path exceeds simple-path node envelope")


class FullSchedulingEnv:
    def __init__(self, scenario: EvaluatorScenario):
        validate_scenario_envelope(scenario)
        self.scenario = scenario
        self.work: dict[str, int] = {}
        self.table = build_options(scenario, self.work)
        if len(self.table) > MAX_COMPLETE_ACTIONS:
            raise ValueError("complete scheduling action table exceeds registered 4096-action bound")
        self.normalizer = 2.0 * math.fsum(flow.priority for flow in scenario.flows)
        self.capacity = np.asarray([edge.capacity_srus for edge in scenario.links], dtype=np.int64)
        self.requirements = np.zeros((len(self.table), len(scenario.links)), dtype=np.int64)
        self.action_flow = np.asarray([option.flow_index for option in self.table], dtype=np.int64)
        for action_index, option in enumerate(self.table):
            self.requirements[action_index, list(option.edges)] = option.allocations
        self.incidence = (
            self.requirements.astype(np.float32) / self.capacity.astype(np.float32)[None, :]
        )
        lookup = {link.edge: index for index, link in enumerate(scenario.links)}
        self.occurrences = np.zeros(len(scenario.links), dtype=np.float32)
        self.rank0_occurrences = np.zeros(len(scenario.links), dtype=np.float32)
        for catalog in scenario.path_catalogs:
            for path_index, path in enumerate(catalog.paths):
                for edge in path.edges:
                    self.occurrences[lookup[edge]] += 1
                    if path_index == 0:
                        self.rank0_occurrences[lookup[edge]] += 1
        self.rates = np.asarray(
            [
                rate_per_sru_bps(edge.sinr_db, scenario.physics.sru_bandwidth_hz)
                for edge in scenario.links
            ],
            dtype=np.float32,
        )
        self.max_catalog_hops = max(
            path.hop_count for catalog in scenario.path_catalogs for path in catalog.paths
        )
        self.mean_catalog_hops = float(
            np.mean(
                [path.hop_count for catalog in scenario.path_catalogs for path in catalog.paths]
            )
        )
        self.max_total_option_srus = max(
            1, max((sum(option.allocations) for option in self.table), default=1)
        )
        self.max_option_level = max(1, max((option.level for option in self.table), default=1))
        self.feasible_path_counts = np.zeros(len(scenario.flows), dtype=np.float32)
        for flow_index in range(len(scenario.flows)):
            self.feasible_path_counts[flow_index] = len(
                {
                    option.path_id
                    for option in self.table
                    if option.flow_index == flow_index and option.path_id >= 0
                }
            )
        self._weight = evaluation_weight(scenario)
        self.reset()

    @property
    def schedule(self) -> Schedule:
        return Schedule(flows=tuple(self._decisions))

    def reset(self) -> dict[str, np.ndarray]:
        self.remaining = self.capacity.copy()
        self.processed = np.zeros(len(self.scenario.flows), dtype=bool)
        self.served = np.zeros(len(self.scenario.flows), dtype=bool)
        self._decisions = list(rejected_schedule(self.scenario).flows)
        self.current_j = 0.0
        self.terminated = False
        self.steps = 0
        self.reward_sum = 0.0
        self._mask = self._calculate_mask()
        return self.observation()

    def _calculate_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.table), dtype=bool)
        for action_index, option in enumerate(self.table):
            if self.processed[option.flow_index]:
                continue
            increment(self.work, "mask_edge_checks", len(option.edges))
            mask[action_index] = all(
                allocation <= self.remaining[edge]
                for edge, allocation in zip(option.edges, option.allocations, strict=True)
            )
        if not self.terminated and not mask.any():
            raise RuntimeError("unprocessed flow must retain a rejection action")
        return mask

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()

    def observation(self) -> dict[str, np.ndarray]:
        flow_count = len(self.scenario.flows)
        path_occurrence_denominator = float(flow_count * PATHS_PER_FLOW)
        edges = np.column_stack(
            (
                self.capacity / MAX_CAPACITY_SRUS,
                self.remaining / MAX_CAPACITY_SRUS,
                self.rates / 1e6,
                [edge.fixed_delay_s / 0.002 for edge in self.scenario.links],
                (self.capacity - self.remaining) / MAX_CAPACITY_SRUS,
                self.occurrences / path_occurrence_denominator,
                self.rank0_occurrences / flow_count,
                [edge.sinr_db / 12.0 for edge in self.scenario.links],
            )
        ).astype(np.float32)
        flows = []
        for flow_index, (flow, catalog) in enumerate(
            zip(self.scenario.flows, self.scenario.path_catalogs, strict=True)
        ):
            path_hops = [path.hop_count for path in catalog.paths]
            flows.append(
                [
                    float(flow.traffic_class is TrafficClass.DELAY),
                    float(flow.traffic_class is TrafficClass.RELIABILITY),
                    flow.priority / 3,
                    flow.data_bits / 2e6,
                    flow.deadline_s / 10,
                    flow.min_reliability,
                    float(self.processed[flow_index]),
                    float(self.served[flow_index]),
                    min(path_hops) / self.max_catalog_hops,
                    max(path_hops) / self.max_catalog_hops,
                    self.feasible_path_counts[flow_index] / PATHS_PER_FLOW,
                ]
            )
        actions = []
        for option in self.table:
            flow = self.scenario.flows[option.flow_index]
            residual = min(
                (
                    (self.remaining[edge] - allocation) / self.capacity[edge]
                    for edge, allocation in zip(option.edges, option.allocations, strict=True)
                ),
                default=0.0,
            )
            occurrence_values = [self.occurrences[edge] for edge in option.edges]
            actions.append(
                [
                    float(option.path_id == -1),
                    option.utility / (2 * flow.priority),
                    option.delay / flow.deadline_s,
                    option.reliability,
                    len(option.edges) / self.max_catalog_hops,
                    sum(option.allocations) / self.max_total_option_srus,
                    residual,
                    option.level / self.max_option_level,
                    (option.path_id + 1) / PATHS_PER_FLOW if option.path_id >= 0 else 0.0,
                    (float(np.mean(occurrence_values)) / path_occurrence_denominator)
                    if occurrence_values
                    else 0.0,
                    (max(occurrence_values) / path_occurrence_denominator)
                    if occurrence_values
                    else 0.0,
                ]
            )
        node_count = len(self.scenario.node_ids)
        global_features = np.asarray(
            [
                self.current_j / self.normalizer,
                1 - self.processed.mean(),
                (self.capacity - self.remaining).sum() / self.capacity.sum(),
                flow_count / MAX_FLOWS,
                len(self.scenario.links) / (node_count * (node_count - 1)),
                self.mean_catalog_hops / self.max_catalog_hops,
            ],
            dtype=np.float32,
        )
        increment(self.work, "observation_edge_rows", len(edges))
        increment(self.work, "observation_flow_rows", len(flows))
        increment(self.work, "observation_action_rows", len(actions))
        return {
            "edges": edges,
            "flows": np.asarray(flows, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "action_edges": self.incidence.copy(),
            "action_flows": self.action_flow.copy(),
            "mask": self._mask.copy(),
            "global": global_features,
        }

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("episode has terminated; call reset before step")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral):
            raise TypeError("action must be an integer table index")
        action = int(action)
        if not 0 <= action < len(self.table) or not self._mask[action]:
            raise ValueError("invalid or masked action; environment does not repair")
        option = self.table[action]
        self._decisions[option.flow_index] = decision_for_option(self.scenario, option)
        self.remaining -= self.requirements[action]
        self.processed[option.flow_index] = True
        self.served[option.flow_index] = option.path_id >= 0
        increment(self.work, "state_allocation_entries", len(option.edges))
        result = evaluate_schedule(self.scenario, self.schedule)
        increment(self.work, "step_evaluations")
        increment(self.work, "step_evaluation_weight", self._weight)
        if not result.feasible or result.objective is None:
            raise RuntimeError("scheduling mask and original evaluator disagree")
        previous = self.current_j
        self.current_j = result.objective
        if not math.isclose(
            self.current_j - previous, option.utility, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError("original evaluator objective is not additive over decoded options")
        reward = (self.current_j - previous) / self.normalizer
        self.reward_sum += reward
        self.steps += 1
        self.terminated = bool(self.processed.all())
        self._mask = self._calculate_mask()
        if self.terminated and (
            self.steps != len(self.scenario.flows)
            or not math.isclose(
                self.reward_sum,
                self.current_j / self.normalizer,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("scheduling terminal reward invariant failed")
        info = {
            "objective": self.current_j,
            "normalized_objective": self.current_j / self.normalizer,
            "feasible": result.feasible,
            "steps": self.steps,
            "truncated": False,
        }
        return self.observation(), reward, self.terminated, info

    def greedy_action(self) -> int:
        if self.terminated:
            raise RuntimeError("no action exists after termination")
        candidates = np.flatnonzero(self._mask)
        increment(self.work, "greedy_action_comparisons", len(candidates))

        def key(action_index: int):
            option = self.table[int(action_index)]
            return (
                -option.utility,
                sum(option.allocations),
                option.flow_id,
                option.path_id,
                option.level,
            )

        return int(min(candidates, key=key))


def solve_control(
    scenario: EvaluatorScenario, mode: str = "greedy", seed: int = 0
) -> dict[str, object]:
    if mode not in ("greedy", "random"):
        raise ValueError("unknown scheduling control")
    environment = FullSchedulingEnv(scenario)
    generator = np.random.default_rng(seed)
    actions: list[int] = []
    rewards: list[float] = []
    while not environment.terminated:
        action = (
            environment.greedy_action()
            if mode == "greedy"
            else int(generator.choice(np.flatnonzero(environment.action_masks())))
        )
        _, reward, _, _ = environment.step(action)
        actions.append(action)
        rewards.append(float(reward))
    result = evaluate_schedule(scenario, environment.schedule)
    increment(environment.work, "terminal_independent_evaluations")
    increment(environment.work, "terminal_independent_evaluation_weight", environment._weight)
    if not result.feasible or result.objective != environment.current_j:
        raise RuntimeError("independent final evaluator disagrees with scheduling control")
    return {
        "schedule": environment.schedule,
        "objective": result.objective,
        "normalized_objective": environment.current_j / environment.normalizer,
        "actions": actions,
        "rewards": rewards,
        "work": dict(environment.work),
    }
