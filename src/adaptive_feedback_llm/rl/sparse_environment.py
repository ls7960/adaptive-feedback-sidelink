"""Sparse, budget-matched constructive environment for scheduling.

Each flow has exactly three actions: rejection and two served actions.  All
five catalog paths are screened with closed-form physics, the two best
feasible paths under a fixed score-independent rank are retained, and one
QoS-knee allocation is built for each retained path.  No per-option complete
schedule evaluation is used.  The original evaluator remains the sole step
and terminal oracle, so masks cannot silently repair an invalid schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from adaptive_feedback_llm.evaluator.core import evaluate_schedule
from adaptive_feedback_llm.models.domain import EvaluatorScenario, Schedule, TrafficClass
from adaptive_feedback_llm.physics.core import (
    classified_margin,
    flow_utility,
    link_success_probability,
    path_reliability,
    pipelined_completion_delay_s,
    rate_per_sru_bps,
)
from adaptive_feedback_llm.rl.decoder import (
    Option,
    decision_for_option,
    evaluation_weight,
    increment,
    rejected_schedule,
)
from adaptive_feedback_llm.rl.full_environment import (
    ACTION_FEATURES,
    EDGE_FEATURES,
    FLOW_FEATURES,
    GLOBAL_FEATURES,
    MAX_CAPACITY_SRUS,
    MAX_FLOWS,
    MAX_NODES,
    PATHS_PER_FLOW,
    validate_scenario_envelope,
)


PROFILE_ID = "sparse-scheduling"
PATHS_RETAINED = 2
PROFILES_PER_PATH = 1
ACTIONS_PER_FLOW = 1 + PATHS_RETAINED * PROFILES_PER_PATH
KNEE_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class _ScreenedPath:
    path_id: int
    edges: tuple[int, ...]
    unit_rates: tuple[float, ...]
    capacities: tuple[int, ...]
    fixed_delay: float
    reliability: float
    required_rate: float
    maximum_bottleneck_rate: float
    minimum_allocations: tuple[int, ...]
    potential_utility: float


def _allocation_for_rate(
    unit_rates: tuple[float, ...], capacities: tuple[int, ...], target: float
) -> tuple[int, ...] | None:
    output: list[int] = []
    for unit_rate, capacity in zip(unit_rates, capacities, strict=True):
        allocation = next(
            (count for count in range(1, capacity + 1) if count * unit_rate >= target),
            None,
        )
        if allocation is None:
            return None
        output.append(allocation)
    return tuple(output)


def _metrics(
    scenario: EvaluatorScenario,
    flow_index: int,
    screened: _ScreenedPath,
    allocations: tuple[int, ...],
) -> tuple[float, float, float]:
    flow = scenario.flows[flow_index]
    bottleneck = min(
        count * rate
        for count, rate in zip(allocations, screened.unit_rates, strict=True)
    )
    delay = pipelined_completion_delay_s(
        flow.data_bits,
        bottleneck,
        (scenario.links[edge].fixed_delay_s for edge in screened.edges),
    )
    margin = classified_margin(
        flow.traffic_class,
        completion_delay_s=delay,
        deadline_s=flow.deadline_s,
        reliability=screened.reliability,
        minimum_reliability=flow.min_reliability,
        tolerance=scenario.numerics.float_tolerance,
    )
    return flow_utility(flow.priority, margin), delay, screened.reliability


def _screen_paths(
    scenario: EvaluatorScenario,
    flow_index: int,
    edge_lookup: dict[tuple[int, int], int],
    unit_rates: tuple[float, ...],
    work: dict[str, int],
) -> list[_ScreenedPath]:
    flow = scenario.flows[flow_index]
    catalog = scenario.path_catalogs[flow_index]
    tolerance = scenario.numerics.float_tolerance
    screened: list[_ScreenedPath] = []
    for path in catalog.paths:
        increment(work, "path_screen_candidates")
        edges = tuple(edge_lookup[edge] for edge in path.edges)
        increment(work, "path_screen_hops", len(edges))
        rates = tuple(unit_rates[edge] for edge in edges)
        capacities = tuple(scenario.links[edge].capacity_srus for edge in edges)
        fixed_delay = math.fsum(scenario.links[edge].fixed_delay_s for edge in edges)
        reliability = path_reliability(
            link_success_probability(
                scenario.links[edge].sinr_db,
                scenario.physics.operating_sinr_db,
            )
            for edge in edges
        )
        if reliability + tolerance < flow.min_reliability:
            continue
        available = flow.deadline_s + tolerance - fixed_delay
        if available <= 0.0:
            continue
        required_rate = flow.data_bits / available
        maximum = min(
            capacity * rate
            for capacity, rate in zip(capacities, rates, strict=True)
        )
        if maximum < required_rate:
            continue
        minimum = _allocation_for_rate(rates, capacities, required_rate)
        if minimum is None:
            continue
        maximum_delay = pipelined_completion_delay_s(
            flow.data_bits, maximum, (scenario.links[edge].fixed_delay_s for edge in edges)
        )
        potential_margin = classified_margin(
            flow.traffic_class,
            completion_delay_s=maximum_delay,
            deadline_s=flow.deadline_s,
            reliability=reliability,
            minimum_reliability=flow.min_reliability,
            tolerance=tolerance,
        )
        screened.append(
            _ScreenedPath(
                path_id=path.path_id,
                edges=edges,
                unit_rates=rates,
                capacities=capacities,
                fixed_delay=fixed_delay,
                reliability=reliability,
                required_rate=required_rate,
                maximum_bottleneck_rate=maximum,
                minimum_allocations=minimum,
                potential_utility=flow_utility(flow.priority, potential_margin),
            )
        )
    screened.sort(
        key=lambda item: (
            -item.potential_utility,
            sum(item.minimum_allocations),
            len(item.edges),
            item.path_id,
        )
    )
    increment(work, "path_rank_items", len(screened))
    return screened


def _knee_allocations(
    flow_class: TrafficClass, screened: _ScreenedPath
) -> tuple[int, ...]:
    if flow_class is TrafficClass.RELIABILITY:
        return screened.minimum_allocations
    target = screened.required_rate + KNEE_FRACTION * (
        screened.maximum_bottleneck_rate - screened.required_rate
    )
    allocations = _allocation_for_rate(screened.unit_rates, screened.capacities, target)
    if allocations is None:
        raise RuntimeError("registered QoS-knee target is outside path capacity")
    return allocations


def build_budget_options(
    scenario: EvaluatorScenario, work: dict[str, int] | None = None
) -> tuple[Option, ...]:
    """Build exactly ``3F`` analytical actions without invoking the evaluator."""

    work = {} if work is None else work
    edge_lookup = {link.edge: index for index, link in enumerate(scenario.links)}
    unit_rates = tuple(
        rate_per_sru_bps(link.sinr_db, scenario.physics.sru_bandwidth_hz)
        for link in scenario.links
    )
    table: list[Option] = []
    for flow_index, flow in enumerate(scenario.flows):
        table.append(Option(flow_index, flow.flow_id, -1, 0, (), (), 0.0, 0.0, 0.0))
        candidates = _screen_paths(
            scenario, flow_index, edge_lookup, unit_rates, work
        )
        if not candidates:
            raise ValueError(
                f"flow {flow.flow_id} has no individually QoS-feasible catalog path"
            )
        chosen = candidates[:PATHS_RETAINED]
        if len(chosen) == 1:
            # Preserve the registered 3F contract.  The fallback remains a
            # one-path analytical profile and never consults method outcomes.
            chosen = [chosen[0], chosen[0]]
        for rank, screened in enumerate(chosen):
            allocations = _knee_allocations(flow.traffic_class, screened)
            if rank == 1 and screened.path_id == chosen[0].path_id:
                alternative = screened.capacities
                if alternative != allocations:
                    allocations = alternative
            utility, delay, reliability = _metrics(
                scenario, flow_index, screened, allocations
            )
            increment(work, "analytic_profile_hops", len(screened.edges))
            table.append(
                Option(
                    flow_index,
                    flow.flow_id,
                    screened.path_id,
                    rank,
                    screened.edges,
                    allocations,
                    utility,
                    delay,
                    reliability,
                )
            )
    if len(table) != ACTIONS_PER_FLOW * len(scenario.flows):
        raise RuntimeError("budget-matched decoder violated the exact 3F action contract")
    increment(work, "retained_actions", len(table))
    return tuple(table)


class SparseSchedulingEnv:
    def __init__(self, scenario: EvaluatorScenario):
        validate_scenario_envelope(scenario)
        self.scenario = scenario
        self.work: dict[str, int] = {}
        self.table = build_budget_options(scenario, self.work)
        self.normalizer = 2.0 * math.fsum(flow.priority for flow in scenario.flows)
        self.capacity = np.asarray(
            [edge.capacity_srus for edge in scenario.links], dtype=np.int64
        )
        self.action_flow = np.asarray(
            [option.flow_index for option in self.table], dtype=np.int64
        )
        sparse_actions: list[int] = []
        sparse_edges: list[int] = []
        sparse_weights: list[float] = []
        for action_index, option in enumerate(self.table):
            for edge, allocation in zip(option.edges, option.allocations, strict=True):
                sparse_actions.append(action_index)
                sparse_edges.append(edge)
                sparse_weights.append(allocation / self.capacity[edge])
        self.sparse_actions = np.asarray(sparse_actions, dtype=np.int64)
        self.sparse_edges = np.asarray(sparse_edges, dtype=np.int64)
        self.sparse_weights = np.asarray(sparse_weights, dtype=np.float32)
        increment(self.work, "sparse_action_edge_entries", len(sparse_edges))

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
        self.max_option_level = PATHS_RETAINED - 1
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
                    option.level / max(1, self.max_option_level),
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
            "action_edge_actions": self.sparse_actions.copy(),
            "action_edge_indices": self.sparse_edges.copy(),
            "action_edge_weights": self.sparse_weights.copy(),
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
        for edge, allocation in zip(option.edges, option.allocations, strict=True):
            self.remaining[edge] -= allocation
        self.processed[option.flow_index] = True
        self.served[option.flow_index] = option.path_id >= 0
        increment(self.work, "state_allocation_entries", len(option.edges))
        result = evaluate_schedule(self.scenario, self.schedule)
        increment(self.work, "step_evaluations")
        increment(self.work, "step_evaluation_weight", self._weight)
        if not result.feasible or result.objective is None:
            raise RuntimeError("budget-matched mask and original evaluator disagree")
        previous = self.current_j
        self.current_j = result.objective
        if not math.isclose(
            self.current_j - previous, option.utility, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError("analytical option utility disagrees with original evaluator")
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
            raise RuntimeError("budget-matched terminal reward invariant failed")
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
        raise ValueError("unknown budget-matched control")
    environment = SparseSchedulingEnv(scenario)
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
    increment(
        environment.work,
        "terminal_independent_evaluation_weight",
        environment._weight,
    )
    if not result.feasible or result.objective != environment.current_j:
        raise RuntimeError("terminal evaluator disagrees with budget-matched control")
    return {
        "schedule": environment.schedule,
        "objective": result.objective,
        "normalized_objective": environment.current_j / environment.normalizer,
        "actions": actions,
        "rewards": rewards,
        "work": dict(environment.work),
    }
