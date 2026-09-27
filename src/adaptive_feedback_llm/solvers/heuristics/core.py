"""Three deterministic heuristics differing only in path ordering."""

from __future__ import annotations

import math
from enum import Enum

from adaptive_feedback_llm.evaluator import evaluate_schedule
from adaptive_feedback_llm.guidance import StaticGuidance, build_static_guidance
from adaptive_feedback_llm.models import (
    CandidatePath,
    EvaluatorScenario,
    Flow,
    FlowDecision,
    Schedule,
)
from adaptive_feedback_llm.solvers.common import (
    PathConstruction,
    construct_path_allocation,
    ordered_flows,
    reserve_allocation,
)
from adaptive_feedback_llm.solvers.results import SolverResult


class HeuristicName(str, Enum):
    """The only deterministic heuristic baselines in the frozen protocol."""

    PRIORITY_AWARE = "priority_aware"
    MAX_SINR = "max_sinr"
    LOAD_AWARE = "load_aware"


def _unserved(flow_id: int) -> FlowDecision:
    return FlowDecision(
        flow_id=flow_id,
        served=False,
        path_id=-1,
        sru_allocations=(),
    )


def _served(
    flow_id: int,
    path: CandidatePath,
    construction: PathConstruction,
) -> FlowDecision:
    return FlowDecision(
        flow_id=flow_id,
        served=True,
        path_id=path.path_id,
        sru_allocations=construction.allocations,
    )


def _priority_paths(
    flow: Flow,
    paths: tuple[CandidatePath, ...],
    scenario: EvaluatorScenario,
    guidance: StaticGuidance,
) -> tuple[CandidatePath, ...]:
    del scenario
    return tuple(
        sorted(paths, key=lambda path: (guidance.rank_for(flow.flow_id, path.path_id), path.path_id))
    )


def _max_sinr_paths(
    flow: Flow,
    paths: tuple[CandidatePath, ...],
    scenario: EvaluatorScenario,
    guidance: StaticGuidance,
) -> tuple[CandidatePath, ...]:
    del flow, guidance
    links = scenario.link_by_edge()
    return tuple(
        sorted(
            paths,
            key=lambda path: (
                -min(links[edge].sinr_db for edge in path.edges),
                -math.fsum(links[edge].sinr_db for edge in path.edges) / path.hop_count,
                path.path_id,
            ),
        )
    )


def _load_aware_options(
    flow: Flow,
    paths: tuple[CandidatePath, ...],
    scenario: EvaluatorScenario,
    guidance: StaticGuidance,
    remaining: dict[tuple[int, int], int],
) -> tuple[tuple[CandidatePath, PathConstruction | None], ...]:
    links = scenario.link_by_edge()
    options: list[
        tuple[tuple[float, float, int, int], CandidatePath, PathConstruction | None]
    ] = []
    for path in paths:
        construction = construct_path_allocation(scenario, flow, path, remaining)
        if construction is None:
            key = (math.inf, math.inf, guidance.rank_for(flow.flow_id, path.path_id), path.path_id)
        else:
            added = {allocation.edge: allocation.num_srus for allocation in construction.allocations}
            projected = tuple(
                (
                    link.capacity_srus
                    - remaining[link.edge]
                    + added.get(link.edge, 0)
                )
                / link.capacity_srus
                for link in scenario.links
            )
            key = (
                max(projected),
                math.fsum(projected),
                guidance.rank_for(flow.flow_id, path.path_id),
                path.path_id,
            )
        options.append((key, path, construction))
    options.sort(key=lambda item: item[0])
    return tuple((path, construction) for _, path, construction in options)


def solve_heuristic(
    scenario: EvaluatorScenario,
    method: HeuristicName,
    *,
    guidance: StaticGuidance | None = None,
) -> SolverResult:
    """Construct one complete schedule and invoke the sole evaluator once."""

    static_guidance = guidance or build_static_guidance(scenario)
    catalogs = scenario.catalog_by_flow_id()
    remaining = {link.edge: link.capacity_srus for link in scenario.links}
    decisions: dict[int, FlowDecision] = {}
    for flow in ordered_flows(scenario):
        paths = catalogs[flow.flow_id].paths
        selected: tuple[CandidatePath, PathConstruction] | None = None
        if method is HeuristicName.LOAD_AWARE:
            for path, construction in _load_aware_options(
                flow, paths, scenario, static_guidance, remaining
            ):
                if construction is not None:
                    selected = (path, construction)
                    break
        else:
            if method is HeuristicName.PRIORITY_AWARE:
                ordered_paths = _priority_paths(
                    flow, paths, scenario, static_guidance
                )
            elif method is HeuristicName.MAX_SINR:
                ordered_paths = _max_sinr_paths(
                    flow, paths, scenario, static_guidance
                )
            else:
                raise ValueError(f"unknown heuristic method: {method!r}")
            for path in ordered_paths:
                construction = construct_path_allocation(
                    scenario, flow, path, remaining
                )
                if construction is not None:
                    selected = (path, construction)
                    break

        if selected is None:
            decisions[flow.flow_id] = _unserved(flow.flow_id)
            continue
        path, construction = selected
        reserve_allocation(remaining, construction)
        decisions[flow.flow_id] = _served(flow.flow_id, path, construction)

    schedule = Schedule(
        flows=tuple(decisions[flow.flow_id] for flow in scenario.flows)
    )
    return SolverResult(
        method=method.value,
        schedule=schedule,
        evaluation=evaluate_schedule(scenario, schedule),
    )


def solve_priority_aware(
    scenario: EvaluatorScenario,
    *,
    guidance: StaticGuidance | None = None,
) -> SolverResult:
    """Solve with static-guidance path rank."""

    return solve_heuristic(
        scenario, HeuristicName.PRIORITY_AWARE, guidance=guidance
    )


def solve_max_sinr(
    scenario: EvaluatorScenario,
    *,
    guidance: StaticGuidance | None = None,
) -> SolverResult:
    """Solve with maximum bottleneck and then mean path SINR."""

    return solve_heuristic(scenario, HeuristicName.MAX_SINR, guidance=guidance)


def solve_load_aware(
    scenario: EvaluatorScenario,
    *,
    guidance: StaticGuidance | None = None,
) -> SolverResult:
    """Solve with dynamic projected utilization path ordering."""

    return solve_heuristic(scenario, HeuristicName.LOAD_AWARE, guidance=guidance)
