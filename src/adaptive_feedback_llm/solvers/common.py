"""Shared deterministic schedule construction used by all solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from adaptive_feedback_llm.models import (
    CandidatePath,
    EvaluatorScenario,
    Flow,
    SRUAllocation,
)
from adaptive_feedback_llm.physics import (
    bottleneck_rate_bps,
    link_success_probability,
    path_reliability,
    pipelined_completion_delay_s,
    rate_per_sru_bps,
)


@dataclass(frozen=True, slots=True)
class PathConstruction:
    """A single-flow allocation and its independently computed metrics."""

    allocations: tuple[SRUAllocation, ...]
    bottleneck_rate_bps: float
    completion_delay_s: float
    reliability: float


def ordered_flows(scenario: EvaluatorScenario) -> tuple[Flow, ...]:
    """Return the one shared `(-priority, flow_id)` heuristic order."""

    return tuple(sorted(scenario.flows, key=lambda flow: (-flow.priority, flow.flow_id)))


def construct_path_allocation(
    scenario: EvaluatorScenario,
    flow: Flow,
    path: CandidatePath,
    remaining_capacity: Mapping[tuple[int, int], int],
) -> PathConstruction | None:
    """Greedily add SRUs to the current slowest hop until both QoS tests pass."""

    links = scenario.link_by_edge()
    tolerance = scenario.numerics.float_tolerance
    if any(remaining_capacity.get(edge, 0) < 1 for edge in path.edges):
        return None

    reliability = path_reliability(
        link_success_probability(
            links[edge].sinr_db,
            scenario.physics.operating_sinr_db,
        )
        for edge in path.edges
    )
    if reliability + tolerance < flow.min_reliability:
        return None

    rates_per_sru = tuple(
        rate_per_sru_bps(links[edge].sinr_db, scenario.physics.sru_bandwidth_hz)
        for edge in path.edges
    )
    allocations = [1] * len(path.edges)
    while True:
        hop_rates = tuple(
            count * rate for count, rate in zip(allocations, rates_per_sru, strict=True)
        )
        bottleneck = bottleneck_rate_bps(hop_rates)
        delay = pipelined_completion_delay_s(
            flow.data_bits,
            bottleneck,
            (links[edge].fixed_delay_s for edge in path.edges),
        )
        if delay <= flow.deadline_s + tolerance:
            return PathConstruction(
                allocations=tuple(
                    SRUAllocation(src=edge[0], dst=edge[1], num_srus=count)
                    for edge, count in zip(path.edges, allocations, strict=True)
                ),
                bottleneck_rate_bps=bottleneck,
                completion_delay_s=delay,
                reliability=reliability,
            )

        slowest_index = select_increment_edge_index(
            path.edges,
            tuple(allocations),
            rates_per_sru,
            remaining_capacity,
        )
        if slowest_index is None:
            return None
        allocations[slowest_index] += 1


def select_increment_edge_index(
    path_edges: tuple[tuple[int, int], ...],
    allocations: tuple[int, ...],
    rates_per_sru_bps: tuple[float, ...],
    remaining_capacity: Mapping[tuple[int, int], int],
) -> int | None:
    """Select the slowest growable hop; exact rate ties keep path-edge order."""

    if not (len(path_edges) == len(allocations) == len(rates_per_sru_bps)):
        raise ValueError("path, allocation, and rate lengths must match")
    growable = tuple(
        index
        for index, edge in enumerate(path_edges)
        if allocations[index] < remaining_capacity.get(edge, 0)
    )
    if not growable:
        return None
    return min(
        growable,
        key=lambda index: (allocations[index] * rates_per_sru_bps[index], index),
    )


def reserve_allocation(
    remaining_capacity: dict[tuple[int, int], int],
    construction: PathConstruction,
) -> None:
    """Reserve a successful construction in place for the caller's local state."""

    for allocation in construction.allocations:
        remaining_capacity[allocation.edge] -= allocation.num_srus
