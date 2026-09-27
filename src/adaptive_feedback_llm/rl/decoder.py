"""Finite rate-breakpoint options verified by the original complete evaluator.

Use multiplication comparisons, not floating ceil(rate / c). Every achievable
bottleneck is an element of these same floating products. This preserves the
original evaluator's finite arithmetic and QoS tolerance at boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

from adaptive_feedback_llm.evaluator.core import evaluate_schedule
from adaptive_feedback_llm.models.domain import (
    EvaluatorScenario, FlowDecision, Schedule, SRUAllocation, TrafficClass,
)
from adaptive_feedback_llm.physics.core import rate_per_sru_bps


@dataclass(frozen=True, slots=True)
class Option:
    flow_index: int
    flow_id: int
    path_id: int
    level: int
    edges: tuple[int, ...]
    allocations: tuple[int, ...]
    utility: float
    delay: float
    reliability: float


def rejected_decision(flow_id: int) -> FlowDecision:
    return FlowDecision(flow_id=flow_id, served=False, path_id=-1, sru_allocations=())


def rejected_schedule(scenario: EvaluatorScenario) -> Schedule:
    return Schedule(flows=tuple(rejected_decision(f.flow_id) for f in scenario.flows))


def decision_for_option(scenario: EvaluatorScenario, option: Option) -> FlowDecision:
    if option.path_id == -1:
        return rejected_decision(option.flow_id)
    return FlowDecision(
        flow_id=option.flow_id, served=True, path_id=option.path_id,
        sru_allocations=tuple(SRUAllocation(
            src=scenario.links[e].src, dst=scenario.links[e].dst, num_srus=n,
        ) for e, n in zip(option.edges, option.allocations, strict=True)),
    )


def increment(work: dict[str, int], name: str, count: int = 1) -> None:
    work[name] = work.get(name, 0) + count


def evaluation_weight(scenario: EvaluatorScenario) -> int:
    hops = max(p.hop_count for cat in scenario.path_catalogs for p in cat.paths)
    return len(scenario.flows) + len(scenario.links) + len(scenario.flows) * hops


def build_options(scenario: EvaluatorScenario, work: dict[str, int] | None = None) -> tuple[Option, ...]:
    work = {} if work is None else work
    edge_index = {link.edge: i for i, link in enumerate(scenario.links)}
    unit_rates = tuple(rate_per_sru_bps(link.sinr_db, scenario.physics.sru_bandwidth_hz)
                       for link in scenario.links)
    template = rejected_schedule(scenario)
    table = []
    weight = evaluation_weight(scenario)
    for fi, (flow, catalog) in enumerate(zip(scenario.flows, scenario.path_catalogs, strict=True)):
        table.append(Option(fi, flow.flow_id, -1, 0, (), (), 0.0, 0.0, 0.0))
        for path in catalog.paths:
            edges = tuple(edge_index[e] for e in path.edges)
            products = tuple(tuple(k * unit_rates[e] for k in range(1, scenario.links[e].capacity_srus + 1))
                             for e in edges)
            breakpoints = sorted({rate for rates in products for rate in rates})
            increment(work, "raw_breakpoints", sum(map(len, products)))
            vectors = set()
            retained: list[Option] = []
            for rate in breakpoints:
                vector = []
                for rates in products:
                    allocation = None
                    for k, achievable in enumerate(rates, start=1):
                        increment(work, "rate_product_comparisons")
                        if achievable >= rate:
                            allocation = k
                            break
                    if allocation is None:
                        break
                    vector.append(allocation)
                if len(vector) != len(edges):
                    continue
                allocation_tuple = tuple(vector)
                if allocation_tuple in vectors:
                    continue
                vectors.add(allocation_tuple)
                increment(work, "option_vector_entries", len(edges))
                option = Option(fi, flow.flow_id, path.path_id, len(retained), edges,
                                allocation_tuple, 0.0, 0.0, 0.0)
                decisions = list(template.flows)
                decisions[fi] = decision_for_option(scenario, option)
                result = evaluate_schedule(scenario, Schedule(flows=tuple(decisions)))
                increment(work, "option_evaluations")
                increment(work, "option_evaluation_weight", weight)
                if not result.feasible:
                    continue
                item = next(f for f in result.flow_evaluations if f.flow_id == flow.flow_id)
                if item.utility is None or item.completion_delay_s is None or item.reliability is None:
                    raise RuntimeError("original evaluator returned incomplete feasible flow metrics")
                option = Option(fi, flow.flow_id, path.path_id, len(retained), edges,
                                allocation_tuple, item.utility, item.completion_delay_s, item.reliability)
                # Increasing rate creates a componentwise nested vector chain.
                # Keep the first (least resource) member of every exact-utility plateau.
                if retained:
                    previous = retained[-1]
                    increment(work, "dominance_edge_comparisons", len(edges))
                    if not all(a <= b for a, b in zip(previous.allocations, option.allocations, strict=True)):
                        raise RuntimeError("breakpoint options are not componentwise monotone")
                    if previous.utility >= option.utility:
                        continue
                retained.append(option)
                if flow.traffic_class is TrafficClass.RELIABILITY:
                    break
            table.extend(retained)
    increment(work, "retained_actions", len(table))
    return tuple(table)
