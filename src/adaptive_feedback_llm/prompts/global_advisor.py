"""paper prompt global strategy with verifiable, non-parametric obligations.

The selector reads only the immutable public LLM view.  It never evaluates or
constructs a candidate.  Detailed aggregate audit fields stay outside the
rendered prompt; the visible brief exposes one strategy and no concrete flow,
path, edge, SRU, admission-count, or candidate target.
"""
from __future__ import annotations

from collections import Counter
from math import ceil

from adaptive_feedback_llm.models import (
    DiagnosticCode,
    FailureCategory,
    LLMScenarioView,
    TrafficClass,
)
from adaptive_feedback_llm.prompts.opportunities import _schedule_key


_STRUCTURAL_CODES = {
    DiagnosticCode.MISSING_FLOW,
    DiagnosticCode.DUPLICATE_FLOW,
    DiagnosticCode.UNKNOWN_FLOW,
    DiagnosticCode.INVALID_UNSERVED_STATE,
    DiagnosticCode.INVALID_PATH_ID,
    DiagnosticCode.PATH_NOT_IN_CATALOG,
    DiagnosticCode.INVALID_CATALOG_PATH,
    DiagnosticCode.MISSING_ALLOCATION,
    DiagnosticCode.DUPLICATE_ALLOCATION,
    DiagnosticCode.PATH_EXTERNAL_ALLOCATION,
    DiagnosticCode.INVALID_SRU_COUNT,
}

_DIRECTIVES = {
    "repair_structure": (
        "Make the next complete candidate structurally legal: verify flow coverage, "
        "catalog paths, path-edge allocations, and positive integer SRUs."
    ),
    "priority_reduction": (
        "Use a lower nonzero admission scale than the repeatedly unsuccessful recent "
        "scale; normally reduce priority 1 before priority 2, and priority 3 last. "
        "Choose the concrete membership, routes, and SRUs from the complete evidence."
    ),
    "priority_exchange": (
        "Change the admission membership, not only routes or SRUs. Explore a coherent "
        "priority-aware exchange that restores higher-priority effective service while "
        "reducing lower-priority admitted load when necessary, and rebuild affected "
        "routes and SRUs in the same complete candidate."
    ),
    "route_exploration": (
        "Use a joint admission-route signature not previously evaluated and rebuild its "
        "SRUs; do not submit an allocation-only variant or reuse an old route pattern."
    ),
    "sru_redistribution": (
        "Use an SRU allocation signature not previously evaluated; redistribute "
        "resources coherently instead of uniformly filling capacity, and preserve joint "
        "capacity and QoS feasibility."
    ),
    "joint_reconfiguration": (
        "Change at least two of admission membership, route pattern, and SRU pattern "
        "relative to recent evidence; do not repeat an old complete schedule, and close "
        "released and received capacity in the same candidate."
    ),
}


def _admission_signature(record) -> tuple[int, ...]:
    return tuple(sorted(flow.flow_id for flow in record.schedule.flows if flow.served))


def _route_signature(record) -> tuple:
    return tuple(sorted(
        (flow.flow_id, flow.served, flow.path_id)
        for flow in record.schedule.flows
    ))


def _allocation_signature(record) -> tuple:
    return tuple(sorted(
        (
            flow.flow_id,
            tuple(sorted(
                (item.src, item.dst, item.num_srus)
                for item in flow.sru_allocations
            )),
        )
        for flow in record.schedule.flows
    ))


def _level(numerator: int, denominator: int) -> str:
    if denominator <= 0 or numerator * 3 < denominator:
        return "low"
    if numerator * 3 < denominator * 2:
        return "medium"
    return "high"


def _is_structurally_valid(record) -> bool:
    return not any(
        diagnostic.code in _STRUCTURAL_CODES
        for diagnostic in record.evaluation.diagnostics
    )


def _priority_profile(record, priority_by_flow: dict[int, float], levels: tuple[float, ...]) -> tuple[int, ...]:
    served = Counter(
        priority_by_flow[flow.flow_id]
        for flow in record.schedule.flows
        if flow.served and flow.flow_id in priority_by_flow
    )
    return tuple(served[level] for level in levels)


def build_global_advisor(view: LLMScenarioView) -> tuple[dict, dict]:
    """Return an internal aggregate audit and the minimal paper prompt brief."""

    history = view.evaluated_history
    best = next((record for record in history if record.best_so_far), None)
    useful = bool(
        best is not None
        and best.evaluation.feasible
        and best.evaluation.objective is not None
        and best.evaluation.objective > 0.0
        and any(flow.served for flow in best.schedule.flows)
    )
    latest_round = max(
        (item.round_index for item in (*history, *view.failure_feedback)), default=0
    )
    next_round = min(20, latest_round + 1)
    recent_start = max(1, latest_round - 2)
    recent = tuple(record for record in history if record.round_index >= recent_start)
    latest_records = tuple(record for record in history if record.round_index == latest_round)
    latest_failures = tuple(
        item for item in view.failure_feedback if item.round_index == latest_round
    )

    first_seen: dict[tuple, tuple[int, int]] = {}
    recent_duplicates = 0
    recent_cross_round_repeats = 0
    recent_same_round_repeats = 0
    for record in history:
        key = _schedule_key(record.schedule)
        previous = first_seen.get(key)
        if record.round_index >= recent_start and previous is not None:
            recent_duplicates += 1
            if previous[0] < record.round_index:
                recent_cross_round_repeats += 1
            else:
                recent_same_round_repeats += 1
        first_seen.setdefault(key, (record.round_index, record.call_index))
    repetition = _level(recent_duplicates, len(recent))

    running_best: float | None = None
    last_progress_round = 0
    for record in history:
        objective = record.evaluation.objective
        if (
            record.evaluation.feasible
            and objective is not None
            and objective > 0.0
            and (running_best is None or objective > running_best)
        ):
            running_best = objective
            last_progress_round = record.round_index
    rounds_since_progress = (
        max(0, latest_round - last_progress_round) if useful else latest_round
    )

    latest_parse_failures = sum(
        item.category is FailureCategory.PARSE_FAILURE for item in latest_failures
    )
    latest_structural = sum(
        any(item.code in _STRUCTURAL_CODES for item in record.evaluation.diagnostics)
        for record in latest_records
    ) + latest_parse_failures
    latest_attempts = len(latest_records) + len(latest_failures)
    structure_unstable = (
        latest_attempts > 0 and latest_structural >= ceil(latest_attempts / 2)
    )

    diagnostics = Counter()
    capacity_delay_joint = 0
    for record in recent:
        codes = {item.code for item in record.evaluation.diagnostics}
        if DiagnosticCode.LINK_CAPACITY_EXCEEDED in codes:
            diagnostics["capacity"] += 1
        if DiagnosticCode.DELAY_QOS_VIOLATION in codes:
            diagnostics["delay"] += 1
        if DiagnosticCode.RELIABILITY_QOS_VIOLATION in codes:
            diagnostics["reliability"] += 1
        if {
            DiagnosticCode.LINK_CAPACITY_EXCEEDED,
            DiagnosticCode.DELAY_QOS_VIOLATION,
        } <= codes:
            capacity_delay_joint += 1
    coupled = bool(recent) and capacity_delay_joint >= ceil(len(recent) / 2)
    reliability_dominant = (
        diagnostics["reliability"] > 0
        and diagnostics["reliability"] >= diagnostics["delay"]
        and diagnostics["reliability"] >= diagnostics["capacity"]
    )
    delay_without_capacity = (
        diagnostics["delay"] > 0 and diagnostics["capacity"] == 0
    )

    prior_records = tuple(record for record in history if record.round_index < latest_round)
    prior_admissions = {_admission_signature(record) for record in prior_records}
    prior_routes = {_route_signature(record) for record in prior_records}
    prior_allocations = {_allocation_signature(record) for record in prior_records}
    prior_complete = {_schedule_key(record.schedule) for record in prior_records}
    latest_new_admissions = sum(
        _admission_signature(record) not in prior_admissions for record in latest_records
    )
    latest_new_routes = sum(
        _route_signature(record) not in prior_routes for record in latest_records
    )
    latest_new_allocations = sum(
        _allocation_signature(record) not in prior_allocations for record in latest_records
    )
    latest_new_complete = sum(
        _schedule_key(record.schedule) not in prior_complete for record in latest_records
    )

    allocation_only = 0
    latest_new_joint = 0
    if useful and best is not None:
        best_admission = _admission_signature(best)
        best_route = _route_signature(best)
        best_allocation = _allocation_signature(best)
        for record in recent:
            if (
                _admission_signature(record) == best_admission
                and _route_signature(record) == best_route
                and _allocation_signature(record) != best_allocation
            ):
                allocation_only += 1
        for record in latest_records:
            changed_dimensions = sum((
                _admission_signature(record) != best_admission,
                _route_signature(record) != best_route,
                _allocation_signature(record) != best_allocation,
            ))
            if changed_dimensions >= 2 and _schedule_key(record.schedule) not in prior_complete:
                latest_new_joint += 1
    allocation_heavy = len(recent) >= 3 and allocation_only * 2 >= len(recent)

    valid_positive_counts = [
        sum(flow.served for flow in record.schedule.flows)
        for record in history
        if _is_structurally_valid(record)
        and any(flow.served for flow in record.schedule.flows)
    ]
    recent_round_minima = []
    for round_index in range(recent_start, latest_round + 1):
        counts = [
            sum(flow.served for flow in record.schedule.flows)
            for record in history
            if record.round_index == round_index
            and _is_structurally_valid(record)
            and any(flow.served for flow in record.schedule.flows)
        ]
        if counts:
            recent_round_minima.append(min(counts))
    admission_stagnant = (
        not useful
        and latest_round >= 3
        and len(recent_round_minima) == 3
        and recent_round_minima[-1] >= recent_round_minima[0]
    )
    recent_minimum_admission = min(recent_round_minima, default=0)
    can_reduce_nonzero = recent_minimum_admission > 1

    priority_by_flow = {flow.flow_id: flow.priority for flow in view.static.flows}
    priority_levels = tuple(sorted(set(priority_by_flow.values()), reverse=True))
    served_by_priority: Counter = Counter()
    rejected_by_priority: Counter = Counter()
    priority_inversion = False
    partial_admission = False
    latest_new_priority_exchange = 0
    if useful and best is not None:
        for flow in best.schedule.flows:
            target = served_by_priority if flow.served else rejected_by_priority
            if flow.flow_id in priority_by_flow:
                target[priority_by_flow[flow.flow_id]] += 1
        priority_inversion = bool(
            served_by_priority and rejected_by_priority
            and max(rejected_by_priority) > min(served_by_priority)
        )
        partial_admission = bool(rejected_by_priority)
        best_profile = _priority_profile(best, priority_by_flow, priority_levels)
        best_admission = _admission_signature(best)
        for record in latest_records:
            membership = _admission_signature(record)
            if membership == best_admission or membership in prior_admissions:
                continue
            profile = _priority_profile(record, priority_by_flow, priority_levels)
            if (priority_inversion and profile > best_profile) or (
                not priority_inversion and len(membership) > len(best_admission)
            ):
                latest_new_priority_exchange += 1

    delay_headroom = 0.0
    reliability_headroom = 0.0
    saturated_edges = 0
    used_edges = 0
    if useful and best is not None:
        flow_meta = {flow.flow_id: flow for flow in view.static.flows}
        for metric in best.evaluation.flow_evaluations:
            if not metric.served or metric.utility is None or metric.flow_id not in flow_meta:
                continue
            metadata = flow_meta[metric.flow_id]
            headroom = max(0.0, 2.0 * metadata.priority - metric.utility)
            if metadata.traffic_class is TrafficClass.DELAY:
                delay_headroom += headroom
            else:
                reliability_headroom += headroom
        used_edges = len(best.evaluation.link_usage)
        saturated_edges = sum(
            item.used_srus == item.capacity_srus
            for item in best.evaluation.link_usage
        )
    capacity_pressure_high = used_edges > 0 and saturated_edges * 3 >= used_edges

    possible_pairs = len(view.static.paths)
    tried_pairs = {
        (flow.flow_id, flow.path_id)
        for record in history
        for flow in record.schedule.flows
        if flow.served and flow.path_id >= 0
        and any(
            path.flow_id == flow.flow_id and path.path_id == flow.path_id
            for path in view.static.paths
        )
    }
    route_underexplored = possible_pairs > 0 and len(tried_pairs) * 5 < possible_pairs * 2

    if not useful and latest_round >= 15 and can_reduce_nonzero:
        strategy, strength, basis = (
            "priority_reduction", "required", "late_no_feasible_adjust_admission_scale"
        )
    elif structure_unstable:
        strategy, strength, basis = (
            "repair_structure", "required", "latest_round_structure_failures"
        )
    elif useful and priority_inversion:
        strategy, basis = "priority_exchange", "priority_inversion_persists"
        strength = (
            "required"
            if rounds_since_progress >= 1 and latest_new_priority_exchange == 0
            else "preferred"
        )
    elif useful and partial_admission and rounds_since_progress >= 1:
        strategy, basis = "priority_exchange", "partial_admission_plateau"
        strength = (
            "required"
            if latest_new_priority_exchange == 0 else "preferred"
        )
    elif not useful and admission_stagnant and can_reduce_nonzero:
        strategy, strength, basis = (
            "priority_reduction", "preferred", "construction_admission_stagnant"
        )
    elif not useful and reliability_dominant:
        strategy, strength, basis = (
            "route_exploration", "preferred", "reliability_failures_dominate"
        )
    elif not useful and delay_without_capacity:
        strategy, strength, basis = (
            "sru_redistribution", "preferred", "delay_failures_without_capacity"
        )
    elif not useful and coupled:
        strategy, strength, basis = (
            "joint_reconfiguration", "preferred", "capacity_delay_coupling"
        )
    elif not useful and route_underexplored and repetition != "low":
        strategy, strength, basis = (
            "route_exploration", "preferred", "low_route_coverage_with_repetition"
        )
    elif not useful:
        strategy, strength, basis = (
            "joint_reconfiguration", "preferred", "default_joint_construction"
        )
    elif allocation_heavy or reliability_headroom > delay_headroom:
        strategy, basis = (
            "route_exploration",
            "allocation_only_plateau" if allocation_heavy else "reliability_headroom_dominates",
        )
        strength = (
            "required"
            if rounds_since_progress >= 2 and latest_new_routes == 0 else "preferred"
        )
    elif delay_headroom > 0.0 and not capacity_pressure_high:
        strategy, basis = "sru_redistribution", "delay_headroom_with_capacity_slack"
        strength = (
            "required"
            if rounds_since_progress >= 2 and latest_new_allocations == 0 else "preferred"
        )
    else:
        strategy, basis = "joint_reconfiguration", "mixed_objective_or_capacity_pressure"
        strength = (
            "required"
            if rounds_since_progress >= 2 and repetition == "high" and latest_new_joint == 0
            else "preferred"
        )

    audit = {
        "schema_version": "1.1",
        "public_view_only": True,
        "latest_completed_round": latest_round,
        "next_round": next_round,
        "mode": "improve_feasible" if useful else "construct_feasible",
        "positive_feasible": useful,
        "rounds_since_progress": rounds_since_progress,
        "recent_window": {
            "first_round": recent_start if latest_round else None,
            "evaluated_candidates": len(recent),
            "all_duplicate_candidates": recent_duplicates,
            "cross_round_repeats": recent_cross_round_repeats,
            "same_round_repeats": recent_same_round_repeats,
            "repetition": repetition,
            "capacity_failure_candidates": diagnostics["capacity"],
            "delay_failure_candidates": diagnostics["delay"],
            "reliability_failure_candidates": diagnostics["reliability"],
            "capacity_delay_joint_candidates": capacity_delay_joint,
        },
        "latest_progress_obligations": {
            "new_admission_signatures": latest_new_admissions,
            "new_route_signatures": latest_new_routes,
            "new_allocation_signatures": latest_new_allocations,
            "new_complete_signatures": latest_new_complete,
            "new_joint_changes": latest_new_joint,
            "new_priority_exchange_attempts": latest_new_priority_exchange,
        },
        "construction": {
            "positive_admitted_counts_tried": sorted(set(valid_positive_counts)),
            "recent_round_minimum_admissions": recent_round_minima,
            "admission_stagnant": admission_stagnant,
            "can_reduce_to_lower_nonzero_scale": can_reduce_nonzero,
        },
        "incumbent_aggregate": {
            "partial_admission": partial_admission,
            "priority_inversion": priority_inversion,
            "served_by_priority": dict(sorted(served_by_priority.items(), reverse=True)),
            "rejected_by_priority": dict(sorted(rejected_by_priority.items(), reverse=True)),
            "delay_objective_headroom": delay_headroom,
            "reliability_objective_headroom": reliability_headroom,
            "used_selected_edges": used_edges,
            "saturated_selected_edges": saturated_edges,
            "capacity_pressure_high": capacity_pressure_high,
        },
        "search_coverage": {
            "possible_flow_path_pairs": possible_pairs,
            "distinct_flow_path_pairs_tried": len(tried_pairs),
            "route_underexplored": route_underexplored,
            "allocation_only_candidates_vs_incumbent": allocation_only,
        },
        "selector": {
            "strategy": strategy,
            "strength": strength,
            "basis": basis,
        },
    }
    brief = {
        "version": "1.1",
        "mode": audit["mode"],
        "round": next_round,
        "state": {
            "rounds_since_progress": rounds_since_progress,
            "repetition": repetition,
        },
        "recommendation": {
            "strategy": strategy,
            "strength": strength,
            "basis": basis,
            "directive": _DIRECTIVES[strategy],
        },
    }
    return audit, brief


def build_global_brief(view: LLMScenarioView) -> dict:
    """Return only the paper prompt global projection."""

    return build_global_advisor(view)[1]


__all__ = ["build_global_advisor", "build_global_brief"]
