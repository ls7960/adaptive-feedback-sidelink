"""paper prompt non-prescriptive evidence matrix derived from the public view.

The matrix exposes exact historical observations and symmetric public capacity
relationships.  It never constructs a candidate, recommends a path/allocation,
calls the evaluator/provider, predicts J, or reads private link parameters.
"""
from __future__ import annotations

from collections import Counter

from adaptive_feedback_llm.models import LLMScenarioView
from adaptive_feedback_llm.prompts.opportunities import _schedule_key


_QOS_CODES = {"delay_qos_violation", "reliability_qos_violation"}


def _path_edges(view: LLMScenarioView) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    return {
        (path.flow_id, path.path_id): tuple(
            zip(path.node_sequence[:-1], path.node_sequence[1:], strict=True)
        )
        for path in view.static.paths
    }


def _structurally_usable_schedule(view: LLMScenarioView, record) -> bool:
    """Check only public Schedule/path structure needed for capacity arithmetic."""

    flow_ids = {flow.flow_id for flow in view.static.flows}
    decisions = record.schedule.flows
    if len(decisions) != len(flow_ids) or {gene.flow_id for gene in decisions} != flow_ids:
        return False
    paths = _path_edges(view)
    for gene in decisions:
        if not gene.served:
            if gene.path_id != -1 or gene.sru_allocations:
                return False
            continue
        expected = paths.get((gene.flow_id, gene.path_id))
        actual = tuple(allocation.edge for allocation in gene.sru_allocations)
        if expected is None or actual != expected:
            return False
        if any(allocation.num_srus <= 0 for allocation in gene.sru_allocations):
            return False
    return True


def build_evidence_matrix(view: LLMScenarioView) -> dict:
    """Build canonically ordered facts without an action menu or target values."""

    static, history = view.static, view.evaluated_history
    flows = {flow.flow_id: flow for flow in static.flows}
    paths = _path_edges(view)
    guidance = {(item.flow_id, item.path_id): item for item in static.guidance}
    capacities = {
        (src, dst): static.capacity_adjacency_matrix[i][j]
        for i, src in enumerate(static.node_ids)
        for j, dst in enumerate(static.node_ids)
    }

    schedule_groups: dict[tuple, dict] = {}
    qos_patterns: dict[tuple, dict] = {}
    capacity_conflicts: dict[tuple, dict] = {}
    for record in history:
        schedule_key = _schedule_key(record.schedule)
        group = schedule_groups.setdefault(schedule_key, {
            "first_candidate_id": record.candidate_id,
            "last_candidate_id": record.candidate_id,
            "submission_count": 0,
            "first_round_index": record.round_index,
            "last_round_index": record.round_index,
            "feasible": record.evaluation.feasible,
            "diagnostic_codes": sorted({item.code.value for item in record.evaluation.diagnostics}),
        })
        group.update(last_candidate_id=record.candidate_id, last_round_index=record.round_index)
        group["submission_count"] += 1

        diagnostic_codes_by_flow = {
            flow_id: sorted({
                item.code.value for item in record.evaluation.diagnostics
                if flow_id in item.flow_ids
            })
            for flow_id in flows
        }
        decisions = {gene.flow_id: gene for gene in record.schedule.flows if gene.flow_id in flows}
        for flow_id, gene in sorted(decisions.items()):
            expected_edges = paths.get((flow_id, gene.path_id))
            allocation_edges = tuple(item.edge for item in gene.sru_allocations)
            if not gene.served or expected_edges is None or allocation_edges != expected_edges:
                continue
            allocation = tuple(item.num_srus for item in gene.sru_allocations)
            if any(value <= 0 for value in allocation):
                continue
            codes = diagnostic_codes_by_flow[flow_id]
            qos_codes = sorted(set(codes) & _QOS_CODES)
            if qos_codes:
                pattern_key = (flow_id, gene.path_id, allocation, tuple(qos_codes))
                pattern = qos_patterns.setdefault(pattern_key, {
                    "flow_id": flow_id,
                    "priority": flows[flow_id].priority,
                    "path_id": gene.path_id,
                    "observed_sru_allocations": [
                        item.model_dump(mode="json") for item in gene.sru_allocations
                    ],
                    "diagnostic_codes": qos_codes,
                    "observed_count": 0,
                    "first_candidate_id": record.candidate_id,
                    "scope": "exact_flow_path_allocation_and_joint_candidate_context_only",
                })
                pattern["observed_count"] += 1
                pattern["last_candidate_id"] = record.candidate_id

        for diagnostic in record.evaluation.diagnostics:
            if diagnostic.code.value != "link_capacity_exceeded" or diagnostic.link is None:
                continue
            users = []
            for flow_id in sorted(fid for fid in diagnostic.flow_ids if fid in decisions):
                claimed = sum(
                    item.num_srus for item in decisions[flow_id].sru_allocations
                    if item.edge == diagnostic.link
                )
                users.append({
                    "flow_id": flow_id,
                    "priority": flows[flow_id].priority,
                    "observed_srus": claimed,
                })
            conflict_key = (
                diagnostic.link, diagnostic.actual, diagnostic.required,
                tuple((item["flow_id"], item["observed_srus"]) for item in users),
            )
            conflict = capacity_conflicts.setdefault(conflict_key, {
                "src": diagnostic.link[0],
                "dst": diagnostic.link[1],
                "official_actual_srus": diagnostic.actual,
                "official_capacity_srus": diagnostic.required,
                "observed_users": users,
                "observed_count": 0,
                "first_candidate_id": record.candidate_id,
                "scope": "exact_observed_joint_candidate_context_only",
            })
            conflict["observed_count"] += 1
            conflict["last_candidate_id"] = record.candidate_id

    best = next((record for record in history if record.best_so_far), None)
    useful = bool(
        best is not None and best.evaluation.feasible
        and any(gene.served for gene in best.schedule.flows)
    )
    latest = history[-1] if history else None
    anchor = best if best is not None else latest
    anchor_role = "official_incumbent" if best is not None else (
        "latest_evaluated_not_incumbent" if latest is not None else "none"
    )
    anchor_usable = anchor is not None and _structurally_usable_schedule(view, anchor)

    latest_round = max(
        (item.round_index for item in (*history, *view.failure_feedback)), default=None
    )
    latest_records = [item for item in history if item.round_index == latest_round]
    older_keys = {
        _schedule_key(item.schedule) for item in history if item.round_index != latest_round
    }

    path_capacity_evidence = []
    if anchor_usable:
        usage: Counter = Counter()
        owners: dict[tuple[int, int], Counter] = {}
        decisions = {gene.flow_id: gene for gene in anchor.schedule.flows}
        for gene in anchor.schedule.flows:
            if not gene.served:
                continue
            for allocation in gene.sru_allocations:
                usage[allocation.edge] += allocation.num_srus
                owners.setdefault(allocation.edge, Counter())[gene.flow_id] += allocation.num_srus

        for flow_id in sorted(flows):
            gene = decisions[flow_id]
            own_usage = Counter(
                {item.edge: item.num_srus for item in gene.sru_allocations}
                if gene.served else {}
            )
            flow_entry = {
                "flow_id": flow_id,
                "priority": flows[flow_id].priority,
                "traffic_class": flows[flow_id].traffic_class.value,
                "anchor_served": gene.served,
                "anchor_selected_path_id": gene.path_id if gene.served else None,
                "paths": [],
            }
            for path_id in sorted(pid for fid, pid in paths if fid == flow_id):
                ref = guidance[(flow_id, path_id)]
                edge_facts = []
                for src, dst in paths[(flow_id, path_id)]:
                    edge = (src, dst)
                    other_usage = usage[edge] - own_usage[edge]
                    other_users = [
                        {
                            "flow_id": owner_id,
                            "priority": flows[owner_id].priority,
                            "claimed_srus": claimed,
                        }
                        for owner_id, claimed in sorted(owners.get(edge, {}).items())
                        if owner_id != flow_id
                    ]
                    edge_facts.append({
                        "src": src,
                        "dst": dst,
                        "capacity_srus": capacities[edge],
                        "other_flow_usage_srus": other_usage,
                        "available_to_target_srus": max(0, capacities[edge] - other_usage),
                        "other_usage_exceeds_capacity_by_srus": max(
                            0, other_usage - capacities[edge]
                        ),
                        "other_users": other_users,
                    })
                flow_entry["paths"].append({
                    "path_id": path_id,
                    "single_flow_feasible": ref.single_flow_feasible,
                    "anchor_selected": gene.served and gene.path_id == path_id,
                    "edge_capacity_envelope_after_releasing_target": edge_facts,
                })
            path_capacity_evidence.append(flow_entry)

    return {
        "schema_version": "1.0",
        "nonprescriptive": True,
        "display_order": "canonical_flow_id_path_id_edge_order_not_a_search_ranking",
        "mode": "improve_feasible" if useful else "construct_feasible",
        "anchor": {
            "candidate_id": anchor.candidate_id if anchor is not None else None,
            "role": anchor_role,
            "schedule_usable_for_public_capacity_arithmetic": anchor_usable,
        },
        "history_coverage": {
            "evaluated_calls": len(history),
            "distinct_complete_schedules": len(schedule_groups),
            "repeated_calls": len(history) - len(schedule_groups),
            "latest_round_index": latest_round,
            "latest_round_evaluated_calls": len(latest_records),
            "latest_round_repeats_of_older_rounds": sum(
                _schedule_key(item.schedule) in older_keys for item in latest_records
            ),
        },
        "repeated_schedule_groups": [
            item for item in schedule_groups.values() if item["submission_count"] > 1
        ],
        "repeated_qos_failure_patterns": [
            item for item in qos_patterns.values() if item["observed_count"] > 1
        ],
        "observed_capacity_conflicts": list(capacity_conflicts.values()),
        "flow_path_capacity_evidence": path_capacity_evidence,
    }
