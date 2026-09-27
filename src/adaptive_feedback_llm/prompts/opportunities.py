"""Deterministic, nonbinding facts derived ONLY from an existing public view.

No evaluator call, hidden physics, candidate construction, or predicted score.
The implementation is included in the 1.4 prompt bundle digest.
"""
from __future__ import annotations

from collections import Counter

from adaptive_feedback_llm.models import LLMScenarioView, Schedule


def _schedule_key(schedule: Schedule) -> tuple:
    """Ignore JSON/flow/edge order without ignoring semantic decisions."""
    return tuple(sorted(
        (flow.flow_id, flow.served, flow.path_id,
         tuple(sorted((a.src, a.dst, a.num_srus) for a in flow.sru_allocations)))
        for flow in schedule.flows
    ))


def build_optimization_facts(view: LLMScenarioView) -> dict:
    """Explain the marked incumbent and previously evaluated search footprint."""
    best = next((r for r in view.evaluated_history if r.best_so_far), None)
    if best is None or not best.evaluation.feasible or not any(
        flow.served for flow in best.schedule.flows
    ):
        return {"schema_version": "1.0", "status": "no_served_feasible_best"}

    static = view.static
    genes = {f.flow_id: f for f in best.schedule.flows}
    caps = {(src, dst): static.capacity_adjacency_matrix[i][j]
            for i, src in enumerate(static.node_ids)
            for j, dst in enumerate(static.node_ids)}
    usage: Counter = Counter()
    owners: dict = {}
    for gene in best.schedule.flows:
        if gene.served:
            for a in gene.sru_allocations:
                usage[a.edge] += a.num_srus
                owners.setdefault(a.edge, []).append(gene.flow_id)

    def path_reference_facts(ref, *, release_flow_id=None):
        released = {
            a.edge: a.num_srus for a in genes[release_flow_id].sru_allocations
        } if release_flow_id is not None else {}
        deficits = []
        for a in ref.reference_sru_allocations:
            residual = caps[a.edge] - usage[a.edge] + released.get(a.edge, 0)
            if a.num_srus > residual:
                deficits.append({
                    "src": a.src, "dst": a.dst,
                    "reference_required_srus": a.num_srus,
                    "residual_after_releasing_this_flow_srus": residual,
                    "shortfall_srus": a.num_srus - residual,
                    "other_incumbent_flow_ids": sorted(
                        f for f in owners.get(a.edge, []) if f != release_flow_id
                    ),
                })
        return {"path_id": ref.path_id, "static_rank": ref.rank,
                "class_quality_rank": ref.class_quality_rank,
                "reference_fits_residual_capacity": not deficits,
                "reference_capacity_shortfalls": deficits}

    rejected = []
    served_paths = []
    highest_priority = max(f.priority for f in static.flows)
    for f in sorted(static.flows, key=lambda f: (-f.priority, f.flow_id)):
        gene = genes[f.flow_id]
        refs = [r for r in static.guidance if r.flow_id == f.flow_id and r.single_flow_feasible]
        if not gene.served:
            rejected.append({"flow_id": f.flow_id, "priority": f.priority,
                "traffic_class": f.traffic_class.value,
                "is_highest_scenario_priority": f.priority == highest_priority,
                "single_flow_feasible_references": [path_reference_facts(r) for r in refs]})
        else:
            served_paths.append({"flow_id": f.flow_id, "priority": f.priority,
                "traffic_class": f.traffic_class.value, "path_id": gene.path_id,
                "selected_edges": [
                    {"src": a.src, "dst": a.dst, "this_flow_srus": a.num_srus,
                     "joint_used_srus": usage[a.edge], "capacity_srus": caps[a.edge],
                     "residual_srus": caps[a.edge] - usage[a.edge]}
                    for a in gene.sru_allocations
                ],
                "alternative_references_after_releasing_this_flow": [
                    path_reference_facts(r, release_flow_id=f.flow_id)
                    for r in refs if r.path_id != gene.path_id
                ]})

    seen: dict = {}
    route_patterns = set()
    comparisons = []
    for r in view.evaluated_history:
        signature = _schedule_key(r.schedule)
        repeat_of = seen.get(signature)
        seen.setdefault(signature, r.candidate_id)
        route_patterns.add(tuple((x[0], x[1], x[2]) for x in signature))
        supplied_ids = Counter(f.flow_id for f in r.schedule.flows)
        changes = {"admitted": [], "rejected": [], "rerouted": [], "allocation_changed": [],
                   "missing_flow_ids": sorted(set(genes) - set(supplied_ids)),
                   "unknown_flow_ids": sorted(set(supplied_ids) - set(genes)),
                   "duplicate_flow_ids": sorted(f for f, count in supplied_ids.items() if count > 1)}
        for other in r.schedule.flows:
            anchor = genes.get(other.flow_id)
            if anchor is None:
                continue  # Invalid submitted candidates may contain unknown flow IDs.
            if other.served and not anchor.served:
                changes["admitted"].append(other.flow_id)
            elif anchor.served and not other.served:
                changes["rejected"].append(other.flow_id)
            elif anchor.served and other.served:
                if other.path_id != anchor.path_id:
                    changes["rerouted"].append(other.flow_id)
                elif sorted((a.src, a.dst, a.num_srus) for a in other.sru_allocations) != sorted(
                    (a.src, a.dst, a.num_srus) for a in anchor.sru_allocations
                ):
                    changes["allocation_changed"].append(other.flow_id)
        comparisons.append({"candidate_id": r.candidate_id,
            "feasible": r.evaluation.feasible,
            "delta_j_vs_incumbent": (
                r.evaluation.objective - best.evaluation.objective
                if r.evaluation.feasible else None),
            "repeat_of_candidate_id": repeat_of,
            "changes_vs_incumbent": changes,
            "diagnostic_codes": sorted({d.code.value for d in r.evaluation.diagnostics})})

    return {"schema_version": "1.0", "status": "active",
        "anchor_candidate_id": best.candidate_id,
        "nonbinding": True,
        "history_coverage": {"evaluated_candidates": len(comparisons),
            "distinct_complete_schedules": len(seen),
            "repeated_complete_schedules": len(comparisons) - len(seen),
            "distinct_admission_route_patterns": len(route_patterns)},
        "candidate_comparisons": comparisons,
        "unserved_flows_by_priority": rejected,
        "served_path_capacity": served_paths}
