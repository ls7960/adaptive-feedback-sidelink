"""Positive-allowlist projection from evaluator scenario to LLM-visible data."""

from __future__ import annotations

from collections import Counter

from adaptive_feedback_llm.evaluator import evaluate_schedule, retain_stable_best
from adaptive_feedback_llm.exceptions import SolverConfigurationError
from adaptive_feedback_llm.guidance import build_static_guidance
from adaptive_feedback_llm.models import (
    EvaluatedCandidateRecord,
    DiagnosticCode,
    EvaluationResult,
    EvaluatorScenario,
    LLMFlowView,
    LLMGuidanceView,
    LLMHotspotView,
    LLMPathView,
    LLMFailureFeedback,
    LLMScenarioView,
    LLMStaticScenarioView,
    FlowQoSGap,
)


def derive_qos_gaps(evaluation) -> tuple[FlowQoSGap, ...]:
    """Derive explicit per-flow gaps only from official QoS diagnostics."""

    gaps: dict[int, dict[str, float | None]] = {}
    for diagnostic in evaluation.diagnostics:
        if diagnostic.code not in {
            DiagnosticCode.DELAY_QOS_VIOLATION,
            DiagnosticCode.RELIABILITY_QOS_VIOLATION,
        }:
            continue
        if diagnostic.required is None or diagnostic.actual is None:
            raise SolverConfigurationError(
                "official QoS diagnostic lacks required/actual values"
            )
        for flow_id in diagnostic.flow_ids:
            item = gaps.setdefault(
                flow_id, {"delay_gap_s": None, "reliability_gap": None}
            )
            if diagnostic.code is DiagnosticCode.DELAY_QOS_VIOLATION:
                item["delay_gap_s"] = max(
                    0.0, float(diagnostic.actual) - float(diagnostic.required)
                )
            else:
                item["reliability_gap"] = max(
                    0.0, float(diagnostic.required) - float(diagnostic.actual)
                )
    return tuple(
        FlowQoSGap(flow_id=flow_id, **values)
        for flow_id, values in sorted(gaps.items())
    )


def validate_evaluated_history(
    scenario: EvaluatorScenario,
    history: tuple[EvaluatedCandidateRecord, ...],
) -> None:
    """Re-evaluate alleged official records before exposing exact metrics."""

    previous_position = (0, 0)
    seen_positions: set[tuple[int, int]] = set()
    seen_candidate_ids: set[str] = set()
    official_records: list[tuple[EvaluatedCandidateRecord, EvaluationResult]] = []
    for record in history:
        position = (record.round_index, record.call_index)
        expected_candidate_id = f"r{record.round_index}c{record.call_index}"
        if record.candidate_id != expected_candidate_id:
            raise SolverConfigurationError(
                f"history candidate identity {record.candidate_id!r} is invalid"
            )
        if position in seen_positions or record.candidate_id in seen_candidate_ids:
            raise SolverConfigurationError("history candidate identities must be unique")
        if position <= previous_position:
            raise SolverConfigurationError(
                "history records must be strictly ordered by round and call"
            )
        previous_position = position
        seen_positions.add(position)
        seen_candidate_ids.add(record.candidate_id)
        if record.scenario_id != scenario.scenario_id:
            raise SolverConfigurationError(
                f"history scenario {record.scenario_id!r} does not match current scenario"
            )
        official = evaluate_schedule(scenario, record.schedule)
        if official != record.evaluation:
            raise SolverConfigurationError(
                f"history candidate {record.candidate_id!r} evaluation is not official"
            )
        if derive_qos_gaps(official) != record.qos_gaps:
            raise SolverConfigurationError(
                f"history candidate {record.candidate_id!r} QoS gaps are not official"
            )
        official_records.append((record, official))

    stable_record: EvaluatedCandidateRecord | None = None
    stable_evaluation = None
    for record, official in official_records:
        if not official.feasible:
            continue
        retained = retain_stable_best(
            stable_evaluation,
            official,
            tolerance=scenario.numerics.float_tolerance,
        )
        if retained is official:
            stable_evaluation = official
            stable_record = record
    marked = tuple(record for record in history if record.best_so_far)
    if stable_record is None:
        if marked:
            raise SolverConfigurationError(
                "history without a feasible candidate cannot mark a best record"
            )
    elif len(marked) != 1 or marked[0] is not stable_record:
        raise SolverConfigurationError(
            "history best marker must identify the stable first official optimum"
        )


def build_llm_scenario_view(
    scenario: EvaluatorScenario,
    evaluated_history: tuple[EvaluatedCandidateRecord, ...] = (),
    failure_feedback: tuple[LLMFailureFeedback, ...] = (),
) -> LLMScenarioView:
    """Project evaluator-private state through a positive allowlist."""

    short = tuple(
        catalog.flow_id for catalog in scenario.path_catalogs if len(catalog.paths) != 5
    )
    if short:
        raise SolverConfigurationError(
            f"Adaptive LLM view requires exactly five paths per flow; invalid flows={short}"
        )
    validate_evaluated_history(scenario, evaluated_history)

    node_index = {node_id: index for index, node_id in enumerate(scenario.node_ids)}
    matrix = [[0 for _ in scenario.node_ids] for _ in scenario.node_ids]
    for link in scenario.links:
        matrix[node_index[link.src]][node_index[link.dst]] = link.capacity_srus

    counts = Counter(
        edge
        for catalog in scenario.path_catalogs
        for path in catalog.paths
        for edge in path.edges
    )
    links = scenario.link_by_edge()
    guidance = build_static_guidance(scenario)
    return LLMScenarioView(
        static=LLMStaticScenarioView(
            scenario_id=scenario.scenario_id,
            node_ids=scenario.node_ids,
            capacity_adjacency_matrix=tuple(tuple(row) for row in matrix),
            flows=tuple(LLMFlowView.model_validate(flow.model_dump()) for flow in scenario.flows),
            paths=tuple(
                LLMPathView(
                    flow_id=path.flow_id,
                    path_id=path.path_id,
                    node_sequence=path.node_sequence,
                )
                for catalog in scenario.path_catalogs
                for path in catalog.paths
            ),
            hotspots=tuple(
                LLMHotspotView(
                    src=edge[0],
                    dst=edge[1],
                    catalog_occurrences=count,
                    capacity_srus=links[edge].capacity_srus,
                    static_load=count / links[edge].capacity_srus,
                )
                for edge, count in sorted(counts.items())
            ),
            guidance=tuple(
                LLMGuidanceView(
                    flow_id=item.flow_id,
                    path_id=item.path_id,
                    rank=item.rank,
                    single_flow_feasible=item.single_flow_feasible,
                    required_sru_total=item.required_sru_total,
                    reference_sru_allocations=item.reference_sru_allocations,
                    hotspot_burden=item.hotspot_burden,
                    class_quality_rank=item.class_quality_rank,
                    hop_count=item.hop_count,
                )
                for item in guidance.paths
            ),
        ),
        evaluated_history=evaluated_history,
        failure_feedback=failure_feedback,
    )
