"""The sole deterministic feasibility and objective evaluator."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from adaptive_feedback_llm.exceptions import EvaluationInvariantError
from adaptive_feedback_llm.models.domain import (
    CandidatePath,
    EvaluatorScenario,
    Flow,
    FlowDecision,
    Schedule,
)
from adaptive_feedback_llm.models.results import (
    Diagnostic,
    DiagnosticCode,
    EvaluationResult,
    FlowEvaluation,
    LinkUsage,
)
from adaptive_feedback_llm.physics import (
    bottleneck_rate_bps,
    classified_margin,
    flow_utility,
    link_success_probability,
    path_reliability,
    pipelined_completion_delay_s,
    rate_per_sru_bps,
)


@dataclass(frozen=True, slots=True)
class _ComputedFlow:
    flow: Flow
    decision: FlowDecision
    path: CandidatePath
    bottleneck_rate_bps: float
    completion_delay_s: float
    reliability: float
    delay_qos_met: bool
    reliability_qos_met: bool


def _diagnostic(
    code: DiagnosticCode,
    message: str,
    *,
    flow_ids: tuple[int, ...] = (),
    link: tuple[int, int] | None = None,
    required: float | int | None = None,
    actual: float | int | None = None,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        message=message,
        flow_ids=flow_ids,
        link=link,
        required=required,
        actual=actual,
    )


def evaluate_schedule(
    scenario: EvaluatorScenario,
    schedule: Schedule,
) -> EvaluationResult:
    """Evaluate one schedule without repair or input mutation.

    Structural semantics are checked before shared capacity, physics, both QoS
    constraints, and finally the unscaled class-continuous objective.
    """

    diagnostics: list[Diagnostic] = []
    flow_by_id = scenario.flow_by_id()
    catalog_by_flow = scenario.catalog_by_flow_id()
    link_by_edge = scenario.link_by_edge()
    decisions_by_flow: dict[int, list[FlowDecision]] = defaultdict(list)
    for decision in schedule.flows:
        decisions_by_flow[decision.flow_id].append(decision)

    # Step 2: every scenario flow exactly once and no unknown flow.
    for flow in scenario.flows:
        count = len(decisions_by_flow.get(flow.flow_id, ()))
        if count == 0:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.MISSING_FLOW,
                    f"flow {flow.flow_id} is missing from the candidate",
                    flow_ids=(flow.flow_id,),
                    required=1,
                    actual=0,
                )
            )
        elif count > 1:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.DUPLICATE_FLOW,
                    f"flow {flow.flow_id} occurs {count} times",
                    flow_ids=(flow.flow_id,),
                    required=1,
                    actual=count,
                )
            )

    seen_unknown: set[int] = set()
    for decision in schedule.flows:
        if decision.flow_id not in flow_by_id and decision.flow_id not in seen_unknown:
            seen_unknown.add(decision.flow_id)
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.UNKNOWN_FLOW,
                    f"candidate contains unknown flow {decision.flow_id}",
                    flow_ids=(decision.flow_id,),
                )
            )

    valid_decisions: dict[int, tuple[FlowDecision, CandidatePath | None]] = {}
    for flow in scenario.flows:
        decisions = decisions_by_flow.get(flow.flow_id, ())
        if len(decisions) != 1:
            continue
        decision = decisions[0]

        # Step 3: served/path/unserved semantics.
        if not decision.served:
            if decision.path_id != -1 or decision.sru_allocations:
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.INVALID_UNSERVED_STATE,
                        f"unserved flow {flow.flow_id} must use path_id=-1 and no SRUs",
                        flow_ids=(flow.flow_id,),
                    )
                )
            else:
                valid_decisions[flow.flow_id] = (decision, None)
            continue

        if decision.path_id < 0:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.INVALID_PATH_ID,
                    f"served flow {flow.flow_id} has negative path_id",
                    flow_ids=(flow.flow_id,),
                    required=0,
                    actual=decision.path_id,
                )
            )
            continue

        # Step 4: catalog membership and directed continuity.
        path = catalog_by_flow[flow.flow_id].get_path(decision.path_id)
        if path is None:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.PATH_NOT_IN_CATALOG,
                    f"path_id {decision.path_id} is not in flow {flow.flow_id}'s catalog",
                    flow_ids=(flow.flow_id,),
                    actual=decision.path_id,
                )
            )
            continue
        if (
            path.node_sequence[0] != flow.source
            or path.node_sequence[-1] != flow.target
            or len(set(path.node_sequence)) != len(path.node_sequence)
            or any(edge not in link_by_edge for edge in path.edges)
        ):
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.INVALID_CATALOG_PATH,
                    f"catalog path {decision.path_id} for flow {flow.flow_id} is invalid",
                    flow_ids=(flow.flow_id,),
                )
            )
            continue

        # Step 5: exact per-edge allocation coverage, uniqueness, and positivity.
        path_edges = set(path.edges)
        allocation_counts = Counter(allocation.edge for allocation in decision.sru_allocations)
        for allocation in decision.sru_allocations:
            if allocation.num_srus <= 0:
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.INVALID_SRU_COUNT,
                        f"flow {flow.flow_id} has a non-positive SRU allocation",
                        flow_ids=(flow.flow_id,),
                        link=allocation.edge,
                        required=1,
                        actual=allocation.num_srus,
                    )
                )
        for edge, count in allocation_counts.items():
            if count > 1:
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.DUPLICATE_ALLOCATION,
                        f"flow {flow.flow_id} allocates edge {edge} {count} times",
                        flow_ids=(flow.flow_id,),
                        link=edge,
                        required=1,
                        actual=count,
                    )
                )
            if edge not in path_edges:
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.PATH_EXTERNAL_ALLOCATION,
                        f"flow {flow.flow_id} allocates edge {edge} outside its path",
                        flow_ids=(flow.flow_id,),
                        link=edge,
                    )
                )
        for edge in path.edges:
            if allocation_counts.get(edge, 0) == 0:
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.MISSING_ALLOCATION,
                        f"flow {flow.flow_id} omits allocation for path edge {edge}",
                        flow_ids=(flow.flow_id,),
                        link=edge,
                    )
                )
        valid_decisions[flow.flow_id] = (decision, path)

    # Capacity and physics are undefined until the whole candidate is structurally sound.
    if diagnostics:
        return EvaluationResult(
            feasible=False,
            objective=None,
            diagnostics=tuple(diagnostics),
            flow_evaluations=(),
            link_usage=(),
        )

    # Step 6: aggregate and check directed-link capacity.
    used_srus: dict[tuple[int, int], int] = defaultdict(int)
    using_flows: dict[tuple[int, int], list[int]] = defaultdict(list)
    for flow in scenario.flows:
        decision, path = valid_decisions[flow.flow_id]
        if path is None:
            continue
        for allocation in decision.sru_allocations:
            used_srus[allocation.edge] += allocation.num_srus
            using_flows[allocation.edge].append(flow.flow_id)

    link_usage: list[LinkUsage] = []
    for link in scenario.links:
        used = used_srus.get(link.edge, 0)
        if used == 0:
            continue
        flow_ids = tuple(using_flows[link.edge])
        link_usage.append(
            LinkUsage(
                src=link.src,
                dst=link.dst,
                used_srus=used,
                capacity_srus=link.capacity_srus,
                flow_ids=flow_ids,
            )
        )
        if used > link.capacity_srus:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.LINK_CAPACITY_EXCEEDED,
                    f"directed link {link.edge} uses {used}/{link.capacity_srus} SRUs",
                    flow_ids=flow_ids,
                    link=link.edge,
                    required=link.capacity_srus,
                    actual=used,
                )
            )

    # Steps 7-8: evaluator-only physical metrics and both QoS checks.
    computed: dict[int, _ComputedFlow] = {}
    preliminary_evaluations: list[FlowEvaluation] = []
    tolerance = scenario.numerics.float_tolerance
    for flow in scenario.flows:
        decision, path = valid_decisions[flow.flow_id]
        if path is None:
            preliminary_evaluations.append(
                FlowEvaluation(
                    flow_id=flow.flow_id,
                    served=False,
                    path_id=-1,
                    utility=0.0,
                )
            )
            continue

        allocations = {allocation.edge: allocation.num_srus for allocation in decision.sru_allocations}
        try:
            hop_rates = tuple(
                allocations[edge]
                * rate_per_sru_bps(
                    link_by_edge[edge].sinr_db,
                    scenario.physics.sru_bandwidth_hz,
                )
                for edge in path.edges
            )
            bottleneck = bottleneck_rate_bps(hop_rates)
            completion_delay = pipelined_completion_delay_s(
                flow.data_bits,
                bottleneck,
                (link_by_edge[edge].fixed_delay_s for edge in path.edges),
            )
            reliability = path_reliability(
                link_success_probability(
                    link_by_edge[edge].sinr_db,
                    scenario.physics.operating_sinr_db,
                )
                for edge in path.edges
            )
        except (ValueError, OverflowError) as exc:
            raise EvaluationInvariantError(
                f"validated scenario produced invalid physics for flow {flow.flow_id}: {exc}"
            ) from exc

        delay_met = completion_delay <= flow.deadline_s + tolerance
        reliability_met = reliability + tolerance >= flow.min_reliability
        computed[flow.flow_id] = _ComputedFlow(
            flow=flow,
            decision=decision,
            path=path,
            bottleneck_rate_bps=bottleneck,
            completion_delay_s=completion_delay,
            reliability=reliability,
            delay_qos_met=delay_met,
            reliability_qos_met=reliability_met,
        )
        preliminary_evaluations.append(
            FlowEvaluation(
                flow_id=flow.flow_id,
                served=True,
                path_id=path.path_id,
                bottleneck_rate_bps=bottleneck,
                completion_delay_s=completion_delay,
                reliability=reliability,
                delay_qos_met=delay_met,
                reliability_qos_met=reliability_met,
            )
        )
        if not delay_met:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.DELAY_QOS_VIOLATION,
                    f"flow {flow.flow_id} completion delay exceeds its deadline",
                    flow_ids=(flow.flow_id,),
                    required=flow.deadline_s,
                    actual=completion_delay,
                )
            )
        if not reliability_met:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.RELIABILITY_QOS_VIOLATION,
                    f"flow {flow.flow_id} reliability is below its threshold",
                    flow_ids=(flow.flow_id,),
                    required=flow.min_reliability,
                    actual=reliability,
                )
            )

    if diagnostics:
        return EvaluationResult(
            feasible=False,
            objective=None,
            diagnostics=tuple(diagnostics),
            flow_evaluations=tuple(preliminary_evaluations),
            link_usage=tuple(link_usage),
        )

    # Only a completely feasible candidate receives margins and an objective.
    final_evaluations: list[FlowEvaluation] = []
    utilities: list[float] = []
    for preliminary in preliminary_evaluations:
        if not preliminary.served:
            final_evaluations.append(preliminary)
            utilities.append(0.0)
            continue
        item = computed[preliminary.flow_id]
        try:
            margin = classified_margin(
                item.flow.traffic_class,
                completion_delay_s=item.completion_delay_s,
                deadline_s=item.flow.deadline_s,
                reliability=item.reliability,
                minimum_reliability=item.flow.min_reliability,
                tolerance=tolerance,
            )
            utility = flow_utility(item.flow.priority, margin)
        except (ValueError, OverflowError) as exc:
            raise EvaluationInvariantError(
                f"validated scenario produced invalid utility for flow "
                f"{item.flow.flow_id}: {exc}"
            ) from exc
        utilities.append(utility)
        final_evaluations.append(
            FlowEvaluation(
                flow_id=preliminary.flow_id,
                served=True,
                path_id=preliminary.path_id,
                bottleneck_rate_bps=preliminary.bottleneck_rate_bps,
                completion_delay_s=preliminary.completion_delay_s,
                reliability=preliminary.reliability,
                delay_qos_met=preliminary.delay_qos_met,
                reliability_qos_met=preliminary.reliability_qos_met,
                margin=margin,
                utility=utility,
            )
        )

    try:
        objective = math.fsum(utilities)
    except (OverflowError, ValueError) as exc:
        raise EvaluationInvariantError(
            f"feasible flow utilities cannot form a finite objective: {exc}"
        ) from exc
    if not math.isfinite(objective):
        raise EvaluationInvariantError("feasible objective is not finite")

    return EvaluationResult(
        feasible=True,
        objective=objective,
        diagnostics=(),
        flow_evaluations=tuple(final_evaluations),
        link_usage=tuple(link_usage),
    )


def retain_stable_best(
    incumbent: EvaluationResult | None,
    challenger: EvaluationResult,
    *,
    tolerance: float,
) -> EvaluationResult:
    """Prefer feasibility and greater objective; retain first within tolerance."""

    if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)):
        raise TypeError("tolerance must be a real number")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if incumbent is None:
        return challenger
    if challenger.feasible and not incumbent.feasible:
        return challenger
    if incumbent.feasible and not challenger.feasible:
        return incumbent
    if not incumbent.feasible:
        return incumbent
    if incumbent.objective is None or challenger.objective is None:
        raise EvaluationInvariantError("feasible result is missing an objective")
    if challenger.objective > incumbent.objective + tolerance_value:
        return challenger
    return incumbent
