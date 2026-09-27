from __future__ import annotations

import math
import sys

import pytest

from adaptive_feedback_llm.evaluator import evaluate_schedule, retain_stable_best
from adaptive_feedback_llm.exceptions import EvaluationInvariantError
from adaptive_feedback_llm.models import (
    DiagnosticCode,
    EvaluationResult,
    EvaluatorScenario,
    Flow,
    FlowDecision,
    DirectedLink,
    Schedule,
    SRUAllocation,
    model_to_json,
)


def _decision(
    flow_id: int,
    *,
    served: bool,
    path_id: int,
    allocations: tuple[tuple[int, int, int], ...] = (),
) -> FlowDecision:
    return FlowDecision(
        flow_id=flow_id,
        served=served,
        path_id=path_id,
        sru_allocations=tuple(
            SRUAllocation(src=src, dst=dst, num_srus=count)
            for src, dst, count in allocations
        ),
    )


def _codes(result: EvaluationResult) -> set[DiagnosticCode]:
    return {diagnostic.code for diagnostic in result.diagnostics}


def _scenario_with_flow_update(
    scenario: EvaluatorScenario, flow_id: int, **updates: object
) -> EvaluatorScenario:
    flows = []
    for flow in scenario.flows:
        if flow.flow_id == flow_id:
            data = flow.model_dump(mode="python")
            data.update(updates)
            flow = Flow.model_validate(data)
        flows.append(flow)
    data = scenario.model_dump(mode="python")
    data["flows"] = tuple(flows)
    return EvaluatorScenario.model_validate(data)


def test_valid_candidate_computes_unscaled_classified_objective(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    result = evaluate_schedule(fixed_scenario, valid_schedule)

    assert result.feasible is True
    assert result.diagnostics == ()
    assert result.objective == pytest.approx(
        math.fsum(item.utility or 0.0 for item in result.flow_evaluations)
    )
    assert 5.0 < result.objective < 10.0
    assert all(item.delay_qos_met for item in result.flow_evaluations)
    assert all(item.reliability_qos_met for item in result.flow_evaluations)
    assert result.flow_evaluations[0].margin == pytest.approx(
        1.0
        - result.flow_evaluations[0].completion_delay_s
        / fixed_scenario.flows[0].deadline_s
    )
    assert result.flow_evaluations[1].margin == pytest.approx(
        (
            result.flow_evaluations[1].reliability
            - fixed_scenario.flows[1].min_reliability
        )
        / (1.0 - fixed_scenario.flows[1].min_reliability)
    )


def test_all_unserved_is_feasible_with_zero_objective(
    fixed_scenario: EvaluatorScenario, all_unserved_schedule: Schedule
) -> None:
    result = evaluate_schedule(fixed_scenario, all_unserved_schedule)

    assert result.feasible is True
    assert result.objective == 0.0
    assert result.link_usage == ()
    assert tuple(item.utility for item in result.flow_evaluations) == (0.0, 0.0)


def test_missing_duplicate_and_unknown_flows_are_evaluator_diagnostics(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    duplicate = Schedule(
        flows=(valid_schedule.flows[0], valid_schedule.flows[0], valid_schedule.flows[1])
    )
    unknown = Schedule(
        flows=(
            valid_schedule.flows[0],
            valid_schedule.flows[1].model_copy(update={"flow_id": 99}),
        )
    )

    duplicate_result = evaluate_schedule(fixed_scenario, duplicate)
    unknown_result = evaluate_schedule(fixed_scenario, unknown)

    assert DiagnosticCode.DUPLICATE_FLOW in _codes(duplicate_result)
    assert DiagnosticCode.MISSING_FLOW in _codes(unknown_result)
    assert DiagnosticCode.UNKNOWN_FLOW in _codes(unknown_result)
    assert duplicate_result.objective is None
    assert unknown_result.objective is None


def test_unserved_semantics_are_not_silently_repaired(
    fixed_scenario: EvaluatorScenario, all_unserved_schedule: Schedule
) -> None:
    invalid = Schedule(
        flows=(
            all_unserved_schedule.flows[0].model_copy(update={"path_id": 0}),
            all_unserved_schedule.flows[1],
        )
    )
    before = model_to_json(invalid)

    result = evaluate_schedule(fixed_scenario, invalid)

    assert result.feasible is False
    assert result.objective is None
    assert _codes(result) == {DiagnosticCode.INVALID_UNSERVED_STATE}
    assert model_to_json(invalid) == before


@pytest.mark.parametrize(
    ("decision", "expected_code"),
    [
        (
            _decision(0, served=True, path_id=-1),
            DiagnosticCode.INVALID_PATH_ID,
        ),
        (
            _decision(0, served=True, path_id=9),
            DiagnosticCode.PATH_NOT_IN_CATALOG,
        ),
        (
            _decision(0, served=True, path_id=0),
            DiagnosticCode.MISSING_ALLOCATION,
        ),
        (
            _decision(
                0,
                served=True,
                path_id=0,
                allocations=((0, 2, 1), (0, 2, 1)),
            ),
            DiagnosticCode.DUPLICATE_ALLOCATION,
        ),
        (
            _decision(0, served=True, path_id=0, allocations=((0, 1, 1),)),
            DiagnosticCode.PATH_EXTERNAL_ALLOCATION,
        ),
        (
            _decision(0, served=True, path_id=0, allocations=((0, 2, 0),)),
            DiagnosticCode.INVALID_SRU_COUNT,
        ),
        (
            _decision(0, served=True, path_id=0, allocations=((0, 2, -1),)),
            DiagnosticCode.INVALID_SRU_COUNT,
        ),
    ],
)
def test_path_and_allocation_semantic_defects_reach_evaluator(
    fixed_scenario: EvaluatorScenario,
    all_unserved_schedule: Schedule,
    decision: FlowDecision,
    expected_code: DiagnosticCode,
) -> None:
    schedule = Schedule(flows=(decision, all_unserved_schedule.flows[1]))

    result = evaluate_schedule(fixed_scenario, schedule)

    assert result.feasible is False
    assert result.objective is None
    assert expected_code in _codes(result)


def test_directed_shared_capacity_is_aggregated_exactly(
    fixed_scenario: EvaluatorScenario,
) -> None:
    schedule = Schedule(
        flows=(
            _decision(0, served=True, path_id=0, allocations=((0, 2, 2),)),
            _decision(1, served=True, path_id=0, allocations=((0, 2, 1),)),
        )
    )

    result = evaluate_schedule(fixed_scenario, schedule)

    capacity = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is DiagnosticCode.LINK_CAPACITY_EXCEEDED
    )
    assert result.feasible is False
    assert result.objective is None
    assert capacity.link == (0, 2)
    assert capacity.required == 2
    assert capacity.actual == 3
    assert capacity.flow_ids == (0, 1)


def test_capacity_equal_to_directed_link_limit_is_feasible(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    result = evaluate_schedule(fixed_scenario, valid_schedule)
    direct_usage = next(item for item in result.link_usage if (item.src, item.dst) == (0, 2))

    assert result.feasible is True
    assert direct_usage.used_srus == direct_usage.capacity_srus == 2
    assert DiagnosticCode.LINK_CAPACITY_EXCEEDED not in _codes(result)


def test_reverse_edge_allocation_cannot_satisfy_forward_path(
    fixed_scenario: EvaluatorScenario, all_unserved_schedule: Schedule
) -> None:
    schedule = Schedule(
        flows=(
            _decision(0, served=True, path_id=0, allocations=((2, 0, 1),)),
            all_unserved_schedule.flows[1],
        )
    )

    result = evaluate_schedule(fixed_scenario, schedule)

    assert result.feasible is False
    assert DiagnosticCode.PATH_EXTERNAL_ALLOCATION in _codes(result)
    assert DiagnosticCode.MISSING_ALLOCATION in _codes(result)
    assert any(item.link == (2, 0) for item in result.diagnostics)


def test_declared_delay_qos_failure_rejects_entire_candidate(
    fixed_scenario: EvaluatorScenario,
    all_unserved_schedule: Schedule,
) -> None:
    schedule = Schedule(
        flows=(
            _decision(0, served=True, path_id=0, allocations=((0, 2, 1),)),
            all_unserved_schedule.flows[1],
        )
    )

    result = evaluate_schedule(fixed_scenario, schedule)

    assert result.feasible is False
    assert result.objective is None
    assert DiagnosticCode.DELAY_QOS_VIOLATION in _codes(result)
    assert result.flow_evaluations[0].utility is None


def test_declared_reliability_qos_failure_rejects_entire_candidate(
    fixed_scenario: EvaluatorScenario,
    all_unserved_schedule: Schedule,
) -> None:
    schedule = Schedule(
        flows=(
            all_unserved_schedule.flows[0],
            _decision(1, served=True, path_id=0, allocations=((0, 2, 1),)),
        )
    )

    result = evaluate_schedule(fixed_scenario, schedule)

    assert result.feasible is False
    assert result.objective is None
    assert DiagnosticCode.RELIABILITY_QOS_VIOLATION in _codes(result)


def test_both_qos_constraints_apply_to_both_traffic_classes(
    fixed_scenario: EvaluatorScenario,
    all_unserved_schedule: Schedule,
) -> None:
    delay_class_reliability_hard = _scenario_with_flow_update(
        fixed_scenario, 0, min_reliability=0.999
    )
    reliability_class_deadline_hard = _scenario_with_flow_update(
        fixed_scenario, 1, deadline_s=0.05
    )
    delay_class_schedule = Schedule(
        flows=(
            _decision(0, served=True, path_id=0, allocations=((0, 2, 2),)),
            all_unserved_schedule.flows[1],
        )
    )
    reliability_class_schedule = Schedule(
        flows=(
            all_unserved_schedule.flows[0],
            _decision(
                1,
                served=True,
                path_id=1,
                allocations=((0, 1, 1), (1, 2, 1)),
            ),
        )
    )

    delay_class_result = evaluate_schedule(
        delay_class_reliability_hard, delay_class_schedule
    )
    reliability_class_result = evaluate_schedule(
        reliability_class_deadline_hard, reliability_class_schedule
    )

    assert DiagnosticCode.RELIABILITY_QOS_VIOLATION in _codes(delay_class_result)
    assert DiagnosticCode.DELAY_QOS_VIOLATION in _codes(reliability_class_result)


def test_evaluator_float_tolerance_accepts_only_micro_delay_violation(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    baseline = evaluate_schedule(fixed_scenario, valid_schedule)
    delay = baseline.flow_evaluations[0].completion_delay_s
    assert delay is not None
    tolerance = fixed_scenario.numerics.float_tolerance
    within = _scenario_with_flow_update(
        fixed_scenario, 0, deadline_s=delay - tolerance / 2.0
    )
    outside = _scenario_with_flow_update(
        fixed_scenario, 0, deadline_s=delay - 2.0 * tolerance
    )

    within_result = evaluate_schedule(within, valid_schedule)
    outside_result = evaluate_schedule(outside, valid_schedule)

    assert within_result.feasible is True
    assert within_result.flow_evaluations[0].margin == 0.0
    assert outside_result.feasible is False
    assert DiagnosticCode.DELAY_QOS_VIOLATION in _codes(outside_result)


def test_evaluator_float_tolerance_accepts_only_micro_reliability_violation(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    baseline = evaluate_schedule(fixed_scenario, valid_schedule)
    reliability = baseline.flow_evaluations[1].reliability
    assert reliability is not None
    tolerance = fixed_scenario.numerics.float_tolerance
    within = _scenario_with_flow_update(
        fixed_scenario, 1, min_reliability=reliability + tolerance / 2.0
    )
    outside = _scenario_with_flow_update(
        fixed_scenario, 1, min_reliability=reliability + 2.0 * tolerance
    )

    within_result = evaluate_schedule(within, valid_schedule)
    outside_result = evaluate_schedule(outside, valid_schedule)

    assert within_result.feasible is True
    assert within_result.flow_evaluations[1].margin == 0.0
    assert outside_result.feasible is False
    assert DiagnosticCode.RELIABILITY_QOS_VIOLATION in _codes(outside_result)


def test_extra_nonbottleneck_sru_does_not_change_objective(
    fixed_scenario: EvaluatorScenario, all_unserved_schedule: Schedule
) -> None:
    base = Schedule(
        flows=(
            _decision(
                0,
                served=True,
                path_id=1,
                allocations=((0, 1, 1), (1, 2, 1)),
            ),
            all_unserved_schedule.flows[1],
        )
    )
    extra = Schedule(
        flows=(
            _decision(
                0,
                served=True,
                path_id=1,
                allocations=((0, 1, 2), (1, 2, 1)),
            ),
            all_unserved_schedule.flows[1],
        )
    )

    base_result = evaluate_schedule(fixed_scenario, base)
    extra_result = evaluate_schedule(fixed_scenario, extra)

    assert base_result.feasible and extra_result.feasible
    assert extra_result.objective == base_result.objective
    assert extra_result.objective == base_result.flow_evaluations[0].utility
    assert extra_result.objective < 100.0


def test_hop_count_is_not_an_objective_term(
    fixed_scenario: EvaluatorScenario, all_unserved_schedule: Schedule
) -> None:
    links = []
    for link in fixed_scenario.links:
        if link.edge == (0, 2):
            link = DirectedLink(
                src=0,
                dst=2,
                capacity_srus=2,
                sinr_db=10.0,
                fixed_delay_s=0.02,
            )
        links.append(link)
    data = fixed_scenario.model_dump(mode="python")
    data["links"] = tuple(links)
    equal_metric_scenario = EvaluatorScenario.model_validate(data)
    direct = Schedule(
        flows=(
            _decision(0, served=True, path_id=0, allocations=((0, 2, 1),)),
            all_unserved_schedule.flows[1],
        )
    )
    multihop = Schedule(
        flows=(
            _decision(
                0,
                served=True,
                path_id=1,
                allocations=((0, 1, 1), (1, 2, 1)),
            ),
            all_unserved_schedule.flows[1],
        )
    )

    direct_result = evaluate_schedule(equal_metric_scenario, direct)
    multihop_result = evaluate_schedule(equal_metric_scenario, multihop)

    assert direct_result.feasible and multihop_result.feasible
    assert direct_result.flow_evaluations[0].completion_delay_s == pytest.approx(
        multihop_result.flow_evaluations[0].completion_delay_s
    )
    assert direct_result.objective == pytest.approx(multihop_result.objective)


def test_stable_comparison_retains_first_seen_tie_and_prefers_feasibility(
    fixed_scenario: EvaluatorScenario,
    valid_schedule: Schedule,
    all_unserved_schedule: Schedule,
) -> None:
    first = evaluate_schedule(fixed_scenario, valid_schedule)
    equal = evaluate_schedule(fixed_scenario, valid_schedule)
    zero = evaluate_schedule(fixed_scenario, all_unserved_schedule)
    infeasible_schedule = Schedule(
        flows=(valid_schedule.flows[0].model_copy(update={"path_id": 99}), valid_schedule.flows[1])
    )
    infeasible = evaluate_schedule(fixed_scenario, infeasible_schedule)

    assert retain_stable_best(first, equal, tolerance=1.0e-9) is first
    assert retain_stable_best(zero, first, tolerance=1.0e-9) is first
    assert retain_stable_best(infeasible, zero, tolerance=1.0e-9) is zero
    assert retain_stable_best(infeasible, infeasible, tolerance=1.0e-9) is infeasible


def test_evaluation_result_json_round_trip(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    result = evaluate_schedule(fixed_scenario, valid_schedule)

    restored = EvaluationResult.model_validate_json(model_to_json(result))

    assert restored == result


def test_infeasible_result_none_serializes_to_null_and_round_trips(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    invalid = Schedule(
        flows=(valid_schedule.flows[0].model_copy(update={"path_id": 99}), valid_schedule.flows[1])
    )
    result = evaluate_schedule(fixed_scenario, invalid)
    raw = model_to_json(result)

    assert '"objective": null' in raw
    assert EvaluationResult.model_validate_json(raw) == result


def test_valid_and_infeasible_evaluation_leave_full_inputs_unchanged(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    infeasible = Schedule(
        flows=(
            _decision(0, served=True, path_id=0, allocations=((0, 2, 1),)),
            valid_schedule.flows[1],
        )
    )
    scenario_before = model_to_json(fixed_scenario)
    valid_before = model_to_json(valid_schedule)
    infeasible_before = model_to_json(infeasible)

    evaluate_schedule(fixed_scenario, valid_schedule)
    evaluate_schedule(fixed_scenario, infeasible)

    assert model_to_json(fixed_scenario) == scenario_before
    assert model_to_json(valid_schedule) == valid_before
    assert model_to_json(infeasible) == infeasible_before


def test_single_flow_utility_overflow_is_evaluation_invariant_error(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    huge = _scenario_with_flow_update(
        fixed_scenario, 0, priority=sys.float_info.max
    )

    with pytest.raises(EvaluationInvariantError, match="invalid utility for flow 0"):
        evaluate_schedule(huge, valid_schedule)


def test_total_objective_overflow_is_evaluation_invariant_error(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    huge = _scenario_with_flow_update(fixed_scenario, 0, priority=8.0e307)
    huge = _scenario_with_flow_update(huge, 1, priority=8.0e307)

    with pytest.raises(EvaluationInvariantError, match="finite objective"):
        evaluate_schedule(huge, valid_schedule)
