from __future__ import annotations

import pytest

import adaptive_feedback_llm.evaluator as evaluator_module
import adaptive_feedback_llm.solvers.heuristics.core as heuristic_core
from adaptive_feedback_llm.guidance import build_static_guidance
from adaptive_feedback_llm.models import DirectedLink, EvaluatorScenario, Flow, TrafficClass
from adaptive_feedback_llm.solvers.common import (
    construct_path_allocation,
    ordered_flows,
    select_increment_edge_index,
)
from adaptive_feedback_llm.solvers.heuristics import (
    solve_load_aware,
    solve_max_sinr,
    solve_priority_aware,
)


def _scenario_with_updates(
    scenario: EvaluatorScenario,
    *,
    flow_updates: dict[int, dict[str, object]] | None = None,
    link_updates: dict[tuple[int, int], dict[str, object]] | None = None,
) -> EvaluatorScenario:
    flows = []
    for flow in scenario.flows:
        data = flow.model_dump(mode="python")
        data.update((flow_updates or {}).get(flow.flow_id, {}))
        flows.append(Flow.model_validate(data))
    links = []
    for link in scenario.links:
        data = link.model_dump(mode="python")
        data.update((link_updates or {}).get(link.edge, {}))
        links.append(DirectedLink.model_validate(data))
    data = scenario.model_dump(mode="python")
    data["flows"] = tuple(flows)
    data["links"] = tuple(links)
    return EvaluatorScenario.model_validate(data)


def test_all_heuristics_share_priority_then_flow_id_order(
    fixed_scenario: EvaluatorScenario,
) -> None:
    assert tuple(flow.flow_id for flow in ordered_flows(fixed_scenario)) == (1, 0)
    tied = _scenario_with_updates(
        fixed_scenario,
        flow_updates={0: {"priority": 3.0}, 1: {"priority": 3.0}},
    )
    assert tuple(flow.flow_id for flow in ordered_flows(tied)) == (0, 1)


def test_common_constructor_increments_current_slowest_link(
    fixed_scenario: EvaluatorScenario,
) -> None:
    scenario = _scenario_with_updates(
        fixed_scenario,
        link_updates={(0, 1): {"sinr_db": 4.0}},
    )
    flow = scenario.flows[0]
    path = scenario.path_catalogs[0].paths[1]
    remaining = {link.edge: link.capacity_srus for link in scenario.links}

    construction = construct_path_allocation(scenario, flow, path, remaining)

    assert construction is not None
    assert tuple(item.num_srus for item in construction.allocations) == (2, 1)


def test_increment_rate_tie_uses_path_edge_order() -> None:
    assert select_increment_edge_index(
        ((0, 1), (1, 2)),
        (1, 1),
        (100.0, 100.0),
        {(0, 1): 2, (1, 2): 2},
    ) == 0


def test_common_constructor_rejects_fixed_reliability_failure_even_with_capacity(
    fixed_scenario: EvaluatorScenario,
) -> None:
    path = fixed_scenario.path_catalogs[1].paths[0]
    remaining = {link.edge: link.capacity_srus for link in fixed_scenario.links}

    assert construct_path_allocation(
        fixed_scenario, fixed_scenario.flows[1], path, remaining
    ) is None


def test_three_heuristics_are_feasible_and_differ_only_by_path_order(
    fixed_scenario: EvaluatorScenario,
) -> None:
    guidance = build_static_guidance(fixed_scenario)
    priority = solve_priority_aware(fixed_scenario, guidance=guidance)
    max_sinr = solve_max_sinr(fixed_scenario, guidance=guidance)
    load = solve_load_aware(fixed_scenario, guidance=guidance)

    assert priority.evaluation.feasible
    assert max_sinr.evaluation.feasible
    assert load.evaluation.feasible
    assert tuple(item.flow_id for item in priority.schedule.flows) == (0, 1)
    assert priority.schedule.flows[0].path_id == 0
    assert max_sinr.schedule.flows[0].path_id == 1
    assert load.schedule.flows[0].path_id == 1


@pytest.mark.parametrize(
    "solver",
    [solve_priority_aware, solve_max_sinr, solve_load_aware],
)
def test_each_heuristic_invokes_the_unique_evaluator_once(
    fixed_scenario: EvaluatorScenario,
    monkeypatch: pytest.MonkeyPatch,
    solver,
) -> None:
    calls = 0
    original = evaluator_module.evaluate_schedule

    def counted(scenario, schedule):
        nonlocal calls
        calls += 1
        return original(scenario, schedule)

    monkeypatch.setattr(heuristic_core, "evaluate_schedule", counted)
    result = solver(fixed_scenario)

    assert calls == 1
    assert result.evaluation == original(fixed_scenario, result.schedule)


def test_constructor_does_not_use_traffic_class_to_skip_either_qos(
    fixed_scenario: EvaluatorScenario,
) -> None:
    delay_with_hard_reliability = _scenario_with_updates(
        fixed_scenario, flow_updates={0: {"min_reliability": 0.999}}
    )
    reliability_with_hard_delay = _scenario_with_updates(
        fixed_scenario,
        flow_updates={1: {"traffic_class": TrafficClass.RELIABILITY, "deadline_s": 0.04}},
    )
    remaining_delay = {link.edge: link.capacity_srus for link in delay_with_hard_reliability.links}
    remaining_reliability = {
        link.edge: link.capacity_srus for link in reliability_with_hard_delay.links
    }

    assert construct_path_allocation(
        delay_with_hard_reliability,
        delay_with_hard_reliability.flows[0],
        delay_with_hard_reliability.path_catalogs[0].paths[0],
        remaining_delay,
    ) is None
    assert construct_path_allocation(
        reliability_with_hard_delay,
        reliability_with_hard_delay.flows[1],
        reliability_with_hard_delay.path_catalogs[1].paths[1],
        remaining_reliability,
    ) is None
