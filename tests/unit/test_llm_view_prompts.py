from __future__ import annotations

import json
import subprocess
import sys

import pytest

from adaptive_feedback_llm.evaluator import evaluate_schedule
from adaptive_feedback_llm.models import (
    EvaluatedCandidateRecord,
    EvaluatorScenario,
    model_to_json,
    schedule_from_json,
)
from adaptive_feedback_llm.exceptions import SolverConfigurationError
from adaptive_feedback_llm.prompts import (
    PROMPT_VERSION,
    SUPPORTED_PROMPT_VERSIONS,
    meta_prompt,
    minimal_example,
    prompt_bundle_sha256,
    render_prompt,
    schedule_json_schema,
)
from adaptive_feedback_llm.solvers.llm import build_llm_scenario_view, derive_qos_gaps


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value))
    return set()


def _numbers(value: object) -> list[float | int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        return sum((_numbers(item) for item in value.values()), [])
    if isinstance(value, list):
        return sum((_numbers(item) for item in value), [])
    return []


def test_static_llm_view_is_allowlisted_and_recursively_has_no_hidden_fields(
    catalog_scenario: EvaluatorScenario,
) -> None:
    hidden_sinr = tuple(37.123456 + index for index in range(len(catalog_scenario.links)))
    hidden_delay = tuple(
        0.07123456 + index / 1000 for index in range(len(catalog_scenario.links))
    )
    links = tuple(
        link.model_copy(
            update={"sinr_db": hidden_sinr[index], "fixed_delay_s": hidden_delay[index]}
        )
        for index, link in enumerate(catalog_scenario.links)
    )
    scenario = catalog_scenario.model_copy(update={"links": links})
    payload = json.loads(build_llm_scenario_view(scenario).model_dump_json())
    static = payload["static"]

    assert payload["schema_version"] == "1.1"

    assert set(static) == {
        "scenario_id",
        "node_ids",
        "capacity_adjacency_matrix",
        "flows",
        "paths",
        "hotspots",
        "guidance",
    }
    assert {
        "sinr_db",
        "rate_bps",
        "loss_probability",
        "success_probability",
        "fixed_delay_s",
        "completion_delay_s",
        "reliability",
        "margin",
        "evaluation",
        "genetic_algorithm",
    }.isdisjoint(_keys(static))
    numeric_values = _numbers(static)
    assert all(value not in numeric_values for value in hidden_sinr + hidden_delay)


def test_evaluated_history_is_the_only_exact_metric_exception(
    catalog_scenario: EvaluatorScenario,
    catalog_valid_schedule,
) -> None:
    evaluation = evaluate_schedule(catalog_scenario, catalog_valid_schedule)
    record = EvaluatedCandidateRecord(
        scenario_id=catalog_scenario.scenario_id,
        candidate_id="r1c1",
        round_index=1,
        call_index=1,
        schedule=catalog_valid_schedule,
        evaluation=evaluation,
        qos_gaps=derive_qos_gaps(evaluation),
        best_so_far=True,
    )

    view = build_llm_scenario_view(catalog_scenario, (record,))

    assert view.evaluated_history[0].evaluation is evaluation
    assert view.evaluated_history[0].evaluation.objective is not None
    assert view.evaluated_history[0].evaluation.flow_evaluations


def test_history_integrity_accepts_official_and_rejects_forged_or_cross_scenario(
    catalog_scenario: EvaluatorScenario,
    catalog_valid_schedule,
) -> None:
    official = evaluate_schedule(catalog_scenario, catalog_valid_schedule)
    record = EvaluatedCandidateRecord(
        scenario_id=catalog_scenario.scenario_id,
        candidate_id="r1c1",
        round_index=1,
        call_index=1,
        schedule=catalog_valid_schedule,
        evaluation=official,
        qos_gaps=derive_qos_gaps(official),
        best_so_far=True,
    )
    assert build_llm_scenario_view(catalog_scenario, (record,)).evaluated_history == (
        record,
    )

    forged_flow = official.flow_evaluations[0].model_copy(
        update={"completion_delay_s": 12345.6789}
    )
    forged_evaluation = official.model_copy(
        update={"flow_evaluations": (forged_flow, *official.flow_evaluations[1:])}
    )
    forged = record.model_copy(update={"evaluation": forged_evaluation})
    with pytest.raises(SolverConfigurationError, match="not official"):
        build_llm_scenario_view(catalog_scenario, (forged,))

    cross_scenario = record.model_copy(update={"scenario_id": "different-scenario"})
    with pytest.raises(SolverConfigurationError, match="does not match"):
        build_llm_scenario_view(catalog_scenario, (cross_scenario,))


def test_history_integrity_enforces_identity_order_and_stable_first_best(
    catalog_scenario: EvaluatorScenario,
    catalog_valid_schedule,
) -> None:
    official = evaluate_schedule(catalog_scenario, catalog_valid_schedule)
    first = EvaluatedCandidateRecord(
        scenario_id=catalog_scenario.scenario_id,
        candidate_id="r1c1",
        round_index=1,
        call_index=1,
        schedule=catalog_valid_schedule,
        evaluation=official,
        qos_gaps=derive_qos_gaps(official),
        best_so_far=True,
    )
    sparse_equal = first.model_copy(
        update={
            "candidate_id": "r2c2",
            "round_index": 2,
            "call_index": 2,
            "best_so_far": False,
        }
    )

    accepted = build_llm_scenario_view(catalog_scenario, (first, sparse_equal))
    assert accepted.evaluated_history == (first, sparse_equal)

    wrong_first = first.model_copy(update={"best_so_far": False})
    wrong_second = sparse_equal.model_copy(update={"best_so_far": True})
    with pytest.raises(SolverConfigurationError, match="stable first"):
        build_llm_scenario_view(catalog_scenario, (wrong_first, wrong_second))

    with pytest.raises(SolverConfigurationError, match="unique"):
        build_llm_scenario_view(catalog_scenario, (first, first))

    with pytest.raises(SolverConfigurationError, match="strictly ordered"):
        build_llm_scenario_view(catalog_scenario, (sparse_equal, first))

    mismatched_id = sparse_equal.model_copy(update={"candidate_id": "r2c1"})
    with pytest.raises(SolverConfigurationError, match="identity"):
        build_llm_scenario_view(catalog_scenario, (first, mismatched_id))


def test_qos_gaps_are_exactly_derived_and_never_fabricated(
    catalog_scenario: EvaluatorScenario,
    catalog_valid_schedule,
) -> None:
    feasible = evaluate_schedule(catalog_scenario, catalog_valid_schedule)
    assert derive_qos_gaps(feasible) == ()

    completion = feasible.flow_evaluations[0].completion_delay_s
    assert completion is not None
    deadline = completion / 2
    delay_scenario = catalog_scenario.model_copy(
        update={
            "flows": (
                catalog_scenario.flows[0].model_copy(update={"deadline_s": deadline}),
                catalog_scenario.flows[1],
            )
        }
    )
    delay_result = evaluate_schedule(delay_scenario, catalog_valid_schedule)
    delay_gaps = derive_qos_gaps(delay_result)
    assert delay_result.objective is None
    assert delay_gaps[0].flow_id == 0
    assert delay_gaps[0].delay_gap_s == pytest.approx(completion - deadline)
    assert delay_gaps[0].reliability_gap is None

    reliability = feasible.flow_evaluations[1].reliability
    assert reliability is not None
    threshold = (reliability + 1.0) / 2.0
    reliability_scenario = catalog_scenario.model_copy(
        update={
            "flows": (
                catalog_scenario.flows[0],
                catalog_scenario.flows[1].model_copy(
                    update={"min_reliability": threshold}
                ),
            )
        }
    )
    reliability_result = evaluate_schedule(
        reliability_scenario, catalog_valid_schedule
    )
    reliability_gaps = derive_qos_gaps(reliability_result)
    assert reliability_result.objective is None
    assert reliability_gaps[0].flow_id == 1
    assert reliability_gaps[0].delay_gap_s is None
    assert reliability_gaps[0].reliability_gap == pytest.approx(
        threshold - reliability
    )

    structural = catalog_valid_schedule.model_copy(
        update={
            "flows": (
                catalog_valid_schedule.flows[0].model_copy(update={"path_id": 99}),
                catalog_valid_schedule.flows[1],
            )
        }
    )
    structural_result = evaluate_schedule(catalog_scenario, structural)
    assert structural_result.objective is None
    assert structural_result.flow_evaluations == ()
    assert derive_qos_gaps(structural_result) == ()


def test_formal_llm_view_rejects_basic_catalog_with_fewer_than_five_paths(
    fixed_scenario: EvaluatorScenario,
) -> None:
    try:
        build_llm_scenario_view(fixed_scenario)
    except SolverConfigurationError as exc:
        assert "exactly five" in str(exc)
    else:
        raise AssertionError("formal LLM projection accepted a short catalog")


PROMPT_REQUIREMENTS = (
    "quasi-static, directed, tactical, and multihop",
    "traffic admission, one catalog path, and per-directed-link SRU",
    "K=5",
    "Different hops",
    "equal bandwidth W_SRU",
    "r_e(n)=",
    "loss_e=",
    "Gamma_f=product_e q_e",
    "simultaneously satisfy",
    "m_D=",
    "Maximize J=",
    "Priority rho_f",
    "total SRUs",
    "nonbinding static",
    "autonomously handle priorities",
    "evaluator owns hidden link parameters",
    "never automatically repaired",
    "complete candidate covering every flow",
    "served=false, path_id=-1, sru_allocations=[]",
    "Output only one legal JSON object",
)


def test_meta_prompt_covers_each_of_the_twenty_required_items() -> None:
    prompt = meta_prompt()
    assert PROMPT_VERSION == "1.0"
    for index, phrase in enumerate(PROMPT_REQUIREMENTS, start=1):
        assert f"{index}." in prompt
        assert phrase in prompt


def test_meta_prompt_preserves_priority_and_joint_repair():
    prompt = meta_prompt()
    for phrase in (
        "Protect higher-priority effective service",
        "not a new lexicographic objective",
        "Necessary admission restriction is allowed",
        "normally reduce priority 1 before priority 2, and priority 3 last",
        "same complete candidate must also change the receiver's route or allocation",
    ):
        assert phrase in prompt


def test_paper_capacity_self_check_and_diagnostic_actions():
    prompt = meta_prompt()
    for phrase in (
        "explicitly sum all served-flow SRUs on each selected directed edge",
        "A path with single_flow_feasible=false cannot become feasible",
        "change at least one decision causally connected to its diagnostic",
        "Capacity release alone is not a completed repair",
        "Extra SRUs do not improve fixed-path reliability-class utility",
        "Do not resubmit a known complete schedule",
    ):
        assert phrase in prompt


def test_only_paper_prompt_is_supported():
    assert SUPPORTED_PROMPT_VERSIONS == ("1.0",)
    assert prompt_bundle_sha256() == "d35e6ad71900f775134f95707566448a85274defb5ae9f07fa096e919cacfad2"
    for unsupported in ("unsupported", "", "99"):
        with pytest.raises(ValueError, match="unsupported"):
            meta_prompt(unsupported)
        with pytest.raises(ValueError, match="unsupported"):
            prompt_bundle_sha256(unsupported)


def test_meta_prompt_includes_complete_pipeline_and_reliability_semantics() -> None:
    prompt = meta_prompt()
    for phrase in (
        "whole scheduling object",
        "pipelined fluid approximation",
        "all flow data to drain through the bottleneck",
        "propagation plus processing delay",
        "Packet size/count",
        "whole-batch store-and-forward",
        "does not vary with SINR",
        "a=ln(11)/4",
        "Hop deliveries are independent",
        "SRUs add bandwidth but do not directly change q_e",
    ):
        assert phrase in prompt


def test_meta_prompt_has_exact_served_gated_base_plus_margin_objective() -> None:
    prompt = meta_prompt()
    assert "J=sum_f rho_f*s_f*(1+m_f)" in prompt
    assert "An unserved flow contributes 0" in prompt
    assert "exactly at its class threshold contributes rho_f" in prompt
    assert "m_f=1 contributes 2*rho_f" in prompt
    assert "rho_f*m_D plus" not in prompt


def test_versioned_schema_and_minimal_example_match_candidate_boundary() -> None:
    schema = json.loads(schedule_json_schema())
    example = json.loads(minimal_example())

    assert schema["$id"].endswith("/1.0")
    item = schema["properties"]["flows"]["items"]
    assert item["additionalProperties"] is False
    assert item["properties"]["flow_id"]["minimum"] == 0
    assert item["properties"]["path_id"]["minimum"] == -1
    allocation = item["properties"]["sru_allocations"]["items"]
    assert allocation["properties"]["src"]["minimum"] == 0
    assert allocation["properties"]["dst"]["minimum"] == 0
    assert allocation["properties"]["num_srus"]["minimum"] == 1
    assert item["allOf"][0]["then"]["properties"]["path_id"]["const"] == -1
    assert item["allOf"][0]["then"]["properties"]["sru_allocations"]["maxItems"] == 0
    assert item["allOf"][1]["then"]["properties"]["path_id"]["minimum"] == 0
    assert item["allOf"][1]["then"]["properties"]["sru_allocations"]["minItems"] == 1
    assert schedule_from_json(json.dumps(example)).flows[0].served is False


def test_first_round_prompt_has_no_history_or_failure_feedback(
    catalog_scenario: EvaluatorScenario,
) -> None:
    prompt = render_prompt(build_llm_scenario_view(catalog_scenario))
    assert '"evaluated_events":[]' in prompt
    assert '"failure_feedback":[]' in prompt
    assert '"reference_sru_allocations":[' in prompt
    assert "single-flow full-capacity construction" in prompt
    assert "sinr_db" not in prompt
    assert "fixed_delay_s" not in prompt


def test_rendered_prompt_embeds_versioned_schema_example_and_no_tokens(
    catalog_scenario: EvaluatorScenario,
) -> None:
    prompt = render_prompt(build_llm_scenario_view(catalog_scenario))
    assert '"$id":"adaptive-feedback-llm/schedule/1.0"' in prompt
    assert json.dumps(json.loads(minimal_example()), separators=(",", ":")) in prompt
    assert "{{" not in prompt
    assert "OUTPUT_JSON_SCHEMA_VERSION=1.0" in prompt
    assert "MINIMAL_OUTPUT_EXAMPLE_VERSION=1.0" in prompt


def test_models_cold_import_does_not_load_guidance_or_solvers() -> None:
    script = (
        "import sys; import adaptive_feedback_llm.models; "
        "assert 'adaptive_feedback_llm.guidance' not in sys.modules; "
        "assert not any(n.startswith('adaptive_feedback_llm.solvers') for n in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
