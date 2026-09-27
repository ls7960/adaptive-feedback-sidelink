from __future__ import annotations

import json

from adaptive_feedback_llm.evaluator import evaluate_schedule
from adaptive_feedback_llm.models import EvaluatedCandidateRecord, Schedule
from adaptive_feedback_llm.prompts import meta_prompt, prompt_bundle_sha256, render_prompt
from adaptive_feedback_llm.prompts.evidence_matrix import build_evidence_matrix
from adaptive_feedback_llm.solvers.llm import build_llm_scenario_view, derive_qos_gaps


def record(sc, schedule, r=1, c=1, best=False):
    evaluation = evaluate_schedule(sc, schedule)
    return EvaluatedCandidateRecord(
        scenario_id=sc.scenario_id,
        candidate_id=f"r{r}c{c}",
        round_index=r,
        call_index=c,
        schedule=schedule,
        evaluation=evaluation,
        qos_gaps=derive_qos_gaps(evaluation),
        best_so_far=best,
    )


def test_first_round_matrix_has_no_anchor_or_generated_menu(catalog_scenario):
    matrix = build_evidence_matrix(build_llm_scenario_view(catalog_scenario))
    assert matrix["mode"] == "construct_feasible"
    assert matrix["anchor"] == {
        "candidate_id": None,
        "role": "none",
        "schedule_usable_for_public_capacity_arithmetic": False,
    }
    assert matrix["flow_path_capacity_evidence"] == []
    assert matrix["display_order"].endswith("not_a_search_ranking")
    encoded = json.dumps(matrix)
    for forbidden in (
        "action_hints",
        "target_sru_allocations",
        "recommended_allocation",
        "reference_sru_allocations",
        "untried_complete_candidate",
    ):
        assert forbidden not in encoded


def test_feasible_anchor_has_symmetric_capacity_envelopes_not_target_values(
    catalog_scenario, catalog_valid_schedule,
):
    view = build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_valid_schedule, best=True),
    ))
    matrix = build_evidence_matrix(view)
    assert matrix["mode"] == "improve_feasible"
    assert matrix["anchor"]["role"] == "official_incumbent"
    assert matrix["anchor"]["schedule_usable_for_public_capacity_arithmetic"]
    assert [item["flow_id"] for item in matrix["flow_path_capacity_evidence"]] == [0, 1]
    assert all(len(item["paths"]) == 5 for item in matrix["flow_path_capacity_evidence"])
    selected = matrix["flow_path_capacity_evidence"][0]["paths"][0]
    assert selected["anchor_selected"]
    assert selected["edge_capacity_envelope_after_releasing_target"] == [{
        "src": 0,
        "dst": 5,
        "capacity_srus": 2,
        "other_flow_usage_srus": 0,
        "available_to_target_srus": 2,
        "other_usage_exceeds_capacity_by_srus": 0,
        "other_users": [],
    }]
    encoded = json.dumps(matrix)
    assert "target_sru_allocations" not in encoded
    assert "reference_sru_allocations" not in encoded


def test_failure_patterns_remain_allocation_and_context_bound(
    catalog_scenario, catalog_valid_schedule,
):
    scenario = catalog_scenario.model_copy(update={
        "flows": tuple(
            flow.model_copy(update={"deadline_s": 0.0001})
            for flow in catalog_scenario.flows
        )
    })
    history = (
        record(scenario, catalog_valid_schedule),
        record(scenario, catalog_valid_schedule, c=2),
        record(scenario, catalog_valid_schedule, r=2),
    )
    matrix = build_evidence_matrix(build_llm_scenario_view(scenario, history))
    assert matrix["mode"] == "construct_feasible"
    assert matrix["history_coverage"]["repeated_calls"] == 2
    assert matrix["history_coverage"]["latest_round_repeats_of_older_rounds"] == 1
    assert matrix["repeated_schedule_groups"][0]["submission_count"] == 3
    assert matrix["repeated_qos_failure_patterns"]
    assert all(item["observed_count"] == 3 for item in matrix["repeated_qos_failure_patterns"])
    assert all(
        item["scope"] == "exact_flow_path_allocation_and_joint_candidate_context_only"
        for item in matrix["repeated_qos_failure_patterns"]
    )
    assert matrix["anchor"]["role"] == "latest_evaluated_not_incumbent"
    assert matrix["flow_path_capacity_evidence"]


def test_capacity_conflict_preserves_exact_users_without_recommending_a_donor(
    catalog_scenario, catalog_valid_schedule,
):
    flow = catalog_valid_schedule.flows[0]
    bad = catalog_valid_schedule.model_copy(update={"flows": (
        flow.model_copy(update={"sru_allocations": (
            flow.sru_allocations[0].model_copy(update={"num_srus": 3}),
        )}),
        catalog_valid_schedule.flows[1],
    )})
    matrix = build_evidence_matrix(build_llm_scenario_view(
        catalog_scenario, (record(catalog_scenario, bad),)
    ))
    conflict = matrix["observed_capacity_conflicts"][0]
    assert (conflict["src"], conflict["dst"]) == (0, 5)
    assert conflict["official_actual_srus"] == 3
    assert conflict["official_capacity_srus"] == 2
    assert conflict["observed_users"] == [{
        "flow_id": 0, "priority": 2.0, "observed_srus": 3,
    }]
    assert "low_priority_first" not in json.dumps(matrix)


def test_paper_prompt_renders_fact_matrix(
    catalog_scenario, catalog_valid_schedule,
):
    view = build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_valid_schedule, best=True),
    ))
    current = render_prompt(view, version="1.0")
    assert "META_PROMPT_VERSION=1.0" in current
    assert "FACT_MATRIX_JSON_VERSION=1.0" in current
    assert "SEARCH_STATE_AND_UNTRIED_ACTION_HINTS_JSON_VERSION" not in current
    assert '"target_sru_allocations"' not in current


def test_paper_prompt_policy_explicitly_rejects_anchors_and_menus():
    prompt = meta_prompt()
    for phrase in (
        "contains no generated target allocation and no action menu",
        "serialization order only, never a search ranking",
        "It is not a recommended allocation",
        "change at least one decision causally connected to its diagnostic",
        "Capacity release alone is not a completed repair",
        "A low-resource trial evaluates only that resource level",
        "not a minimum, typical, central, preferred, or recommended allocation",
    ):
        assert phrase in prompt


def test_paper_prompt_bundle_binds_evidence_matrix(monkeypatch):
    import adaptive_feedback_llm.prompts as prompts
    before = prompt_bundle_sha256()
    read = prompts._asset_text
    monkeypatch.setattr(prompts, "_asset_text", lambda name: read(name) + (
        "\n# change" if name == "evidence_matrix.py" else ""))
    assert prompt_bundle_sha256() != before


def test_matrix_is_deterministic_private_free_and_non_mutating(
    catalog_scenario, catalog_valid_schedule, monkeypatch,
):
    view = build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_valid_schedule, best=True),
    ))
    before = view.model_dump_json()

    def forbidden(*args, **kwargs):
        raise AssertionError("evidence matrix called evaluator")

    monkeypatch.setattr("adaptive_feedback_llm.evaluator.evaluate_schedule", forbidden)
    matrix = build_evidence_matrix(view)
    assert build_evidence_matrix(view) == matrix
    assert view.model_dump_json() == before
    encoded = json.dumps(matrix)
    for private in ("sinr_db", "fixed_delay_s", "predicted_j", "predicted_utility"):
        assert private not in encoded


def test_structurally_broken_anchor_does_not_get_capacity_envelopes(
    catalog_scenario, catalog_valid_schedule,
):
    broken = Schedule(flows=catalog_valid_schedule.flows[:1])
    matrix = build_evidence_matrix(build_llm_scenario_view(
        catalog_scenario, (record(catalog_scenario, broken),)
    ))
    assert not matrix["anchor"]["schedule_usable_for_public_capacity_arithmetic"]
    assert matrix["flow_path_capacity_evidence"] == []
