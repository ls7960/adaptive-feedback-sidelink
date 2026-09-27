from __future__ import annotations

import json

from adaptive_feedback_llm.evaluator import evaluate_schedule
from adaptive_feedback_llm.models import EvaluatedCandidateRecord, Schedule
from adaptive_feedback_llm.prompts import meta_prompt, prompt_bundle_sha256, render_prompt
from adaptive_feedback_llm.prompts.global_advisor import (
    build_global_advisor,
    build_global_brief,
)
from adaptive_feedback_llm.solvers.llm import build_llm_scenario_view, derive_qos_gaps


def record(scenario, schedule, round_index=1, call_index=1, best=False):
    evaluation = evaluate_schedule(scenario, schedule)
    return EvaluatedCandidateRecord(
        scenario_id=scenario.scenario_id,
        candidate_id=f"r{round_index}c{call_index}",
        round_index=round_index,
        call_index=call_index,
        schedule=schedule,
        evaluation=evaluation,
        qos_gaps=derive_qos_gaps(evaluation),
        best_so_far=best,
    )


def priority_inversion_schedule(valid_schedule):
    rejected_high = valid_schedule.flows[1].model_copy(update={
        "served": False,
        "path_id": -1,
        "sru_allocations": (),
    })
    return Schedule(flows=(valid_schedule.flows[0], rejected_high))


def test_first_round_brief_is_smaller_and_nonprescriptive(catalog_scenario):
    audit, brief = build_global_advisor(build_llm_scenario_view(catalog_scenario))
    assert brief == {
        "version": "1.1",
        "mode": "construct_feasible",
        "round": 1,
        "state": {"rounds_since_progress": 0, "repetition": "low"},
        "recommendation": {
            "strategy": "joint_reconfiguration",
            "strength": "preferred",
            "basis": "default_joint_construction",
            "directive": (
                "Change at least two of admission membership, route pattern, and SRU "
                "pattern relative to recent evidence; do not repeat an old complete "
                "schedule, and close released and received capacity in the same candidate."
            ),
        },
    }
    assert audit["public_view_only"]
    encoded = json.dumps(brief)
    for forbidden in (
        "flow_id", "path_id", "src", "dst", "num_srus", "admission_count",
        "candidate_id", "target_", "action_hints", "predicted_j",
    ):
        assert forbidden not in encoded


def test_round_16_keeps_required_priority_reduction(
    catalog_scenario, catalog_valid_schedule,
):
    impossible = catalog_scenario.model_copy(update={
        "flows": tuple(
            flow.model_copy(update={"deadline_s": 0.0001})
            for flow in catalog_scenario.flows
        )
    })
    history = tuple(
        record(impossible, catalog_valid_schedule, round_index=index)
        for index in range(1, 16)
    )
    audit, brief = build_global_advisor(
        build_llm_scenario_view(impossible, history)
    )
    assert brief["round"] == 16
    assert brief["state"] == {"rounds_since_progress": 15, "repetition": "high"}
    assert brief["recommendation"]["strategy"] == "priority_reduction"
    assert brief["recommendation"]["strength"] == "required"
    assert brief["recommendation"]["basis"] == "late_no_feasible_adjust_admission_scale"
    assert "lower nonzero admission scale" in brief["recommendation"]["directive"]
    assert "priority 1 before priority 2" in brief["recommendation"]["directive"]
    assert "priority 3 last" in brief["recommendation"]["directive"]
    assert audit["construction"]["can_reduce_to_lower_nonzero_scale"]


def test_construction_stagnation_prefers_early_priority_reduction(
    catalog_scenario, catalog_valid_schedule,
):
    impossible = catalog_scenario.model_copy(update={
        "flows": tuple(
            flow.model_copy(update={"deadline_s": 0.0001})
            for flow in catalog_scenario.flows
        )
    })
    history = tuple(
        record(impossible, catalog_valid_schedule, round_index=index)
        for index in range(1, 4)
    )
    brief = build_global_brief(build_llm_scenario_view(impossible, history))
    assert brief["round"] == 4
    assert brief["recommendation"]["strategy"] == "priority_reduction"
    assert brief["recommendation"]["strength"] == "preferred"
    assert brief["recommendation"]["basis"] == "construction_admission_stagnant"


def test_priority_exchange_escalates_when_membership_does_not_change(
    catalog_scenario, catalog_valid_schedule,
):
    inverted = priority_inversion_schedule(catalog_valid_schedule)
    first = record(catalog_scenario, inverted, 1, best=True)
    preferred = build_global_brief(
        build_llm_scenario_view(catalog_scenario, (first,))
    )
    assert preferred["recommendation"]["strategy"] == "priority_exchange"
    assert preferred["recommendation"]["strength"] == "preferred"
    assert "Change the admission membership" in preferred["recommendation"]["directive"]

    ignored = record(catalog_scenario, inverted, 2)
    audit, required = build_global_advisor(
        build_llm_scenario_view(catalog_scenario, (first, ignored))
    )
    assert audit["incumbent_aggregate"]["priority_inversion"]
    assert audit["latest_progress_obligations"]["new_priority_exchange_attempts"] == 0
    assert required["state"]["rounds_since_progress"] == 1
    assert required["recommendation"]["strategy"] == "priority_exchange"
    assert required["recommendation"]["strength"] == "required"


def test_repetition_includes_same_round_sibling_duplicates(
    catalog_scenario, catalog_valid_schedule,
):
    history = (
        record(catalog_scenario, catalog_valid_schedule, 1, 1, best=True),
        record(catalog_scenario, catalog_valid_schedule, 1, 2),
        record(catalog_scenario, catalog_valid_schedule, 1, 3),
    )
    audit, brief = build_global_advisor(
        build_llm_scenario_view(catalog_scenario, history)
    )
    assert audit["recent_window"]["same_round_repeats"] == 2
    assert audit["recent_window"]["cross_round_repeats"] == 0
    assert brief["state"]["repetition"] == "high"


def test_feasible_plateau_requires_a_new_allocation_signature(
    catalog_scenario, catalog_valid_schedule,
):
    history = (
        record(catalog_scenario, catalog_valid_schedule, 1, best=True),
        record(catalog_scenario, catalog_valid_schedule, 2),
        record(catalog_scenario, catalog_valid_schedule, 3),
    )
    audit, brief = build_global_advisor(
        build_llm_scenario_view(catalog_scenario, history)
    )
    assert audit["latest_progress_obligations"]["new_allocation_signatures"] == 0
    assert brief["state"] == {"rounds_since_progress": 2, "repetition": "high"}
    assert brief["recommendation"]["strategy"] == "sru_redistribution"
    assert brief["recommendation"]["strength"] == "required"
    assert "not previously evaluated" in brief["recommendation"]["directive"]


def test_paper_prompt_renders_brief(
    catalog_scenario, catalog_valid_schedule,
):
    view = build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_valid_schedule, best=True),
    ))
    current = render_prompt(view, version="1.0")
    assert "META_PROMPT_VERSION=1.0" in current
    assert "GLOBAL_BRIEF_JSON_VERSION=1.1" in current
    assert '"rounds_since_progress":0' in current
    assert '"basis":"delay_headroom_with_capacity_slack"' in current
    assert "FACT_MATRIX_JSON_VERSION=1.0" in current
    assert '"scope":"global_only_no_flow_path_edge_or_sru_target"' not in current


def test_paper_prompt_policy_and_bundle_bind_only_advisor(monkeypatch):
    prompt = meta_prompt("1.0")
    for phrase in (
        "one global strategy and a corresponding decision-layer change obligation",
        "reads no hidden link parameter",
        "no concrete flow, path, edge, SRU, admission-count",
        "Official feasible J remains the sole acceptance objective",
        "priority-aware exchange is a search policy, not a lexicographic objective",
        "must change admission membership",
        "requires a previously unevaluated joint admission-route signature",
        "requires a previously unevaluated allocation signature",
        "After 15 completed rounds without a positive feasible result",
        "normally reduce priority 1 before priority 2, and priority 3 last",
        "includes exact duplicates within the same completed round",
    ):
        assert phrase in prompt

    import adaptive_feedback_llm.prompts as prompts

    before = {
        version: prompt_bundle_sha256(version)
        for version in prompts.SUPPORTED_PROMPT_VERSIONS
    }
    read = prompts._asset_text
    monkeypatch.setattr(
        prompts,
        "_asset_text",
        lambda name: read(name) + ("\n# change" if name == "global_advisor.py" else ""),
    )
    assert prompt_bundle_sha256("1.0") != before["1.0"]
    assert all(
        prompt_bundle_sha256(version) == digest
        for version, digest in before.items()
        if version != "1.0"
    )


def test_advisor_is_deterministic_private_free_and_non_mutating(
    catalog_scenario, catalog_valid_schedule, monkeypatch,
):
    view = build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_valid_schedule, best=True),
    ))
    before = view.model_dump_json()

    def forbidden(*args, **kwargs):
        raise AssertionError("global advisor called evaluator")

    monkeypatch.setattr("adaptive_feedback_llm.evaluator.evaluate_schedule", forbidden)
    audit, brief = build_global_advisor(view)
    assert build_global_advisor(view) == (audit, brief)
    assert view.model_dump_json() == before
    visible = json.dumps(brief)
    for private_or_target in (
        "sinr_db", "fixed_delay_s", "bottleneck_rate_bps", "completion_delay_s",
        "reliability", "margin", "utility", "flow_id", "path_id", "num_srus",
        "admission_count", "candidate_id", "predicted_j",
    ):
        assert private_or_target not in visible


def test_paper_prompt_keeps_paper_bundle_hash_frozen():
    assert prompt_bundle_sha256("1.0") == (
        "d35e6ad71900f775134f95707566448a85274defb5ae9f07fa096e919cacfad2"
    )
