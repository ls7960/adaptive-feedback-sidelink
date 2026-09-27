from __future__ import annotations

from copy import deepcopy
import json

import pytest

from adaptive_feedback_llm.evaluator import evaluate_schedule
from adaptive_feedback_llm.models import EvaluatedCandidateRecord
from adaptive_feedback_llm.models.llm import (
    FailureCategory, FailureFact, LLMFailureFeedback,
)
from adaptive_feedback_llm.prompts import meta_prompt, prompt_bundle_sha256, render_prompt
from adaptive_feedback_llm.prompts.compact_context import (
    compact_json, decode_context, encode_context,
)
from adaptive_feedback_llm.prompts.opportunities import build_optimization_facts
from adaptive_feedback_llm.solvers.llm import build_llm_scenario_view, derive_qos_gaps


def record(scenario, schedule, round_index=1, call_index=1, best=False):
    evaluation = evaluate_schedule(scenario, schedule)
    return EvaluatedCandidateRecord(
        scenario_id=scenario.scenario_id,
        candidate_id=f"r{round_index}c{call_index}",
        round_index=round_index, call_index=call_index, schedule=schedule,
        evaluation=evaluation, qos_gaps=derive_qos_gaps(evaluation), best_so_far=best,
    )


def assert_roundtrip(view):
    before = view.model_dump_json()
    encoded = encode_context(view)
    serialized = json.loads(compact_json(encoded))
    restored = decode_context(serialized)
    assert restored.model_dump_json() == before
    assert view.model_dump_json() == before
    assert encode_context(view) == encoded
    # Prove the ACTUAL prompt, not merely the in-memory packer, is reversible.
    prompt = render_prompt(view)
    parsed = {"encoding_version": "1.0"}
    for key, marker in (
        ("decision", "CURRENT_DECISION_JSON_ENCODING_VERSION=1.0"),
        ("evidence", "COMPLETE_EVIDENCE_JSON_ENCODING_VERSION=1.0"),
    ):
        parsed[key], _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())
    assert decode_context(parsed).model_dump_json() == before
    return encoded


def test_empty_first_round(catalog_scenario):
    encoded = assert_roundtrip(build_llm_scenario_view(catalog_scenario))
    assert encoded["decision"]["incumbent_payload_ref"] is None
    assert encoded["decision"]["latest_round_index"] is None
    assert encoded["evidence"]["evaluated_events"] == []
    assert encoded["decision"]["focus_payloads"] == {}


def test_all_60_events_retained_with_one_exact_payload(catalog_scenario, catalog_valid_schedule):
    history = tuple(record(catalog_scenario, catalog_valid_schedule, r, c, r == c == 1)
                    for r in range(1, 21) for c in range(1, 4))
    view = build_llm_scenario_view(catalog_scenario, history)
    encoded = assert_roundtrip(view)
    assert len(encoded["evidence"]["evaluated_events"]) == 60
    assert len(encoded["decision"]["focus_payloads"]) == 1
    assert encoded["evidence"]["other_payloads"] == {}
    facts = encoded["decision"]["optimization_facts"]
    assert len(facts["comparisons_by_payload"]) == 1
    assert len(facts["schedule_repetitions"]) == 59
    assert facts["history_coverage"]["evaluated_candidates"] == 60
    assert len(render_prompt(view)) < len(view.model_dump_json(indent=2)) * 0.3
    assert encoded["decision"]["incumbent_candidate_id"] == "r1c1"
    assert encoded["decision"]["latest_evaluated_candidate_ids"] == ["r20c1", "r20c2", "r20c3"]


def test_array_order_not_erased_but_semantic_repeats_recognized(
    catalog_scenario, catalog_valid_schedule,
):
    shuffled = catalog_valid_schedule.model_copy(update={"flows": tuple(
        f.model_copy(update={"sru_allocations": tuple(reversed(f.sru_allocations))})
        for f in reversed(catalog_valid_schedule.flows))})
    view = build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_valid_schedule, best=True),
        record(catalog_scenario, shuffled, 2),
    ))
    encoded = assert_roundtrip(view)
    assert len(encoded["decision"]["focus_payloads"]) == 2
    assert encoded["decision"]["optimization_facts"]["schedule_repetitions"] == [
        {"candidate_id": "r2c1", "repeat_of_candidate_id": "r1c1"}]


def test_invalid_payloads_archived_and_failure_only_latest_round(
    catalog_scenario, catalog_valid_schedule,
):
    invalid = catalog_valid_schedule.model_copy(update={"flows": (
        catalog_valid_schedule.flows[0].model_copy(update={"path_id": 99}),
        catalog_valid_schedule.flows[1],
    )})
    failure = LLMFailureFeedback(
        candidate_id="r3c2", round_index=3, call_index=2,
        category=FailureCategory.PARSE_FAILURE, fact=FailureFact.PARSE_FAILURE,
        transport_attempts=1,
    )
    view = build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_valid_schedule, best=True),
        record(catalog_scenario, invalid, 2),
    ), (failure,))
    encoded = assert_roundtrip(view)
    assert encoded["decision"]["latest_evaluated_candidate_ids"] == []
    assert encoded["decision"]["latest_failure_candidate_ids"] == ["r3c2"]
    assert set(encoded["decision"]["focus_payloads"]) == {"p0"}
    assert set(encoded["evidence"]["other_payloads"]) == {"p1"}
    assert encoded["evidence"]["other_payloads"]["p1"]["evaluation"]["objective"] is None
    assert encoded["evidence"]["other_payloads"]["p1"]["evaluation"]["diagnostics"]


def test_zero_best_and_no_best(catalog_scenario, catalog_all_unserved_schedule, catalog_valid_schedule):
    zero = build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_all_unserved_schedule, best=True),))
    encoded = assert_roundtrip(zero)
    assert encoded["decision"]["optimization_facts"]["status"] == "no_served_feasible_best"
    assert encoded["decision"]["incumbent_candidate_id"] == "r1c1"
    invalid = catalog_valid_schedule.model_copy(update={"flows": ()})
    bad = build_llm_scenario_view(catalog_scenario, (record(catalog_scenario, invalid),))
    assert assert_roundtrip(bad)["decision"]["incumbent_candidate_id"] is None


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(encoding_version="99"),
    lambda x: x["evidence"]["evaluated_events"][0].update(payload_ref="missing"),
    lambda x: x["evidence"]["other_payloads"].update(p0=x["decision"]["focus_payloads"]["p0"]),
    lambda x: x["evidence"]["other_payloads"].update(orphan=x["decision"]["focus_payloads"]["p0"]),
    lambda x: x["decision"].update(incumbent_candidate_id="wrong"),
    lambda x: x["decision"]["optimization_facts"].update(status="wrong"),
    lambda x: x["evidence"]["evaluated_events"][0].update(schedule={}),
    lambda x: x.update(unexpected=True),
])
def test_decoder_rejects_inconsistent_references_and_indexes(
    catalog_scenario, catalog_valid_schedule, mutation,
):
    encoded = encode_context(build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_valid_schedule, best=True),)))
    corrupted = deepcopy(encoded)
    mutation(corrupted)
    with pytest.raises(ValueError):
        decode_context(corrupted)


def test_exact_official_qos_values_and_gaps_survive(catalog_scenario, catalog_valid_schedule):
    tight = catalog_scenario.model_copy(update={"flows": tuple(
        f.model_copy(update={"deadline_s": 0.000123456789012345})
        for f in catalog_scenario.flows)})
    view = build_llm_scenario_view(tight, (record(tight, catalog_valid_schedule),))
    assert view.evaluated_history[0].qos_gaps
    assert_roundtrip(view)


def test_no_evaluator_calls_and_all_facts_preserved(catalog_scenario, catalog_valid_schedule, monkeypatch):
    view = build_llm_scenario_view(catalog_scenario, (
        record(catalog_scenario, catalog_valid_schedule, best=True),
        record(catalog_scenario, catalog_valid_schedule, 2),
    ))
    old = build_optimization_facts(view)
    def forbidden(*args, **kwargs):
        raise AssertionError("encoding must not evaluate candidates")
    monkeypatch.setattr("adaptive_feedback_llm.evaluator.evaluate_schedule", forbidden)
    encoded = assert_roundtrip(view)
    new = encoded["decision"]["optimization_facts"]
    for key in old.keys() - {"schema_version", "candidate_comparisons"}:
        assert old[key] == new[key]
    repeats = {x["candidate_id"]: x["repeat_of_candidate_id"] for x in new["schedule_repetitions"]}
    for event, comparison in zip(encoded["evidence"]["evaluated_events"], old["candidate_comparisons"], strict=True):
        restored = {**new["comparisons_by_payload"][event["payload_ref"]],
                    "candidate_id": event["candidate_id"],
                    "repeat_of_candidate_id": repeats.get(event["candidate_id"])}
        assert restored == comparison


def test_paper_bundle_binds_encoding_source(monkeypatch):
    import adaptive_feedback_llm.prompts as prompts
    before = prompt_bundle_sha256()
    read = prompts._asset_text
    monkeypatch.setattr(prompts, "_asset_text", lambda name: read(name) + (
        "\n# changed encoder" if name == "compact_context.py" else ""))
    assert prompt_bundle_sha256() != before
    assert "Every evaluated event remains in original order" in meta_prompt()
    assert "Do not output evidence fields, references" in meta_prompt()
