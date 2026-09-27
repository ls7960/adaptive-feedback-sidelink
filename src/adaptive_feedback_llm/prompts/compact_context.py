"""paper prompt: lossless public history pooling, not a history summarizer.

Never imports a provider, evaluator, scenario artifact, or private physics.
The source of this module is bound into the paper prompt bundle digest.
"""
from __future__ import annotations

from copy import deepcopy
import json

from adaptive_feedback_llm.models import LLMScenarioView
from adaptive_feedback_llm.prompts.opportunities import build_optimization_facts


PAYLOAD_FIELDS = ("schedule", "evaluation", "qos_gaps")


def compact_json(value: object) -> str:
    """Keep field names, array order and full JSON float precision."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def encode_context(view: LLMScenarioView) -> dict:
    """Pool only exactly equal payloads; preserve every original event field.

    List reordering is NOT an exact duplicate. The separate nonbinding facts
    still recognize semantic schedule repeats irrespective of list order.
    """
    raw = view.model_dump(mode="json")
    pool: dict = {}
    seen: dict = {}
    events = []
    for record in raw["evaluated_history"]:
        payload = {key: record[key] for key in PAYLOAD_FIELDS}
        signature = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, separators=(",", ":"))
        if signature not in seen:
            ref = f"p{len(pool)}"
            seen[signature] = ref
            pool[ref] = payload
        event = {key: value for key, value in record.items() if key not in PAYLOAD_FIELDS}
        event["payload_ref"] = seen[signature]
        events.append(event)

    best = next((event for event in events if event["best_so_far"]), None)
    failures = raw["failure_feedback"]
    latest_round = max((event["round_index"] for event in [*events, *failures]), default=None)
    latest_ids = [event["candidate_id"] for event in events
                  if event["round_index"] == latest_round]
    latest_failures = [event["candidate_id"] for event in failures
                       if event["round_index"] == latest_round]
    # Put the incumbent and last evaluated round up front, each payload ONCE.
    focus_refs = {event["payload_ref"] for event in events
                  if event["best_so_far"] or event["round_index"] == latest_round}
    focus = {ref: payload for ref, payload in pool.items() if ref in focus_refs}
    archive = {ref: payload for ref, payload in pool.items() if ref not in focus_refs}

    facts = build_optimization_facts(view)
    comparisons = facts.pop("candidate_comparisons", [])
    # Per-state comparisons are identical for exact payload repeats. Keep the
    # per-call repeat provenance separately instead of repeating change arrays.
    comparisons_by_payload = {}
    repetitions = []
    for event, comparison in zip(events, comparisons, strict=bool(comparisons)):
        comparison = dict(comparison)
        candidate_id = comparison.pop("candidate_id")
        repeat_of = comparison.pop("repeat_of_candidate_id")
        comparisons_by_payload.setdefault(event["payload_ref"], comparison)
        if repeat_of is not None:
            repetitions.append({"candidate_id": candidate_id,
                                "repeat_of_candidate_id": repeat_of})
    facts["schema_version"] = "1.1"
    if comparisons:
        facts["comparisons_by_payload"] = comparisons_by_payload
        facts["schedule_repetitions"] = repetitions

    return {
        "encoding_version": "1.0",
        "decision": {
            "static": raw["static"],
            "incumbent_candidate_id": best["candidate_id"] if best else None,
            "incumbent_payload_ref": best["payload_ref"] if best else None,
            "latest_round_index": latest_round,
            "latest_evaluated_candidate_ids": latest_ids,
            "latest_failure_candidate_ids": latest_failures,
            "focus_payloads": focus,
            "optimization_facts": facts,
        },
        "evidence": {
            "view_schema_version": raw["schema_version"],
            "evaluated_events": events,
            "other_payloads": archive,
            "failure_feedback": failures,
        },
    }


def decode_context(encoded: dict) -> LLMScenarioView:
    """Offline inverse with canonical integrity checks; not an evaluator.

    Validation checks references, focus/navigation and derived facts too, but
    cannot establish that metrics are official. The runner's existing public
    projection remains responsible for evaluator replay before encoding.
    """
    data = deepcopy(encoded)
    try:
        if data["encoding_version"] != "1.0":
            raise ValueError("unsupported encoding version")
        decision, evidence = data["decision"], data["evidence"]
        focus, archive = decision["focus_payloads"], evidence["other_payloads"]
        if set(focus) & set(archive):
            raise ValueError("duplicate payload reference across zones")
        pool = {**focus, **archive}
        records = []
        for event in evidence["evaluated_events"]:
            metadata = dict(event)
            payload = pool[metadata.pop("payload_ref")]
            if set(payload) != set(PAYLOAD_FIELDS) or set(metadata) & set(PAYLOAD_FIELDS):
                raise ValueError("invalid payload fields")
            records.append({**metadata, **payload})
        raw = {"schema_version": evidence["view_schema_version"],
               "static": decision["static"], "evaluated_history": records,
               "failure_feedback": evidence["failure_feedback"]}
        view = LLMScenarioView.model_validate_json(compact_json(raw))
        # Also reject orphan states, wrong navigation, unexpected fields and
        # tampered derived indexes, rather than silently ignoring them.
        expected = encode_context(view)
        canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False,
                                             allow_nan=False, separators=(",", ":"))
        if canonical(expected) != canonical(data):
            raise ValueError("noncanonical or inconsistent compact context")
        return view
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid compact context references or structure") from exc
