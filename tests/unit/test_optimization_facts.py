from __future__ import annotations

import json

from adaptive_feedback_llm.evaluator import evaluate_schedule
from adaptive_feedback_llm.models import EvaluatedCandidateRecord, EvaluatorScenario, Schedule
from adaptive_feedback_llm.prompts import render_prompt, prompt_bundle_sha256
from adaptive_feedback_llm.prompts.opportunities import build_optimization_facts
from adaptive_feedback_llm.solvers.llm import build_llm_scenario_view, derive_qos_gaps


def record(scenario, schedule, *, candidate_id="r1c1", best=True):
    evaluation = evaluate_schedule(scenario, schedule)
    return EvaluatedCandidateRecord(
        scenario_id=scenario.scenario_id, candidate_id=candidate_id,
        round_index=int(candidate_id[1]), call_index=int(candidate_id[-1]),
        schedule=schedule, evaluation=evaluation, qos_gaps=derive_qos_gaps(evaluation),
        best_so_far=best,
    )


def altered(schedule, update):
    raw = schedule.model_dump(mode="json")
    update(raw)
    return Schedule.model_validate_json(json.dumps(raw))


def test_facts_inactive_without_useful_incumbent(catalog_scenario, catalog_valid_schedule):
    view = build_llm_scenario_view(catalog_scenario)
    assert build_optimization_facts(view) == {
        "schema_version": "1.0", "status": "no_served_feasible_best"}
    unserved = altered(catalog_valid_schedule, lambda r: [
        f.update(served=False, path_id=-1, sru_allocations=[]) for f in r["flows"]])
    zero = record(catalog_scenario, unserved)
    view = build_llm_scenario_view(catalog_scenario, (zero,))
    assert build_optimization_facts(view)["status"] == "no_served_feasible_best"
    invalid = altered(catalog_valid_schedule, lambda r: r["flows"][0].update(path_id=99))
    rejected = record(catalog_scenario, invalid, best=False)
    view = build_llm_scenario_view(catalog_scenario, (rejected,))
    assert build_optimization_facts(view)["status"] == "no_served_feasible_best"


def test_rejected_priority_and_reference_deficits_are_current_joint_facts(
    catalog_scenario, catalog_valid_schedule,
):
    def edit(raw):
        raw["flows"][0]["sru_allocations"][0]["num_srus"] = 2
        raw["flows"][1].update(served=False, path_id=-1, sru_allocations=[])
    schedule = altered(catalog_valid_schedule, edit)
    view = build_llm_scenario_view(catalog_scenario, (record(catalog_scenario, schedule),))
    facts = build_optimization_facts(view)
    rejected = facts["unserved_flows_by_priority"][0]
    assert rejected["flow_id"] == 1 and rejected["priority"] == 3
    assert rejected["is_highest_scenario_priority"]
    refs = {x["path_id"]: x for x in rejected["single_flow_feasible_references"]}
    assert not refs[0]["reference_fits_residual_capacity"]
    assert refs[0]["reference_capacity_shortfalls"] == [{
        "src": 0, "dst": 5, "reference_required_srus": 1,
        "residual_after_releasing_this_flow_srus": 0,
        "shortfall_srus": 1, "other_incumbent_flow_ids": [0]}]
    assert refs[1]["reference_fits_residual_capacity"]
    assert facts["served_path_capacity"][0]["selected_edges"][0]["residual_srus"] == 0


def test_residual_uses_incumbent_not_latest_trial(catalog_scenario, catalog_valid_schedule):
    best = record(catalog_scenario, catalog_valid_schedule)
    bad = altered(catalog_valid_schedule, lambda r: r["flows"][0]["sru_allocations"][0].update(num_srus=3))
    trial = record(catalog_scenario, bad, candidate_id="r2c1", best=False)
    view = build_llm_scenario_view(catalog_scenario, (best, trial))
    facts = build_optimization_facts(view)
    assert facts["anchor_candidate_id"] == "r1c1"
    f = next(f for f in facts["served_path_capacity"] if f["flow_id"] == 0)
    assert f["selected_edges"][0]["residual_srus"] == 1
    comparison = facts["candidate_comparisons"][1]
    assert comparison["delta_j_vs_incumbent"] is None
    assert comparison["diagnostic_codes"] == ["link_capacity_exceeded"]


def test_reroute_reference_releases_own_allocation_on_shared_path_edge(
    catalog_scenario, catalog_valid_schedule,
):
    raw = catalog_scenario.model_dump(mode="json")
    for link in raw["links"]:
        if (link["src"], link["dst"]) == (0, 1):
            link["capacity_srus"] = 1
    raw["links"].append({"src": 1, "dst": 2, "capacity_srus": 1,
                         "sinr_db": 10.0, "fixed_delay_s": 0.01})
    raw["links"].sort(key=lambda link: (link["src"], link["dst"]))
    raw["path_catalogs"][1]["paths"][4]["node_sequence"] = [0, 1, 2, 5]
    scenario = EvaluatorScenario.model_validate_json(json.dumps(raw))
    view = build_llm_scenario_view(scenario, (record(scenario, catalog_valid_schedule),))
    facts = build_optimization_facts(view)
    item = next(x for x in facts["served_path_capacity"] if x["flow_id"] == 1)
    edge = next(e for e in item["selected_edges"] if (e["src"], e["dst"]) == (0, 1))
    assert edge["residual_srus"] == 0
    alternative = next(x for x in item["alternative_references_after_releasing_this_flow"]
                       if x["path_id"] == 4)
    assert alternative["reference_fits_residual_capacity"]
    assert alternative["reference_capacity_shortfalls"] == []


def test_repeats_ignore_order_but_distinguish_allocations(catalog_scenario, catalog_valid_schedule):
    first = record(catalog_scenario, catalog_valid_schedule)
    shuffled = altered(catalog_valid_schedule, lambda r: (
        r["flows"].reverse(), [f["sru_allocations"].reverse() for f in r["flows"]]))
    second = record(catalog_scenario, shuffled, candidate_id="r2c1", best=False)
    # Reliability-class allocation changes can preserve objective and route.
    changed = altered(catalog_valid_schedule, lambda r: r["flows"][1]["sru_allocations"][0].update(num_srus=2))
    third = record(catalog_scenario, changed, candidate_id="r2c2", best=False)
    view = build_llm_scenario_view(catalog_scenario, (first, second, third))
    facts = build_optimization_facts(view)
    assert facts["history_coverage"] == {"evaluated_candidates": 3,
        "distinct_complete_schedules": 2, "repeated_complete_schedules": 1,
        "distinct_admission_route_patterns": 1}
    assert facts["candidate_comparisons"][1]["repeat_of_candidate_id"] == "r1c1"
    last = facts["candidate_comparisons"][2]
    assert last["repeat_of_candidate_id"] is None
    assert last["delta_j_vs_incumbent"] == 0
    assert last["changes_vs_incumbent"]["allocation_changed"] == [1]


def test_structurally_invalid_history_is_indexed_without_invented_metrics(
    catalog_scenario, catalog_valid_schedule,
):
    first = record(catalog_scenario, catalog_valid_schedule)
    missing = altered(catalog_valid_schedule, lambda r: r["flows"].pop())
    second = record(catalog_scenario, missing, candidate_id="r2c1", best=False)
    facts = build_optimization_facts(build_llm_scenario_view(catalog_scenario, (first, second)))
    item = facts["candidate_comparisons"][1]
    assert item["changes_vs_incumbent"]["missing_flow_ids"] == [1]
    assert item["delta_j_vs_incumbent"] is None


def test_no_hidden_access_no_mutation_and_full_history_stays_in_render(
    catalog_scenario, catalog_valid_schedule, monkeypatch,
):
    view = build_llm_scenario_view(catalog_scenario, (record(catalog_scenario, catalog_valid_schedule),))
    before = view.model_dump_json()
    def forbidden(*args, **kwargs):
        raise AssertionError("facts must not evaluate any candidate")
    monkeypatch.setattr("adaptive_feedback_llm.evaluator.evaluate_schedule", forbidden)
    facts = build_optimization_facts(view)
    text = json.dumps(facts)
    for hidden in ("sinr_db", "fixed_delay_s", "bottleneck_rate_bps", "completion_delay_s", "predicted_j"):
        assert hidden not in text
    rendered = render_prompt(view)
    assert '"optimization_facts"' in rendered
    assert 'COMPLETE_EVIDENCE_JSON_ENCODING_VERSION=1.0' in rendered
    assert '{{' not in rendered
    assert before == view.model_dump_json()


def test_paper_bundle_binds_facts_implementation(monkeypatch):
    import adaptive_feedback_llm.prompts as prompts
    current = prompt_bundle_sha256()
    read = prompts._asset_text
    monkeypatch.setattr(prompts, "_asset_text", lambda name: read(name) + (
        "\n# changed implementation" if name == "opportunities.py" else ""))
    assert current != prompt_bundle_sha256()
