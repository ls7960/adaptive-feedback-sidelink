from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_feedback_llm.exceptions import CandidateParseError, ScenarioValidationError
from adaptive_feedback_llm.models import (
    CandidatePath,
    EvaluatorScenario,
    Flow,
    PathCatalog,
    Schedule,
    evaluator_scenario_from_json,
    model_to_json,
    schedule_from_json,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_candidate_path_edges_single_and_multi_hop_regression() -> None:
    direct = CandidatePath(flow_id=0, path_id=0, node_sequence=(0, 2))
    multihop = CandidatePath(flow_id=0, path_id=1, node_sequence=(0, 1, 2))

    assert direct.edges == ((0, 2),)
    assert multihop.edges == ((0, 1), (1, 2))


def test_fixed_scenario_is_directed_asymmetric_and_allows_fewer_than_five_paths(
    fixed_scenario: EvaluatorScenario,
) -> None:
    links = fixed_scenario.link_by_edge()

    assert links[(0, 2)].capacity_srus == 2
    assert links[(2, 0)].capacity_srus == 1
    assert links[(0, 2)].sinr_db != links[(2, 0)].sinr_db
    assert all(len(catalog.paths) == 2 for catalog in fixed_scenario.path_catalogs)


def test_scenario_and_schedule_json_round_trip(
    fixed_scenario: EvaluatorScenario, valid_schedule: Schedule
) -> None:
    assert evaluator_scenario_from_json(model_to_json(fixed_scenario)) == fixed_scenario
    assert schedule_from_json(model_to_json(valid_schedule)) == valid_schedule


@pytest.mark.parametrize(
    "raw_json",
    [
        '{"flows":[],"flows":[]}',
        '{"flows":[{"flow_id":0,"served":false,"served":true,'
        '"path_id":-1,"sru_allocations":[]}]}',
    ],
)
def test_schedule_json_rejects_top_level_and_nested_duplicate_keys(
    raw_json: str,
) -> None:
    with pytest.raises(CandidateParseError, match="duplicate object key"):
        schedule_from_json(raw_json)


def test_scenario_json_rejects_top_level_duplicate_key() -> None:
    raw = (FIXTURES / "basic_scenario.json").read_text(encoding="utf-8")
    invalid = raw.replace(
        '"schema_version": "1.0",',
        '"schema_version": "1.0",\n  "schema_version": "1.0",',
        1,
    )

    with pytest.raises(ScenarioValidationError, match="duplicate object key"):
        evaluator_scenario_from_json(invalid)


def test_scenario_json_rejects_nested_duplicate_key() -> None:
    raw = (FIXTURES / "basic_scenario.json").read_text(encoding="utf-8")
    invalid = raw.replace(
        '"sinr_db": 10.0,',
        '"sinr_db": 10.0,\n      "sinr_db": 10.0,',
        1,
    )

    with pytest.raises(ScenarioValidationError, match="duplicate object key"):
        evaluator_scenario_from_json(invalid)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_numeric_tokens_are_rejected_before_pydantic(
    token: str,
) -> None:
    schedule = (
        '{"flows":[{"flow_id":0,"served":true,"path_id":0,'
        f'"sru_allocations":[{{"src":0,"dst":2,"num_srus":{token}}}]}}]}}'
    )
    scenario = (FIXTURES / "basic_scenario.json").read_text(encoding="utf-8").replace(
        '"sinr_db": 10.0', f'"sinr_db": {token}', 1
    )

    with pytest.raises(CandidateParseError, match="non-standard numeric token"):
        schedule_from_json(schedule)
    with pytest.raises(ScenarioValidationError, match="non-standard numeric token"):
        evaluator_scenario_from_json(scenario)


def test_domain_models_are_immutable(fixed_scenario: EvaluatorScenario) -> None:
    with pytest.raises(ValidationError):
        fixed_scenario.links[0].sinr_db = 99.0


@pytest.mark.parametrize(
    "raw_json",
    [
        '{"flows":[{"flow_id":true,"served":false,"path_id":-1,"sru_allocations":[]}]}',
        '{"flows":[{"flow_id":0,"served":1,"path_id":-1,"sru_allocations":[]}]}',
        '{"flows":[{"flow_id":0,"served":true,"path_id":0,"sru_allocations":'
        '[{"src":0,"dst":2,"num_srus":true}]}]}',
        '{"flows":[{"flow_id":0,"served":true,"path_id":0,"sru_allocations":'
        '[{"src":0,"dst":2,"num_srus":1.5}]}]}',
        '{"flows":[],"extra":1}',
        '{"flows":[]} trailing text',
    ],
)
def test_schedule_parse_boundary_rejects_type_schema_and_non_json_errors(
    raw_json: str,
) -> None:
    with pytest.raises(CandidateParseError):
        schedule_from_json(raw_json)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_bits", 0.0),
        ("data_bits", float("nan")),
        ("priority", 0.0),
        ("priority", float("inf")),
        ("deadline_s", 0.0),
        ("min_reliability", -0.1),
        ("min_reliability", 1.0),
        ("traffic_class", "unknown"),
    ],
)
def test_flow_rejects_invalid_values(
    fixed_scenario: EvaluatorScenario, field: str, value: object
) -> None:
    data = fixed_scenario.flows[0].model_dump(mode="python")
    data[field] = value

    with pytest.raises(ValidationError):
        Flow.model_validate(data)


def test_scenario_json_rejects_invalid_relations() -> None:
    raw = (FIXTURES / "basic_scenario.json").read_text(encoding="utf-8")
    invalid = raw.replace('"sinr_db": 0.0', '"sinr_db": -0.1')

    with pytest.raises(ScenarioValidationError, match="below operating SINR"):
        evaluator_scenario_from_json(invalid)


def test_scenario_rejects_noncontinuous_or_wrong_direction_path() -> None:
    raw = (FIXTURES / "basic_scenario.json").read_text(encoding="utf-8")
    invalid = raw.replace(
        '{"flow_id": 0, "path_id": 1, "node_sequence": [0, 1, 2]}',
        '{"flow_id": 0, "path_id": 1, "node_sequence": [0, 2, 1]}',
    )

    with pytest.raises(ScenarioValidationError):
        evaluator_scenario_from_json(invalid)


def test_scenario_rejects_catalog_order_not_sorted_by_hops() -> None:
    raw = (FIXTURES / "basic_scenario.json").read_text(encoding="utf-8")
    direct = '{"flow_id": 0, "path_id": 0, "node_sequence": [0, 2]}'
    via = '{"flow_id": 0, "path_id": 1, "node_sequence": [0, 1, 2]}'
    invalid = raw.replace(
        direct,
        '{"flow_id": 0, "path_id": 0, "node_sequence": [0, 1, 2]}',
    ).replace(
        via,
        '{"flow_id": 0, "path_id": 1, "node_sequence": [0, 2]}',
    )

    with pytest.raises(ScenarioValidationError):
        evaluator_scenario_from_json(invalid)


def test_path_catalog_rejects_duplicate_node_sequence() -> None:
    paths = (
        CandidatePath(flow_id=0, path_id=0, node_sequence=(0, 2)),
        CandidatePath(flow_id=0, path_id=1, node_sequence=(0, 2)),
    )

    with pytest.raises(ValidationError, match="unique node sequences"):
        PathCatalog(flow_id=0, paths=paths)


def test_path_catalog_rejects_more_than_five_paths() -> None:
    paths = tuple(
        CandidatePath(flow_id=0, path_id=index, node_sequence=(0, index + 1, 10))
        for index in range(6)
    )

    with pytest.raises(ValidationError):
        PathCatalog(flow_id=0, paths=paths)


def test_scenario_rejects_unknown_schema_version() -> None:
    raw = (FIXTURES / "basic_scenario.json").read_text(encoding="utf-8")
    invalid = raw.replace('"schema_version": "1.0"', '"schema_version": "2.0"', 1)

    with pytest.raises(ScenarioValidationError):
        evaluator_scenario_from_json(invalid)
