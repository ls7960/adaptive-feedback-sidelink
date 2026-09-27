from __future__ import annotations

from pydantic import ValidationError
import pytest

from adaptive_feedback_llm.exceptions import InsufficientCandidatePathsError
from adaptive_feedback_llm.guidance import build_static_guidance
from adaptive_feedback_llm.models import DirectedLink, EvaluatorScenario, Flow, model_to_json
from adaptive_feedback_llm.paths import build_candidate_path_catalog


def _link(src: int, dst: int) -> DirectedLink:
    return DirectedLink(
        src=src,
        dst=dst,
        capacity_srus=3,
        sinr_db=5.0,
        fixed_delay_s=0.01,
    )


def test_formal_path_catalog_has_deterministic_five_path_identity(
    fixed_scenario: EvaluatorScenario,
) -> None:
    flow = fixed_scenario.flows[0].model_copy(update={"target": 5})
    links = tuple(
        reversed(
            (
                _link(0, 5),
                _link(0, 1),
                _link(1, 5),
                _link(0, 2),
                _link(2, 5),
                _link(0, 3),
                _link(3, 5),
                _link(0, 4),
                _link(4, 5),
            )
        )
    )

    catalog = build_candidate_path_catalog(flow, links, formal=True)
    repeated = build_candidate_path_catalog(flow, tuple(reversed(links)), formal=True)

    expected = ((0, 5), (0, 1, 5), (0, 2, 5), (0, 3, 5), (0, 4, 5))
    assert tuple(path.node_sequence for path in catalog.paths) == expected
    assert tuple(path.path_id for path in catalog.paths) == tuple(range(5))
    assert catalog == repeated
    assert all(len(set(path.node_sequence)) == len(path.node_sequence) for path in catalog.paths)


def test_more_than_five_equal_hop_paths_select_global_lexicographic_first_five(
    fixed_scenario: EvaluatorScenario,
) -> None:
    flow = fixed_scenario.flows[0].model_copy(update={"target": 9})
    ordered_links = tuple(
        link
        for intermediate in range(1, 8)
        for link in (_link(0, intermediate), _link(intermediate, 9))
    )
    shuffled_links = tuple(ordered_links[index] for index in (9, 2, 13, 0, 7, 4, 11, 6, 1, 12, 3, 10, 5, 8))

    first = build_candidate_path_catalog(flow, shuffled_links, formal=True)
    second = build_candidate_path_catalog(flow, tuple(reversed(shuffled_links)), formal=True)

    expected = tuple((0, intermediate, 9) for intermediate in range(1, 6))
    assert tuple(path.node_sequence for path in first.paths) == expected
    assert tuple(path.path_id for path in first.paths) == tuple(range(5))
    assert first == second


def test_nonformal_path_catalog_can_have_fewer_than_five(
    fixed_scenario: EvaluatorScenario,
) -> None:
    catalog = build_candidate_path_catalog(
        fixed_scenario.flows[0], fixed_scenario.links, formal=False
    )

    assert len(catalog.paths) == 2


def test_formal_path_catalog_rejects_fewer_than_five(
    fixed_scenario: EvaluatorScenario,
) -> None:
    with pytest.raises(InsufficientCandidatePathsError):
        build_candidate_path_catalog(
            fixed_scenario.flows[0], fixed_scenario.links, formal=True
        )


def test_static_guidance_is_deterministic_nonbinding_and_has_no_exact_quality_leak(
    fixed_scenario: EvaluatorScenario,
) -> None:
    first = build_static_guidance(fixed_scenario)
    second = build_static_guidance(fixed_scenario)
    raw = model_to_json(first).lower()

    assert first == second
    assert "class_quality_rank" in raw
    assert '"class_quality"' not in raw
    for forbidden in (
        "sinr_db",
        "fixed_delay_s",
        "completion_delay",
        '"reliability"',
        '"schedule"',
    ):
        assert forbidden not in raw

    flow_zero = tuple(item for item in first.paths if item.flow_id == 0)
    flow_one = tuple(item for item in first.paths if item.flow_id == 1)
    assert flow_zero[0].path_id == 0
    assert flow_one[0].path_id == 1
    assert flow_zero[0].hotspot_burden == pytest.approx(1.0)
    assert next(item for item in flow_zero if item.path_id == 1).hotspot_burden == pytest.approx(4 / 3)
    assert all(not hasattr(item, "class_quality") for item in first.paths)
    for item in first.paths:
        if item.single_flow_feasible:
            assert item.reference_sru_allocations
            assert tuple(allocation.edge for allocation in item.reference_sru_allocations) == tuple(
                zip(item.node_sequence, item.node_sequence[1:])
            )
            assert sum(
                allocation.num_srus for allocation in item.reference_sru_allocations
            ) == item.required_sru_total
        else:
            assert item.reference_sru_allocations == ()
            assert item.required_sru_total is None
