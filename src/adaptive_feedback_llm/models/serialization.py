"""JSON boundaries for scenarios, schedules, and evaluator results."""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from adaptive_feedback_llm.exceptions import CandidateParseError, ScenarioValidationError
from adaptive_feedback_llm.models.domain import EvaluatorScenario, Schedule


class _StrictJSONError(ValueError):
    """Internal marker for JSON extensions forbidden by the project schema."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(token: str) -> object:
    raise _StrictJSONError(f"non-standard numeric token {token!r} is forbidden")


def _preflight_strict_json(raw_json: str) -> None:
    """Check standard JSON syntax, recursive key uniqueness, and finite tokens."""

    json.loads(
        raw_json,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonstandard_constant,
    )


def model_to_json(model: BaseModel, *, indent: int | None = 2) -> str:
    """Serialize an immutable model without modifying it."""

    return model.model_dump_json(indent=indent)


def schedule_from_json(raw_json: str) -> Schedule:
    """Parse exactly one strictly typed JSON schedule.

    Syntax, unknown fields, and primitive-type errors are parse failures.
    Representable schedule semantics are intentionally left to the evaluator.
    """

    try:
        _preflight_strict_json(raw_json)
        return Schedule.model_validate_json(raw_json)
    except (json.JSONDecodeError, _StrictJSONError, ValidationError, ValueError) as exc:
        raise CandidateParseError(f"Invalid schedule JSON: {exc}") from exc


def evaluator_scenario_from_json(raw_json: str) -> EvaluatorScenario:
    """Parse and fully validate one deterministic evaluator scenario."""

    try:
        _preflight_strict_json(raw_json)
        return EvaluatorScenario.model_validate_json(raw_json)
    except (json.JSONDecodeError, _StrictJSONError, ValidationError, ValueError) as exc:
        raise ScenarioValidationError(f"Invalid evaluator scenario JSON: {exc}") from exc
