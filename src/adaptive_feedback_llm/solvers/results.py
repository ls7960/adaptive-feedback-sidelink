"""Immutable unified solver result records."""

from __future__ import annotations

from pydantic import Field

from adaptive_feedback_llm.models import EvaluationResult, Schedule
from adaptive_feedback_llm.models.domain import FrozenStrictModel


class SolverResult(FrozenStrictModel):
    """Common schedule-plus-evaluation interface for every solver."""

    method: str = Field(min_length=1)
    schedule: Schedule
    evaluation: EvaluationResult
