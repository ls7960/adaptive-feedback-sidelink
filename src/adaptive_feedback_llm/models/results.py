"""Immutable evaluator diagnostics and results."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from adaptive_feedback_llm.models.domain import FrozenStrictModel


class DiagnosticCode(str, Enum):
    """Stable machine-readable infeasibility reasons."""

    MISSING_FLOW = "missing_flow"
    DUPLICATE_FLOW = "duplicate_flow"
    UNKNOWN_FLOW = "unknown_flow"
    INVALID_UNSERVED_STATE = "invalid_unserved_state"
    INVALID_PATH_ID = "invalid_path_id"
    PATH_NOT_IN_CATALOG = "path_not_in_catalog"
    INVALID_CATALOG_PATH = "invalid_catalog_path"
    MISSING_ALLOCATION = "missing_allocation"
    DUPLICATE_ALLOCATION = "duplicate_allocation"
    PATH_EXTERNAL_ALLOCATION = "path_external_allocation"
    INVALID_SRU_COUNT = "invalid_sru_count"
    LINK_CAPACITY_EXCEEDED = "link_capacity_exceeded"
    DELAY_QOS_VIOLATION = "delay_qos_violation"
    RELIABILITY_QOS_VIOLATION = "reliability_qos_violation"


class Diagnostic(FrozenStrictModel):
    """One deterministic validation fact; it never prescribes a repair."""

    code: DiagnosticCode
    message: str = Field(min_length=1)
    flow_ids: tuple[int, ...] = ()
    link: tuple[int, int] | None = None
    required: float | int | None = None
    actual: float | int | None = None


class LinkUsage(FrozenStrictModel):
    """Aggregated exact SRU usage for a directed link."""

    src: int
    dst: int
    used_srus: int = Field(ge=0)
    capacity_srus: int = Field(gt=0)
    flow_ids: tuple[int, ...]


class FlowEvaluation(FrozenStrictModel):
    """Evaluator-derived metrics for one scenario flow."""

    flow_id: int = Field(ge=0)
    served: bool
    path_id: int
    bottleneck_rate_bps: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    completion_delay_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    delay_qos_met: bool | None = None
    reliability_qos_met: bool | None = None
    margin: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    utility: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)


class EvaluationResult(FrozenStrictModel):
    """Complete deterministic evaluation result."""

    feasible: bool
    objective: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    diagnostics: tuple[Diagnostic, ...]
    flow_evaluations: tuple[FlowEvaluation, ...]
    link_usage: tuple[LinkUsage, ...]

    @model_validator(mode="after")
    def check_objective_feasibility_relation(self) -> "EvaluationResult":
        if self.feasible != (self.objective is not None):
            raise ValueError("objective must be non-null exactly for feasible candidates")
        if self.feasible and self.diagnostics:
            raise ValueError("feasible candidates cannot carry infeasibility diagnostics")
        return self
