"""Strict allowlisted LLM view and evaluated-history records."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from adaptive_feedback_llm.models.domain import (
    FrozenStrictModel,
    Schedule,
    SRUAllocation,
    TrafficClass,
)
from adaptive_feedback_llm.models.results import EvaluationResult


class LLMFlowView(FrozenStrictModel):
    flow_id: int = Field(ge=0)
    source: int = Field(ge=0)
    target: int = Field(ge=0)
    data_bits: float = Field(gt=0.0, allow_inf_nan=False)
    priority: float = Field(gt=0.0, allow_inf_nan=False)
    traffic_class: TrafficClass
    deadline_s: float = Field(gt=0.0, allow_inf_nan=False)
    min_reliability: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)


class LLMPathView(FrozenStrictModel):
    flow_id: int = Field(ge=0)
    path_id: int = Field(ge=0)
    node_sequence: tuple[int, ...]


class LLMHotspotView(FrozenStrictModel):
    src: int = Field(ge=0)
    dst: int = Field(ge=0)
    catalog_occurrences: int = Field(gt=0)
    capacity_srus: int = Field(gt=0)
    static_load: float = Field(gt=0.0, allow_inf_nan=False)


class LLMGuidanceView(FrozenStrictModel):
    flow_id: int = Field(ge=0)
    path_id: int = Field(ge=0)
    rank: int = Field(ge=0)
    single_flow_feasible: bool
    required_sru_total: int | None = Field(default=None, gt=0)
    reference_sru_allocations: tuple[SRUAllocation, ...] = ()
    hotspot_burden: float = Field(ge=0.0, allow_inf_nan=False)
    class_quality_rank: int | None = Field(default=None, ge=0)
    hop_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_reference_allocation(self) -> "LLMGuidanceView":
        if self.single_flow_feasible:
            if self.required_sru_total is None or not self.reference_sru_allocations:
                raise ValueError(
                    "feasible guidance requires a nonempty reference allocation"
                )
            if sum(item.num_srus for item in self.reference_sru_allocations) != self.required_sru_total:
                raise ValueError("reference allocation must sum to required_sru_total")
            if len(self.reference_sru_allocations) != self.hop_count:
                raise ValueError("reference allocation must cover every path hop")
        elif self.required_sru_total is not None or self.reference_sru_allocations:
            raise ValueError(
                "infeasible guidance cannot expose a reference allocation or required total"
            )
        return self


class LLMStaticScenarioView(FrozenStrictModel):
    scenario_id: str = Field(min_length=1)
    node_ids: tuple[int, ...]
    capacity_adjacency_matrix: tuple[tuple[int, ...], ...]
    flows: tuple[LLMFlowView, ...]
    paths: tuple[LLMPathView, ...]
    hotspots: tuple[LLMHotspotView, ...]
    guidance: tuple[LLMGuidanceView, ...]

    @model_validator(mode="after")
    def validate_square_capacity_matrix(self) -> "LLMStaticScenarioView":
        size = len(self.node_ids)
        if len(self.capacity_adjacency_matrix) != size or any(
            len(row) != size for row in self.capacity_adjacency_matrix
        ):
            raise ValueError("capacity adjacency matrix must be square")
        if any(value < 0 for row in self.capacity_adjacency_matrix for value in row):
            raise ValueError("capacity adjacency values cannot be negative")
        return self


class EvaluatedCandidateRecord(FrozenStrictModel):
    scenario_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=20)
    call_index: int = Field(ge=1, le=3)
    schedule: Schedule
    evaluation: EvaluationResult
    qos_gaps: tuple["FlowQoSGap", ...] = ()
    best_so_far: bool


class FlowQoSGap(FrozenStrictModel):
    flow_id: int = Field(ge=0)
    delay_gap_s: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    reliability_gap: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_a_reported_violation(self) -> "FlowQoSGap":
        if self.delay_gap_s is None and self.reliability_gap is None:
            raise ValueError("QoS gap record requires at least one violated QoS")
        return self


EvaluatedCandidateRecord.model_rebuild()


class FailureCategory(str, Enum):
    PARSE_FAILURE = "parse_failure"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVICE_ERROR = "service_error"


class FailureFact(str, Enum):
    PARSE_FAILURE = "Candidate response failed strict JSON/schema parsing."
    TIMEOUT = "Provider timeout or connection attempts were exhausted."
    RATE_LIMIT = "Provider rate-limit attempts were exhausted."
    SERVICE_ERROR = "Provider service request failed."


FAILURE_FACT_BY_CATEGORY = {
    FailureCategory.PARSE_FAILURE: FailureFact.PARSE_FAILURE,
    FailureCategory.TIMEOUT: FailureFact.TIMEOUT,
    FailureCategory.RATE_LIMIT: FailureFact.RATE_LIMIT,
    FailureCategory.SERVICE_ERROR: FailureFact.SERVICE_ERROR,
}


class LLMFailureFeedback(FrozenStrictModel):
    candidate_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=20)
    call_index: int = Field(ge=1, le=3)
    category: FailureCategory
    fact: FailureFact
    transport_attempts: int = Field(ge=1, le=3)

    @model_validator(mode="after")
    def validate_fixed_fact(self) -> "LLMFailureFeedback":
        if self.fact is not FAILURE_FACT_BY_CATEGORY[self.category]:
            raise ValueError("failure fact must match its fixed category mapping")
        return self


class LLMScenarioView(FrozenStrictModel):
    schema_version: Literal["1.1"] = "1.1"
    static: LLMStaticScenarioView
    evaluated_history: tuple[EvaluatedCandidateRecord, ...] = ()
    failure_feedback: tuple[LLMFailureFeedback, ...] = ()
