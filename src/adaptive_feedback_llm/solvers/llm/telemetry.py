"""Credential-free provider observations, kept outside the model's prompt view."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from adaptive_feedback_llm.models.domain import FrozenStrictModel


class TokenUsage(FrozenStrictModel):
    """Reported counters only. Missing counters remain unknown, not zero."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    prompt_cache_hit_tokens: int | None = Field(default=None, ge=0)
    prompt_cache_miss_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    invalid_fields: tuple[str, ...] = ()

    @property
    def non_reasoning_completion_tokens(self) -> int | None:
        if self.completion_tokens is None or self.reasoning_tokens is None:
            return None
        difference = self.completion_tokens - self.reasoning_tokens
        return difference if difference >= 0 else None


class ProviderAttemptAudit(FrozenStrictModel):
    attempt: int = Field(ge=1, le=3)
    outcome: Literal[
        "success", "timeout", "rate_limit", "http_error", "transport_error",
        "invalid_envelope", "empty_content", "output_truncated",
    ]
    elapsed_s: float = Field(ge=0.0, allow_inf_nan=False)
    http_status: int | None = Field(default=None, ge=100, le=999)
    response_id: str | None = Field(default=None, max_length=256)
    returned_model: str | None = Field(default=None, max_length=256)
    system_fingerprint: str | None = Field(default=None, max_length=256)
    finish_reason: Literal[
        "stop", "length", "content_filter", "tool_calls", "function_call",
        "insufficient_system_resource", "unknown",
    ] | None = None
    usage: TokenUsage | None = None
    final_content_characters: int | None = Field(default=None, ge=0)
    reasoning_content_characters: int | None = Field(default=None, ge=0)


class ProviderRequestAudit(FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1)
    started_at_utc: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    thinking_mode: Literal["enabled", "disabled"]
    reasoning_effort: Literal["low", "high", "max"] | None = None
    temperature_sent: float | None = Field(default=None, allow_inf_nan=False)
    temperature_applicable: bool
    max_tokens: int = Field(gt=0)
    timeout_s: float = Field(gt=0.0, allow_inf_nan=False)
    total_elapsed_s: float = Field(ge=0.0, allow_inf_nan=False)
    attempts: tuple[ProviderAttemptAudit, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_attempts(self) -> "ProviderRequestAudit":
        if tuple(a.attempt for a in self.attempts) != tuple(range(1, len(self.attempts) + 1)):
            raise ValueError("provider attempts must be consecutive")
        if self.thinking_mode == "disabled" and self.reasoning_effort is not None:
            raise ValueError("disabled thinking cannot carry a reasoning effort")
        if self.temperature_applicable != (self.thinking_mode == "disabled"):
            raise ValueError("temperature applicability must match thinking mode")
        if any(a.outcome == "success" for a in self.attempts[:-1]):
            raise ValueError("successful attempt must terminate a request")
        return self


def extract_usage(payload: object) -> TokenUsage | None:
    if not isinstance(payload, dict):
        return None
    values = {}
    invalid = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "reasoning_tokens"):
        source = payload
        if key == "reasoning_tokens":
            source = payload.get("completion_tokens_details", {})
        value = source.get(key) if isinstance(source, dict) else None
        if value is not None and (type(value) is not int or value < 0):
            invalid.append(key)
            value = None
        values[key] = value
    return TokenUsage(**values, invalid_fields=tuple(invalid))
