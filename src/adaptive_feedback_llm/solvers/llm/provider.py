"""LLM provider protocol, deterministic mock, and OpenAI-compatible transport."""

from __future__ import annotations

import os
import time
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator

from adaptive_feedback_llm.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from adaptive_feedback_llm.config import LLMConfig
from adaptive_feedback_llm.models.domain import FrozenStrictModel
from adaptive_feedback_llm.solvers.llm.telemetry import (
    ProviderAttemptAudit, ProviderRequestAudit, extract_usage,
)


SUPPORTED_DEEPSEEK_MODELS = (
    "deepseek-v4-flash",
)
SupportedDeepSeekModel = Literal["deepseek-v4-flash"]


def endpoint_identity_sha256(base_url: str) -> str:
    """Return the non-secret identity digest used by experiment provenance."""

    if not isinstance(base_url, str) or not base_url:
        raise ValueError("provider base URL must be non-empty text")
    return hashlib.sha256(base_url.rstrip("/").encode("utf-8")).hexdigest()


class CompletionResponse(FrozenStrictModel):
    raw_response: str
    transport_attempts: int = Field(ge=1, le=3)
    provider_audit: ProviderRequestAudit | None = None


class ProviderSettings(FrozenStrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)
    api_key: SecretStr = Field(min_length=1, repr=False, exclude=True)
    base_url: str = Field(min_length=1, repr=False)
    model: SupportedDeepSeekModel = "deepseek-v4-flash"
    thinking_mode: Literal["enabled"] = "enabled"
    reasoning_effort: Literal["low"] = "low"
    request_timeout_s: Literal[240.0] = 240.0
    max_transport_attempts: Literal[1] = 1

    @field_validator("api_key")
    @classmethod
    def validate_key(cls, value):
        raw = value.get_secret_value()
        if raw != raw.strip() or any(ord(c) < 33 or ord(c) > 126 for c in raw):
            raise ValueError("invalid credential format")
        if raw in {"PASTE_API_KEY_HERE", "your-api-key", "changeme"}:
            raise ValueError("placeholder credentials are not accepted")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_endpoint(cls, value):
        try:
            parts = urlsplit(value)
            valid = (parts.scheme == "https" and parts.hostname and not parts.username
                     and not parts.password and not parts.query and not parts.fragment
                     and not any(c.isspace() for c in value) and parts.port != 0)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("endpoint must be HTTPS without credentials, query or fragment")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_reasoning_mode(self) -> "ProviderSettings":
        if self.api_key.get_secret_value() in self.base_url:
            raise ValueError("credentials must not occur in the endpoint")
        if self.thinking_mode == "disabled" and self.reasoning_effort is not None:
            raise ValueError("reasoning effort requires thinking enabled")
        return self

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        config: LLMConfig | None = None,
        thinking_mode: Literal["enabled"] = "enabled",
        reasoning_effort: Literal["low"] = "low",
    ) -> "ProviderSettings":
        values = os.environ if environment is None else environment
        try:
            api_key = values["LLM_API_KEY"]
            base_url = values["LLM_BASE_URL"]
        except KeyError as exc:
            raise LLMServiceError(
                f"missing required environment variable {exc.args[0]}", attempts=0
            ) from exc
        environment_model = values.get("LLM_MODEL", "deepseek-v4-flash")
        if environment_model not in SUPPORTED_DEEPSEEK_MODELS:
            raise LLMServiceError(
                "LLM_MODEL must be deepseek-v4-flash for the paper profile",
                attempts=0,
            )
        values_config = config
        try:
            return cls._from_values(api_key, base_url, environment_model, values_config,
                                    thinking_mode, reasoning_effort)
        except ValidationError:
            raise LLMServiceError("invalid provider configuration", attempts=0) from None

    @classmethod
    def _from_values(cls, api_key, base_url, environment_model, values_config,
                     thinking_mode, reasoning_effort):
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=environment_model,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            request_timeout_s=(
                values_config.request_timeout_s if values_config else 240.0
            ),
            max_transport_attempts=(
                values_config.max_transport_attempts if values_config else 1
            ),

        )


def load_provider_environment_file(path: str | Path) -> dict[str, str]:
    """Read an explicit local credential file without mutating process state."""

    credential_path = Path(path)
    try:
        raw = credential_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise LLMServiceError(
            f"cannot read local LLM credential file {credential_path}", attempts=0
        ) from exc

    allowed = {"LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or key not in allowed:
            raise LLMServiceError(
                f"invalid credential-file entry at line {line_number}", attempts=0
            )
        if key in values:
            raise LLMServiceError(
                f"duplicate credential-file key {key}", attempts=0
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value:
            raise LLMServiceError(
                f"credential-file value for {key} is empty", attempts=0
            )
        values[key] = value

    missing = allowed.difference(values)
    if missing:
        raise LLMServiceError(
            f"credential file is missing {','.join(sorted(missing))}", attempts=0
        )
    if values["LLM_API_KEY"] == "PASTE_API_KEY_HERE":
        raise LLMServiceError(
            "credential file still contains the API-key placeholder", attempts=0
        )
    if values["LLM_BASE_URL"] == "PASTE_OPENAI_COMPATIBLE_BASE_URL_HERE":
        raise LLMServiceError(
            "credential file still contains the base-URL placeholder", attempts=0
        )
    return values


class LLMProvider(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        request_id: str,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResponse: ...


class DeterministicMockProvider:
    """Repeat a fixed response sequence without network access."""

    def __init__(self, responses: str | tuple[str, ...]) -> None:
        self._responses = (responses,) if isinstance(responses, str) else responses
        if not self._responses:
            raise ValueError("mock provider requires at least one response")
        self._index = 0
        self.prompts: list[str] = []
        self.request_ids: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        request_id: str,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResponse:
        del temperature, max_tokens
        self.prompts.append(prompt)
        self.request_ids.append(request_id)
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return CompletionResponse(raw_response=response, transport_attempts=1)


class OpenAICompatibleProvider:
    """Single-attempt OpenAI-compatible transport for the paper protocol."""

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        allow_network: bool = False,
    ) -> None:
        if transport is None and not allow_network:
            raise LLMServiceError("network access requires explicit opt-in", attempts=0)
        self._settings = settings
        self._transport = transport

    def _contains_secret(self, value: str) -> bool:
        key = self._settings.api_key.get_secret_value()
        if key in value:
            return True
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return False
        return key in json.dumps(decoded, ensure_ascii=False)

    @property
    def model_name(self) -> str:
        return self._settings.model

    @property
    def endpoint_identity_sha256(self) -> str:
        return endpoint_identity_sha256(self._settings.base_url)

    def _safe_identifier(self, value: object) -> str | None:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,256}", value):
            return None
        if self._settings.api_key.get_secret_value() in value:
            return None
        if self._settings.base_url in value:
            return None
        return value

    def complete(
        self,
        prompt: str,
        *,
        request_id: str,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResponse:
        if self._contains_secret(prompt) or self._contains_secret(request_id):
            raise LLMServiceError("credential found in request content", attempts=0)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", request_id):
            raise LLMServiceError("invalid request identifier", attempts=0)
        url = f"{self._settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        }
        payload = {
            "model": self._settings.model,
            "thinking": {"type": self._settings.thinking_mode},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._settings.reasoning_effort is not None:
            payload["reasoning_effort"] = self._settings.reasoning_effort
        # Omit an ineffective sampling parameter in the new explicit reasoning arm.
        # Legacy implicit-effort requests retain their original request shape.
        if self._settings.thinking_mode == "enabled" and self._settings.reasoning_effort is not None:
            payload.pop("temperature")
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        audits: list[ProviderAttemptAudit] = []

        def request_audit() -> ProviderRequestAudit:
            return ProviderRequestAudit(
                request_id=request_id, started_at_utc=started_at,
                requested_model=self._settings.model,
                thinking_mode=self._settings.thinking_mode,
                reasoning_effort=self._settings.reasoning_effort,
                temperature_sent=payload.get("temperature"),
                temperature_applicable=self._settings.thinking_mode == "disabled",
                max_tokens=max_tokens, timeout_s=self._settings.request_timeout_s,
                total_elapsed_s=time.perf_counter() - started, attempts=tuple(audits),
            )

        with httpx.Client(
            timeout=self._settings.request_timeout_s,
            transport=self._transport,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            for attempt in range(1, self._settings.max_transport_attempts + 1):
                attempt_started = time.perf_counter()
                status = None
                metadata = {}
                content = None
                category = LLMServiceError
                try:
                    response = client.post(url, headers=headers, json=payload)
                    status = response.status_code
                    # Never retain HTTP error bodies, request headers or exception text.
                    if status == 429:
                        outcome, category = "rate_limit", LLMRateLimitError
                    elif 500 <= status < 600:
                        outcome = "http_error"
                    elif status >= 300:
                        outcome = "http_error"
                    else:
                        outcome = "invalid_envelope"
                        try:
                            body = response.json()
                            if isinstance(body, dict):
                                metadata = {
                                    "response_id": self._safe_identifier(body.get("id")),
                                    "returned_model": self._safe_identifier(body.get("model")),
                                    "system_fingerprint": self._safe_identifier(body.get("system_fingerprint")),
                                    "usage": extract_usage(body.get("usage")),
                                }
                                choice = body["choices"][0]
                                finish = choice.get("finish_reason")
                                metadata["finish_reason"] = finish if finish in (
                                    None, "stop", "length", "content_filter", "tool_calls",
                                    "function_call", "insufficient_system_resource",
                                ) else "unknown"
                                message = choice["message"]
                                content = message.get("content")
                                reasoning = message.get("reasoning_content")
                                metadata["final_content_characters"] = len(content) if isinstance(content, str) else None
                                metadata["reasoning_content_characters"] = len(reasoning) if isinstance(reasoning, str) else None
                                if finish == "length":
                                    outcome = "output_truncated"
                                elif isinstance(content, str):
                                    outcome = "success" if content.strip() else "empty_content"
                                    if self._contains_secret(content):
                                        content = None
                                        outcome = "invalid_envelope"
                        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                            outcome = "invalid_envelope"
                except (httpx.TimeoutException, httpx.ConnectError):
                    outcome, category = "timeout", LLMTimeoutError
                except httpx.TransportError:
                    outcome = "transport_error"
                audits.append(ProviderAttemptAudit(
                    attempt=attempt, outcome=outcome, http_status=status,
                    elapsed_s=time.perf_counter() - attempt_started, **metadata,
                ))
                if outcome == "success":
                    return CompletionResponse(raw_response=content, transport_attempts=attempt,
                                              provider_audit=request_audit())
                raise category("provider request failed", attempts=attempt,
                               provider_audit=request_audit())

        raise LLMServiceError(
            "provider request ended without a response",
            attempts=self._settings.max_transport_attempts,
        )


def select_provider(
    config: LLMConfig,
    *,
    environment: Mapping[str, str] | None = None,
    mock_responses: str | tuple[str, ...] = '{"flows":[]}',
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    thinking_mode: Literal["enabled"] = "enabled",
    reasoning_effort: Literal["low"] = "low",
    allow_network: bool = False,
) -> LLMProvider:
    """Use HTTP only with complete credentials; otherwise select the mock."""

    values = os.environ if environment is None else environment
    environment_model = values.get("LLM_MODEL")
    if (
        environment_model is not None
        and environment_model not in SUPPORTED_DEEPSEEK_MODELS
    ):
        raise LLMServiceError(
            "LLM_MODEL must be deepseek-v4-flash for the paper profile", attempts=0
        )
    has_key = bool(values.get("LLM_API_KEY"))
    has_base = bool(values.get("LLM_BASE_URL"))
    if not has_key and not has_base:
        return DeterministicMockProvider(mock_responses)
    if has_key != has_base:
        raise LLMServiceError(
            "LLM_API_KEY and LLM_BASE_URL must either both be set or both be absent",
            attempts=0,
        )
    return OpenAICompatibleProvider(
        ProviderSettings.from_environment(
            values,
            config=config,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        ),
        transport=transport,
        sleep=sleep,
        allow_network=allow_network,
    )
