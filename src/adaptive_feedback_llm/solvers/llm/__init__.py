"""Adaptive Feedback-LLM solver and provider boundaries."""

from adaptive_feedback_llm.solvers.llm.projection import (
    build_llm_scenario_view,
    derive_qos_gaps,
    validate_evaluated_history,
)
from adaptive_feedback_llm.solvers.llm.provider import (
    CompletionResponse,
    DeterministicMockProvider,
    LLMProvider,
    OpenAICompatibleProvider,
    ProviderSettings,
    SUPPORTED_DEEPSEEK_MODELS,
    SupportedDeepSeekModel,
    endpoint_identity_sha256,
    load_provider_environment_file,
    select_provider,
)
from adaptive_feedback_llm.solvers.llm.runner import (
    AdaptiveLLMResult,
    AdaptiveRunStatus,
    AdaptiveStopReason,
    InteractionLog,
    run_adaptive_llm,
)

__all__ = [
    "AdaptiveLLMResult",
    "AdaptiveRunStatus",
    "AdaptiveStopReason",
    "CompletionResponse",
    "DeterministicMockProvider",
    "InteractionLog",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "ProviderSettings",
    "SUPPORTED_DEEPSEEK_MODELS",
    "SupportedDeepSeekModel",
    "endpoint_identity_sha256",
    "load_provider_environment_file",
    "build_llm_scenario_view",
    "derive_qos_gaps",
    "run_adaptive_llm",
    "select_provider",
    "validate_evaluated_history",
]
