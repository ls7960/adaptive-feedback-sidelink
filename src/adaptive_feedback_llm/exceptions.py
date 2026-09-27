"""Project-specific exception hierarchy."""


class AdaptiveFeedbackLLMError(Exception):
    """Base class for expected project errors."""


class ConfigurationError(AdaptiveFeedbackLLMError):
    """Base class for configuration loading or validation failures."""


class ConfigurationFileError(ConfigurationError):
    """The configuration file cannot be read or decoded."""


class ConfigurationValidationError(ConfigurationError):
    """Decoded configuration data violates the frozen contract."""


class DomainValidationError(AdaptiveFeedbackLLMError):
    """Base class for invalid network, traffic, path, or scenario data."""


class ScenarioValidationError(DomainValidationError):
    """A complete evaluator scenario violates domain invariants."""


class ScenarioGenerationError(DomainValidationError):
    """A scenario-generation operation failed its frozen contract."""


class ScenarioGenerationExhaustedError(ScenarioGenerationError):
    """A scenario could not be generated within the configured attempt cap."""


class ScenarioArtifactError(ScenarioGenerationError):
    """A generated scenario artifact or its isolated path is invalid."""


class ScenarioCalibrationError(ScenarioGenerationError):
    """No candidate zeta passed the frozen calibration gates."""

    def __init__(self, message: str, *, report: object) -> None:
        super().__init__(message)
        self.report = report


class ScenarioBatchError(ScenarioGenerationError):
    """A batch or partition manifest could not be completed safely."""


class ExperimentError(AdaptiveFeedbackLLMError):
    """A experiment execution or checkpoint operation failed."""


class ExperimentRecordError(ExperimentError):
    """A immutable raw-result record is invalid or conflicting."""


class ExperimentExecutionError(ExperimentError):
    """A method attempt ended without a reusable formal checkpoint."""

    def __init__(self, message: str, *, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result


class PathCatalogError(DomainValidationError):
    """Base class for deterministic path-catalog construction failures."""


class InsufficientCandidatePathsError(PathCatalogError):
    """Formal mode could not construct the required five simple paths."""


class CandidateError(AdaptiveFeedbackLLMError):
    """Base class for candidate parsing or evaluation failures."""


class CandidateParseError(CandidateError):
    """A raw candidate is not exactly one supported JSON document."""


class EvaluationInvariantError(AdaptiveFeedbackLLMError):
    """The evaluator encountered invalid internal state, not candidate infeasibility."""


class SolverConfigurationError(AdaptiveFeedbackLLMError):
    """A solver run configuration violates a frozen protocol."""


class LLMProviderError(AdaptiveFeedbackLLMError):
    """Base class for exhausted or non-retryable provider failures."""

    def __init__(self, message: str, *, attempts: int, provider_audit=None) -> None:
        super().__init__(message)
        self.attempts = attempts
        # The runner accepts only the typed, allowlisted audit model, never error text.
        self.provider_audit = provider_audit


class LLMTimeoutError(LLMProviderError):
    """Timeout or connection attempts were exhausted."""


class LLMRateLimitError(LLMProviderError):
    """HTTP 429 attempts were exhausted."""


class LLMServiceError(LLMProviderError):
    """The provider returned another HTTP/protocol failure."""
