"""Adaptive three-calls-per-round LLM orchestration with factual history only."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from adaptive_feedback_llm.config import LLMConfig
from adaptive_feedback_llm.evaluator import evaluate_schedule, retain_stable_best
from adaptive_feedback_llm.exceptions import (
    CandidateParseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from adaptive_feedback_llm.models import (
    EvaluatedCandidateRecord,
    EvaluationResult,
    EvaluatorScenario,
    FailureCategory,
    FailureFact,
    LLMFailureFeedback,
    Schedule,
    schedule_from_json,
)
from adaptive_feedback_llm.models.domain import FrozenStrictModel
from adaptive_feedback_llm.models.llm import FAILURE_FACT_BY_CATEGORY
from adaptive_feedback_llm.prompts import render_prompt
from adaptive_feedback_llm.solvers.llm.projection import (
    build_llm_scenario_view,
    derive_qos_gaps,
)
from adaptive_feedback_llm.solvers.llm.provider import LLMProvider
from adaptive_feedback_llm.solvers.llm.telemetry import ProviderRequestAudit


class AdaptiveRunStatus(str, Enum):
    FEASIBLE = "feasible"
    REJECTED = "rejected"
    SERVICE_FAILURE = "service_failure"


class AdaptiveStopReason(str, Enum):
    FEASIBLE_STAGNATION = "feasible_stagnation"
    MAX_ROUNDS = "max_rounds"


def _qualifies_for_stagnation(
    evaluation: EvaluationResult | None,
    *,
    tolerance: float,
) -> bool:
    """Require a nonzero feasible incumbent before patience can advance."""

    return bool(
        evaluation is not None
        and evaluation.feasible
        and evaluation.objective is not None
        and evaluation.objective > tolerance
    )


class InteractionLog(FrozenStrictModel):
    candidate_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=20)
    call_index: int = Field(ge=1, le=3)
    prompt: str = Field(min_length=1)
    raw_response: str | None = None
    transport_attempts: int = Field(default=1, ge=1, le=3)
    provider_audit: ProviderRequestAudit | None = None
    failure_category: FailureCategory | None = None
    failure_fact: FailureFact | None = None
    schedule: Schedule | None = None
    evaluation: EvaluationResult | None = None
    selected: bool = False

    @model_serializer(mode="wrap")
    def serialize_optional_audit(self, handler):
        data = handler(self)
        if self.provider_audit is None:
            data.pop("provider_audit", None)
        return data

    @model_validator(mode="after")
    def validate_exactly_one_outcome(self) -> "InteractionLog":
        if self.provider_audit is not None:
            if len(self.provider_audit.attempts) != self.transport_attempts:
                raise ValueError("provider audit must match actual transport attempts")
        if (self.failure_category is None) != (self.failure_fact is None):
            raise ValueError("failure category and fact must occur together")
        if (
            self.failure_category is not None
            and self.failure_fact is not FAILURE_FACT_BY_CATEGORY[self.failure_category]
        ):
            raise ValueError("interaction failure fact must match fixed category")
        provider_failure = self.failure_category in {
            FailureCategory.TIMEOUT,
            FailureCategory.RATE_LIMIT,
            FailureCategory.SERVICE_ERROR,
        }
        parse_failure = self.failure_category is FailureCategory.PARSE_FAILURE
        evaluated = self.schedule is not None or self.evaluation is not None
        if sum((provider_failure, parse_failure, evaluated)) != 1:
            raise ValueError("interaction must have exactly one outcome state")
        if provider_failure and (
            self.raw_response is not None
            or self.schedule is not None
            or self.evaluation is not None
        ):
            raise ValueError("provider failure cannot contain response or evaluation")
        if parse_failure and (
            self.raw_response is None
            or self.schedule is not None
            or self.evaluation is not None
            or provider_failure
        ):
            raise ValueError("parse failure must contain raw response only")
        if evaluated and (
            self.raw_response is None
            or self.schedule is None
            or self.evaluation is None
            or provider_failure
            or parse_failure
        ):
            raise ValueError("evaluated interaction requires raw, schedule, and evaluation")
        if self.selected and (
            self.evaluation is None or not self.evaluation.feasible
        ):
            raise ValueError("only a feasible evaluated interaction may be selected")
        return self


class AdaptiveLLMResult(FrozenStrictModel):
    method: Literal["adaptive_feedback_llm"] = "adaptive_feedback_llm"
    scenario_id: str = Field(min_length=1)
    float_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    status: AdaptiveRunStatus
    stop_reason: AdaptiveStopReason
    schedule: Schedule | None = None
    evaluation: EvaluationResult | None = None
    evaluated_history: tuple[EvaluatedCandidateRecord, ...]
    interactions: tuple[InteractionLog, ...]
    rounds_executed: int = Field(ge=1, le=20)
    independent_candidate_calls: int = Field(ge=3, le=60)

    @property
    def rounds_completed(self) -> int:
        """Backward-compatible read-only name for the executed-round count."""

        return self.rounds_executed

    @model_validator(mode="after")
    def validate_result_state(self) -> "AdaptiveLLMResult":
        if len(self.interactions) != self.independent_candidate_calls:
            raise ValueError("logical calls must equal interaction records")
        if self.independent_candidate_calls != self.rounds_executed * 3:
            raise ValueError("every completed round must contain exactly three calls")
        expected_identity = tuple(
            (
                f"r{position // 3 + 1}c{position % 3 + 1}",
                position // 3 + 1,
                position % 3 + 1,
            )
            for position in range(self.independent_candidate_calls)
        )
        actual_identity = tuple(
            (item.candidate_id, item.round_index, item.call_index)
            for item in self.interactions
        )
        if actual_identity != expected_identity:
            raise ValueError("interactions must be complete, unique, and chronological")

        provider_failure_categories = {
            FailureCategory.TIMEOUT,
            FailureCategory.RATE_LIMIT,
            FailureCategory.SERVICE_ERROR,
        }
        all_provider_failures = all(
            item.failure_category in provider_failure_categories
            for item in self.interactions
        )
        feasible_interactions = tuple(
            item
            for item in self.interactions
            if item.evaluation is not None and item.evaluation.feasible
        )
        if feasible_interactions:
            derived_status = AdaptiveRunStatus.FEASIBLE
        elif all_provider_failures:
            derived_status = AdaptiveRunStatus.SERVICE_FAILURE
        else:
            derived_status = AdaptiveRunStatus.REJECTED
        if self.status is not derived_status:
            raise ValueError("status must be derived from all interaction outcomes")

        if not feasible_interactions and self.rounds_executed != 20:
            raise ValueError(
                "a run without a feasible candidate must exhaust twenty rounds"
            )

        if self.status is AdaptiveRunStatus.FEASIBLE:
            if (
                self.schedule is None
                or self.evaluation is None
                or not self.evaluation.feasible
            ):
                raise ValueError("feasible run requires a feasible schedule evaluation")
        elif self.schedule is not None or self.evaluation is not None:
            raise ValueError("non-feasible run cannot return a fallback schedule")
        selected = tuple(item for item in self.interactions if item.selected)
        evaluated_logs = tuple(
            item for item in self.interactions if item.evaluation is not None
        )
        if len(evaluated_logs) != len(self.evaluated_history):
            raise ValueError("history must contain every and only evaluated interaction")
        for interaction, record in zip(
            evaluated_logs, self.evaluated_history, strict=True
        ):
            if (
                record.scenario_id != self.scenario_id
                or record.candidate_id != interaction.candidate_id
                or record.round_index != interaction.round_index
                or record.call_index != interaction.call_index
                or record.schedule != interaction.schedule
                or record.evaluation != interaction.evaluation
                or record.qos_gaps != derive_qos_gaps(record.evaluation)
            ):
                raise ValueError("history record does not match evaluated interaction")

        stable_best: EvaluatedCandidateRecord | None = None
        no_improvement_rounds = 0
        records_by_round: dict[int, list[EvaluatedCandidateRecord]] = {
            round_index: [] for round_index in range(1, self.rounds_executed + 1)
        }
        for record in self.evaluated_history:
            records_by_round[record.round_index].append(record)
        replay_best: EvaluationResult | None = None
        expected_stop_round: int | None = None
        for round_index in range(1, self.rounds_executed + 1):
            had_qualifying_best_at_round_start = _qualifies_for_stagnation(
                replay_best,
                tolerance=self.float_tolerance,
            )
            round_strictly_improved = False
            for record in records_by_round[round_index]:
                if not record.evaluation.feasible:
                    continue
                retained = retain_stable_best(
                    replay_best,
                    record.evaluation,
                    tolerance=self.float_tolerance,
                )
                if retained is record.evaluation:
                    replay_best = record.evaluation
                    round_strictly_improved = True
            if not _qualifies_for_stagnation(
                replay_best,
                tolerance=self.float_tolerance,
            ):
                no_improvement_rounds = 0
                continue
            if not had_qualifying_best_at_round_start:
                no_improvement_rounds = 0
            elif round_strictly_improved:
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1
            if no_improvement_rounds < 3 or round_index == 20:
                continue
            expected_stop_round = round_index
            break

        if expected_stop_round is not None and expected_stop_round < self.rounds_executed:
            raise ValueError("execution continued beyond the stagnation stop")
        if self.rounds_executed == 20:
            expected_stop_reason = AdaptiveStopReason.MAX_ROUNDS
        else:
            expected_stop_reason = AdaptiveStopReason.FEASIBLE_STAGNATION
            if expected_stop_round != self.rounds_executed:
                raise ValueError(
                    "early stop requires three complete no-improvement rounds "
                    "after a positive feasible incumbent"
                )
        if self.stop_reason is not expected_stop_reason:
            raise ValueError("stop reason does not match replayed round progress")
        if (
            self.status is not AdaptiveRunStatus.FEASIBLE
            and self.stop_reason is not AdaptiveStopReason.MAX_ROUNDS
        ):
            raise ValueError("non-feasible status can stop only at max rounds")

        for record in self.evaluated_history:
            if not record.evaluation.feasible:
                continue
            retained = retain_stable_best(
                None if stable_best is None else stable_best.evaluation,
                record.evaluation,
                tolerance=self.float_tolerance,
            )
            if retained is record.evaluation:
                stable_best = record
        marked_best = tuple(item for item in self.evaluated_history if item.best_so_far)
        if self.status is AdaptiveRunStatus.FEASIBLE:
            if len(selected) != 1 or len(marked_best) != 1 or stable_best is None:
                raise ValueError("feasible result requires one selected stable best")
            selected_item = selected[0]
            if (
                selected_item.candidate_id != stable_best.candidate_id
                or marked_best[0] != stable_best
                or self.schedule != stable_best.schedule
                or self.evaluation != stable_best.evaluation
                or selected_item.schedule != self.schedule
                or selected_item.evaluation != self.evaluation
            ):
                raise ValueError("top-level selection must equal the stable official best")
        elif selected or marked_best or stable_best is not None:
            raise ValueError("non-feasible result cannot select or mark a best record")
        return self


def _mark_best(
    history: list[EvaluatedCandidateRecord],
    best_candidate_id: str | None,
) -> tuple[EvaluatedCandidateRecord, ...]:
    return tuple(
        item.model_copy(update={"best_so_far": item.candidate_id == best_candidate_id})
        for item in history
    )


def run_adaptive_llm(
    scenario: EvaluatorScenario,
    provider: LLMProvider,
    config: LLMConfig,
    *,
    prompt_version: str | None = None,
) -> AdaptiveLLMResult:
    """Run complete rounds to the cap or post-feasibility stagnation."""

    history: list[EvaluatedCandidateRecord] = []
    logs: list[InteractionLog] = []
    best_candidate_id: str | None = None
    best_schedule: Schedule | None = None
    best_evaluation: EvaluationResult | None = None
    stop_reason: AdaptiveStopReason | None = None
    no_improvement_rounds = 0
    logical_calls = 0
    provider_failures = 0
    previous_round_failures: tuple[LLMFailureFeedback, ...] = ()
    for round_index in range(1, config.max_rounds + 1):
        had_qualifying_best_at_round_start = _qualifies_for_stagnation(
            best_evaluation,
            tolerance=scenario.numerics.float_tolerance,
        )
        round_strictly_improved = False
        view = build_llm_scenario_view(
            scenario,
            _mark_best(history, best_candidate_id),
            previous_round_failures,
        )
        render_arguments = {"float_tolerance": scenario.numerics.float_tolerance}
        if prompt_version is not None:
            render_arguments["version"] = prompt_version
        prompt = render_prompt(view, **render_arguments)
        round_failures: list[LLMFailureFeedback] = []
        for call_index in range(1, config.calls_per_round + 1):
            logical_calls += 1
            candidate_id = f"r{round_index}c{call_index}"
            try:
                completion = provider.complete(
                    prompt,
                    request_id=candidate_id,
                    temperature=config.temperature,
                    max_tokens=config.completion_token_limit,
                )
            except LLMProviderError as exc:
                provider_failures += 1
                if isinstance(exc, LLMTimeoutError):
                    category = FailureCategory.TIMEOUT
                elif isinstance(exc, LLMRateLimitError):
                    category = FailureCategory.RATE_LIMIT
                else:
                    category = FailureCategory.SERVICE_ERROR
                fact = FAILURE_FACT_BY_CATEGORY[category]
                round_failures.append(
                    LLMFailureFeedback(
                        candidate_id=candidate_id,
                        round_index=round_index,
                        call_index=call_index,
                        category=category,
                        fact=fact,
                        transport_attempts=exc.attempts,
                    )
                )
                logs.append(
                    InteractionLog(
                        candidate_id=candidate_id,
                        round_index=round_index,
                        call_index=call_index,
                        prompt=prompt,
                        failure_category=category,
                        failure_fact=fact,
                        transport_attempts=exc.attempts,
                        provider_audit=(exc.provider_audit if isinstance(exc.provider_audit, ProviderRequestAudit) else None),
                    )
                )
                continue

            try:
                schedule = schedule_from_json(completion.raw_response)
            except CandidateParseError:
                category = FailureCategory.PARSE_FAILURE
                fact = FAILURE_FACT_BY_CATEGORY[category]
                round_failures.append(
                    LLMFailureFeedback(
                        candidate_id=candidate_id,
                        round_index=round_index,
                        call_index=call_index,
                        category=category,
                        fact=fact,
                        transport_attempts=completion.transport_attempts,
                    )
                )
                logs.append(
                    InteractionLog(
                        candidate_id=candidate_id,
                        round_index=round_index,
                        call_index=call_index,
                        prompt=prompt,
                        raw_response=completion.raw_response,
                        transport_attempts=completion.transport_attempts,
                        provider_audit=completion.provider_audit,
                        failure_category=category,
                        failure_fact=fact,
                    )
                )
                continue

            evaluation = evaluate_schedule(scenario, schedule)
            record = EvaluatedCandidateRecord(
                scenario_id=scenario.scenario_id,
                candidate_id=candidate_id,
                round_index=round_index,
                call_index=call_index,
                schedule=schedule,
                evaluation=evaluation,
                qos_gaps=derive_qos_gaps(evaluation),
                best_so_far=False,
            )
            history.append(record)
            if evaluation.feasible:
                retained = retain_stable_best(
                    best_evaluation,
                    evaluation,
                    tolerance=scenario.numerics.float_tolerance,
                )
                if retained is evaluation:
                    best_candidate_id = candidate_id
                    best_schedule = schedule
                    best_evaluation = evaluation
                    round_strictly_improved = True
            logs.append(
                InteractionLog(
                    candidate_id=candidate_id,
                    round_index=round_index,
                    call_index=call_index,
                    prompt=prompt,
                    raw_response=completion.raw_response,
                    transport_attempts=completion.transport_attempts,
                    provider_audit=completion.provider_audit,
                    schedule=schedule,
                    evaluation=evaluation,
                )
            )
        previous_round_failures = tuple(round_failures)
        if round_index == config.max_rounds:
            stop_reason = AdaptiveStopReason.MAX_ROUNDS
            break
        if _qualifies_for_stagnation(
            best_evaluation,
            tolerance=scenario.numerics.float_tolerance,
        ):
            if not had_qualifying_best_at_round_start:
                no_improvement_rounds = 0
            elif round_strictly_improved:
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1
            if no_improvement_rounds >= config.early_stopping_patience_rounds:
                stop_reason = AdaptiveStopReason.FEASIBLE_STAGNATION
                break

    marked_history = _mark_best(history, best_candidate_id)
    final_logs = tuple(
        item.model_copy(update={"selected": item.candidate_id == best_candidate_id})
        for item in logs
    )
    if best_evaluation is not None:
        status = AdaptiveRunStatus.FEASIBLE
    elif provider_failures == logical_calls:
        status = AdaptiveRunStatus.SERVICE_FAILURE
    else:
        status = AdaptiveRunStatus.REJECTED
    if stop_reason is None:
        raise RuntimeError("adaptive runner exited without a stop reason")
    return AdaptiveLLMResult(
        scenario_id=scenario.scenario_id,
        float_tolerance=scenario.numerics.float_tolerance,
        status=status,
        stop_reason=stop_reason,
        schedule=best_schedule,
        evaluation=best_evaluation,
        evaluated_history=marked_history,
        interactions=final_logs,
        rounds_executed=logical_calls // config.calls_per_round,
        independent_candidate_calls=logical_calls,
    )
