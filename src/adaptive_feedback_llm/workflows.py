"""Public LLM workflows with explicit network access and durable request admission."""

import hashlib
import os
import re
from pathlib import Path

from adaptive_feedback_llm.evaluator import evaluate_schedule, retain_stable_best
from adaptive_feedback_llm.exceptions import CandidateParseError, LLMProviderError, LLMServiceError
from adaptive_feedback_llm.io import write_json
from adaptive_feedback_llm.models import schedule_from_json
from adaptive_feedback_llm.profiles import PROMPT_SHA256, PROMPT_VERSION, ThinkingConfig, ThinkingSettings
from adaptive_feedback_llm.prompts import prompt_bundle_sha256, render_prompt
from adaptive_feedback_llm.solvers.llm.projection import build_llm_scenario_view
from adaptive_feedback_llm.solvers.llm.provider import DeterministicMockProvider, OpenAICompatibleProvider
from adaptive_feedback_llm.solvers.llm.runner import run_adaptive_llm


class RequestJournal:
    """Persist intent before dispatch; never retry or resume an ambiguous request."""

    def __init__(self, provider, output):
        self.provider = provider
        self.output = Path(output)
        self.failed = False

    def complete(self, prompt, *, request_id, **kwargs):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", request_id) or request_id in {".", ".."}:
            raise LLMServiceError("invalid request identifier", attempts=0)
        if self.failed:
            raise LLMServiceError("request dispatch halted after earlier failure", attempts=1)
        write_json(self.output / f"{request_id}.intent.json", {
            "request_id": request_id,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "state": "admitted; absence of a terminal record means unknown outcome",
        })
        try:
            result = self.provider.complete(prompt, request_id=request_id, **kwargs)
        except LLMProviderError:
            self.failed = True
            write_json(self.output / f"{request_id}.terminal.json", {
                "request_id": request_id, "state": "failed_or_remote_outcome_unknown",
                "retry_allowed": False,
            })
            raise
        except Exception:
            self.failed = True
            # Keep the intent intact; do not persist exception text or traceback.
            raise LLMServiceError("request failed; inspect admission journal", attempts=1) from None
        write_json(self.output / f"{request_id}.terminal.json", {
            "request_id": request_id, "state": "received",
            "response_sha256": hashlib.sha256(result.raw_response.encode()).hexdigest(),
        })
        return result


def one_round(scenario, provider):
    config = ThinkingConfig()
    prompt = render_prompt(build_llm_scenario_view(scenario, (), ()),
                           version=PROMPT_VERSION,
                           float_tolerance=scenario.numerics.float_tolerance)
    best = schedule = None
    records = []
    for call in range(1, 4):
        request_id = f"r1c{call}"
        try:
            completion = provider.complete(prompt, request_id=request_id,
                                           temperature=config.temperature,
                                           max_tokens=config.completion_token_limit)
            candidate = schedule_from_json(completion.raw_response)
        except (LLMProviderError, CandidateParseError):
            records.append({"candidate_id": request_id, "state": "invalid_or_unavailable"})
            continue
        evaluation = evaluate_schedule(scenario, candidate)
        records.append({"candidate_id": request_id,
                        "evaluation": evaluation.model_dump(mode="json")})
        if evaluation.feasible:
            retained = retain_stable_best(best, evaluation,
                                          tolerance=scenario.numerics.float_tolerance)
            if retained is evaluation:
                best, schedule = evaluation, candidate
    return {"method": "one_round_llm", "scenario_id": scenario.scenario_id,
            "schedule": schedule.model_dump(mode="json") if schedule else None,
            "evaluation": best.model_dump(mode="json") if best else None,
            "candidates": records}


def solve_llm(scenario, output, *, live=False, response=None, first_round_only=False):
    if prompt_bundle_sha256(PROMPT_VERSION) != PROMPT_SHA256:
        raise ValueError("paper prompt bundle identity mismatch")
    if live:
        settings = ThinkingSettings.from_environment(
            os.environ, config=ThinkingConfig(), thinking_mode="enabled", reasoning_effort="low")
        # Credentials must never appear in a user-supplied scenario, even in identifiers.
        if settings.api_key.get_secret_value() in scenario.model_dump_json():
            raise LLMServiceError("credential found in scenario input", attempts=0)
        provider = OpenAICompatibleProvider(settings, allow_network=True)
    else:
        if response is None:
            raise ValueError("mock mode requires an explicit response JSON file")
        provider = DeterministicMockProvider(response)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    journal = RequestJournal(provider, output / "requests")
    write_json(output / "run.json", {"profile": "paper", "live": live,
               "scenario_id": scenario.scenario_id, "prompt_sha256": PROMPT_SHA256,
               "method": "one_round_llm" if first_round_only else "full_llm",
               "max_http_requests": 3 if first_round_only else 60,
               "automatic_retry": False, "automatic_resume": False})
    result = (one_round(scenario, journal) if first_round_only else
              run_adaptive_llm(scenario, journal, ThinkingConfig(),
                               prompt_version=PROMPT_VERSION).model_dump(mode="json"))
    result["transport_halted"] = journal.failed
    write_json(output / "result.json", result)
    return result
