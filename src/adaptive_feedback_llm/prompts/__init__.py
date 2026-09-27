"""Deterministic assembly of the paper prompt."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from adaptive_feedback_llm.models import LLMScenarioView
from adaptive_feedback_llm.prompts.compact_context import compact_json, encode_context
from adaptive_feedback_llm.prompts.evidence_matrix import build_evidence_matrix
from adaptive_feedback_llm.prompts.global_advisor import build_global_brief

PROMPT_VERSION = "1.0"
SUPPORTED_PROMPT_VERSIONS = (PROMPT_VERSION,)


def _asset_text(name: str) -> str:
    return files(__package__).joinpath(name).read_text(encoding="utf-8")


def _require_paper_prompt(version: str) -> None:
    if version != PROMPT_VERSION:
        raise ValueError(f"unsupported prompt version: {version!r}; only the public paper prompt is supported")


def meta_prompt(version: str = PROMPT_VERSION) -> str:
    _require_paper_prompt(version)
    return _asset_text("instructions.txt")


def schedule_json_schema() -> str:
    return _asset_text("schedule_schema.json")


def minimal_example() -> str:
    return _asset_text("minimal_example.json")


def prompt_bundle_sha256(version: str = PROMPT_VERSION) -> str:
    """Hash the public paper bundle, including implementation hashes."""
    _require_paper_prompt(version)
    bundle = {
        "prompt_version": version,
        "instructions.txt": meta_prompt(version),
        "minimal_example.json": minimal_example(),
        "schedule_schema.json": schedule_json_schema(),
        "request_template.txt": _asset_text("request_template.txt"),
    }
    for name, key in (
        ("opportunities.py", "optimization_facts_implementation_sha256"),
        ("compact_context.py", "compact_context_implementation_sha256"),
        ("evidence_matrix.py", "evidence_matrix_implementation_sha256"),
        ("global_advisor.py", "global_advisor_implementation_sha256"),
    ):
        bundle[key] = hashlib.sha256(_asset_text(name).encode("utf-8")).hexdigest()
    canonical = json.dumps(bundle, ensure_ascii=False, allow_nan=False,
                           sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def render_prompt(view: LLMScenarioView, *, version: str = PROMPT_VERSION,
                  float_tolerance: float = 1e-9) -> str:
    """Render the paper prompt; tolerance is kept for caller compatibility."""
    _require_paper_prompt(version)
    context = encode_context(view)
    template = _asset_text("request_template.txt")
    template = template.replace("{{FACT_MATRIX_JSON}}", compact_json(build_evidence_matrix(view)))
    template = template.replace("{{GLOBAL_BRIEF_JSON}}", compact_json(build_global_brief(view)))
    return (
        template
        .replace("{{META_PROMPT}}", meta_prompt(version).strip())
        .replace("{{DECISION_JSON}}", compact_json(context["decision"]))
        .replace("{{EVIDENCE_JSON}}", compact_json(context["evidence"]))
        .replace("{{OUTPUT_SCHEMA}}", compact_json(json.loads(schedule_json_schema())))
        .replace("{{MINIMAL_EXAMPLE}}", compact_json(json.loads(minimal_example())))
    )


__all__ = ["PROMPT_VERSION", "SUPPORTED_PROMPT_VERSIONS", "meta_prompt", "minimal_example",
           "prompt_bundle_sha256", "render_prompt", "schedule_json_schema"]
