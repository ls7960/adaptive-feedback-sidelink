import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from adaptive_feedback_llm.cli import main
from adaptive_feedback_llm.io import load_dataset
from adaptive_feedback_llm.profiles import PROMPT_SHA256, PROMPT_VERSION
from adaptive_feedback_llm.prompts import prompt_bundle_sha256
from adaptive_feedback_llm.solvers import ga_portfolio

ROOT = Path(__file__).resolve().parents[1]


def test_published_summary_script():
    result = subprocess.run([sys.executable, str(ROOT / "tools/summarize_results.py")],
                            capture_output=True, text=True, check=True)
    assert "99.78%" in result.stdout and "fixed reference" in result.stdout


def test_published_prompt_identity():
    assert prompt_bundle_sha256(PROMPT_VERSION) == PROMPT_SHA256


def test_paper_config_matches_runtime_contracts():
    from adaptive_feedback_llm.config import load_config
    from adaptive_feedback_llm.profiles import ThinkingConfig, ThinkingSettings
    from adaptive_feedback_llm.solvers.genetic_algorithm import GARunConfig

    assert [p.name for p in (ROOT / "configs").glob("*.yaml")] == ["paper.yaml"]
    profile = load_config(ROOT / "configs/paper.yaml")
    assert profile.llm == ThinkingConfig()
    transport = ThinkingSettings(api_key="test-only", base_url="https://example.invalid")
    for field in ("model", "thinking_mode", "reasoning_effort", "request_timeout_s", "max_transport_attempts"):
        assert getattr(transport, field) == getattr(profile.llm, field)
    ga = profile.genetic_algorithm
    assert {k: tuple(v) for k, v in ga.components.items()} == ga_portfolio.COMPONENTS
    assert ga.independent_restarts == ga_portfolio.RESTARTS == GARunConfig().independent_restarts
    assert ga.population_evaluations_per_scenario == ga_portfolio.POPULATION_EVALUATIONS
    for field, value in ga.model_dump().items():
        if field not in {"components", "population_evaluations_per_scenario"}:
            assert getattr(GARunConfig(), field) == value


@pytest.mark.parametrize("field,value", [
    ("model", "deepseek-v4-pro"), ("completion_token_limit", 512),
    ("max_transport_attempts", 3), ("request_timeout_s", 30.0),
    ("thinking_mode", "disabled"), ("reasoning_effort", "high"),
    ("prompt_version", "unsupported"),
])
def test_obsolete_config_values_are_rejected(field, value):
    from pydantic import ValidationError
    from adaptive_feedback_llm.config import LLMConfig
    with pytest.raises(ValidationError):
        LLMConfig(**{field: value})


def test_public_example_prompt_identity():
    from adaptive_feedback_llm.io import load_scenario
    from adaptive_feedback_llm.prompts import render_prompt
    from adaptive_feedback_llm.solvers.llm import build_llm_scenario_view
    view = build_llm_scenario_view(load_scenario(ROOT / "examples/scenario.json"))
    assert hashlib.sha256(render_prompt(view).encode()).hexdigest() == (
        "8936ed99c3f153d081bc677d88972f9be9ebc5a499046af6ece9daca85d25e23"
    )
    with pytest.raises(ValueError, match="unsupported"):
        render_prompt(view, version="unsupported")


def test_benchmark_domain_and_membership():
    meta, scenes = load_dataset(ROOT / "data/benchmark/manifest.json")
    assert len(scenes) == 120 and meta["split"] == "test"
    assert {d: sum(e["difficulty"] == d for e in meta["entries"])
            for d in ("easy", "medium", "hard")} == {"easy": 40, "medium": 40, "hard": 40}


def test_first_round_and_mock_cli(tmp_path):
    assert main(["evaluate", str(ROOT / "examples/scenario.json"), str(ROOT / "examples/schedule.json")]) == 0
    assert main(["demo", str(ROOT / "examples/scenario.json"), "--output", str(tmp_path / "demo")]) == 0
    assert main(["llm", str(ROOT / "examples/scenario.json"), "--mock-response",
                 str(ROOT / "examples/schedule.json"), "--one-round", "--output", str(tmp_path / "round")]) == 0
    result = json.loads((tmp_path / "round/result.json").read_text())
    assert len(result["candidates"]) == 3
    assert result["evaluation"]["feasible"]


def test_published_aggregation():
    rows = json.loads((ROOT / "data/published/llm_metrics.json").read_text())
    stats = json.loads((ROOT / "data/published/statistics.json").read_text())
    assert len(rows) == 360
    for method in ("full_llm", "one_round_llm"):
        repeats = []
        for repeat in (1, 2, 3):
            group = [r for r in rows if r["repeat"] == repeat]
            assert len(group) == 120
            repeats.append(statistics.mean(100 * r[method + "_J"] / r["ga_J"] for r in group))
        expected = stats["methods"][method]["overall"]["quality_percent"]
        assert statistics.mean(repeats) == pytest.approx(expected["mean"], abs=1e-10)
        assert statistics.stdev(repeats) == pytest.approx(expected["sample_sd"], abs=1e-10)


def test_ga_portfolio_budget_and_selection_without_full_search(monkeypatch, catalog_scenario):
    calls = []
    class Result:
        def __init__(self, value):
            self.value = value
        def model_dump(self, **kwargs):
            return {"evaluation": {"objective": self.value}, "schedule": {"flows": []}}
    def solver(scenario, *, master_seed, config, parallel_restarts):
        calls.append(config)
        assert config.independent_restarts == 15 and not parallel_restarts
        return Result(2 if len(calls) < 3 else 1)
    monkeypatch.setattr(ga_portfolio, "solve_genetic_algorithm", solver)
    result = ga_portfolio.solve_portfolio(catalog_scenario)
    assert len(calls) == 3 and result["population_evaluations"] == 5760000
    assert result["selected_component"] == "balanced"


def test_dataset_traversal_rejected(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"schema": "sidelink-public-benchmark-v1",
        "split": "test", "count": 1, "entries": [{"path": "../outside.json",
        "scenario_id": "x", "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}]}))
    with pytest.raises(ValueError, match="escapes"):
        load_dataset(directory / "manifest.json")
