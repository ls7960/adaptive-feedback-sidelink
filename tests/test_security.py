import importlib.util
import json
import socket
import subprocess
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from adaptive_feedback_llm.cli import main
from adaptive_feedback_llm.exceptions import LLMServiceError, LLMProviderError
from adaptive_feedback_llm.profiles import ThinkingConfig, ThinkingSettings
from adaptive_feedback_llm.solvers.llm.provider import OpenAICompatibleProvider, ProviderSettings
from adaptive_feedback_llm.workflows import RequestJournal, solve_llm

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "test-" + "qR8z" * 12


def scanner():
    spec = importlib.util.spec_from_file_location("release_secret_scan", ROOT / "tools/check_secrets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(**kwargs):
    return ThinkingSettings(api_key=FAKE_KEY, base_url="https://provider.invalid/v1", **kwargs)


def test_default_provider_cannot_open_network():
    with pytest.raises(LLMServiceError, match="opt-in"):
        OpenAICompatibleProvider(settings())
    with pytest.raises(AssertionError, match="forbidden"):
        socket.create_connection(("example.invalid", 443))


@pytest.mark.parametrize("url", ["http://example.invalid/v1", "https://" + "u:p@" + "example.invalid",
                                "https://example.invalid/v1?key=x", "https://example.invalid/#x",
                                "https://example.invalid:bad", "https://example.invalid /v1"])
def test_endpoint_validation_does_not_echo_credentials(url):
    with pytest.raises(ValidationError) as error:
        ProviderSettings(api_key=FAKE_KEY, base_url=url)
    assert FAKE_KEY not in str(error.value)


def test_environment_validation_and_serialization_hide_key():
    value = settings()
    assert FAKE_KEY not in repr(value)
    assert "api_key" not in value.model_dump()
    assert FAKE_KEY not in value.model_dump_json()
    with pytest.raises(LLMServiceError) as error:
        ThinkingSettings.from_environment({"LLM_API_KEY": FAKE_KEY,
            "LLM_BASE_URL": "http://invalid", "LLM_MODEL": "deepseek-v4-flash"},
            config=ThinkingConfig(), thinking_mode="enabled", reasoning_effort="low")
    assert FAKE_KEY not in str(error.value)


def test_endpoint_cannot_embed_key_and_journal_cannot_escape(tmp_path):
    with pytest.raises(ValidationError):
        ThinkingSettings(api_key=FAKE_KEY, base_url="https://provider.invalid/" + FAKE_KEY)
    journal = RequestJournal(None, tmp_path / "requests")
    with pytest.raises(LLMServiceError):
        journal.complete("test", request_id="../escape", temperature=0.3, max_tokens=512)
    assert not list(tmp_path.iterdir())


def test_json_escaped_credential_echo_is_rejected():
    escaped = "".join("\\u" + f"{ord(char):04x}" for char in FAKE_KEY)
    content = '{"value":"' + escaped + '"}'
    provider = OpenAICompatibleProvider(settings(), transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": content}}]})))
    with pytest.raises(LLMServiceError):
        provider.complete("test", request_id="r1c1", temperature=0.3, max_tokens=512)


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_failures_never_echo_or_retry(status, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, text=FAKE_KEY,
                              headers={"Location": "https://untrusted.invalid"})
    provider = OpenAICompatibleProvider(settings(), transport=httpx.MockTransport(handler))
    journal = RequestJournal(provider, tmp_path)
    for request_id in ["r1c1", "r1c2"]:
        with pytest.raises(LLMProviderError) as error:
            journal.complete("test", request_id=request_id, temperature=0.3, max_tokens=32768)
        assert FAKE_KEY not in str(error.value)
    assert len(calls) == 1
    assert all(FAKE_KEY not in p.read_text() for p in tmp_path.glob("*.json"))


def test_echoed_key_and_secret_in_prompt_are_rejected():
    provider = OpenAICompatibleProvider(settings(), transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": FAKE_KEY}}]})))
    for prompt in ["test", FAKE_KEY]:
        with pytest.raises(LLMServiceError):
            provider.complete(prompt, request_id="r1c1", temperature=0.3, max_tokens=512)


def test_live_flag_and_output_overwrite_gates(tmp_path, monkeypatch, catalog_scenario):
    from adaptive_feedback_llm.models import model_to_json
    from adaptive_feedback_llm.io import load_schedule
    response = model_to_json(load_schedule(ROOT / "examples/schedule.json"))
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.invalid/v1")
    out = tmp_path / "run"
    result = solve_llm(catalog_scenario, out, response=response)
    assert result["evaluation"]["feasible"]
    with pytest.raises(FileExistsError):
        solve_llm(catalog_scenario, out, response=response)
    assert FAKE_KEY not in (out / "result.json").read_text()
    with pytest.raises(SystemExit):
        main(["llm", str(ROOT / "examples/scenario.json"), "--output", str(tmp_path / "bad")])


def test_scan_reports_locations_without_values():
    scan = scanner()
    token = "sk-" + "A9b" * 15
    findings = scan.inspect("config.py", f'LLM_API_KEY = "{token}"'.encode())
    assert findings and token not in json.dumps(findings)
    assert scan.inspect(".env.local", b"x")
    assert scan.inspect("leak.zip", b"PK\x00opaque")
    assert not scan.inspect(".env.example", b"LLM_API_KEY=\n")
    assert scan.inspect("notes.txt", b"xx" + FAKE_KEY.encode(), [FAKE_KEY.encode()])


def test_staged_scan_uses_index_not_clean_worktree_and_scans_history(tmp_path):
    scan = scanner()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL)
    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@localhost")
    token = "sk-" + "B8c" * 15
    p = tmp_path / "settings.txt"
    p.write_text(token)
    git("add", "settings.txt")
    p.write_text("clean worktree")
    assert any(scan.inspect(name, raw) for name, raw in scan.collect("staged", tmp_path))
    git("commit", "-m", "synthetic scanner fixture")
    git("add", "settings.txt")
    git("commit", "-m", "remove fixture from current tree")
    assert not any(scan.inspect(name, raw) for name, raw in scan.collect("staged", tmp_path))
    assert any(scan.inspect(name, raw) for name, raw in scan.collect("history", tmp_path))
    # An unpacked source tree inside another Git repository must scan its own files.
    nested = tmp_path / "unpacked"
    nested.mkdir()
    (nested / "local.txt").write_text(token)
    own_files = list(scan.collect("worktree", nested))
    assert any(name == "local.txt" and scan.inspect(name, raw) for name, raw in own_files)
    with pytest.raises(ValueError, match="repository root"):
        list(scan.collect("history", nested))
