from pathlib import Path
import socket

import pytest

from adaptive_feedback_llm.models import (
    EvaluatorScenario,
    Schedule,
    evaluator_scenario_from_json,
    schedule_from_json,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Keep tests offline even when real credentials exist in the shell."""
    def blocked(*args, **kwargs):
        raise AssertionError("network connections are forbidden in the offline test suite")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def fixed_scenario() -> EvaluatorScenario:
    return evaluator_scenario_from_json(
        (FIXTURES / "basic_scenario.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def valid_schedule() -> Schedule:
    return schedule_from_json(
        (FIXTURES / "basic_valid_schedule.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def all_unserved_schedule() -> Schedule:
    return schedule_from_json(
        (FIXTURES / "basic_all_unserved_schedule.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def catalog_scenario() -> EvaluatorScenario:
    return evaluator_scenario_from_json(
        (FIXTURES / "catalog_scenario.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def catalog_valid_schedule() -> Schedule:
    return schedule_from_json(
        (FIXTURES / "catalog_valid_schedule.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def catalog_all_unserved_schedule() -> Schedule:
    return schedule_from_json(
        (FIXTURES / "catalog_all_unserved_schedule.json").read_text(encoding="utf-8")
    )
