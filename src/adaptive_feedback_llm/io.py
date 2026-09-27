"""Portable JSON inputs and exclusive output publication."""

import hashlib
import json
from pathlib import Path

from adaptive_feedback_llm.models import evaluator_scenario_from_json, schedule_from_json


def load_scenario(path):
    return evaluator_scenario_from_json(Path(path).read_text(encoding="utf-8-sig"))


def load_schedule(path):
    return schedule_from_json(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content + "\n")


def load_dataset(manifest):
    manifest = Path(manifest).resolve(strict=True)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if value.get("schema") != "sidelink-public-benchmark-v1":
        raise ValueError("unsupported dataset schema")
    entries = value["entries"]
    if not entries or len(entries) != value["count"]:
        raise ValueError("dataset count mismatch")
    seen = set()
    scenarios = []
    for row in entries:
        path = (manifest.parent / row["path"]).resolve(strict=True)
        if not path.is_relative_to(manifest.parent) or not path.is_file():
            raise ValueError("dataset path escapes its root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("scenario hash mismatch")
        scene = load_scenario(path)
        if scene.scenario_id in seen or scene.scenario_id != row["scenario_id"]:
            raise ValueError("duplicate or inconsistent scenario identity")
        seen.add(scene.scenario_id)
        scenarios.append(scene)
    return value, scenarios
