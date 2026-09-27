import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_paper_rl_settings_match_runtime_defaults():
    import inspect
    from adaptive_feedback_llm.config import load_config
    from adaptive_feedback_llm.rl.ppo import PPOConfig
    from adaptive_feedback_llm.rl.training import train
    profile = load_config(Path(__file__).resolve().parents[1] / "configs/paper.yaml").rl
    ppo = PPOConfig()
    for field in ("learning_rate", "epochs", "batch_size"):
        assert getattr(ppo, field) == getattr(profile, field)
    signature = inspect.signature(train)
    for field in ("hidden", "rollout_steps", "validation_interval"):
        assert signature.parameters[field].default == getattr(profile, field)

from adaptive_feedback_llm.io import load_scenario
from adaptive_feedback_llm.rl.ppo import PPOConfig, PPOTrainer, complete_episode
from adaptive_feedback_llm.rl.training import components, infer_checkpoint, rollout, training_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scene():
    manifest = json.loads((ROOT / "data/benchmark/manifest.json").read_text())
    first = next(row for row in manifest["entries"] if row["difficulty"] == "easy")
    return load_scenario(ROOT / "data/benchmark" / first["path"])


@pytest.mark.parametrize("variant", ["full", "sparse"])
def test_policy_episode_update_and_safe_checkpoint(variant, scene, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    env_type, policy_type = components(variant)
    policy = policy_type(hidden=8)
    transitions, record = complete_episode(policy, env_type(scene))
    assert len(transitions) == len(scene.flows)
    assert sum(t.reward for t in transitions) == pytest.approx(record["normalized_objective"])
    trainer = PPOTrainer(policy, PPOConfig(epochs=1), seed=42)
    trainer.update(transitions)
    assert trainer.optimizer_steps > 0
    result = rollout(scene, policy, env_type)
    path = tmp_path / "selected.pt"
    torch.save({"format": "sidelink-public-rl-v1", "variant": variant,
                "hidden": 8, "state_dict": policy.state_dict()}, path)
    restored = infer_checkpoint(scene, path)
    assert result["schedule"] == restored["schedule"]
    assert result["evaluation"] == restored["evaluation"]


def test_final_test_set_rejected_for_training():
    path = ROOT / "data/benchmark/manifest.json"
    with pytest.raises(ValueError, match="test data is forbidden"):
        training_inputs(path, path)


def test_training_budget_and_exclusive_output(monkeypatch, scene, tmp_path):
    from adaptive_feedback_llm.rl import training
    # Exercise orchestration with one tiny episode; dataset boundary checks are tested separately.
    monkeypatch.setattr(training, "training_inputs", lambda *args, **kwargs: ([scene], [scene]))
    manifest = tmp_path / "inputs.json"
    manifest.write_text("{}")
    out = tmp_path / "run"
    result = training.train(manifest, manifest, out, variant="sparse", seed=1,
                            steps=len(scene.flows), validation_interval=len(scene.flows), hidden=8)
    assert result["steps"] == len(scene.flows)
    assert len(result["validation"]) == 2
    assert infer_checkpoint(scene, out / "selected.pt")["evaluation"]["feasible"]
    with pytest.raises(FileExistsError):
        training.train(manifest, manifest, out, variant="sparse", seed=1,
                       steps=len(scene.flows), validation_interval=len(scene.flows), hidden=8)


def test_source_only_selection_is_explicit(scene, tmp_path):
    import hashlib
    path = tmp_path / "scene.json"
    path.write_text(scene.model_dump_json())
    entry = {"path": "scene.json", "scenario_id": scene.scenario_id,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "sidelink-public-benchmark-v1",
                         "split": "train", "count": 1, "entries": [entry]}))
    assert len(training_inputs(manifest, manifest, source_only=True)[0]) == 1
    with pytest.raises(ValueError):
        training_inputs(manifest, manifest)
