"""Portable PPO training/inference, using the frozen full and sparse model kernels.

This replaces dated campaign orchestration, not the archived scientific runs.
Training inputs are explicit and checkpoint selection reads validation only.
"""

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from adaptive_feedback_llm.evaluator import evaluate_schedule
from adaptive_feedback_llm.io import load_dataset, write_json
from adaptive_feedback_llm.rl.ppo import PPOConfig, PPOTrainer, complete_episode


def components(variant):
    if variant == "full":
        from .full_environment import FullSchedulingEnv
        from .full_policy import FullActorCritic
        return FullSchedulingEnv, FullActorCritic
    if variant == "sparse":
        from .sparse_environment import SparseSchedulingEnv
        from .sparse_policy import SparseActorCritic
        return SparseSchedulingEnv, SparseActorCritic
    raise ValueError("variant must be full or sparse")


def rollout(scenario, policy, environment_type):
    env = environment_type(scenario)
    observation = env.reset()
    policy.eval()
    while not env.terminated:
        action, _, _ = policy.act(observation, deterministic=True)
        observation, _, _, _ = env.step(action)
    result = evaluate_schedule(scenario, env.schedule)
    if not result.feasible or result.objective != env.current_j:
        raise RuntimeError("RL terminal schedule failed independent evaluation")
    return {"schedule": env.schedule.model_dump(mode="json"),
            "evaluation": result.model_dump(mode="json"),
            "normalized_objective": result.objective / env.normalizer,
            "work": dict(env.work)}


def _identity(scene):
    value = scene.model_dump(mode="json")
    value.pop("scenario_id")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def training_inputs(train_manifest, validation_manifest, *, source_only=False):
    train_meta, training = load_dataset(train_manifest)
    valid_meta, validation = load_dataset(validation_manifest)
    if source_only:
        if (train_meta["split"] != "train" or valid_meta["split"] != "train"
                or len(training) != 1 or len(validation) != 1
                or training[0] != validation[0]):
            raise ValueError("source-only selection requires the same single train instance")
        return training, validation
    if train_meta["split"] != "train" or valid_meta["split"] != "validation":
        raise ValueError("training requires train/validation splits; benchmark test data is forbidden")
    if ({s.scenario_id for s in training} & {s.scenario_id for s in validation}
            or {_identity(s) for s in training} & {_identity(s) for s in validation}):
        raise ValueError("training and validation overlap")
    return training, validation


def train(train_manifest, validation_manifest, output, *, variant, seed, steps,
          validation_interval=5000, hidden=128, rollout_steps=512, source_only=False):
    if source_only and variant != "full":
        raise ValueError("the source-only baseline uses the full RL architecture")
    training, validation = training_inputs(train_manifest, validation_manifest,
                                           source_only=source_only)
    minimum_episode = min(len(s.flows) for s in training)
    if steps < minimum_episode or validation_interval < minimum_episode or rollout_steps < 1:
        raise ValueError("step budget and validation interval must fit a complete episode")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    env_type, policy_type = components(variant)
    policy = policy_type(hidden=hidden)
    trainer = PPOTrainer(policy, PPOConfig(), seed=seed)
    rng = np.random.default_rng(seed)
    used, best, best_state, rows = 0, -math.inf, None, []

    def validate():
        nonlocal best, best_state
        score = float(np.mean([rollout(s, policy, env_type)["normalized_objective"]
                               for s in validation]))
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        rows.append({"steps": used, "validation_mean_normalized_objective": score})

    validate()
    boundary = min(validation_interval, steps)
    while steps - used >= minimum_episode:
        transitions = []
        while len(transitions) < rollout_steps:
            eligible = [s for s in training if len(s.flows) <= boundary - used]
            if not eligible:
                break
            scene = eligible[int(rng.integers(len(eligible)))]
            episode, _ = complete_episode(policy, env_type(scene))
            transitions.extend(episode)
            used += len(episode)
        if transitions:
            trainer.update(transitions)
        if boundary - used < minimum_episode:
            validate()
            if boundary == steps:
                break
            boundary = min(boundary + validation_interval, steps)
    assert best_state is not None
    checkpoint = {"format": "sidelink-public-rl-v1", "variant": variant, "hidden": hidden,
                  "seed": seed, "environment_steps": used, "state_dict": best_state}
    with (output / "selected.pt").open("xb") as stream:
        torch.save(checkpoint, stream)
    summary = {"variant": variant, "seed": seed, "steps": used, "step_ceiling": steps,
               "selection_data": "single training source" if source_only else "held-out validation",
               "validation": rows, "best_validation_score": best,
               "selection": "strict improvement; earliest exact tie",
               "train_manifest_sha256": hashlib.sha256(Path(train_manifest).read_bytes()).hexdigest(),
               "validation_manifest_sha256": hashlib.sha256(Path(validation_manifest).read_bytes()).hexdigest(),
               "optimizer_steps": trainer.optimizer_steps, "new_run": True}
    write_json(output / "summary.json", summary)
    return summary


def infer_checkpoint(scenario, checkpoint):
    # Reject arbitrary pickle objects. Historical private checkpoint formats are not accepted.
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if value.get("format") != "sidelink-public-rl-v1":
        raise ValueError("unsupported public checkpoint format")
    env_type, policy_type = components(value["variant"])
    policy = policy_type(hidden=value["hidden"])
    policy.load_state_dict(value["state_dict"], strict=True)
    return rollout(scenario, policy, env_type)
