"""Complete-episode PPO with gamma=lambda=1 and immutable rollout masks.

The categorical probability ratio is for the complete joint action. There are no
separate flow/path/resource ratios and no final reward added a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
import torch

from .full_policy import FullActorCritic as MaskedActorCritic


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    epochs: int = 4
    batch_size: int = 32
    clip: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5

    def __post_init__(self):
        for name in ("learning_rate", "clip", "value_clip", "value_coefficient",
                     "entropy_coefficient", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.learning_rate <= 0 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("learning rate, epochs and batch size must be positive")
        if not 0 < self.clip < 1 or self.value_clip <= 0 or self.max_grad_norm <= 0:
            raise ValueError("invalid clipping or gradient limit")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("loss coefficients must be nonnegative")


@dataclass(frozen=True)
class Transition:
    observation: dict[str, np.ndarray]
    action: int
    reward: float
    old_log_probability: float
    old_value: float
    return_to_go: float


def freeze_observation(observation):
    """Detach copies so env reset/step cannot mutate an old policy's support."""
    copied = {key: np.array(value, copy=True) for key, value in observation.items()}
    for value in copied.values():
        value.setflags(write=False)
    return copied


def complete_episode(policy, env):
    """Collect one F-step episode; returns complete transitions and audit record."""
    observation = env.reset()
    pending = []
    actions, rewards = [], []
    while not env.terminated:
        saved = freeze_observation(observation)
        action, old_log_probability, old_value = policy.act(observation)
        if not saved["mask"][action]:
            raise AssertionError("sampled action was invalid under saved mask")
        observation, reward, terminated, _ = env.step(action)
        pending.append((saved, action, float(reward), old_log_probability, old_value))
        actions.append(action)
        rewards.append(float(reward))
        if terminated != env.terminated:
            raise AssertionError("environment termination disagreement")
    if len(pending) != len(env.scenario.flows):
        raise AssertionError("episode must process exactly every flow")
    transitions, running_return = [], 0.0
    for saved, action, reward, old_lp, old_v in reversed(pending):
        running_return += reward
        transitions.append(Transition(saved, action, reward, old_lp, old_v, running_return))
    transitions.reverse()
    normalized = env.current_j / env.normalizer
    if not np.isclose(sum(rewards), normalized, rtol=1e-10, atol=1e-12):
        raise AssertionError("reward sum differs from normalized terminal objective")
    return transitions, {"scenario_id": env.scenario.scenario_id, "actions": actions,
                         "rewards": rewards, "objective": env.current_j,
                         "normalized_objective": normalized, "work": dict(env.work)}


def clipped_losses(new_log_probs, old_log_probs, advantages, values, old_values,
                   returns, entropies, config: PPOConfig):
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - config.clip, 1 + config.clip) * advantages
    actor_loss = -torch.minimum(unclipped, clipped).mean()
    clipped_values = old_values + (values - old_values).clamp(-config.value_clip, config.value_clip)
    value_loss = 0.5 * torch.maximum((values - returns).square(),
                                   (clipped_values - returns).square()).mean()
    entropy = entropies.mean()
    loss = actor_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
    with torch.no_grad():
        log_ratio = new_log_probs - old_log_probs
        approximate_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > config.clip).float().mean()
    return loss, {"actor_loss": actor_loss, "value_loss": value_loss, "entropy": entropy,
                  "approximate_kl": approximate_kl, "clip_fraction": clip_fraction}


class PPOTrainer:
    def __init__(self, policy: MaskedActorCritic, config: PPOConfig, *, seed: int):
        self.policy, self.config = policy, config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(seed)
        self.optimizer_steps = 0
        self.sample_evaluations = 0

    def update(self, transitions: list[Transition]):
        if not transitions:
            raise ValueError("PPO requires nonempty complete-episode rollout data")
        device = next(self.policy.parameters()).device
        returns = torch.tensor([t.return_to_go for t in transitions], dtype=torch.float32, device=device)
        old_values = torch.tensor([t.old_value for t in transitions], dtype=torch.float32, device=device)
        old_lps = torch.tensor([t.old_log_probability for t in transitions], dtype=torch.float32, device=device)
        advantages = returns - old_values
        # Population standard deviation avoids a NaN for a single sample.
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        totals, count = {}, 0
        self.policy.train()
        for _ in range(self.config.epochs):
            order = self.rng.permutation(len(transitions))
            for start in range(0, len(order), self.config.batch_size):
                indices = order[start:start + self.config.batch_size]
                lps, vals, entropies = [], [], []
                for index in indices:
                    transition = transitions[int(index)]
                    if not bool(transition.observation["mask"][transition.action]):
                        raise ValueError("rollout action is not legal under its stored mask")
                    distribution, value = self.policy.distribution_value(transition.observation)
                    self.sample_evaluations += 1
                    action = torch.tensor(transition.action, dtype=torch.long, device=device)
                    lps.append(distribution.log_prob(action))
                    vals.append(value)
                    entropies.append(distribution.entropy())
                selected = torch.as_tensor(indices, dtype=torch.long, device=device)
                loss, metrics = clipped_losses(torch.stack(lps), old_lps[selected], advantages[selected],
                                               torch.stack(vals), old_values[selected], returns[selected],
                                               torch.stack(entropies), self.config)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm,
                                                               error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer_steps += 1
                metrics.update(loss=loss, gradient_norm=gradient_norm)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(indices)
                count += len(indices)
        return {**{key: value / count for key, value in totals.items()},
                "samples": len(transitions), "sample_evaluations": count,
                "optimizer_steps_total": self.optimizer_steps}
