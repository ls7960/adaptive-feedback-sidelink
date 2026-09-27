"""Permutation-equivariant actor-critic with scheduling structural features."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .full_environment import ACTION_FEATURES, EDGE_FEATURES, FLOW_FEATURES, GLOBAL_FEATURES


POLICY_ID = "full-actor-critic"


def _mlp(inputs: int, hidden: int, outputs: int | None = None) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, outputs or hidden)
    )


class _MessageRound(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_update = _mlp(2 * hidden, hidden)
        self.flow_update = _mlp(2 * hidden, hidden)
        self.action_update = _mlp(3 * hidden, hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.flow_norm = nn.LayerNorm(hidden)
        self.action_norm = nn.LayerNorm(hidden)

    def forward(self, edge, flow, action, incidence, membership, legal):
        active_incidence = incidence * legal[:, None]
        edge_message = active_incidence.T @ action
        edge_message = edge_message / active_incidence.sum(0)[:, None].clamp_min(1)
        flow_message = torch.zeros_like(flow)
        flow_message.index_add_(0, membership, action * legal[:, None])
        flow_count = torch.zeros(flow.shape[0], dtype=flow.dtype, device=flow.device)
        flow_count.index_add_(0, membership, legal)
        flow_message = flow_message / flow_count[:, None].clamp_min(1)
        edge = self.edge_norm(edge + self.edge_update(torch.cat((edge, edge_message), -1)))
        flow = self.flow_norm(flow + self.flow_update(torch.cat((flow, flow_message), -1)))
        path_message = incidence @ edge
        path_message = path_message / incidence.sum(1)[:, None].clamp_min(1)
        action = self.action_norm(
            action
            + self.action_update(torch.cat((action, path_message, flow[membership]), -1))
        )
        return edge, flow, action


class FullActorCritic(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        if hidden < 4:
            raise ValueError("hidden must be at least four")
        self.hidden = hidden
        self.edge_encoder = _mlp(EDGE_FEATURES, hidden)
        self.flow_encoder = _mlp(FLOW_FEATURES, hidden)
        self.action_encoder = _mlp(ACTION_FEATURES, hidden)
        self.message_rounds = nn.ModuleList([_MessageRound(hidden) for _ in range(2)])
        self.actor = _mlp(5 * hidden + GLOBAL_FEATURES, hidden, 1)
        self.critic = _mlp(3 * hidden + GLOBAL_FEATURES, hidden, 1)

    def distribution_value(self, observation: Mapping[str, np.ndarray | torch.Tensor]):
        device = next(self.parameters()).device
        obs = {
            key: (
                torch.tensor(value, device=device)
                if isinstance(value, np.ndarray) and not value.flags.writeable
                else torch.as_tensor(value, device=device)
            )
            for key, value in observation.items()
        }
        mask = obs["mask"].bool()
        if mask.ndim != 1 or not bool(mask.any()):
            raise ValueError("policy requires a one-dimensional nonempty action mask")
        edges = obs["edges"].float()
        flows = obs["flows"].float()
        actions = obs["actions"].float()
        incidence = obs["action_edges"].float()
        membership = obs["action_flows"].long()
        if (
            edges.ndim != 2
            or edges.shape[1] != EDGE_FEATURES
            or flows.ndim != 2
            or flows.shape[1] != FLOW_FEATURES
            or actions.shape != (mask.shape[0], ACTION_FEATURES)
            or incidence.shape != (mask.shape[0], edges.shape[0])
            or membership.shape != mask.shape
            or obs["global"].shape != (GLOBAL_FEATURES,)
        ):
            raise ValueError("observation shape does not match scheduling policy contract")
        if edges.shape[0] == 0 or flows.shape[0] == 0:
            raise ValueError("edge and flow sets must be nonempty")
        if int(membership.min()) < 0 or int(membership.max()) >= flows.shape[0]:
            raise ValueError("action flow membership is outside the flow set")
        for values in (edges, flows, actions, incidence, obs["global"]):
            if not bool(torch.isfinite(values).all()):
                raise ValueError("nonfinite scheduling observation")
        edge = self.edge_encoder(edges)
        flow = self.flow_encoder(flows)
        action = self.action_encoder(actions)
        legal = mask.to(actions.dtype)
        for message_round in self.message_rounds:
            edge, flow, action = message_round(
                edge, flow, action, incidence, membership, legal
            )
        pooled = torch.cat(
            (
                edge.mean(0),
                flow.mean(0),
                (action * legal[:, None]).sum(0) / legal.sum(),
                obs["global"].float(),
            )
        )
        context = pooled.expand(action.shape[0], -1)
        logits = self.actor(torch.cat((action, flow[membership], context), -1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        value = self.critic(pooled).squeeze(-1)
        return Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, observation, *, deterministic: bool = False):
        distribution, value = self.distribution_value(observation)
        action = torch.argmax(distribution.logits) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())
