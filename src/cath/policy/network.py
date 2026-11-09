"""Masked hierarchical action heads and shared critic."""
import torch
from torch import nn
from torch.distributions import Categorical
from ..neural import SetEncoder, mlp
from .records import Observation, Decision
from . import features


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.dim
        self.roles = nn.Embedding(6, d)
        self.task_embedding = nn.Embedding(3, d)
        self.action_embedding = nn.Embedding(6, d)
        self.special = nn.Embedding(2, d)  # STOP, EXECUTE
        self.global_encoder = mlp(4 * d + 3, d)
        self.member_input = mlp(2 * d + 1, d)
        self.member_set = SetEncoder(d, config.heads, config.set_layers)
        self.action_input = mlp(3 * d + 4, d)
        self.scorers = nn.ModuleList([mlp(3 * d, 1, d) for _ in range(4)])
        self.critic = mlp(2 * d, 1, d)

    def evaluate(self, observation):
        o = observation
        global_state = self.global_encoder(o.global_features)
        selected = self.member_set(self.member_input(o.selected))
        actions = self.action_input(o.action_features)
        actor_global = torch.zeros_like(global_state) if o.kind == 3 else global_state
        context = torch.cat((actor_global, selected)).expand(actions.shape[0], -1)
        logits = self.scorers[o.kind](torch.cat((context, actions), -1)).squeeze(-1)
        if not o.mask.any():
            raise ValueError("cannot sample an all-masked action space")
        distribution = Categorical(logits=logits.masked_fill(~o.mask, -torch.inf))
        return distribution, self.critic(torch.cat((global_state, selected))).squeeze(-1)

    def choose(self, observation, mode="greedy", forced_key=None):
        # Environment features are detached; policy embeddings remain trainable.
        o = observation.detached()
        distribution, value = self(o)
        if forced_key is not None:
            action = o.action_keys.index(forced_key)
            if not o.mask[action]:
                raise ValueError(f"illegal replay action {forced_key}")
        else:
            action = int(distribution.sample() if mode == "sample" else distribution.logits.argmax())
        record = Decision(o, action, float(distribution.log_prob(torch.tensor(action, device=o.mask.device)).detach()),
                          float(value.detach()))
        return action, record

    def global_features(self, request, state, remaining, maximum):
        return features.global_features(self, request, state, remaining, maximum)

    def selected_features(self, state, members, available):
        return features.selected_features(self, state, members, available)

    def decorate(self, observation):
        # Role/task/operation embeddings are applied during every PPO re-evaluation.
        o = observation
        a = o.action_features.clone()
        d = self.roles.embedding_dim
        ids = a[:, -4].long()
        tags = a[:, -3].long()  # 0=task, 1=role, 2=operation, 3=special
        embedding = torch.zeros_like(a[:, :d])
        for tag, table in ((0, self.task_embedding), (1, self.roles), (2, self.action_embedding), (3, self.special)):
            selected = tags == tag
            if selected.any():
                embedding[selected] = table(ids[selected])
        a[:, :d] = a[:, :d] + embedding
        if o.kind in {1, 2}:
            a[:, d:2*d] = a[:, d:2*d] + self.task_embedding(a[:, -2].long())
        if o.kind == 3:
            a[:, d:2*d] = a[:, d:2*d] + self.roles(a[:, -2].long())
        members = o.selected.clone()
        if members.shape[0]:
            members[:, d:2*d] = self.roles(members[:, -1].long())
            members[:, -1] = 1.0  # selected members are eligible for this task
        return Observation(o.global_features, a, members, o.mask, o.kind, o.action_keys)

    def forward(self, observation):
        return self.evaluate(self.decorate(observation))
