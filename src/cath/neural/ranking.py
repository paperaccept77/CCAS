"""Recurrent task evidence and shared ranking corrections."""
import torch
from torch import nn
from ..types import EpisodeState, Role, TaskType
from .layers import mlp
from .set_encoder import SetEncoder


class RankingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.dim
        self.context_input = mlp(d, d)
        self.candidate_input = mlp(d + 2, d)
        self.message_meta = nn.Embedding(6 * 6, d)
        self.message_set = SetEncoder(d, config.heads, config.set_layers)
        self.summary_set = SetEncoder(d, config.heads, config.set_layers)
        self.edge_gru = nn.GRUCell(d, d)
        self.local_gru = nn.GRUCell(3 * d, d)
        self.local_init = mlp(3 * d, d)
        self.peer_request = nn.Parameter(torch.zeros(d))
        self.memory_gru = nn.GRUCell(d, d)
        self.candidate_gru = nn.GRUCell(3 * d, d)
        self.context_gru = nn.GRUCell(d, d)
        self.rank_head = mlp(3 * d + 2, 1, d)

    def correction(self, context, candidates, initial, masks):
        return self.rank_head(torch.cat((context.expand_as(candidates), candidates, initial, masks), -1)).squeeze(-1)

    def initialize(self, request):
        c = self.context_input(request.request_features)
        v = self.candidate_input(torch.cat((request.candidate_features, request.masks), -1))
        reference = self.correction(c, v, v, request.masks)
        state = EpisodeState(c, v, c, v, reference, v.new_zeros((6, v.shape[-1])), request.base_scores)
        state.scores = self.score(request, state)
        return state

    def score(self, request, state):
        # Deterministic identical terms cancel at t=0, including their gradients.
        return request.base_scores + (self.correction(state.context, state.candidates,
                                                      state.initial_candidates, request.masks) - state.initial_reference)

    def aggregate(self, latest, candidate_results, member_summaries):
        updated = dict(candidate_results)
        for p, previous in candidate_results.items():
            messages = [message + self.message_meta.weight[int(role) * 6 + operation]
                        for (role, target), (message, operation) in latest.items() if target == p]
            if messages:
                updated[p] = self.edge_gru(self.message_set(torch.stack(messages)), previous)
        values = list(updated.values()) + list(member_summaries.values())
        summary = self.summary_set(torch.stack(values))
        return updated, summary

    def complete(self, request, state, task, members, results, summary):
        # Functional stacks preserve warm-up gradients and isolate untouched states.
        state.memories = torch.stack([
            self.memory_gru(summary, h) if Role(a) in members else h
            for a, h in enumerate(state.memories)])
        state.candidates = torch.stack([
            self.candidate_gru(torch.cat((results[p], summary, state.context)), v)
            if p in results else v for p, v in enumerate(state.candidates)])
        if task.kind == TaskType.REQUEST:
            state.context = self.context_gru(summary, state.context)
        state.summaries.append(summary)
        state.scores = self.score(request, state)
