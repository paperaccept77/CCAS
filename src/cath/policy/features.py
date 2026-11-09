"""Detached environment features for policy decisions."""
import torch


def global_features(policy, request, state, remaining, maximum):
    # Only completed task summaries enter global state; never private messages.
    summary = torch.stack(state.summaries).mean(0) if state.summaries else torch.zeros_like(state.context)
    numeric = state.scores.new_tensor([remaining / max(maximum, 1e-9),
                                     float(state.scores.detach().max()), float(state.scores.detach().mean())])
    return torch.cat((state.context, state.candidates.mean(0), state.memories.mean(0), summary, numeric)).detach()


def selected_features(policy, state, members, available):
    d = state.context.shape[0]
    # Store role id as a one-hot-like scalar in the last column, not detached embeddings.
    if not members:
        return state.context.new_empty((0, 2 * d + 1))
    return torch.stack([torch.cat((state.memories[int(a)], torch.zeros_like(state.context),
                                  state.context.new_tensor([int(a)]))) for a in members]).detach()
