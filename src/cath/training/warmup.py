"""Ranking supervision on completed task states."""
import torch
from .objectives import utility


def warmup_loss(engine, request, positive_id, script=None):
    """Input request + separate label + fixed optional actions; output scalar loss.

    A retrieval miss has no ranking supervision. Immediate STOP has zero executor
    gradient. Random sampled discrete trajectories are held fixed for backprop.
    """
    ids = [p.poi_id for p in request.candidates]
    if positive_id not in ids:
        return None, None
    result = engine.run(request, mode="sample", script=script)
    if not result.task_scores:
        return None, result
    positive = ids.index(positive_id)
    return torch.stack([-utility(scores, positive) for scores in result.task_scores]).mean(), result
