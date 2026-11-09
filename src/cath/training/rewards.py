"""Task-level utility changes and cost reconciliation."""
from .objectives import utility


def assign_rewards(result, request, positive_id, cost_weight):
    """Add utility improvement once to the final sampled decision of each task."""
    ids = [p.poi_id for p in request.candidates]
    if positive_id not in ids:
        raise ValueError("PPO utility is defined only for retrieved training positives")
    positive = ids.index(positive_id)
    previous = float(utility(request.base_scores, positive))
    for edge, scores in zip(result.record["tasks"], result.task_scores):
        current = float(utility(scores, positive))
        end, start = edge["decision_end"], edge["decision_start"]
        if end > start:
            result.decisions[end - 1].reward += current - previous
        previous = current
    expected = float(utility(result.scores, positive) - utility(request.base_scores, positive))
    expected -= cost_weight * result.record["cost"]
    actual = sum(d.reward for d in result.decisions)
    if result.decisions and abs(actual - expected) > 1e-5:
        raise RuntimeError(f"reward ledger does not telescope: {actual} vs {expected}")
