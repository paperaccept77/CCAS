"""Request-macro coverage, covered-only metrics, and all-request metrics."""
import math


def evaluate(predictions, labels, cutoff=20):
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    totals = {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0}
    covered, count, seen = 0, 0, set()
    for prediction in predictions:
        key = str(prediction["request_id"])
        if key in seen:
            raise ValueError(f"duplicate prediction {key}")
        seen.add(key)
        if key not in labels:
            raise ValueError(f"missing label for {key}")
        pool, ranking = prediction["candidate_ids"], prediction["ranking"]
        if len(pool) != len(set(pool)) or len(ranking) != len(pool) or set(ranking) != set(pool):
            raise ValueError("ranking must be a permutation of the original pool")
        count += 1
        positive = labels[key]
        if positive not in pool:
            continue
        covered += 1
        rank = ranking.index(positive) + 1
        totals["recall"] += float(rank <= cutoff)
        totals["ndcg"] += 1 / math.log2(rank + 1) if rank <= cutoff else 0.0
        totals["mrr"] += 1 / rank
    if not count:
        raise ValueError("no predictions")
    if seen != set(labels):
        raise ValueError("prediction and label request IDs must match exactly")
    return {"requests": count, "covered": covered, "coverage": covered/count, "cutoff": cutoff,
            "end_to_end": {k: v/count for k, v in totals.items()},
            "conditional": {k: v/covered if covered else None for k, v in totals.items()}}
