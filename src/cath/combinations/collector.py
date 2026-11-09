"""Matched task and initiator group execution."""
import itertools
import torch
from ..policy import Observation
from ..training.records import CombinationRecord
from ..training.objectives import utility
from ..types import Role


def collect(engine, request, positive_id, task_key, initiator, repetitions=3):
    """Enumerate at most 32 groups containing the same initiator at initial state.

    Executes every recorded group; does not score unexecuted groups. A deterministic
    two-round rule prefers each role's first legal evidence operation, then refine.
    Other member orders are seed-shuffled for invitation imitation.
    """
    initiator = Role[initiator.upper()]
    state = engine.model.initialize(request)
    tasks = {task.key(): task for task in engine.legal_tasks(request, state)}
    if task_key not in tasks:
        raise ValueError(f"illegal initial task {task_key}; available: {list(tasks)}")
    task = tasks[task_key]
    roles = engine.eligible_roles(request, task)
    if initiator not in roles:
        raise ValueError("initiator is unavailable")
    others = [a for a in roles if a != initiator]
    ids = [p.poi_id for p in request.candidates]
    if positive_id not in ids:
        return
    positive = ids.index(positive_id)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    for size in range(min(len(others), engine.config.max_members - 1) + 1):
        for rest in itertools.combinations(others, size):
            group = (initiator, *rest)
            for repetition in range(repetitions):
                order = list(rest)
                if order:
                    permutation = torch.randperm(len(order)).tolist()
                    order = [order[i] for i in permutation]
                invited, counts, completed = [], {}, [False]

                def rule(o):
                    if o.kind == 0:
                        return "STOP" if completed[0] else task_key
                    if o.kind == 1:
                        invited.append(initiator)
                        return initiator.name
                    if o.kind == 2:
                        remaining = [a for a in order if a not in invited]
                        if not remaining:
                            return "EXECUTE"
                        invited.append(remaining[0])
                        return remaining[0].name
                    role = Role(int(o.action_features[0, -2]))
                    count = counts.get(role, 0)
                    counts[role] = count + 1
                    completed[0] = True
                    legal = [key for key, valid in zip(o.action_keys, o.mask.tolist()) if valid]
                    if count >= 2:
                        return "finish"
                    if count == 1 and "refine" in legal:
                        return "refine"
                    return next((key for key in legal if key not in {"finish", "request_peer", "refine"}), "finish")

                try:
                    result = engine.run(request, script=rule)
                except ValueError as error:
                    if "illegal replay action" in str(error):
                        continue  # group not affordable under the exact legal mask
                    raise
                if not result.record["tasks"] or set(result.record["tasks"][0]["members"]) != {a.name for a in group}:
                    continue
                original = result.decisions[0].observation
                selected_task = original.action_keys.index(task_key)
                first = Observation(original.global_features,
                                    original.action_features[selected_task:selected_task+1],
                                    original.selected, original.mask[selected_task:selected_task+1],
                                    original.kind, (task_key,))
                gain = float(utility(result.task_scores[0], positive) - utility(request.base_scores, positive))
                gain -= engine.config.cost_weight * result.record["tasks"][0]["cost"]
                invitations = [d for d in result.decisions if d.observation.kind == 2]
                key = f"{request.request_id}|initial|{task_key}|{initiator.name}|two_round_rule"
                yield CombinationRecord(key, first, tuple(a.name for a in group), gain, invitations)
