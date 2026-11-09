"""Request-level tasks orchestration."""
from ..types import Role, Task, TaskType, stable_order


def eligible_roles(engine, request, task):
    return [role for role in Role if engine.tools.eligible(request, task, role)]


def cheapest_operation(engine, request, task, members):
    return min((engine.cost.upper(a, op) for a in members
                for op in engine.tools.operations(request, task, a)
                if op not in {"finish", "request_peer", "refine"}), default=float("inf"))


def execution_reserve(engine, request, task, members):
    return (engine.cost.routing_upper * len(members) + engine.cost.aggregation_upper +
            engine.cost.completion_upper + engine.cheapest_operation(request, task, members))


def legal_tasks(engine, request, state):
    order = stable_order(state.scores, request)
    pairs = {tuple(sorted((a, b))) for a, b in zip(order, order[1:])}
    if engine.config.within_category_pairs:
        categories = {}
        for p in order:
            categories.setdefault(request.candidates[p].category, []).append(p)
        for indices in categories.values():
            pairs.update(tuple(sorted((a, b))) for a, b in zip(indices, indices[1:]))
    tasks = [Task(TaskType.REQUEST, tuple(range(len(order))))]
    tasks.extend(Task(TaskType.COMPARISON, pair) for pair in sorted(pairs))
    tasks.extend(Task(TaskType.COMPENSATION, (p,)) for p, poi in enumerate(request.candidates) if not all(poi.mask))
    return [task for task in tasks if engine.eligible_roles(request, task)]
