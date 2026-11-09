"""Request-level observations orchestration."""
import torch
from ..policy import Observation
from ..types import OPERATIONS


def action(engine, state, first=None, second=None, third=None, numeric=(0, 0, 0, 0)):
    zero = torch.zeros_like(state.context)
    return torch.cat((zero if first is None else first, zero if second is None else second,
                      zero if third is None else third, zero.new_tensor(numeric))).detach()


def observation(engine, request, state, ledger, kind, actions, keys, mask, members=(), role=None):
    selected = engine.policy.selected_features(state, members, None)
    return Observation(engine.policy.global_features(request, state, ledger.remaining, ledger.budget),
                       torch.stack(actions), selected,
                       torch.tensor(mask, device=state.context.device, dtype=torch.bool), kind, tuple(keys))


def task_observation(engine, request, state, ledger, tasks):
    actions = [engine.action(state, numeric=(0, 3, 1, ledger.remaining))]
    keys, mask = ["STOP"], [True]
    for task in tasks:
        roles = engine.eligible_roles(request, task)
        minimum = min(engine.execution_reserve(request, task, [a]) for a in roles)
        # Current task decision + initiator + EXECUTE decision + first execution.
        legal = minimum + 3 * engine.cost.routing_upper <= ledger.remaining
        actions.append(engine.action(state, second=state.candidates[list(task.targets)].mean(0),
                                   numeric=(int(task.kind), 0, len(task.targets), ledger.remaining)))
        keys.append(task.key())
        mask.append(legal)
    return engine.observation(request, state, ledger, 0, actions, keys, mask)


def member_observation(engine, request, state, ledger, task, members, initiator):
    roles = engine.eligible_roles(request, task)
    actions, keys, mask = [], [], []
    if not initiator:
        actions.append(engine.action(state, numeric=(1, 3, int(task.kind), ledger.remaining)))
        keys.append("EXECUTE")
        mask.append(bool(members))
    for role in roles:
        proposed = members + [role]
        # Current routing, later EXECUTE, and enough first-round execution budget.
        reserve = engine.execution_reserve(request, task, proposed) + 2 * engine.cost.routing_upper
        legal = role not in members and len(members) < engine.config.max_members and reserve <= ledger.remaining
        actions.append(engine.action(state, second=state.candidates[list(task.targets)].mean(0),
                                   third=state.memories[int(role)], numeric=(int(role), 1, int(task.kind), ledger.remaining)))
        keys.append(role.name)
        mask.append(legal)
    return engine.observation(request, state, ledger, 1 if initiator else 2, actions, keys, mask, members)


def operation_observation(engine, request, state, ledger, task, role, local, summary, round_budget, round_number, members):
    available = engine.tools.operations(request, task, role)
    actions, keys, mask = [], [], []
    for i, op in enumerate(OPERATIONS[role]):
        legal = op in available
        if op in {"refine", "request_peer"} and round_number == 0:
            legal = False
        if op == "request_peer" and (len(members) < 2 or round_number + 1 == engine.config.max_rounds):
            legal = False
        if op not in {"finish", "request_peer"}:
            upper = engine.cost.upper(role, op)
            legal = legal and upper + engine.cost.aggregation_upper + engine.cost.completion_upper <= round_budget
        actions.append(engine.action(state, second=local, third=state.memories[int(role)],
                                   numeric=(i, 2, int(role), round_budget)))
        keys.append(op)
        mask.append(legal)
    return engine.observation(request, state, ledger, 3, actions, keys, mask)
