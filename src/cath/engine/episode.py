"""Request-level episode orchestration."""
from ..cost import Ledger
from ..types import Role, stable_order
from .records import EpisodeResult


def run(engine, request, mode="greedy", script=None):
    """Input: prediction-time Request. Output: scores [K], trace, PPO transitions.

    script is an optional list of action keys for fixed legal warm-up replay.
    It contains STOP/task keys, role names, EXECUTE, then operation names.
    """
    cfg = engine.config
    ledger = Ledger(cfg.budget, engine.cost, request.base_scores.device)
    if cfg.budget < engine.cost.completion_upper:
        order = stable_order(request.base_scores, request)
        record = {"request_id": request.request_id, "candidate_ids": [p.poi_id for p in request.candidates],
                  "base_scores": request.base_scores.detach().cpu().tolist(),
                  "scores": request.base_scores.detach().cpu().tolist(),
                  "ranking": [request.candidates[p].poi_id for p in order], "termination": "initialization_budget",
                  "cost": 0.0, "seconds": 0.0, "cost_calibrated": engine.cost.calibrated,
                  "cost_ledger": [], "incidence": [[] for _ in Role], "roles": [a.name for a in Role], "tasks": []}
        return EpisodeResult(request.base_scores, [], [], record)
    # Initial encoding/reference is measured and charged, also for immediate STOP.
    state, _, _ = ledger.measure("initialize", engine.cost.completion_upper,
                                 lambda: engine.model.initialize(request))
    records, task_scores = [], []
    replay = script if callable(script) else iter(script) if script is not None else None

    def choose(builder):
        def decide():
            observation = builder()
            forced = replay(observation) if callable(replay) else next(replay) if replay is not None else None
            index, transition = engine.policy.choose(observation, mode, forced)
            return observation.action_keys[index], transition
        (key, transition), _, _ = ledger.measure("routing", engine.cost.routing_upper, decide)
        if replay is None or callable(replay):
            records.append(transition)
        return key

    termination = "task_limit"
    for task_step in range(cfg.max_tasks):
        start_record = len(records)
        start_cost = ledger.spent
        # Keep reward-only score references outside all policy/tool observations.
        if ledger.remaining < engine.cost.routing_upper:
            termination = "budget"
            break
        tasks = engine.legal_tasks(request, state)
        affordable = [task for task in tasks if min(engine.execution_reserve(request, task, [a])
                      for a in engine.eligible_roles(request, task)) + 3 * engine.cost.routing_upper <= ledger.remaining]
        if not affordable:
            termination = "no_affordable_task"
            break
        selected = choose(lambda: engine.task_observation(request, state, ledger, tasks))
        if selected == "STOP":
            termination = "sampled_stop"
            if records:
                records[-1].reward = -cfg.cost_weight * (ledger.spent - start_cost)
            break
        task = next(task for task in tasks if task.key() == selected)
        members = []
        first = choose(lambda: engine.member_observation(request, state, ledger, task, members, True))
        members.append(Role[first])
        while True:
            member = choose(lambda: engine.member_observation(request, state, ledger, task, members, False))
            if member == "EXECUTE":
                break
            members.append(Role[member])
            if len(members) == cfg.max_members:
                break  # forced execution is not a sampled action
        task_record = engine.execute_task(request, state, ledger, task, members, choose)
        task_record["step"] = task_step
        task_record["cost"] = ledger.spent - start_cost
        task_record["decision_start"] = start_record
        task_record["decision_end"] = len(records)
        state.edges.append(task_record)
        task_scores.append(state.scores)
        # Label-dependent utility deltas are assigned by training/rewards.py after execution.
        if len(records) > start_record:
            records[-1].reward = -cfg.cost_weight * (ledger.spent - start_cost)
    if replay is not None and not callable(replay) and next(replay, None) is not None:
        raise ValueError("warm-up script has unused actions")
    if records:
        # Attach initialization and forced-termination costs once to the final transition.
        charged = sum(edge["cost"] for edge in state.edges)
        if termination == "sampled_stop":
            charged += ledger.entries[-1]["cost"]
        records[-1].reward -= cfg.cost_weight * (ledger.spent - charged)
        records[-1].done = True
    order = stable_order(state.scores, request)
    incidence = [[int(role.name in edge["members"]) for edge in state.edges] for role in Role]
    record = {"request_id": request.request_id, "candidate_ids": [p.poi_id for p in request.candidates],
              "base_scores": request.base_scores.detach().cpu().tolist(),
              "scores": state.scores.detach().cpu().tolist(),
              "ranking": [request.candidates[p].poi_id for p in order], "termination": termination,
              "cost": ledger.spent, "seconds": sum(e["seconds"] for e in ledger.entries),
              "cost_calibrated": engine.cost.calibrated, "cost_ledger": ledger.entries,
              "incidence": incidence, "roles": [a.name for a in Role], "tasks": state.edges,
              "initial_scores": (request.base_scores).detach().cpu().tolist()}
    return EpisodeResult(state.scores, task_scores, records, record)
