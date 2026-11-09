"""Request-level rounds orchestration."""
import torch
from ..types import OPERATIONS


def execute_task(engine, request, state, ledger, task, members, choose):
    cfg = engine.config
    zero = torch.zeros_like(state.context)
    results = {p: zero for p in task.targets}
    latest, summaries, local, finished = {}, {}, {}, set()
    summary = zero
    for a in members:
        local[a] = engine.model.local_init(torch.cat((state.memories[int(a)], state.context,
                                                   state.candidates[list(task.targets)].mean(0))))
    rounds, pending_peers = [], set()
    for round_number in range(cfg.max_rounds):
        active = [a for a in members if a not in finished]
        if not active:
            break
        # Reserve enough routing for all active members, one aggregation and completion.
        overhead = len(active) * engine.cost.routing_upper + engine.cost.aggregation_upper + engine.cost.completion_upper
        if ledger.remaining < overhead:
            break
        round_budget = ledger.remaining - len(active) * engine.cost.routing_upper
        snapshot_summary, snapshot_local = summary, dict(local)
        selected, requested = [], set()
        decisions = []
        for a in active:
            key = choose(lambda a=a: engine.operation_observation(request, state, ledger, task, a,
                         snapshot_local[a], snapshot_summary, round_budget, round_number, members))
            decisions.append({"role": a.name, "operation": key, "peer_requested": a in pending_peers})
            if key == "finish":
                finished.add(a)
            elif key == "request_peer":
                # Deterministic active peer, fixed task targets, delivery only next round.
                peers = [p for p in active if p != a]
                if peers:
                    requested.add(min(peers, key=int))
            else:
                selected.append((engine.cost.upper(a, key), a, key))
        pending_peers = requested
        admitted = []
        available = ledger.remaining - engine.cost.aggregation_upper - engine.cost.completion_upper
        for upper, a, key in sorted(selected, key=lambda x: (x[0], int(x[1]))):
            if upper <= available:
                admitted.append((upper, a, key))
                available -= upper
        if not admitted:
            if not requested:
                rounds.append({"round": round_number, "decisions": decisions, "outputs": [], "status": "no_admitted_work"})
                break
        # Joint reservation precedes any execution. The round snapshot stays immutable.
        joint = sum(upper for upper, _, _ in admitted)
        ledger.reserve(joint)
        outputs = []
        for upper, a, key in admitted:
            def execute(a=a, key=key):
                admitted_shared = snapshot_summary if cfg.exchange else zero
                evidence = engine.tools.bind(request, task, a, key, admitted_shared)
                if key == "refine" and not cfg.exchange:
                    admitted_shared = summaries.get(a, zero)
                return engine.executor(a, key, task, request, evidence, state.memories[int(a)],
                                     snapshot_local[a], admitted_shared)
            output, seconds, cost = ledger.measure_reserved(f"{a.name.lower()}.{key}", upper, execute)
            output.seconds, output.normalized_cost = seconds, cost
            op_index = OPERATIONS[a].index(key)
            for p, message in output.messages.items():
                if p not in results:
                    raise ValueError("executor covered a non-target candidate")
                latest[a, p] = message, op_index  # replace, never append duplicate contribution
            if output.messages:
                summaries[a] = output.summary
            outputs.append({"role": a.name, "operation": key, "coverage": sorted(output.messages),
                            "status": output.status, "objects": output.objects, "seconds": seconds,
                            "cost": cost, "input_tokens": output.input_tokens, "output_tokens": output.output_tokens})
        def aggregate():
            updated, shared = engine.model.aggregate(latest, results, summaries)
            updated_local = dict(local)
            for a in active:
                read_summary = shared if cfg.exchange else summaries.get(a, zero)
                own = [message for (author, _), (message, _) in latest.items() if author == a]
                read_candidates = torch.stack(list(updated.values())).mean(0) if cfg.exchange else torch.stack(own).mean(0) if own else zero
                # Peer requests are visible next round as a deterministic local-state signal.
                peer_signal = engine.model.peer_request if a in pending_peers else zero
                read = torch.cat((state.memories[int(a)], read_summary + peer_signal, read_candidates))
                updated_local[a] = engine.model.local_gru(read, snapshot_local[a])
            return updated, shared, updated_local
        (results, summary, local), _, _ = ledger.measure("aggregation", engine.cost.aggregation_upper, aggregate)
        rounds.append({"round": round_number, "decisions": decisions, "outputs": outputs, "status": "executed"})
    ledger.measure("complete_and_rank", engine.cost.completion_upper,
                   lambda: engine.model.complete(request, state, task, members, results, summary))
    return {"task": task.key(), "type": task.kind.name, "targets": list(task.targets),
            "members": [a.name for a in members], "rounds": rounds}
