"""Training stage orchestration and optimizer steps."""
import json
import random
import torch
from ..runtime import build
from ..checkpoints import save_checkpoint
from ..data import labels, read_jsonl, requests
from ..combinations import decode_record
from . import SetEvaluator, assign_rewards, fit_set_evaluator, initialize_invitations, ppo_update, set_phase, warmup_loss


def train(args):
    engine = build(args)
    cfg = engine.config
    set_phase(engine, args.stage)
    if args.stage == "selection":
        if not args.combinations:
            raise ValueError("selection requires --combinations")
        records = [decode_record(row, cfg, args.device) for row in read_jsonl(args.combinations)]
        evaluator = SetEvaluator(cfg).to(args.device)
        q_optimizer = torch.optim.AdamW(evaluator.parameters(), lr=cfg.learning_rate)
        for step in range(args.steps):
            loss = fit_set_evaluator(evaluator, q_optimizer, records)
            if step % 100 == 0:
                print(json.dumps({"step": step, "evaluator_mse": loss}))
        optimizer = torch.optim.AdamW(engine.policy.parameters(), lr=cfg.policy_learning_rate)
        for step in range(args.steps):
            initialize_invitations(engine.policy, optimizer, evaluator, records)
        save_checkpoint(engine, args.output, args.stage, evaluator)
        return
    if not args.requests or not args.labels:
        raise ValueError("warmup and ppo require --requests and --labels")
    training_requests = list(requests(args.requests, cfg, args.device))
    ground_truth = labels(args.labels)
    if set(ground_truth) != {q.request_id for q in training_requests}:
        raise ValueError("training request IDs and labels must match exactly")
    training_requests = [q for q in training_requests if ground_truth[q.request_id] in {p.poi_id for p in q.candidates}]
    if not training_requests:
        raise ValueError("no covered training requests")
    scripts = {str(row["request_id"]): row["actions"] for row in read_jsonl(args.scripts)} if args.scripts else {}
    parameters = (list(engine.model.parameters()) + list(engine.executor.parameters())) if args.stage == "warmup" else list(engine.policy.parameters())
    parameters = [p for p in parameters if p.requires_grad]
    rate = cfg.learning_rate if args.stage == "warmup" else cfg.policy_learning_rate
    optimizer = torch.optim.AdamW(parameters, lr=rate)
    updates = 0
    for step in range(args.steps):
        batch = random.choices(training_requests, k=args.batch_size)
        if args.stage == "warmup":
            optimizer.zero_grad(set_to_none=True)
            losses = []
            # Serial gradient accumulation avoids retaining a batch of LLM graphs.
            for request in batch:
                loss, _ = warmup_loss(engine, request, ground_truth[request.request_id], scripts.get(request.request_id))
                if loss is not None:
                    (loss / args.batch_size).backward()
                    losses.append(float(loss.detach()))
            if losses:
                # Average over supervised episodes, excluding immediate STOP episodes.
                for p in parameters:
                    if p.grad is not None:
                        p.grad.mul_(args.batch_size / len(losses))
                torch.nn.utils.clip_grad_norm_(parameters, cfg.gradient_clip)
                for group in optimizer.param_groups:
                    group["lr"] = rate * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
                optimizer.step()
                updates += 1
            info = {"loss": sum(losses)/len(losses) if losses else None, "supervised": len(losses)}
        else:
            episodes = []
            with torch.no_grad():
                for request in batch:
                    result = engine.run(request, mode="sample")
                    assign_rewards(result, request, ground_truth[request.request_id], cfg.cost_weight)
                    episodes.append(result.decisions)
            info = ppo_update(engine.policy, optimizer, episodes, cfg)
            updates += int(bool(info["transitions"]))
        if step % 10 == 0:
            print(json.dumps({"step": step, **info}))
    if not updates:
        raise RuntimeError("no optimizer update occurred; inspect budget and warm-up trajectories")
    save_checkpoint(engine, args.output, args.stage)
