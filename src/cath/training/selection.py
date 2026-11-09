"""Invitation imitation from preferred executed groups."""
import torch


def initialize_invitations(policy, optimizer, evaluator, records):
    """Behavior cloning of the evaluator's best *executed* group per matched state.

    Records must be refreshed after executor changes. Initiator is part of the
    comparison key. Supply multiple legal invitation orders in the records to
    obtain the paper's order augmentation.
    """
    grouped = {}
    for record in records:
        grouped.setdefault(record.comparison_key, []).append(record)
    losses = []
    for group in grouped.values():
        with torch.no_grad():
            best = max(group, key=lambda r: float(evaluator(r.observation, r.members)))
        for decision in best.invitation_decisions:
            if decision.observation.kind != 2:
                raise ValueError("combination initialization expects invitation decisions")
            distribution, _ = policy(decision.observation)
            action = torch.tensor(decision.action, device=decision.observation.mask.device)
            losses.append(-distribution.log_prob(action))
    if not losses:
        raise ValueError("no invitation records")
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
