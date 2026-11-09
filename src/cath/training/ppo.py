"""Clipped policy and value updates with original masks."""
import torch
from torch import nn
from .advantages import gae


def ppo_update(policy, optimizer, episodes, config):
    """Re-evaluate saved observations and exact original masks; tools are not rerun."""
    samples = [(decision, advantage, target) for episode in episodes
               for decision, (advantage, target) in zip(episode, gae(episode, config.gae_lambda))]
    if not samples:
        return {"loss": None, "transitions": 0}
    device = next(policy.parameters()).device
    advantages = torch.tensor([a for _, a, _ in samples], device=device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    losses = []
    for _ in range(config.ppo_epochs):
        objectives = []
        for index, (decision, _, target) in enumerate(samples):
            distribution, value = policy(decision.observation)
            action = torch.tensor(decision.action, device=device)
            ratio = (distribution.log_prob(action) - decision.old_log_prob).exp()
            advantage = advantages[index]
            clipped = torch.clamp(ratio, 1-config.ppo_clip, 1+config.ppo_clip) * advantage
            actor = -torch.minimum(ratio * advantage, clipped)
            critic = (value - value.new_tensor(target)).square()
            objectives.append(actor + config.value_weight * critic - config.entropy_weight * distribution.entropy())
        loss = torch.stack(objectives).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"loss": sum(losses)/len(losses), "transitions": len(samples)}
