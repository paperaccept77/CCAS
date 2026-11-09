"""Executed-group set utility model and regression."""
import torch
import torch.nn.functional as F
from torch import nn
from ..neural import SetEncoder, mlp
from ..types import Role


class SetEvaluator(nn.Module):
    """Q_omega fitted solely to executed combinations, never unexecuted pseudo-labels."""
    def __init__(self, config):
        super().__init__()
        d = config.dim
        self.roles = nn.Embedding(6, d)
        self.set_encoder = SetEncoder(d, config.heads, config.set_layers)
        self.context = mlp(4*d+3, d)
        self.task = mlp(3*d+4, d)
        self.head = mlp(2*d, 1, d)

    def forward(self, observation, members):
        role_ids = torch.tensor([int(Role[name]) for name in members], device=observation.global_features.device)
        group = self.set_encoder(self.roles(role_ids))
        context = self.context(observation.global_features) + self.task(observation.action_features[0])
        return self.head(torch.cat((context, group))).squeeze(-1)


def fit_set_evaluator(evaluator, optimizer, records):
    if not records:
        raise ValueError("executed combination records are required")
    predictions = torch.stack([evaluator(r.observation, r.members) for r in records])
    targets = predictions.new_tensor([r.measured_gain for r in records])
    loss = F.mse_loss(predictions, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
