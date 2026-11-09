"""Shared feed-forward layers."""
from torch import nn


def mlp(input_dim, output_dim, hidden=None):
    return nn.Sequential(nn.Linear(input_dim, hidden or output_dim), nn.GELU(),
                         nn.Linear(hidden or output_dim, output_dim))
