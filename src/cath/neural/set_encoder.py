"""Permutation-invariant attention pooling."""
import torch
from torch import nn


class SetEncoder(nn.Module):
    """Self-attention blocks followed by learned-seed pooling; no positions/dropout."""
    def __init__(self, dim, heads, layers):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0, batch_first=True)
            for _ in range(layers)])
        self.seed = nn.Parameter(torch.zeros(1, 1, dim))
        self.pool = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, items):
        # Input [N,D]; output [D]. Empty sets have an exact zero summary.
        if items.shape[0] == 0:
            return items.new_zeros(items.shape[-1])
        x = items.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.norm(self.pool(self.seed, x, x, need_weights=False)[0][0, 0])
