"""Episode tensors and serialized trace results."""
from dataclasses import dataclass
import torch


@dataclass
class EpisodeResult:
    scores: torch.Tensor
    task_scores: list[torch.Tensor]
    decisions: list
    record: dict
