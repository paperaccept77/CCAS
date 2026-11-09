"""Masked policy observations and sampled transitions."""
from dataclasses import dataclass
import torch


@dataclass
class Observation:
    global_features: torch.Tensor  # [4D+3]
    action_features: torch.Tensor  # [A,3D+4]
    selected: torch.Tensor  # [N,2D+1]
    mask: torch.Tensor  # bool [A]
    kind: int  # task=0, initiator=1, invitation=2, operation=3
    action_keys: tuple[str, ...]

    def detached(self):
        return Observation(self.global_features.detach().clone(), self.action_features.detach().clone(),
                           self.selected.detach().clone(), self.mask.detach().clone(), self.kind, self.action_keys)


@dataclass
class Decision:
    observation: Observation
    action: int
    old_log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False
