"""Candidate-linked legal evidence containers."""
from dataclasses import dataclass, field
import torch


@dataclass
class Evidence:
    candidate_text: dict[int, list[str]] = field(default_factory=dict)
    candidate_images: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    retrieval_vectors: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    objects: list[str] = field(default_factory=list)
    status: str = "ok"

    @property
    def coverage(self):
        return sorted(set(self.candidate_text) | set(self.candidate_images) | set(self.retrieval_vectors))
