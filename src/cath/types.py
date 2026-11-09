"""Typed prediction-time interfaces. Labels deliberately live outside Request."""
from dataclasses import dataclass, field
from enum import IntEnum
from torch import Tensor


class Role(IntEnum):
    PREFERENCE = 0
    TRAJECTORY = 1
    SPATIOTEMPORAL = 2
    TEXT = 3
    VISUAL = 4
    COLLABORATIVE = 5


class TaskType(IntEnum):
    REQUEST = 0
    COMPARISON = 1
    COMPENSATION = 2


OPERATIONS = {
    Role.PREFERENCE: ("history_summary", "relevant_history", "compare", "refine", "request_peer", "finish"),
    Role.TRAJECTORY: ("recent_sequence", "retrieve_sequence", "compare", "refine", "request_peer", "finish"),
    Role.SPATIOTEMPORAL: ("distance", "transitions", "compare", "refine", "request_peer", "finish"),
    Role.TEXT: ("passages", "attributes", "compare", "refine", "request_peer", "finish"),
    Role.VISUAL: ("target_images", "history_images", "compare", "refine", "request_peer", "finish"),
    Role.COLLABORATIVE: ("similar_users", "related_pois", "retrieve_sequence", "refine", "request_peer", "finish"),
}


@dataclass(frozen=True)
class POI:
    poi_id: str
    category: str
    latitude: float
    longitude: float
    mask: tuple[bool, bool]  # text, image
    text: str | None = None
    image: Tensor | None = None  # frozen CLIP vector [image_dim]


@dataclass(frozen=True)
class Visit:
    poi: POI
    timestamp: float


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    cutoff: float
    location: tuple[float, float]
    history: tuple[Visit, ...]
    candidates: tuple[POI, ...]
    request_features: Tensor  # [D], frozen sequence/context representation
    candidate_features: Tensor  # [K,D], frozen, already masked side information
    base_scores: Tensor  # [K], original retriever order

    @property
    def masks(self):
        return self.base_scores.new_tensor([p.mask for p in self.candidates])


@dataclass(frozen=True)
class Task:
    kind: TaskType
    targets: tuple[int, ...]  # indices into the unchanged candidate pool

    def key(self):
        return f"{self.kind.name.lower()}:" + ",".join(map(str, self.targets))


@dataclass
class OperationOutput:
    messages: dict[int, Tensor]  # covered candidate -> [D]
    summary: Tensor  # [D]
    status: str
    objects: list[str] = field(default_factory=list)
    seconds: float = 0.0
    normalized_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class EpisodeState:
    context: Tensor  # [D]
    candidates: Tensor  # [K,D]
    initial_context: Tensor
    initial_candidates: Tensor
    initial_reference: Tensor  # [K], retains warm-up gradient graph
    memories: Tensor  # [6,D]
    scores: Tensor  # [K]
    summaries: list[Tensor] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)


def stable_order(scores: Tensor, request: Request) -> list[int]:
    values = scores.detach().cpu().tolist()
    return sorted(range(len(values)), key=lambda i: (-values[i], request.candidates[i].poi_id))
