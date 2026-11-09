"""Paper settings and explicit implementation choices; see docs/METHOD.md."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    dim: int = 256
    image_dim: int = 512
    set_layers: int = 2
    heads: int = 4
    backbone: str = "models/Llama-3.1-8B-Instruct"
    precision: str = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 32
    input_tokens: int = 4096
    budget: float = 3.0
    max_tasks: int = 4
    max_rounds: int = 3
    max_members: int = 6
    history_limit: int = 50
    retrieval_top_k: int = 20
    within_category_pairs: bool = True
    exchange: bool = True
    cost_weight: float = 0.15
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    learning_rate: float = 1e-4
    policy_learning_rate: float = 3e-5
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.dim % self.heads or min(self.dim, self.heads, self.set_layers) < 1:
            raise ValueError("dim must be positive and divisible by heads")
        if not 1 <= self.max_members <= 6:
            raise ValueError("max_members must be in [1, 6]")
        if min(self.max_rounds, self.max_tasks, self.history_limit, self.retrieval_top_k) < 1:
            raise ValueError("execution limits must be positive")
        if self.budget < 0 or self.input_tokens < 256:
            raise ValueError("invalid budget/token limit")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported precision")

    @classmethod
    def load(cls, path: str):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)
