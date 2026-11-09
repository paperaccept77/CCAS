"""Normalized operation upper bounds."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class CostProfile:
    unit_seconds: float
    routing_upper: float
    aggregation_upper: float
    completion_upper: float
    operation_upper: dict[str, float]  # e.g. preference.history_summary, normalized units
    calibrated: bool = False

    def __post_init__(self):
        values = [self.unit_seconds, self.routing_upper, self.aggregation_upper, self.completion_upper,
                  *self.operation_upper.values()]
        if any(not 0 < x < float("inf") for x in values):
            raise ValueError("cost scales and upper bounds must be finite and positive")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def upper(self, role, operation):
        key = f"{role.name.lower()}.{operation}"
        return self.operation_upper[key]
