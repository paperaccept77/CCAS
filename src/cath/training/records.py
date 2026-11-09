"""Matched executed-combination training records."""
from dataclasses import dataclass


@dataclass
class CombinationRecord:
    comparison_key: str  # identical request/state/task/targets/initiator/local rules
    observation: object  # task-specific Observation, with exactly one task action row
    members: tuple[str, ...]
    measured_gain: float
    invitation_decisions: list
