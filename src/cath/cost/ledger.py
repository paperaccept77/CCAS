"""Synchronized timing and joint budget reservations."""
from dataclasses import dataclass, field
from time import perf_counter
import torch
from .profile import CostProfile


@dataclass
class Ledger:
    budget: float
    profile: CostProfile
    device: torch.device
    spent: float = 0.0
    reserved: float = 0.0
    entries: list[dict] = field(default_factory=list)

    @property
    def remaining(self):
        return max(0.0, self.budget - self.spent - self.reserved)

    def reserve(self, amount):
        if amount > self.remaining + 1e-10:
            raise ValueError("insufficient budget for reservation")
        self.reserved += amount

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def measure_reserved(self, name, upper, function):
        self.sync()
        start = perf_counter()
        result = function()
        self.sync()
        seconds = perf_counter() - start
        cost = seconds / self.profile.unit_seconds
        self.reserved -= upper
        self.spent += cost
        self.entries.append({"name": name, "seconds": seconds, "cost": cost, "reserved_upper": upper})
        if cost > upper + 1e-9:
            raise RuntimeError(f"{name}: actual cost {cost:.6f} exceeds calibrated upper bound {upper}; recalibrate")
        return result, seconds, cost

    def measure(self, name, upper, function):
        self.reserve(upper)
        return self.measure_reserved(name, upper, function)
