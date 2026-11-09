"""Model and adapter weight checkpoint IO."""
from pathlib import Path
import torch


def save_checkpoint(engine, path, stage, evaluator=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"config": engine.config.to_dict(), "model": engine.model.state_dict(),
            "policy": engine.policy.state_dict(), "executor": engine.executor.compact_state_dict(), "stage": stage}
    if evaluator is not None:
        data["evaluator"] = evaluator.state_dict()
    torch.save(data, path)
