"""Seeded construction of a configured CATH engine."""
import random
import torch
from .config import Config
from .cost import CostProfile
from .engine import Engine
from .executor import LLMExecutor
from .neural import RankingModel
from .policy import Policy
from .tools import EvidenceTools


def build(args):
    config = Config.load(args.config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(args.device)
    model = RankingModel(config).to(device)
    policy = Policy(config).to(device)
    executor = LLMExecutor(config, device)
    engine = Engine(config, model, policy, executor, EvidenceTools(config, args.index, device), CostProfile.load(args.costs))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if checkpoint["config"] != config.to_dict():
            raise ValueError("checkpoint configuration must exactly match --config")
        model.load_state_dict(checkpoint["model"])
        policy.load_state_dict(checkpoint["policy"])
        missing, unexpected = executor.load_state_dict(checkpoint["executor"], strict=False)
        if unexpected or any(not name.startswith("backbone.") or "lora_" in name for name in missing):
            raise ValueError("incomplete executor checkpoint")
    return engine
