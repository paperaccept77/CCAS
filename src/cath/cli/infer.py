"""Ranking prediction command."""
import torch
from ..runtime import build
from ..data import requests, write_jsonl
from ..training.phases import set_phase


def infer(args):
    engine = build(args)
    set_phase(engine, "ppo")
    engine.policy.eval()
    def predictions():
        with torch.no_grad():
            for request in requests(args.requests, engine.config, args.device):
                yield engine.run(request).record
    write_jsonl(args.output, predictions())
