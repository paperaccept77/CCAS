"""Executed-combination collection command."""
import torch
from ..runtime import build
from ..data import labels, requests, write_jsonl
from ..training.phases import set_phase
from ..combinations import collect, encode_record


def collect_combinations(args):
    engine = build(args)
    set_phase(engine, "ppo")
    truth = labels(args.labels)

    def records():
        with torch.no_grad():
            for request in requests(args.requests, engine.config, args.device):
                for record in collect(engine, request, truth[request.request_id], args.task, args.initiator, args.repetitions):
                    yield encode_record(record)

    write_jsonl(args.output, records())
