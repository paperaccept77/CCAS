"""Chronological prediction-time request loading."""
import math
import torch
from ..config import Config
from ..types import Request, Visit
from .jsonl import read_jsonl
from .validation import parse_poi, vector


def parse_request(row: dict, config: Config, device="cpu") -> Request:
    forbidden = {"positive_poi_id", "label", "next_visit", "next_visit_time"}.intersection(row)
    if forbidden:
        raise ValueError(f"Labels must be in labels.jsonl, not request input: {forbidden}")
    cutoff = float(row["cutoff"])
    if not math.isfinite(cutoff):
        raise ValueError("cutoff must be finite Unix seconds")
    candidates = tuple(parse_poi(p, config, device, cutoff) for p in row["candidates"])
    if not candidates or len({p.poi_id for p in candidates}) != len(candidates):
        raise ValueError("candidate pool must be nonempty and unique")
    visits = tuple(Visit(parse_poi(v["poi"], config, device, cutoff), float(v["timestamp"])) for v in row["history"])
    if any(not math.isfinite(v.timestamp) or v.timestamp >= cutoff for v in visits):
        raise ValueError("history must precede cutoff")
    if [v.timestamp for v in visits] != sorted(v.timestamp for v in visits):
        raise ValueError("history must be chronological")
    features = torch.stack([vector(p["features"], config.dim, "candidate features", device) for p in row["candidates"]])
    location = tuple(map(float, row["location"]))
    if len(location) != 2 or not (-90 <= location[0] <= 90 and -180 <= location[1] <= 180):
        raise ValueError("location must be [latitude, longitude]")
    return Request(str(row["request_id"]), str(row["user_id"]), cutoff, location,
                   visits[-config.history_limit:], candidates,
                   vector(row["request_features"], config.dim, "request features", device), features,
                   vector(row["base_scores"], len(candidates), "base_scores", device))


def requests(path, config, device="cpu"):
    seen = set()
    for row in read_jsonl(path):
        request = parse_request(row, config, device)
        if request.request_id in seen:
            raise ValueError(f"duplicate request_id {request.request_id}")
        seen.add(request.request_id)
        yield request
