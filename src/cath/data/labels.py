"""Recommendation labels keyed by request ID."""
from .jsonl import read_jsonl


def labels(path):
    result = {}
    for row in read_jsonl(path):
        key = str(row["request_id"])
        if key in result:
            raise ValueError(f"duplicate label {key}")
        result[key] = str(row["positive_poi_id"])
    return result
