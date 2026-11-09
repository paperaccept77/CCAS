"""JSON serialization of observations and executed groups."""
import torch
from ..policy import Decision, Observation
from ..training.records import CombinationRecord


def encode_observation(o):
    return {"global_features": o.global_features.cpu().tolist(), "action_features": o.action_features.cpu().tolist(),
            "selected": o.selected.cpu().tolist(), "mask": o.mask.cpu().tolist(),
            "kind": o.kind, "action_keys": list(o.action_keys)}


def decode_observation(row, config, device):
    return Observation(torch.tensor(row["global_features"], device=device),
                       torch.tensor(row["action_features"], device=device),
                       torch.tensor(row["selected"], device=device).reshape(-1, 2*config.dim+1),
                       torch.tensor(row["mask"], dtype=torch.bool, device=device),
                       int(row["kind"]), tuple(row["action_keys"]))


def encode_record(record):
    return {"comparison_key": record.comparison_key, "observation": encode_observation(record.observation),
            "members": list(record.members), "measured_gain": record.measured_gain,
            "invitations": [{"observation": encode_observation(d.observation), "action": d.action,
                             "old_log_prob": d.old_log_prob, "value": d.value} for d in record.invitation_decisions]}


def decode_record(row, config, device):
    decisions = [Decision(decode_observation(d["observation"], config, device), d["action"], d["old_log_prob"], d["value"])
                 for d in row["invitations"]]
    return CombinationRecord(row["comparison_key"], decode_observation(row["observation"], config, device),
                             tuple(row["members"]), float(row["measured_gain"]), decisions)
