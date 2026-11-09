"""Finite tensor and POI content validation."""
import torch
from ..types import POI


def vector(value, size, name, device):
    result = torch.tensor(value, dtype=torch.float32, device=device)
    if result.shape != (size,) or not torch.isfinite(result).all():
        raise ValueError(f"{name}: expected finite [{size}]")
    return result


def parse_poi(row, config, device, cutoff):
    mask = row["mask"]
    if len(mask) != 2 or any(type(x) is not bool for x in mask):
        raise ValueError("mask must be [bool text_available, bool image_available]")
    if mask[0] != (row.get("text") is not None) or mask[1] != (row.get("image_embedding") is not None):
        raise ValueError("unavailable content must be null; available content must be supplied")
    if row.get("content_timestamp") is None:
        if not row.get("static_side_information", False):
            raise ValueError("untimestamped content requires static_side_information=true")
    elif row["content_timestamp"] > cutoff:
        raise ValueError("future content is forbidden")
    lat, lon = float(row["latitude"]), float(row["longitude"])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("invalid coordinates")
    return POI(str(row["poi_id"]), str(row["category"]), lat, lon, tuple(mask),
               row.get("text"), vector(row["image_embedding"], config.image_dim, "image", device) if mask[1] else None)
