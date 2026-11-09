"""Generate public IO-format fixtures only; all embeddings are zero placeholders."""
import json
from pathlib import Path


def main():
    output = Path(__file__).resolve().parents[1] / "examples"
    output.mkdir(exist_ok=True)

    def poi(identifier, text=None, image=False, candidate=False):
        record = {"poi_id": identifier, "category": "cafe", "latitude": 40.73,
                  "longitude": -73.99, "mask": [text is not None, image], "text": text,
                  "image_embedding": [0.0] * 512 if image else None,
                  "content_timestamp": None, "static_side_information": True}
        if candidate:
            record["features"] = [0.0] * 256
        return record

    request = {"request_id": "example-001", "user_id": "user-001", "cutoff": 1700000000.0,
               "location": [40.73, -73.99], "request_features": [0.0] * 256,
               "history": [{"timestamp": 1699990000.0, "poi": poi("past-cafe", "Quiet cafe", True)}],
               "candidates": [poi("poi-a", "Coffee and outdoor seating", candidate=True),
                              poi("poi-b", candidate=True)], "base_scores": [1.2, 0.9]}
    (output / "requests.jsonl").write_text(json.dumps(request) + "\n", encoding="utf-8")
    index = [{"object_id": f"train-{role}-001", "user_id": "other-user", "split": "train",
              "timestamp": 1699980000.0, "role": role, "category": "cafe", "embedding": [0.0] * 256,
              "text": "Synthetic format example: cafe visit in training history."}
             for role in ("trajectory", "collaborative")]
    (output / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index), encoding="utf-8")


if __name__ == "__main__":
    main()
