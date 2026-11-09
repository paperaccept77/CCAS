"""Training-only role-scoped dense retrieval."""
import math
import torch.nn.functional as F
from ..data import read_jsonl, vector
from ..types import Role


class TrainingIndex:
    def __init__(self, config, index_path=None, device="cpu"):
        self.config = config
        self.index = []
        if index_path:
            for row in read_jsonl(index_path):
                if row["split"] != "train" or not math.isfinite(float(row["timestamp"])):
                    raise ValueError("retrieval records require split=train and finite timestamp")
                role = Role[row["role"].upper()]
                if role not in (Role.TRAJECTORY, Role.COLLABORATIVE):
                    raise ValueError("index roles must be trajectory or collaborative")
                self.index.append({**row, "role_enum": role,
                                   "embedding": vector(row["embedding"], config.dim, "index embedding", device)})

    def eligible_records(self, request, role):
        return [r for r in self.index if r["role_enum"] == role and r["timestamp"] < request.cutoff
                and str(r["user_id"]) != request.user_id]

    def retrieve(self, request, role, p, shared, evidence, operation):
        # Query changes deterministically after broadcast. No discrete parameter sampling.
        query = request.candidate_features[p] + shared
        records = self.eligible_records(request, role)
        if operation == "related_pois":
            records = [r for r in records if r["category"] == request.candidates[p].category]
        scored = [(float(F.cosine_similarity(query, r["embedding"], dim=0)), str(r["object_id"]), r) for r in records]
        top = sorted(scored, key=lambda x: (-x[0], x[1]))[:self.config.retrieval_top_k]
        if top:
            evidence.candidate_text[p] = [f"retrieved={r['object_id']}; {r['text']}" for _, _, r in top]
            evidence.retrieval_vectors[p] = [r["embedding"] for _, _, r in top]
            evidence.objects.extend(str(r["object_id"]) for _, _, r in top)
