# Input and output

Files use UTF-8 JSONL: one object per line. Examples are in `examples/`.

## Requests

| Field | Format |
|---|---|
| `request_id`, `user_id` | String identifiers |
| `cutoff` | Prediction time, Unix UTC seconds |
| `location` | `[latitude, longitude]` |
| `request_features` | Float array `[256]` |
| `history` | Chronological visits: `{"timestamp": ..., "poi": {...}}` |
| `candidates` | Fixed candidate POIs |
| `base_scores` | Float array `[K]`, in candidate order |

Each POI contains `poi_id`, `category`, `latitude`, `longitude`, `mask`, `text`, and `image_embedding`.

- `mask`: `[text_available, image_available]` as booleans.
- `text`: string, or null when missing.
- `image_embedding`: float array `[512]`, or null when missing.
- `features`: float array `[256]`, required for candidates.
- `content_timestamp`: content availability time. Use `static_side_information: true` for content without timestamps.

History timestamps must precede `cutoff`. Apply modality masks before computing features and indices. Request and candidate features are precomputed encoder outputs; example vectors use zeros to illustrate shapes.

## Labels

```json
{"request_id":"example-001","positive_poi_id":"poi-b"}
```

Labels are used by training and evaluation. Prediction requests contain no labels. Keep retrieval misses in the evaluation set.

## Retrieval index

Optional records contain `object_id`, `user_id`, `split`, `timestamp`, `role`, `category`, `embedding`, and `text`.

Use `split: "train"`, role `trajectory` or `collaborative`, and a 256-dimensional embedding. Retrieved records must precede the request cutoff.

## Predictions

| Field | Meaning |
|---|---|
| `request_id` | Input request identifier |
| `candidate_ids` | Original candidate order |
| `base_scores`, `scores` | Base and final scores in original order |
| `ranking` | Candidate IDs sorted by final score |
| `tasks` | Members, operations, coverage, accessed objects, and costs |
| `incidence`, `roles` | Agent–task incidence matrix and row labels |
| `termination` | Episode termination reason |
| `cost`, `seconds`, `cost_ledger` | Normalized cost and measured timing records |

## Metrics and checkpoints

Evaluation outputs coverage and conditional/end-to-end Recall@L, NDCG@L, and MRR. Retrieval misses contribute zero to end-to-end metrics.

Checkpoints contain model configuration, ranking parameters, policy parameters, role adapters, and projections. The local backbone is loaded from the configured model directory.
