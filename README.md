# CATH

Next-POI recommendation with complementary agent groups and task hypergraphs.

## Setup

```sh
pip install -e .
```

Set the local Llama model path in `configs/paper.json` and hardware costs in `configs/costs.local.json`. See `examples/costs.example.json` for the cost format.

## Data

Requests contain visit history, a fixed candidate pool, base scores, precomputed features, and modality masks. Labels are stored separately.

- State features: 256 dimensions.
- Image features: 512 dimensions.
- Output: candidate scores, ranking, task records, and costs.

See [input format](docs/IO.md) and `examples/`.

## Train

```sh
cath train --stage warmup --requests data/train.requests.jsonl --labels data/train.labels.jsonl --costs configs/costs.local.json --output checkpoints/warmup.pt
cath train --stage ppo --checkpoint checkpoints/warmup.pt --requests data/train.requests.jsonl --labels data/train.labels.jsonl --costs configs/costs.local.json --output checkpoints/cath.pt
```

Use `--index data/train.index.jsonl` to enable retrieval. Optional group-selection warm-up is available through `cath collect-combinations` and `cath train --stage selection`.

## Predict

```sh
cath infer --checkpoint checkpoints/cath.pt --requests data/test.requests.jsonl --costs configs/costs.local.json --output outputs/predictions.jsonl
cath evaluate --predictions outputs/predictions.jsonl --labels data/test.labels.jsonl --output outputs/metrics.json
```

Run `cath --help` for all commands.
