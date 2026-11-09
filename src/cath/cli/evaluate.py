"""Request-macro ranking evaluation command."""
import json
from pathlib import Path
from ..data import labels, read_jsonl
from ..metrics import evaluate


def evaluate_predictions(args):
    result = evaluate(read_jsonl(args.predictions), labels(args.labels), args.cutoff)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
