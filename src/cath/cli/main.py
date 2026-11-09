"""CATH command dispatch."""
from .parser import build_parser
from .infer import infer
from .collect import collect_combinations
from .evaluate import evaluate_predictions
from ..training.runner import train


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "infer":
        infer(args)
    elif args.command == "train":
        if args.steps < 1 or args.batch_size < 1:
            parser.error("steps and batch-size must be positive")
        if args.stage != "warmup" and not args.checkpoint:
            parser.error("selection and PPO require a warm-up/selection checkpoint")
        train(args)
    elif args.command == "collect-combinations":
        collect_combinations(args)
    else:
        evaluate_predictions(args)
