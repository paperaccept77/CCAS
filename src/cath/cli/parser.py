"""CATH command-line argument definitions."""
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="CATH paper-based reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("infer", "train", "collect-combinations"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/paper.json")
        command.add_argument("--costs", required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--index")
        command.add_argument("--checkpoint", required=name != "train")
        command.add_argument("--requests", required=name != "train")
        command.add_argument("--output", required=True)
        if name == "train":
            command.add_argument("--stage", choices=("warmup", "selection", "ppo"), required=True)
            command.add_argument("--labels")
            command.add_argument("--combinations")
            command.add_argument("--scripts")
            command.add_argument("--steps", type=int, default=20000)
            command.add_argument("--batch-size", type=int, default=64)
        if name == "collect-combinations":
            command.add_argument("--labels", required=True)
            command.add_argument("--task", required=True, help="e.g. comparison:0,1")
            command.add_argument("--initiator", required=True)
            command.add_argument("--repetitions", type=int, default=3)
    metric = commands.add_parser("evaluate")
    metric.add_argument("--predictions", required=True)
    metric.add_argument("--labels", required=True)
    metric.add_argument("--cutoff", type=int, default=20)
    metric.add_argument("--output", required=True)
    return parser
