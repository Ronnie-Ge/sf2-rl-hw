from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sf2-rl-hw",
        description="Utilities for the Street Fighter II RL course project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a PPO agent.")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baseline.yaml"),
        help="Path to the experiment config.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint.")
    eval_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baseline.yaml"),
        help="Path to the experiment config.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override.",
    )

    record_parser = subparsers.add_parser("record", help="Record a policy video.")
    record_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baseline.yaml"),
        help="Path to the experiment config.",
    )
    record_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Single checkpoint to record.",
    )
    record_parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Record the latest N checkpoints for this experiment.",
    )
    record_parser.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default=None,
        help="Glob pattern for batch checkpoint selection.",
    )

    play_parser = subparsers.add_parser("play", help="Run one-off agent play sessions.")
    play_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baseline.yaml"),
        help="Path to the experiment config.",
    )
    play_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        from .train import run_training

        run_training(args.config)
        return

    if args.command == "evaluate":
        from .evaluate import run_evaluation

        run_evaluation(args.config, args.checkpoint)
        return

    if args.command == "record":
        from .record import run_recording

        run_recording(args.config, args.checkpoint, args.latest, args.glob_pattern)
        return

    if args.command == "play":
        from .play import run_play

        run_play(args.config, args.checkpoint)
        return

    parser.error(f"Unsupported command: {args.command}")
