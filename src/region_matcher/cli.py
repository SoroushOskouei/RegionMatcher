from __future__ import annotations

import argparse
from collections.abc import Sequence

from .constants import DEFAULT_CHECKPOINT, DEFAULT_OUTPUT_DIR
from .inference import InferenceOptions, run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="region-match",
        description="Match a query crop against one full image or a gallery directory.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint path (default: models/model.pt)",
    )
    parser.add_argument("--query", required=True, help="Small query crop image")
    parser.add_argument(
        "--gallery",
        required=True,
        help="One full image or a directory searched recursively",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--full-size", type=int, default=None, help="Normally read from checkpoint")
    parser.add_argument("--query-size", type=int, default=None, help="Normally read from checkpoint")
    parser.add_argument("--device", default=None, help="Examples: cpu, cuda, cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_inference(
        InferenceOptions(
            checkpoint=args.checkpoint,
            query=args.query,
            gallery=args.gallery,
            output=args.output,
            top_k=args.top_k,
            batch_size=args.batch_size,
            full_size=args.full_size,
            query_size=args.query_size,
            device=args.device,
            seed=args.seed,
        )
    )
    return 0
