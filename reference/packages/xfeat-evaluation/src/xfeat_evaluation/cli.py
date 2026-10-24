"""`xfeat-megadepth1500`: score one reference configuration on MegaDepth-1500."""

import argparse
import json
from pathlib import Path

import torch

from xfeat_evaluation import megadepth1500
from xfeat_reference.pipeline import GridNormalisation, ReferenceConfig, ReferencePipeline

README_RESIZE = 640
README_TOP_K = 1300
"""Settings of the XFeat README's LighterGlue MegaDepth-1500 row (AUC 0.444/0.610/0.746)."""
DEFAULT_DATA_DIR = Path.home() / ".cache" / "xfeat-rt" / "gluefactory-data"


def main() -> None:
    """Parse arguments, run the evaluation and print the summary."""
    args = _parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.use_deterministic_algorithms(mode=True)
    config = ReferenceConfig(grid=args.grid, pruning=args.pruning, top_k=args.top_k)
    settings = megadepth1500.EvaluationSettings(
        resize=args.resize, data_dir=args.data_dir, pair_limit=args.pair_limit
    )
    summary = megadepth1500.run(ReferencePipeline(config, device), settings, args.output_dir)
    print(json.dumps(summary, indent=2))  # noqa: T201


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--grid", type=GridNormalisation, choices=list(GridNormalisation), required=True
    )
    parser.add_argument("--pruning", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resize", type=int, default=README_RESIZE)
    parser.add_argument("--top-k", type=int, default=README_TOP_K)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pair-limit", type=int, default=megadepth1500.PAIR_COUNT)
    return parser
