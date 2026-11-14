"""Command-line options shared by the TensorRT entry points."""

import argparse
from pathlib import Path

from xfeat_trt.engine import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OPTIMIZATION_LEVEL,
    MAX_OPTIMIZATION_LEVEL,
    BuildOptions,
)


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """Add `--cache-dir`, `--tf32` and `--optimization-level` to `parser`."""
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--tf32", action="store_true", help="allow TF32 in FP32 layers (faster, less precise)"
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=DEFAULT_OPTIMIZATION_LEVEL,
        choices=range(MAX_OPTIMIZATION_LEVEL + 1),
        help="TensorRT builder optimization level (higher builds slower, runs faster)",
    )


def build_options(args: argparse.Namespace) -> BuildOptions:
    """Builder options from arguments parsed with `add_engine_arguments`."""
    return BuildOptions(args.tf32, args.optimization_level)
