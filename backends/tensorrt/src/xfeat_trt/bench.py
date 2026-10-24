"""`xfeat-trt-bench`: latency of the XFeat and LighterGlue engines on random inputs."""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from xfeat_bench.timing import Measurement, measure, write_results
from xfeat_core.features import Features
from xfeat_trt.engine import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OPTIMIZATION_LEVEL,
    MAX_OPTIMIZATION_LEVEL,
    BuildOptions,
)
from xfeat_trt.models import LighterGlueModel, XFeatModel

RANDOM_SEED = 0


def _random_features(count: int, size: tuple[int, int], rng: np.random.Generator) -> Features:
    descriptors = rng.standard_normal((count, 64)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    keypoints = (rng.random((count, 2)) * (np.asarray(size) - 1)).astype(np.float32)
    return Features(keypoints, np.ones(count, dtype=np.float32), descriptors, size)


def main() -> None:
    """Entry point of `xfeat-trt-bench`.

    Each model is timed with its inputs resident on the device (`*_engine`: execute and stream
    sync) and end to end (`*_end_to_end`: upload, execute, download). `--graph` runs the engine
    through a CUDA graph.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--xfeat", type=Path, required=True)
    parser.add_argument("--lighterglue", type=Path, required=True)
    parser.add_argument("--image", type=Path, help=".npy gray image; random noise if omitted")
    parser.add_argument("--graph", action="store_true", help="execute through a CUDA graph")
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    options = BuildOptions(args.tf32, args.optimization_level)

    rng = np.random.default_rng(RANDOM_SEED)
    xfeat = XFeatModel(args.xfeat, args.cache_dir, options)
    lighterglue = LighterGlueModel(args.lighterglue, args.cache_dir, options)
    meta = xfeat.metadata
    shape = (meta.batch, 1, meta.height, meta.width)
    images = (
        np.load(args.image).astype(np.float32)
        if args.image
        else rng.random(shape, dtype=np.float32)
    )
    size = (meta.width, meta.height)
    first = lighterglue.pad(_random_features(lighterglue.metadata.top_k, size, rng))
    second = lighterglue.pad(_random_features(lighterglue.metadata.top_k, size, rng))
    feeds = lighterglue.feeds(first, second)
    settings = {
        "top_k": str(meta.top_k),
        "batch": str(meta.batch),
        "graph": str(args.graph),
        "tf32": str(args.tf32),
        "optimization_level": str(args.optimization_level),
    }

    xfeat.runner.upload({"image": images})
    lighterglue.runner.upload(feeds)
    if args.graph:
        xfeat.runner.capture_graph()
        lighterglue.runner.capture_graph()

    def timed(name: str, call: Callable[[], object]) -> Measurement:
        return measure(name, call, warmup=args.warmup, runs=args.runs, settings=settings)

    results = [
        timed("xfeat_engine", xfeat.runner.execute),
        timed("xfeat_end_to_end", lambda: xfeat.runner.run({"image": images})),
        timed("lighterglue_engine", lighterglue.runner.execute),
        timed("lighterglue_end_to_end", lambda: lighterglue.runner.run(feeds)),
    ]
    print(write_results(results, args.out_dir))  # noqa: T201  # CLI output
    for result in results:
        print(result.name, result.summary())  # noqa: T201
