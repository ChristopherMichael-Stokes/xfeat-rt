"""`xfeat-trt-bench`: latency of the XFeat and LighterGlue engines on random inputs."""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from xfeat_bench.timing import Measurement, measure, write_results
from xfeat_trt.arguments import add_engine_arguments, build_options
from xfeat_trt.models import LighterGlueModel, XFeatModel
from xfeat_trt.random_inputs import random_features, random_images

RANDOM_SEED = 0


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
    add_engine_arguments(parser)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    options = build_options(args)

    rng = np.random.default_rng(RANDOM_SEED)
    xfeat = XFeatModel(args.xfeat, args.cache_dir, options)
    lighterglue = LighterGlueModel(args.lighterglue, args.cache_dir, options)
    meta = xfeat.metadata
    images = np.load(args.image).astype(np.float32) if args.image else random_images(meta, rng)
    size = (meta.width, meta.height)
    first = lighterglue.pad(random_features(lighterglue.metadata.top_k, size, rng))
    second = lighterglue.pad(random_features(lighterglue.metadata.top_k, size, rng))
    feeds = lighterglue.feeds([(first, second)] * lighterglue.metadata.batch)
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
