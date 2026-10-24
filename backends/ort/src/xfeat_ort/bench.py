"""`xfeat-ort-bench`: latency of the XFeat and LighterGlue graphs on random inputs."""

import argparse
from pathlib import Path

import numpy as np

from xfeat_bench.timing import measure, write_results
from xfeat_core.features import Features
from xfeat_ort.models import LighterGlueModel, XFeatModel

RANDOM_SEED = 0


def _random_features(count: int, size: tuple[int, int], rng: np.random.Generator) -> Features:
    descriptors = rng.standard_normal((count, 64)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    keypoints = (rng.random((count, 2)) * (np.asarray(size) - 1)).astype(np.float32)
    return Features(keypoints, np.ones(count, dtype=np.float32), descriptors, size)


def main() -> None:
    """Entry point of `xfeat-ort-bench`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xfeat", type=Path, required=True)
    parser.add_argument("--lighterglue", type=Path, required=True)
    parser.add_argument("--image", type=Path, help=".npy gray image; random noise if omitted")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    xfeat = XFeatModel(args.xfeat, args.threads)
    lighterglue = LighterGlueModel(args.lighterglue, args.threads)
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
    settings = {"threads": str(args.threads), "top_k": str(meta.top_k), "batch": str(meta.batch)}

    results = [
        measure(
            "xfeat",
            lambda: xfeat.run_raw(images),
            warmup=args.warmup,
            runs=args.runs,
            settings=settings,
        ),
        measure(
            "lighterglue",
            lambda: lighterglue.run_raw(first, second),
            warmup=args.warmup,
            runs=args.runs,
            settings=settings,
        ),
    ]
    print(write_results(results, args.out_dir))  # noqa: T201  # CLI output
    for result in results:
        print(result.name, result.summary())  # noqa: T201
