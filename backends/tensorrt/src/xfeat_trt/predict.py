"""`xfeat-trt-predict`: extract and match image pairs, writing one `.npz` per pair."""

import argparse
from pathlib import Path

import numpy as np

from xfeat_trt.engine import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OPTIMIZATION_LEVEL,
    MAX_OPTIMIZATION_LEVEL,
    BuildOptions,
)
from xfeat_trt.models import LighterGlueModel, XFeatModel


def main() -> None:
    """Entry point of `xfeat-trt-predict`.

    Images are `.npy` float32 (1, 1, H, W) arrays in [0, 1], consumed two at a time as pairs.
    Each `pair{i}.npz` holds `{keypoints,scores,descriptors}{0,1}` of the valid features and
    `matches` (M, 2).
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--xfeat", type=Path, required=True, help="exported XFeat ONNX file")
    parser.add_argument(
        "--lighterglue", type=Path, required=True, help="exported LighterGlue ONNX file"
    )
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
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
    args = parser.parse_args()
    options = BuildOptions(args.tf32, args.optimization_level)
    if len(args.images) % 2:
        parser.error("--images takes an even number of files (pairs)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xfeat = XFeatModel(args.xfeat, args.cache_dir, options)
    lighterglue = LighterGlueModel(args.lighterglue, args.cache_dir, options)
    for index, (path0, path1) in enumerate(zip(args.images[::2], args.images[1::2], strict=True)):
        first, second = (xfeat.extract(np.load(p).astype(np.float32))[0] for p in (path0, path1))
        matches = lighterglue.match(first, second)
        np.savez(
            args.out_dir / f"pair{index}.npz",
            keypoints0=first.keypoints,
            scores0=first.scores,
            descriptors0=first.descriptors,
            keypoints1=second.keypoints,
            scores1=second.scores,
            descriptors1=second.descriptors,
            matches=matches,
        )
