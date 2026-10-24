"""`xfeat-ort-predict`: extract and match image pairs, writing one `.npz` per pair."""

import argparse
from pathlib import Path

import numpy as np

from xfeat_ort.models import LighterGlueModel, XFeatModel


def main() -> None:
    """Entry point of `xfeat-ort-predict`.

    Images are `.npy` float32 (1, 1, H, W) arrays in [0, 1], consumed two at a time as pairs.
    Each `pair{i}.npz` holds `{keypoints,scores,descriptors}{0,1}` of the valid features and
    `matches` (M, 2).
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--xfeat", type=Path, required=True)
    parser.add_argument("--lighterglue", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if len(args.images) % 2:
        parser.error("--images takes an even number of files (pairs)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xfeat = XFeatModel(args.xfeat, args.threads)
    lighterglue = LighterGlueModel(args.lighterglue, args.threads)
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
