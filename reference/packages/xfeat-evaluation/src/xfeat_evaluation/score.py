"""`xfeat-score`: score a prediction directory against its manifest's ground truth.

Pose AUC for pairs with relative-pose ground truth, corner-error AUC for pairs with a homography,
and with `--reference` the agreement with a second prediction directory (pairs without ground
truth take part in that). Latency columns come from the per-pair times in the index.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from xfeat_evaluation.agreement import pair_agreement, summarise
from xfeat_evaluation.scoring import score_homographies, score_poses
from xfeat_pairs.manifest import PairRecord, load_manifest
from xfeat_pairs.predictions import PairPrediction, read_index, read_pair

README_LONG_SIDE = 640
"""Long side of the XFeat README's MegaDepth-1500 evaluation (GlueFactory default)."""
AGREEMENT_TOLERANCE_PX = 1.0


def _load(directory: Path, records: list[PairRecord]) -> list[tuple[PairRecord, PairPrediction]]:
    index = read_index(directory)
    entries = {entry.name: entry for entry in index.pairs}
    missing = [record.name for record in records if record.name not in entries]
    if missing:
        msg = f"{directory} has no prediction for pairs {missing[:5]}"
        raise ValueError(msg)
    return [(record, read_pair(directory, entries[record.name])) for record in records]


def _timing(pairs: list[tuple[PairRecord, PairPrediction]]) -> dict[str, float]:
    extract = np.array([prediction.extract_ms for _, prediction in pairs])
    match = np.array([prediction.match_ms for _, prediction in pairs])
    return {
        "extract_ms_p50": float(np.percentile(extract, 50)),
        "match_ms_p50": float(np.percentile(match, 50)),
        "keypoints_mean": float(np.mean([len(p.keypoints0) + len(p.keypoints1) for _, p in pairs]))
        / 2,
        "matches_mean": float(np.mean([len(p.matches) for _, p in pairs])),
    }


def main() -> None:
    """Entry point of `xfeat-score`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="prediction directory to compare against")
    parser.add_argument("--resize", type=int, default=README_LONG_SIDE, help="scoring long side")
    parser.add_argument("--agreement-tolerance", type=float, default=AGREEMENT_TOLERANCE_PX)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    pairs = _load(args.predictions, records)
    index = read_index(args.predictions)
    result: dict[str, object] = {
        "predictions": str(args.predictions),
        "backend": index.backend,
        "settings": index.settings,
        "model_size": index.model_size,
        "resize_mode": index.resize_mode.value,
        "scoring_long_side": args.resize,
        "timing": _timing(pairs),
        "pose": score_poses(pairs, args.resize),
        "homography": score_homographies(pairs, args.resize),
    }
    if args.reference is not None:
        reference = _load(args.reference, records)
        result["reference"] = str(args.reference)
        result["agreement"] = summarise(
            [
                pair_agreement(
                    prediction,
                    oracle,
                    long_side=args.resize,
                    tolerance=args.agreement_tolerance,
                )
                for (_, prediction), (_, oracle) in zip(pairs, reference, strict=True)
            ]
        )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))  # noqa: T201  # CLI output
