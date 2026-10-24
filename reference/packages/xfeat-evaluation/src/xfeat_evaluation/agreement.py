"""Agreement between two prediction directories on one manifest (typically reference vs backend).

Keypoints are compared at the scoring resolution (long image side scaled to a common length), so
the tolerance does not depend on the image size. A keypoint of one run agrees with the nearest
keypoint of the other within the tolerance. Matches agree when both of their ends map to a
keypoint pair the other run also matched.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from xfeat_pairs.predictions import PairPrediction

UNMAPPED = -1


@dataclass(frozen=True)
class PairAgreement:
    """Agreement of run `a` with run `b` on one pair.

    Attributes:
        keypoint_recall: Fraction of `b`'s keypoints that have a keypoint of `a` within tolerance.
        keypoint_precision: Fraction of `a`'s keypoints that have one of `b`.
        match_jaccard: Jaccard index of the two runs' matches (1 if both are empty).
    """

    keypoint_recall: float
    keypoint_precision: float
    match_jaccard: float


def _nearest_within(
    source: npt.NDArray[np.float32], target: npt.NDArray[np.float32], tolerance: float
) -> npt.NDArray[np.int64]:
    """Index of each `source` point's nearest `target` point within `tolerance`, else -1."""
    if len(source) == 0 or len(target) == 0:
        return np.full(len(source), UNMAPPED, dtype=np.int64)
    first, second = source.astype(np.float64), target.astype(np.float64)
    squared = (
        (first**2).sum(axis=1)[:, None] + (second**2).sum(axis=1)[None, :] - 2 * first @ second.T
    )
    nearest = squared.argmin(axis=1)
    within = squared[np.arange(len(first)), nearest] <= tolerance**2
    return np.where(within, nearest, UNMAPPED).astype(np.int64)


def _side_scale(size: tuple[int, int], long_side: int) -> float:
    return long_side / max(size)


def pair_agreement(
    a: PairPrediction, b: PairPrediction, *, long_side: int, tolerance: float
) -> PairAgreement:
    """Agreement of `a` with `b`; `tolerance` is in pixels at the `long_side` resolution."""
    scale0 = _side_scale(a.original_size0, long_side)
    scale1 = _side_scale(a.original_size1, long_side)
    a0, a1 = a.keypoints0 * scale0, a.keypoints1 * scale1
    b0, b1 = b.keypoints0 * scale0, b.keypoints1 * scale1
    a_to_b0, a_to_b1 = _nearest_within(a0, b0, tolerance), _nearest_within(a1, b1, tolerance)
    b_to_a0, b_to_a1 = _nearest_within(b0, a0, tolerance), _nearest_within(b1, a1, tolerance)

    mapped_a = {
        (int(a_to_b0[i]), int(a_to_b1[j]))
        for i, j in a.matches
        if a_to_b0[i] != UNMAPPED and a_to_b1[j] != UNMAPPED
    }
    unmapped_count = len(a.matches) - len(mapped_a)
    b_matches = {(int(i), int(j)) for i, j in b.matches}
    union = len(mapped_a) + unmapped_count + len(b_matches) - len(mapped_a & b_matches)
    jaccard = len(mapped_a & b_matches) / union if union else 1.0

    hits_recall = (b_to_a0 != UNMAPPED).sum() + (b_to_a1 != UNMAPPED).sum()
    hits_precision = (a_to_b0 != UNMAPPED).sum() + (a_to_b1 != UNMAPPED).sum()
    return PairAgreement(
        keypoint_recall=float(hits_recall / max(len(b0) + len(b1), 1)),
        keypoint_precision=float(hits_precision / max(len(a0) + len(a1), 1)),
        match_jaccard=jaccard,
    )


def summarise(agreements: Sequence[PairAgreement]) -> dict[str, float]:
    """Mean and minimum of each agreement measure over the pairs."""
    summary: dict[str, float] = {"pairs": float(len(agreements))}
    for name in ("keypoint_recall", "keypoint_precision", "match_jaccard"):
        values = np.array([getattr(agreement, name) for agreement in agreements])
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_min"] = float(values.min())
    return summary
