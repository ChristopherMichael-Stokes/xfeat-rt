"""Result types and the NumPy glue between the XFeat and LighterGlue graphs."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DESCRIPTOR_DIMENSION = 64


@dataclass(frozen=True)
class Features:
    """Valid features of one image.

    Attributes:
        keypoints: (N, 2) float32 x, y in pixels of the model input, pixel centres at integers.
        scores: (N,) float32 keypoint scores, descending.
        descriptors: (N, 64) float32, L2-normalised.
        image_size: Model input (width, height) in pixels.
    """

    keypoints: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    descriptors: npt.NDArray[np.float32]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class PaddedFeatures:
    """One image's features padded to the matcher's K, as the LighterGlue graph takes them.

    Attributes:
        keypoints: (K, 2) float32; padded rows are zero.
        descriptors: (K, 64) float32; padded rows are zero.
        valid: (K,) bool.
        image_size: (2,) float32 (width, height).
    """

    keypoints: npt.NDArray[np.float32]
    descriptors: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]
    image_size: npt.NDArray[np.float32]


def compact_features(
    keypoints: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    descriptors: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
    image_size: tuple[int, int],
) -> Features:
    """Drop the invalid slots of one image's fixed-size XFeat outputs."""
    return Features(keypoints[valid], scores[valid], descriptors[valid], image_size)


def pad_features(features: Features, top_k: int) -> PaddedFeatures:
    """Pad one image's features to `top_k` rows for LighterGlue.

    Raises:
        ValueError: If the image has more than `top_k` features.
    """
    count = len(features.keypoints)
    if count > top_k:
        msg = f"{count} features exceed the matcher's K={top_k}"
        raise ValueError(msg)
    keypoints = np.zeros((top_k, 2), dtype=np.float32)
    descriptors = np.zeros((top_k, DESCRIPTOR_DIMENSION), dtype=np.float32)
    valid = np.zeros(top_k, dtype=np.bool_)
    keypoints[:count] = features.keypoints
    descriptors[:count] = features.descriptors
    valid[:count] = True
    return PaddedFeatures(
        keypoints, descriptors, valid, np.asarray(features.image_size, dtype=np.float32)
    )


def match_pairs(matches0: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """(M, 2) index pairs (image 0, image 1) from LighterGlue's `m0` (-1 = unmatched).

    Valid features come first in the padded layout, so padded and compact indices coincide.
    """
    matched = np.flatnonzero(matches0 >= 0)
    return np.stack([matched, matches0[matched]], axis=1).astype(np.int64)
