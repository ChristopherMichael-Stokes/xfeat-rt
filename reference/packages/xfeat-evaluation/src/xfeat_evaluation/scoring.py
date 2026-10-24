"""Score prediction directories against ground truth with GlueFactory's estimators and metrics.

Scoring happens at a common resolution: original-image keypoints and ground truth are scaled so
the long image side is `long_side` pixels, because RANSAC thresholds and corner errors are in
pixels of the resolution they are computed at. Pose AUC follows GlueFactory's MegaDepth-1500
evaluation (poselib relative pose, thresholds swept, best by mean AUC); homography accuracy follows
its HPatches evaluation.

Conventions: predictions hold original-image pixel coordinates with pixel centres at integers. The
pose estimator takes COLMAP-style coordinates (image corner at 0, so `x' = (x + 1/2) s`) with
intrinsics scaled by `s`; the homography is mapped through `x' = (x + 1/2) s - 1/2`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from gluefactory.eval.utils import eval_poses
from gluefactory.geometry.epipolar import relative_pose_error
from gluefactory.geometry.homography import homography_corner_error
from gluefactory.geometry.wrappers import Camera
from gluefactory.robust_estimators import load_estimator

from xfeat_evaluation.megadepth1500 import RANSAC_THRESHOLDS_PX
from xfeat_pairs.manifest import Homography, PairRecord, RelativePose
from xfeat_pairs.predictions import PairPrediction

POSE_AUC_THRESHOLDS_DEG = (5, 10, 20)
"""GlueFactory's MegaDepth-1500 thresholds."""
HOMOGRAPHY_AUC_THRESHOLDS_PX = (1, 3, 5)
"""GlueFactory's HPatches thresholds."""
MIN_POINTS_FOR_HOMOGRAPHY = 4
MIN_POINTS_FOR_RELATIVE_POSE = 5


@dataclass(frozen=True)
class MatchedPoints:
    """Matched keypoints of one pair at the scoring resolution.

    Attributes:
        points0, points1: (M, 2) float32 coordinates in the image-0 and image-1 frames.
        scale0, scale1: Ratio of the scoring resolution to the original image.
    """

    points0: npt.NDArray[np.float32]
    points1: npt.NDArray[np.float32]
    scale0: float
    scale1: float


def _long_side_scale(size: tuple[int, int], long_side: int) -> float:
    return long_side / max(size)


def matched_points(
    prediction: PairPrediction, long_side: int, *, corner_origin: bool
) -> MatchedPoints:
    """Matched keypoints scaled to `long_side`; `corner_origin` selects COLMAP coordinates."""
    scales = (
        _long_side_scale(prediction.original_size0, long_side),
        _long_side_scale(prediction.original_size1, long_side),
    )
    centre_shift = 0.0 if corner_origin else -0.5

    def scaled(keypoints: npt.NDArray[np.float32], scale: float) -> npt.NDArray[np.float32]:
        return ((keypoints + 0.5) * scale + centre_shift).astype(np.float32)

    first = scaled(prediction.keypoints0[prediction.matches[:, 0]], scales[0])
    second = scaled(prediction.keypoints1[prediction.matches[:, 1]], scales[1])
    return MatchedPoints(first, second, *scales)


def _camera(
    intrinsics: tuple[tuple[float, ...], ...], size: tuple[int, int], scale: float
) -> Camera:
    focal_x, focal_y = intrinsics[0][0], intrinsics[1][1]
    centre_x, centre_y = intrinsics[0][2], intrinsics[1][2]
    camera = Camera.from_colmap(
        {
            "model": "PINHOLE",
            "width": size[0] * scale,
            "height": size[1] * scale,
            "params": np.array([focal_x, focal_y, centre_x, centre_y]) * scale,
        }
    )
    if not isinstance(camera, Camera):  # GlueFactory is untyped; narrows its Unknown
        raise TypeError(camera)
    return camera


def pose_error(
    prediction: PairPrediction, truth: RelativePose, long_side: int, ransac_threshold: float
) -> float:
    """Larger of the rotation and translation-direction errors in degrees; inf on failure."""
    points = matched_points(prediction, long_side, corner_origin=True)
    if len(points.points0) < MIN_POINTS_FOR_RELATIVE_POSE:
        return float("inf")
    estimator = load_estimator("relative_pose", "poselib")(
        {"estimator": "poselib", "ransac_th": ransac_threshold}
    )
    estimate = estimator(
        {
            "m_kpts0": torch.from_numpy(points.points0),
            "m_kpts1": torch.from_numpy(points.points1),
            "camera0": _camera(truth.intrinsics0, prediction.original_size0, points.scale0),
            "camera1": _camera(truth.intrinsics1, prediction.original_size1, points.scale1),
        }
    )
    if not estimate["success"]:
        return float("inf")
    ground_truth = torch.eye(4)
    ground_truth[:3, :3] = torch.tensor(truth.rotation, dtype=torch.float32)
    ground_truth[:3, 3] = torch.tensor(truth.translation, dtype=torch.float32)
    translation_error, rotation_error = relative_pose_error(
        ground_truth, estimate["M_0to1"].R, estimate["M_0to1"].t
    )
    return float(max(translation_error, rotation_error))


def _resize_matrix(scale: float) -> torch.Tensor:
    """Homogeneous map of pixel-centre coordinates, `x' = (x + 1/2) s - 1/2`."""
    shift = 0.5 * scale - 0.5
    return torch.tensor([[scale, 0, shift], [0, scale, shift], [0, 0, 1]], dtype=torch.float32)


def homography_error(
    prediction: PairPrediction, truth: Homography, long_side: int, ransac_threshold: float
) -> float:
    """Mean corner error in pixels at the scoring resolution; inf on failure."""
    points = matched_points(prediction, long_side, corner_origin=False)
    if len(points.points0) < MIN_POINTS_FOR_HOMOGRAPHY:
        return float("inf")
    estimator = load_estimator("homography", "poselib")(
        {"estimator": "poselib", "ransac_th": ransac_threshold}
    )
    estimate = estimator(
        {
            "m_kpts0": torch.from_numpy(points.points0),
            "m_kpts1": torch.from_numpy(points.points1),
        }
    )
    if not estimate["success"]:
        return float("inf")
    ground_truth = (
        _resize_matrix(points.scale1)
        @ torch.tensor(truth.matrix, dtype=torch.float32)
        @ torch.linalg.inv(_resize_matrix(points.scale0))
    )
    size = torch.tensor(
        [prediction.original_size0[0] * points.scale0, prediction.original_size0[1] * points.scale0]
    )
    return float(homography_corner_error(estimate["M_0to1"], ground_truth, size))


def _auc_summary(
    errors: dict[float, list[float]], thresholds: Sequence[int], key: str, unit: str
) -> dict[str, float]:
    summaries, best_threshold = eval_poses(
        {threshold: {key: values} for threshold, values in errors.items()},
        list(thresholds),
        key,
        unit,
    )
    result: dict[str, float] = {str(name): float(value) for name, value in summaries.items()}
    result["best_ransac_threshold_px"] = float(best_threshold)
    return result


def score_poses(
    pairs: Sequence[tuple[PairRecord, PairPrediction]], long_side: int
) -> dict[str, float] | None:
    """Pose AUC over the pairs with relative-pose ground truth; None if there are none."""
    errors: dict[float, list[float]] = {threshold: [] for threshold in RANSAC_THRESHOLDS_PX}
    count = 0
    for record, prediction in pairs:
        if not isinstance(record.ground_truth, RelativePose):
            continue
        count += 1
        for threshold in RANSAC_THRESHOLDS_PX:
            errors[threshold].append(
                pose_error(prediction, record.ground_truth, long_side, threshold)
            )
    if count == 0:
        return None
    return {
        "pairs": float(count),
        **_auc_summary(errors, POSE_AUC_THRESHOLDS_DEG, "rel_pose_error", "°"),
    }


def score_homographies(
    pairs: Sequence[tuple[PairRecord, PairPrediction]], long_side: int
) -> dict[str, float] | None:
    """Corner-error AUC over the pairs with homography ground truth; None if there are none."""
    errors: dict[float, list[float]] = {threshold: [] for threshold in RANSAC_THRESHOLDS_PX}
    count = 0
    for record, prediction in pairs:
        if not isinstance(record.ground_truth, Homography):
            continue
        count += 1
        for threshold in RANSAC_THRESHOLDS_PX:
            errors[threshold].append(
                homography_error(prediction, record.ground_truth, long_side, threshold)
            )
    if count == 0:
        return None
    return {
        "pairs": float(count),
        **_auc_summary(errors, HOMOGRAPHY_AUC_THRESHOLDS_PX, "H_error_ransac", "px"),
    }
