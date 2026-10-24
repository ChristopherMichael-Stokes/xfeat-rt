"""Scoring on synthetic geometry: matches projected from 3-D points or a known homography."""

import dataclasses

import numpy as np
import numpy.typing as npt
import pytest

from xfeat_evaluation.agreement import pair_agreement
from xfeat_evaluation.scoring import score_homographies, score_poses
from xfeat_pairs.manifest import PairRecord, RelativePose
from xfeat_pairs.predictions import PairPrediction

SIZE = (640, 480)
FOCAL = 800.0
INTRINSICS = ((FOCAL, 0.0, 320.0), (0.0, FOCAL, 240.0), (0.0, 0.0, 1.0))
POINT_COUNT = 300
LONG_SIDE = 640


def _prediction(
    points0: npt.NDArray[np.float64], points1: npt.NDArray[np.float64]
) -> PairPrediction:
    count = len(points0)
    indices = np.arange(count, dtype=np.int64)
    return PairPrediction(
        keypoints0=points0.astype(np.float32),
        scores0=np.ones(count, np.float32),
        descriptors0=np.zeros((count, 64), np.float32),
        keypoints1=points1.astype(np.float32),
        scores1=np.ones(count, np.float32),
        descriptors1=np.zeros((count, 64), np.float32),
        matches=np.stack([indices, indices], axis=1),
        original_size0=SIZE,
        original_size1=SIZE,
        extract_ms=0.0,
        match_ms=0.0,
    )


def _record(name: str, truth: dict[str, object] | None) -> PairRecord:
    return PairRecord.model_validate(
        {"name": name, "image0": "a.png", "image1": "b.png", "ground_truth": truth}
    )


def _rotation_y(degrees: float) -> npt.NDArray[np.float64]:
    angle = np.radians(degrees)
    return np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )


def _project_centred(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Pixel coordinates with pixel centres at integers: COLMAP u minus 1/2."""
    u = FOCAL * points[:, 0] / points[:, 2] + INTRINSICS[0][2]
    v = FOCAL * points[:, 1] / points[:, 2] + INTRINSICS[1][2]
    return np.stack([u, v], axis=1) - 0.5


def _pose_pair(seed: int) -> tuple[PairRecord, PairPrediction]:
    rng = np.random.default_rng(seed)
    rotation = _rotation_y(5.0)
    translation = np.array([0.5, 0.05, 0.1])
    points = np.column_stack(
        [
            rng.uniform(-2, 2, POINT_COUNT),
            rng.uniform(-1.5, 1.5, POINT_COUNT),
            rng.uniform(4, 10, POINT_COUNT),
        ]
    )
    truth = {
        "kind": "relative_pose",
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "intrinsics0": INTRINSICS,
        "intrinsics1": INTRINSICS,
    }
    second = points @ rotation.T + translation
    return _record(f"pose{seed}", truth), _prediction(
        _project_centred(points), _project_centred(second)
    )


def test_exact_matches_give_full_pose_auc() -> None:
    pairs = [_pose_pair(seed) for seed in range(4)]
    scores = score_poses(pairs, LONG_SIDE)
    assert scores is not None
    assert scores["pairs"] == 4
    # noise-free projections of a valid configuration: errors far below the 5 degree threshold
    assert scores["rel_pose_error@5°"] > 0.95


def test_wrong_pose_ground_truth_lowers_auc() -> None:
    record, prediction = _pose_pair(0)
    assert isinstance(record.ground_truth, RelativePose)
    wrong = record.ground_truth.model_dump() | {"rotation": _rotation_y(40.0).tolist()}
    scores = score_poses([(_record("wrong", wrong), prediction)], LONG_SIDE)
    assert scores is not None
    assert scores["rel_pose_error@5°"] < 0.1


def test_homography_corner_error_vanishes_for_exact_matches() -> None:
    matrix = np.array([[1.02, 0.03, 12.0], [-0.02, 0.98, -7.0], [1e-5, 2e-5, 1.0]])
    rng = np.random.default_rng(0)
    points = rng.uniform([0, 0], SIZE, (POINT_COUNT, 2))
    homogeneous = np.column_stack([points, np.ones(POINT_COUNT)]) @ matrix.T
    warped = homogeneous[:, :2] / homogeneous[:, 2:]
    truth = {"kind": "homography", "matrix": matrix.tolist()}
    scores = score_homographies([(_record("h", truth), _prediction(points, warped))], LONG_SIDE)
    assert scores is not None
    assert scores["H_error_ransac@1px"] > 0.95


def test_pairs_without_ground_truth_are_not_scored() -> None:
    _, prediction = _pose_pair(0)
    bare = _record("bare", None)
    assert score_poses([(bare, prediction)], LONG_SIDE) is None
    assert score_homographies([(bare, prediction)], LONG_SIDE) is None


def test_agreement_is_full_for_identical_runs_and_zero_when_shifted_beyond_tolerance() -> None:
    _, prediction = _pose_pair(1)
    same = pair_agreement(prediction, prediction, long_side=LONG_SIDE, tolerance=1.0)
    assert (same.keypoint_recall, same.keypoint_precision, same.match_jaccard) == (1.0, 1.0, 1.0)

    shifted = dataclasses.replace(
        prediction, keypoints0=prediction.keypoints0 + 50, keypoints1=prediction.keypoints1 + 50
    )
    apart = pair_agreement(prediction, shifted, long_side=LONG_SIDE, tolerance=1.0)
    assert apart.keypoint_recall < 0.05
    assert apart.match_jaccard == pytest.approx(0.0, abs=0.05)


def test_two_runs_without_matches_agree() -> None:
    _, prediction = _pose_pair(2)
    empty = dataclasses.replace(prediction, matches=np.zeros((0, 2), dtype=np.int64))
    agreement = pair_agreement(empty, empty, long_side=LONG_SIDE, tolerance=1.0)
    assert agreement.match_jaccard == 1.0
