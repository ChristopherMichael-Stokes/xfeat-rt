import math
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from xfeat_core.features import DESCRIPTOR_DIMENSION, Features
from xfeat_pairs.manifest import PairRecord
from xfeat_pairs.predictions import PairPrediction
from xfeat_pairs.preprocess import PreparedImage, ResizeMode, prepare_image
from xfeat_pairs.run import predict_pairs, predict_pairs_batched

MODEL_SIZE = (40, 30)
MODE = ResizeMode.LETTERBOX
EXTRACT_BATCH = 2
MATCH_BATCH = 3
IMAGE_SIZES = ((80, 60), (50, 70), (64, 64), (31, 17), (90, 45))
PAIRS = ((0, 1), (1, 2), (0, 3), (4, 4), (3, 2), (2, 0), (4, 1))
"""Five unique images and seven pairs: neither count divides by its batch size."""


def _fake_extract(images: npt.NDArray[np.float32]) -> list[Features]:
    """Brightest pixels of each image as keypoints; depends on that image only."""
    features = []
    for image in images[:, 0]:
        flat = image.ravel()
        count = 1 + int(flat.mean() * 8)
        order = np.argsort(-flat, kind="stable")[:count]
        keypoints = np.stack([order % image.shape[1], order // image.shape[1]], axis=1)
        descriptors = np.outer(flat[order], np.arange(DESCRIPTOR_DIMENSION, dtype=np.float32))
        features.append(
            Features(
                keypoints.astype(np.float32),
                flat[order].astype(np.float32),
                descriptors.astype(np.float32),
                MODEL_SIZE,
            )
        )
    return features


def _fake_match(pairs: Sequence[tuple[Features, Features]]) -> list[npt.NDArray[np.int64]]:
    """Index i to i where keypoint i of image 0 lies right of image 1's; per pair only."""
    matches = []
    for first, second in pairs:
        shared = min(len(first.keypoints), len(second.keypoints))
        rows = np.flatnonzero(first.keypoints[:shared, 0] > second.keypoints[:shared, 0])
        matches.append(np.stack([rows, rows], axis=1).astype(np.int64))
    return matches


def _records(tmp_path: Path) -> list[PairRecord]:
    rng = np.random.default_rng(0)
    paths = []
    for number, (width, height) in enumerate(IMAGE_SIZES):
        path = tmp_path / f"{number}.png"
        assert cv2.imwrite(str(path), rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        paths.append(path)
    return [
        PairRecord(name=f"pair{number}", image0=paths[first], image1=paths[second])
        for number, (first, second) in enumerate(PAIRS)
    ]


def _load(path: Path) -> PreparedImage:
    return prepare_image(path, MODEL_SIZE, MODE)


def _assert_same_prediction(batched: PairPrediction, single: PairPrediction) -> None:
    for field in fields(PairPrediction):
        if field.name in {"extract_ms", "match_ms"}:
            continue
        np.testing.assert_array_equal(
            getattr(batched, field.name), getattr(single, field.name), err_msg=field.name
        )


def test_batched_predictions_equal_batch_one(tmp_path: Path) -> None:
    records = _records(tmp_path)
    single = list(
        predict_pairs(
            records,
            model_size=MODEL_SIZE,
            resize_mode=MODE,
            extract=lambda pixels: _fake_extract(pixels)[0],
            match=lambda first, second: _fake_match([(first, second)])[0],
        )
    )

    batched = list(
        predict_pairs_batched(
            records,
            load_image=_load,
            extract=_fake_extract,
            extract_batch_size=EXTRACT_BATCH,
            match=_fake_match,
            match_batch_size=MATCH_BATCH,
        )
    )

    assert [record.name for record, _ in batched] == [record.name for record in records]
    for (_, batched_prediction), (_, single_prediction) in zip(batched, single, strict=True):
        _assert_same_prediction(batched_prediction, single_prediction)


def test_each_image_is_extracted_once_in_full_padded_batches(tmp_path: Path) -> None:
    records = _records(tmp_path)
    extract_batches: list[npt.NDArray[np.float32]] = []
    match_batches: list[Sequence[tuple[Features, Features]]] = []
    loaded: list[Path] = []

    def extract(images: npt.NDArray[np.float32]) -> list[Features]:
        extract_batches.append(images.copy())
        return _fake_extract(images)

    def match(pairs: Sequence[tuple[Features, Features]]) -> list[npt.NDArray[np.int64]]:
        match_batches.append(pairs)
        return _fake_match(pairs)

    def load(path: Path) -> PreparedImage:
        loaded.append(path)
        return _load(path)

    list(
        predict_pairs_batched(
            records,
            load_image=load,
            extract=extract,
            extract_batch_size=EXTRACT_BATCH,
            match=match,
            match_batch_size=MATCH_BATCH,
        )
    )

    unique = len(IMAGE_SIZES)
    assert sorted(loaded) == sorted({path for r in records for path in (r.image0, r.image1)})
    assert len(extract_batches) == math.ceil(unique / EXTRACT_BATCH)
    assert all(batch.shape == (EXTRACT_BATCH, 1, 30, 40) for batch in extract_batches)
    padded_images = EXTRACT_BATCH * len(extract_batches) - unique
    assert not extract_batches[-1][EXTRACT_BATCH - padded_images :].any()
    assert len(match_batches) == math.ceil(len(PAIRS) / MATCH_BATCH)
    assert all(len(batch) == MATCH_BATCH for batch in match_batches)
    padded_pairs = MATCH_BATCH * len(match_batches) - len(PAIRS)
    for first, second in match_batches[-1][MATCH_BATCH - padded_pairs :]:
        assert len(first.keypoints) == len(second.keypoints) == 0
