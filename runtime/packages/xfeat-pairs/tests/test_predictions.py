from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from xfeat_core.features import Features
from xfeat_pairs.manifest import PairRecord
from xfeat_pairs.predictions import RunHeader, read_index, read_pair, write_predictions
from xfeat_pairs.preprocess import ResizeMode
from xfeat_pairs.run import predict_pairs

MODEL_SIZE = (64, 48)
HEADER = RunHeader("fake 1.0", ResizeMode.STRETCH, MODEL_SIZE, {"model": "xfeat"}, {}, {})


def _record(tmp_path: Path, name: str) -> PairRecord:
    for side in ("0", "1"):
        image = np.full((96, 128, 3), 50, dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / f"{name}_{side}.png"), image)
    return PairRecord(
        name=name, image0=tmp_path / f"{name}_0.png", image1=tmp_path / f"{name}_1.png"
    )


def _extract(pixels: npt.NDArray[np.float32]) -> Features:
    assert pixels.shape == (1, 1, MODEL_SIZE[1], MODEL_SIZE[0])
    keypoints = np.array([[0, 0], [63, 47]], dtype=np.float32)
    return Features(keypoints, np.ones(2, np.float32), np.zeros((2, 64), np.float32), MODEL_SIZE)


def _match(_first: Features, _second: Features) -> npt.NDArray[np.int64]:
    return np.array([[0, 1], [1, 0]], dtype=np.int64)


def test_round_trip_in_original_coordinates(tmp_path: Path) -> None:
    records = [_record(tmp_path, "a"), _record(tmp_path, "b")]
    out = tmp_path / "run"
    results = predict_pairs(
        records,
        model_size=MODEL_SIZE,
        resize_mode=ResizeMode.STRETCH,
        extract=_extract,
        match=_match,
    )
    write_predictions(out, HEADER, results)

    index = read_index(out)
    assert [entry.name for entry in index.pairs] == ["a", "b"]
    prediction = read_pair(out, index.pairs[1])
    # model (0, 0) and (63, 47) in a 2x upscale to 128x96: x' = (x + 1/2) 2 - 1/2
    np.testing.assert_allclose(prediction.keypoints0, [[0.5, 0.5], [126.5, 94.5]])
    np.testing.assert_array_equal(prediction.matches, [[0, 1], [1, 0]])
    assert prediction.original_size0 == (128, 96)
    assert prediction.extract_ms >= 0
    assert index.model_size == MODEL_SIZE


def test_existing_run_is_not_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "run"
    write_predictions(out, HEADER, [])
    with pytest.raises(FileExistsError):
        write_predictions(out, HEADER, [])
