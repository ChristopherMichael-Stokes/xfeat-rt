from pathlib import Path

import cv2
import numpy as np
import pytest
from input_store import load_or_prepare

from xfeat_pairs.preprocess import ResizeMode, prepare_image

MODEL_SIZE = (64, 48)


def _write_noise(path: Path, width: int, height: int, seed: int) -> None:
    rgb = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), rgb)


@pytest.mark.parametrize("mode", list(ResizeMode))
def test_stored_input_reads_back_bit_for_bit(tmp_path: Path, mode: ResizeMode) -> None:
    image = tmp_path / "noise.png"
    _write_noise(image, 97, 41, seed=0)
    store = tmp_path / "store"
    expected = prepare_image(image, MODEL_SIZE, mode)

    load_or_prepare(store, image, MODEL_SIZE, mode)
    image.unlink()  # a second read must come from the store, not the file
    stored = load_or_prepare(store, image, MODEL_SIZE, mode)

    np.testing.assert_array_equal(stored.pixels, expected.pixels)
    assert stored.pixels.dtype == expected.pixels.dtype
    assert stored.original_size == expected.original_size
    assert stored.scale == expected.scale
    assert stored.offset == expected.offset


def test_model_size_and_mode_are_separate_entries(tmp_path: Path) -> None:
    image = tmp_path / "noise.png"
    _write_noise(image, 97, 41, seed=1)
    store = tmp_path / "store"
    for size, mode in ((MODEL_SIZE, ResizeMode.STRETCH), ((32, 32), ResizeMode.LETTERBOX)):
        load_or_prepare(store, image, size, mode)

    letterboxed = load_or_prepare(store, image, (32, 32), ResizeMode.LETTERBOX)

    assert letterboxed.pixels.shape == (1, 1, 32, 32)
    assert letterboxed.offset == prepare_image(image, (32, 32), ResizeMode.LETTERBOX).offset
