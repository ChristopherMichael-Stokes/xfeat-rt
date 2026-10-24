from pathlib import Path

import cv2
import numpy as np
import pytest

from xfeat_pairs.preprocess import (
    ImageReadError,
    ResizeMode,
    prepare_image,
    to_original_coordinates,
)


def _write_rgb(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = rgb
    assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def test_gray_is_the_channel_mean_in_unit_range(tmp_path: Path) -> None:
    path = tmp_path / "red.png"
    _write_rgb(path, 8, 6, (255, 0, 0))
    prepared = prepare_image(path, (8, 6), ResizeMode.STRETCH)
    assert prepared.pixels.shape == (1, 1, 6, 8)
    np.testing.assert_allclose(prepared.pixels, 1 / 3, atol=1e-6)


def test_letterbox_centres_content_and_leaves_margin_empty(tmp_path: Path) -> None:
    path = tmp_path / "wide.png"
    _write_rgb(path, 200, 100, (102, 102, 102))
    prepared = prepare_image(path, (100, 100), ResizeMode.LETTERBOX)
    assert prepared.scale == (0.5, 0.5)
    assert prepared.offset == (0, 25)
    canvas = prepared.pixels[0, 0]
    assert canvas[:25].max() == 0
    assert canvas[75:].max() == 0
    np.testing.assert_allclose(canvas[25:75], 102 / 255, atol=1e-6)


@pytest.mark.parametrize("mode", list(ResizeMode))
def test_keypoint_mapping_inverts_the_resize(tmp_path: Path, mode: ResizeMode) -> None:
    path = tmp_path / "image.png"
    _write_rgb(path, 300, 200, (10, 20, 30))
    model_size = (128, 96)
    prepared = prepare_image(path, model_size, mode)

    original = np.array([[0, 0], [299, 199], [150, 100], [37, 181]], dtype=np.float32)
    # forward map of pixel centres: x_model = (x + 1/2) s - 1/2 + offset
    scale = np.array(prepared.scale)
    offset = np.array(prepared.offset)
    in_model = ((original + 0.5) * scale - 0.5 + offset).astype(np.float32)

    np.testing.assert_allclose(to_original_coordinates(in_model, prepared), original, atol=1e-4)


def test_unreadable_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "not_an_image.png"
    path.write_text("nope")
    with pytest.raises(ImageReadError):
        prepare_image(path, (8, 8), ResizeMode.STRETCH)
