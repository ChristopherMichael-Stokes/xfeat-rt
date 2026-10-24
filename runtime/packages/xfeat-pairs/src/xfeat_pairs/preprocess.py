"""Image loading and resizing to a graph's static input size, and keypoint mapping back.

Images are resized as 8-bit RGB with bilinear interpolation, then become grayscale by the mean of
the channels scaled to [0, 1]. That is the order of the reference pipeline; resizing after the
mean changes pixels by up to 1/255 and, through near-tied keypoints, the keypoint set.

`stretch` resizes to the model size ignoring aspect; `letterbox` keeps the aspect, centres the
image on a zero canvas and leaves the margin empty.

Keypoints are pixel coordinates with pixel centres at integers. A model-input point `x` maps to
the original image as `x_original = (x - offset + 1/2) / scale - 1/2` per axis, where `scale` is
the resized-to-original size ratio and `offset` the canvas margin.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

MAX_PIXEL_VALUE = 255.0


class ResizeMode(StrEnum):
    """How an image is fitted to the model input size."""

    STRETCH = "stretch"
    LETTERBOX = "letterbox"


@dataclass(frozen=True)
class PreparedImage:
    """An image fitted to the model input.

    Attributes:
        pixels: (1, 1, H, W) float32 gray image in [0, 1].
        original_size: (width, height) of the file.
        scale: Per-axis (x, y) ratio of the resized content to the original size.
        offset: Per-axis (x, y) margin in pixels between the canvas and the content.
    """

    pixels: npt.NDArray[np.float32]
    original_size: tuple[int, int]
    scale: tuple[float, float]
    offset: tuple[int, int]


class ImageReadError(OSError):
    """The file is not a readable image."""


def prepare_image(path: Path, model_size: tuple[int, int], mode: ResizeMode) -> PreparedImage:
    """Load `path` and fit it to `model_size` (width, height)."""
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)
    if rgb is None:
        msg = f"cannot read image {path}"
        raise ImageReadError(msg)
    height, width = rgb.shape[:2]
    model_width, model_height = model_size

    if mode is ResizeMode.STRETCH:
        content_size = model_size
    else:
        fit = min(model_width / width, model_height / height)
        content_size = (round(width * fit), round(height * fit))
    resized = cv2.resize(rgb, content_size, interpolation=cv2.INTER_LINEAR)
    content = resized.astype(np.float32).mean(axis=2) / MAX_PIXEL_VALUE

    offset = ((model_width - content_size[0]) // 2, (model_height - content_size[1]) // 2)
    canvas = np.zeros((model_height, model_width), dtype=np.float32)
    canvas[offset[1] : offset[1] + content_size[1], offset[0] : offset[0] + content_size[0]] = (
        content
    )
    scale = (content_size[0] / width, content_size[1] / height)
    return PreparedImage(canvas[None, None], (width, height), scale, offset)


def to_original_coordinates(
    keypoints: npt.NDArray[np.float32], image: PreparedImage
) -> npt.NDArray[np.float32]:
    """Map (N, 2) model-input keypoints to pixel coordinates of the original image."""
    scale = np.asarray(image.scale, dtype=np.float64)
    offset = np.asarray(image.offset, dtype=np.float64)
    return ((keypoints - offset + 0.5) / scale - 0.5).astype(np.float32)
