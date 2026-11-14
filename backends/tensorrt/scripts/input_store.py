"""Persisted model inputs: each image prepared once and read back on later runs.

Layout under the store directory::

    <width>x<height>_<mode>/<digest>.npy    pixels of `PreparedImage`, (1, 1, H, W) float32
    <width>x<height>_<mode>/<digest>.json   resize geometry; written last, marks a complete entry

`digest` is the SHA-256 of the image path as given. The key covers the path, the model size and
the resize mode only: a changed image file or a change to `prepare_image` is not detected, so a
store is cleared by hand when either changes.
"""

import hashlib
import io
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from xfeat_pairs.preprocess import PreparedImage, ResizeMode, prepare_image

PARTIAL_SUFFIX = ".partial"


class _Geometry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    image: Path
    original_size: tuple[int, int]
    scale: tuple[float, float]
    offset: tuple[int, int]


def _entry_stem(store: Path, image: Path, model_size: tuple[int, int], mode: ResizeMode) -> Path:
    digest = hashlib.sha256(str(image).encode()).hexdigest()
    return store / f"{model_size[0]}x{model_size[1]}_{mode}" / digest


def _write_atomically(path: Path, data: bytes) -> None:
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    partial.write_bytes(data)
    partial.replace(path)


def _save(stem: Path, image: Path, prepared: PreparedImage) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pixels = io.BytesIO()
    np.save(pixels, prepared.pixels)
    _write_atomically(stem.with_suffix(".npy"), pixels.getvalue())
    geometry = _Geometry(
        image=image,
        original_size=prepared.original_size,
        scale=prepared.scale,
        offset=prepared.offset,
    )
    _write_atomically(stem.with_suffix(".json"), geometry.model_dump_json().encode())


def _load(stem: Path) -> PreparedImage:
    geometry = _Geometry.model_validate_json(stem.with_suffix(".json").read_text())
    pixels = np.load(stem.with_suffix(".npy"))
    return PreparedImage(pixels, geometry.original_size, geometry.scale, geometry.offset)


def load_or_prepare(
    store: Path, image: Path, model_size: tuple[int, int], mode: ResizeMode
) -> PreparedImage:
    """`prepare_image(image, model_size, mode)`, read from `store` or prepared and saved there.

    A stored entry reads back equal bit for bit to the `prepare_image` result it was saved from.
    """
    stem = _entry_stem(store, image, model_size, mode)
    if stem.with_suffix(".json").exists():
        return _load(stem)
    prepared = prepare_image(image, model_size, mode)
    _save(stem, image, prepared)
    return prepared
