"""Prediction directory: what a backend writes and `xfeat-evaluation` scores.

Layout::

    index.json            PredictionIndex (written last: its presence marks a complete run)
    pairs/000000.npz      one PairPrediction per manifest pair, in manifest order

Keypoints are in pixel coordinates of the original images, pixel centres at integers.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from xfeat_pairs.manifest import PairRecord
from xfeat_pairs.preprocess import ResizeMode

FORMAT_VERSION = 1
INDEX_NAME = "index.json"
PAIRS_DIRECTORY = "pairs"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PairEntry(_Frozen):
    """Index record of one pair."""

    name: str
    file: str
    image0: Path
    image1: Path
    original_size0: tuple[int, int]
    original_size1: tuple[int, int]
    extract_ms: float
    match_ms: float


class PredictionIndex(_Frozen):
    """Provenance and per-pair records of a run.

    Attributes:
        backend: Name and version of the runtime that produced the run.
        resize_mode: Fit of the images to the model input.
        model_size: Model input (width, height).
        xfeat: Raw `metadata_props` of the XFeat graph.
        lighterglue: Raw `metadata_props` of the LighterGlue graph.
        settings: Backend settings that affect results (precision, options).
        pairs: Entries in manifest order; `extract_ms` and `match_ms` are the wall time of feature
            extraction of both images and of matching, including host-device transfers.
    """

    format_version: Literal[1]
    backend: str
    resize_mode: ResizeMode
    model_size: tuple[int, int]
    xfeat: dict[str, str]
    lighterglue: dict[str, str]
    settings: dict[str, str]
    pairs: list[PairEntry]


@dataclass(frozen=True)
class PairPrediction:
    """Features and matches of one pair.

    Attributes:
        keypoints0, keypoints1: (N, 2) float32 original-image pixel coordinates.
        scores0, scores1: (N,) float32.
        descriptors0, descriptors1: (N, 64) float32.
        matches: (M, 2) int64 indices into the keypoints of image 0 and image 1.
        original_size0, original_size1: (width, height) of the source images.
        extract_ms, match_ms: See `PredictionIndex`.
    """

    keypoints0: npt.NDArray[np.float32]
    scores0: npt.NDArray[np.float32]
    descriptors0: npt.NDArray[np.float32]
    keypoints1: npt.NDArray[np.float32]
    scores1: npt.NDArray[np.float32]
    descriptors1: npt.NDArray[np.float32]
    matches: npt.NDArray[np.int64]
    original_size0: tuple[int, int]
    original_size1: tuple[int, int]
    extract_ms: float
    match_ms: float


@dataclass(frozen=True)
class RunHeader:
    """Run-level fields of the index, known before the first pair."""

    backend: str
    resize_mode: ResizeMode
    model_size: tuple[int, int]
    xfeat: dict[str, str]
    lighterglue: dict[str, str]
    settings: dict[str, str]


_ARRAY_FIELDS = (
    "keypoints0",
    "scores0",
    "descriptors0",
    "keypoints1",
    "scores1",
    "descriptors1",
    "matches",
)


def write_predictions(
    directory: Path,
    header: RunHeader,
    results: Iterable[tuple[PairRecord, PairPrediction]],
) -> Path:
    """Write `results` to `directory` as they arrive; returns the index path.

    Raises:
        FileExistsError: If `directory` already holds an index.
    """
    index_path = directory / INDEX_NAME
    if index_path.exists():
        msg = f"{index_path} exists; predictions are not overwritten"
        raise FileExistsError(msg)
    (directory / PAIRS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    entries = []
    for number, (record, prediction) in enumerate(results):
        file = f"{PAIRS_DIRECTORY}/{number:06d}.npz"
        np.savez(directory / file, **{name: getattr(prediction, name) for name in _ARRAY_FIELDS})
        entries.append(
            PairEntry(
                name=record.name,
                file=file,
                image0=record.image0,
                image1=record.image1,
                original_size0=prediction.original_size0,
                original_size1=prediction.original_size1,
                extract_ms=prediction.extract_ms,
                match_ms=prediction.match_ms,
            )
        )
    index = PredictionIndex(
        format_version=FORMAT_VERSION,
        backend=header.backend,
        resize_mode=header.resize_mode,
        model_size=header.model_size,
        xfeat=header.xfeat,
        lighterglue=header.lighterglue,
        settings=header.settings,
        pairs=entries,
    )
    index_path.write_text(index.model_dump_json(indent=1))
    return index_path


def read_index(directory: Path) -> PredictionIndex:
    """The index of the run in `directory`."""
    return PredictionIndex.model_validate_json((directory / INDEX_NAME).read_text())


def read_pair(directory: Path, entry: PairEntry) -> PairPrediction:
    """The arrays of one pair of the run in `directory`."""
    with np.load(directory / entry.file) as archive:
        arrays = {name: archive[name] for name in _ARRAY_FIELDS}
    return PairPrediction(
        keypoints0=arrays["keypoints0"],
        scores0=arrays["scores0"],
        descriptors0=arrays["descriptors0"],
        keypoints1=arrays["keypoints1"],
        scores1=arrays["scores1"],
        descriptors1=arrays["descriptors1"],
        matches=arrays["matches"],
        original_size0=entry.original_size0,
        original_size1=entry.original_size1,
        extract_ms=entry.extract_ms,
        match_ms=entry.match_ms,
    )
