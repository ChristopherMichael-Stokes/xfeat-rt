"""Run a backend over the pairs of a manifest."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import numpy.typing as npt

from xfeat_core.features import DESCRIPTOR_DIMENSION, Features
from xfeat_pairs.manifest import PairRecord
from xfeat_pairs.predictions import PairPrediction
from xfeat_pairs.preprocess import PreparedImage, ResizeMode, prepare_image, to_original_coordinates

NANOSECONDS_PER_MILLISECOND = 1e6


def predict_pairs(
    records: Sequence[PairRecord],
    *,
    model_size: tuple[int, int],
    resize_mode: ResizeMode,
    extract: Callable[[npt.NDArray[np.float32]], Features],
    match: Callable[[Features, Features], npt.NDArray[np.int64]],
) -> Iterator[tuple[PairRecord, PairPrediction]]:
    """Extract and match each pair, reporting keypoints in original-image coordinates.

    `extract` takes a (1, 1, H, W) image; `match` returns (M, 2) index pairs. The reported times
    include everything the callables do, transfers and synchronisation among them.
    """
    for record in records:
        images = [
            prepare_image(path, model_size, resize_mode) for path in (record.image0, record.image1)
        ]
        start = perf_counter_ns()
        first, second = (extract(image.pixels) for image in images)
        extracted = perf_counter_ns()
        matches = match(first, second)
        matched = perf_counter_ns()
        yield (
            record,
            _prediction(
                images,
                (first, second),
                matches,
                (extracted - start) / NANOSECONDS_PER_MILLISECOND,
                (matched - extracted) / NANOSECONDS_PER_MILLISECOND,
            ),
        )


def predict_pairs_batched(  # noqa: PLR0913  # one callable and one batch size per model
    records: Sequence[PairRecord],
    *,
    load_image: Callable[[Path], PreparedImage],
    extract: Callable[[npt.NDArray[np.float32]], Sequence[Features]],
    extract_batch_size: int,
    match: Callable[[Sequence[tuple[Features, Features]]], Sequence[npt.NDArray[np.int64]]],
    match_batch_size: int,
) -> Iterator[tuple[PairRecord, PairPrediction]]:
    """`predict_pairs` with each unique image extracted once and both models run on batches.

    All unique images are extracted first, `extract_batch_size` per call, in order of first
    appearance; then the pairs are matched, `match_batch_size` per call, in manifest order. A
    final short batch is padded (zero images, feature sets with no keypoints) and the padded
    outputs are dropped. Features of every unique image stay in memory until the pairs are
    matched. Calls run serially on the calling thread.

    `load_image` returns the model input of a path; its time is not reported. `extract` takes
    (`extract_batch_size`, 1, H, W) images and returns one `Features` per image; `match` takes
    `match_batch_size` pairs and returns (M, 2) index pairs for each.

    `extract_ms` and `match_ms` are amortised shares of the batched calls: an extract call's time
    is split equally over its real images, and an image's share equally over the pairs that use
    it; a match call's time is split equally over its real pairs. Summed over all pairs they give
    the total time of the extract and match calls.

    Raises:
        ValueError: If a callable returns a number of results other than its batch size.
    """
    extracted = _extract_unique_images(records, load_image, extract, extract_batch_size)
    for chunk in batched(records, match_batch_size, strict=False):
        sides = [(extracted[record.image0], extracted[record.image1]) for record in chunk]
        pairs = [(first.features, second.features) for first, second in sides]
        matches, elapsed_ms = _timed(match, _padded_pairs(pairs, match_batch_size))
        _check_count(matches, match_batch_size, "match")
        for record, (first, second), pair_matches in zip(
            chunk, sides, matches[: len(chunk)], strict=True
        ):
            yield record, _batched_prediction(first, second, pair_matches, elapsed_ms / len(chunk))


@dataclass(frozen=True)
class _ExtractedImage:
    """What matching and the prediction need of one image, without its pixels."""

    features: Features
    original_keypoints: npt.NDArray[np.float32]
    original_size: tuple[int, int]
    extract_ms: float


def _extract_unique_images(
    records: Sequence[PairRecord],
    load_image: Callable[[Path], PreparedImage],
    extract: Callable[[npt.NDArray[np.float32]], Sequence[Features]],
    batch_size: int,
) -> dict[Path, _ExtractedImage]:
    uses = _image_uses(records)
    extracted = {}
    for paths in batched(uses, batch_size, strict=False):
        loaded = [load_image(path) for path in paths]
        batch, elapsed_ms = _timed(
            extract, _padded_images([image.pixels for image in loaded], batch_size)
        )
        _check_count(batch, batch_size, "extract")
        for path, image, image_features in zip(paths, loaded, batch[: len(paths)], strict=True):
            extracted[path] = _ExtractedImage(
                image_features,
                to_original_coordinates(image_features.keypoints, image),
                image.original_size,
                elapsed_ms / len(paths) / uses[path],
            )
    return extracted


def _batched_prediction(
    first: _ExtractedImage,
    second: _ExtractedImage,
    matches: npt.NDArray[np.int64],
    match_ms: float,
) -> PairPrediction:
    return PairPrediction(
        keypoints0=first.original_keypoints,
        scores0=first.features.scores,
        descriptors0=first.features.descriptors,
        keypoints1=second.original_keypoints,
        scores1=second.features.scores,
        descriptors1=second.features.descriptors,
        matches=matches,
        original_size0=first.original_size,
        original_size1=second.original_size,
        extract_ms=first.extract_ms + second.extract_ms,
        match_ms=match_ms,
    )


def _image_uses(records: Sequence[PairRecord]) -> dict[Path, int]:
    """Pairs using each image, keyed in order of first appearance."""
    uses: dict[Path, int] = {}
    for record in records:
        for path in (record.image0, record.image1):
            uses[path] = uses.get(path, 0) + 1
    return uses


def _timed[Batch, Result](call: Callable[[Batch], Result], batch: Batch) -> tuple[Result, float]:
    start = perf_counter_ns()
    result = call(batch)
    return result, (perf_counter_ns() - start) / NANOSECONDS_PER_MILLISECOND


def _check_count(results: Sequence[object], batch_size: int, name: str) -> None:
    if len(results) != batch_size:
        msg = f"{name} returned {len(results)} results for a batch of {batch_size}"
        raise ValueError(msg)


def _padded_images(
    pixels: Sequence[npt.NDArray[np.float32]], batch_size: int
) -> npt.NDArray[np.float32]:
    """(batch_size, 1, H, W) from (1, 1, H, W) images, zero images after the real ones."""
    batch = np.zeros((batch_size, *pixels[0].shape[1:]), dtype=np.float32)
    batch[: len(pixels)] = np.concatenate(pixels)
    return batch


def _padded_pairs(
    pairs: Sequence[tuple[Features, Features]], batch_size: int
) -> list[tuple[Features, Features]]:
    """`pairs` followed by pairs of empty feature sets up to `batch_size`."""
    empty = Features(
        np.zeros((0, 2), dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros((0, DESCRIPTOR_DIMENSION), dtype=np.float32),
        pairs[0][0].image_size,
    )
    return [*pairs, *[(empty, empty)] * (batch_size - len(pairs))]


def _prediction(
    images: Sequence[PreparedImage],
    features: tuple[Features, Features],
    matches: npt.NDArray[np.int64],
    extract_ms: float,
    match_ms: float,
) -> PairPrediction:
    return PairPrediction(
        keypoints0=to_original_coordinates(features[0].keypoints, images[0]),
        scores0=features[0].scores,
        descriptors0=features[0].descriptors,
        keypoints1=to_original_coordinates(features[1].keypoints, images[1]),
        scores1=features[1].scores,
        descriptors1=features[1].descriptors,
        matches=matches,
        original_size0=images[0].original_size,
        original_size1=images[1].original_size,
        extract_ms=extract_ms,
        match_ms=match_ms,
    )
