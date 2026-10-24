"""Run a backend over the pairs of a manifest."""

from collections.abc import Callable, Iterator, Sequence
from time import perf_counter_ns

import numpy as np
import numpy.typing as npt

from xfeat_core.features import Features
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
