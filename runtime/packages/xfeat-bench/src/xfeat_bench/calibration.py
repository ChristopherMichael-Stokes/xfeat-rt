"""Batch-size calibration: choosing a batch size from measured throughput, and its stored result.

A backend sweeps the batch sizes of an exported ladder in increasing order, measuring repeated
end-to-end throughput (items per second) for each, and stops early with `throughput_fell_twice`.
`select_batch_size` picks from the trials; `save_calibration` and `load_calibration` keep the
result in a directory, one JSON file per key. The key is a string map the backend composes from
whatever invalidates a measurement (device, driver, runtime version, build options, model hashes).
"""

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt

FALLS_TO_STOP = 2
"""Consecutive throughput falls that end a sweep; one fall can be noise."""
KEY_DIGEST_CHARACTERS = 16
"""Hex characters of the key digest in a file name; 64 bits make collisions negligible."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MeasuredBatchSize(_Frozen):
    """A batch size that ran.

    Attributes:
        batch_size: Items per call.
        items_per_second: Throughput of each repeat.
        build_seconds: Wall time to build and load the engine, or None when it came from a
            cache.
    """

    kind: Literal["measured"] = "measured"
    batch_size: PositiveInt
    items_per_second: tuple[PositiveFloat, ...] = Field(min_length=1)
    build_seconds: NonNegativeFloat | None

    @property
    def median(self) -> float:
        """Median throughput over the repeats, items per second."""
        return statistics.median(self.items_per_second)

    @property
    def spread(self) -> float:
        """Range (max - min) of the repeats, items per second."""
        return max(self.items_per_second) - min(self.items_per_second)


class FailedBatchSize(_Frozen):
    """A batch size whose engine could not be built or whose buffers could not be allocated."""

    kind: Literal["failed"] = "failed"
    batch_size: PositiveInt
    reason: str


BatchSizeTrial = Annotated[MeasuredBatchSize | FailedBatchSize, Field(discriminator="kind")]


class NoBatchSizeError(ValueError):
    """No batch size of a sweep ran."""


def select_batch_size(trials: Sequence[MeasuredBatchSize | FailedBatchSize]) -> int:
    """Smallest batch size whose median throughput is within noise of the best median.

    The noise band is the spread (max - min) of the repeats of the batch size with the best
    median: a batch size qualifies when `median >= best_median - best_spread`. The band only keeps
    measurement noise from deciding between equivalent batch sizes; it gives no weight to latency,
    and a larger batch inside the band is not preferred even though it costs more latency per item.

    Raises:
        NoBatchSizeError: If no trial was measured.
    """
    measured = [trial for trial in trials if isinstance(trial, MeasuredBatchSize)]
    if not measured:
        msg = "no batch size ran"
        raise NoBatchSizeError(msg)
    best = max(measured, key=lambda trial: trial.median)
    floor = best.median - best.spread
    return min(trial.batch_size for trial in measured if trial.median >= floor)


def throughput_fell_twice(trials: Sequence[MeasuredBatchSize | FailedBatchSize]) -> bool:
    """Whether the median throughput fell at each of the last two measured batch sizes."""
    medians = [trial.median for trial in trials if isinstance(trial, MeasuredBatchSize)]
    last = medians[-(FALLS_TO_STOP + 1) :]
    return len(last) > FALLS_TO_STOP and all(a > b for a, b in pairwise(last))


class ModelCalibration(_Frozen):
    """Sweep of one model, trials in increasing batch size."""

    trials: tuple[BatchSizeTrial, ...] = Field(min_length=1)

    @property
    def batch_size(self) -> int:
        """`select_batch_size` of the trials."""
        return select_batch_size(self.trials)


class Calibration(_Frozen):
    """Batch-size sweeps of XFeat and LighterGlue under one key."""

    key: dict[str, str]
    xfeat: ModelCalibration
    lighterglue: ModelCalibration


def _calibration_path(directory: Path, key: Mapping[str, str]) -> Path:
    canonical = json.dumps(dict(key), sort_keys=True).encode()
    return directory / f"{hashlib.sha256(canonical).hexdigest()[:KEY_DIGEST_CHARACTERS]}.json"


def load_calibration(directory: Path, key: Mapping[str, str]) -> Calibration | None:
    """The calibration stored in `directory` under `key`, or None if there is none."""
    path = _calibration_path(directory, key)
    if not path.exists():
        return None
    calibration = Calibration.model_validate_json(path.read_text())
    return calibration if calibration.key == dict(key) else None


def save_calibration(directory: Path, calibration: Calibration) -> Path:
    """Store `calibration` in `directory` under its key; returns the file path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = _calibration_path(directory, calibration.key)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(calibration.model_dump_json(indent=1))
    partial.replace(path)
    return path
