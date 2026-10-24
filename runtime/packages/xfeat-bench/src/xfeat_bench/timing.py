"""Latency measurement with raw samples, percentiles and run conditions."""

import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PERCENTILES = (50.0, 95.0, 99.0)
NANOSECONDS_PER_MILLISECOND = 1_000_000


@dataclass(frozen=True)
class RunConditions:
    """What the machine was doing during a measurement.

    Attributes:
        load_average: 1, 5 and 15 minute load averages after the run.
        logical_cpus: CPU count the OS reports.
        platform: `platform.platform()`.
        python: Interpreter version.
    """

    load_average: tuple[float, float, float]
    logical_cpus: int
    platform: str
    python: str


@dataclass(frozen=True)
class Measurement:
    """Latency samples of one callable.

    Attributes:
        name: Label of the measured operation.
        samples_ms: Per-call wall time in milliseconds, in call order.
        warmup: Calls made and discarded before timing.
        settings: Free-form description of the configuration (model, K, threads, ...).
        conditions: Machine state during the run.
    """

    name: str
    samples_ms: list[float]
    warmup: int
    settings: dict[str, str]
    conditions: RunConditions

    def percentile(self, percent: float) -> float:
        """Percentile of the raw samples in milliseconds (linear interpolation)."""
        return float(np.percentile(self.samples_ms, percent))

    def summary(self) -> dict[str, float]:
        """Percentiles `p50`, `p95`, `p99` of the samples in milliseconds."""
        return {f"p{percent:g}": self.percentile(percent) for percent in PERCENTILES}


def measure(
    name: str,
    call: Callable[[], object],
    *,
    warmup: int,
    runs: int,
    settings: dict[str, str],
) -> Measurement:
    """Time `call` for `runs` calls after `warmup` discarded calls.

    `call` must return only when its work is finished; a GPU backend synchronises inside it.
    """
    for _ in range(warmup):
        call()
    samples_ms = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        call()
        samples_ms.append((time.perf_counter_ns() - start) / NANOSECONDS_PER_MILLISECOND)
    conditions = RunConditions(
        load_average=os.getloadavg(),
        logical_cpus=os.cpu_count() or 0,
        platform=platform.platform(),
        python=platform.python_version(),
    )
    return Measurement(name, samples_ms, warmup, settings, conditions)


def write_results(measurements: list[Measurement], directory: Path) -> Path:
    """Write raw samples and summaries to a timestamped JSON file in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    records = [{**asdict(m), "summary_ms": m.summary()} for m in measurements]
    path.write_text(json.dumps(records, indent=1))
    return path
