"""Latency and throughput measurement with raw samples, percentiles and run conditions."""

import json
import math
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
NANOSECONDS_PER_SECOND = 1_000_000_000


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


def current_conditions() -> RunConditions:
    """The machine state right now."""
    return RunConditions(
        load_average=os.getloadavg(),
        logical_cpus=os.cpu_count() or 0,
        platform=platform.platform(),
        python=platform.python_version(),
    )


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
    return Measurement(name, samples_ms, warmup, settings, current_conditions())


def measure_throughput(
    call: Callable[[], object],
    *,
    items_per_call: int,
    warmup_seconds: float,
    repeats: int,
    repeat_seconds: float,
) -> tuple[float, ...]:
    """Items per second of `call`, one value per repeat.

    Calls are made and discarded for `warmup_seconds` first (at least one), since GPU clocks ramp
    up under load; their mean time sizes each repeat to about `repeat_seconds` (at least one
    call). `call` must return only when its work is finished.
    """
    warmup_ns = warmup_seconds * NANOSECONDS_PER_SECOND
    warmup_calls = 0
    start = time.perf_counter_ns()
    while True:
        call()
        warmup_calls += 1
        warmup_elapsed_ns = time.perf_counter_ns() - start
        if warmup_elapsed_ns >= warmup_ns:
            break
    mean_call_ns = max(warmup_elapsed_ns / warmup_calls, 1)  # a call under the clock's resolution
    calls = max(1, math.ceil(repeat_seconds * NANOSECONDS_PER_SECOND / mean_call_ns))
    throughputs = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(calls):
            call()
        elapsed_seconds = (time.perf_counter_ns() - start) / NANOSECONDS_PER_SECOND
        throughputs.append(items_per_call * calls / elapsed_seconds)
    return tuple(throughputs)


def write_results(measurements: list[Measurement], directory: Path) -> Path:
    """Write raw samples and summaries to a timestamped JSON file in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    records = [{**asdict(m), "summary_ms": m.summary()} for m in measurements]
    path.write_text(json.dumps(records, indent=1))
    return path
