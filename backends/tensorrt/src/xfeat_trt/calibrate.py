"""`xfeat-trt-calibrate`: measure the batch ladder of a models directory and pick each batch size.

Each model is swept separately in increasing batch size. A batch size is timed end to end as
`xfeat-trt-predict-batch` runs it: upload, CUDA graph launch, download and the host-side
unpacking of the outputs. The sweep of a model stops after its throughput fell twice in a row, or
at the first batch size whose engine cannot be built or allocated. The result is stored under a
key of everything that invalidates it and reused by later calls with the same key.
"""

import argparse
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.core import Device, system

# cuda.core raises CUDAError but exports it only from this private module
from cuda.core._utils.cuda_utils import CUDAError

from xfeat_bench.calibration import (
    Calibration,
    FailedBatchSize,
    MeasuredBatchSize,
    ModelCalibration,
    load_calibration,
    save_calibration,
    throughput_fell_twice,
)
from xfeat_bench.timing import measure_throughput
from xfeat_trt.arguments import add_engine_arguments, build_options
from xfeat_trt.engine import BuildOptions, EngineBuildError
from xfeat_trt.ladder import BatchLadder, find_ladder
from xfeat_trt.models import LighterGlueModel, XFeatModel
from xfeat_trt.random_inputs import random_features, random_images
from xfeat_trt.runner import EngineAllocationError

LIGHTERGLUE_BATCH_SIZES = (1, 2, 4)
"""LighterGlue gained nothing per pair at any B > 1 measured (TensorRT FP16, K 2048 and 4096,
RTX 4070 Ti), so larger batches are not swept."""
WARMUP_SECONDS = 2.0
"""The GPU idles at low clocks during an engine build: with three warm-up calls the first two 1 s
repeats ran at 30 % and 91 % of the steady rate (LighterGlue K=256, RTX 4070 Ti)."""
REPEATS = 5
REPEAT_SECONDS = 1.0
"""Long enough to average over GPU clock changes, short enough for a sweep of about a minute."""
RANDOM_SEED = 0
CALIBRATION_DIRECTORY = "calibration"

_NOT_POSSIBLE = (EngineBuildError, EngineAllocationError, CUDAError)

Trial = MeasuredBatchSize | FailedBatchSize


def calibration_key(ladder: BatchLadder, device: Device, options: BuildOptions) -> dict[str, str]:
    """Everything a stored calibration of `ladder` depends on."""
    driver = ".".join(str(part) for part in system.get_user_mode_driver_version())
    return {
        "device": device.name,
        "compute_capability": device.arch,
        "cuda_driver": driver,
        "tensorrt": trt.__version__,
        "tf32": str(options.tf32),
        "optimization_level": str(options.optimization_level),
        **{f"sha256:{name}": digest for name, digest in ladder.onnx_sha256().items()},
    }


def load_or_calibrate(
    ladder: BatchLadder, cache_dir: Path, options: BuildOptions
) -> tuple[Calibration, bool]:
    """Stored calibration of `ladder`, measured and stored on a miss; and whether it was stored.

    Runs on the device the models load on by default.
    """
    device = Device()
    device.set_current()
    directory = cache_dir / CALIBRATION_DIRECTORY
    key = calibration_key(ladder, device, options)
    stored = load_calibration(directory, key)
    if stored is not None:
        return stored, True
    image_size = (ladder.xfeat_metadata.width, ladder.xfeat_metadata.height)
    calibration = Calibration(
        key=key,
        xfeat=_sweep(
            "xfeat",
            ladder.xfeat.items(),
            lambda batch, path: _xfeat_trial(batch, path, cache_dir, options),
        ),
        lighterglue=_sweep(
            "lighterglue",
            ((b, p) for b, p in ladder.lighterglue.items() if b in LIGHTERGLUE_BATCH_SIZES),
            lambda batch, path: _lighterglue_trial(batch, path, image_size, cache_dir, options),
        ),
    )
    save_calibration(directory, calibration)
    return calibration, False


def _sweep(
    model: str, rungs: Iterable[tuple[int, Path]], trial: Callable[[int, Path], Trial]
) -> ModelCalibration:
    trials: list[Trial] = []
    for batch, path in rungs:
        trials.append(trial(batch, path))
        print(f"{model} {_format_trial(trials[-1])}", flush=True)  # noqa: T201  # CLI progress
        if isinstance(trials[-1], FailedBatchSize) or throughput_fell_twice(trials):
            break
    return ModelCalibration(trials=tuple(trials))


def _xfeat_trial(batch: int, path: Path, cache_dir: Path, options: BuildOptions) -> Trial:
    start = time.perf_counter()
    try:
        model = XFeatModel(path, cache_dir, options)
        model.runner.capture_graph()
    except _NOT_POSSIBLE as error:
        return FailedBatchSize(batch_size=batch, reason=str(error))
    build_seconds = time.perf_counter() - start if model.built else None
    images = random_images(model.metadata, np.random.default_rng(RANDOM_SEED))
    return MeasuredBatchSize(
        batch_size=batch,
        items_per_second=_throughput(lambda: model.extract(images), batch),
        build_seconds=build_seconds,
    )


def _lighterglue_trial(
    batch: int, path: Path, image_size: tuple[int, int], cache_dir: Path, options: BuildOptions
) -> Trial:
    start = time.perf_counter()
    try:
        model = LighterGlueModel(path, cache_dir, options)
        model.runner.capture_graph()
    except _NOT_POSSIBLE as error:
        return FailedBatchSize(batch_size=batch, reason=str(error))
    build_seconds = time.perf_counter() - start if model.built else None
    rng = np.random.default_rng(RANDOM_SEED)
    top_k = model.metadata.top_k
    pairs = [
        (random_features(top_k, image_size, rng), random_features(top_k, image_size, rng))
        for _ in range(batch)
    ]
    return MeasuredBatchSize(
        batch_size=batch,
        items_per_second=_throughput(lambda: model.match_batch(pairs), batch),
        build_seconds=build_seconds,
    )


def _throughput(call: Callable[[], object], batch: int) -> tuple[float, ...]:
    return measure_throughput(
        call,
        items_per_call=batch,
        warmup_seconds=WARMUP_SECONDS,
        repeats=REPEATS,
        repeat_seconds=REPEAT_SECONDS,
    )


def _format_trial(trial: Trial) -> str:
    if isinstance(trial, FailedBatchSize):
        return f"B={trial.batch_size}: not possible ({trial.reason})"
    build = "cached" if trial.build_seconds is None else f"built in {trial.build_seconds:.1f} s"
    return (
        f"B={trial.batch_size}: median {trial.median:.1f}/s, spread {trial.spread:.1f}/s, "
        f"engine {build}"
    )


def format_calibration(calibration: Calibration) -> str:
    """Per-model trials and chosen batch size, one line each."""
    lines: list[str] = []
    for model, sweep in (("xfeat", calibration.xfeat), ("lighterglue", calibration.lighterglue)):
        lines += [f"{model} {_format_trial(trial)}" for trial in sweep.trials]
        lines.append(f"{model} chosen B={sweep.batch_size}")
    return "\n".join(lines)


def main() -> None:
    """Entry point of `xfeat-trt-calibrate`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=Path, required=True, help="directory holding one batch ladder"
    )
    add_engine_arguments(parser)
    args = parser.parse_args()

    calibration, stored = load_or_calibrate(
        find_ladder(args.models), args.cache_dir, build_options(args)
    )
    print("stored calibration" if stored else "measured calibration")  # noqa: T201  # CLI output
    print(format_calibration(calibration))  # noqa: T201
