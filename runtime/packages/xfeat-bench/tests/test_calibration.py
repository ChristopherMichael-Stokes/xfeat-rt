from pathlib import Path

import pytest

from xfeat_bench.calibration import (
    Calibration,
    FailedBatchSize,
    MeasuredBatchSize,
    ModelCalibration,
    NoBatchSizeError,
    load_calibration,
    save_calibration,
    select_batch_size,
    throughput_fell_twice,
)
from xfeat_bench.timing import measure_throughput


def _measured(batch_size: int, *items_per_second: float) -> MeasuredBatchSize:
    return MeasuredBatchSize(
        batch_size=batch_size, items_per_second=items_per_second, build_seconds=None
    )


def test_smallest_batch_within_the_spread_of_the_best_is_chosen() -> None:
    # best median 120 (B=8), its spread 125 - 110 = 15, so the floor is 105:
    # B=2 (median 100) is below it, B=4 (median 110) is the smallest inside
    trials = [
        _measured(1, 50, 52, 51),
        _measured(2, 99, 100, 101),
        _measured(4, 108, 110, 112),
        _measured(8, 110, 120, 125),
        _measured(16, 118, 119, 119.5),
    ]
    assert select_batch_size(trials) == 4


def test_a_tight_best_wins_over_a_close_smaller_batch() -> None:
    # best median 120 with spread 1 gives floor 119; B=1 at median 118 is outside it
    trials = [_measured(1, 117, 118, 119), _measured(2, 119.5, 120, 120.5)]
    assert select_batch_size(trials) == 2


def test_failed_batch_sizes_are_never_chosen() -> None:
    trials = [_measured(1, 10, 10, 10), FailedBatchSize(batch_size=2, reason="out of memory")]
    assert select_batch_size(trials) == 1


def test_a_sweep_with_nothing_measured_has_no_choice() -> None:
    with pytest.raises(NoBatchSizeError):
        select_batch_size([FailedBatchSize(batch_size=1, reason="out of memory")])


@pytest.mark.parametrize(
    ("medians", "expected"),
    [
        ((10, 9), False),  # one fall
        ((10, 9, 9.5), False),  # fell, then rose
        ((10, 9, 8), True),
        ((5, 10, 9, 8), True),
        ((10, 9, 8, 8.5), False),  # only the last two steps count
    ],
)
def test_sweep_stops_after_two_consecutive_falls(
    medians: tuple[float, ...], *, expected: bool
) -> None:
    trials = [_measured(2**index, median) for index, median in enumerate(medians)]
    assert throughput_fell_twice(trials) is expected


def test_calibration_round_trips_under_its_key(tmp_path: Path) -> None:
    calibration = Calibration(
        key={"device": "gpu", "onnx": "abc"},
        xfeat=ModelCalibration(
            trials=(
                MeasuredBatchSize(batch_size=1, items_per_second=(1.5, 2.25), build_seconds=3.0),
                FailedBatchSize(batch_size=2, reason="out of memory"),
            )
        ),
        lighterglue=ModelCalibration(trials=(_measured(1, 4.0),)),
    )
    save_calibration(tmp_path, calibration)

    assert load_calibration(tmp_path, {"onnx": "abc", "device": "gpu"}) == calibration
    assert load_calibration(tmp_path, {"device": "other gpu", "onnx": "abc"}) is None


def test_throughput_counts_items_of_every_call() -> None:
    calls = []
    throughputs = measure_throughput(
        lambda: calls.append(1), items_per_call=4, warmup_seconds=0.0, repeats=3, repeat_seconds=0.0
    )
    # zero durations still make one call each: 1 warm-up + 3 repeats
    assert len(calls) == 4
    assert len(throughputs) == 3
    assert all(value > 0 for value in throughputs)
