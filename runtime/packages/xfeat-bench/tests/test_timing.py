import numpy as np

from xfeat_bench.timing import measure


def test_measure_discards_warmup_and_keeps_one_sample_per_run() -> None:
    calls = []
    result = measure("noop", lambda: calls.append(1), warmup=3, runs=7, settings={})
    assert len(calls) == 10
    assert len(result.samples_ms) == 7


def test_summary_percentiles_follow_linear_interpolation_of_the_samples() -> None:
    result = measure("noop", lambda: None, warmup=0, runs=5, settings={})
    object.__setattr__(result, "samples_ms", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.summary() == {
        "p50": 3.0,
        "p95": float(np.percentile([1, 2, 3, 4, 5], 95)),
        "p99": float(np.percentile([1, 2, 3, 4, 5], 99)),
    }
