"""Engine cache behaviour on a tiny ONNX graph; needs an NVIDIA GPU."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from cuda.core import Device
from onnx import TensorProto, helper
from xfeat_trt.engine import (
    BuildOptions,
    engine_cache_key,
    load_or_build_engine,
    read_onnx_metadata,
)
from xfeat_trt.runner import EngineRunner

METADATA = {"model": "tiny", "top_k": "4"}


def _write_graph(path: Path, addend: float) -> None:
    """Y = x + addend over a (1, 4) float tensor."""
    constant = helper.make_tensor("c", TensorProto.FLOAT, [1, 4], [addend] * 4)
    node = helper.make_node("Add", ["x", "c"], ["y"])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializer=[constant],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    for key, value in METADATA.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, path)


@pytest.fixture
def device() -> Device:
    return Device(0)


def test_second_load_hits_the_cache_and_runs(tmp_path: Path, device: Device) -> None:
    graph = tmp_path / "tiny.onnx"
    _write_graph(graph, 1.0)
    cache = tmp_path / "cache"

    first = load_or_build_engine(graph, device, cache)
    second = load_or_build_engine(graph, device, cache)

    assert first.built
    assert not second.built
    assert second.metadata == METADATA
    x = np.arange(4, dtype=np.float32).reshape(1, 4)
    outputs = EngineRunner(second.engine, device).run({"x": x})
    np.testing.assert_array_equal(outputs["y"], x + 1.0)


def test_corrupt_plan_is_rebuilt(tmp_path: Path, device: Device) -> None:
    graph = tmp_path / "tiny.onnx"
    _write_graph(graph, 1.0)
    cache = tmp_path / "cache"
    built = load_or_build_engine(graph, device, cache)
    built.path.write_bytes(b"not a TensorRT plan")

    rebuilt = load_or_build_engine(graph, device, cache)

    assert rebuilt.built
    assert rebuilt.path.stat().st_size > len(b"not a TensorRT plan")


def test_changed_graph_changes_the_key(tmp_path: Path, device: Device) -> None:
    first, second = tmp_path / "a.onnx", tmp_path / "b.onnx"
    _write_graph(first, 1.0)
    _write_graph(second, 2.0)
    assert engine_cache_key(first, device, BuildOptions()) != engine_cache_key(
        second, device, BuildOptions()
    )


def test_onnx_metadata_is_read_back(tmp_path: Path) -> None:
    graph = tmp_path / "tiny.onnx"
    _write_graph(graph, 1.0)
    assert read_onnx_metadata(graph) == METADATA


def test_build_options_change_the_key(tmp_path: Path, device: Device) -> None:
    graph = tmp_path / "tiny.onnx"
    _write_graph(graph, 1.0)
    keys = {
        engine_cache_key(graph, device, options)
        for options in (BuildOptions(), BuildOptions(tf32=True), BuildOptions(optimization_level=5))
    }
    assert len(keys) == 3


def test_optimization_level_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="optimization_level"):
        BuildOptions(optimization_level=6)
