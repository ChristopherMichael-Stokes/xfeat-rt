"""Batch ladder discovery from ONNX metadata; the graphs are placeholders, only metadata is read."""

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper
from xfeat_trt.ladder import LadderError, find_ladder

COMMON = {
    "opset": "20",
    "weights_sha256": "w",
    "git_hash": "g",
    "precision": "fp16",
    "target": "tensorrt",
}
XFEAT = {
    **COMMON,
    "model": "xfeat",
    "height": "48",
    "width": "64",
    "top_k": "8",
    "detection_threshold": "0.05",
    "grid": "pixel_centre",
    "topk_mode": "plain",
}
LIGHTERGLUE = {
    **COMMON,
    "model": "lighterglue",
    "top_k": "8",
    "match_threshold": "0.1",
    "pruning": "off",
}


def _write(path: Path, metadata: dict[str, str]) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "placeholder",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, path)


def _write_ladder(directory: Path, batches: tuple[int, ...]) -> None:
    for batch in batches:
        _write(directory / f"x_b{batch}.onnx", {**XFEAT, "batch": str(batch)})
        _write(directory / f"l_b{batch}.onnx", {**LIGHTERGLUE, "batch": str(batch)})


def test_ladder_maps_each_batch_size_to_its_file(tmp_path: Path) -> None:
    _write_ladder(tmp_path, (4, 1, 2))
    ladder = find_ladder(tmp_path)
    assert list(ladder.xfeat) == [1, 2, 4]
    assert ladder.xfeat[2] == tmp_path / "x_b2.onnx"
    assert ladder.lighterglue[4] == tmp_path / "l_b4.onnx"
    assert (ladder.xfeat_metadata.width, ladder.xfeat_metadata.height) == (64, 48)


def test_mixed_resolutions_are_rejected(tmp_path: Path) -> None:
    _write_ladder(tmp_path, (1,))
    _write(tmp_path / "x_b2_wide.onnx", {**XFEAT, "batch": "2", "width": "128"})
    with pytest.raises(LadderError, match="differ"):
        find_ladder(tmp_path)


def test_a_repeated_batch_size_is_rejected(tmp_path: Path) -> None:
    _write_ladder(tmp_path, (1,))
    _write(tmp_path / "x_b1_copy.onnx", {**XFEAT, "batch": "1"})
    with pytest.raises(LadderError, match="both have batch 1"):
        find_ladder(tmp_path)


def test_mismatched_k_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "x_b1.onnx", {**XFEAT, "batch": "1"})
    _write(tmp_path / "l_b1.onnx", {**LIGHTERGLUE, "batch": "1", "top_k": "16"})
    with pytest.raises(LadderError, match="K="):
        find_ladder(tmp_path)
