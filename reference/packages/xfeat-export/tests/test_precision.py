"""FP16 export keeps FP32 graph I/O and converts only the networks' weights."""

from pathlib import Path

import onnx
import pytest
import torch
from onnx import TensorProto

from xfeat_export.cli import export_lighterglue, export_xfeat
from xfeat_export.precision import Precision
from xfeat_reference.pipeline import GridNormalisation, ReferenceConfig, ReferencePipeline

HEIGHT, WIDTH, TOP_K = 64, 64, 16


@pytest.fixture(scope="module")
def pipeline() -> ReferencePipeline:
    config = ReferenceConfig(GridNormalisation.PIXEL_CENTRE, pruning=False, top_k=TOP_K)
    return ReferencePipeline(config, torch.device("cpu"))


def _tensor_types(model: onnx.ModelProto) -> set[int]:
    return {initializer.data_type for initializer in model.graph.initializer}


def _io_types(model: onnx.ModelProto) -> set[int]:
    values = [*model.graph.input, *model.graph.output]
    return {value.type.tensor_type.elem_type for value in values}


def _metadata(model: onnx.ModelProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in model.metadata_props}


@pytest.mark.parametrize("precision", list(Precision))
def test_xfeat_weights_follow_precision_and_io_stays_fp32(
    pipeline: ReferencePipeline, tmp_path: Path, precision: Precision
) -> None:
    path = export_xfeat(
        pipeline.xfeat_network,
        tmp_path,
        height=HEIGHT,
        width=WIDTH,
        top_k=TOP_K,
        batch=1,
        chunking=None,
        precision=precision,
    )
    model = onnx.load(str(path))
    assert (TensorProto.FLOAT16 in _tensor_types(model)) == (precision is Precision.FP16)
    assert TensorProto.FLOAT16 not in _io_types(model)
    assert _metadata(model)["precision"] == str(precision)


def test_lighterglue_fp16_keeps_assignment_head_fp32(
    pipeline: ReferencePipeline, tmp_path: Path
) -> None:
    path = export_lighterglue(
        pipeline.lighterglue_network,
        tmp_path,
        top_k=TOP_K,
        batch=1,
        width=WIDTH,
        height=HEIGHT,
        precision=Precision.FP16,
    )
    model = onnx.load(str(path))
    assert {TensorProto.FLOAT16, TensorProto.FLOAT} <= _tensor_types(model)
    assert TensorProto.FLOAT16 not in _io_types(model)
    assert _metadata(model)["precision"] == "fp16"
