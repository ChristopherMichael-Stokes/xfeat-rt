"""Dynamo ONNX export of the static graphs, opset 20, fixed shapes."""

from pathlib import Path

import onnx
import torch
from onnxscript import FLOAT
from onnxscript import opset20 as op
from torch import nn

from xfeat_export.metadata import attach_metadata

OPSET = 20
"""Opset 23 exports attention as a fused `Attention` node, but TensorRT 11.4 segfaults on it."""


def space_to_depth(x: FLOAT, downscale_factor: int) -> FLOAT:
    """ONNX `SpaceToDepth` for `aten.pixel_unshuffle`; the default lowering is Reshape/Transpose."""
    return op.SpaceToDepth(x, blocksize=downscale_factor)


TRANSLATION_TABLE = {torch.ops.aten.pixel_unshuffle.default: space_to_depth}


def export_model(  # noqa: PLR0913  # one call writes one fully described file
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    input_names: list[str],
    output_names: list[str],
    metadata: dict[str, str],
    path: Path,
) -> None:
    """Export `module` to `path` with `metadata` attached, then run the ONNX checker."""
    with torch.no_grad():
        torch.onnx.export(
            module.eval(),
            example_inputs,
            str(path),
            dynamo=True,
            opset_version=OPSET,
            input_names=input_names,
            output_names=output_names,
            external_data=False,
            custom_translation_table=TRANSLATION_TABLE,  # ty: ignore[invalid-argument-type]
        )
    model = onnx.load(str(path))
    attach_metadata(model, {**metadata, "opset": str(OPSET)})
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
