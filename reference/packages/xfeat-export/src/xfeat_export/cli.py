"""`xfeat-export`: write XFeat and LighterGlue ONNX files for one shape and keypoint budget.

One file pair is written per batch size, so `--batch 1 2 4 8` gives the ladder that
batch-size calibration sweeps.
"""

import argparse
import platform
from enum import StrEnum
from pathlib import Path

import torch

from xfeat_export.coreml_graph import CoreMLLighterGlue, CoreMLXFeat
from xfeat_export.lighterglue_graph import StaticLighterGlue
from xfeat_export.metadata import git_description
from xfeat_export.onnx_export import export_model
from xfeat_export.precision import Precision
from xfeat_export.xfeat_graph import DEFAULT_CHUNKING, ChunkedTopK, StaticXFeat
from xfeat_reference.pipeline import GridNormalisation, ReferenceConfig, ReferencePipeline
from xfeat_reference.upstream import LIGHTERGLUE_WEIGHTS_SHA256, XFEAT_WEIGHTS_SHA256

GRID = GridNormalisation.PIXEL_CENTRE
DETECTION_THRESHOLD = 0.05
"""Upstream XFeat default."""
MATCH_THRESHOLD = 0.1
"""Upstream LighterGlue default."""
IMAGE_CHANNELS = 1
KEYPOINT_COORDINATES = 2
DESCRIPTOR_DIMENSION = 64


class Target(StrEnum):
    """Runtime the exported graph is rewritten for."""

    TENSORRT = "tensorrt"
    COREML = "coreml"


NATIVE_TARGET = "native"
"""`--target` value that resolves to the target of the machine running the export."""


def native_target() -> Target:
    """Core ML on Apple silicon, where the Core ML graphs measured fastest; TensorRT elsewhere."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return Target.COREML
    return Target.TENSORRT


def _suffix(precision: Precision, target: Target) -> str:
    precision_part = "" if precision is Precision.FP32 else f"_{precision}"
    target_part = "" if target is Target.TENSORRT else f"_{target}"
    return precision_part + target_part


def export_xfeat(  # noqa: PLR0913  # one parameter per exported shape setting
    network: object,
    out_dir: Path,
    *,
    height: int,
    width: int,
    top_k: int,
    batch: int,
    chunking: ChunkedTopK | None,
    precision: Precision,
    target: Target,
) -> Path:
    """Export the XFeat graph and return its path."""
    module_class = CoreMLXFeat if target is Target.COREML else StaticXFeat
    module = module_class(
        network,
        height=height,
        width=width,
        top_k=top_k,
        detection_threshold=DETECTION_THRESHOLD,
        chunking=chunking,
        precision=precision,
    )
    path = out_dir / f"xfeat_{width}x{height}_k{top_k}_b{batch}{_suffix(precision, target)}.onnx"
    metadata = {
        "model": "xfeat",
        "height": str(height),
        "width": str(width),
        "batch": str(batch),
        "top_k": str(top_k),
        "detection_threshold": str(DETECTION_THRESHOLD),
        "grid": str(GRID),
        "topk_mode": "chunked" if chunking else "plain",
        "precision": str(precision),
        "target": str(target),
        "weights_sha256": XFEAT_WEIGHTS_SHA256,
        "git_hash": git_description(),
    }
    if chunking:
        metadata |= {
            "chunk_size": str(chunking.chunk_size),
            "max_k_per_op": str(chunking.max_k_per_op),
        }
    export_model(
        module,
        (torch.rand(batch, IMAGE_CHANNELS, height, width),),
        input_names=["image"],
        output_names=["kpts", "scores", "desc", "valid"],
        metadata=metadata,
        path=path,
    )
    return path


def export_lighterglue(  # noqa: PLR0913  # shape settings of the paired XFeat export
    network: object,
    out_dir: Path,
    *,
    top_k: int,
    batch: int,
    width: int,
    height: int,
    precision: Precision,
    target: Target,
) -> Path:
    """Export the padded LighterGlue graph and return its path."""
    module_class = CoreMLLighterGlue if target is Target.COREML else StaticLighterGlue
    module = module_class(network, MATCH_THRESHOLD, precision)
    path = out_dir / f"lighterglue_k{top_k}_b{batch}{_suffix(precision, target)}.onnx"
    example: list[torch.Tensor] = []
    names: list[str] = []
    for side in (0, 1):
        example += [
            torch.rand(batch, top_k, KEYPOINT_COORDINATES) * torch.tensor([width - 1, height - 1]),
            torch.nn.functional.normalize(torch.randn(batch, top_k, DESCRIPTOR_DIMENSION), dim=-1),
            torch.rand(batch, top_k) > 0.1,  # noqa: PLR2004  # ~10 % padding exercises the mask
            torch.tensor([[width, height]], dtype=torch.float32).repeat(batch, 1),
        ]
        names += [f"{name}{side}" for name in ("kpts", "desc", "valid", "image_size")]
    metadata = {
        "model": "lighterglue",
        "batch": str(batch),
        "top_k": str(top_k),
        "match_threshold": str(MATCH_THRESHOLD),
        "precision": str(precision),
        "target": str(target),
        "pruning": "off",
        "weights_sha256": LIGHTERGLUE_WEIGHTS_SHA256,
        "git_hash": git_description(),
    }
    export_model(
        module,
        tuple(example),
        input_names=names,
        output_names=["m0", "mscores0"],
        metadata=metadata,
        path=path,
    )
    return path


def main() -> None:
    """Entry point of `xfeat-export`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1], help="one file pair per batch size"
    )
    parser.add_argument(
        "--chunked-topk",
        action="store_true",
        help="split TopK so no node exceeds K=2048 (TensorRT builds K <= 3840 only)",
    )
    parser.add_argument(
        "--precision",
        type=Precision,
        choices=list(Precision),
        default=Precision.FP32,
        help="backbone and transformer precision; detection and assignment stay FP32",
    )
    parser.add_argument(
        "--target",
        choices=[*Target, NATIVE_TARGET],
        default=NATIVE_TARGET,
        help="runtime the graphs are rewritten for (coreml: ONNX Runtime Core ML provider; "
        "native: coreml on Apple silicon, tensorrt elsewhere)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = native_target() if args.target == NATIVE_TARGET else Target(args.target)

    config = ReferenceConfig(grid=GRID, pruning=False, top_k=args.top_k)
    pipeline = ReferencePipeline(config, torch.device("cpu"))
    chunking = DEFAULT_CHUNKING if args.chunked_topk else None
    for batch in args.batch:
        for path in (
            export_xfeat(
                pipeline.xfeat_network,
                args.out_dir,
                height=args.height,
                width=args.width,
                top_k=args.top_k,
                batch=batch,
                chunking=chunking,
                precision=args.precision,
                target=target,
            ),
            export_lighterglue(
                pipeline.lighterglue_network,
                args.out_dir,
                top_k=args.top_k,
                batch=batch,
                width=args.width,
                height=args.height,
                precision=args.precision,
                target=target,
            ),
        ):
            print(path)  # noqa: T201  # CLI output
