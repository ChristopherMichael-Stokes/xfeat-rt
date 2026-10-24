"""`xfeat-trt-predict`: extract and match the pairs of a manifest into a prediction directory."""

import argparse
from pathlib import Path

import tensorrt

from xfeat_pairs.manifest import load_manifest
from xfeat_pairs.predictions import RunHeader, write_predictions
from xfeat_pairs.preprocess import ResizeMode
from xfeat_pairs.run import predict_pairs
from xfeat_trt.engine import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OPTIMIZATION_LEVEL,
    MAX_OPTIMIZATION_LEVEL,
    BuildOptions,
)
from xfeat_trt.models import LighterGlueModel, XFeatModel


def main() -> None:
    """Entry point of `xfeat-trt-predict`; see `xfeat_pairs.manifest` for the manifest format."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--xfeat", type=Path, required=True, help="exported XFeat ONNX file")
    parser.add_argument("--lighterglue", type=Path, required=True, help="LighterGlue ONNX file")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resize", type=ResizeMode, choices=list(ResizeMode), required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--tf32", action="store_true", help="allow TF32 in FP32 layers (faster, less precise)"
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=DEFAULT_OPTIMIZATION_LEVEL,
        choices=range(MAX_OPTIMIZATION_LEVEL + 1),
        help="TensorRT builder optimization level (higher builds slower, runs faster)",
    )
    args = parser.parse_args()
    options = BuildOptions(args.tf32, args.optimization_level)

    xfeat = XFeatModel(args.xfeat, args.cache_dir, options)
    lighterglue = LighterGlueModel(args.lighterglue, args.cache_dir, options)
    meta = xfeat.metadata
    model_size = (meta.width, meta.height)
    header = RunHeader(
        backend=f"tensorrt {tensorrt.__version__}",
        resize_mode=args.resize,
        model_size=model_size,
        xfeat=xfeat.raw_metadata,
        lighterglue=lighterglue.raw_metadata,
        settings={
            "tf32": str(args.tf32),
            "optimization_level": str(args.optimization_level),
        },
    )
    results = predict_pairs(
        load_manifest(args.manifest),
        model_size=model_size,
        resize_mode=args.resize,
        extract=lambda pixels: xfeat.extract(pixels)[0],
        match=lighterglue.match,
    )
    print(write_predictions(args.out_dir, header, results))  # noqa: T201  # CLI output
