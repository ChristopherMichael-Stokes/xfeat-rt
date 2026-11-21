"""`xfeat-ort-predict`: extract and match the pairs of a manifest into a prediction directory."""

import argparse
from pathlib import Path

import onnxruntime

from xfeat_ort.options import add_model_arguments, load_models, provider_settings
from xfeat_pairs.manifest import load_manifest
from xfeat_pairs.predictions import RunHeader, write_predictions
from xfeat_pairs.preprocess import ResizeMode
from xfeat_pairs.run import predict_pairs


def main() -> None:
    """Entry point of `xfeat-ort-predict`; see `xfeat_pairs.manifest` for the manifest format."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    add_model_arguments(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resize", type=ResizeMode, choices=list(ResizeMode), required=True)
    args = parser.parse_args()

    xfeat, lighterglue = load_models(args)
    meta = xfeat.metadata
    model_size = (meta.width, meta.height)
    header = RunHeader(
        backend=f"onnxruntime {onnxruntime.__version__}",
        resize_mode=args.resize,
        model_size=model_size,
        xfeat=xfeat.raw_metadata,
        lighterglue=lighterglue.raw_metadata,
        settings={"threads": str(args.threads), **provider_settings(args)},
    )
    results = predict_pairs(
        load_manifest(args.manifest),
        model_size=model_size,
        resize_mode=args.resize,
        extract=lambda pixels: xfeat.extract(pixels)[0],
        match=lighterglue.match,
    )
    print(write_predictions(args.out_dir, header, results))  # noqa: T201  # CLI output
