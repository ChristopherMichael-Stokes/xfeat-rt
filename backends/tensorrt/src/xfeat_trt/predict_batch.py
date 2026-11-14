"""`xfeat-trt-predict-batch`: the pairs of a manifest at the calibrated batch sizes.

Throughput mode beside the batch-1 `xfeat-trt-predict`: each unique image is extracted once,
both models run on batches through CUDA graphs, and images are prepared as they are read. The
batch sizes come from `xfeat-trt-calibrate`'s stored result for the models directory, measured
first when there is none. The output is the same prediction directory as `xfeat-trt-predict` writes.
"""

import argparse
from pathlib import Path

import tensorrt

from xfeat_pairs.manifest import load_manifest
from xfeat_pairs.predictions import RunHeader, write_predictions
from xfeat_pairs.preprocess import ResizeMode, prepare_image
from xfeat_pairs.run import predict_pairs_batched
from xfeat_trt.arguments import add_engine_arguments, build_options
from xfeat_trt.calibrate import load_or_calibrate
from xfeat_trt.ladder import find_ladder
from xfeat_trt.models import LighterGlueModel, XFeatModel


def main() -> None:
    """Entry point of `xfeat-trt-predict-batch`; see `xfeat_pairs.manifest` for the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=Path, required=True, help="directory holding one batch ladder"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resize", type=ResizeMode, choices=list(ResizeMode), required=True)
    add_engine_arguments(parser)
    args = parser.parse_args()
    options = build_options(args)

    ladder = find_ladder(args.models)
    calibration, _ = load_or_calibrate(ladder, args.cache_dir, options)
    xfeat_batch = calibration.xfeat.batch_size
    lighterglue_batch = calibration.lighterglue.batch_size
    xfeat = XFeatModel(ladder.xfeat[xfeat_batch], args.cache_dir, options)
    lighterglue = LighterGlueModel(ladder.lighterglue[lighterglue_batch], args.cache_dir, options)
    xfeat.runner.capture_graph()
    lighterglue.runner.capture_graph()

    model_size = (xfeat.metadata.width, xfeat.metadata.height)
    header = RunHeader(
        backend=f"tensorrt {tensorrt.__version__}",
        resize_mode=args.resize,
        model_size=model_size,
        xfeat=xfeat.raw_metadata,
        lighterglue=lighterglue.raw_metadata,
        settings={
            "tf32": str(args.tf32),
            "optimization_level": str(args.optimization_level),
            "cuda_graph": "True",
        },
    )
    results = predict_pairs_batched(
        load_manifest(args.manifest),
        load_image=lambda path: prepare_image(path, model_size, args.resize),
        extract=xfeat.extract,
        extract_batch_size=xfeat_batch,
        match=lighterglue.match_batch,
        match_batch_size=lighterglue_batch,
    )
    print(f"batch sizes: xfeat {xfeat_batch}, lighterglue {lighterglue_batch}")  # noqa: T201  # CLI output
    print(write_predictions(args.out_dir, header, results))  # noqa: T201
