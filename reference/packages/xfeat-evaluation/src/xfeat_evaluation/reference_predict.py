"""`xfeat-reference-predict`: the PyTorch reference over a manifest, as a prediction directory.

Runs with the same preprocessing, model size and keypoint budget as the backends so that its
predictions are the oracle for `xfeat-score --reference`.
"""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from xfeat_core.features import Features
from xfeat_pairs.manifest import load_manifest
from xfeat_pairs.predictions import RunHeader, write_predictions
from xfeat_pairs.preprocess import ResizeMode
from xfeat_pairs.run import predict_pairs
from xfeat_reference import pipeline as reference
from xfeat_reference.pipeline import GridNormalisation, ReferenceConfig, ReferencePipeline

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 960


def _extract(
    pipeline: ReferencePipeline, pixels: npt.NDArray[np.float32], size: tuple[int, int]
) -> Features:
    features = pipeline.extract(torch.from_numpy(pixels))
    return Features(
        features.keypoints.cpu().numpy().astype(np.float32),
        features.scores.cpu().numpy().astype(np.float32),
        features.descriptors.cpu().numpy().astype(np.float32),
        size,
    )


def _match(pipeline: ReferencePipeline, first: Features, second: Features) -> npt.NDArray[np.int64]:
    def as_reference(features: Features) -> reference.Features:
        return reference.Features(
            torch.from_numpy(features.keypoints).to(pipeline.device),
            torch.from_numpy(features.scores).to(pipeline.device),
            torch.from_numpy(features.descriptors).to(pipeline.device),
            features.image_size,
        )

    return pipeline.match(as_reference(first), as_reference(second))


def main() -> None:
    """Entry point of `xfeat-reference-predict`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resize", type=ResizeMode, choices=list(ResizeMode), required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.use_deterministic_algorithms(mode=True)
    config = ReferenceConfig(GridNormalisation.PIXEL_CENTRE, pruning=False, top_k=args.top_k)
    pipeline = ReferencePipeline(config, device)
    model_size = (args.width, args.height)
    header = RunHeader(
        backend=f"pytorch reference {torch.__version__} on {device}",
        resize_mode=args.resize,
        model_size=model_size,
        xfeat={"top_k": str(config.top_k), "grid": str(config.grid)},
        lighterglue={"match_threshold": str(config.match_threshold), "pruning": "off"},
        settings={"device": str(device)},
    )
    results = predict_pairs(
        load_manifest(args.manifest),
        model_size=model_size,
        resize_mode=args.resize,
        extract=lambda pixels: _extract(pipeline, pixels, model_size),
        match=lambda first, second: _match(pipeline, first, second),
    )
    print(write_predictions(args.out_dir, header, results))  # noqa: T201  # CLI output
