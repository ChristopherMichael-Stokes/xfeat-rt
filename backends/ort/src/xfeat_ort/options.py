"""Command-line options shared by `xfeat-ort-predict` and `xfeat-ort-bench`."""

import argparse
from enum import StrEnum
from pathlib import Path

from xfeat_ort.models import (
    DEFAULT_LIGHTERGLUE_UNITS,
    DEFAULT_XFEAT_UNITS,
    ComputeUnits,
    CoreMLProvider,
    LighterGlueModel,
    XFeatModel,
)

DEFAULT_CACHE_DIRECTORY = Path("uncommitted/coreml-cache")


class Provider(StrEnum):
    """ONNX Runtime execution provider the graphs run on."""

    CPU = "cpu"
    COREML = "coreml"


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the model files, thread count and execution provider options."""
    parser.add_argument("--xfeat", type=Path, required=True)
    parser.add_argument("--lighterglue", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--provider", type=Provider, choices=list(Provider), default=Provider.CPU)
    parser.add_argument(
        "--xfeat-units",
        type=ComputeUnits,
        choices=list(ComputeUnits),
        default=DEFAULT_XFEAT_UNITS,
        help="Core ML compute units for XFeat",
    )
    parser.add_argument(
        "--lighterglue-units",
        type=ComputeUnits,
        choices=list(ComputeUnits),
        default=DEFAULT_LIGHTERGLUE_UNITS,
        help="Core ML compute units for LighterGlue",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIRECTORY,
        help="Core ML compiled-model cache",
    )


def load_models(args: argparse.Namespace) -> tuple[XFeatModel, LighterGlueModel]:
    """Sessions for `args.xfeat` and `args.lighterglue` on `args.provider`."""
    if args.provider is Provider.CPU:
        return (
            XFeatModel(args.xfeat, args.threads),
            LighterGlueModel(args.lighterglue, args.threads),
        )
    return (
        XFeatModel(args.xfeat, args.threads, CoreMLProvider(args.xfeat_units, args.cache_dir)),
        LighterGlueModel(
            args.lighterglue, args.threads, CoreMLProvider(args.lighterglue_units, args.cache_dir)
        ),
    )


def provider_settings(args: argparse.Namespace) -> dict[str, str]:
    """Run settings that name the provider and, for Core ML, the compute units."""
    settings = {"provider": str(args.provider)}
    if args.provider is Provider.COREML:
        settings |= {
            "xfeat_units": str(args.xfeat_units),
            "lighterglue_units": str(args.lighterglue_units),
        }
    return settings
