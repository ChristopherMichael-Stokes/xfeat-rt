"""MegaDepth-1500 relative-pose evaluation of the reference pipeline through GlueFactory."""

import importlib
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import gluefactory.settings
import h5py
import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from xfeat_reference.pipeline import Features, ReferencePipeline

PAIR_COUNT = 1500
COLMAP_PIXEL_CENTRE_OFFSET = 0.5
"""GlueFactory uses COLMAP pixel coordinates (image corner at 0, first pixel centre at 0.5);
XFeat keypoints are pixel indices (first pixel centre at 0)."""
RANSAC_THRESHOLDS_PX = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
"""GlueFactory's sweep for `ransac_th: -1`; the best threshold by AUC is reported."""
UNMATCHED = -1


@dataclass(frozen=True)
class EvaluationSettings:
    """MegaDepth-1500 protocol settings.

    Attributes:
        resize: Long image side in pixels after resizing.
        data_dir: GlueFactory data directory holding (or receiving) `megadepth1500/`.
        pair_limit: Pairs to run and score; a published comparison needs all `PAIR_COUNT`.
    """

    resize: int
    data_dir: Path
    pair_limit: int = PAIR_COUNT


@dataclass(frozen=True)
class Timing:
    """Mean wall time per pair, in milliseconds, without device synchronisation.

    Attributes:
        extract_ms: Both images' extraction.
        match_ms: LighterGlue matching.
    """

    extract_ms: float
    match_ms: float


def run(
    pipeline: ReferencePipeline, settings: EvaluationSettings, output_dir: Path
) -> dict[str, Any]:
    """Write predictions for every pair, score them, and save `summary.json` in `output_dir`.

    A `predictions.h5` already in `output_dir` is complete (it is renamed into place only after
    the last pair) and is scored as is, so a run that failed while scoring resumes there.

    Returns:
        The summary: GlueFactory metrics over the scored pairs, timing and settings.

    Raises:
        PredictionsMismatchError: An existing `predictions.h5` came from other settings.
    """
    eval_module = _gluefactory_megadepth1500(settings.data_dir)
    eval_pipeline = eval_module.MegaDepth1500Pipeline(_gluefactory_conf(settings.resize))
    loader = eval_pipeline.get_dataloader(eval_pipeline.conf.data)

    output_dir.mkdir(parents=True, exist_ok=True)
    # gluefactory's CacheLoader reads a relative path against DATA_PATH
    predictions_path = output_dir.resolve() / "predictions.h5"
    provenance = _provenance(pipeline, settings)
    if not predictions_path.exists():
        partial_path = predictions_path.with_suffix(".partial.h5")
        timing = write_predictions(pipeline, _first(loader, settings.pair_limit), partial_path)
        _write_attributes(partial_path, provenance, timing)
        partial_path.rename(predictions_path)
    timing = _read_timing(predictions_path, provenance)
    metrics, _, _ = eval_pipeline.run_eval(_first(loader, settings.pair_limit), predictions_path)

    summary = {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "timing": asdict(timing),
        **json.loads(provenance),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


class PredictionsMismatchError(RuntimeError):
    """An existing predictions file was written with other pipeline or protocol settings."""


def _provenance(pipeline: ReferencePipeline, settings: EvaluationSettings) -> str:
    return json.dumps(
        {
            "settings": {**asdict(settings), "data_dir": str(settings.data_dir)},
            "pipeline": {**asdict(pipeline.config), "device": str(pipeline.device)},
        }
    )


def _write_attributes(path: Path, provenance: str, timing: Timing) -> None:
    with h5py.File(path, "a") as predictions:
        predictions.attrs["provenance"] = provenance
        predictions.attrs["timing"] = json.dumps(asdict(timing))


def _read_timing(path: Path, expected_provenance: str) -> Timing:
    with h5py.File(path, "r") as predictions:
        provenance = str(predictions.attrs["provenance"])
        timing = json.loads(str(predictions.attrs["timing"]))
    if provenance != expected_provenance:
        msg = f"{path} was written with {provenance}, not {expected_provenance}"
        raise PredictionsMismatchError(msg)
    return Timing(**timing)


def write_predictions(
    pipeline: ReferencePipeline, pairs: Iterable[Mapping[str, Any]], path: Path
) -> Timing:
    """Run the pipeline on GlueFactory pairs and write a GlueFactory predictions file."""
    extract_s = match_s = 0.0
    pair_count = 0
    with h5py.File(path, "w") as predictions:
        for pair in tqdm(pairs, desc="pairs"):
            start = time.perf_counter()
            features = [pipeline.extract(pair[view]["image"]) for view in ("view0", "view1")]
            extracted = time.perf_counter()
            matches = pipeline.match(*features)
            extract_s += extracted - start
            match_s += time.perf_counter() - extracted
            pair_count += 1
            scales = [pair[view]["scales"][0] for view in ("view0", "view1")]
            _write_pair(predictions.create_group(pair["name"][0]), features, scales, matches)
    return Timing(extract_ms=extract_s / pair_count * 1e3, match_ms=match_s / pair_count * 1e3)


def _write_pair(
    group: h5py.Group,
    features: list[Features],
    scales: list[torch.Tensor],
    matches: npt.NDArray[np.int64],
) -> None:
    for view, (view_features, scale) in enumerate(zip(features, scales, strict=True)):
        keypoints = (view_features.keypoints.cpu() + COLMAP_PIXEL_CENTRE_OFFSET) / scale
        group.create_dataset(f"keypoints{view}", data=keypoints.numpy())
        group.create_dataset(f"keypoint_scores{view}", data=view_features.scores.cpu().numpy())
        partner = _partner_indices(
            len(view_features.keypoints), matches[:, view], matches[:, 1 - view]
        )
        group.create_dataset(f"matches{view}", data=partner)
        # upstream match_lighterglue returns no confidences; every kept match counts as certain
        group.create_dataset(
            f"matching_scores{view}", data=(partner != UNMATCHED).astype(np.float32)
        )


def _partner_indices(
    keypoint_count: int, own: npt.NDArray[np.int64], other: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    partner = np.full(keypoint_count, UNMATCHED, dtype=np.int64)
    partner[own] = other
    return partner


def _gluefactory_megadepth1500(data_dir: Path) -> ModuleType:
    # gluefactory modules bind settings.DATA_PATH at import; set it first so the dataset and its
    # download both use data_dir instead of gluefactory's install directory
    gluefactory.settings.DATA_PATH = data_dir
    return importlib.import_module("gluefactory.eval.megadepth1500")


def _gluefactory_conf(resize: int) -> DictConfig:
    return OmegaConf.create(
        {
            "data": {"preprocessing": {"resize": resize, "side": "long"}},
            "eval": {"estimator": "poselib", "ransac_th": list(RANSAC_THRESHOLDS_PX)},
        }
    )


def _first(loader: Iterable[Mapping[str, Any]], limit: int) -> Iterable[Mapping[str, Any]]:
    for index, pair in enumerate(loader):
        if index >= limit:
            return
        yield pair
