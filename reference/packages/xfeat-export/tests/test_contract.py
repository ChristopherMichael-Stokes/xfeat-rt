"""Cross-workspace contract: export with `reference`, run with each backend, compare.

Each backend runs in its own virtualenv through its predict CLI, so no torch is imported there.
Reference is FP32 PyTorch on CPU with the `pixel_centre` grid and pruning off.
"""

import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest
import torch

from xfeat_export.cli import export_lighterglue, export_xfeat
from xfeat_export.xfeat_graph import DEFAULT_CHUNKING, ChunkedTopK
from xfeat_pairs.predictions import read_index, read_pair
from xfeat_reference.pipeline import (
    Features,
    GridNormalisation,
    ReferenceConfig,
    ReferencePipeline,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MEGADEPTH = Path.home() / ".cache/xfeat-rt/gluefactory-data/megadepth1500"
HEIGHT, WIDTH = 960, 1280
PAIR_COUNT = 3

MAX_TIE_SWAPS = 1
TIE_RELATIVE_SCORE_GAP = 1e-5
"""FP32 reassociation reorders scores that differ below this at the K-th slot (C1 spike: gaps of
1.7e-5 and 7.5e-6 relative)."""
MIN_DESCRIPTOR_COSINE = 0.99999
"""C1 spike minimum over 10 images: 0.9999988."""
MAX_SCORE_ABSOLUTE_DIFFERENCE = 1e-4
"""C1 spike maximum over 10 images: 7.6e-5."""
MIN_MATCHES = 100
"""Fewer matches than this per pair would make the Jaccard comparison meaningless (spike: 227+)."""
MIN_MATCH_JACCARD = 0.99
"""One tie-swapped keypoint removes at most its own few matches from the 227+ per pair."""


@dataclass(frozen=True)
class Case:
    top_k: int
    chunking: ChunkedTopK | None


CASES = [Case(2048, None), Case(2048, DEFAULT_CHUNKING), Case(4096, DEFAULT_CHUNKING)]


@dataclass(frozen=True)
class Backend:
    name: str
    predict_command: str

    @property
    def project(self) -> Path:
        return REPOSITORY_ROOT / "backends" / self.name


BACKENDS = [Backend("ort", "xfeat-ort-predict"), Backend("tensorrt", "xfeat-trt-predict")]


@dataclass(frozen=True)
class BackendPair:
    keypoints: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    scores: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    descriptors: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    matches: npt.NDArray[np.int64]


@dataclass(frozen=True)
class Outcome:
    reference_features: list[Features]
    reference_matches: list[npt.NDArray[np.int64]]
    backend: list[BackendPair]


def _load_backend_pairs(directory: Path) -> list[BackendPair]:
    """Predictions with keypoints mapped from original pixels back to the model input.

    Inverse of the stretch resize, written from its definition: pixel centres at integers, so
    `x_model = (x_original + 1/2) W_model / W_original - 1/2`.
    """
    index = read_index(directory)
    pairs = []
    for entry in index.pairs:
        prediction = read_pair(directory, entry)

        def to_model(
            keypoints: npt.NDArray[np.float32], original_size: tuple[int, int]
        ) -> npt.NDArray[np.float32]:
            ratio = np.asarray(index.model_size) / np.asarray(original_size)
            return ((keypoints + 0.5) * ratio - 0.5).astype(np.float32)

        pairs.append(
            BackendPair(
                (
                    to_model(prediction.keypoints0, prediction.original_size0),
                    to_model(prediction.keypoints1, prediction.original_size1),
                ),
                (prediction.scores0, prediction.scores1),
                (prediction.descriptors0, prediction.descriptors1),
                prediction.matches,
            )
        )
    return pairs


def _gray_image(name: str) -> npt.NDArray[np.float32]:
    rgb = cv2.imread(str(MEGADEPTH / "images" / name), cv2.IMREAD_COLOR_RGB)
    assert rgb is not None, f"missing MegaDepth-1500 image {name}; run xfeat-megadepth1500 once"
    resized = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32).mean(axis=2) / 255.0)[None, None]


@pytest.fixture(
    scope="module",
    params=list(itertools.product(BACKENDS, CASES)),
    ids=lambda param: f"{param[0].name}-k{param[1].top_k}{'-chunked' if param[1].chunking else ''}",
)
def outcome(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Outcome:
    backend, case = request.param
    assert isinstance(backend, Backend)
    assert isinstance(case, Case)
    work = tmp_path_factory.mktemp(f"{backend.name}-k{case.top_k}-{bool(case.chunking)}")
    torch.use_deterministic_algorithms(mode=True)
    config = ReferenceConfig(GridNormalisation.PIXEL_CENTRE, pruning=False, top_k=case.top_k)
    pipeline = ReferencePipeline(config, torch.device("cpu"))
    xfeat_path = export_xfeat(
        pipeline.xfeat_network,
        work,
        height=HEIGHT,
        width=WIDTH,
        top_k=case.top_k,
        batch=1,
        chunking=case.chunking,
    )
    lighterglue_path = export_lighterglue(
        pipeline.lighterglue_network, work, top_k=case.top_k, batch=1, width=WIDTH, height=HEIGHT
    )

    names = [
        line.split()[:2]
        for line in (MEGADEPTH / "pairs_calibrated.txt").read_text().splitlines()[:PAIR_COUNT]
    ]
    reference_features: list[Features] = []
    reference_matches: list[npt.NDArray[np.int64]] = []
    for pair in names:
        pair_features = [pipeline.extract(torch.from_numpy(_gray_image(name))) for name in pair]
        reference_features += pair_features
        reference_matches.append(pipeline.match(*pair_features))

    manifest = work / "pairs.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "name": f"pair{index}",
                    "image0": str(MEGADEPTH / "images" / pair[0]),
                    "image1": str(MEGADEPTH / "images" / pair[1]),
                }
            )
            + "\n"
            for index, pair in enumerate(names)
        )
    )
    subprocess.run(  # noqa: S603  # fixed argv, no shell
        [  # noqa: S607
            "uv",
            "run",
            "--frozen",
            "--project",
            str(backend.project),
            backend.predict_command,
            "--xfeat",
            str(xfeat_path),
            "--lighterglue",
            str(lighterglue_path),
            "--manifest",
            str(manifest),
            "--resize",
            "stretch",
            "--out-dir",
            str(work / "predictions"),
        ],
        check=True,
    )
    return Outcome(reference_features, reference_matches, _load_backend_pairs(work / "predictions"))


def _pixel_set(keypoints: npt.NDArray[np.float32]) -> dict[tuple[int, int], int]:
    rounded = np.rint(keypoints).astype(np.int64)
    return {(int(x), int(y)): row for row, (x, y) in enumerate(rounded)}


def test_keypoints_agree_up_to_tie_swaps(outcome: Outcome) -> None:
    for index, reference in enumerate(outcome.reference_features):
        backend = outcome.backend[index // 2]
        side = index % 2
        reference_scores = reference.scores.numpy()
        backend_scores = backend.scores[side]
        reference_rows = _pixel_set(reference.keypoints.numpy())
        backend_rows = _pixel_set(backend.keypoints[side])
        assert len(reference_rows) == len(backend_rows) == len(reference_scores) > 0
        only_reference = reference_rows.keys() - backend_rows.keys()
        only_backend = backend_rows.keys() - reference_rows.keys()
        assert len(only_reference) <= MAX_TIE_SWAPS
        assert len(only_backend) <= MAX_TIE_SWAPS
        kth_score = min(reference_scores.min(), backend_scores.min())
        for point in only_reference:
            assert (
                abs(reference_scores[reference_rows[point]] - kth_score)
                <= TIE_RELATIVE_SCORE_GAP * kth_score
            )
        for point in only_backend:
            assert (
                abs(backend_scores[backend_rows[point]] - kth_score)
                <= TIE_RELATIVE_SCORE_GAP * kth_score
            )


def test_scores_and_descriptors_agree_on_common_keypoints(outcome: Outcome) -> None:
    for index, reference in enumerate(outcome.reference_features):
        backend = outcome.backend[index // 2]
        side = index % 2
        reference_rows = _pixel_set(reference.keypoints.numpy())
        backend_rows = _pixel_set(backend.keypoints[side])
        common = sorted(reference_rows.keys() & backend_rows.keys())
        reference_index = [reference_rows[p] for p in common]
        backend_index = [backend_rows[p] for p in common]
        score_difference = np.abs(
            reference.scores.numpy()[reference_index] - backend.scores[side][backend_index]
        )
        assert score_difference.max() <= MAX_SCORE_ABSOLUTE_DIFFERENCE
        cosine = np.einsum(
            "kd,kd->k",
            reference.descriptors.numpy()[reference_index],
            backend.descriptors[side][backend_index],
        )
        assert cosine.min() >= MIN_DESCRIPTOR_COSINE


def test_matches_agree_by_keypoint_coordinates(outcome: Outcome) -> None:
    for index, matches in enumerate(outcome.reference_matches):
        first, second = outcome.reference_features[2 * index : 2 * index + 2]
        backend = outcome.backend[index]
        reference_pairs = {
            (
                tuple(np.rint(first.keypoints[a].numpy()).astype(int)),
                tuple(np.rint(second.keypoints[b].numpy()).astype(int)),
            )
            for a, b in matches
        }
        backend_pairs = {
            (
                tuple(np.rint(backend.keypoints[0][a]).astype(int)),
                tuple(np.rint(backend.keypoints[1][b]).astype(int)),
            )
            for a, b in backend.matches
        }
        assert len(reference_pairs) > MIN_MATCHES
        jaccard = len(reference_pairs & backend_pairs) / len(reference_pairs | backend_pairs)
        assert jaccard >= MIN_MATCH_JACCARD
