"""Cross-workspace contract: export with `reference`, run with `backends/ort`, compare.

The ORT side runs in its own virtualenv through the `xfeat-ort-predict` CLI, so no torch is
imported there. Reference is FP32 PyTorch on CPU with the `pixel_centre` grid and pruning off.
"""

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
from xfeat_reference.pipeline import (
    Features,
    GridNormalisation,
    ReferenceConfig,
    ReferencePipeline,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ORT_PROJECT = REPOSITORY_ROOT / "backends" / "ort"
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


CASES = [Case(2048, None), Case(4096, DEFAULT_CHUNKING)]


@dataclass(frozen=True)
class OrtPair:
    keypoints: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    scores: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    descriptors: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    matches: npt.NDArray[np.int64]


@dataclass(frozen=True)
class Outcome:
    reference_features: list[Features]
    reference_matches: list[npt.NDArray[np.int64]]
    ort: list[OrtPair]


def _load_ort_pair(path: Path) -> OrtPair:
    with np.load(path) as archive:

        def float_pair(name: str) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            return (
                np.asarray(archive[f"{name}0"], dtype=np.float32),
                np.asarray(archive[f"{name}1"], dtype=np.float32),
            )

        return OrtPair(
            float_pair("keypoints"),
            float_pair("scores"),
            float_pair("descriptors"),
            np.asarray(archive["matches"], dtype=np.int64),
        )


def _gray_image(name: str) -> npt.NDArray[np.float32]:
    rgb = cv2.imread(str(MEGADEPTH / "images" / name), cv2.IMREAD_COLOR_RGB)
    assert rgb is not None, f"missing MegaDepth-1500 image {name}; run xfeat-megadepth1500 once"
    resized = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32).mean(axis=2) / 255.0)[None, None]


@pytest.fixture(scope="module", params=CASES, ids=lambda case: f"k{case.top_k}")
def outcome(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Outcome:
    case = request.param
    assert isinstance(case, Case)
    work = tmp_path_factory.mktemp(f"k{case.top_k}")
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
    image_paths: list[Path] = []
    reference_features: list[Features] = []
    reference_matches: list[npt.NDArray[np.int64]] = []
    for index, pair in enumerate(names):
        pair_features = []
        for view, name in enumerate(pair):
            image = _gray_image(name)
            path = work / f"pair{index}_{view}.npy"
            np.save(path, image)
            image_paths.append(path)
            pair_features.append(pipeline.extract(torch.from_numpy(image)))
        reference_features += pair_features
        reference_matches.append(pipeline.match(*pair_features))

    subprocess.run(  # noqa: S603  # fixed argv, no shell
        [  # noqa: S607
            "uv",
            "run",
            "--frozen",
            "--project",
            str(ORT_PROJECT),
            "xfeat-ort-predict",
            "--xfeat",
            str(xfeat_path),
            "--lighterglue",
            str(lighterglue_path),
            "--images",
            *map(str, image_paths),
            "--out-dir",
            str(work / "ort"),
        ],
        check=True,
    )
    ort = [_load_ort_pair(work / "ort" / f"pair{i}.npz") for i in range(PAIR_COUNT)]
    return Outcome(reference_features, reference_matches, ort)


def _pixel_set(keypoints: npt.NDArray[np.float32]) -> dict[tuple[int, int], int]:
    rounded = np.rint(keypoints).astype(np.int64)
    return {(int(x), int(y)): row for row, (x, y) in enumerate(rounded)}


def test_keypoints_agree_up_to_tie_swaps(outcome: Outcome) -> None:
    for index, reference in enumerate(outcome.reference_features):
        ort = outcome.ort[index // 2]
        side = index % 2
        reference_scores = reference.scores.numpy()
        ort_scores = ort.scores[side]
        reference_rows = _pixel_set(reference.keypoints.numpy())
        ort_rows = _pixel_set(ort.keypoints[side])
        assert len(reference_rows) == len(ort_rows) == len(reference_scores) > 0
        only_reference = reference_rows.keys() - ort_rows.keys()
        only_ort = ort_rows.keys() - reference_rows.keys()
        assert len(only_reference) <= MAX_TIE_SWAPS
        assert len(only_ort) <= MAX_TIE_SWAPS
        kth_score = min(reference_scores.min(), ort_scores.min())
        for point in only_reference:
            assert (
                abs(reference_scores[reference_rows[point]] - kth_score)
                <= TIE_RELATIVE_SCORE_GAP * kth_score
            )
        for point in only_ort:
            assert (
                abs(ort_scores[ort_rows[point]] - kth_score) <= TIE_RELATIVE_SCORE_GAP * kth_score
            )


def test_scores_and_descriptors_agree_on_common_keypoints(outcome: Outcome) -> None:
    for index, reference in enumerate(outcome.reference_features):
        ort = outcome.ort[index // 2]
        side = index % 2
        reference_rows = _pixel_set(reference.keypoints.numpy())
        ort_rows = _pixel_set(ort.keypoints[side])
        common = sorted(reference_rows.keys() & ort_rows.keys())
        reference_index = [reference_rows[p] for p in common]
        ort_index = [ort_rows[p] for p in common]
        score_difference = np.abs(
            reference.scores.numpy()[reference_index] - ort.scores[side][ort_index]
        )
        assert score_difference.max() <= MAX_SCORE_ABSOLUTE_DIFFERENCE
        cosine = np.einsum(
            "kd,kd->k",
            reference.descriptors.numpy()[reference_index],
            ort.descriptors[side][ort_index],
        )
        assert cosine.min() >= MIN_DESCRIPTOR_COSINE


def test_matches_agree_by_keypoint_coordinates(outcome: Outcome) -> None:
    for index, matches in enumerate(outcome.reference_matches):
        first, second = outcome.reference_features[2 * index : 2 * index + 2]
        ort = outcome.ort[index]
        reference_pairs = {
            (
                tuple(np.rint(first.keypoints[a].numpy()).astype(int)),
                tuple(np.rint(second.keypoints[b].numpy()).astype(int)),
            )
            for a, b in matches
        }
        ort_pairs = {
            (
                tuple(np.rint(ort.keypoints[0][a]).astype(int)),
                tuple(np.rint(ort.keypoints[1][b]).astype(int)),
            )
            for a, b in ort.matches
        }
        assert len(reference_pairs) > MIN_MATCHES
        jaccard = len(reference_pairs & ort_pairs) / len(reference_pairs | ort_pairs)
        assert jaccard >= MIN_MATCH_JACCARD
