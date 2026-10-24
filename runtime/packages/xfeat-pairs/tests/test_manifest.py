import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from xfeat_pairs.manifest import Homography, ManifestError, RelativePose, load_manifest

IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _write(tmp_path: Path, lines: Sequence[Mapping[str, object]]) -> Path:
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(b"")
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return manifest


def test_paths_resolve_against_the_manifest_directory(tmp_path: Path) -> None:
    manifest = _write(tmp_path, [{"name": "p", "image0": "a.png", "image1": "b.png"}])
    (record,) = load_manifest(manifest)
    assert record.image0 == tmp_path / "a.png"
    assert record.ground_truth is None


def test_ground_truth_kinds_are_parsed(tmp_path: Path) -> None:
    pose = {
        "kind": "relative_pose",
        "rotation": IDENTITY,
        "translation": [0, 0, 1],
        "intrinsics0": IDENTITY,
        "intrinsics1": IDENTITY,
    }
    lines = [
        {"name": "p", "image0": "a.png", "image1": "b.png", "ground_truth": pose},
        {
            "name": "h",
            "image0": "a.png",
            "image1": "b.png",
            "ground_truth": {"kind": "homography", "matrix": IDENTITY},
        },
    ]
    pose_record, homography_record = load_manifest(_write(tmp_path, lines))
    assert isinstance(pose_record.ground_truth, RelativePose)
    assert isinstance(homography_record.ground_truth, Homography)


def test_duplicate_names_are_rejected(tmp_path: Path) -> None:
    line: dict[str, object] = {"name": "p", "image0": "a.png", "image1": "b.png"}
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write(tmp_path, [line, line]))


def test_error_names_the_line(tmp_path: Path) -> None:
    good: dict[str, object] = {"name": "p", "image0": "a.png", "image1": "b.png"}
    bad: dict[str, object] = {"name": "q", "image0": "a.png"}
    with pytest.raises(ManifestError, match=r"pairs\.jsonl:2"):
        load_manifest(_write(tmp_path, [good, bad]))


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    manifest = _write(tmp_path, [{"name": "p", "image0": "a.png", "image1": "gone.png"}])
    with pytest.raises(ManifestError, match=r"gone\.png"):
        load_manifest(manifest)
