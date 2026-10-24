from pathlib import Path

from xfeat_evaluation.megadepth_manifest import manifest_lines
from xfeat_pairs.manifest import RelativePose, load_manifest


def test_pairs_calibrated_line_becomes_a_relative_pose_pair(tmp_path: Path) -> None:
    root = tmp_path / "megadepth1500"
    (root / "images").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (root / "images" / name).write_bytes(b"")
    k0 = "1 2 3 4 5 6 7 8 9"
    k1 = "11 12 13 14 15 16 17 18 19"
    rotation = "21 22 23 24 25 26 27 28 29"
    translation = "31 32 33"
    (root / "pairs_calibrated.txt").write_text(f"a.jpg b.jpg {k0} {k1} {rotation} {translation}\n")

    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(manifest_lines(tmp_path, pair_limit=10)) + "\n")
    (record,) = load_manifest(manifest)

    truth = record.ground_truth
    assert isinstance(truth, RelativePose)
    assert truth.intrinsics0 == ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    assert truth.intrinsics1[2] == (17, 18, 19)
    assert truth.rotation[1] == (24, 25, 26)
    assert truth.translation == (31, 32, 33)
    assert record.image0 == root / "images" / "a.jpg"
