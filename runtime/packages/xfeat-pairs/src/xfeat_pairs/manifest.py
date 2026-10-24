"""Pair manifest: JSON Lines, one image pair per line, with optional ground truth.

Example line::

    {"name": "a_b", "image0": "a.jpg", "image1": "b.jpg",
     "ground_truth": {"kind": "homography", "matrix": [[1,0,0],[0,1,0],[0,0,1]]}}

Image paths are relative to the manifest file. A `relative_pose` ground truth gives `T_0to1`
(rotation `R`, translation `t`, so `x1 = R x0 + t` in camera coordinates) and the intrinsics of
both cameras at the original image resolution. A `homography` maps image 0 to image 1 in original
pixel coordinates with pixel centres at integers.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RelativePose(_Frozen):
    """Relative camera pose and intrinsics of a pair."""

    kind: Literal["relative_pose"]
    rotation: Matrix3
    translation: Vector3
    intrinsics0: Matrix3
    intrinsics1: Matrix3


class Homography(_Frozen):
    """Planar homography from image 0 to image 1."""

    kind: Literal["homography"]
    matrix: Matrix3


GroundTruth = Annotated[RelativePose | Homography, Field(discriminator="kind")]


class PairRecord(_Frozen):
    """One image pair."""

    name: str = Field(min_length=1)
    image0: Path
    image1: Path
    ground_truth: GroundTruth | None = None


class ManifestError(ValueError):
    """The manifest file is malformed."""


class _Manifest(_Frozen):
    pairs: list[PairRecord]

    @model_validator(mode="after")
    def _names_are_unique(self) -> "_Manifest":
        names = [pair.name for pair in self.pairs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"duplicate pair names: {duplicates}"
            raise ValueError(msg)
        return self


def load_manifest(path: Path) -> list[PairRecord]:
    """Pairs of the JSON Lines manifest at `path`, with image paths resolved against it.

    Raises:
        ManifestError: On a malformed line, a duplicate name or a missing image.
    """
    records = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = PairRecord.model_validate_json(line)
        except ValueError as error:
            msg = f"{path}:{number}: {error}"
            raise ManifestError(msg) from error
        records.append(
            record.model_copy(
                update={
                    "image0": path.parent / record.image0,
                    "image1": path.parent / record.image1,
                }
            )
        )
    try:
        _Manifest(pairs=records)
    except ValueError as error:
        msg = f"{path}: {error}"
        raise ManifestError(msg) from error
    missing = [p for r in records for p in (r.image0, r.image1) if not p.is_file()]
    if missing:
        msg = f"{path}: missing images: {[str(p) for p in missing]}"
        raise ManifestError(msg)
    return records
