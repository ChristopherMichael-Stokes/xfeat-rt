"""Batch ladder: XFeat and LighterGlue exports differing only in batch size, in one directory.

`xfeat-export --batch 1 2 4 ...` writes one; every other setting of the XFeat files (size, K,
precision, ...) must agree, likewise for the LighterGlue files, and both must share K.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from xfeat_core.metadata import LighterGlueMetadata, XFeatMetadata, parse_metadata
from xfeat_trt.engine import read_onnx_metadata

_SINGLE_BATCH = {"batch": 1}


class LadderError(ValueError):
    """The directory does not hold one consistent batch ladder."""


@dataclass(frozen=True)
class BatchLadder:
    """ONNX files of one ladder by batch size, and the settings they share.

    Attributes:
        xfeat: Batch size to XFeat file.
        lighterglue: Batch size to LighterGlue file.
        xfeat_metadata: XFeat settings of the ladder, with batch 1.
    """

    xfeat: dict[int, Path]
    lighterglue: dict[int, Path]
    xfeat_metadata: XFeatMetadata

    def onnx_sha256(self) -> dict[str, str]:
        """SHA-256 of every file of the ladder, by file name."""
        paths = [*self.xfeat.values(), *self.lighterglue.values()]
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def find_ladder(directory: Path) -> BatchLadder:
    """The batch ladder of the `.onnx` files in `directory`.

    Raises:
        LadderError: If a model is missing, a batch size repeats, or settings other than the batch
            size differ between files of one model.
    """
    xfeat: dict[int, tuple[Path, XFeatMetadata]] = {}
    lighterglue: dict[int, tuple[Path, LighterGlueMetadata]] = {}
    for path in sorted(directory.glob("*.onnx")):
        metadata = parse_metadata(read_onnx_metadata(path))
        if isinstance(metadata, XFeatMetadata):
            _add_rung(xfeat, path, metadata)
        else:
            _add_rung(lighterglue, path, metadata)
    xfeat_metadata = _shared_settings(directory, xfeat, "xfeat")
    lighterglue_metadata = _shared_settings(directory, lighterglue, "lighterglue")
    if xfeat_metadata.top_k != lighterglue_metadata.top_k:
        msg = (
            f"{directory}: XFeat K={xfeat_metadata.top_k}, "
            f"LighterGlue K={lighterglue_metadata.top_k}"
        )
        raise LadderError(msg)
    return BatchLadder(
        {batch: path for batch, (path, _) in sorted(xfeat.items())},
        {batch: path for batch, (path, _) in sorted(lighterglue.items())},
        xfeat_metadata,
    )


def _add_rung[Metadata: (XFeatMetadata, LighterGlueMetadata)](
    rungs: dict[int, tuple[Path, Metadata]], path: Path, metadata: Metadata
) -> None:
    if metadata.batch in rungs:
        msg = f"{path} and {rungs[metadata.batch][0]} both have batch {metadata.batch}"
        raise LadderError(msg)
    rungs[metadata.batch] = (path, metadata)


def _shared_settings[Metadata: (XFeatMetadata, LighterGlueMetadata)](
    directory: Path, rungs: dict[int, tuple[Path, Metadata]], model: str
) -> Metadata:
    if not rungs:
        msg = f"{directory} holds no {model} graph"
        raise LadderError(msg)
    settings = {metadata.model_copy(update=_SINGLE_BATCH) for _, metadata in rungs.values()}
    if len(settings) > 1:
        msg = f"{directory}: {model} graphs differ in more than the batch size"
        raise LadderError(msg)
    return settings.pop()
