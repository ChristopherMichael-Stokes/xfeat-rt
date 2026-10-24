"""ONNX Runtime sessions for the exported XFeat and LighterGlue graphs."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from onnxruntime import (
    InferenceSession,
    SessionOptions,  # ty: ignore[possibly-missing-import]  # onnxruntime imports it in a try block
)

from xfeat_core.features import (
    Features,
    PaddedFeatures,
    compact_features,
    match_pairs,
    pad_features,
)
from xfeat_core.metadata import LighterGlueMetadata, XFeatMetadata, parse_metadata

CPU_PROVIDER = "CPUExecutionProvider"


def _session(path: Path, intra_op_threads: int) -> InferenceSession:
    options = SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    return InferenceSession(path, options, providers=[CPU_PROVIDER])


class XFeatModel:
    """Exported XFeat graph: gray images in [0, 1] to features."""

    def __init__(self, path: Path, intra_op_threads: int = 0) -> None:
        """Load `path`; `intra_op_threads` 0 lets ORT choose."""
        self._session = _session(path, intra_op_threads)
        metadata = parse_metadata(self._session.get_modelmeta().custom_metadata_map)
        if not isinstance(metadata, XFeatMetadata):
            msg = f"{path} is a {metadata.model} graph, not xfeat"
            raise TypeError(msg)
        self.metadata = metadata

    def run_raw(self, images: npt.NDArray[np.float32]) -> list[npt.NDArray[np.generic]]:
        """Run the graph on (B, 1, H, W) images; returns kpts, scores, desc, valid."""
        return [np.asarray(output) for output in self._session.run(None, {"image": images})]

    def extract(self, images: npt.NDArray[np.float32]) -> list[Features]:
        """Features of each image of a (B, 1, H, W) float32 batch in [0, 1]."""
        meta = self.metadata
        expected = (meta.batch, 1, meta.height, meta.width)
        if images.shape != expected:
            msg = f"images have shape {images.shape}, the graph takes {expected}"
            raise ValueError(msg)
        keypoints, scores, descriptors, valid = self.run_raw(images)
        size = (meta.width, meta.height)
        return [
            compact_features(keypoints[i], scores[i], descriptors[i], valid[i], size)
            for i in range(meta.batch)
        ]


class LighterGlueModel:
    """Exported LighterGlue graph over feature sets padded to K."""

    def __init__(self, path: Path, intra_op_threads: int = 0) -> None:
        """Load `path`; `intra_op_threads` 0 lets ORT choose."""
        self._session = _session(path, intra_op_threads)
        metadata = parse_metadata(self._session.get_modelmeta().custom_metadata_map)
        if not isinstance(metadata, LighterGlueMetadata):
            msg = f"{path} is a {metadata.model} graph, not lighterglue"
            raise TypeError(msg)
        self.metadata = metadata

    def pad(self, features: Features) -> PaddedFeatures:
        """Pad `features` to this graph's K."""
        return pad_features(features, self.metadata.top_k)

    def run_raw(
        self, first: PaddedFeatures, second: PaddedFeatures
    ) -> list[npt.NDArray[np.generic]]:
        """Run the graph on one pair; returns m0 (1, K) and mscores0 (1, K)."""
        feeds = {}
        for side, padded in enumerate((first, second)):
            feeds[f"kpts{side}"] = padded.keypoints[None]
            feeds[f"desc{side}"] = padded.descriptors[None]
            feeds[f"valid{side}"] = padded.valid[None]
            feeds[f"image_size{side}"] = padded.image_size[None]
        return [np.asarray(output) for output in self._session.run(None, feeds)]

    def match(self, first: Features, second: Features) -> npt.NDArray[np.int64]:
        """(M, 2) index pairs into `first` and `second` of the mutual matches."""
        if self.metadata.batch != 1:
            msg = "match takes one pair; this graph has batch > 1"
            raise ValueError(msg)
        matches0 = self.run_raw(self.pad(first), self.pad(second))[0]
        return match_pairs(np.asarray(matches0[0], dtype=np.int64))
