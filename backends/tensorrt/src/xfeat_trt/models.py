"""TensorRT engines for the exported XFeat and LighterGlue graphs."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from cuda.core import Device

from xfeat_core.features import (
    Features,
    PaddedFeatures,
    compact_features,
    match_pairs,
    pad_features,
)
from xfeat_core.metadata import LighterGlueMetadata, XFeatMetadata, parse_metadata
from xfeat_trt.engine import DEFAULT_CACHE_DIR, BuildOptions, load_or_build_engine
from xfeat_trt.runner import EngineRunner

XFEAT_OUTPUTS = ("kpts", "scores", "desc", "valid")
LIGHTERGLUE_OUTPUTS = ("m0", "mscores0")


def _as_float32(array: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    return array.astype(np.float32) if np.issubdtype(array.dtype, np.floating) else array


class XFeatModel:
    """Exported XFeat graph: gray images in [0, 1] to features."""

    def __init__(
        self,
        path: Path,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        options: BuildOptions = BuildOptions(),  # noqa: B008  # frozen, safe default
        device_index: int = 0,
    ) -> None:
        """Load the engine for the ONNX file `path`, building it on a cache miss."""
        device = Device(device_index)
        device.set_current()
        loaded = load_or_build_engine(path, device, cache_dir, options)
        self.raw_metadata = loaded.metadata
        metadata = parse_metadata(self.raw_metadata)
        if not isinstance(metadata, XFeatMetadata):
            msg = f"{path} is a {metadata.model} graph, not xfeat"
            raise TypeError(msg)
        self.metadata = metadata
        self.runner = EngineRunner(loaded.engine, device)

    def run_raw(self, images: npt.NDArray[np.float32]) -> list[npt.NDArray[np.generic]]:
        """Run the graph on (B, 1, H, W) images; returns kpts, scores, desc, valid."""
        outputs = self.runner.run({"image": images})
        return [_as_float32(outputs[name]) for name in XFEAT_OUTPUTS]

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

    def __init__(
        self,
        path: Path,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        options: BuildOptions = BuildOptions(),  # noqa: B008  # frozen, safe default
        device_index: int = 0,
    ) -> None:
        """Load the engine for the ONNX file `path`, building it on a cache miss."""
        device = Device(device_index)
        device.set_current()
        loaded = load_or_build_engine(path, device, cache_dir, options)
        self.raw_metadata = loaded.metadata
        metadata = parse_metadata(self.raw_metadata)
        if not isinstance(metadata, LighterGlueMetadata):
            msg = f"{path} is a {metadata.model} graph, not lighterglue"
            raise TypeError(msg)
        self.metadata = metadata
        self.runner = EngineRunner(loaded.engine, device)

    def pad(self, features: Features) -> PaddedFeatures:
        """Pad `features` to this graph's K."""
        return pad_features(features, self.metadata.top_k)

    @staticmethod
    def feeds(first: PaddedFeatures, second: PaddedFeatures) -> dict[str, npt.NDArray[np.generic]]:
        """Engine inputs of one pair, with the batch axis added."""
        feeds = {}
        for side, padded in enumerate((first, second)):
            feeds[f"kpts{side}"] = padded.keypoints[None]
            feeds[f"desc{side}"] = padded.descriptors[None]
            feeds[f"valid{side}"] = padded.valid[None]
            feeds[f"image_size{side}"] = padded.image_size[None]
        return feeds

    def run_raw(
        self, first: PaddedFeatures, second: PaddedFeatures
    ) -> list[npt.NDArray[np.generic]]:
        """Run the graph on one pair; returns m0 (1, K) and mscores0 (1, K)."""
        outputs = self.runner.run(self.feeds(first, second))
        return [_as_float32(outputs[name]) for name in LIGHTERGLUE_OUTPUTS]

    def match(self, first: Features, second: Features) -> npt.NDArray[np.int64]:
        """(M, 2) index pairs into `first` and `second` of the mutual matches."""
        if self.metadata.batch != 1:
            msg = "match takes one pair; this graph has batch > 1"
            raise ValueError(msg)
        matches0 = self.run_raw(self.pad(first), self.pad(second))[0]
        return match_pairs(np.asarray(matches0[0], dtype=np.int64))
