"""TensorRT engines for the exported XFeat and LighterGlue graphs."""

from collections.abc import Sequence
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
        self.built = loaded.built
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
        self.built = loaded.built
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
    def feeds(
        pairs: Sequence[tuple[PaddedFeatures, PaddedFeatures]],
    ) -> dict[str, npt.NDArray[np.generic]]:
        """Engine inputs of a batch of pairs, stacked on the batch axis."""
        feeds = {}
        for side in (0, 1):
            padded = [pair[side] for pair in pairs]
            feeds[f"kpts{side}"] = np.stack([features.keypoints for features in padded])
            feeds[f"desc{side}"] = np.stack([features.descriptors for features in padded])
            feeds[f"valid{side}"] = np.stack([features.valid for features in padded])
            feeds[f"image_size{side}"] = np.stack([features.image_size for features in padded])
        return feeds

    def run_raw(
        self, pairs: Sequence[tuple[PaddedFeatures, PaddedFeatures]]
    ) -> list[npt.NDArray[np.generic]]:
        """Run the graph on a batch of pairs; returns m0 (B, K) and mscores0 (B, K)."""
        outputs = self.runner.run(self.feeds(pairs))
        return [_as_float32(outputs[name]) for name in LIGHTERGLUE_OUTPUTS]

    def match_batch(
        self, pairs: Sequence[tuple[Features, Features]]
    ) -> list[npt.NDArray[np.int64]]:
        """(M, 2) index pairs of the mutual matches of each pair; takes exactly B pairs."""
        if len(pairs) != self.metadata.batch:
            msg = f"{len(pairs)} pairs given; this graph takes {self.metadata.batch}"
            raise ValueError(msg)
        matches0 = self.run_raw([(self.pad(first), self.pad(second)) for first, second in pairs])[0]
        return [match_pairs(np.asarray(row, dtype=np.int64)) for row in matches0]

    def match(self, first: Features, second: Features) -> npt.NDArray[np.int64]:
        """(M, 2) index pairs into `first` and `second` of the mutual matches; batch-1 graphs."""
        return self.match_batch([(first, second)])[0]
