"""XFeat sparse extraction and LighterGlue matching, run through the pinned upstream code."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import torch

from xfeat_reference.upstream import UpstreamCheckout, load_upstream

PRUNING_DISABLED = -1.0
"""kornia LightGlue `width_confidence` value that turns point pruning off."""


class GridNormalisation(StrEnum):
    """How keypoint pixel positions map to `grid_sample` coordinates in [-1, 1].

    Upstream calls `grid_sample(align_corners=False)`, where -1 and 1 are the outer edges of the
    corner pixels (PyTorch `torch.nn.functional.grid_sample` docs).

    - `UPSTREAM_EXACT`: upstream's `g = 2x / (W - 1) - 1`. Pixel x lands on source coordinate
      `x W' / (W - 1) - 1/2` in a map of width W', a skew that grows towards the far edge. In
      the full-resolution heatmap the last row and column land on `W - 1/2`, outside the map, so
      their score is 0 and they are never kept.
    - `PIXEL_CENTRE`: `g = 2 (x + 1/2) / W - 1`, the inverse of `align_corners=False`, so pixel
      x samples its own centre at every resolution and edge pixels stay in bounds.

    Here x is a pixel index (pixel centres at integers) and W the input width; likewise for y, H.
    """

    UPSTREAM_EXACT = "upstream_exact"
    PIXEL_CENTRE = "pixel_centre"


@dataclass(frozen=True)
class ReferenceConfig:
    """Settings of one reference pipeline.

    Attributes:
        grid: Keypoint sampling convention.
        pruning: LighterGlue point pruning (upstream default 0.95 width confidence) when true.
        top_k: Keypoints kept per image.
        detection_threshold: Minimum heatmap value for an NMS candidate (upstream default 0.05).
        match_threshold: Minimum LighterGlue match confidence (upstream default 0.1).
    """

    grid: GridNormalisation
    pruning: bool
    top_k: int
    detection_threshold: float = 0.05
    match_threshold: float = 0.1


@dataclass(frozen=True)
class Features:
    """Sparse features of one image.

    Attributes:
        keypoints: (N, 2) float x, y in input-image pixels, pixel centres at integers.
        scores: (N,) keypoint scores.
        descriptors: (N, 64) L2-normalised descriptors.
        image_size: Input image (width, height) in pixels.
    """

    keypoints: torch.Tensor
    scores: torch.Tensor
    descriptors: torch.Tensor
    image_size: tuple[int, int]


class ReferencePipeline:
    """Pinned upstream XFeat and LighterGlue on one device, configured by a `ReferenceConfig`."""

    def __init__(self, config: ReferenceConfig, device: torch.device) -> None:
        """Load the pinned upstream code and weights onto `device`."""
        self.config = config
        self.device = device
        self._upstream = load_upstream()
        self._xfeat = _build_xfeat(self._upstream, config, device)

    @property
    def xfeat_network(self) -> Any:  # noqa: ANN401  # upstream XFeatModel is untyped
        """The upstream XFeat backbone with the pinned weights loaded."""
        return self._xfeat.net

    @property
    def lighterglue_network(self) -> Any:  # noqa: ANN401  # kornia LightGlue
        """The kornia LightGlue network with the pinned weights loaded."""
        return self._xfeat.lighterglue.net

    def extract(self, image: torch.Tensor) -> Features:
        """Detect and describe keypoints in one image.

        Args:
            image: (1, C, H, W) float image, grayscale or RGB.

        Returns:
            Features of the image, in its own pixel coordinates.
        """
        with self._grid_normalisation():
            upstream_features = self._xfeat.detectAndCompute(image.to(self.device))[0]
        height, width = image.shape[-2:]
        return Features(
            keypoints=upstream_features["keypoints"],
            scores=upstream_features["scores"],
            descriptors=upstream_features["descriptors"],
            image_size=(width, height),
        )

    def match(self, features0: Features, features1: Features) -> npt.NDArray[np.int64]:
        """Match two images' features with LighterGlue.

        Returns:
            (M, 2) indices into `features0` and `features1` of each mutual match.
        """
        _, _, indices = self._xfeat.match_lighterglue(
            _as_upstream(features0), _as_upstream(features1), min_conf=self.config.match_threshold
        )
        return np.asarray(indices, dtype=np.int64)

    @contextmanager
    def _grid_normalisation(self) -> Generator[None]:
        if self.config.grid is GridNormalisation.UPSTREAM_EXACT:
            yield
            return
        # detectAndCompute builds its samplers inside the call, so the class method is the only seam
        sampler_class = self._upstream.interpolator_module.InterpolateSparse2d
        with mock.patch.object(sampler_class, "normgrid", _pixel_centre_normgrid):
            yield


def _build_xfeat(upstream: UpstreamCheckout, config: ReferenceConfig, device: torch.device) -> Any:  # noqa: ANN401  # upstream XFeat is untyped, only known at runtime through torch.hub
    xfeat = upstream.xfeat_class(
        weights=None, top_k=config.top_k, detection_threshold=config.detection_threshold
    )
    xfeat.net.load_state_dict(
        torch.load(upstream.xfeat_weights, map_location=device, weights_only=True)
    )
    xfeat.dev = device
    xfeat.net.to(device)

    lighterglue = upstream.lighterglue_module.LighterGlue(weights=str(upstream.lighterglue_weights))
    if not config.pruning:
        lighterglue.net.conf.width_confidence = PRUNING_DISABLED
    lighterglue.dev = device
    lighterglue.net.to(device)
    xfeat.lighterglue = lighterglue
    return xfeat


def _pixel_centre_normgrid(_sampler: object, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    size = torch.tensor([W, H], device=x.device, dtype=x.dtype)
    return 2.0 * (x + 0.5) / size - 1.0


def _as_upstream(features: Features) -> dict[str, Any]:
    return {
        "keypoints": features.keypoints,
        "scores": features.scores,
        "descriptors": features.descriptors,
        "image_size": features.image_size,
    }
