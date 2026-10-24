"""XFeat with detection, top-K selection and descriptor sampling inside one static graph.

Follows optocycle/accelerated_features `deployment/export_xfeat_onnx.py` at
f8baecb4d1e5e6a78481541097f11b458be21fb4: dense peak map from max-pool equality and a threshold,
dense score map, `TopK(K)`, bicubic `GridSample` of the descriptor map. Chunked top-K stage 1
follows fabio-sim/LightGlue-ONNX `lightglue_dynamo/models/raco.py::_chunked_topk` at
d12b4ba1632f558234e3f084e1f3d8bdf9147890.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, override

import torch
from torch import nn
from torch.nn import functional

from xfeat_export.precision import Precision

NMS_KERNEL = 5
"""Upstream `XFeat.NMS` kernel size."""
HEATMAP_CELL = 8
"""Upstream keypoint head: 64 logits per 8x8 cell plus a dustbin."""
INVALID_SCORE = -1.0
"""Score of a non-candidate pixel; upstream's padding marker (`xfeat.py:80`)."""


@dataclass(frozen=True)
class ChunkedTopK:
    """Settings that keep every `TopK` node at K <= `max_k_per_op`.

    Attributes:
        chunk_size: Flattened score-map elements per chunk (fabio-sim default 65536).
        max_k_per_op: Largest K of any single `TopK` node. TensorRT builds K <= 3840 only.
    """

    chunk_size: int
    max_k_per_op: int


DEFAULT_CHUNKING = ChunkedTopK(chunk_size=65536, max_k_per_op=2048)


def pixel_centre_grid(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Pixel positions (x, y) to `grid_sample` coordinates, `g = 2 (p + 1/2) / size - 1`."""
    size = torch.tensor([width, height], dtype=points.dtype)
    return 2.0 * (points + 0.5) / size - 1.0


def chunked_topk(
    flat: torch.Tensor, k: int, settings: ChunkedTopK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the last axis of (B, N) with every `TopK` node at K <= `settings.max_k_per_op`.

    Stage 1 (fabio-sim): `TopK(c)` in each disjoint chunk, c = min(k, max_k_per_op).
    Stage 2: when k > c, k / c passes of `TopK(c)` over the union of chunk winners, each pass
    setting the previous winners to -inf with `ScatterElements`, concatenated in descending order.
    Exact whenever no chunk holds more than c members of the global top-k, which is always true for
    c = k. No published precedent for the pass split.

    Returns:
        Values and flat indices, each (B, k).
    """
    per_op = min(k, settings.max_k_per_op)
    passes = math.ceil(k / per_op)
    if passes * per_op != k:
        msg = f"k={k} must be a multiple of {per_op}"
        raise ValueError(msg)
    batch, length = flat.shape
    padding = -length % settings.chunk_size
    chunks = functional.pad(flat, (0, padding), value=-torch.inf).reshape(
        batch, -1, settings.chunk_size
    )
    local_values, local_indices = chunks.topk(per_op, dim=-1, sorted=False)
    offsets = torch.arange(chunks.shape[1], dtype=local_indices.dtype).reshape(1, -1, 1)
    union_indices = (local_indices + offsets * settings.chunk_size).flatten(1)
    union_values = local_values.flatten(1)
    values, orders = [], []
    for _ in range(passes):
        pass_values, pass_orders = union_values.topk(per_op, dim=-1)
        values.append(pass_values)
        orders.append(pass_orders)
        union_values = union_values.scatter(1, pass_orders, -torch.inf)
    return torch.cat(values, 1), union_indices.gather(1, torch.cat(orders, 1))


def peak_mask(heatmap: torch.Tensor, threshold: float) -> torch.Tensor:
    """Pixels that equal their 5x5 neighbourhood maximum and exceed `threshold`."""
    local_maximum = functional.max_pool2d(heatmap, NMS_KERNEL, stride=1, padding=NMS_KERNEL // 2)
    return (heatmap == local_maximum) & (heatmap > threshold)


def dense_reliability(reliability: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Reliability map (1/8 resolution) bilinearly sampled at every full-resolution pixel centre.

    Equal to a dense `grid_sample(align_corners=False, padding_mode="zeros")` at pixel-centre
    coordinates, computed as a 1-px zero pad, bilinear x8 resize, crop. The pad keeps the resize
    source coordinate `(x + 8 + 1/2) / 8 - 1/2` inside the padded map, so no edge clamp applies and
    the border matches the zeros padding. ORT CPU `GridSample` costs 60 ms at 1.2 Mpx against 3 ms
    for this form.
    """
    padded = functional.pad(reliability, (1, 1, 1, 1))
    upsampled = functional.interpolate(
        padded, scale_factor=HEATMAP_CELL, mode="bilinear", align_corners=False
    )
    return upsampled[..., HEATMAP_CELL : HEATMAP_CELL + height, HEATMAP_CELL : HEATMAP_CELL + width]


def _unfold_as_pixel_unshuffle(x: torch.Tensor, ws: int = 2) -> torch.Tensor:
    """Upstream `_unfold2d` for one channel; the tensors are identical (checked in the tests)."""
    return functional.pixel_unshuffle(x, ws)


class StaticXFeat(nn.Module):
    """Image (B, 1, H, W) in [0, 1] to keypoints, scores, descriptors and a validity mask.

    Outputs are `kpts` (B, K, 2) float x, y in pixels, `scores` (B, K) with 0 where invalid,
    `desc` (B, K, 64) L2-normalised and `valid` (B, K). Invalid slots come last.
    Sampling follows the `pixel_centre` grid convention.

    `pixel_unshuffle` and `pixel_shuffle` replace upstream's 6-D unfold/permute, which costs 43 ms
    on ORT CPU. Both are bit-identical to upstream. Precedent: BCDL XFeat port.
    """

    def __init__(  # noqa: PLR0913  # one graph parameter each, set at export
        self,
        network: Any,  # noqa: ANN401  # upstream XFeatModel is untyped, known only at runtime
        *,
        height: int,
        width: int,
        top_k: int,
        detection_threshold: float,
        chunking: ChunkedTopK | None,
        precision: Precision,
    ) -> None:
        """Wrap `network` (a copy is patched, the argument is left unchanged)."""
        super().__init__()
        self.network = _with_pixel_unshuffle(network).to(precision.dtype)
        self.compute_dtype = precision.dtype
        self.height, self.width, self.top_k = height, width, top_k
        self.detection_threshold = detection_threshold
        self.chunking = chunking

    @override
    def forward(  # ty: ignore[invalid-mutable-override]  # torch types forward as a writable attribute
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run the graph on `image`."""
        features, logits, reliability = (
            output.to(torch.float32) for output in self.network(image.to(self.compute_dtype))
        )
        features = functional.normalize(features, dim=1)
        heatmap = functional.pixel_shuffle(functional.softmax(logits, 1)[:, :64], HEATMAP_CELL)
        peaks = peak_mask(heatmap, self.detection_threshold)
        # pixel_centre: upstream's `nearest` sample of the full-resolution heatmap is the pixel
        score_map = torch.where(
            peaks,
            heatmap * dense_reliability(reliability, self.height, self.width),
            INVALID_SCORE,
        )
        flat = score_map.reshape(score_map.shape[0], self.height * self.width)
        if self.chunking is None:
            scores, index = flat.topk(self.top_k, dim=1)
        else:
            scores, index = chunked_topk(flat, self.top_k, self.chunking)
        # integer div/mod: a float flat index overflows FP16 (fabio-sim)
        x = index % self.width
        y = torch.div(index, self.width, rounding_mode="floor")
        keypoints = torch.stack([x, y], -1).to(torch.float32)
        valid = scores > 0
        grid = pixel_centre_grid(keypoints, self.height, self.width)[:, :, None, :]
        descriptors = functional.grid_sample(features, grid, mode="bicubic", align_corners=False)
        descriptors = functional.normalize(descriptors[..., 0].permute(0, 2, 1), dim=-1)
        return keypoints, torch.where(valid, scores, 0.0), descriptors, valid


def _with_pixel_unshuffle(network: Any) -> Any:  # noqa: ANN401  # see StaticXFeat.__init__
    patched = copy.deepcopy(network).eval()
    patched._unfold2d = _unfold_as_pixel_unshuffle  # noqa: SLF001  # upstream seam, no other hook
    return patched
