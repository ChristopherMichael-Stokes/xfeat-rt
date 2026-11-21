"""XFeat and LighterGlue rewritten for the ONNX Runtime Core ML execution provider.

The provider hands Core ML only the ops it supports, and every unsupported node splits the graph
and costs a CPU round trip. Each rewrite below is exact in exact arithmetic and replaces ops the
provider rejects (ONNX Runtime 1.31). `TopK`, the bicubic `GridSample` of the descriptors and the
mutual-match check stay on the CPU provider, under 2 ms together on an Apple M2.
"""

from collections.abc import Callable
from typing import Any, override

import torch
from kornia.feature.lightglue import normalize_keypoints
from torch import nn
from torch.nn import functional

from xfeat_export.lighterglue_graph import MASK_LOGIT, StaticLighterGlue, as_tensor
from xfeat_export.precision import Precision
from xfeat_export.xfeat_graph import StaticXFeat

L2_EPSILON = 1e-12
"""`torch.nn.functional.normalize` default `eps`."""
PIXEL_UNSHUFFLE_CELL = 8
"""Upstream keypoint head unshuffles the image into 8x8 cells."""
ROTARY_PAIR = 2
"""Rotary encoding rotates adjacent feature pairs."""


def unfold_as_strided_conv(x: torch.Tensor, ws: int = 2) -> torch.Tensor:
    """Pixel unshuffle of a one-channel image as a stride-`ws` convolution with a one-hot kernel.

    Output channel `i * ws + j` selects the pixel at offset (i, j) of each `ws` x `ws` cell, which
    is `pixel_unshuffle`. Avoids `SpaceToDepth`, which the provider rejects.
    """
    kernel = torch.eye(ws * ws, dtype=x.dtype).reshape(ws * ws, 1, ws, ws)
    return functional.conv2d(x, kernel, stride=ws)


def rotate_half_matrix(dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Signed permutation `R` with `x @ R == rotate_half(x)`.

    Kornia's `rotate_half` maps `out[2i] = -x[2i+1]` and `out[2i+1] = x[2i]`, so `R[2i+1, 2i] = -1`
    and `R[2i, 2i+1] = 1`. Avoids the `Neg` of the rotation, which the provider rejects.
    """
    matrix = torch.zeros(dim, dim, dtype=dtype)
    even = torch.arange(0, dim, ROTARY_PAIR)
    matrix[even + 1, even] = -1
    matrix[even, even + 1] = 1
    return matrix


class CoreMLXFeat(StaticXFeat):
    """`StaticXFeat` with the pixel unshuffle and the L2 normalisation rewritten.

    - Pixel unshuffle (`ws=8`) is a stride-8 convolution (`unfold_as_strided_conv`) instead of
      `SpaceToDepth`.
    - `x / max(sqrt(sum(x^2)), eps)` instead of `ReduceL2` and its `Expand`; equal to
      `functional.normalize` for the same `eps`.
    """

    @override
    def _unfold2d_replacement(self) -> Callable[[torch.Tensor, int], torch.Tensor]:
        return unfold_as_strided_conv

    @override
    def _normalize(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return x / torch.sqrt((x * x).sum(dim, keepdim=True)).clamp_min(L2_EPSILON)


class CoreMLLighterGlue(StaticLighterGlue):
    """`StaticLighterGlue` with the attention inputs and the assignment rewritten.

    - `Wqkv` is three linears over the weight rows `[i::3]`, which is the layout of upstream's
      `unflatten(-1, (heads, -1, 3))`. Avoids the rank-5 `Gather` of the split.
    - `rotate_half` is a matmul with `rotate_half_matrix`. Avoids `Neg`.
    - The positional encoding keeps cos and sin as 4-D tensors, concatenated on the last axis, and
      builds `repeat_interleave(2)` as stack plus reshape. Avoids rank-6 `Unsqueeze` and `Expand`.
    - The assignment runs in probability space, see `_assign`. Avoids `LogSoftmax` and `Log`.
    """

    rotate_half: torch.Tensor

    def __init__(
        self,
        lightglue: Any,  # noqa: ANN401  # kornia LightGlue
        match_threshold: float,
        precision: Precision,
    ) -> None:
        """Wrap a loaded kornia `LightGlue` network; see `StaticLighterGlue`."""
        super().__init__(lightglue, match_threshold, precision)
        for layer in self.lightglue.transformers:
            block = layer.self_attn
            block.split_projections = nn.ModuleList(
                _projection_from_rows(block.Wqkv, offset) for offset in range(3)
            )
            del block.Wqkv
        head_dim = self.lightglue.transformers[0].self_attn.head_dim
        self.register_buffer("rotate_half", rotate_half_matrix(head_dim, precision.dtype))

    @override
    def _encode(self, keypoints: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
        projected = self.lightglue.posenc.Wr(normalize_keypoints(keypoints, image_size))
        batch, count, half = projected.shape
        cosine, sine = (
            torch.stack([trigonometric(projected)] * 2, -1).reshape(batch, 1, count, 2 * half)
            for trigonometric in (torch.cos, torch.sin)
        )
        return torch.cat([cosine, sine], -1).to(self.compute_dtype)

    @override
    def _query_key_value(
        self,
        block: Any,  # kornia SelfBlock
        x: torch.Tensor,
        encoding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cosine, sine = encoding.chunk(2, -1)
        query, key, value = (
            as_tensor(projection(x).unflatten(-1, (block.num_heads, -1))).transpose(1, 2)
            for projection in block.split_projections
        )
        return (
            as_tensor(query * cosine + torch.matmul(query, self.rotate_half) * sine),
            as_tensor(key * cosine + torch.matmul(key, self.rotate_half) * sine),
            value,
        )

    @override
    def _assign(
        self,
        assignment: Any,  # kornia MatchAssignment
        x0: torch.Tensor,
        x1: torch.Tensor,
        valid0: torch.Tensor,
        valid1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Argmax and confidence of the match probability.

        With S the scaled similarity, m0 and m1 the matchability logits and the masks as additive
        biases, upstream scores `log_softmax_row(S) + log_softmax_col(S) + logsigmoid(m0) +
        logsigmoid(m1)^T` and reports `exp(max score)`. Here P = softmax_row(S) * softmax_col(S) *
        sigmoid(m0) * sigmoid(m1)^T is exp of that sum, so the argmax is unchanged and the
        confidence is `max P`. Avoids `LogSoftmax` and `Log`.
        """
        projected0, projected1 = assignment.final_proj(x0), assignment.final_proj(x1)
        dim = projected0.shape[-1]
        similarity = torch.matmul(projected0 / dim**0.25, (projected1 / dim**0.25).transpose(1, 2))
        column_bias = torch.where(valid1, 0.0, MASK_LOGIT)[:, None, :]
        row_bias = torch.where(valid0, 0.0, MASK_LOGIT)[:, :, None]
        probability = (
            functional.softmax(similarity + column_bias, 2)
            * functional.softmax(similarity + row_bias, 1)
            * torch.sigmoid(assignment.matchability(x0))
            * torch.sigmoid(assignment.matchability(x1)).transpose(1, 2)
        )
        probability = torch.where(valid0[:, :, None] & valid1[:, None, :], probability, 0.0)
        best0, best_probability0, mutual = self._best_mutual(probability, valid0)
        return self._thresholded(best0, mutual, best_probability0)


def _projection_from_rows(combined: nn.Linear, offset: int) -> nn.Linear:
    """Linear holding rows `offset::3` of `combined`'s weight and bias."""
    projection = nn.Linear(combined.in_features, combined.out_features // 3).to(
        combined.weight.dtype
    )
    projection.weight.data = combined.weight.data[offset::3].clone()
    projection.bias.data = combined.bias.data[offset::3].clone()
    return projection
