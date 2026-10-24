"""LighterGlue as a static graph over inputs padded to K keypoints, with validity masks.

Mirrors the CPU path of kornia 0.8.3 `kornia/feature/lightglue.py` (SelfBlock, CrossBlock einsum
branch, MatchAssignment, `filter_matches`) with pruning and early exit off. Attention is explicit
MatMul/Softmax with an additive key mask (scaled dot-product attention, Vaswani et al. 2017).
The dustbin row and column are not built because `filter_matches` never reads them.
"""

from typing import Any, override

import torch
from kornia.feature.lightglue import apply_cached_rotary_emb, normalize_keypoints
from torch import nn
from torch.nn import functional

MASK_LOGIT = -30000.0
"""Additive mask for padded keypoints. Finite in FP16 (max 65504), so a row with every key masked
stays finite, and `exp(-30000)` underflows to exactly 0 in FP32 and FP16."""


def masked_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_bias: torch.Tensor
) -> torch.Tensor:
    """`softmax(q k^T / sqrt(d) + bias) v` with `bias` broadcast over the key axis."""
    similarity = torch.matmul(query, key.transpose(-1, -2)) * query.shape[-1] ** -0.5 + key_bias
    return torch.matmul(similarity.softmax(-1), value)


def _tensor(value: object) -> torch.Tensor:
    """Narrow the untyped output of an upstream module to `Tensor`."""
    if not isinstance(value, torch.Tensor):
        msg = f"expected a Tensor, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    return _tensor(x.unflatten(-1, (heads, -1))).transpose(1, 2)


class StaticLighterGlue(nn.Module):
    """Match two padded keypoint sets.

    Inputs per image: `kpts` (B, K, 2), `desc` (B, K, 64), `valid` (B, K) bool and `image_size`
    (B, 2) float (w, h). Outputs `m0` (B, K) int64, the matched index into image 1 or -1, and
    `mscores0` (B, K), the match confidence (0 where unmatched or invalid).
    """

    def __init__(self, lightglue: Any, match_threshold: float) -> None:  # noqa: ANN401  # kornia LightGlue
        """Wrap a loaded kornia `LightGlue` network."""
        super().__init__()
        self.lightglue = lightglue.eval()
        self.match_threshold = match_threshold

    def _self_block(
        self,
        block: Any,  # noqa: ANN401  # kornia SelfBlock
        x: torch.Tensor,
        encoding: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        qkv = block.Wqkv(x).unflatten(-1, (block.num_heads, -1, 3)).transpose(1, 2)
        query, key, value = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        query = apply_cached_rotary_emb(encoding, query)
        key = apply_cached_rotary_emb(encoding, key)
        context = masked_attention(query, key, value, bias)
        message = block.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + _tensor(block.ffn(torch.cat([x, message], -1)))

    def _cross_block(
        self,
        block: Any,  # noqa: ANN401  # kornia CrossBlock
        x0: torch.Tensor,
        x1: torch.Tensor,
        bias0: torch.Tensor,
        bias1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = block.scale**0.5
        qk0 = _split_heads(block.to_qk(x0), block.heads) * scale
        qk1 = _split_heads(block.to_qk(x1), block.heads) * scale
        value0 = _split_heads(block.to_v(x0), block.heads)
        value1 = _split_heads(block.to_v(x1), block.heads)
        similarity = torch.matmul(qk0, qk1.transpose(-1, -2))
        message0 = torch.matmul((similarity + bias1).softmax(-1), value1)
        message1 = torch.matmul((similarity.transpose(-1, -2) + bias0).softmax(-1), value0)
        message0, message1 = (
            block.to_out(m.transpose(1, 2).flatten(start_dim=-2)) for m in (message0, message1)
        )
        return (
            x0 + _tensor(block.ffn(torch.cat([x0, message0], -1))),
            x1 + _tensor(block.ffn(torch.cat([x1, message1], -1))),
        )

    @override
    def forward(  # ty: ignore[invalid-mutable-override]  # torch types forward as a writable attribute
        self,
        kpts0: torch.Tensor,
        desc0: torch.Tensor,
        valid0: torch.Tensor,
        size0: torch.Tensor,
        kpts1: torch.Tensor,
        desc1: torch.Tensor,
        valid1: torch.Tensor,
        size1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the matcher."""
        net = self.lightglue
        encoding0 = net.posenc(normalize_keypoints(kpts0, size0))
        encoding1 = net.posenc(normalize_keypoints(kpts1, size1))
        x0, x1 = net.input_proj(desc0), net.input_proj(desc1)
        bias0 = torch.where(valid0, 0.0, MASK_LOGIT)[:, None, None, :]
        bias1 = torch.where(valid1, 0.0, MASK_LOGIT)[:, None, None, :]
        for layer in net.transformers:
            x0 = self._self_block(layer.self_attn, x0, encoding0, bias0)
            x1 = self._self_block(layer.self_attn, x1, encoding1, bias1)
            x0, x1 = self._cross_block(layer.cross_attn, x0, x1, bias0, bias1)
        return self._assign(net.log_assignment[-1], x0, x1, valid0, valid1)

    def _assign(
        self,
        assignment: Any,  # noqa: ANN401  # kornia MatchAssignment
        x0: torch.Tensor,
        x1: torch.Tensor,
        valid0: torch.Tensor,
        valid1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected0, projected1 = assignment.final_proj(x0), assignment.final_proj(x1)
        dim = projected0.shape[-1]
        similarity = torch.matmul(projected0 / dim**0.25, (projected1 / dim**0.25).transpose(1, 2))
        matchability0, matchability1 = assignment.matchability(x0), assignment.matchability(x1)
        column_bias = torch.where(valid1, 0.0, MASK_LOGIT)[:, None, :]
        row_bias = torch.where(valid0, 0.0, MASK_LOGIT)[:, :, None]
        scores = (
            functional.log_softmax(similarity + column_bias, 2)
            + functional.log_softmax(similarity + row_bias, 1)
            + functional.logsigmoid(matchability0)
            + functional.logsigmoid(matchability1).transpose(1, 2)
        )
        scores = torch.where(valid0[:, :, None] & valid1[:, None, :], scores, MASK_LOGIT)
        best_score0, best0 = scores.max(2)
        best1 = scores.max(1).indices
        mutual = torch.arange(best0.shape[1])[None] == best1.gather(1, best0)
        mutual &= valid0
        confidence = torch.where(mutual, best_score0.exp(), 0.0)
        matched = mutual & (confidence > self.match_threshold)
        return torch.where(matched, best0, -1), confidence
