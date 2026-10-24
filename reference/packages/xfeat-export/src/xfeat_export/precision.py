"""Compute precision of the exported networks."""

from enum import StrEnum

import torch


class Precision(StrEnum):
    """Precision of the convolutional backbone (XFeat) and transformer (LighterGlue).

    Detection (softmax, NMS, TopK), descriptor sampling and normalisation, and LighterGlue's
    assignment (log-softmax, log-sigmoid) stay FP32 in both modes, and graph inputs and outputs are
    FP32 either way.
    """

    FP32 = "fp32"
    FP16 = "fp16"

    @property
    def dtype(self) -> torch.dtype:
        """Torch dtype of the networks' tensors."""
        return torch.float16 if self is Precision.FP16 else torch.float32
