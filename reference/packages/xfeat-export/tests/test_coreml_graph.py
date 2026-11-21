"""The Core ML rewrites reproduce the base graphs in PyTorch at FP32 on distinct inputs."""

import pytest
import torch
from kornia.feature.lightglue import rotate_half
from torch.nn import functional

from xfeat_export.coreml_graph import (
    CoreMLLighterGlue,
    CoreMLXFeat,
    rotate_half_matrix,
    unfold_as_strided_conv,
)
from xfeat_export.lighterglue_graph import StaticLighterGlue
from xfeat_export.precision import Precision
from xfeat_export.xfeat_graph import StaticXFeat
from xfeat_reference.pipeline import GridNormalisation, ReferenceConfig, ReferencePipeline

HEIGHT, WIDTH, TOP_K = 256, 256, 128
DETECTION_THRESHOLD = 0.05
MATCH_THRESHOLD = 0.1
SHIFT = (3, 5)
"""Pixel shift between the two images of the pair, so they differ yet share structure."""
COARSE_CELL = 16
"""Noise is drawn at 1/16 resolution and upsampled, which gives corner-like structure."""
MIN_KEYPOINTS = 10
"""Fewer keypoints would make the equality of keypoints trivial."""
MIN_MATCHES = 10
"""Fewer matches would make the equality of match indices trivial."""
FLOAT32_EPSILON = torch.finfo(torch.float32).eps
DESCRIPTOR_ATOL = 4 * FLOAT32_EPSILON
"""Descriptor entries are at most 1 in magnitude. The rewrites change only the order of the L2
reduction and the sqrt form, a few units in the last place per entry."""
CONFIDENCE_RTOL = 1e-5
"""Confidence is `exp` of a log-space sum of four terms of magnitude below about 10 against a
product of the four probabilities; each term carries a relative rounding error of epsilon, so the
ratio differs by at most about 4 * 10 * epsilon = 5e-6."""


@pytest.fixture(scope="module")
def pipeline() -> ReferencePipeline:
    config = ReferenceConfig(GridNormalisation.PIXEL_CENTRE, pruning=False, top_k=TOP_K)
    return ReferencePipeline(config, torch.device("cpu"))


@pytest.fixture(scope="module")
def images() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    coarse = torch.rand(1, 1, HEIGHT // COARSE_CELL, WIDTH // COARSE_CELL, generator=generator)
    first = functional.interpolate(coarse, size=(HEIGHT, WIDTH), mode="nearest")
    first = first + 0.2 * torch.rand(1, 1, HEIGHT, WIDTH, generator=generator)
    first = first.clamp(0.0, 1.0)
    return first, first.roll(shifts=SHIFT, dims=(2, 3))


def _xfeat(module_class: type[StaticXFeat], pipeline: ReferencePipeline) -> StaticXFeat:
    return module_class(
        pipeline.xfeat_network,
        height=HEIGHT,
        width=WIDTH,
        top_k=TOP_K,
        detection_threshold=DETECTION_THRESHOLD,
        chunking=None,
        precision=Precision.FP32,
    ).eval()


def test_unfold_as_strided_conv_is_pixel_unshuffle() -> None:
    x = torch.randn(2, 1, 32, 48)
    assert torch.equal(unfold_as_strided_conv(x, 8), functional.pixel_unshuffle(x, 8))


def test_rotate_half_matrix_is_rotate_half() -> None:
    x = torch.randn(2, 4, 10, 24)
    assert torch.equal(x @ rotate_half_matrix(24, torch.float32), rotate_half(x))


def test_coreml_xfeat_matches_base_on_each_image(
    pipeline: ReferencePipeline, images: tuple[torch.Tensor, torch.Tensor]
) -> None:
    base = _xfeat(StaticXFeat, pipeline)
    coreml = _xfeat(CoreMLXFeat, pipeline)
    for image in images:
        with torch.no_grad():
            expected, actual = base(image), coreml(image)
        assert expected[3].sum() > MIN_KEYPOINTS
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[3], expected[3])
        assert torch.equal(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=DESCRIPTOR_ATOL)


def test_coreml_lighterglue_matches_base_on_a_shifted_pair(
    pipeline: ReferencePipeline, images: tuple[torch.Tensor, torch.Tensor]
) -> None:
    extractor = _xfeat(StaticXFeat, pipeline)
    size = torch.tensor([[WIDTH, HEIGHT]], dtype=torch.float32)
    inputs: list[torch.Tensor] = []
    with torch.no_grad():
        for image in images:
            keypoints, _, descriptors, valid = extractor(image)
            inputs += [keypoints, descriptors, valid, size]
        base = StaticLighterGlue(
            pipeline.lighterglue_network, MATCH_THRESHOLD, Precision.FP32
        ).eval()
        coreml = CoreMLLighterGlue(
            pipeline.lighterglue_network, MATCH_THRESHOLD, Precision.FP32
        ).eval()
        expected, actual = base(*inputs), coreml(*inputs)
    assert (expected[0] >= 0).sum() > MIN_MATCHES
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=CONFIDENCE_RTOL, atol=0.0)
