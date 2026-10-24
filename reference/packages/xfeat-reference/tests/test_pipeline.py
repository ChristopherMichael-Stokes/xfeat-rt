import pytest
import torch

from xfeat_reference.pipeline import GridNormalisation, ReferenceConfig, ReferencePipeline

HEIGHT, WIDTH = 128, 192


def _edge_keypoint_count(grid: GridNormalisation) -> int:
    # threshold 0 and an unbounded K keep every NMS candidate, so only scoring can drop one
    config = ReferenceConfig(
        grid=grid, pruning=False, top_k=HEIGHT * WIDTH, detection_threshold=0.0
    )
    pipeline = ReferencePipeline(config, torch.device("cpu"))
    image = torch.rand(1, 1, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(0))
    keypoints = pipeline.extract(image).keypoints
    on_edge = (keypoints[:, 0] == WIDTH - 1) | (keypoints[:, 1] == HEIGHT - 1)
    return int(on_edge.sum())


@pytest.mark.parametrize(
    ("grid", "keeps_edge"),
    [(GridNormalisation.UPSTREAM_EXACT, False), (GridNormalisation.PIXEL_CENTRE, True)],
)
def test_last_row_and_column_survive_only_with_pixel_centre_grid(
    grid: GridNormalisation, *, keeps_edge: bool
) -> None:
    assert (_edge_keypoint_count(grid) > 0) == keeps_edge
