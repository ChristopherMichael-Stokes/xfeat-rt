import numpy as np
import pytest
from pydantic import ValidationError

from xfeat_core.features import (
    Features,
    compact_features,
    match_pairs,
    pad_features,
    rescale_keypoints,
)
from xfeat_core.metadata import LighterGlueMetadata, XFeatMetadata, parse_metadata

XFEAT_RAW = {
    "model": "xfeat",
    "height": "960",
    "width": "1280",
    "batch": "1",
    "top_k": "4096",
    "detection_threshold": "0.05",
    "grid": "pixel_centre",
    "topk_mode": "chunked",
    "chunk_size": "65536",
    "max_k_per_op": "2048",
    "opset": "20",
    "weights_sha256": "ab",
    "git_hash": "cd-dirty",
}
LIGHTERGLUE_RAW = {
    "model": "lighterglue",
    "batch": "1",
    "top_k": "2048",
    "match_threshold": "0.1",
    "pruning": "off",
    "opset": "20",
    "weights_sha256": "ab",
    "git_hash": "cd",
}


def _features(count: int) -> Features:
    keypoints = np.arange(2 * count, dtype=np.float32).reshape(count, 2)
    return Features(
        keypoints,
        np.linspace(1, 0.5, count, dtype=np.float32),
        np.ones((count, 64), dtype=np.float32),
        (1280, 960),
    )


def test_xfeat_metadata_parses_strings_to_typed_values() -> None:
    metadata = parse_metadata(XFEAT_RAW)
    assert isinstance(metadata, XFeatMetadata)
    assert (metadata.top_k, metadata.detection_threshold, metadata.chunk_size) == (
        4096,
        0.05,
        65536,
    )


def test_lighterglue_metadata_selected_by_model_key() -> None:
    assert isinstance(parse_metadata(LIGHTERGLUE_RAW), LighterGlueMetadata)


def test_plain_topk_metadata_has_no_chunk_settings() -> None:
    raw = {k: v for k, v in XFEAT_RAW.items() if k not in {"chunk_size", "max_k_per_op"}}
    raw["topk_mode"] = "plain"
    assert parse_metadata(raw).model_dump()["chunk_size"] is None


@pytest.mark.parametrize("extra", [{"unknown_key": "1"}, {"top_k": "many"}])
def test_malformed_metadata_is_rejected(extra: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        parse_metadata({**XFEAT_RAW, **extra})


def test_compact_features_keeps_only_valid_slots_in_order() -> None:
    features = _features(4)
    valid = np.array([True, True, False, True])
    compact = compact_features(
        features.keypoints, features.scores, features.descriptors, valid, (1280, 960)
    )
    np.testing.assert_array_equal(compact.keypoints, features.keypoints[[0, 1, 3]])
    np.testing.assert_array_equal(compact.scores, features.scores[[0, 1, 3]])


def test_pad_features_puts_valid_rows_first_and_zero_fills_the_rest() -> None:
    padded = pad_features(_features(3), 5)
    np.testing.assert_array_equal(padded.valid, [True, True, True, False, False])
    assert not padded.keypoints[3:].any()
    assert not padded.descriptors[3:].any()
    np.testing.assert_array_equal(padded.image_size, [1280, 960])


def test_pad_features_rejects_more_features_than_k() -> None:
    with pytest.raises(ValueError, match="exceed"):
        pad_features(_features(6), 5)


def test_match_pairs_lists_matched_indices_only() -> None:
    matches0 = np.array([2, -1, 0, -1], dtype=np.int64)
    np.testing.assert_array_equal(match_pairs(matches0), [[0, 2], [2, 0]])


def test_match_pairs_of_no_matches_has_shape_zero_by_two() -> None:
    assert match_pairs(np.full(4, -1, dtype=np.int64)).shape == (0, 2)


def test_rescale_maps_pixel_centres_not_pixel_indices() -> None:
    # pixel 0 of a 4 px row spans [0, 4); its centre 0.5 doubles to 1.0 in an 8 px row, index 0.5
    rescaled = rescale_keypoints(
        np.array([[0.0, 0.0], [3.0, 3.0]], dtype=np.float32), (4, 4), (8, 8)
    )
    np.testing.assert_allclose(rescaled, [[0.5, 0.5], [6.5, 6.5]])
