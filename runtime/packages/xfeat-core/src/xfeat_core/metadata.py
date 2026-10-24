"""Typed view of the `metadata_props` that `xfeat-export` writes into each ONNX file.

A backend reads the raw string map from its runtime (ORT `custom_metadata_map`, or the ONNX file
TensorRT builds from) and passes it to `parse_metadata`.
"""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class XFeatMetadata(BaseModel):
    """Settings baked into an exported XFeat graph.

    Attributes:
        height: Input image height in pixels.
        width: Input image width in pixels.
        batch: Images per call.
        top_k: Keypoints per image; slots beyond the detected count are invalid.
        detection_threshold: Minimum heatmap value of an NMS candidate.
        grid: Keypoint sampling convention (`GridNormalisation` of the reference).
        topk_mode: `chunked` when no `TopK` node exceeds `max_k_per_op`.
        chunk_size: Score-map elements per chunk, chunked mode only.
        max_k_per_op: Largest K of one `TopK` node, chunked mode only.
        opset: ONNX opset of the file.
        weights_sha256: SHA-256 of the pinned upstream weights.
        git_hash: Commit of this repository at export, `-dirty` with uncommitted changes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: Literal["xfeat"]
    height: int
    width: int
    batch: int
    top_k: int
    detection_threshold: float
    grid: Literal["pixel_centre", "upstream_exact"]
    topk_mode: Literal["plain", "chunked"]
    chunk_size: int | None = None
    max_k_per_op: int | None = None
    opset: int
    weights_sha256: str
    git_hash: str


class LighterGlueMetadata(BaseModel):
    """Settings baked into an exported LighterGlue graph.

    Attributes:
        batch: Pairs per call.
        top_k: Keypoints per image the inputs are padded to.
        match_threshold: Minimum match confidence.
        pruning: LighterGlue point pruning; the exported graph has none.
        opset: ONNX opset of the file.
        weights_sha256: SHA-256 of the pinned upstream weights.
        git_hash: Commit of this repository at export, `-dirty` with uncommitted changes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: Literal["lighterglue"]
    batch: int
    top_k: int
    match_threshold: float
    pruning: Literal["off"]
    opset: int
    weights_sha256: str
    git_hash: str


Metadata = Annotated[XFeatMetadata | LighterGlueMetadata, Field(discriminator="model")]
_METADATA_ADAPTER: TypeAdapter[Metadata] = TypeAdapter(Metadata)


def parse_metadata(raw: Mapping[str, str]) -> XFeatMetadata | LighterGlueMetadata:
    """Parse the string map of an exported model; raises `pydantic.ValidationError` if malformed."""
    return _METADATA_ADAPTER.validate_python(dict(raw))
