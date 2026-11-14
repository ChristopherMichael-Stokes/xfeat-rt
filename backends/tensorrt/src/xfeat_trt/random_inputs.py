"""Random model inputs for timing; every graph is static, so compute does not depend on them."""

import numpy as np
import numpy.typing as npt

from xfeat_core.features import DESCRIPTOR_DIMENSION, Features
from xfeat_core.metadata import XFeatMetadata


def random_images(metadata: XFeatMetadata, rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """(B, 1, H, W) uniform noise in [0, 1) for the XFeat graph of `metadata`."""
    shape = (metadata.batch, 1, metadata.height, metadata.width)
    return rng.random(shape, dtype=np.float32)


def random_features(count: int, size: tuple[int, int], rng: np.random.Generator) -> Features:
    """`count` features with unit descriptors at uniform positions in an image of `size`."""
    descriptors = rng.standard_normal((count, DESCRIPTOR_DIMENSION)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    keypoints = (rng.random((count, 2)) * (np.asarray(size) - 1)).astype(np.float32)
    return Features(keypoints, np.ones(count, dtype=np.float32), descriptors, size)
