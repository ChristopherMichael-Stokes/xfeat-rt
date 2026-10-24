"""Pinned upstream XFeat source and weights, fetched through the repository's own `hubconf.py`.

verlab/accelerated_features has no `pyproject.toml`, so it cannot be a package dependency.
`torch.hub` downloads the repository at a fixed commit into the hub cache and imports it from
there; nothing upstream is copied into this repository.
"""

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

UPSTREAM_REPOSITORY = "verlab/accelerated_features"
UPSTREAM_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
XFEAT_WEIGHTS_SHA256 = "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
LIGHTERGLUE_WEIGHTS_SHA256 = "766102df37f11189efe5b0811d1f47c72b22629b79bfabfcfff9d2a2f84654b8"


class WeightsMismatchError(RuntimeError):
    """A weights file does not have the pinned SHA-256."""


@dataclass(frozen=True)
class UpstreamCheckout:
    """The pinned upstream checkout in the torch.hub cache.

    Attributes:
        root: Repository root inside the hub cache.
        xfeat_class: `modules.xfeat.XFeat`.
        lighterglue_module: `modules.lighterglue`.
        interpolator_module: `modules.interpolator`.
    """

    root: Path
    xfeat_class: type[Any]
    lighterglue_module: ModuleType
    interpolator_module: ModuleType

    @property
    def xfeat_weights(self) -> Path:
        """`weights/xfeat.pt`, verified against its pinned SHA-256."""
        return _verified(self.root / "weights" / "xfeat.pt", XFEAT_WEIGHTS_SHA256)

    @property
    def lighterglue_weights(self) -> Path:
        """`weights/xfeat-lighterglue.pt`, verified against its pinned SHA-256."""
        return _verified(self.root / "weights" / "xfeat-lighterglue.pt", LIGHTERGLUE_WEIGHTS_SHA256)


def load_upstream() -> UpstreamCheckout:
    """Fetch the pinned upstream commit (cached after the first call) and import its modules."""
    # pretrained=False: hubconf's pretrained path downloads weights from the unpinned main branch
    untrained = torch.hub.load(
        f"{UPSTREAM_REPOSITORY}:{UPSTREAM_COMMIT}",
        "XFeat",
        pretrained=False,
        trust_repo=True,
        # validation queries the GitHub API, which rate-limits anonymous callers
        skip_validation=True,
        verbose=False,
    )
    xfeat_class = type(untrained)
    # hub removes the checkout from sys.path after loading, but `modules` stays in sys.modules
    return UpstreamCheckout(
        root=Path(inspect.getfile(xfeat_class)).parents[1],
        xfeat_class=xfeat_class,
        lighterglue_module=importlib.import_module("modules.lighterglue"),
        interpolator_module=importlib.import_module("modules.interpolator"),
    )


def _verified(path: Path, expected_sha256: str) -> Path:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        msg = f"{path}: SHA-256 {actual}, expected {expected_sha256}"
        raise WeightsMismatchError(msg)
    return path
