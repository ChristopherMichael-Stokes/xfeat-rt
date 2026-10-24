"""ONNX `metadata_props` written on every exported model.

All values are strings. `xfeat_core.metadata` parses the same keys; the cross-workspace contract
test fails if the two disagree.
"""

import subprocess
from pathlib import Path

import onnx

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def git_description() -> str:
    """Full commit hash of this repository, suffixed `-dirty` with uncommitted changes."""
    return subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],  # noqa: S607  # git on PATH
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def attach_metadata(model: onnx.ModelProto, entries: dict[str, str]) -> None:
    """Add `entries` to `model.metadata_props`."""
    for key, value in entries.items():
        model.metadata_props.add(key=key, value=value)
