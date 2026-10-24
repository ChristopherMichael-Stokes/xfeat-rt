"""`xfeat-megadepth1500-manifest`: pair manifest of MegaDepth-1500 with relative-pose ground truth.

Reads `pairs_calibrated.txt` of the GlueFactory data directory. Each line holds 32 columns: two
image names, then K0 and K1 (9 values each, row-major, original resolution, COLMAP convention) and
`T_0to1` as R (9 values, row-major) followed by t (3 values).
"""

import argparse
import json
from pathlib import Path

NAME_COLUMNS = 2
MATRIX_VALUES = 9
TRANSLATION_VALUES = 3
COLUMNS = NAME_COLUMNS + 3 * MATRIX_VALUES + TRANSLATION_VALUES
DEFAULT_DATA_DIR = Path.home() / ".cache" / "xfeat-rt" / "gluefactory-data"
MEGADEPTH_DIRECTORY = "megadepth1500"


def _matrix(values: list[float]) -> list[list[float]]:
    return [values[row * 3 : row * 3 + 3] for row in range(3)]


def manifest_lines(data_dir: Path, pair_limit: int) -> list[str]:
    """JSON Lines of the first `pair_limit` MegaDepth-1500 pairs."""
    root = data_dir / MEGADEPTH_DIRECTORY
    lines = []
    for number, text in enumerate((root / "pairs_calibrated.txt").read_text().splitlines()):
        if number == pair_limit:
            break
        columns = text.split()
        if len(columns) != COLUMNS:
            msg = (
                f"pairs_calibrated.txt line {number + 1} has {len(columns)} columns, not {COLUMNS}"
            )
            raise ValueError(msg)
        numbers = [float(value) for value in columns[NAME_COLUMNS:]]
        intrinsics0 = _matrix(numbers[0:9])
        intrinsics1 = _matrix(numbers[9:18])
        rotation = _matrix(numbers[18:27])
        translation = numbers[27:30]
        lines.append(
            json.dumps(
                {
                    "name": f"pair{number:04d}",
                    "image0": str(root / "images" / columns[0]),
                    "image1": str(root / "images" / columns[1]),
                    "ground_truth": {
                        "kind": "relative_pose",
                        "rotation": rotation,
                        "translation": translation,
                        "intrinsics0": intrinsics0,
                        "intrinsics1": intrinsics1,
                    },
                }
            )
        )
    return lines


def main() -> None:
    """Entry point of `xfeat-megadepth1500-manifest`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pair-limit", type=int, default=1500)
    args = parser.parse_args()
    args.out.write_text("\n".join(manifest_lines(args.data_dir, args.pair_limit)) + "\n")
    print(args.out)  # noqa: T201  # CLI output
