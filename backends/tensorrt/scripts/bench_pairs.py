"""Development benchmark, not packaged: inference-only throughput over a pair manifest.

Every unique image of the manifest is loaded from the input store into RAM before any timing,
and each timed pass follows an untimed warm-up of `WARMUP_SECONDS` on the first pairs.
Timed: upload, engine execution, download and the host-side unpacking of the outputs. Not timed:
image decoding and resizing, disk reads.

Variants, each on freshly built models since a captured CUDA graph cannot be released:

- per pair at batch 1 without CUDA graphs: both images of each pair extracted, then the pair
  matched, the shape of the real-time path;
- the same with CUDA graphs;
- for every XFeat batch size of the ladder: `predict_pairs_batched` with CUDA graphs, each unique
  image extracted once and LighterGlue at batch 1.
"""

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from input_store import load_or_prepare
from xfeat_trt.arguments import add_engine_arguments, build_options
from xfeat_trt.engine import BuildOptions
from xfeat_trt.ladder import BatchLadder, find_ladder
from xfeat_trt.models import LighterGlueModel, XFeatModel

from xfeat_bench.timing import RunConditions, current_conditions
from xfeat_pairs.manifest import PairRecord, load_manifest
from xfeat_pairs.preprocess import PreparedImage, ResizeMode
from xfeat_pairs.run import predict_pairs_batched

DEFAULT_REPEATS = 2
MILLISECONDS_PER_SECOND = 1000
LIGHTERGLUE_BATCH_SIZE = 1
PER_PAIR_VARIANT = "per pair"
UNIQUE_IMAGES_VARIANT = "unique images"
WARMUP_SECONDS = 2.0
"""GPU clocks idle low after an engine build: a cold first pass ran at 30 % of the steady rate in
calibration (LighterGlue K=256, RTX 4070 Ti)."""
WARMUP_PAIRS = 32

Images = dict[Path, PreparedImage]
Timing = tuple[float, float]
"""Extract seconds and match seconds."""
TimeVariant = Callable[[Sequence[PairRecord], XFeatModel, LighterGlueModel], Timing]


@dataclass(frozen=True)
class Variant:
    """One way of running the models over the manifest."""

    name: str
    xfeat_batch: int
    cuda_graph: bool
    time: TimeVariant


@dataclass(frozen=True)
class BenchRow:
    """One timed pass over the manifest.

    Attributes:
        variant: `per pair` (every image of every pair extracted) or `unique images` (each image
            extracted once, in batches).
        xfeat_batch: XFeat batch size of the models.
        cuda_graph: Whether both models ran through captured CUDA graphs.
        repeat: Index of the pass among the repeats of its variant.
        extract_seconds: Time in XFeat calls.
        match_seconds: Time in LighterGlue calls.
        total_seconds: `extract_seconds + match_seconds`.
        pairs_per_second: Manifest pairs divided by `total_seconds`.
    """

    variant: str
    xfeat_batch: int
    cuda_graph: bool
    repeat: int
    extract_seconds: float
    match_seconds: float
    total_seconds: float
    pairs_per_second: float


def load_images(
    records: Sequence[PairRecord], store: Path, size: tuple[int, int], mode: ResizeMode
) -> Images:
    """Model inputs of every unique image of `records`, read from `store` (prepared on a miss)."""
    paths = dict.fromkeys(path for record in records for path in (record.image0, record.image1))
    return {path: load_or_prepare(store, path, size, mode) for path in paths}


def _models(
    ladder: BatchLadder, variant: Variant, cache_dir: Path, options: BuildOptions
) -> tuple[XFeatModel, LighterGlueModel]:
    xfeat = XFeatModel(ladder.xfeat[variant.xfeat_batch], cache_dir, options)
    glue = LighterGlueModel(ladder.lighterglue[LIGHTERGLUE_BATCH_SIZE], cache_dir, options)
    if variant.cuda_graph:
        xfeat.runner.capture_graph()
        glue.runner.capture_graph()
    return xfeat, glue


def _time_per_pair(
    records: Sequence[PairRecord], images: Images, xfeat: XFeatModel, glue: LighterGlueModel
) -> Timing:
    extract_seconds = match_seconds = 0.0
    for record in records:
        start = time.perf_counter()
        first = xfeat.extract(images[record.image0].pixels)[0]
        second = xfeat.extract(images[record.image1].pixels)[0]
        extracted = time.perf_counter()
        glue.match(first, second)
        extract_seconds += extracted - start
        match_seconds += time.perf_counter() - extracted
    return extract_seconds, match_seconds


def _time_unique_images(
    records: Sequence[PairRecord],
    images: Images,
    xfeat: XFeatModel,
    glue: LighterGlueModel,
    xfeat_batch: int,
) -> Timing:
    predictions = [
        prediction
        for _, prediction in predict_pairs_batched(
            records,
            load_image=images.__getitem__,
            extract=xfeat.extract,
            extract_batch_size=xfeat_batch,
            match=glue.match_batch,
            match_batch_size=LIGHTERGLUE_BATCH_SIZE,
        )
    ]
    return (
        sum(p.extract_ms for p in predictions) / MILLISECONDS_PER_SECOND,
        sum(p.match_ms for p in predictions) / MILLISECONDS_PER_SECOND,
    )


def _row(variant: Variant, repeat: int, timing: Timing, pairs: int) -> BenchRow:
    extract_seconds, match_seconds = timing
    total_seconds = extract_seconds + match_seconds
    return BenchRow(
        variant.name,
        variant.xfeat_batch,
        variant.cuda_graph,
        repeat,
        extract_seconds,
        match_seconds,
        total_seconds,
        pairs / total_seconds,
    )


def format_row(row: BenchRow) -> str:
    """One console line of `row`."""
    graph = "graph" if row.cuda_graph else "no graph"
    return (
        f"{row.variant}, XFeat B{row.xfeat_batch}, {graph}, repeat {row.repeat}: "
        f"extract {row.extract_seconds:.2f} s, match {row.match_seconds:.2f} s, "
        f"total {row.total_seconds:.2f} s, {row.pairs_per_second:.1f} pairs/s"
    )


def run_variants(  # noqa: PLR0913  # one parameter per run setting
    records: Sequence[PairRecord],
    images: Images,
    ladder: BatchLadder,
    *,
    cache_dir: Path,
    options: BuildOptions,
    repeats: int,
    on_row: Callable[[BenchRow], None],
) -> list[BenchRow]:
    """Time every variant `repeats` times on fresh models; `on_row` sees each row when finished."""
    variants = [
        Variant(PER_PAIR_VARIANT, 1, graph, lambda r, x, g: _time_per_pair(r, images, x, g))
        for graph in (False, True)
    ]
    variants += [
        Variant(
            UNIQUE_IMAGES_VARIANT,
            batch,
            True,  # noqa: FBT003  # positional field of a dataclass
            lambda r, x, g, batch=batch: _time_unique_images(r, images, x, g, batch),
        )
        for batch in ladder.xfeat
    ]
    rows: list[BenchRow] = []
    for variant in variants:
        for repeat in range(repeats):
            xfeat, glue = _models(ladder, variant, cache_dir, options)
            _warm_up(variant.time, records[:WARMUP_PAIRS], xfeat, glue)
            timing = variant.time(records, xfeat, glue)
            rows.append(_row(variant, repeat, timing, len(records)))
            on_row(rows[-1])
    return rows


def _warm_up(
    time_variant: TimeVariant,
    records: Sequence[PairRecord],
    xfeat: XFeatModel,
    glue: LighterGlueModel,
) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_SECONDS:
        time_variant(records, xfeat, glue)


def write_rows(rows: list[BenchRow], conditions: RunConditions, directory: Path) -> Path:
    """Write `rows` and `conditions` to a timestamped JSON file in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    record = {"conditions": asdict(conditions), "rows": [asdict(row) for row in rows]}
    path.write_text(json.dumps(record, indent=1))
    return path


def main() -> None:
    """Run every variant and write the rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=Path, required=True, help="directory holding one batch ladder"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resize", type=ResizeMode, choices=list(ResizeMode), required=True)
    parser.add_argument(
        "--input-store", type=Path, required=True, help="directory of persisted model inputs"
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--out-dir", type=Path, required=True)
    add_engine_arguments(parser)
    args = parser.parse_args()

    ladder = find_ladder(args.models)
    records = load_manifest(args.manifest)
    size = (ladder.xfeat_metadata.width, ladder.xfeat_metadata.height)
    images = load_images(records, args.input_store, size, args.resize)
    print(f"{len(records)} pairs, {len(images)} unique images in RAM")  # noqa: T201  # CLI output
    rows = run_variants(
        records,
        images,
        ladder,
        cache_dir=args.cache_dir,
        options=build_options(args),
        repeats=args.repeats,
        on_row=lambda row: print(format_row(row), flush=True),  # noqa: T201
    )
    print(write_rows(rows, current_conditions(), args.out_dir))  # noqa: T201


if __name__ == "__main__":
    main()
