# C2 results: export, `xfeat-core`, `backends/ort`, contract test, harness

Checkpoint C2 of `implementation-plan.md`. The cross-workspace contract test passes and a first CPU
latency table exists. `make check` and `make test` pass.

## Built

| Path | Holds |
| ---- | ----- |
| `reference/packages/xfeat-export` | Static XFeat graph (in-graph NMS, `TopK`, bicubic descriptor sampling), padded masked LighterGlue graph, metadata writer, `xfeat-export` CLI |
| `runtime/packages/xfeat-core` | `parse_metadata` (pydantic, frozen, `extra="forbid"`), `Features`, valid-mask compaction, padding, match pairs, keypoint rescale |
| `runtime/packages/xfeat-bench` | `measure` (warm-up, `perf_counter_ns`, raw samples, p50/p95/p99, load average) and timestamped JSON writer |
| `backends/ort` | `XFeatModel`, `LighterGlueModel`, `xfeat-ort-predict`, `xfeat-ort-bench`; own lock, Python 3.13 pin |

- `ReferencePipeline` gained `xfeat_network` and `lighterglue_network`; the export no longer reads
  private attributes.
- The empty `xfeat-rt-pipeline` stub is deleted.
- Make targets: `test`, `export TOP_K=<k>`; `setup` also syncs `backends/ort`.
- No `uv.lock` mentions torch except `reference/`.

## Export graph choices (from C1)

- Reliability through pad + `Resize` x8 + crop; `pixel_unshuffle` as `SpaceToDepth` through a
  translation table. The `gridsample` and upstream-layout variants are not carried over.
- Chunked `TopK` is the `--chunked-topk` flag, off by default.
- Opset 20 only. Opset 23 `Attention` segfaults TensorRT 11.4 at build (C1 TensorRT report).
- File names: `xfeat_<W>x<H>_k<K>_b<B>.onnx`, `lighterglue_k<K>_b<B>.onnx`.
- Metadata keys: `model`, `height`, `width`, `batch`, `top_k`, `detection_threshold`, `grid`,
  `topk_mode`, `chunk_size`, `max_k_per_op`, `match_threshold`, `pruning`, `opset`,
  `weights_sha256`, `git_hash`.

## Contract test

`reference/packages/xfeat-export/tests/test_contract.py`: exports in the test process, runs
`xfeat-ort-predict` in the ORT venv, compares with `ReferencePipeline` (FP32, CPU, `pixel_centre`,
pruning off, deterministic algorithms) on the first 3 MegaDepth-1500 pairs resized to 1280x960.

| Case | Keypoints | Scores, descriptors | Matches |
| ---- | --------- | ------------------- | ------- |
| K = 2048, plain `TopK` | sets equal up to 1 tie swap per image | max score diff 1e-4, cosine 0.99999 | Jaccard 1.0 |
| K = 4096, chunked `TopK` | same | same | Jaccard at least 0.99 |

- Bounds are the C1 spike's measured extremes; each constant carries its source in the test.
- Checked live: raising the Jaccard bound to 1.01 fails the test.
- The ORT CLI parses the metadata through `xfeat-core`, so a key mismatch between writer and parser
  fails the test.

## CPU latency (ORT 1.31.0, Ryzen 7 5800X3D, WSL2, 1280x960, batch 1)

p50 ms of `session.run`, 30 runs after 5 warm-up, real MegaDepth image for XFeat, random padded
features for LighterGlue. Load average 2 to 9 on 16 threads during the sweep, partly the sweep itself;
the box was not idle, so treat as indicative. Raw samples are in `backends/ort/uncommitted/bench/`.

| K | XFeat, default threads | XFeat, 8 threads | LighterGlue, default threads | LighterGlue, 8 threads |
| --- | --- | --- | --- | --- |
| 1024 | 46.4 | 46.9 | 41.0 | 46.5 |
| 2048 | 46.9 | 49.7 | 124.8 | 150.5 |
| 4096 | 47.5 | 49.9 | 656.3 | 722.5 |

- XFeat is flat in K (47 to 50 ms): `TopK` and sampling are cheap next to the backbone.
- LighterGlue grows 3x then 5.3x per K doubling and is 14x XFeat at K = 4096. This is the
  optimisation target.
- 8 threads is not faster than ORT's default on this machine.
- Matches the C1 spike (XFeat 58 to 67 ms, LighterGlue 170 and 740 ms under heavier load).

## Findings that change the plan

- `xfeat-core` now depends on pydantic as well as numpy (typed, validated metadata parse). Pure
  wheels exist for aarch64; not yet confirmed on the Pi.
- The prediction-file writer is not built. Its format belongs to `xfeat-evaluation`, which scores
  predictions; define both together when the custom pair loader is built in C3.
- `prek` skips untracked files and ignores an empty (intent-to-add) `prek.toml`; new projects need
  `git add` of their `prek.toml` and `prek run --refresh` once.
- The `typos` hook rewrites files: it rewrote a two-letter einsum subscript in a test into a word and
  broke the test. Check diffs after the first `make check` on new code.
- `ty` has no fix for `torch.nn.Module.forward` (invalid-mutable-override, one `ty: ignore` per
  override) and for `onnxruntime.SessionOptions` (possibly-missing-import, one `ty: ignore`).

## Not done

- Batch > 1 graphs (C4), FP16 (C3, TensorRT converter issues in `c1-tensorrt-spike.md`),
  idle-machine rerun of the latency table.
- `onnxruntime_perf_test` cross-check of the headline numbers.
