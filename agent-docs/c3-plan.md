# C3 plan: `backends/tensorrt`, pair loader, prediction format, eval on own pairs

Status: awaiting approval. Evidence: `c1-tensorrt-spike.md`, `c2-results.md`, a TensorRT 11.4 /
`cuda.core` API check (below) and a survey of the current code.

## Verified facts the design rests on

- TensorRT 11.4.0.106, `cuda-python` 13.4.4 (`cuda.core` 1.2.2), RTX 4070 Ti sm_89, Python 3.13/3.14.
- FP16/INT8 are not builder flags; precision comes from ONNX tensor types (`STRONGLY_TYPED`).
- `trt.Runtime(...).deserialize_cuda_engine` returns `None` and logs on a bad plan; it does not
  raise. A real TensorRT-version or GPU mismatch could not be reproduced on one machine.
- int64 and bool I/O are native. `execute_async_v3` captures into a CUDA graph (after a warm-up
  execute); ~2x faster on a tiny model, untested on the real engines.
- `cuda.core` covers device buffers, pinned host memory (NumPy view through DLPack), async copies,
  streams, timing events (`e1 - e0` in ms) and graph capture. Used instead of raw `cuda.bindings`.
- Timing cache (`create_timing_cache` / `set_timing_cache` / `serialize`) round-trips; its header
  rejects a different device unless `ignore_mismatch`.
- `tensorrt-cu13-libs` is an unhashed sdist stub in the default lock. Pinning that package to
  `https://pypi.nvidia.com/` (`explicit = true` index) locks the real wheels with sha256 hashes.
  Lock-level only; the install from those wheels is not yet run.
- Existing prediction format: ORT `pair{i}.npz` only (keypoints in model-input pixels, no names,
  scales or timing). `xfeat-evaluation` scores a GlueFactory `predictions.h5`
  (`keypoints{0,1}`, `keypoint_scores{0,1}`, `matches{0,1}`, `matching_scores{0,1}`, keypoints in
  the original-image frame as `(kp + 0.5) / scale`) through `run_eval`; only the in-process
  `ReferencePipeline` produces it today. `run_eval` needs ground truth (pose); pairs without it
  cannot go through it.

## Proposed design (forks for approval)

### D1. Prediction format (backend-neutral, torch-free)

A run directory:

| File | Holds |
| ---- | ----- |
| `index.json` | Pydantic-validated: format version, backend name and version, model metadata (both ONNX `metadata_props` maps), preprocessing, per-pair record (name, image paths, original sizes, model-input size, scale), timing summary, provenance |
| `pairs/<name>.npz` | `keypoints0/1` (N,2) f32 in **original-image pixel-centre coordinates** (0.5 offset convention stated in `index.json`), `scores0/1`, `descriptors0/1` (kept for agreement checks), `matches` (M,2) i64 |

- Writer and reader live in a new runtime package `xfeat-pairs` (see D2), not `xfeat-core`; both
  use numpy + pydantic + stdlib only. `xfeat-core` keeps features and metadata.
- `xfeat-evaluation` reads the directory, converts to the GlueFactory h5 layout (scale, per-view
  partner arrays) and calls `run_eval`. The ORT `pair{i}.npz` path is replaced by this format
  (`xfeat-ort-predict` and the contract test move over).

### D2. Pair manifest and loader

- Manifest: JSON Lines, one pair per line: `name`, `image0`, `image1` (paths relative to the
  manifest) and optional `ground_truth`: `{"relative_pose": {R, t, K0, K1}}` or
  `{"homography": [[...]]}`. Pairs with no ground truth still run for latency and for
  reference-vs-backend agreement.
- Preprocessing is declarative in the manifest header (or CLI flags): grayscale conversion
  (channel mean, as the reference), `stretch` or `letterbox` resize to the exported H x W, bilinear
  through OpenCV. Letterbox keeps aspect (pad value 0) and rescales keypoints back, which
  MegaDepth-1500 needs because its aspect ratios vary and the exported shapes are static.
  No user-supplied Python hook in v1.
- New package `runtime/packages/xfeat-pairs`: manifest model, image loading and preprocessing,
  keypoint rescale, prediction writer and reader. New dependency: `opencv-python-headless`
  (bit-matching the reference's resize; aarch64 wheels exist; already in `reference/`).

### D3. Scoring custom pairs (`xfeat-evaluation`)

- With relative-pose ground truth: same GlueFactory path as MegaDepth-1500 (pose AUC at 5/10/20,
  RANSAC via poselib).
- With homography ground truth: corner-error accuracy (MHA / AUC) as in HPatches, built on
  GlueFactory's homography estimator where it exists.
- Without ground truth: agreement report between two prediction directories (typically reference
  vs backend): keypoint-set recall/precision at 1 px, match Jaccard, match count.
- One new CLI `xfeat-score` (`--predictions DIR [--reference DIR] [--dataset megadepth1500|manifest]`).
  `xfeat-megadepth1500` stays as the reference-pipeline producer and writes the new directory format.

### D4. `backends/tensorrt`

- Layout mirrors `backends/ort`: `xfeat_trt/{engine,models,predict,bench}.py`; same public surface
  (`XFeatModel.extract`, `LighterGlueModel.match`, `run_raw`), Python 3.13 pin, own lock.
- Dependencies: `tensorrt-cu13`, `cuda-python` (uses `cuda.core`), `xfeat-core`, `xfeat-bench`,
  `xfeat-pairs`; libs pinned to the NVIDIA index for a hashed lock. Dev: `onnx` (tiny test graphs).
- Engine cache: `~/.cache/xfeat-rt/tensorrt/<key>.engine`, key = SHA-256 of ONNX bytes + TensorRT
  version + compute capability + GPU name + builder options. `None` from deserialize, or a missing
  file, rebuilds. A timing cache per (TensorRT version, GPU) shortens rebuilds. Engines are never
  committed.
- Inference: one stream, pinned host buffers, device buffers allocated once; optional CUDA graph.
- Hardware compatibility levels stay off (they disable cuDNN/cuBLAS tactics and bloat the plan).
- K handling: plain TopK at K <= 2048, `--chunked-topk` at 4096; the table also times chunked at
  2048 (build 16 s vs 52 s for plain).
- Bench: wall-clock `xfeat-bench.measure` with stream sync inside the call (no change to
  `measure`), reported three ways: engine only (inputs resident), end to end with H2D/D2H, and
  CUDA-graph. Cross-check headline numbers against `trtexec`. GPU clocks and power state recorded.

### D5. FP16 (decision for you)

FP16 is where TensorRT earns its speed on LighterGlue (explicit MatMul/Softmax attention is only
fused in FP16), but the spike's FP16 route (`onnxconverter-common`) produced invalid graphs.
Options:

| Option | What | Cost |
| ------ | ---- | ---- |
| A (recommended) | C3 ships FP32 end to end first; then `xfeat-export --precision fp16` exports mixed precision from torch (backbone convs and attention FP16, NMS/TopK/log-assignment FP32, explicit casts), gated on AUC delta | Export work; stays in C3 as checkpoint 3d |
| B | FP32 only in C3, FP16 in C6 as planned | Latency table lacks the main TensorRT win until C6 |

### Contract test

Parametrise `test_contract.py` over the backend (ORT, TensorRT); same 3 pairs, K = 2048 plain and
K = 4096 chunked. GPU tolerances come from a measured noise floor (reference CPU vs TensorRT FP32),
stored with their source as in C2; no copied outputs.

## Checkpoints

| # | Milestone | Demonstrable result |
| - | --------- | ------------------- |
| 3a | `backends/tensorrt` FP32, engine cache, TensorRT contract test, K = 1024/2048/4096 latency table | Contract test passes; cache hit skips the build; table in `c3-results.md` |
| 3b | `xfeat-pairs`: manifest, loader, prediction format; ORT and TensorRT predict CLIs write it | Round trip test (write with backends, read with scorer) |
| 3c | `xfeat-score`: MegaDepth-1500 from TensorRT predictions, custom pairs with and without ground truth | AUC within the reference's noise at the same canvas; your pairs scored |
| 3d | FP16 export and engines (if D5 = A) | FP16 table and AUC delta |

Each checkpoint: `make check` and `make test` pass and the real flow is run.

## Open items

- MegaDepth-1500 scoring at the exported canvas (1280x960 letterbox) differs from the C0 reference
  settings (640 long side, K = 1300). The reference is rerun at the same canvas for comparison;
  absolute AUC against the README gate is not expected to match.
- Timings need a rerun on an idle machine; WSL2 numbers are dev-only.
- Behaviour on a machine without a system CUDA toolkit is untested (`libcudart` came from
  `/usr/local/cuda-13.3`).
