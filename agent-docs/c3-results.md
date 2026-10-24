# C3 results

## 3a: `backends/tensorrt` FP32, engine cache, contract test, latency

`make check` and `make test` pass. The contract test runs ORT and TensorRT over K = 2048 plain,
K = 2048 chunked and K = 4096 chunked (18 cases).

### Built

| Path | Holds |
| ---- | ----- |
| `backends/tensorrt/src/xfeat_trt/engine.py` | Engine build from ONNX (strongly typed, TF32 off, timing cache), cache keyed on ONNX bytes + TensorRT version + compute capability + device name + options, rebuild when a plan fails to deserialize |
| `.../runner.py` | `EngineRunner`: stream, persistent device and pinned buffers, optional CUDA graph |
| `.../models.py` | `XFeatModel`, `LighterGlueModel` with the same surface as `backends/ort` |
| `.../predict.py`, `bench.py` | `xfeat-trt-predict` (same `.npz` as ORT until the prediction format lands), `xfeat-trt-bench` |
| `backends/tensorrt/stubs/` | Typing stubs for the `tensorrt` and `cuda.core` subset in use (neither package is typed for `ty`) |
| `backends/tensorrt/tests/` | Cache hit, corrupt-plan rebuild, key change, metadata read-back on a tiny graph |

- `tensorrt-cu13-libs` is pinned to `https://pypi.nvidia.com/`; `uv.lock` holds hashed wheels and
  no torch. `uv sync` installs and imports on the WSL2 box.
- `onnx` is a runtime dependency of the TensorRT backend: the TensorRT parser does not expose
  `metadata_props`, so the build step reads them with `onnx` and stores them beside the engine.

### Findings

- TF32 is on by default in TensorRT 11 FP32 builds. The backend clears it unless `--tf32` is
  given (`BuildOptions.tf32`, part of the cache key). Against the FP32 reference (5 pairs, 10
  images):

  | Build | Keypoint swaps per image (max) | Max score diff | Min descriptor cosine | Min match Jaccard |
  | ----- | ------------------------------ | -------------- | --------------------- | ----------------- |
  | K = 2048, TF32 off | 0 | 2.6e-6 | 0.9999996 | 1.0 |
  | K = 2048, `--tf32` | 17 | 6.7e-3 | 0.99998 | 0.94 |
  | K = 4096, `--tf32` | 77 | 6.7e-3 | 0.99997 | 0.91 |
  | ORT CPU, K = 2048 (same run) | 1 | 6.0e-5 | 0.9999993 | 0.98 |

- Plain `TopK` over the 1.2 Mpx score map costs about 5.5 ms on TensorRT. The chunked graph is
  faster even at K = 1024 and 2048 (XFeat 7.2 ms to 1.8 ms at K = 2048). Chunked is the TensorRT
  export; the plain graph only matters for the K <= 3840 cap.
- K = 4096 builds and passes the contract test with the chunked graph.

### CUDA latency (TensorRT 11.4.0.106, RTX 4070 Ti, WSL2, FP32, 1280x960, batch 1)

p50 ms, 100 runs after 20 warm-up, chunked export, random input, load average about 2. "engine" is
execute plus stream sync with inputs resident; "end to end" adds upload and download. Raw samples
are in `backends/tensorrt/uncommitted/bench/`.

| K | XFeat engine | XFeat end to end | XFeat engine, graph | LighterGlue engine | LighterGlue end to end | LighterGlue engine, graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1.56 | 2.21 | 1.29 | 3.85 | 4.09 | 1.56 |
| 2048 | 1.83 | 2.72 | 1.54 | 5.10 | 5.35 | 4.46 |
| 4096 | 1.90 | 2.78 | 2.02 | 22.99 | 23.93 | 24.62 |

- Against ORT CPU (C2): XFeat about 25x faster, LighterGlue 24x at K = 2048 and 29x at K = 4096.
- XFeat is flat in K. LighterGlue grows about 4.5x from 2048 to 4096 and is 12x XFeat at 4096: the
  FP16 and fused-attention work of 3d targets it.
- The CUDA graph helps LighterGlue at K = 1024 (3.85 to 1.56 ms, launch-bound) and not at 4096.
- Single machine, WSL2; dev-only until rechecked on native Linux. `trtexec` cross-check not done.

### Build options (K = 2048, chunked, p50 ms of the engine, fresh timing cache, two builds each)

| `--tf32` | `--optimization-level` | XFeat | LighterGlue |
| -------- | ---------------------- | ----- | ----------- |
| off | 3 (default) | 1.85, 2.27 | 4.84, 5.58 |
| on | 3 | 1.70, 2.19 | 4.25, 4.46 |
| off | 5 | 1.70, 1.90 | 5.06, 5.42 |
| on | 5 | 1.60, 1.69 | 4.28, 4.33 |

- TF32 saves about 10 to 20 % of LighterGlue and 5 to 10 % of XFeat at K = 2048 (LighterGlue 10 %
  at K = 4096: 22.6 to 20.2 ms) for the accuracy loss above. The score error is the same at both K
  and the keypoint swaps grow with K because more near-tied candidates sit at the K-th slot.
- Optimization level 5 gives at most a few percent over 3 and is within build-to-build spread.
- Identical settings rebuilt with a fresh timing cache differ by up to 1.2x (XFeat 1.85 vs 2.27 ms;
  one rebuild measured 3.1 ms): tactic selection is noisy. Latency tables compare medians of
  repeated builds, not one engine.

### Not done in 3a

- `trtexec` cross-check of the headline numbers.
- `make export` has no chunked option; the chunked models were exported with
  `xfeat-export --chunked-topk --out-dir uncommitted/models-chunked`.

## 3b: `xfeat-pairs`, manifest and prediction directory

- `runtime/packages/xfeat-pairs`: JSON Lines pair manifest with optional `relative_pose` or
  `homography` ground truth; `stretch` and `letterbox` preprocessing (8-bit RGB resize, then channel
  mean, as the reference); prediction directory (`index.json` written last, `pairs/NNNNNN.npz`,
  keypoints in original-image pixel coordinates, per-pair extract and match wall time, both models'
  `metadata_props`, backend and settings); `predict_pairs` loop shared by the backends.
- `xfeat-ort-predict` and `xfeat-trt-predict` take `--manifest --resize --out-dir` and write that
  directory. The `.npy`/`pair{i}.npz` interface is gone, and `rescale_keypoints` left `xfeat-core`
  (the preprocessing mapping replaces it and handles letterbox).
- The contract test builds a manifest from MegaDepth-1500, runs the backend CLI and reads the
  directory with `xfeat-pairs`, mapping keypoints back to model pixels by its own derivation.
  18 cases pass.
- Resizing the gray image instead of the RGB image changed pixels by up to 1/255 and the keypoint
  set by up to 156 per image (near-tied candidates); the loader resizes RGB first.
- `xfeat-evaluation` depends on `xfeat-pairs` by path; `reference/uv.lock` resolves it under the
  `numpy<2.4` override.

## 3c: `xfeat-score`, reference predictions, own-pair scoring

- `xfeat-megadepth1500-manifest` writes the 1500 MegaDepth-1500 pairs with relative-pose ground
  truth (`pairs_calibrated.txt`: K0, K1 at original resolution, R and t row-major).
- `xfeat-reference-predict` runs the PyTorch reference over a manifest with the backends'
  preprocessing, canvas and K, writing the same prediction directory.
- `xfeat-score` scores a prediction directory with GlueFactory's poselib relative-pose and
  homography estimators and its AUC code, at a common long side (`--resize`, default 640), and
  with `--reference` reports keypoint recall/precision and match Jaccard against a second run.
  Pairs without ground truth contribute to the agreement report only.
- Tests build matches from known 3-D points and a known homography (independent of the code under
  test); a wrong ground-truth rotation drops AUC@5 below 0.1.

### MegaDepth-1500, all 1500 pairs, K = 2048, 1280x960 letterbox, FP32, scored at 640 long side

| Run | AUC@5 | AUC@10 | AUC@20 | mAA | Keypoint recall (min) | Match Jaccard (mean, min) |
| --- | ----- | ------ | ------ | --- | --------------------- | ------------------------- |
| PyTorch reference, CPU | 0.5626 | 0.7076 | 0.8162 | 0.6955 | 1 | 1, 1 |
| TensorRT, TF32 off | 0.5626 | 0.7072 | 0.8157 | 0.6952 | 0.9998 | 0.9999, 0.982 |
| ORT CPU | 0.5622 | 0.7068 | 0.8158 | 0.6949 | 0.9993 | 0.9988, 0.941 |

- Backend AUCs sit within 0.0005 of the reference. poselib is unseeded, so repeated scoring of the
  same predictions moves AUC by about that much.
- These are not comparable to the README gate (0.444/0.610/0.746 at 640 px input, K = 1300): the
  canvas and K are larger here.
- Per-pair wall time p50 (both images' extraction, then matching, with preprocessing excluded):
  reference 236 + 258 ms, ORT 94 + 184 ms, TensorRT 13.1 + 12.3 ms. The TensorRT extraction time
  is 2 images at 6.5 ms each against 2.7 ms in the bench loop; host-side conversion between
  pairs and an idle GPU clock are the likely cause and were not investigated.
- Raw predictions and scores: `reference/uncommitted/c3/full2048/`.

## 3d: FP16 export and engines

- `xfeat-export --precision fp16`: the XFeat backbone and the LighterGlue input projection and
  transformer layers are converted to half precision in PyTorch before export. Detection (softmax,
  NMS, TopK), descriptor sampling and normalisation, LighterGlue positional encoding and the
  assignment head (log-softmax, log-sigmoid, mutual filter) stay FP32, and graph inputs and outputs
  stay FP32, so backends and the prediction format are unchanged. Files carry an `_fp16` suffix and
  `precision` in their metadata (new required key in `xfeat-core`).
- The converter route of the C1 spike (`onnxconverter-common`) is not used: it produced invalid
  graphs. Strongly typed TensorRT builds the half-precision layers on tensor cores and fuses the
  explicit attention.

### CUDA latency, FP32 vs FP16 (p50 ms of the engine, RTX 4070 Ti, WSL2, 1280x960, batch 1, chunked)

| K | XFeat FP32 | XFeat FP16 | LighterGlue FP32 | LighterGlue FP16 | LighterGlue FP16, CUDA graph |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1.56 | 1.06 | 3.85 | 3.29 | 0.82 |
| 2048 | 1.83 | 1.30 | 5.10 | 4.28 | 1.81 |
| 4096 | 1.90 | 1.53 | 22.99 | 8.42 | 7.84 |

- FP16 LighterGlue at K = 4096 is 2.7x faster than FP32; at K = 1024 and 2048 the eager engine is
  launch-bound and the CUDA graph takes it to 0.8 and 1.8 ms.
- XFeat with graph: 0.78 / 0.99 / 1.20 ms at K = 1024 / 2048 / 4096.

### MegaDepth-1500, all 1500 pairs, TensorRT, 1280x960 letterbox, scored at 640 long side

| K | Precision | AUC@5 | AUC@10 | AUC@20 | mAA | Keypoint recall (min) | Match Jaccard (mean) | Matches per pair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | FP32 (= reference) | 0.5626 | 0.7072 | 0.8157 | 0.6952 | 0.9998 | 0.9999 | 632.3 |
| 2048 | FP16 | 0.5602 | 0.7069 | 0.8157 | 0.6943 | 0.9878 | 0.919 | 631.1 |
| 4096 | FP32 | 0.5828 | 0.7251 | 0.8282 | 0.7120 | 1 | 1 | 1366.0 |
| 4096 | FP16 | 0.5828 | 0.7249 | 0.8273 | 0.7117 | 0.9893 | 0.923 | 1364.1 |

- FP16 costs at most 0.0024 AUC (at 5 degrees, K = 2048) and 0.0003 mAA at K = 4096, the size of
  poselib's run-to-run spread (about 0.0005 between repeated scorings). The matches differ
  (Jaccard 0.92) without a measurable loss of pose accuracy.
- K = 4096 raises AUC@5 by 0.020 over K = 2048 (FP32 and FP16 alike), the first point of the K
  sweep: FP16 TensorRT runs XFeat plus LighterGlue at K = 4096 in about 10 ms, 14 ms with
  transfers.
- The per-pair wall times in the score files varied 2x between runs on a busy machine (other jobs
  ran concurrently) and are not latency results; the bench table above is.
- Not done: bootstrap confidence intervals on the AUC differences (the plan's acceptance rule) and
  the INT8 question.
