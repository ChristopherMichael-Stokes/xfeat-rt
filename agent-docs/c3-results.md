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
