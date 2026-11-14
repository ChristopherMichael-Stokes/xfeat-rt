# TensorRT backend results

CUDA backend on WSL2, RTX 4070 Ti: install spike, FP32 backend, pair loader and scoring, FP16.
Open: behaviour on a machine without a system CUDA toolkit; `trtexec` cross-check; rerun of
timings on an idle native-Linux machine.

## Install and engine-build spike

Run on 2026-10-24, WSL2, RTX 4070 Ti (sm_89), driver CUDA UMD 13.4, system CUDA toolkit 13.3
at `/usr/local/cuda-13.3`, uv 0.13.5. Scratch project (not in the repo):
`/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/trt/`
(scripts `mk.py`, `build.py`, `run.py`, `batch.py`, `fp16b.py`). ONNX inputs came from
`.../scratchpad/c1/onnx/` (produced by another agent).

### `uv lock` / `uv sync` for a torch-free TensorRT project

`pyproject.toml`: `requires-python = ">=3.13"`, dependencies `tensorrt-cu13`, `cuda-python`,
`numpy`. Dev group (added later): `onnx`, `onnxconverter-common`.

| Item | Result |
| --- | --- |
| `uv lock` | works, 0.28 s, 9 packages |
| `uv sync --python 3.13` (cold cache) | works, 54 s (stubs build, 4.4 GB of libs downloaded) |
| `uv sync --python 3.14.9` in a copy (warm cache) | works, 5.7 s, identical lock; `import tensorrt` and `Builder` creation work on 3.14.9 |
| `uv lock` default interpreter | picked `CPython 3.14.0+freethreaded` (the only local 3.14); lock content identical because the lock is keyed on `requires-python` only |
| `grep -ci torch uv.lock` | 0 (also 0 after adding `onnx` and `onnxconverter-common`) |

Resolved versions: tensorrt-cu13 11.4.0.106, tensorrt-cu13-bindings 11.4.0.106,
tensorrt-cu13-libs 11.4.0.106, cuda-python 13.4.4, cuda-bindings 13.4.4, cuda-core 1.2.2,
cuda-pathfinder 1.8.4, numpy 2.5.4; dev: onnx 1.23.2, protobuf 7.36.2, ml-dtypes 0.6.0,
onnxconverter-common 1.16.0, packaging 26.3.

Stub packages:

- `tensorrt-cu13` and `tensorrt-cu13-libs` are sdist-only in the lock (17,970 B and 15,508 B).
  `uv sync` printed `Building tensorrt-cu13==11.4.0.106` and `Building tensorrt-cu13-libs==11.4.0.106`.
- The libs stub does pull the real libraries at build time: `.venv/.../tensorrt_libs` is 4.4 GB
  (`libnvinfer.so.11` 715 MB, builder resources for sm75/80/86/89/90/100/120, plus `win_*`
  resource files on Linux), `.venv` total 4.5 GB.
- The real library wheel is not in `uv.lock` (no URL or hash for it): the lock pins only the
  15 KB sdist. The download from NVIDIA's index is not covered by the lock.
- Warm cache reuse works (second venv synced in 5.7 s).
- `tensorrt-cu13-bindings` is a normal 1.2 MB wheel.

### Runtime environment on WSL2

- `import tensorrt` prints 11.4.0.106; `trt.Builder(logger)` creates on the GPU
  (`[MemUsageChange] Init CUDA`); `cuda.bindings.runtime` sees 1 device,
  `NVIDIA GeForce RTX 4070 Ti`.
- No `LD_LIBRARY_PATH` or other environment fix was needed (`LD_LIBRARY_PATH` was empty).
- `libcuda.so.1` comes from `/usr/lib/wsl/lib`. `cuda-python` located `libcudart.so.13` through
  `cuda.pathfinder` with `found_via = system-search`, at `/usr/local/cuda/targets/x86_64-linux/lib/`
  (also libnvrtc 13.3.33 from `/usr/local/cuda-13.3`). The venv contains no `libcudart`.
  Whether a machine without a system CUDA toolkit works was not tested. `cuda-bindings` has an
  extra `all` that adds `cuda-toolkit[...]==13.*` wheels.

### API differences in TensorRT 11 (checked from the installed package)

- `Builder.platform_has_fast_fp16` does not exist.
- `trt.BuilderFlag` has no `FP16`, `INT8` or `BF16`. The only network creation flag is
  `STRONGLY_TYPED`; `create_network(0)` still works.
- Consequence: FP16 is selected by the ONNX tensor types (strongly typed network), not by a
  builder flag. An FP32 ONNX built with `STRONGLY_TYPED` runs in FP32. An FP16 engine needs an
  FP16-typed ONNX.
- Engine plan: `builder.build_serialized_network` returns `IHostMemory` (`nbytes`, `dtype`;
  no `len`). Inference used `execute_async_v3` with `set_tensor_address` and cuda-python
  `cudaMalloc`/`cudaMemcpy`.

### Tiny graph: TopK, GridSample cubic, MaxPool

Graph made with `onnx.helper` (opset 20, no torch): `MaxPool` 5x5 stride 1 pad 2 on 64x64,
`Reshape`, `TopK(K)`, and `GridSample(mode="cubic", padding_mode="border", align_corners=0)`.
Built with `STRONGLY_TYPED`, FP32 and FP16 typed models. Both ran on the GPU with finite
outputs of the expected shapes.

| K | FP32 | FP16 |
| --- | --- | --- |
| 1024 | build ok | build ok |
| 3839 | build ok | build ok |
| 3840 | build ok | build ok |
| 3841 | fail | fail |
| 4096 | fail | fail |

Error text for K=4096 (identical for K=3841):

```text
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error (ITopKLayer
    node_of_vals: node_of_vals: K exceeds the maximum value allowed (3840). In processCheck at
    /_src/optimizer/shapeof/graphShapeAnalyzer.cpp:1066 In analyzeShapes at
    /_src/optimizer/shapeof/graphShapeAnalyzer.cpp:2826)
```

The limit is K <= 3840 inclusive (the documented "K < 3840" is off by one). The largest K that
builds in a single TopK is 3840. Cubic GridSample parsed and built without warnings in both
precisions.

### Real XFeat and LighterGlue ONNX

Build code (`build.py`): `create_network(STRONGLY_TYPED)`, `OnnxParser.parse_from_file`,
default `BuilderConfig`, `build_serialized_network`. FP16 ONNX made with
`onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True, ...)`.
Model: 1280x960, batch 1. All timings are single builds on a cold timing cache.

FP32 (all parse with no unsupported-op errors):

| ONNX | Build s | Engine MB | Result |
| --- | --- | --- | --- |
| xfeat_1280x960_k2048 (plain TopK) | 51.7 | 28.91 | ok, runs |
| xfeat_1280x960_k2048_chunked | 16.4 | 28.96 | ok, runs |
| xfeat_1280x960_k2048_relresize | 13.0 | 21.91 | ok |
| xfeat_1280x960_k2048_chunked_relresize | 15.7 | 22.00 | ok |
| xfeat_1280x960_k2048_relresize_s2d | 11.5 | 21.90 | ok |
| xfeat_1280x960_k2048_chunked_relresize_s2d | 11.8 | 21.95 | ok |
| xfeat_1280x960_k4096 (plain TopK) | n/a | n/a | fail: TopK K exceeds 3840 |
| xfeat_1280x960_k4096_relresize | n/a | n/a | fail: same |
| xfeat_1280x960_k4096_relresize_s2d | n/a | n/a | fail: same |
| xfeat_1280x960_k4096_chunked | 12.4 | 29.25 | ok, runs |
| xfeat_1280x960_k4096_chunked_relresize | 12.2 | 22.04 | ok |
| xfeat_1280x960_k4096_chunked_relresize_s2d | 12.0 | 22.00 | ok, runs |
| xfeat_backbone_1280x960 | 28.8 | 21.98 | ok, runs |
| xfeat_backbone_1280x960_pixelops | 11.4 | 21.80 | ok |
| xfeat_backbone_1280x960_s2d | 20.6 | 21.80 | ok |
| lighterglue_k2048 | 8.0 | 5.32 | ok, runs |
| lighterglue_k4096 | 8.8 | 5.34 | ok, runs |
| stage_*: nms, topk_plain_k2048, topk_chunked_k2048/k4096, desc_k2048, rel_candidates_n16384, rel_dense_resize, rel_dense_gridsample | 0.5 to 1.9 | 0.02 to 9.84 | ok |
| stage_topk_plain_k4096 | n/a | n/a | fail: TopK K exceeds 3840 |
| probe_sdpa_opset20 | 3.0 | 0.06 | ok |
| probe_sdpa_opset23 (ONNX `Attention` op) | n/a | n/a | process segfault (exit -11, SIGSEGV) during build; the build log shows `Unused Input: q`, `Unused Input: k` warnings before the crash |

K=4096 failure text (plain TopK models):

```text
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error (ITopKLayer
    node_topk__1: node_topk__1: K exceeds the maximum value allowed (3840). In processCheck at
    /_src/optimizer/shapeof/graphShapeAnalyzer.cpp:1066 In analyzeShapes at
    /_src/optimizer/shapeof/graphShapeAnalyzer.cpp:2826)
```

Chunked TopK (three 2048-wide chunks plus merge, per the export metadata) builds at K=4096 and
runs. Largest plain K that builds is 3840 (tiny graph); no real XFeat model at K=3840 was
exported, so that figure comes from the tiny graph only.

FP16 (ONNX converted with onnxconverter-common):

| Model | Converter setting | Result |
| --- | --- | --- |
| Full XFeat models (k2048, k2048_chunked, k2048_relresize, k4096, k4096_chunked) | default op block list | fail at parse, see below |
| xfeat_1280x960_k2048, k2048_chunked, k2048_relresize, k4096_chunked | op block list extended with `Div, Mod, Floor, TopK, Gather, Greater, Where, Equal, And, Clip, Sub, Add, Mul, Expand, Slice, Concat, MaxPool` | build ok, 22.3 to 23.6 s, 46.2 / 46.2 / 41.3 / 46.3 MB, runs |
| xfeat_backbone_1280x960 | default | ok, 45.5 s, 19.10 MB (FP32: 21.98 MB), runs |
| backbone pixelops / s2d, k2048 and k4096 chunked_relresize (with and without s2d), k2048_relresize_s2d | extended block list | ok, 18.9 to 52.5 s, 41.2 to 41.5 MB |
| lighterglue_k2048, lighterglue_k4096 | default; two extended block lists tried on k2048 | fail at parse |
| stage_* models, probe_sdpa_opset20/23 | default | ok |

FP16 failure texts:

```text
[TRT] [E] ITensor::getDimensions: Error Code 4: API Usage Error (node_Div_544:
    ElementWiseOperation DIV must have same input types. But they are of types Float and Half. In
    validateTypes at /_src/optimizer/common/nodes/elementWiseNode.cpp:59)
[TRT] [E] ITensor::getDimensions: Error Code 4: API Usage Error (node_gt: ElementWiseOperation
    GREATER must have same input types. But they are of types Half and Float. In validateTypes at
    /_src/optimizer/common/nodes/elementWiseNode.cpp:59)
```

Cause found for the XFeat `Div`: the converter turns the constant divisor (`1280`, from
`index // width`) into FP16 but leaves the `Cast(to=1)` of the integer index in FP32. For
LighterGlue, `Greater` consumes `mscores0` (a graph output) through a converter-inserted
`Cast(to=10)` on input 0 and `Cast(to=1)` on input 1. These are converter output defects, not
TensorRT parse limits. No FP16 LighterGlue engine was produced; no manual fix was attempted.

Observations on FP16:

- The block list keeps the post-processing and elementwise ops in FP32 and gives FP16
  engines larger than FP32 (41 to 46 MB versus 22 to 29 MB). The cause was not investigated.
- The `stage_nms` FP16 engine is 17.23 MB versus 0.02 MB in FP32 (cause not investigated).
- Converter warnings: `-inf` truncated to `-10000.0`; `9.99e-13` and `2.57e-08` truncated to
  `1e-07`.
- Precision accuracy was not checked (out of scope).

Execution check with `run.py` on random inputs (uniform [0,1) image; zero-filled
non-float inputs):

| Engine | Outputs | Finite | Valid count |
| --- | --- | --- | --- |
| xfeat k2048 FP32, k2048_chunked FP32 | kpts [1,2048,2] f32, scores [1,2048], desc [1,2048,64], valid [1,2048] bool | yes | 2048 |
| xfeat k2048 FP16 (block list) | kpts f32, scores and desc f16 | yes | 2048 |
| xfeat k4096_chunked FP32, FP16 (block list), k4096_chunked_relresize_s2d FP32 | same with 4096 | yes | 4096 |
| xfeat_backbone FP16 | feats [1,64,120,160], prob [1,1,960,1280], reliability [1,1,120,160] f32 | yes | n/a |
| lighterglue k2048 / k4096 FP32 | m0 int64 [1,K], mscores0 f32 [1,K] | yes | m0 all -1 (all-invalid zero inputs, so uninformative) |

Plain and chunked K=2048 produced the same output ranges (kpts 0 to 1278; scores
0.00106 to 0.0205). Numerical cross-checks against the reference were not done.

### Spike commands

```bash
cd <scratch>/trt                     # pyproject.toml as described above
uv lock
uv sync --python 3.13
uv add --dev onnx onnxconverter-common
uv run python t2.py                  # import, Builder, cuda-python device query
uv run python mk.py w/t_fp32_3840.onnx 3840 fp32   # tiny graph, args: out K dtype
uv run python build.py w/t_fp32_3840.onnx w/t_fp32_3840.plan
uv run python batch.py <onnx basenames>             # FP32 + FP16 builds, log in logs/trt.log
uv run python fp16b.py <basename>                   # FP16 with extended block list
uv run python run.py plans/<name>.plan              # deserialize, run once, print outputs
```

### Spike answers

| Question | Answer |
| --- | --- |
| Does a torch-free uv project lock and sync TensorRT 11.4 cu13 + cuda-python on 3.13 and 3.14? | Yes, no torch in the lock; 4.5 GB venv; libs are fetched by the stub build and are not hashed in `uv.lock` |
| Does it work on WSL2 without environment fixes? | Yes; `libcudart` came from the system CUDA 13.3 toolkit |
| Do cubic GridSample, TopK and MaxPool build in FP32 and FP16? | Yes |
| TopK limit | K <= 3840; K=3841 and 4096 fail with the error above |
| Do real XFeat and LighterGlue parse and build? | FP32: all do, except plain TopK at K=4096; chunked K=4096 builds. FP16: XFeat builds only with a custom converter block list; LighterGlue FP16 failed with onnxconverter-common |
| ONNX `Attention` (opset 23) | segfault during TensorRT 11.4 FP32 build |

## `backends/tensorrt` FP32, engine cache, contract test, latency

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

- Against ORT CPU (`ort-backend-results.md`): XFeat about 25x faster, LighterGlue 24x at
  K = 2048 and 29x at K = 4096.
- XFeat is flat in K. LighterGlue grows about 4.5x from 2048 to 4096 and is 12x XFeat at 4096: the
  FP16 and fused-attention work below targets it.
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

### Not done in the FP32 stage

- `trtexec` cross-check of the headline numbers.
- The chunked models were exported with
  `xfeat-export --chunked-topk --out-dir uncommitted/models-chunked`.

## `xfeat-pairs`, manifest and prediction directory

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

## `xfeat-score`, reference predictions, own-pair scoring

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
- Raw predictions and scores were deleted on 2026-11-14 (15 GB); regenerate with
  `xfeat-trt-predict` and `xfeat-score`.

## FP16 export and engines

- `xfeat-export --precision fp16`: the XFeat backbone and the LighterGlue input projection and
  transformer layers are converted to half precision in PyTorch before export. Detection (softmax,
  NMS, TopK), descriptor sampling and normalisation, LighterGlue positional encoding and the
  assignment head (log-softmax, log-sigmoid, mutual filter) stay FP32, and graph inputs and outputs
  stay FP32, so backends and the prediction format are unchanged. Files carry an `_fp16` suffix and
  `precision` in their metadata (new required key in `xfeat-core`).
- The converter route of the install spike (`onnxconverter-common`) is not used: it produced invalid
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
