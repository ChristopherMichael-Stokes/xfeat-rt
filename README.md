# xfeat-rt

XFeat feature extraction and LighterGlue matching, optimized for CPU, CUDA GPUs and Apple MPS.

- `reference/`: PyTorch reference implementation, model export and scoring.
- `runtime/`: shared packages for the backends (metadata, pair manifests, timing), without PyTorch.
- `backends/`: one inference backend per directory (`ort`, `tensorrt`), without PyTorch.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and [prek](https://prek.j178.dev/).

```sh
uv tool install prek@0.5.7
make setup
```

`make setup` syncs `reference`, `runtime` and the backends with wheels for the machine: `ort` on
macOS arm64, `ort` and `tensorrt` on Linux x86_64 and aarch64. `make setup BACKENDS="ort"` picks
them explicitly; `make test` runs the tests of the same backends.

## Checks

```sh
make check
```

## Export

```sh
make export TOP_K=2048
```

writes the FP16 XFeat and LighterGlue graphs for the machine to `reference/uncommitted/models/`:
the Core ML target on Apple silicon, the TensorRT target elsewhere (`--target native`, also the
`xfeat-export` default).
`make export TARGET=tensorrt` or `TARGET=coreml` picks one explicitly.
`xfeat-export --help` (from `reference/`) covers FP32, the image size, batch sizes and plain TopK.

## Apple silicon

Core ML runs through ONNX Runtime's CoreML execution provider on an export rewritten for it
(`--target coreml`, the native target on Apple silicon):

```sh
make setup
make export TOP_K=2048
cd backends/ort
M=../../reference/uncommitted/models
uv run --frozen xfeat-ort-bench --provider coreml \
  --xfeat $M/xfeat_1280x960_k2048_b1_fp16_coreml.onnx \
  --lighterglue $M/lighterglue_k2048_b1_fp16_coreml.onnx --out-dir bench
uv run --frozen xfeat-ort-predict --provider coreml \
  --xfeat $M/xfeat_1280x960_k2048_b1_fp16_coreml.onnx \
  --lighterglue $M/lighterglue_k2048_b1_fp16_coreml.onnx \
  --manifest pairs.jsonl --resize letterbox --out-dir coreml
```

XFeat runs on `--xfeat-units ALL` (GPU and Neural Engine) and LighterGlue on
`--lighterglue-units CPUAndGPU`; the other choices are `CPUAndNeuralEngine` and `CPUOnly`.
Compiled models are cached in `--cache-dir` (default `uncommitted/coreml-cache`), so the first
load of a model takes seconds and later loads under 0.1 s.

## Results

Machine `wsl2-4070ti`: Ryzen 7 5800X3D (8C/16T) · RTX 4070 Ti 12 GB · WSL2, CUDA driver 13.4 ·
PyTorch 2.14.1+cu130, ONNX Runtime 1.31.0, TensorRT 11.4.0.106. Images are 1280x960 letterbox,
batch 1. Measured 2026-11-14 at 032d54c unless a row says otherwise.

### Latency per image pair

Pair = two XFeat calls + one LighterGlue match, wall-clock p50 ms including host-device copies,
excluding image loading. First 100 MegaDepth-1500 pairs. TopK is chunked on TensorRT.

| Backend                 | Precision       | K=2048 | K=4096 |
| ----------------------- | --------------- | -----: | -----: |
| PyTorch CPU (reference) | FP32            |    464 |   1154 |
| ONNX Runtime CPU        | FP32            |    282 |    802 |
| PyTorch CUDA            | FP32, FP16 attn |   27.8 |   31.3 |
| TensorRT                | FP32            |   11.5 |   29.9 |
| TensorRT                | FP16            |    9.6 |   15.7 |

PyTorch CUDA runs LighterGlue attention in FP16, so TensorRT FP16 is its like-for-like row.

### TensorRT engine time

`eager / CUDA graph` p50 ms: inputs on device, 20 warm-up + 100 runs, random input.

| Model              | Precision | K=1024       | K=2048      | K=4096        |
| ------------------ | --------- | ------------ | ----------- | ------------- |
| XFeat (image)      | FP32      | 3.60* / 1.30 | 1.66 / 1.53 | 1.90 / 1.75   |
| XFeat (image)      | FP16      | 1.15 / 0.75  | 1.19 / 0.98 | 1.46 / 1.18   |
| LighterGlue (pair) | FP32      | 4.60 / 1.60  | 5.17 / 4.29 | 22.16 / 21.70 |
| LighterGlue (pair) | FP16      | 2.90 / 0.77  | 3.97 / 1.62 | 8.00 / 6.10   |

\* Disturbed run: the same run's time with transfers included was 2.32 ms. A fresh timing cache
moves engine times by up to 1.2x.

### Accuracy, MegaDepth-1500

All 1500 pairs, relative-pose AUC (poselib RANSAC, best threshold), scored at 640 px long side.
Scoring noise is about 0.0005. `—` = not run yet.

| Backend                 | Precision | K    |  AUC@5 | AUC@10 | AUC@20 | Measured    |
| ----------------------- | --------- | ---- | -----: | -----: | -----: | ----------- |
| PyTorch CPU (reference) | FP32      | 2048 | 0.5626 | 0.7076 | 0.8162 | 2026-10-24  |
| ONNX Runtime CPU        | FP32      | 2048 | 0.5622 | 0.7068 | 0.8158 | 2026-10-24† |
| TensorRT                | FP32      | 2048 | 0.5626 | 0.7072 | 0.8157 | 2026-10-24† |
| TensorRT                | FP16      | 2048 | 0.5601 | 0.7048 | 0.8136 | 2026-11-14  |
| PyTorch CPU (reference) | FP32      | 4096 |      — |      — |      — |             |
| ONNX Runtime CPU        | FP32      | 4096 |      — |      — |      — |             |
| TensorRT                | FP32      | 4096 | 0.5828 | 0.7251 | 0.8282 | 2026-10-24† |
| TensorRT                | FP16      | 4096 | 0.5847 | 0.7274 | 0.8287 | 2026-11-14  |

† Measured on the matcher graph exported before 3bf54d8 (cross-attention with one shared
similarity matrix); not yet re-run on the current export.

### Batch throughput, TensorRT FP16

Pairs per second over all 1500 MegaDepth-1500 pairs (806 unique images), K=4096, model calls
only (upload, execute, download), inputs preloaded. Median of four runs.

| Mode                                                | Pairs/s |
| --------------------------------------------------- | ------: |
| Both images of every pair extracted, CUDA graphs    |      86 |
| Each unique image extracted once, XFeat batch 1     |     117 |
| Each unique image extracted once, XFeat batch 4     |    125‡ |

‡ Median of the two fast runs; the other two gave 107 with a slower LighterGlue pass.

## Scoring your own pairs

Describe the pairs in a JSON Lines manifest, one pair per line, image paths relative to the file.
Ground truth is optional: a `relative_pose` (`rotation`, `translation`, `intrinsics0`,
`intrinsics1` at the original image resolution) or a `homography` (`matrix`, original pixels).

```json
{"name": "a_b", "image0": "a.jpg", "image1": "b.jpg",
 "ground_truth": {"kind": "homography", "matrix": [[1,0,0],[0,1,0],[0,0,1]]}}
```

(One pair per line in the file; the example is wrapped for reading.)

A backend writes a prediction directory, `xfeat-score` scores it, and with `--reference` also
compares it with the PyTorch reference run on the same manifest:

```sh
cd reference
uv run --frozen xfeat-reference-predict --manifest pairs.jsonl --out-dir ref \
  --resize letterbox --top-k 2048
cd ../backends/tensorrt
uv run --frozen xfeat-trt-predict --xfeat X.onnx --lighterglue L.onnx \
  --manifest pairs.jsonl --resize letterbox --out-dir trt
cd ../../reference
uv run --frozen xfeat-score --predictions trt --manifest pairs.jsonl --reference ref \
  --output score.json
```

`--resize` fits each image to the exported size: `letterbox` keeps the aspect ratio and pads,
`stretch` ignores it. MegaDepth-1500 has its own manifest writer, `xfeat-megadepth1500-manifest`.

## Batch processing

For throughput rather than latency, export a ladder of batch sizes into its own directory and let
the TensorRT backend pick the batch size per model on this machine:

```sh
cd reference
uv run --frozen xfeat-export --target tensorrt --top-k 2048 --chunked-topk --precision fp16 \
  --batch 1 2 4 8 16 32 --out-dir uncommitted/ladder
cd ../backends/tensorrt
uv run --frozen xfeat-trt-calibrate --models ../../reference/uncommitted/ladder
uv run --frozen xfeat-trt-predict-batch --models ../../reference/uncommitted/ladder \
  --manifest pairs.jsonl --resize letterbox --out-dir trt-batch
```

`xfeat-trt-calibrate` times each batch size end to end and keeps the smallest one within
measurement noise of the best throughput; the result is stored with the engine cache and reused
until the GPU, driver, TensorRT version, build options or models change.
`xfeat-trt-predict-batch` runs the calibration first if there is none, extracts each unique image
once and writes the same prediction directory as `xfeat-trt-predict`. Its per-pair times are
shares of the batched calls.

## Addendum: ablations

Same machine and settings as the results above. None of the variants in the last table is in the
repo.

### Cross-attention export (3bf54d8)

LighterGlue's cross-attention is exported as one standard attention per direction, which TensorRT
fuses into its attention kernel, instead of one shared similarity matrix with two softmaxes.
FP16 engine with CUDA graph, p50 ms, and MegaDepth-1500 AUC@5:

| K    | Shared similarity | One attention per direction | AUC@5 before | AUC@5 after |
| ---- | ----------------: | --------------------------: | -----------: | ----------: |
| 2048 |              1.80 |                        1.65 |       0.5609 |      0.5601 |
| 4096 |              7.97 |                        5.96 |       0.5847 |      0.5847 |

### Batch-processing steps

Pairs per second at K=2048, FP16, 1500 pairs, measured before 3bf54d8. Each row adds one change.

| Variant                                        | Pairs/s |
| ---------------------------------------------- | ------: |
| Both images of every pair, batch 1, no graph   |     105 |
| + CUDA graph                                   |     151 |
| + each unique image extracted once             |     278 |
| + XFeat batch 2                                |     306 |

### TopK

Unchunked TopK adds about 5.5 ms per XFeat call on TensorRT, which also builds no single TopK
above K=3840.

### Precision below FP16

Timing probes with uncalibrated scales on TensorRT; accuracy not measured. Change against the FP16
engine with CUDA graph.

| Variant                                         | K=2048 | K=4096 |
| ----------------------------------------------- | -----: | -----: |
| FP8 attention, LighterGlue engine               |   +1 % |  -15 % |
| FP8 attention, whole pair                       |    0 % |   -8 % |
| INT8 linear layers, LighterGlue engine          |  +30 % |  +12 % |
| INT8 XFeat backbone, whole pair                 |   +3 % |      — |
