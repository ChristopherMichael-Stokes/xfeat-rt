# xfeat-rt

XFeat feature extraction and LighterGlue matching, optimized for CPU, CUDA GPUs and Apple MPS.

- `reference/`: PyTorch reference implementation and model export.
- `runtime/`: inference backends and pipeline, without PyTorch.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and [prek](https://prek.j178.dev/).

```sh
uv tool install prek@0.5.7
make setup
```

## Checks

```sh
make check
```

## Performance log

### Latency per image pair

Pair = two XFeat calls + one LighterGlue match, wall-clock p50 ms including host-device copies,
excluding image loading. K=2048, 1280x960 letterbox, batch 1, first 100 MegaDepth-1500 pairs,
machine idle.

| Backend                 | Precision       | wsl2-4070ti |
| ----------------------- | --------------- | ----------: |
| PyTorch CPU (reference) | FP32            |         420 |
| ONNX Runtime CPU        | FP32            |         291 |
| PyTorch CUDA            | FP32, FP16 attn |        28.9 |
| TensorRT                | FP32            |        11.7 |
| TensorRT                | FP16            |    **10.3** |

### Accuracy, MegaDepth-1500

All 1500 pairs, relative-pose AUC (poselib RANSAC, best threshold), scored at 640 px long side.
Scoring noise is about 0.0005. `—` = not run yet.

| Backend                 | Precision | K    |  AUC@5 | AUC@10 | AUC@20 | Machine     |
| ----------------------- | --------- | ---- | -----: | -----: | -----: | ----------- |
| PyTorch CPU (reference) | FP32      | 2048 | 0.5626 | 0.7076 | 0.8162 | wsl2-4070ti |
| ONNX Runtime CPU        | FP32      | 2048 | 0.5622 | 0.7068 | 0.8158 | wsl2-4070ti |
| TensorRT                | FP32      | 2048 | 0.5626 | 0.7072 | 0.8157 | wsl2-4070ti |
| TensorRT                | FP16      | 2048 | 0.5602 | 0.7069 | 0.8157 | wsl2-4070ti |
| PyTorch CPU (reference) | FP32      | 4096 |      — |      — |      — | wsl2-4070ti |
| ONNX Runtime CPU        | FP32      | 4096 |      — |      — |      — | wsl2-4070ti |
| TensorRT                | FP32      | 4096 | 0.5828 | 0.7251 | 0.8282 | wsl2-4070ti |
| TensorRT                | FP16      | 4096 | 0.5828 | 0.7249 | 0.8273 | wsl2-4070ti |

### wsl2-4070ti

Ryzen 7 5800X3D (8C/16T) · RTX 4070 Ti 12 GB · WSL2, CUDA driver 13.4 · PyTorch 2.14.1+cu130,
ONNX Runtime 1.31.0, TensorRT 11.4.0.106 · 2026-10-24, c4636b0.

Pair p50 ms, 1280x960 letterbox, batch 1, 100 pairs, idle. TopK chunked on TensorRT.

| Backend          | Precision       | K=2048 | K=4096 |
| ---------------- | --------------- | -----: | -----: |
| PyTorch CPU      | FP32            |    420 |   1180 |
| ONNX Runtime CPU | FP32            |    291 |    786 |
| PyTorch CUDA     | FP32, FP16 attn |   28.9 |   32.0 |
| TensorRT         | FP32            |   11.7 |   30.8 |
| TensorRT         | FP16            |   10.3 |   15.7 |

TensorRT engine time, `eager / CUDA graph` p50 ms: inputs on device, 20 warm-up + 100 runs,
random input.

| Model              | Precision | K=1024      | K=2048      | K=4096        |
| ------------------ | --------- | ----------- | ----------- | ------------- |
| XFeat (image)      | FP32      | 1.56 / 1.29 | 1.83 / 1.54 | 1.90 / 2.02   |
| XFeat (image)      | FP16      | 1.06 / 0.78 | 1.30 / 0.99 | 1.53 / 1.20   |
| LighterGlue (pair) | FP32      | 3.85 / 1.56 | 5.10 / 4.46 | 22.99 / 24.62 |
| LighterGlue (pair) | FP16      | 3.29 / 0.82 | 4.28 / 1.81 | 8.42 / 7.84   |

PyTorch CUDA runs LighterGlue attention in FP16, so TensorRT FP16 is its like-for-like row.
Unchunked TopK adds about 5.5 ms per XFeat call; a fresh timing cache moves engine times by up to
1.2x.

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
