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
