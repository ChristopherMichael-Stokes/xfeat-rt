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
