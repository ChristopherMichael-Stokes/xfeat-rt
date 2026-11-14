# Research: precision below FP16 for XFeat and LighterGlue on TensorRT

Question: which route below FP16 gives the most speed on the RTX 4070 Ti and the Orin NX, and
what does it cost in accuracy? Gathered 2026-11-14. [V] = source opened or probed locally;
[U] = unverified.

## TensorRT 11.4 mechanics

- INT8 and FP8 exist only as `QuantizeLinear`/`DequantizeLinear` pairs in the ONNX, built strongly
  typed. `BuilderFlag.INT8`, calibrators and `ITensor.dynamic_range` are removed. [V, installed
  package; [10-to-11 migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-reference.html)]
- Zero point must be 0; per-tensor scales on activations, per-channel (axis 0) on weights; QLinear
  operators are rejected. [V, [explicit quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html)]
- Placement: quantize the inputs of weighted layers (Conv, Gemm), not their outputs; quantize the
  residual input of a skip connection; max pooling commutes with the pairs. [V, same page]
- Opset 20 with float16 scales on float16 tensors builds; no Cast nodes are needed. [V, probe]
- Conv + BatchNorm + ReLU with pairs on the conv input and after the ReLU builds as one INT8
  kernel. [V, probe]
- A bare product of two activations with pairs on both inputs runs as INT8 gemm with a Float
  output and a separate Cast kernel. [V, probe] Softmax stays float.
- InstanceNorm, LayerNorm and GridSample take FP32/FP16/BF16 only. [V, [onnx-tensorrt operators](https://raw.githubusercontent.com/onnx/onnx-tensorrt/main/docs/operators.md)]
- Which layers ran in which type: build with `ProfilingVerbosity.DETAILED`, read
  `create_engine_inspector().get_engine_information(LayerInformationFormat.JSON)`. [V, probe]

## Fused attention

Source: [TensorRT fused attention](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/transformers-fused-attention.html) [V].

| Type | GPUs | Head size H | Sequence length S |
| ---- | ---- | ----------- | ----------------- |
| FP16 | SM75 to SM90 | 16 to 256, multiple of 8 | any |
| INT8 | | 16, 32 or 64 | at most 512 |
| FP8 | SM89 and later | 32 to 256, multiple of 16 | at least 32 |

- LighterGlue: 1 head, H = 96, S = 2048 or 4096 [V, verlab `modules/lighterglue.py`]. INT8 fused
  attention cannot apply on either device. FP8 can on the 4070 Ti (SM89), not the Orin (SM87, [U]
  inferred from Ampere).
- "FP8 fused Attention is expected to outperform FP16 fused Attention when H ≥ 128 or S ≥ 128." [V]
- Pattern: MatMul, Softmax, MatMul with Q or K pre-scaled and at most one mask op; the docs advise
  against pairs inside attention that cannot fuse. [V]
- The cached FP16 LighterGlue engines hold 12 `_gemm_mha_v2` kernels (self-attention) and leave the
  6 cross-attention blocks as 18 plain products and 12 softmax kernels. [V, engine inspector]
- kornia 0.8.3 `CrossBlock` has a flash branch that computes one standard attention per direction;
  `CrossBlock.scale == dim_head**-0.5`. [V, installed package]

## Tooling

- NVIDIA Model Optimizer, `nvidia-modelopt` 0.48.0: INT8 and FP8 pairs placed by TensorRT's rules;
  entropy or max calibration; node and op exclusion; `--autotune` keeps pairs only where the engine
  is at least 1.02x faster; needs torch and pins `onnx~=1.21` on Linux. Not installed or run.
  [V, PyPI and [guide](https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html)]
- ONNX Runtime `quantize_static` (QDQ format): placement not TensorRT-aware. [U] Fallback only.
- `pytorch-quantization`: last release 2023-11-03, superseded. [V, PyPI]

## Methods

- Calibration: per-channel max for weights; per-tensor max, entropy or 99.99 to 99.999 percentile
  for activations, no single winner; partial quantization from a one-layer-at-a-time sensitivity
  sweep; then quantization-aware training. Wu et al., "Integer Quantization for Deep Learning
  Inference", 2020, <https://arxiv.org/abs/2004.09602> [V]
- Calibration set of 500 to 1000 unlabeled images. Nagel et al., "A White Paper on Neural Network
  Quantization", 2021, <https://arxiv.org/abs/2106.08295> [V]
- Transformers: activations, not weights, are the problem; the residual sum after the feed-forward
  block is the worst tensor. Bondarenko et al., 2021, <https://arxiv.org/abs/2109.12948> [V]
- Score and location heads of a keypoint network need float; the descriptor head tolerates INT8.
  Kanakis et al., "ZippyPoint", CVPRW 2023, <https://arxiv.org/abs/2203.03610> [V]

## Published results

- FP8 SuperPoint + LightGlue with Model Optimizer, RTX 4080 Laptop, TensorRT 10.9: up to 5.97x
  over FP32 at K = 3840; no FP16 comparison, no AUC; assignment scores fell and the match
  threshold removed most matches on a hard pair; calibrated on 2 pairs. [V,
  [Sim 2026](https://fabio-sim.github.io/blog/fp8-quantized-lightglue-tensorrt-nvidia-model-optimizer/)]
- INT8 against FP16 at batch 1: RT-DETR on Jetson Thor 5.79 against 5.73 ms; ViT-S on Orin Nano
  35.4 against 36.7 ms, and a 2.7x regression with Model Optimizer on the full model. [V, NVIDIA
  forum threads 353308 and 357835]
- None found: INT8 or FP8 accuracy for XFeat, LighterGlue, LightGlue, SuperGlue or LoFTR.

## Orin NX

- JetPack 7.2 ships TensorRT 10.16.2; TensorRT 11.4.1 names AGX Orin and Thor only. [V,
  [JetPack 7.2](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2),
  [Jetson migration](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/api/migration/tensorrt-10x-to-11x-jetson.html)]
- Explicit pairs load on TensorRT 10 as well. [V, Model Optimizer guide] Opset 20 with float16
  scales on TensorRT 10.16 is untested.

## Other routes, not in the plan

- LighterGlue's weights carry an assignment head per layer, so a graph cut to fewer than 6 layers
  needs no retraining. [V, weight keys] Accuracy cost unmeasured.
- LighterGlue training code is unpublished; glue-factory has no LighterGlue config. [V]
- Halving XFeat's last three blocks cost 5.2 AUC@5 in the XFeat paper (Table 5). [V]
- INT4 is weight-only; 2:4 sparsity applies to constant weights only: neither reaches the
  attention products. [V, TensorRT docs]
