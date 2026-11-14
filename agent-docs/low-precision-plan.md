# Low-precision speedup plan

Goal (owner, 2026-11-14): "squeeze all the juice out this optimisation with some more lee-way for
perf degradation as a tradeoff"; "int8 as a loose guide, if there are better suited or newer
approaches then consider them too"; "just needs to run on my 4070 or orin nx".

Sources and evidence: `research-low-precision.md`. Results: `low-precision-results.md`.

## Decisions

- Accuracy budget: MegaDepth-1500 AUC@5 drop of at most 0.01 against FP16 (0.5602 at K = 2048,
  1280x960 letterbox), with matches per pair within 10 %.
- The gain is shown on the RTX 4070 Ti. Nothing is built or measured on the Orin NX in this line
  of work (it runs TensorRT 10.16 and needs its own build path).
- A precision change is kept only if it is at least 10 % faster per pair, end to end at batch 1 or
  in pairs per second in batch mode, than the best FP16 graph.
- `nvidia-modelopt` (NVIDIA's tool that places quantize/dequantize node pairs) is not added until a
  timing probe passes. It needs its own environment: it pins `onnx~=1.21` and the export uses 1.23.2.
- Calibration data: a few hundred MegaDepth images from scenes outside the 1500 test pairs.
- Out: fewer LighterGlue layers, smaller K, retraining of any kind, the Orin NX's DLA cores.

## Stages

| Stage | What | Stop or accept rule |
| ----- | ---- | ------------------- |
| Per-layer timing | FP16 engines at K = 2048 and 4096, batch 1 and 2, timed layer by layer | Cross-attention a small share of matcher time: the fused cross-attention stage is dropped |
| Fused cross-attention | Cross-attention exported as one standard attention per direction (kornia `CrossBlock` flash branch), so TensorRT fuses it as it does self-attention | 24 fused attention kernels, contract test passes, AUC unchanged; becomes the FP16 baseline |
| FP8 attention timing probe | Quantize/dequantize pairs to FP8 inserted by hand on the fused attention inputs, throwaway scales; assignment head stays FP32 | Under 10 % faster per pair than fused FP16: FP8 stops |
| INT8 backbone timing probe | Quantize/dequantize pairs on the XFeat backbone convolutions, throwaway scales; score head, softmax, NMS, top-K and descriptor sampling stay float | Under 10 % faster per pair: INT8 stops |
| Calibration and accuracy | For each probe that passes: real calibration (entropy against max), AUC and matches per pair | Inside the accuracy budget |
| Accuracy recovery | Only where a stage pays but misses the budget: one-layer-at-a-time sensitivity sweep, worst layers left in float (Wu et al. 2020) | Inside the budget, still 10 % faster |

A timing probe measures the best case: quantize/dequantize placement decides kernels and fusion,
scale values do not, so throwaway scales give true timings; layers later returned to float shrink
the gain.

## Scope of failed tests

A method that misses its rule is recorded as "not yet shown to work" with its conditions: device,
TensorRT version, K, resolution, batch size and where the quantize/dequantize pairs sat.
