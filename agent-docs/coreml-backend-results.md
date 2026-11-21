# Core ML backend results (Apple silicon, real-time path)

Apple M2, 16 GB, macOS (Darwin 27.2), ONNX Runtime 1.31.0, torch 2.14.1 for export, 2026-11-21.
Batch 1, 1280x960 only; the batch-processing path is out of scope. Scratch scripts were not
committed; the export rewrites below are the part to bring into `xfeat-export`.

## Decision

Built: `xfeat-export --target coreml` (`make export` on Apple silicon, through
`--target native`), and `--provider coreml` with `--xfeat-units`, `--lighterglue-units` and
`--cache-dir` on `xfeat-ort-predict` and `xfeat-ort-bench`. Two export targets, so the TensorRT
graphs are unchanged. Repo build on the M2: XFeat 15.1 ms, LighterGlue 31.9 ms p50 at K 2048
FP16; 20-pair MegaDepth slice AUC@5 0.6268. The contract test passes on the Core ML provider.

The Mac real-time backend is `backends/ort` with ONNX Runtime's CoreML execution provider (EP),
fed by a Mac export variant of the same reference modules:

- EP options: `ModelFormat=MLProgram`, `RequireStaticInputShapes=1`, `ModelCacheDirectory` as the
  compiled-model cache (the counterpart of the TensorRT engine cache).
- XFeat on `MLComputeUnits=ALL` (GPU and Neural Engine), LighterGlue on `CPUAndGPU`.
- FP16 export, as on TensorRT.

Rejected, with the evidence below: native Core ML through coremltools (no faster, Python 3.13 only,
XFeat does not convert), PyTorch MPS (slower, and PyTorch is barred from backends), the Neural
Engine for LighterGlue (slower, changes matches), MLX (below).

MLX, both models rewritten by hand (about 1500 lines), discarded unmerged on 2026-11-21. K 2048 FP16
on the M2 at load average 2.2 to 2.5, p50 of 100 calls over 3 rounds alternating with Core ML: MLX
XFeat 16.4 ms, LighterGlue 28.5 ms, 61.3 ms per pair; Core ML 15.4, 32.0, 62.8 ms. MegaDepth-1500
AUC@5 0.5608 (FP16 K 2048), match Jaccard 0.954 against CPU FP32. The 1.5 ms lead needed an
unpublished two-stage top-k (per-chunk top 64, bound check, full top-k fallback); with the plain
top-k (+2.2 ms per image) MLX is slower than Core ML.

## Why the current export is slow on the EP

The EP hands Core ML only the ops it supports; each unsupported node splits the graph and costs a
CPU round trip. Current FP16 export: XFeat 9 Core ML partitions, LighterGlue 16.

| Graph | Rejected ops |
| --- | --- |
| XFeat | `SpaceToDepth` (pixel unshuffle), `ReduceL2` and `Expand` (L2 normalise), `TopK`, cubic `GridSample`, `Equal`/`Greater`/`Mod`/`Floor`/`GatherElements` in detection |
| LighterGlue | 40 rank-5 `Gather` (q/k/v split), 24 `Neg` (rotary half-rotation), `Unsqueeze` past 5-D and `Expand` (positional encoding `repeat_interleave`), `LogSoftmax` and `Log` (assignment) |

## Mac export rewrites

Each rewrite is exact in exact arithmetic. In PyTorch, at FP32 and FP16, it reproduces the repo
modules: keypoints identical, descriptors within 1.5e-7, matches identical. That check used one
random image matched against itself; real-pair agreement is in the MegaDepth table.

- XFeat `_unfold2d(ws=8)`: one-hot 64-channel stride-8 convolution.
- L2 normalise: `x / sqrt(sum(x*x)).clamp_min(1e-12)`.
- LighterGlue `Wqkv`: three linears from weight rows `[0::3]`, `[1::3]`, `[2::3]`.
- Rotary `rotate_half(x)`: `x @ R`, R the fixed signed permutation with `out[2i] = -x[2i+1]`,
  `out[2i+1] = x[2i]`.
- Positional encoding: cos and sin kept as two 4-D tensors; `repeat_interleave` as stack plus
  reshape.
- Assignment in probability space: `P = softmax_row(S) * softmax_col(S) * sigmoid(m0) *
  sigmoid(m1)^T`, the exponential of the log-space sum; argmax unchanged, confidence = `max P`.
- `TopK`, cubic `GridSample` and the mutual check stay on the CPU EP (under 2 ms together).

Partitions after the rewrites: XFeat 6, LighterGlue 3. One Core ML partition carries all of
LighterGlue.

## Latency (p50 ms, `session.run` with copies, 30 to 50 runs after 5 warm-up, random inputs)

| Graph | ORT CPU | EP, current export | EP, Mac export |
| --- | --- | --- | --- |
| XFeat K 2048 FP16 | 55 | 18.7 (ALL) | 15.0 (ALL); 28 (CPUAndGPU); 33 (CPUAndNeuralEngine) |
| LighterGlue K 2048 FP16 | 283 | 78 (CPUAndGPU) | 32 (CPUAndGPU); 50 (ALL) |
| XFeat K 4096 FP16 | 57 | not run | 16.4 (ALL) |
| LighterGlue K 4096 FP16 | 1039 | 264 | 102 (CPUAndGPU) |
| XFeat K 2048 FP32 | 62 | not run | 29.6 (ALL); 30.4 (CPUAndGPU) |
| LighterGlue K 2048 FP32 | 223 | 98 | 46 (CPUAndGPU) |

- One pair with both images new costs 2 x 15 + 32 = 62 ms at K 2048 and 2 x 16.4 + 102 = 135 ms at
  K 4096. TensorRT FP16 on the RTX 4070 Ti: 1.3 + 4.3 ms and 1.5 + 8.4 ms per call
  (`tensorrt-backend-results.md`). The gap is the hardware.
- LighterGlue at K 2048 needs about 40 GFLOP (6 layers, 4 attentions each, d = 96). 32 ms is about
  3x the M2 GPU's FP16 arithmetic floor.
- EP options `SpecializationStrategy=FastPrediction` and `AllowLowPrecisionAccumulationOnGPU`:
  within 0.6 ms of the defaults.
- Model load: first compile 0.7 s (XFeat) and 2.7 s (LighterGlue K 2048 and 4096); from the cache
  0.05 to 0.08 s. LighterGlue with Neural Engine units compiles for 10 to 24 s.
- XFeat at Mac-export FP16 on ALL changes 0.3 % of keypoints against ORT CPU on noise input;
  CPUAndGPU changes none.

## Alternatives measured

| Option | LighterGlue K 2048 FP16 | Notes |
| --- | --- | --- |
| Native Core ML, coremltools 9.0, explicit attention | 30.8 (CPU_AND_GPU), 54 (ALL) | Needs float `valid` inputs and no in-place `&=`; coremltools has tested torch only up to 2.7, has no cp314 wheel, and its last release was 2025-11 |
| Native Core ML, `scaled_dot_product_attention` (24 fused ops) | 31.2 (CPU_AND_GPU), 40 (ALL) | No gain from the fused op |
| PyTorch MPS, reference pipeline, 50 MegaDepth pairs | 85 match, 121 extract per pair | Dynamic reference, not the static modules |

## MegaDepth-1500 (all 1500 pairs, 1280x960 letterbox, scored at 640 long side)

| Run | AUC@5 | AUC@10 | AUC@20 | mAA | Match Jaccard vs CPU FP32 (mean / min) | Keypoint recall (mean) |
| --- | --- | --- | --- | --- | --- | --- |
| ORT CPU FP32, current export, K 2048 (this Mac) | 0.5587 | 0.7057 | 0.8140 | 0.6928 | reference | reference |
| EP Mac FP16 K 2048, XFeat ALL | 0.5595 | 0.7066 | 0.8144 | 0.6935 | 0.938 / 0.40 | 0.9936 |
| EP Mac FP16 K 2048, XFeat CPUAndGPU | 0.5624 | 0.7095 | 0.8172 | 0.6964 | 0.949 / 0.66 | 0.9957 |
| EP Mac FP16 K 4096, XFeat ALL | 0.5862 | 0.7276 | 0.8279 | 0.7139 | not scored | not scored |
| TensorRT FP16 K 2048 / 4096 (RTX 4070 Ti) | 0.5602 / 0.5828 | 0.7069 / 0.7249 | 0.8157 / 0.8273 | 0.6943 / 0.7117 | 0.919 (vs its FP32) | 0.988 |

- The Core ML FP16 runs score at or above the FP32 CPU reference on the same Mac. Match agreement
  is higher than TensorRT FP16's.
- XFeat ALL vs CPUAndGPU: 0.0029 AUC@5 for 13 ms per image. Bootstrap intervals are not computed
  (open item in `STATUS.md`), so whether the difference is real is not settled; ALL is chosen for
  latency.
- The FP32 CPU reference here (0.5587) is 0.004 below TensorRT FP32 on WSL2 (0.5626) for the same
  export. Not investigated; poselib's run-to-run spread is about 0.0005.
- Per-pair times in these score files were taken with scoring jobs running alongside them; they
  are not latency results.

## Open

- How `ModelCacheDirectory` detects a changed model or OS version is unverified. The TensorRT
  cache keys on device, driver, build options and ONNX hashes.
- Sustained (thermal) latency and the `ALL` vs `CPUAndGPU` AUC difference with bootstrap intervals.
