# Low-precision speedup results

Plan: `low-precision-plan.md`. Sources: `research-low-precision.md`.

Scope of every number: RTX 4070 Ti, TensorRT 11.4.0.106, WSL2, FP16 chunked exports at 1280x960,
batch 1 unless stated, random inputs for timing. Single machine; dev-only until rechecked on
native Linux.

## State on 2026-11-14

| Stage | State |
| ----- | ----- |
| Per-layer timing | Done |
| Fused cross-attention | Done; export change uncommitted |
| FP8 attention timing probe | Done: no gain at K = 2048; at K = 4096 8 % per pair at batch 1, 8 to 13 % in batch mode. Stopped by the owner on 2026-11-14 |
| INT8 backbone timing probe | Done: not yet shown to help; stopped |
| INT8 and FP8 on the matcher's linear layers | Done: not yet shown to help; stopped |
| Calibration and accuracy, accuracy recovery | Not run: no probe was taken forward |

Session scratch (cleared by a reboot):
`/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/3c74d365-5871-4776-9b71-4e85d332030d/scratchpad/`
holds `layer-profile/profile_layers.py` (per-layer timing harness), `models-fused/` (FP16 graphs
with the fused cross-attention, regenerable with `xfeat-export`), `auc/` (manifest, predictions,
scores, `run_auc.sh`), `qdq-probe/` (FP8 and INT8 probe scripts, graphs and bench output) and
`batch-fp8/` (K = 4096 ladders and `bench_pairs.py` output).

## Per-layer timing of the FP16 matcher before the change

Per-layer p50 over 50 runs through `IProfiler`, no CUDA graph. Shares are of the per-layer sum,
which sits within about 20 % of the wall time in either direction.

| Group | K = 2048 | K = 4096 | K = 2048, batch 2 |
| ----- | -------- | -------- | ----------------- |
| Fused self-attention (12 kernels) | 15.1 % | 15.3 % | 10.9 % |
| Unfused cross-attention products | 8.7 % | 11.4 % | 15.1 % |
| Unfused cross-attention softmax | 7.1 % | 16.7 % | 14.7 % |
| Linear layers | 22.9 % | 13.4 % | 10.9 % |
| Assignment head (FP32) | 12.1 % | 20.7 % | 15.2 % |
| Other (reshapes, transposes, LayerNorm, GELU) | 34.2 % | 22.5 % | 33.1 % |

- Cross-attention was 16 % of matcher time at K = 2048, 28 % at K = 4096 and 30 % at batch 2.
- XFeat at K = 2048 (1.29 ms per-layer sum): 32 convolution kernels 0.425 ms (33 %), the two
  top-K kernels 0.51 ms (40 %). An INT8 backbone can save at most about 0.2 ms per image.

## Fused cross-attention

`StaticLighterGlue._cross_block` computes one standard attention per direction (kornia
`CrossBlock` flash branch) in place of one shared similarity with two softmaxes. The function is
the same; the similarity product is computed twice.

- Engine: 24 `_gemm_mha_v2` kernels (12 before); the 18 plain cross-attention products and 12
  softmax kernels are gone.
- Assignment head after the change: 11.6 % of matcher time at K = 2048, 26.4 % at K = 4096.

Matcher p50 ms, `xfeat-trt-bench`, 200 runs, two repeats each:

| K | Path | Shared similarity | One attention per direction | Change |
| --- | --- | --- | --- | --- |
| 2048 | engine, CUDA graph | 1.81, 1.80 | 1.62, 1.67 | -9 % |
| 2048 | end to end, CUDA graph | 2.04, 2.08 | 1.89, 1.94 | -7 % |
| 2048 | engine, no graph | 4.11, 4.33 | 3.78, 3.80 | -10 % |
| 4096 | engine, CUDA graph | 8.02, 7.91 | 5.99, 5.93 | -25 % |
| 4096 | end to end, CUDA graph | 8.42, 8.35 | 6.40, 6.35 | -24 % |
| 4096 | engine, no graph | 8.61, 8.64 | 7.76, 7.69 | -10 % |

- Reference tests pass (30, with the contract test); `make check` passes.
- Batch-2 timing of the changed graph: not measured.

MegaDepth-1500, all 1500 pairs, `xfeat-trt-predict` at batch 1, letterbox, scored at 640 long
side; agreement is against the shared-similarity run with the same K:

| K | Cross-attention | AUC@5 | AUC@10 | AUC@20 | mAA | Matches per pair | Match Jaccard (mean, min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | Shared similarity | 0.5609 | 0.7075 | 0.8163 | 0.6949 | 631.1 | 1, 1 |
| 2048 | One attention per direction | 0.5601 | 0.7048 | 0.8136 | 0.6928 | 630.7 | 0.936, 0.560 |
| 4096 | Shared similarity | 0.5847 | 0.7260 | 0.8275 | 0.7127 | 1363.0 | 1, 1 |
| 4096 | One attention per direction | 0.5847 | 0.7274 | 0.8287 | 0.7136 | 1362.4 | 0.927, 0.571 |

- The largest AUC change is -0.0027 (AUC@10, K = 2048) and +0.0014 (K = 4096): the size of the
  FP32-to-FP16 change (0.0024) and a few times poselib's run-to-run spread (about 0.0005). Keypoints
  are identical; matches differ as they do between FP16 and FP32 (Jaccard 0.92 there). No
  bootstrap interval computed.

## Timing probes

Eight probe graphs with throwaway scales, timing only, no calibration and no accuracy measured
(session scratch, `qdq-probe/models/`). Opset 20 unchanged; scales are float16 scalars; FP8 is
expressed by a `FLOAT8E4M3FN` zero-point tensor of 0.

| Probe | Quantize/dequantize placement | Engine inspector |
| ----- | ----------------------------- | ---------------- |
| FP8 attention (matcher) | Query, transposed key, value, softmax output and second product's output of all 24 attentions | 24 fused attention kernels with FP8 inputs; no reformat kernels; 12 Half-to-FP8 casts |
| FP8 attention and linear layers | As above plus 110 linear layers, per-tensor weights | 24 fused FP8 attention kernels; 50 of 53 linear kernels FP8 |
| INT8 linear layers, FP16 attention | 110 linear layers, per-channel weights | 24 fused FP16 attention kernels; 56 of 59 linear kernels INT8 |
| INT8 backbone (XFeat), all 27 convolutions | Convolution inputs, per-channel weights, both inputs of the 3 residual adds | 27 INT8 convolutions; 13 reformat kernels (5 in FP16) |
| INT8 backbone without the three head convolutions and the first convolution | As above on 23 convolutions | 22 INT8 convolutions; 15 reformat kernels |

- BatchNorm is already folded by the export.
- The FP8 attention placement follows the pattern figure of TensorRT's fused attention page, read
  second-hand; the engine confirms the fused FP8 kernel.

### Batch 1, `xfeat-trt-bench --graph --runs 200`, two alternating repeats, p50 ms

Baseline is the fused-cross-attention FP16 graph. Pair = 2 x XFeat end to end + matcher end to
end, transfers included.

| Probe | K | Model engine, baseline | Model engine, probe | Change | Pair end to end |
| ----- | --- | --- | --- | --- | --- |
| FP8 attention | 2048 | 1.624, 1.625 | 1.639, 1.646 | +1 % | -2 %, +1 % |
| FP8 attention | 4096 | 5.948, 5.959 | 5.078, 5.077 | -15 % | -8 %, -8 % |
| FP8 attention and linear layers | 2048 | 1.624, 1.625 | 1.699, 1.699 | +5 % | +1 %, +3 % |
| FP8 attention and linear layers | 4096 | 5.948, 5.959 | 5.019, 5.023 | -16 % | -8 %, -9 % |
| INT8 linear layers, FP16 attention | 2048 | 1.624, 1.625 | 2.117, 2.113 | +30 % | +7 %, +10 % |
| INT8 linear layers, FP16 attention | 4096 | 5.948, 5.959 | 6.666, 6.671 | +12 % | +7 %, +7 % |
| INT8 backbone, 27 convolutions | 2048 | 1.103, 0.983 | 1.959 (outlier), 1.038 | within spread | +2 %, +3 % |
| INT8 backbone, 23 convolutions | 2048 | 1.103, 0.983 | 1.037, 1.036 | within spread | +2 %, +4 % |

### Batch mode at K = 4096, `scripts/bench_pairs.py`, all 1500 MegaDepth pairs, four alternating repeats

FP16 ladder against the same ladder with the FP8-attention matcher; matcher at batch 1; inference
only, stored inputs. Pairs per second per repeat:

| Variant | FP16 | FP8 attention | Change of medians |
| ------- | ---- | ------------- | ----------------- |
| Per pair, CUDA graph | 85.0, 85.8, 85.6, 85.6 | 92.8, 92.6, 92.1, 91.5 | +8 % |
| Unique images, XFeat batch 1 | 116.2, 118.2, 118.1, 112.9 | 125.5, 128.9, 128.6, 124.7 | +8 % |
| Unique images, XFeat batch 4 | 125.3, 107.4, 107.3, 124.2 | 122.5, 141.2, 141.4, 141.3 | +13 % between the fast runs (125 against 141) |

- The matcher pass over 1500 pairs takes 10.4 to 10.5 s in FP16 and 9.0 to 9.1 s in FP8 (-13 %)
  in the fast runs. At XFeat batch 4 the FP16 matcher pass took 12.4 s in two of four repeats and
  the FP8 pass 10.7 s in one; the cause is not investigated, and the medians (107 to 116 against
  141) overstate the gain.

### Conclusions and their scope

Conditions of all of these: RTX 4070 Ti, TensorRT 11.4.0.106, 1280x960, throwaway scales (the best
case: layers returned to float for accuracy shrink any gain). Nothing was run on an Orin NX.

- INT8 on the XFeat backbone: not yet shown to help at K = 2048, batch 1, with quantize/dequantize
  pairs on convolution inputs, per-channel weights and both inputs of the residual adds (13 to 15
  reformat kernels against 5). The ceiling is low whatever the placement: all convolutions take
  0.425 ms per image, so a 1.5 to 2x INT8 speedup saves 5 to 8 % of a batch-1 pair. Stopped.
- INT8 on the matcher's 110 linear layers with attention in FP16: 12 to 30 % slower at K = 2048
  and 4096, batch 1. Stopped.
- FP8 on the linear layers: not yet shown to add anything over FP8 attention alone (+4 % at
  K = 2048, -1 % at K = 4096). Stopped.
- FP8 attention: no gain at K = 2048. At K = 4096 the matcher is 13 to 15 % faster; the pair is
  8 % faster at batch 1 and 8 to 13 % faster in batch mode, on both sides of the plan's 10 % rule.
  Not calibrated; accuracy unknown. Stopped by the owner: the gain is small, at K = 4096 only and
  absent on the Orin NX.
