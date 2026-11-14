# Batch throughput results

Question: does running XFeat and LighterGlue on batches (static batch-B exports) raise
throughput, and is per-call time flat up to some batch size so that a per-machine sweet spot
exists?

Scope of every result here: RTX 4070 Ti and Ryzen 7 5800X3D (16 logical CPUs), WSL2, idle
(load average below 1), GPU clocks not pinned, TensorRT 11.4 FP16 chunked exports, ORT 1.31.0 FP32,
random inputs (noise images, random unit descriptors, all keypoints valid). Every graph is static,
so compute does not depend on the data. Single host thread. MNN out of scope.

## Method

- Exports: `xfeat-export --batch B` for B in 1, 2, 4, 8, 16, 32 (TensorRT) and 1, 2, 4, 8 (ORT),
  at 1280x960 with K = 2048 and 4096, and at 640x480 with K = 2048 (TensorRT only).
- Harnesses (session scratch, not committed): a per-model sweep timing the engine, the CUDA graph,
  and upload plus execute plus download without a graph; and a throughput step of N streams, each
  doing async upload, CUDA graph launch and async download, then a sync of all streams. p50 of 50
  runs after 10 warm-up (TensorRT), 10 after 2 (ORT).
- Batch-B outputs compared slot by slot with the batch-1 engine on the same input.

## End-to-end time per item with CUDA graph, TensorRT FP16, one stream (ms)

Upload, graph launch and download included; per item = step time / B.

| B | XFeat 640x480 | XFeat 1280x960 | LighterGlue K 2048 (per pair) | LighterGlue K 4096 (per pair) |
| --- | --- | --- | --- | --- |
| 1 | 0.955 | 1.91 | 2.03 | 8.66 |
| 2 | 0.606 | 1.39 | 2.19 | 9.51 |
| 4 | 0.452 | 1.40 | 2.55 | 9.47 |
| 8 | 0.373 | 1.42 | 2.65 | 9.67 |
| 16 | 0.366 | 1.55 | 2.87 | 10.29 |
| 32 | 0.425 | 1.52 | 3.15 | 10.29 |

- XFeat 640x480 shows the flat region: a batch of 2 costs 1.27x one image, a batch of 4 1.9x; the
  best throughput is at B = 8 to 16 (2.6x batch 1), and B = 32 is slower again.
- XFeat 1280x960 gains 1.37x at B = 2 to 8, then nothing: one 1.2 Mpx image nearly fills the GPU.
  Engine-only time (inputs resident) is 0.97 ms at B = 1 and 0.66 ms per image at B = 4; transfers
  and the host-side FP32-to-FP16 cast cost more than the model at this size.
- LighterGlue as exported is slower per pair at every B > 1 (K = 2048: graph time 1.78 ms for one
  pair, 4.01 ms for two). Batching only helped the eager path, whose launch overhead the CUDA graph
  already removes. A cache effect from the full K x K FP32 assignment matrices (16 MB per pair at
  K = 2048, 48 MB L2) is a candidate cause; it is unverified (no per-layer profile).

## Concurrent streams at batch B (TensorRT, CUDA graph, end to end)

Two or four streams fed from one host thread gave no gain over one stream for any model or B
(differences within about 10 %, in both directions). Not yet shown to help on this machine; the
host loop issues the pinned-memory copies serially, so overlap from several host threads is
untested.

## ORT CPU, FP32, default threading (ms per item)

| B | XFeat K 2048 | XFeat K 4096 | LighterGlue K 2048 | LighterGlue K 4096 |
| --- | --- | --- | --- | --- |
| 1 | 46.9 | 43.5 | 128.7 | 572 |
| 2 | 47.5 | 44.9 | 150.2 | 630 |
| 4 | 46.0 | 45.8 | 165.7 | 661 |
| 8 | 47.4 | 48.2 | 178.7 | 693 |

ORT already uses every core inside each operator; batching gives XFeat nothing and makes
LighterGlue 15 to 40 % slower per pair. The CPU line stops here for this machine.

## Agreement with batch 1

- XFeat FP16 keypoint Jaccard against the batch-1 engine: 0.989 to 0.997 for B > 1 (TensorRT picks
  different tactics per batch size). Noise images tie many scores near the top-K cutoff, which
  inflates the difference; real-image agreement and pose accuracy are unmeasured.
- LighterGlue matches were identical in every slot on random descriptors, where almost every
  entry is "no match". On real MegaDepth features, LighterGlue B = 4 FP16 gives different matches
  from B = 1 (match Jaccard mean 0.943, min 0.619, 399 pairs); slots do not leak into each other.

## Batch-processing mode on MegaDepth-1500 (all 1500 pairs, 806 unique images)

FP16 chunked, K = 2048, 1280x960 letterbox, idle machine. Calibration chose XFeat B = 4 and
LighterGlue B = 1 (XFeat medians 478, 648, 666, 647, 623 images/s at B = 1 to 16).

### Inference only

Every variant reads the same stored model inputs, all preloaded into RAM before timing; only the
model calls are timed (upload, execute, download, host-side unpacking of outputs). Two repeats
each; the table gives their mean. Each row adds one change to the row above.

| Variant | XFeat | LighterGlue | Total | Pairs/s | Gain over previous row |
| --- | --- | --- | --- | --- | --- |
| Both images of every pair, batch 1, no CUDA graph (real-time path) | 6.83 s | 7.44 s | 14.27 s | 105 | |
| + CUDA graph | 6.30 s | 3.63 s | 9.92 s | 151 | 1.44x |
| + each unique image extracted once (806 calls, not 3000) | 1.77 s | 3.62 s | 5.39 s | 278 | 1.84x |
| + XFeat B = 2 | 1.44 s | 3.47 s | 4.91 s | 306 | 1.10x |
| XFeat B = 4 (calibrated) | 1.39 s | 3.62 s | 5.01 s | 299 | |
| XFeat B = 8 | 1.48 s | 3.69 s | 5.18 s | 290 | |
| XFeat B = 16 | 1.50 s | 3.89 s | 5.39 s | 279 | |

- Batching XFeat cuts its time 1.77 s to 1.39 to 1.44 s (about 1.25x on XFeat, 1.1x on the
  pair). The gain over the real-time path is 2.9x overall: unique images 1.84x, CUDA graph 1.44x,
  batching 1.1x.
- LighterGlue at batch 1 with a CUDA graph is 61 to 74 % of inference time in the batched
  variants; its run-to-run spread (3.45 to 4.02 s) is as large as the XFeat batching gain.
- An earlier comparison of `xfeat-trt-predict` (77.8 s wall, decoding every image) against
  `xfeat-trt-predict-batch` (11.5 s wall, stored inputs) measured preprocessing, not inference,
  and is withdrawn. Model-call times recorded inside `xfeat-trt-predict` (43.4 s) are also
  inflated against the table above (14.3 s), plausibly by the GPU idling between calls while the
  CPU decodes (unverified).

### Accuracy

| Run | AUC@5 | AUC@10 | AUC@20 | mAA |
| --- | --- | --- | --- | --- |
| `xfeat-trt-predict`, batch 1 | 0.5609 | 0.7058 | 0.8145 | 0.6937 |
| `xfeat-trt-predict-batch`, XFeat B = 4 | 0.5658 | 0.7073 | 0.8157 | 0.6963 |

- Keypoint recall against batch 1: mean 0.9985, min 0.9968. All of the difference comes from the
  B = 4 XFeat engine's FP16 tactics: the batched runner at B = 1 reproduces batch 1 exactly.
- AUC does not drop (all four metrics equal or higher). No bootstrap interval computed.

## Open

- Per-layer profile of LighterGlue at B = 1 against B = 2.
