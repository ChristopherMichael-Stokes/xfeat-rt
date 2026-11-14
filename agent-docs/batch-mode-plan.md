# Batch processing mode: plan

Goal (user): "speeding up batch processing with xfeat & lighterglue", "some way to find the sweet
spot for whatever hardware we are deploying on dynamically"; "more interested in throughput than
latency"; "We still want the low latency, real time export to exist alongside, but doesn't have to
be the same thing". Chosen approach: one static engine per batch size, timed on the target machine
on first run, choice cached. MegaDepth-1500 and TensorRT first; MNN out.

## Baseline (current `xfeat-trt-predict`, FP16 chunked, K = 2048, 1280x960 letterbox)

400 MegaDepth-1500 pairs, warm engine cache: 30.6 s wall, 76 ms per pair. The engine benchmark
puts the models at about 10 ms per pair, so most of the time is elsewhere. py-spy, 50 Hz, native,
1792 samples:

| Part | Share of wall time |
| ---- | ------------------ |
| `prepare_image` (decode, resize, grey, letterbox) | 69.5 % |
| of which `resized.astype(np.float32).mean(axis=2)` | 46.1 % |
| of which JPEG decode (`cv2.imread`) | about 16 % |
| XFeat extract (incl. GPU sync waits) | 13.7 % |
| LighterGlue match (incl. GPU sync waits) | 12.7 % |
| writing predictions (`np.savez`) | 2.4 % |

- The predict path decodes and extracts both images of every pair. MegaDepth-1500 has 806 unique
  images in 1500 pairs: 3000 extractions where 806 would do (3.7x).
- The predict path runs without CUDA graphs.

Cost model for all 1500 pairs today: 1500 x 76 ms = 114 s. After the steps below (one image
decoded and extracted once, XFeat batched, LighterGlue at batch 1 with a CUDA graph): about
806 x 1.4 ms + 1500 x 2.0 ms = 4.1 s of GPU time; preprocessing is persisted once and
not timed. The batch-size choice itself is worth
about 0.4 s of that at 1280x960 on this GPU; it matters more at smaller inputs (2.6x per image at
640x480).

## Steps, in order of share

1. **Grey conversion.** Replace the float32 channel mean with an OpenCV call computing the same
   equal-weight mean. Outputs differ at float32 rounding (about 1e-7), not bit-identical: the
   user's call. Shared by both run modes.
2. **Batched pair runner** in `xfeat-pairs` (`predict_pairs_batched`, backend-independent,
   takes batch callables): extracts each unique image once in batches of B (final batch
   zero-padded, outputs dropped), then matches pairs at LighterGlue's B. Per-pair
   `extract_ms`/`match_ms` become the amortized share, stated in the docstring. Writes the same
   prediction directory, so `xfeat-score` works unchanged.
   - Preprocessing is not what is benchmarked (user decision): each unique image is prepared
     once and persisted as its float32 model input (`.npy`, about 4.9 MB at 1280x960, 4 GB for
     MegaDepth-1500) plus its resize geometry, in a gitignored `uncommitted/` directory keyed by
     image path, model size and resize mode. Timed runs read the persisted inputs.
   - Overlapping preprocessing with inference (thread pools, prefetch) is deployment code, not
     this repo (user decision).
3. **Batch ladder export.** `xfeat-export --batch 1 2 4 8 16 32` writes one file pair per B.
4. **Calibration** for TensorRT, `xfeat-trt-calibrate --models DIR`:
   - Times end-to-end throughput (upload, CUDA graph, download, as the predict path runs it) of
     each B in increasing order, repeated 3 to 5 times; stops after throughput falls for two steps
     in a row; a build or allocation failure at large B marks that B as not possible.
   - Picks the smallest B whose throughput is within the measured repeat spread of the best (a
     noise band, not a latency preference). The selection function lives in `xfeat-bench`.
   - XFeat and LighterGlue separately; LighterGlue sweeps 1, 2, 4 only (B = 1 won everywhere here).
   - Cache key: device name, compute capability, driver and TensorRT versions, build options,
     ONNX hashes. Reports engine build time per B, for first-run budgeting.
5. **`xfeat-trt-predict-batch`**: reads the calibration (runs it on a cache miss) and drives step
   2. `xfeat-trt-predict` (real time, batch 1) is unchanged apart from step 1.

ORT gets steps 1 and 2 through the same runner at B = 1; an ORT calibration CLI waits for a Pi 5 or
M2 measurement showing batching gains on CPU.

## Acceptance

- Pairs per second on 100, then all 1500 MegaDepth-1500 pairs: current path, batch 1 with CUDA
  graph, and the batched path, each step's gain reported on its own (grey conversion, unique
  images, batching).
- MegaDepth-1500 AUC of the batched run against the batch-1 run; a drop beyond the spread of
  repeated batch-1 scorings rejects the batched path.
- A manifest whose image count is not divisible by B gives the same predictions as batch 1, within
  the FP16 agreement already measured (keypoint Jaccard 0.99).
- `make check` passes.

## Not covered

Jetson (unified memory changes the transfer cost; the same calibration applies, no claim until
measured), several host threads feeding several CUDA streams, LighterGlue per-layer profile.
