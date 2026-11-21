# Status

As of 2026-11-21. Plan and stage order: `implementation-plan.md`.

## Done

| Line of work | Result | Doc |
| ------------ | ------ | --- |
| Reference pipeline | README MegaDepth-1500 AUC reproduced; pruning delta measured | `reference-results.md` |
| Export | Static XFeat (in-graph NMS, TopK, chunked TopK) and padded LighterGlue at opset 20; FP16 export | `export-results.md` |
| ORT CPU backend | Contract test passes; CPU latency table | `ort-backend-results.md` |
| TensorRT backend | FP32 and FP16 engines, engine cache, CUDA latency tables; FP16 within 0.0024 AUC of FP32 | `tensorrt-backend-results.md` |
| Pair loader and scoring | `xfeat-pairs` manifest and prediction directory; `xfeat-score` on MegaDepth-1500 and own pairs | `tensorrt-backend-results.md` |
| Batch processing (TensorRT) | Batch ladder export, per-machine calibration, `xfeat-trt-predict-batch`: 2.9x pairs per second over the real-time path, no AUC drop | `batch-mode-plan.md`, `batch-throughput-results.md` |
| Precision below FP16 (TensorRT) | Fused cross-attention export: matcher 25 % faster at K = 4096, AUC within 0.003. INT8 and FP8 timing probes: no gain worth calibrating on the 4070 Ti; closed | `low-precision-plan.md`, `low-precision-results.md` |

## In progress

- Core ML backend (Apple silicon, real-time): `xfeat-export --target coreml` and `--provider coreml`
  on `xfeat-ort-predict` / `xfeat-ort-bench`; 15 ms XFeat, 32 ms LighterGlue at K = 2048 FP16 on an
  M2, AUC on par with FP32 CPU. Open: Core ML cache invalidation. `coreml-backend-results.md`.
- MLX backend spike (Apple silicon): running in a separate worktree, not merged.
- `test_keypoints_agree_up_to_tie_swaps[ort-k4096-chunked]` fails on macOS (tie gap 1.95e-5 against
  a 1e-5 bound), also on the commit before the Core ML work.
- ORT CPU runs the TensorRT-target graph; ORT CPU on the Core ML-target graph is unmeasured.

## Open

- `trtexec` and `onnxruntime_perf_test` cross-checks of headline latencies.
- Bootstrap confidence intervals on AUC differences.
- README rows still measured on the matcher export before the fused cross-attention change: ORT
  and TensorRT FP32 accuracy, batch throughput at K = 2048. The README marks them.
- Behaviour without a system CUDA toolkit; timings rerun on idle native Linux (WSL2 numbers are
  dev-only).

## Next stages

Pi 5, ablations (precision, resolution, K), Jetson Orin NX.

## Research

`research-method.md`, `research-export.md`, `research-runtimes.md`, `research-prior-art.md`,
`research-measurement.md`, `research-low-precision.md`.
