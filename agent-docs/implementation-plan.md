# Implementation plan: xfeat-rt

Status: C0 to C3 done (`c2-results.md`, `c3-plan.md`, `c3-results.md`). Evidence
lives in `research-01-method.md` to `research-05-measurement.md` beside this file.

## Targets

- Keypoint budget K: at least 2048 in the low-latency modes, never below 4096 offline. On
  TensorRT, if 4096 turns out unreachable, the maximum K that builds is acceptable. Maximising
  K at a given latency is a primary goal, so the K sweep (U8) is a headline result, not an
  ablation, and every graph, engine and kernel choice is judged at K = 2048 and 4096.
- LighterGlue cost is the pressure point: attention is quadratic in K, and the CPU measurement
  grew about 4x per K doubling (73 ms at 1024, 275 ms at 2048, 1263 ms at 4096; research 01).
  LighterGlue is the dominant compute cost by an order of magnitude, so fusing it is the main
  optimisation target of the project: explicit multi-head attention in the graph and a fused
  attention kernel per backend, started in the first spike and not deferred.
- Latency modes: XFeat at batch 1; LighterGlue on one pair. Offline mode: maximum throughput.

## Design in one table

| Topic              | Choice                                                                              |
| ------------------ | ----------------------------------------------------------------------------------- |
| Export format      | One ONNX per model (XFeat, LighterGlue), opset 20, dynamo exporter, fixed shapes    |
| Default shape      | 1280x960, K and batch set at export (`--height --width --top-k --batch`)            |
| XFeat graph        | Network, NMS mask, `TopK(K)`, bicubic `GridSample`, L2 norm; fixed-size outputs     |
| LighterGlue graph  | Inputs padded to K with a valid mask; outputs `m0`, `mscores0`; pruning and early exit off |
| Runtime layout     | Shared core wheel in `runtime/`; one independent project per backend in `backends/<name>/` |
| Backends           | ORT (CPU, Arm, CoreML EP), native TensorRT (CUDA, Jetson); backends are never installed together |
| Reference "correct" | FP32 PyTorch on CPU, pinned kornia, pruning off, upstream quirks reproduced        |
| Accuracy harness   | GlueFactory pinned as a git dependency in `reference/`; runtime writes predictions to files |

## Package split

`reference/` (Python 3.13, torch):

| Package            | Holds                                                                         |
| ------------------ | ----------------------------------------------------------------------------- |
| `xfeat-reference`  | Pinned upstream XFeat and kornia LighterGlue wrapped as the reference pipeline |
| `xfeat-export`     | Static-shape rewrite, ONNX export, metadata, `xfeat-export` CLI                |
| `xfeat-evaluation` | GlueFactory-based MegaDepth-1500 and HPatches evaluation of prediction files   |

`runtime/` (no torch, shared core, built as a wheel):

| Package       | Holds                                                                               |
| ------------- | ----------------------------------------------------------------------------------- |
| `xfeat-core`  | Metadata read-back, result types, NumPy glue (valid-mask compaction, rescale, match pairs); depends on `numpy` and `pydantic` |
| `xfeat-pairs` | (C3) Pair manifest, image loading and preprocessing, prediction-directory writer and reader; `numpy`, `pydantic`, `opencv-python-headless` |
| `xfeat-bench` | Benchmark harness: explicit per-backend sync, `perf_counter_ns`, percentiles from raw samples, JSON results |

`backends/<name>/` (one independent uv project each, own `uv.lock`, `.venv` and Python pin):

| Project          | Runtime dependency                           | Devices                       |
| ---------------- | -------------------------------------------- | ----------------------------- |
| `backends/tensorrt` | `tensorrt-cu13` (libs pinned to the NVIDIA index for a hashed lock), `cuda-python` (`cuda.core`) | NVIDIA CUDA (first, WSL2 dev) |
| `backends/ort`   | `onnxruntime`                                | x86 CPU, Raspberry Pi 5, Apple (CoreML EP) |
| `backends/openvino` | `openvino`                                | CPU ablation                  |
| `backends/jetson` | system TensorRT of the latest JetPack| Orin NX, in an NGC/CUDA container |
| `backends/coreml` | `coremltools` (cp313 only)                  | Apple, only if the EP falls short |

- Backends are never installed together and share no lock; each depends on `xfeat-core` by
  path in development and by built wheel elsewhere. No `conflicts` entries, no extras.
- A backend project owns loading its model file, running it, and its own benchmark entry point
  using `xfeat-bench`. No protocol or ABC: shared behaviour is plain functions in `xfeat-core`.
- Metadata (H, W, K, batch, threshold, opset, weights SHA-256, git hash) sits in ONNX
  `metadata_props`. A backend reads the raw string map (ORT `custom_metadata_map`; TensorRT reads
  it from the ONNX file it builds from) and `xfeat-core` parses it into a typed value, so no
  backend needs the `onnx` package.
- Each backend pins Python separately: Core ML stays on cp313 without holding the others back.
  `reference/` stays on 3.13 for torch and export tooling.
- `AGENTS.md` layout table changes accordingly (`runtime/` is the core; `backends/` is added).

## Export

- XFeat graph outputs `kpts[B,K,2]`, `scores[B,K]`, `desc[B,K,64]`, `valid[B,K]`. Static
  rewrite: dense score map, `TopK(K)`, validity mask, replacing upstream's data-dependent
  `nonzero` and Python NMS loop. Precedent: optocycle/accelerated_features, fabio-sim top-K.
- LighterGlue graph takes `kpts`, `desc`, `valid`, `image_size`; padded both sides to K with
  masks, through log-assignment and mutual-NN threshold.
- Pruning and early exit stay off, as in every prior port. A masked static version cannot speed
  anything up, and no exported pruning exists anywhere (research 04).
- Disagreement with research 02, which put NMS and top-k outside the graph: outside forces a GPU
  to host copy of the 1.2 Mpx heatmap on CUDA. The C1 spike measured in-graph NMS at about 2 ms on
  ORT CPU (the 55 ms attributed to max-pool was dense `GridSample`, replaced by `Resize`).
  Fallback if the spike fails on a backend: backbone-only graph with NumPy post-processing
  (precedent: BCDL port, kornia/vision-rt).
- Risks in the in-graph route: dense bilinear sampling costs more than sampling candidates only;
  tie order differs from argsort, so keypoints are compared as sets.
- TensorRT `TopK` requires K <= 3840 (measured; K = 3841 and 4096 fail), which rules out the 4096
  offline floor. Chunked top-K (as
  fabio-sim does, 39.5 to 4.1 ms in their report) is therefore required on TensorRT, not an
  option; the spike checks it builds and matches at K = 2048 and 4096, and records the largest K
  that builds if 4096 does not.
- TensorRT: engine built from the ONNX on the target and cached (engines are tied to TensorRT
  version and GPU). That is a build step, not a second export.
- Core ML: no `mlpackage` in the first wave. Core ML `grid_sampler` has no bicubic, `coremltools`
  has no cp314 wheel. ORT CoreML EP is used; a `coremltools` route is added only if a Mac
  measurement shows a large gap.

## Backend per device

Hardware: WSL2 box with an NVIDIA GPU (development and CUDA testing), M2 Mac, Raspberry Pi 5
16 GB, Jetson Orin NX 16 GB. Development stays on the WSL2 box until CUDA is built; the other
devices follow.

| Order | Device                  | Backend project        | Evidence                                              |
| ----- | ----------------------- | ---------------------- | ----------------------------------------------------- |
| 1     | WSL2 x86 CPU            | `backends/ort`         | Contract-test baseline on the dev box; cp313/cp314 wheels (research 03) |
| 2     | WSL2 NVIDIA GPU         | `backends/tensorrt`    | Research 03/04; ORT CUDA EP slower than torch at high K in issues |
| 3     | Raspberry Pi 5          | `backends/ort`         | Same code as x86; aarch64 wheels                      |
| 3     | M2 Mac                  | `backends/ort` (CoreML EP) | cp314-ready; ANE coverage unmeasured              |
| 4     | Orin NX                 | `backends/jetson`      | Latest JetPack (7.2.1 at time of research): no `l4t-jetpack` container, NGC/CUDA base + Container Toolkit. Orin NX support on that release not yet verified |
| Later | x86/Arm CPU ablation    | `backends/openvino`    | OpenVINO had an NMS-loop accuracy bug in 2024.6       |
| Later | Mac, if EP falls short  | `backends/coreml`      | Needs cp313 and no bicubic `grid_sampler`             |

WSL2 numbers are dev-only for latency claims; CUDA results that matter are rechecked off WSL2
when a native Linux box is available.

## Contract between workspaces

- Artifact: the two ONNX files plus their metadata; never committed.
- Cross-workspace test: export with `reference`, load with a backend project, run on fixed image
  pairs, compare to the reference outputs.
- Reference definition of correct: FP32 PyTorch on CPU, kornia pinned to one version,
  `width_confidence=-1`, `depth_confidence=-1`, unpadded kornia forward per pair (avoids the
  0.8.3 padded-path bug at `lightglue.py:739`), `xfeat.pt` pinned by SHA-256, deterministic
  algorithms on. CUDA's FP16 SDPA (kornia runs attention in FP16 on CUDA) serves as the FP16
  oracle, not the reference.
- Upstream quirks (last row and column keypoints dropped, skewed sampling grid): the reference
  pipeline has a named `upstream_exact` config that reproduces them, used to measure the AUC
  effect, and a default config with the fix if it is a simple change in the static rewrite. The
  exported graph follows the default config; the contract test compares against the same one.
- Tolerances (research 05 section 3.4), stored as versioned data:
  - FP32 noise floor measured by perturbing the reference (threads, CPU vs GPU, TF32).
  - FP16 bound: export may be no worse than PyTorch FP16 vs FP32.
  - Keypoints compared as sets within a radius; descriptors by cosine; matches by Jaccard.
  - Acceptance on task-metric delta (AUC, MHA) with bootstrap confidence intervals.

## Benchmark harness and accuracy evaluation

- Harness: custom, thin (no library times CUDA, ORT, TensorRT, Core ML and MPS uniformly).
  Warm-up, raw samples saved as JSON, p50/p95/p99; XFeat at batch 1 and LighterGlue on one pair
  timed separately; offline mode with batch-B engines. Cross-check headline numbers against
  `trtexec` and `onnxruntime_perf_test`. Record clocks and power mode (Jetson `nvpmodel`,
  Pi throttle state, CPU governor). Results land in a gitignored `uncommitted/` directory with
  timestamped names.
- Accuracy: backends write predictions (keypoints, matches) to files; `xfeat-evaluation` in
  `reference/` scores them with GlueFactory (Apache-2.0, pinned commit, not copied). Torch stays
  out of the runtime and the backends.
- Custom pair datasets: `xfeat-evaluation` loads image pairs from a user-supplied source (a
  manifest of image paths with optional ground truth such as relative pose or homography), applies
  user-defined preprocessing, and feeds the same scoring path as MegaDepth-1500. Pairs without
  ground truth still run for latency and for reference-vs-backend agreement.
- Gate 0: reproduce the README LighterGlue MegaDepth-1500 AUC (0.444/0.610/0.746 at 640 px, 1300
  keypoints) before trusting any other number.
- Datasets: MegaDepth-1500 primary, HPatches secondary, the user's own pairs through the loader
  above; public sets are fetched, never redistributed (MegaDepth licence unverified). ScanNet is
  out.
- Ablations (checkpoint 6): precision (FP32/FP16, INT8 only if requested), resolution, K, pruning
  on/off, runtime, ORT vs OpenVINO on CPU.

## Build order

| #  | Checkpoint                                  | Demonstrable result                                                   |
| -- | ------------------------------------------- | --------------------------------------------------------------------- |
| C0 | Reference pipeline                          | Gate 0 AUC reproduced; `upstream_exact` vs default config measured    |
| C1 | Spike (below)                               | Go/no-go on in-graph post-processing, pruning delta, TensorRT install |
| C2 | Export, `xfeat-core`, `backends/ort`, contract test, harness (WSL2 CPU) | Cross-workspace test passes; first CPU latency table |
| C3 | `backends/tensorrt` (WSL2 GPU), custom pair loader | Done: CUDA latency tables (FP32, FP16), engine cache, `xfeat-score` on MegaDepth-1500 and own pairs |
| C4 | Offline throughput                          | Separate batch-B engines (a batch-2 profile slowed batch-1 by 13-16% in vision-rt) |
| C5 | Pi 5 and M2 Mac                             | Arm and CoreML EP tables                                              |
| C6 | Ablations                                   | Accuracy vs latency across precision, resolution, K                   |
| C7 | `backends/jetson` on Orin NX                | Container run on the latest JetPack                                        |

Out of scope for v1: end-to-end fine-tuning on the user's domain data (curriculum training,
distilling high-cost models such as MASt3R to reduce labelling). The custom pair loader and the
prediction-file format are the hooks it will build on.

## Decisions

| #  | Decision                                          | Outcome                                                         |
| -- | ------------------------------------------------- | --------------------------------------------------------------- |
| F1 | LighterGlue pruning                               | Off by default; C0/C1 measure the delta on MegaDepth-1500 and the user decides if it matters |
| F2 | Upstream edge row/column quirk                    | Fix if simple; `upstream_exact` config kept for measurement     |
| F3 | Resize to export resolution                       | Caller's job                                                    |
| F4 | Benchmark hardware                                | WSL2 CUDA first, then M2 Mac, Pi 5 16 GB, Orin NX 16 GB on the latest JetPack |
| F8 | Keypoint budget                                   | K >= 2048 for low latency, K >= 4096 offline; maximise K per latency budget |
| F5 | ScanNet-1500                                      | Out                                                             |
| F6 | Domain data                                       | Custom pair loader for eval; fine-tuning is post-v1             |
| F7 | Package layout                                    | Shared wheel in `runtime/`, independent project per backend     |

## New dependencies

- `reference/`: `torch`, `kornia` (pinned), `accelerated_features` at `e92685f` (git; may lack a
  `pyproject`, to be checked), `onnx`, `onnxscript`, `glue-factory` (git), `h5py`.
- `runtime/` (`xfeat-core`): `numpy`. Each backend project adds its own: `onnxruntime`,
  `tensorrt-cu13` + `cuda-python`, `openvino`; later `coremltools` (cp313).
- After each addition, check the affected `uv.lock` (runtime and every backend) for torch.

## Unknowns and the measurement that settles each

| #  | Unknown                                                                 | Measurement (C1 covers 1-3)                                       |
| -- | ----------------------------------------------------------------------- | ----------------------------------------------------------------- |
| U1 | Full static XFeat and padded LighterGlue graphs export, match reference at set level, and in-graph MaxPool NMS cost vs CNN at 1280x960 | Dynamo export at opset 20, ORT CPU run, set-level comparison, per-stage timing |
| U2 | Accuracy cost of pruning off                                            | MegaDepth-1500 AUC, pruning on vs off in the reference            |
| U3 | `uv lock` in `backends/tensorrt` (15 KB stub packages download libs); TensorRT parses both graphs incl. cubic `GridSample` in FP16 and the `TopK` cap | `uv lock`, TensorRT build of both ONNX files |
| U4 | ORT CoreML EP share of each graph on ANE/GPU                            | M2 Mac run with EP profiling                                         |
| U5 | Pi 5: ORT vs OpenVINO                                                   | Latency on Pi 5 16 GB with throttle state logged                        |
| U6 | Orin NX support on the latest JetPack at that time; container Python version and TensorRT bindings | Container smoke test on the Orin NX with the latest JetPack                       |
| U7 | FP16 and INT8 accuracy (robofans: XFeat FP16 -2 AUC@5, no speedup)      | FP16 measured in C3 (AUC within 0.0024 of FP32, LighterGlue 2.7x faster at K = 4096); INT8 and confidence intervals in C6 |
| U8 | Largest K per device inside the latency budget; LighterGlue attention cost at K = 2048 and 4096 (about 4x per doubling on CPU) | K sweep from C2 onward on each device, XFeat and LighterGlue timed separately |
| U9 | Chunked TopK on TensorRT at K = 4096: builds, matches the reference, cost | TensorRT build and set-level comparison in C1                     |

## Stray items from research

- `dl/lg115.patch` appeared in the scratchpad during the export research and was not created by
  that agent; unexamined.
