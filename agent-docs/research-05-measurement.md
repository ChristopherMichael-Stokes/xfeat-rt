# 05 Measurement, accuracy evaluation, numerical contract

Legend: [V] = verified by fetching the cited page this session. [U] = not verified (from background
knowledge or not on the fetched page).

## 1. Benchmarking practice

### Protocol skeleton (recommended)

- Three scenarios, borrowed from MLPerf Inference vocabulary:
  - SingleStream = batch 1, next query only after the previous finishes; report percentile latency
    (MLPerf uses p90). Maps to XFeat extraction and LighterGlue one-pair.
  - Offline = all samples in one query; report throughput only. Maps to max-throughput mode.
  - MLPerf minimum duration is 600 s and 24,576 queries for p90. That is far too heavy for edge CI;
    adopt the definitions, not the counts. [V]
    <https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>
- Per run: warm-up (discard) -> N timed iterations -> report median, p90/p95/p99, min, max, std, N,
  plus full raw per-iteration times in JSON. NVIDIA's own advice: "use throughput and median latency
  as the numbers to compare"; trtexec also prints min/max/mean/p90/p95/p99. [V]
  <https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html>
- Time end-to-end stages separately: preprocess, H2D, model, postprocess (top-k, NMS, descriptor
  sampling), D2H. Report "model only" and "pipeline" numbers; trtexec excludes data transfers by
  default (`--includeDataTransfers` to add). [V] same page.
- Record metadata with every result: device, driver/runtime versions, power mode, clocks, thread
  count, precision, input res, top_k, git hash. (Jetson docs: "record the mode ID with every
  result", paraphrase of implication; [V] page below.)
- Warm-up/iteration counts are not standardised across vendors. Defaults to cite: trtexec warms
  up >=200 ms, then >=10 iterations and >=3 s [V]; PyTorch Timer.blocked_autorange runs >=0.2 s by
  default [V]. Suggested starting point (my judgement, not a standard): >=50 warm-up iterations
  and >=200 timed iterations or >=10 s, whichever longer; raise until p99 stabilises across repeated
  process launches. Choose by measuring run-to-run variance (see 3.4).
- The XFeat paper's own speed protocol is weak (30 frames averaged, i5-1135G7, VGA, thread count and
  warm-up unreported; embedded test on Orange Pi Zero 3 at 480x360, plain PyTorch). Do not copy it;
  use it only as a sanity reference for the "~27 FPS on i5 CPU" claim. [V]
  <https://arxiv.org/html/2404.19174>

### Timer choice and synchronisation

- CPU: `time.perf_counter_ns` (monotonic, highest resolution); pytest-benchmark defaults to
  `time.perf_counter` and lets you swap via `--benchmark-timer`. [V]
  <https://pytest-benchmark.readthedocs.io/en/latest/usage.html>
- CUDA: PyTorch tutorial: plain timeit "only times the time to launch the kernel"; first CUDA call
  is slow (cuBLAS load) so warm up; `torch.utils.benchmark.Timer` syncs and warms up for you and is
  single-thread by default, so set `num_threads` explicitly. [V]
  <https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html>
- CUDA events: TensorRT docs state events are GPU-timestamped and avoid host-side sync noise;
  wall-clock timing needs an explicit stream/device sync before reading the result. [V] TensorRT
  benchmarking page above. (`torch.cuda.Event(enable_timing=True)`, `elapsed_time` after
  `end.synchronize()`: [U] the torch.cuda.Event page only returned a redirect; standard documented
  API.)
- TensorRT: `trtexec` is the reference tool. Verified flags/behaviour: `--warmUp`, `--duration`,
  `--iterations`, CUDA graphs on by default (`--noCudaGraph` to disable; big drop without graphs
  means enqueue-bound), `--noDataTransfers`/`--includeDataTransfers`, `--noSpinWait` (less stable
  latency, lower CPU use), `--dumpProfile` / `--exportProfile=<json>` per-layer times (build with
  `--profilingVerbosity=detailed`, `--dumpLayerInfo`). Note: the page does NOT contain the names
  `useCudaGraph`, `useSpinWait`, `exportTimes`; do not use those. [V]
- GPU clocks: floating by default; `sudo nvidia-smi -lgc <freq>` locks, `-rgc` releases. NVIDIA
  gives "no definite recommendation" whether to lock: locked = more repeatable tactic
  selection/timing, floating = better average performance. Decide per plan: report locked-clock
  numbers for ablations, floating for "realistic" numbers, and label which. [V] same page.
  (Thermal-throttling section of that page was truncated; not read.)
- ONNX Runtime: `sess_options.enable_profiling = True` writes a JSON trace (per-op latency,
  threading) viewable in Perfetto/chrome://tracing; profiling itself perturbs timings, so use it for
  attribution runs, never for headline latency; `onnxruntime_perf_test` is the CLI tool; CUDA kernel
  profiling needs a build with `--enable_cuda_profiling`. [V]
  <https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html> . Thread
  management and IO binding pages exist but were not read [U]; use IO binding for GPU EPs to avoid
  copy cost in the timed region [U].

### Apple Silicon (MPS / Core ML)

- Core ML first load: loading an .mlpackage compiles to .mlmodelc, instantiates for compute units,
  then device-specialises (ANE can take seconds to minutes). Specialisation cache is tied to the
  .mlmodelc path; the Python workflow uses a temp location so cache is usually lost per process.
  Example (M1 Max): MLModel load ~15-18 s every time; `ct.models.CompiledMLModel` from a fixed path
  ~14.7 s first, 0.1 s later. Therefore: report cold-load time separately, and exclude it from
  latency; warm up several predictions after load. [V]
  <https://apple.github.io/coremltools/docs-guides/source/model-prediction.html>
- Compute units: `compute_units` (`ALL` default, `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE`) set at
  load. A preference, not a guarantee; unsupported ops fall back. The coremltools page confirmed
  only that default uses all units and CPU_ONLY exists; the full enum and fallback semantics came
  from background knowledge because Apple's doc page returned empty. [V partial] Ablate each compute
  unit as an experiment axis. Also `FastPrediction` hint trades specialisation time/memory for
  latency [V].
- Prediction API runs only on macOS, not Linux, so Core ML benchmarks cannot run in WSL2/Linux CI.
  [V]
- MPS async: `torch.mps.synchronize()` "waits for all kernels in all streams on a MPS device to
  complete". Must call it before stopping the timer. `torch.mps.Event` is documented only as a
  wrapper; the page documents no timing method, so use host timers around synchronize. Profiler:
  `torch.mps.profiler` (OS Signpost) and `metal_capture` (needs `MTL_CAPTURE_ENABLED`). [V]
  <https://docs.pytorch.org/docs/2.14/mps.html>
- Not verified: Core ML inherently-async behaviour, Xcode performance report / `MLComputePlan`
  usage, thermal state API (`ProcessInfo.thermalState`) [U]. Have a Mac-side research pass confirm
  before relying.

### CPU pitfalls

- pyperf `system tune`: governor -> performance, raises scaling_min_freq, disables Intel Turbo (via
  MSR or intel_pstate), IRQ affinity, checks AC power; isolate cores with `isolcpus=`/`nohz_full=`;
  Runner pins workers to isolated CPUs, else use `--affinity`; validate by comparing idle vs loaded
  runs; records `runnable_threads` from /proc/loadavg. [V]
  <https://pyperf.readthedocs.io/en/latest/system.html>
- Pin and fix thread counts explicitly (torch `set_num_threads`, ORT intra_op threads, OMP/MKL envs)
  and report them; PyTorch warns winner depends on thread count. [V] PyTorch benchmark tutorial.
- WSL2: Microsoft docs verified only that WSL2 is a managed Hyper-V VM and cross-OS filesystem
  access (/mnt/c) is slow; keep data/models on the Linux filesystem. [V]
  <https://learn.microsoft.com/en-us/windows/wsl/compare-versions> . Not verified [U]:
  governor/turbo control (the VM cannot set host frequency, so `pyperf system tune` is unlikely to
  work), isolcpus, host power plan effects, CUDA-on-WSL timing overhead. Treat WSL2 numbers as
  dev-only, not reportable; reportable CPU numbers need bare-metal Linux.
- Raspberry Pi: SoC limit 85 C; progressive Arm throttling from 80 C; at 85 C both Arm and GPU
  throttled; `vcgencmd measure_temp` is the accurate reading (Linux sysfs temps "can be
  inaccurate"); Pi 5 Active Cooler fan curve 50-75 C. [V]
  <https://www.raspberrypi.com/documentation/computers/raspberry-pi.html> . `vcgencmd get_throttled`
  bit meanings are NOT on that page [U]. Practice: active cooling, log temp + throttle flag per run,
  discard runs where throttling occurred, add cool-down gaps between configs, and for throughput
  mode report sustained (post-soak) not burst numbers.

### Jetson

- Orin power modes via `sudo nvpmodel -q` / `-m <ID>`; modes cap online cores and CPU/GPU/DLA/memory
  max clocks; custom modes in /etc/nvpmodel.conf. Defaults: Orin Nano 8GB 15W (ID 0), Orin NX 15W
  (ID 2), AGX Orin 32/64GB 30W (ID 2). MAXN "does not guarantee the best performance for all use
  cases" (hardware throttling when over TDP). [V]
- `sudo jetson_clocks` pins CPU/GPU/EMC at the max of the active nvpmodel mode; after that you
  cannot change nvpmodel mode without reboot; `--show`, `--store`, `--restore`, `--fan` (fan no
  longer maxed by default since R32.4). [V]
- Fan profile (`nvfancontrol`, quiet vs cool) and thermal limits (SW throttle ~99 C, HW ~103 C)
  affect sustained runs; EMC DFS can distort memory-bound tests. [V]
  <https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html>
  (R36.4; other L4T/Xavier/Nano-classic docs differ, not checked).
- Practice: report each device at (a) its default mode and (b) MAXN + jetson_clocks, with fan
  profile; use `tegrastats` for power/temp logging [U, not fetched].

### Python libraries: fit

| Library | Verified features | Verdict for xfeat-rt |
|---|---|---|
| pyperf | Process spawning, calibration, pinning, `system tune` checks, JSON results, stats compare [V] | Good for CPU-only microbenchmarks and for its system-tuning checks (reuse `pyperf system show` as a pre-flight, Linux only). Not GPU-aware (no sync). Weak fit as the main harness. |
| pytest-benchmark | Rounds/min-time, `pedantic()`, save/compare/JSON, custom `--benchmark-timer` [V]; no GPU awareness [V] | Good for regression tracking in CI; wrap callable with its own sync. Use `benchmark.pedantic` with explicit rounds/warmup (auto warmup is only on PyPy [V]). |
| torch.utils.benchmark | Auto warmup, CUDA sync, `blocked_autorange`, thread control [V] | Best for PyTorch reference path only. Does not time ONNX Runtime/TRT/Core ML engines. |
| trtexec | See above [V] | Authoritative for TensorRT engine-only numbers; cross-check Python harness against it. |
| onnxruntime_perf_test | [V] listed | Cross-check for ORT. |
| MLPerf LoadGen (`mlperf_loadgen` Python bindings) | Implements SingleStream/Offline scenarios, accuracy/performance modes [V] | Possible but heavy; not verified as pip-installable for edge use. Adopt the scenario definitions only. |
Recommendation (judgement): write a thin runtime-agnostic harness (a `Runner` callable + explicit `sync()` per backend + perf_counter_ns + numpy percentiles, results as JSON), use pytest-benchmark only if CI regression gating is wanted, and cross-check headline numbers with trtexec / ORT perf_test. No single mature library covers CUDA/ORT/TRT/CoreML/MPS uniformly; I did not find one. Not verified: other candidates (e.g. google-benchmark, nvbench, asv) were not assessed.

## 2. Accuracy evaluation

### What XFeat reports (paper arXiv 2404.19174, HTML) [V]

- MegaDepth-1500: images resized to max dim 1200; relative-pose AUC@5/10/20 deg, Acc@10, mean inlier
  ratio, #inliers; LO-RANSAC essential matrix, threshold searched per method (values not given); MNN
  matching. XFeat (4096 kpts): AUC 42.6/56.4/67.7; XFeat* (10k semi-dense): 50.2/65.4/77.1.
- ScanNet-1500: VGA; XFeat AUC 16.7/32.6/47.8 (XFeat* 18.4/34.7/50.3).
- HPatches: MHA@3/5/7 px (ALIKE protocol), illumination 95.0/98.1/98.8, viewpoint 68.6/81.1/86.1;
  MAGSAC++ for homography; resolution not stated. Note: the XFeat repo README (as fetched) has NO
  HPatches script.
- Aachen day-night with HLoc, max dim 1024, 4096 kpts.
- XFeat repo MegaDepth-1500/ScanNet-1500 scripts:
  `python3 -m modules.eval.megadepth1500 --dataset-dir ... --matcher xfeat --ransac-the 2.5` and
  `modules.eval.scannet1500`; download helpers
  `python3 -m modules.dataset.download --megadepth-1500 | --scannet-1500`. [V]
  <https://github.com/verlab/accelerated_features> (Apache-2.0)
- LighterGlue numbers (README, via GlueFactory eval): MegaDepth-1500 AUC@5/10/20 = 0.444/0.610/0.746
  at 640 px, 1300 kpts; 0.564/0.710/0.819 at 1024 px, 4096 kpts (SuperPoint+LightGlue
  0.469/0.633/0.762 and 0.591/0.738/0.841). Use these as the PyTorch-reference targets. [V]
  <https://github.com/verlab/accelerated_features/blob/main/README.md>

### Datasets

| Dataset | Source / size | License / terms |
|---|---|---|
| MegaDepth-1500 | ~600 MB test set from LoFTR Google Drive (also fetched by XFeat/GlueFactory download scripts; GlueFactory states ~1.5 GB) [V]. Pair list originates from LoFTR/SuperGlue-style 1500 pairs [pair provenance U] | Underlying MegaDepth (Li & Snavely, CVPR 2018): code MIT; dataset license/terms NOT stated on the GitHub page; project site fetch failed (certificate error). UNVERIFIED. |
| ScanNet-1500 | ~1 GB LoFTR Drive (GlueFactory ~1.1 GB) [V] | ScanNet data under ScanNet Terms of Use, requires signed agreement emailed to <scannet@googlegroups.com>; ToS text not read, so non-commercial restriction UNVERIFIED. Check whether the 1500-pair derivative redistribution by LoFTR is covered [U]. <https://github.com/ScanNet/ScanNet> |
| HPatches | 4.2 GB patches / 1.3 GB sequences, hosted on HuggingFace `vbalnt/hpatches`; repo MIT [V]; GlueFactory downloads ~1.8 GB | Repo MIT shown; dataset-specific terms not described. Sequence count (116) not confirmed. <https://github.com/hpatches/hpatches-dataset> |
Do not redistribute datasets in xfeat-rt; fetch at runtime with the upstream download scripts.

### Evaluation code as library

- GlueFactory (cvg/glue-factory): Apache-2.0 (nonfree models separate); installs only by
  `git clone` + `pip install -e .`, no PyPI release seen on the page [V]. Benchmarks:
  `gluefactory.eval.megadepth1500` (pose AUC 5/10/20), `.hpatches` (homography AUC 1/3/5 px),
  `.scannet1500`, `.eth3d`; two-stage design (export predictions.h5 then evaluate), robust
  estimators selectable (poselib, opencv, pycolmap, ransac_th list auto-selects best), results in
  `summaries.json`. [V] <https://github.com/cvg/glue-factory> and docs/evaluation.md. Its AUC
  definition and the custom-matcher plug-in path were NOT confirmed from docs (the evaluation.md
  page does not cover them); you need to read `gluefactory/eval/*.py` and the model base class.
  Whether upstream GlueFactory ships an XFeat model: not verified (LighterGlue was trained "using
  GlueFactory" per XFeat README, likely via a fork).
- Library vs copy: both XFeat and GlueFactory are installable as editable git dependencies (pin
  commit), not importable from PyPI. Recommended: add GlueFactory as a pinned git dependency in an
  `eval` extra, wrap our runtimes as a GlueFactory model (or export our keypoints/matches to its
  predictions format) so the metric/estimator code is not copied. Verify feasibility with a spike.
  (judgement)
- Alternative tools not assessed: kornia, `lightglue` repo eval (it delegates to GlueFactory [V],
  <https://github.com/cvg/LightGlue>; LightGlue repo has no XFeat weights [V], Apache-2.0 except
  SuperPoint weights).
- Pitfall: GlueFactory uses COLMAP pixel convention (top-left corner of the top-left pixel is (0,0))
  [V]; XFeat's own script uses a different (OpenCV-style) convention probably [U]. Keep one
  convention across reference and exported paths and confirm numbers reproduce the paper/README
  first (reference reproduction is gate 0).
- Keep RANSAC fixed (estimator, threshold, seed) across precisions; otherwise RANSAC noise swamps
  export error. Report AUC with confidence from bootstrap or repeated seeds [practice, no source
  fetched].

## 3. Reference-vs-export numerical contract

### Established tools and conventions

- torch.testing.assert_close: pass if |actual-expected| <= atol + rtol*|expected|; defaults float32
  rtol 1.3e-6 atol 1e-5, float16 rtol 1e-3 atol 1e-5, bfloat16 rtol 1.6e-2 atol 1e-5; mismatched
  dtypes use the larger tolerance. [V] <https://docs.pytorch.org/docs/2.14/testing.html> . These are
  unit-test defaults for bit-near ops, not a contract for whole-network FP16/INT8 exports.
- Polygraphy (`polygraphy run model.onnx --onnxrt --trt [--fp16]`): default `elemwise` check fails
  an element only if both abs and rel error exceed atol and rtol; also
  `--check-error-stat mean|median|max`; per-output tolerances; docs say defaults "usually
  appropriate for FP32 but may not be appropriate for reduced precisions"; example uses
  atol=rtol=1e-3 with median for TF32 TRT vs ORT. Numeric default values not given on page. [V]
  <https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/run/01_comparing_frameworks/README.md>
- ONNX Runtime quantization debugging: `qdq_loss_debug` APIs (create_weight_matching,
  modify_model_output_intermediate_tensors, collect_activations, create_activation_matching) compare
  float vs quantized weights/activations; optimise during pre-processing, not quantization;
  quantization "is not a loss-less transformation"; calibrators MinMax/Entropy/Percentile; QDQ S8S8
  first choice; per-channel, reduce_range, U8U8 as accuracy levers. [V]
  <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html> .
  SQNR/activation-error helper function names were not on the page.
- No authoritative source I found specifies FP16 or INT8 tolerances for feature-matching exports.
  Not verified: LightGlue-ONNX validation approach (not fetched), TensorRT docs on FP16 accuracy,
  any published FP16 tolerance for detectors.

### Proposed contract (design, built from the above; numbers are placeholders to be set by 3.4)

Compare per stage, since tensors-level and decision-level errors differ:

1. Raw network outputs on identical preprocessed input (heatmap/reliability, descriptor map,
   keypoint scores): max/mean/p99 abs error; per-descriptor cosine similarity (L2-normalised 64-D
   XFeat descriptors; report min, p1, mean); NOT elementwise-and.
2. Detection set: after identical top-k/NMS, keypoint set overlap = |matched within r px| / k via
   nearest-neighbour with radius (e.g. 1 px, to be derived), IoU of index sets, rank correlation of
   scores; note top-k boundary flips are expected and benign.
3. Matches: for same detections or each own detections, mutual-NN match set agreement (Jaccard of
   (i,j) pairs on keypoints matched back to reference), inlier-count ratio.
4. Task metrics: MegaDepth/HPatches/ScanNet AUC and MHA delta vs PyTorch FP32 with fixed RANSAC.
   This is the final acceptance criterion; tensor-level numbers are diagnostics.

### Deriving tolerances from measurement (no invented numbers)

1. Establish the noise floor: run the PyTorch FP32 reference vs itself under benign perturbations
   (different thread counts, CPU vs GPU, TF32 on/off, different batch composition) and measure the
   tensor/detection/match metrics. Any export must be within a small multiple of this floor for
   FP32.
2. FP16 tolerance: run PyTorch `.half()`/autocast on GPU as the "FP16 oracle"; the export's error to
   FP32 should be no worse than PyTorch-FP16's error to FP32 (the achievable floor), measured on a
   calibration set of N image pairs (>= several hundred from MegaDepth/HPatches train-disjoint
   images). Set tolerance = e.g. p99.9 of observed error * safety margin, store as data with the
   dataset hash and commit.
3. INT8: use calibration/validation split; accept by task metric delta (e.g. AUC@5 drop <= X, X
   chosen by user; to be ablated), not tensor atol. Use ORT activation matching to find layers to
   exclude.
4. Use bootstrap over pairs to get confidence intervals on metric deltas; a drop within CI of
   FP32-vs-FP32-nondeterministic is "no drop".
5. Tolerance regression test: assert contract on a small fixed pair set at CI time, tolerances
   loaded from a versioned file; full-dataset run is offline.
6. Make comparisons backend-parameterised (CPU/CUDA/TRT/CoreML/MPS) since each has its own floor;
   Core ML ANE runs FP16 internally [U], so ANE/GPU ceilings are FP16 floors.

## Could not verify

- MegaDepth and ScanNet dataset licence text (cs.cornell.edu cert error; ScanNet ToS PDF unread);
  HPatches dataset terms and scene counts.
- GlueFactory AUC definition, custom-matcher plug-in, XFeat model presence; XFeat's pixel
  convention; LighterGlue weight licence (repo badge says Apache-2.0).
- Apple Core ML computeUnits doc (empty fetch), Core ML async behaviour, ANE precision, Xcode perf
  tooling; torch.cuda.Event doc (redirect only).
- WSL2 frequency/governor/isolcpus behaviour; Pi `get_throttled` bits; tegrastats; trtexec thermal
  section (truncated page).
- Any standard tolerance values for FP16/INT8 matcher exports; numeric Polygraphy defaults.
- The XFeat arXiv PDF could not be parsed; HTML v1 used instead (numbers above come from it, not
  independently cross-checked against the PDF).
