# 02 - Export routes: PyTorch -> targets (researched 2026-10-23)

Method: docs fetched, PyPI JSON, and upstream source read directly (onnxscript 0.7.2 wheel, pytorch
main, onnx-tensorrt main, coremltools 9.0 tag, verlab/accelerated_features main, kornia main).
Nothing was run end to end; no export was actually executed.

## Bottom line

- Backbone CNN (XFeat net) and the LighterGlue transformer layers export cleanly. Post-processing
  (NMS, top-k, descriptor sampling, match filtering) is the fragile part.
- Recommended split: export graph = backbone only (heatmaps/descriptor map/reliability) and
  LighterGlue with pruning/early-stop OFF at fixed N. Keypoint NMS + top-k + sparse descriptor
  sampling + mutual-NN/threshold filtering: reimplemented outside graph (NumPy/native). Evidence:
  the only working XFeat-on-NPU port does exactly this (below).
- Versions today: torch 2.14.1 (2026-09-30), onnxscript 0.7.2 (2026-09-09), onnx 1.23.2, onnxruntime
  1.31.0, TensorRT 11.4.0.106 (2026-10-08; NOT on Jetson), OpenVINO 2026.4.1, coremltools 9.0
  (2025-11-10), ExecuTorch 1.5.1.

## 1. PyTorch exporter (torch.onnx.export dynamo=True)

- dynamo=True is the default since 2.9 (<https://docs.pytorch.org/docs/2.14/onnx_export.html>;
  confirmed in torch/onnx/__init__.py main: `dynamo: bool = True`). `fallback` option removed in
  2.11. `dynamic_axes` deprecated in favour of `dynamic_shapes` (torch.export syntax).
  `optimize=True` default (constant folding). `report=True` for failure diagnosis.
  `custom_translation_table` maps an aten target to an ONNX Script function (the hook for custom
  ops). Legacy TorchScript exporter is capped at opset 20.
- Default opset: ONNX_DEFAULT_OPSET = 20, ONNX_MAX_OPSET = 23 (torch/onnx/_constants.py, v2.14.1).
  TorchLib registry base opset 18 (_constants.py in exporter/). Lower than 18 gets auto-upgraded
  with a warning.
- Opset choice: 20 gives GridSample-20 (bicubic ok), TopK-11, Resize, no Attention op. Opset 23 adds
  ONNX Attention (<https://onnx.ai/onnx/operators/onnx__Attention.html>; current is v25 in onnx
  1.23). TensorRT 11.4 parses opsets 9-24; OpenVINO ONNX frontend lists Attention 23/24, GridSample
  16/20/22 (openvino master src/frontends/onnx/docs/supported_ops.md). Pick opset 20 for ORT/TRT/OV
  breadth; 23 only if you want fused Attention and verified runtimes.
- SDPA: onnxscript 0.7.2 torchlib
  (`function_libs/torch_lib/ops/nn.py::aten_scaled_dot_product_attention`) lowers it to
  MatMul/Softmax (separate no-mask/bool-mask/float-mask paths), 4D q/k/v only, dropout path handled.
  I found no torchlib lowering to ONNX `Attention` (grep of wheel); so SDPA exports as plain ops at
  any opset. Runtimes (ORT, TRT) re-fuse attention patterns themselves; LightGlue-ONNX reports
  FlashAttention-2 fusion in ORT>=1.16 (their README).
- grid_sample: `aten_grid_sampler_2d` -> ONNX GridSample with mode/padding_mode preserved (core.py).
  Needs opset >=20 for bicubic (opset 16 lacks it).
- topk: `aten_topk` exists (core.py); argsort exists. nonzero exists but gives data-dependent output
  shape (unbacked symint); max_pool2d exports as MaxPool (indices output variant is separate).
- Dynamic vs fixed keypoint count: NonZero-based NMS yields dynamic N. Fixed N (e.g. top_k padded)
  is what TRT wants: TensorRT ONNX operator table (onnx-tensorrt docs/operators.md, 4.4 main) says
  TopK needs `K < 3840` and sorted=1. XFeat default top_k=4096 therefore exceeds the TRT TopK limit
  (reported also in verlab PR #5 as "TensorRT top-k exceeding maximum", unanswered). Keep K <= 3839
  or topk outside the graph.

## 2. XFeat specifics (verlab/accelerated_features main, modules/xfeat.py, model.py, interpolator.py)

Ops in the pipeline and export status:

| Stage | Op | Status |
|---|---|---|
| preprocess | InstanceNorm2d(1) on input, F.interpolate bilinear | exports; BCDL port moved InstanceNorm to CPU preprocessing for its NPU |
| backbone | convs, `_unfold2d` (tensor.unfold x2 + reshape), F.interpolate bilinear x2 | exports; BCDL replaced _unfold2d with pixel_unshuffle (-> SpaceToDepth) for its compiler |
| heatmap | softmax, reshape | clean |
| NMS | MaxPool2d(5, s1) + `nonzero` + python `for b in range(B)` loop + padding into zeros tensor | Data-dependent loop. Intel KB (reviewed 2026-02-18): ONNX export of this loop is not captured correctly and caused accuracy loss in OpenVINO 2024.6; fix = upgrade to 2025.3+ or rewrite NMS without the loop (max-pool + nonzero + pad) <https://www.intel.com/content/www/us/en/support/articles/000102315.html> |
| reliability scores | InterpolateSparse2d('nearest') and ('bilinear') = F.grid_sample | exports (GridSample) |
| top-k | `argsort(-scores)` + gather + slice [:top_k] | exports (argsort/topk), dynamic slice |
| descriptors | InterpolateSparse2d('bicubic') = grid_sample bicubic | exports at opset>=20 only. NOT portable: Core ML's torch converter `grid_sampler` (coremltools 9.0 ops.py) only maps bilinear/nearest; bicubic would fail. TRT importer (onnx-tensorrt main onnxOpImporters.cpp) maps nearest/linear/bicubic -> kNEAREST/kLINEAR/kCUBIC, so TRT ok. |
| final | F.normalize, mask `scores>0` (dynamic output size) | needs fixed-N + validity mask |
Real examples:

- verlab PR #5 (acai66, open since 2024-05-04, still unmerged; comments into Sept 2025): ONNX export
  of xfeat, xfeat_dualscale, matching, xfeat_matching with dynamic shapes; later commits:
  fewer-than-4800 keypoints, different-sized sets, window-size non-multiples; "argsort -> topk"
  optimization; TRT demo with profiles 64x64..1280x1280.
  <https://github.com/verlab/accelerated_features/pull/5>
- BCDL XFeat for RDK S100 (HF ruisv/bcdl-xfeat): compiles only the conv backbone (3 MB); NMS, top-k,
  sparse sampling stay on CPU; warns bicubic (descriptors) vs bilinear (reliability) mix-up leaves
  descriptor cosine ~0.9965. <https://huggingface.co/ruisv/bcdl-xfeat>
- Luxonis forum (Myriad blob): opset 11 Resize error in OpenVINO, opset 10 then fails on index_put;
  no reply seen.
  <https://discuss.luxonis.com/d/4760-unable-to-convert-xfeat-model-from-openvino-ir-to-myriad-blob-format>
- Hailo community: XFeat compile blocked (calibration data).
  <https://community.hailo.ai/t/xfeat-compilation/14926>
- I found no public XFeat Core ML or ExecuTorch export.

## 3. LighterGlue / LightGlue

- LighterGlue in verlab repo = kornia `LightGlue` with an xfeat conf: n_layers=6, num_heads=1,
  descriptor_dim=96, flash=True, depth_confidence=-1 (early stop OFF), width_confidence=0.95 (point
  pruning ON) (modules/lighterglue.py).
- Export hazards in kornia/lightglue.py main: point pruning uses `torch.where(mask)[1]` with
  data-dependent sizes (lines ~890-906) -> set width_confidence=-1 and depth_confidence=-1 for
  export; `filter_matches` and the per-pair `matches` list are data-dependent (`torch.where(valid)`)
  -> export only m0/m1/mscores (fixed shape [B,M]) and build index pairs outside.
  `_check_keypoints_normalized` is skipped under `is_exporting()` (kornia already has an export
  hook). Attention uses F.scaled_dot_product_attention (flash path only on CUDA device; fine).
  kornia also has `compile()` with static lengths (256..1536) for torch.compile, not export.
- Prior art: fabio-sim/LightGlue-ONNX (Apache-2.0, ~820 stars): full extractor+matcher ONNX via
  "LightGlue Dynamo CLI" (`lightglue_dynamo` pkg), TensorRT via ORT EP or `trtexec`/Polygraphy,
  FP16, FP8 Q/DQ (2026-01-19), dynamic batch (2024-07-17), changelog latest 2026-07-20
  (RaCo-ALIKED). Historic README states early-stopping/adaptive pruning are hard to export (limited
  dynamic control flow) and recommends a static keypoint count for TRT and shape inference on the
  ONNX before TRT. It targets SuperPoint/DISK/ALIKED, not XFeat.
  <https://github.com/fabio-sim/LightGlue-ONNX> . PINTO0309 fork removed edge-keypoint filtering
  because it introduced NonZero.
- No public LighterGlue-specific ONNX/TRT/CoreML export found.

## 4. TensorRT

- TRT 11.4.0 (PyPI 2026-10-08): wheels `tensorrt` -> tensorrt_cu13_libs + bindings; bindings have
  cp313 wheels for manylinux x86_64 and aarch64 (SBSA) and win. ONNX opsets 9-24.
- Jetson: TRT 11.4 does NOT support JetPack; stay on the 10.x that ships with JetPack
  (<https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/api/migration/tensorrt-10x-to-11x-jetson.html>).
  JetPack 7.0/7.1 = Thor, CUDA 13.0, Ubuntu 24.04; JetPack 7.2 adds Orin, TRT 10.16.2 (secondary
  source: RidgeRun wiki, unverified against NVIDIA notes). Hardware-compatibility mode not supported
  on JetPack.
- Portability
  (<https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/inference-library/engine-compatibility.html>):
  engine only loads on same TRT version unless built VERSION_COMPATIBLE (TRT 10 -> 11, 8 -> 10; may
  be slower; needs host-code enable); GPU: exact compute-capability match by default; hardware
  compat levels AMPERE_PLUS / SAME_COMPUTE_CAPABILITY (not on JetPack); not portable across OS. =>
  ship ONNX and build engines on the target (or per-SM), cache them.
- Timing cache: ORT TRT EP docs: trt_timing_cache_enable, reuse across models on same compute
  capability, build time e.g. 34.6s -> 7.7s; engine cache not portable
  (<https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html>; the doc's
  version table stops at ORT 1.22/TRT 10.9, stale vs ORT 1.31). I could not fetch NVIDIA's own
  timing-cache page.
- Build: trtexec --onnx= --fp16 --minShapes/--optShapes/--maxShapes (not re-verified for 11.x, where
  weakly typed FP16 flags are being deprecated in favour of strongly typed; NVIDIA tells users to
  move to strong typing/explicit Q/DQ). Fixed 1280x960 = no profile needed.
- Op restrictions (onnx-tensorrt main): TopK K<3840 & sorted; GridSample 4D/5D, FP32/FP16/BF16
  (cubic mapped); MaxPool no indices output; NonZero supported (FP32/FP16 only); Attention 4D only,
  no past-KV, no qk_matmul_output/softcap; Loop shapes must be static across iterations.

## 5. Core ML (Apple Silicon)

- coremltools 9.0 (2025-11-10, wheels cp313 macOS arm64/x86_64, manylinux x86_64; no cp314).
  torch.export (ExportedProgram) support added in 8.0, "beta", ~70% op coverage per docs
  (<https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html>); 8.2
  notes ~83% parity with the torch.jit.trace converter, which docs still call the stable/more
  performant path. No ONNX->Core ML converter (onnx-coreml legacy only). So route = PyTorch ->
  torch.jit.trace (or torch.export) -> ct.convert -> ML Program (.mlpackage), not via ONNX.
- Defaults: ML Program is FP16 unless compute_precision=FLOAT32; FP32 can't run on the Neural
  Engine; compute_units All/CPU_AND_GPU/CPU_AND_NE/CPU_ONLY
  (<https://apple.github.io/coremltools/docs-guides/source/typed-execution.html>). Flexible shapes:
  Enumerated (<=128) or RangeDim with upper bound; fixed shape simplest (flexible-inputs doc).
  Run-time prediction needs macOS + coremltools (or Swift); Linux can convert but not predict.
- Torch ops present in 9.0 ops.py: grid_sampler (bilinear/nearest only, no bicubic), topk, nonzero,
  argsort, scaled_dot_product_attention, pixel_unshuffle, instance_norm, index_put. nonzero gives
  dynamic shape (works poorly with ANE).
- => On Core ML, change descriptor sampling to bilinear (accuracy impact on descriptors unmeasured;
  weights trained with bicubic) or do it outside graph.

## 6. OpenVINO IR

- 2026.4.1 (2026-10-01): wheels cp313 for linux x86_64, linux aarch64, macOS arm64, win. ONNX
  frontend uses GraphIterator by default from 2026.0 (ONNX_ITERATOR=0 reverts). IR has GridSample-9,
  ScaledDotProductAttention-13, TopK-11, NonZero-3. Convert from ONNX (`ov.convert_model`) is the
  supported route; docs:
  <https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-onnx.html>. Arm
  CPU plugin op coverage for these ops: NOT verified.
- Relevant for x86 CPU + Raspberry-Pi class Arm only if you want an alternative to ORT; Intel KB
  shows XFeat NMS loop broke in 2024.6.

## 7. ExecuTorch (1.5.1, 2026-09-22)

- Needs torch to export (torch.export -> .pte); runtime is C++/pybind. PyPI package pulls torchao,
  coremltools==9.0 (mac/linux x86_64), optional openvino<2026 extra. Backends (XNNPACK, CoreML,
  Vulkan, QNN, openvino...) not verified from docs here. Not recommended: adds a second export
  pipeline, no XFeat/LightGlue example found, runtime Python bindings with no-PyTorch constraint
  unverified.

## 8. Runtime-side fit (Python >=3.13, no torch)

- onnxruntime 1.31.0 (2026-10-09) has cp313 wheels (linux x86_64/aarch64, macOS arm64, win); TRT
  EP/CUDA EP/CoreML EP available via different packages (not verified here). onnx 1.23.2 PyPI shows
  no cp313-specific wheel in the file list (probably abi3 py3-none; unverified).
- Covers: x86 CPU (ORT CPU / OpenVINO), NVIDIA (ORT-TRT or native TRT), Arm CPU (ORT aarch64 wheel),
  Jetson (TRT 10.x from JetPack), Apple (ONNX->ORT CoreML EP unverified; or native Core ML).

## Conclusion: what exports cleanly vs. what to reimplement

Exports cleanly (all targets): XFeat backbone (convs, bilinear interpolate, pixel_unshuffle
replacing unfold, InstanceNorm, softmax heatmap); LighterGlue transformer layers incl. SDPA, rotary
embeddings, log-assignment, sigmoid_log_double_softmax at fixed M,N with pruning/early-stop
disabled. Export with care: grid_sample for reliability scores (bilinear/nearest fine everywhere);
descriptor bicubic sampling (OK ONNX>=20/TRT/OV; not Core ML); topk with fixed K (<3840 for TRT);
mutual-NN match filter at fixed shape. Reimplement outside graph (NumPy or backend-native): (1)
keypoint NMS threshold + dynamic nonzero compaction; (2) top-k selection with variable valid count
(or in-graph fixed-K top-k only on TRT/ORT variants); (3) sparse descriptor sampling if targeting
Core ML (bicubic) or if a backend lacks GridSample; (4) LightGlue point pruning/early stop (drop, or
loop in host); (5) match index extraction `where(valid)`. Padding to fixed N with validity mask +
feeding keypoints to the matcher as inputs is the portable interface.

## Not verified

- No export was actually run (no torch install); XFeat/LighterGlue behaviour under dynamo=True with
  torch 2.14 is inferred from source, not observed. Highest-value follow-up: run the export in the
  reference workspace.
- Whether the dynamo exporter ever emits ONNX `Attention` at opset 23 (not found in onnxscript 0.7.2
  torchlib).
- NVIDIA timing-cache docs and trtexec 11.x flags; JetPack 7.1/7.2 exact TRT versions from NVIDIA;
  OpenVINO Arm plugin coverage; ORT CoreML EP op coverage; ExecuTorch backend status; whether TRT's
  cubic GridSample is supported in FP16 on all archs.
- Scratchpad has a stray file
  /tmp/claude-1000/-home-chris-xfeat-dev/1902fc88-97d1-56bb-8dcc-d233cba4e78e/scratchpad/dl/lg115.patch
  not created by me.
