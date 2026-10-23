# 04 — Prior art: XFeat / LightGlue / LighterGlue ports and optimisations

Checked 2026-10-23. "Last push" = GitHub `pushed_at` from the REST API. Numbers are copied from the
source named next to them; "unverified" means the source gives no hardware/resolution/precision, or
I could not open it.

## Bottom line

- No port targets XFeat + LighterGlue on several backends with published, reproducible latency. The
  XFeat+LighterGlue ONNX exports (noahzhy, acai66 PR #5, stschake fork, Derkai52) publish
  **no latency numbers**. Users in verlab issue #100 report accuracy complaints, which are
  unresolved.
- The most mature reference is **fabio-sim/LightGlue-ONNX** (Apache-2.0, 820★, last push
  2026-07-30). It covers SuperPoint, DISK, ALIKED and RaCo-ALIKED. It does **not** cover XFeat or
  LighterGlue.
- The best measured XFeat numbers on Jetson come from **kornia/vision-rt** (Rust + TensorRT, Orin
  Nano). Its XFeat graph is backbone-only, with NMS, top-K and sampling moved out to CUDA. It pairs
  XFeat with mutual-NN, not LighterGlue.
- **Adaptive depth:** one implementation exists, fabio-sim PR #115 (merged 2026-07-29). No speed or
  accuracy numbers have been published for it.
- **Point pruning (adaptive width):** no exported implementation found anywhere. Every
  XFeat/LighterGlue port disables it, although upstream LighterGlue enables it
  (`width_confidence: 0.95`).
- **Core ML / MLX / ncnn / OpenVINO-specific / Arm-CPU-optimised:**
  - Nothing real exists for LighterGlue on any of these.
  - The one "XFeat & LighterGlue" Core ML export (EliFuzz/cv) actually contains a mutual-NN matcher,
    not LighterGlue.
  - Searches for MLX, ncnn and OpenVINO LightGlue repos returned nothing.

## Reference facts (upstream)

| Item | Value | Source |
|---|---|---|
| LighterGlue config | input_dim 64, descriptor_dim 96, n_layers 6, num_heads 1, depth_confidence −1 (early stop off), width_confidence 0.95 (pruning **on**), filter_threshold 0.1. Wraps `kornia.feature.lightglue.LightGlue` | <https://github.com/verlab/accelerated_features/blob/main/modules/lighterglue.py> |
| LighterGlue accuracy (MegaDepth-1500, AUC@5/10/20) | 640 px, 1300 kpts: 0.444/0.610/0.746. 1024 px, 4096 kpts: 0.564/0.710/0.819. README claims "~3× faster than LightGlue", with no timing table | <https://github.com/verlab/accelerated_features> |
| XFeat speed (paper) | i5-1135G7 CPU, VGA, vanilla PyTorch: XFeat 27.1 ± 0.33 FPS, XFeat* 19.2 ± 1.12 FPS. Orange Pi Zero 3 (Cortex-A53), 480×360: 1.8 FPS. Up to 4096 kpts (sparse) | <https://arxiv.org/html/2404.19174> |
| XFeat GPU (README) | about 1,400 FPS batched VGA on RTX 4090. Over 150 FPS single batch | verlab README |
| LightGlue adaptivity (paper) | Exit when fraction of confident points > α (α = 95%), with λ_ℓ = 0.8 + 0.1·e^(−4ℓ/L). Prune a point if confident and σ < β (β = 0.01). Average exit layer 5.7/9. Adaptive speedup over full model 1.45× overall (1.86× easy, 1.16× hard). RTX 3080, 2048 SuperPoint kpts, matching only: LightGlue 44.2 ms, adaptive 31.4 ms | <https://arxiv.org/html/2306.13643> |
| Upstream LightGlue license | Apache-2.0 (SuperPoint weights restrictive) | <https://github.com/cvg/LightGlue> |

## Ports and exports

### LightGlue (non-XFeat extractors)

**fabio-sim/LightGlue-ONNX** — <https://github.com/fabio-sim/LightGlue-ONNX> — Apache-2.0, last push
2026-07-30, actively maintained.

- **What it exports:** one end-to-end extractor+matcher ONNX graph (SuperPoint, DISK, ALIKED,
  RaCo-ALIKED). Top-K keypoint selection, mutual check and threshold are all inside the graph. Batch
  and H/W can be dynamic.
- **Runtimes:** ORT CPU, CUDA, TensorRT and OpenVINO EPs, plus pure TensorRT through Polygraphy.
  Also has a WebGPU demo, an FP8 Q/DQ path (static shapes) and Kornia integration
  (`kornia.feature.OnnxLightGlue`).
- **2024 write-up**
  (<https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/>):
  - Setup: i9-12900HX + RTX 4080 Laptop, ORT 1.18.1, TRT 10.2, SuperPoint+LightGlue end-to-end.
  - Result: 2–4× faster than FP16 `torch.compile`, TRT FP16 up to 4×. The ORT CUDA EP is about equal
    to torch at low K and **slower at high K**.
  - No absolute latencies in text (charts only).
- **2026 RaCo write-up**
  (<https://fabio-sim.github.io/blog/gpt-5-6-sol-discovers-tensorrt-optimizations-raco-aliked-lightglue/>):
  - Setup: same laptop, TRT 10.16, FP16, MegaDepth-1500, median per pair.
  - 512², K=1024: **11.9 ms**. 768², K=3584: 31.1 ms. 1024², K=3584: 48.4 ms.
  - Median 2.67× over FP16 `torch.compile`.
  - Precision@1px 74.16 → 73.11 %.
- **Techniques (learn, don't copy):**
  - Fixed top-K instead of a threshold, which keeps shapes static.
  - Top-K instead of ArgMax: about 30% faster.
  - `reshape` instead of `flatten`/`unflatten`; non-negative axes.
  - Attention emitted as `com.microsoft::MultiHeadAttention` through `torch.library`, because ORT's
    pattern fusion is brittle. FlashAttention-2 through ORT ≥ 1.16 is "up to 80% faster" at long
    sequences.
  - Attention calls per layer cut from 4 to 2.
  - Chunked hierarchical TopK (65,536-element chunks): TopK 39.5 → 4.1 ms at 1280², K=3584.
  - BN folding.
- **Pitfalls:**
  - TRT TopK caps K at 3840.
  - FP16 overflow in flat-index → coordinate conversion silently gave zero matches; fixed with
    integer floor-div.
  - Shape-inference failures on a naive export.
  - CUDA/cuDNN library-path issues.

**colmap/LightGlue-ONNX** — <https://github.com/colmap/LightGlue-ONNX> — Apache-2.0, last push
2026-02-13, 2★.

- A fork of cvg/LightGlue "ONNX export for LightGlue matcher". It is not the same project as
  fabio-sim's.
- No benchmarks of its own.

**qdLMF/LightGlue-with-FlashAttentionV2-TensorRT** —
<https://github.com/qdLMF/LightGlue-with-FlashAttentionV2-TensorRT> — MIT, last push 2025-03-03.

- A CUTLASS/CuTe FlashAttention-v2 TensorRT plugin for head-dim 64, targeting Jetson Orin NX 8 GB
  with TRT 8.5.2.
- **No numbers on the page — unverified.**
- Relevant only if our head dim is 64. LighterGlue has 1 head of dim 96.

**yuefanhao/SuperPoint-LightGlue-TensorRT** —
<https://github.com/yuefanhao/SuperPoint-LightGlue-TensorRT> — Apache-2.0, last push 2023-12-14
(stale).

- RTX 3080, 320×240: LightGlue 2.54 ms, SuperPoint 0.95 ms (256 pts).
- The README says main "has some bugs".

**fettahyildizz/superpoint_lightglue_tensorrt** —
<https://github.com/fettahyildizz/superpoint_lightglue_tensorrt> — Apache-2.0, last push 2026-02-14.

- TRT 8.5. LightGlue engine on GTX 1650 Ti: mean 1.30 ms enqueue (K up to 1024, 720×1280).
- Claims "onnxsim cut inference time by 35–40%".
- NMS, top-K and descriptor sampling live in C++ post-processing, not in the graph.

**OroChippw/LightGlue-OnnxRunner** — <https://github.com/OroChippw/LightGlue-OnnxRunner> — MIT, last
push 2024-02-01 (stale).

- C++ ORT on Windows, using fabio-sim models.
- i5-13500H CPU: decoupled LightGlue about 2.1 s. RTX 4060 Laptop: about 155–230 ms. Old 2023
  models.
- These figures are mainly evidence that the early unfused graphs were very slow on CPU.

**Raven-Z-CV/Alike_Lightglue_RKNN-TRT_Deploy** —
<https://github.com/Raven-Z-CV/Alike_Lightglue_RKNN-TRT_Deploy> — last push 2026-09-24. License
conflict: API says MIT, README says Apache-2.0.

- **What it exports:** ALIKE + LightGlue cut from 9 to 7 layers (truncation + fine-tune). Backbone,
  NMS, top-K, soft-argmax, descriptor sampling and matcher are all in-graph, for RKNN FP16 and TRT.
  512², K=512.
- **Graph rewrite:** replacing `Einsum` with `MatMul` alone gave 2.1× on CPU (222 → 105 ms). The
  7-layer version is 78 ms.
- Full pipeline: CPU 365 → 140 ms, TRT FP16 3.1 ms.
- **CPU/GPU hardware for the table is not stated — unverified.** The demo was on an RTX 5090.
- HPatches: ONNX bit-exact to torch. FP16 backends keep about 99.8% of inliers but give about 10%
  fewer matches.
- **Notes:**
  - Engines are not portable.
  - "fp16 is a request" — check per-tensor dtypes.
  - TRT builds are not bit-reproducible.
  - Einsum ops are left off the NPU.

**Other LightGlue ports:**

- PIPIKAI/LightGlue-RKNN (Apache-2.0, 2025-12) and newcloud123/lightglue_rknn (RK3588, 2025-12)
  re-export fabio-sim graphs for RKNN. No numbers.
- MapMindAI/EasyTensorRT (MIT, 2026-09) serves the models through Triton. GTX 1650 Ti, LightGlue:
  ONNX 29.31 ms vs TRT 31.36 ms (TRT not faster).
- cvg/LightGlue issue #112: GridSample/opset-15 incompatibility on the NXP i.MX 93 NPU.

### XFeat and XFeat + LighterGlue

**verlab PR #5 (acai66)** — <https://github.com/verlab/accelerated_features/pull/5> — open and
unmerged since 2024-05.

- Exports `xfeat.onnx`, `xfeat_dualscale.onnx`, `matching.onnx` and end-to-end
  `xfeat_matching.onnx`. Later added detectAndCompute and LighterGlue. Dynamic shapes.
- Commit "argsort → topk".
- **Pitfalls reported:**
  - The TRT EP fails in "select top-k" when the max K is exceeded.
  - The ORT CUDA EP is **slower than PyTorch**, blamed on a large `bmm`.
- No numbers.

**verlab PR #4 (IamShubhamGupto)** — <https://github.com/verlab/accelerated_features/pull/4> —
draft, idle since 2024-05.

- Jetson AGX Xavier (JP 5.1): XFeat CUDA 21.4 FPS vs TRT 24.7 FPS.
  **Resolution and precision not stated.**
- Commits say "tensorrt export serialization does not work".

**noahzhy/xfeat_lightglue_onnx** — <https://github.com/noahzhy/xfeat_lightglue_onnx> — Apache-2.0,
last push 2025-05-20.

- Separate XFeat and LighterGlue ONNX graphs (opset 17, onnxsim, dynamic keypoint axes).
- **`width_confidence` set to −1 "because onnx is not supported dynamic control flow"**, so pruning
  is off.
- No latency numbers, and not evaluated on public datasets (the author says so in #100).
- Forks: soraxas/xfeat-lightglue-onnx (identical README), lumi-living/xfeat_lightglue_onnx_mowe,
  nocturnalnoob.

**verlab issue #100** — <https://github.com/verlab/accelerated_features/issues/100> — LighterGlue
ONNX.

- Links the stschake fork commit acecc15 and the noahzhy repo.
- Several users report poor MegaDepth/private-data accuracy after export. One bug was fixed in April
  2025, but complaints continued in June 2025.
- No maintainer reply.
- **Lesson:** validate exported LighterGlue against PyTorch on MegaDepth-1500, not on demo images.

**Derkai52/XFeat-Lightglue-TRT** — <https://github.com/Derkai52/XFeat-Lightglue-TRT> — no license,
last push 2025-07-11.

- JetPack 5.1.3, TRT 8.5.2. Separate XFeat (800×800, top-K 512, `--split_instance_norm`) and
  LighterGlue L6 engines through trtexec. FP32 by default, INT8 optional.
- **No numbers.**
- No license, so it can't be reused anyway.

**kornia/vision-rt `vrt-xfeat`** —
<https://github.com/kornia/vision-rt/blob/main/crates/vrt-xfeat/README.md> — Apache-2.0, last push
2026-10-08, active.

- TensorRT backbone-only ONNX. NMS, top-K, descriptor sampling, L2 norm and mutual-NN run as NVRTC
  CUDA kernels on one stream with one sync.
- Orin Nano MAXN_SUPER, top-K 2048, p50 per pair:
  - 640×480: 6.05 ms (2× batch-1) or 5.35 ms (batch-2 engine).
  - 736×480: 6.78 ms or 6.15 ms.
- Separate Orin Nano run, FP16, 640², K=1024: XFeat extract ×2 6.8 ms + mutual-NN 0.9 ms =
  **7.7 ms**.
- Reports XFeat inliers fall to 0% beyond about 80° rotation.
- **No LighterGlue.** Its LightGlue+ is RaCo-ALIKED only: K=1024 match 21.6 ms on Orin Nano FP16.
- **Lesson:** NMS/TopK moved out of the graph "so TensorRT can parse it". A batch-2 profile slows
  batch-1 runs by 13–16% unless sized to the camera.

**robofans/XFeat_trt** — <https://github.com/robofans/XFeat_trt> — no license, last push 2026-06-09.

- TRT C++, 1184² input. MegaDepth-1500 AUC@5: PyTorch 49.5, TRT FP32 48.5, FP16 46.4, INT8 45.9.
  Latency 16.7 / 16.1 / 16.6 ms. **Hardware not stated — unverified.**
- **Notes:**
  - "Unfold depth-to-space does not support ONNX dynamic shapes", so padding interacts with
    InstanceNorm and costs about 1 AUC point.
  - FP16/INT8 gives no speedup because the model is memory-bound.

**meyiao/xfeatc** — <https://github.com/meyiao/xfeatc> — Unlicense C++ (model Apache-2.0), last push
2024-11-28.

- ORT C++. **Moves InstanceNorm out of the graph into pre-processing.** Outputs are NHWC for faster
  post-processing. Fast exp for softmax.
- Fixed 640×640.
- In verlab #77: about 60 ms/frame on RK3562. Post-processing about 3 ms on a PC (estimated about 20
  ms on RK3562). Mutual-NN called slow.

**Other XFeat ports:**

| Repo | License | Last push | Notes |
|---|---|---|---|
| DavideCatto/XFeat-ONNX | Apache-2.0 | 2024-06-19 | Batch 1 only. C++ keypoint shift when padded. No numbers |
| Kazuhito00/XFeat-Image-Matching-ONNX-Sample | Apache-2.0 | 2024-05 | Author says ONNX on GPU gives little speedup. The interpolator is better in NumPy |
| optocycle/accelerated_features | Apache-2.0 | 2026-10-02 | Whole sparse detector (CNN+NMS+top-K+GridSample) in one graph. Fixed K with padding. Runtime threshold input. Any H, W that is a multiple of 32. Opset ≥ 16 for GridSample. Self-check vs reference |
| PranavNedunghat/XFeatTensorRT | none | 2025-02 | Orin NX. Needs LibTorch |
| udaysankar01/xfeat_cpp | Apache-2.0 | 2024-08 | libtorch, not an export |
| lrm2017/xfeatcpp | none (README says MIT) | 2025-09 | ORT 3072×2048. No LighterGlue numbers |
| lrm2017/xfeat-rknn | none | 2025-11 | RK3568, 1024 kpts, 1024×1536 matching: mutual-NN 181 ms, KNN cross-check 41 ms. Extraction time not given |
| jeffin07/XFeat-Rust | — | — | Empty README |

**adityarasam/tutorial-lighterglue-ptq** — <https://github.com/adityarasam/tutorial-lighterglue-ptq>
— Apache-2.0, last push 2026-08-10.

- LighterGlue ONNX PTQ on a "1-core ARM simulation":
  - FP32 about 508 ms.
  - Dynamic INT8 about 455 ms (success 53.1 → 51.1%).
  - Static QDQ is worse (44.4%).
- Pitfall: ORT `quant_pre_process` symbolic shape inference crashes; needs
  `skip_symbolic_shape=True`.
- **Simulated hardware, and it cites the wrong authors for XFeat** — treat as low-trust.

**EliFuzz/cv** — <https://github.com/EliFuzz/cv> — Apache-2.0, last push 2026-01-08.

- Claims an "XFeat & LighterGlue" ONNX and Core ML export (`ct.convert`, mlprogram, iOS16, FP32,
  static 1024 kpts).
- **Verified from source: the "LighterGlue" wrapper is a mutual-NN `DescriptorMatcher` (threshold
  0.7), not LighterGlue.** So no Core ML LighterGlue exists.

**AirSLAM** (<https://arxiv.org/pdf/2408.03520>) uses C++/TensorRT point+line detection with
LightGlue on Jetson AGX Orin.

- About 40 FPS VIO on Orin vs 73 FPS on RTX 4080, end-to-end only.
- **The per-module LightGlue latency was not verified.**

## Exporting adaptive depth / point pruning

**Adaptive depth (early exit):** one established implementation, fabio-sim PR #115
(<https://github.com/fabio-sim/LightGlue-ONNX/pull/115>). The owner merged it on 2026-07-29, with no
description and no reviews. Read from the commit patch:

- Export uses per-layer conditionals (`torch.cond` → ONNX `If`), one per layer after a minimum
  depth. Graph surgery then rewrites the sequential `If` nodes into nested early-exit `If` subgraphs
  that return the descriptors and the exit depth.
- Constraints enforced by the CLI:
  - opset 20.
  - ONNX must stay FP32; FP16 is left to TensorRT at inference.
  - Batch fixed to exactly one image pair.
  - Two ORT optimisers (`MatMulAddFusion`, `GatherSliceToSplitFusion`) must be disabled for
    control-flow graphs.
- **No latency, accuracy or per-EP numbers have been published for it.** I did not verify how TRT
  handles the nested `If`s.
- Its 2024 blog says adaptivity was removed earlier because the TorchDynamo exporter could not
  handle `torch.cond`.
- Older issue #27 (2023): a model exported with depth/width confidence "can't run".

**Point pruning (adaptive width):** no exported implementation found.

- Every port disables it, because the active point set changes size inside the network: noahzhy
  explicitly, fabio-sim implicitly, and the XFeat TRT repos.
- Upstream LighterGlue runs with pruning on (`width_confidence: 0.95`). Exported LighterGlue
  therefore does not exactly reproduce upstream behaviour. That is a likely partial cause of the
  accuracy gaps reported in #100 (my inference, not verified).
- Possible fixed-shape alternatives (mask pruned points in attention, or a fixed-K ladder of
  sub-graphs) are not published for LightGlue as far as I found.

**Static alternatives that are published:**

- Layer truncation + fine-tune (Raven-Z-CV: 9 → 7 layers).
- Fixed early layer: the LightGlue paper, Table 4, gives 11.9 ms at layer 5/9 vs 19.4 ms full, 512
  kpts, RTX 3080.
- Matching at a smaller K than extraction (vision-rt: extract at K=3072, match at K=1024, 1.63×
  faster).

## Cross-cutting lessons

1. Keep shapes static: fixed top-K with padding and a mask or count, not a threshold. Variable
   output sizes break batching and TRT profiles.
2. TopK is a hotspot on TRT (3840 cap, poor scaling). Chunked TopK or a custom CUDA top-K both help.
   ArgMax/argsort → topk helped in two separate projects.
3. ORT CUDA EP has been reported **slower than PyTorch** for both XFeat (PR #5) and LightGlue at
   high K (fabio-sim). The TRT EP or native TRT is needed on NVIDIA.
4. Attention: emit fused MHA explicitly, rather than relying on pattern fusion. Einsum → MatMul gave
   2.1× on CPU.
5. XFeat InstanceNorm on the input plus Unfold/depth-to-space hurt dynamic shapes and padding.
   Options seen: move InstanceNorm to pre-processing (meyiao), split it (Derkai52), or keep the
   backbone-only graph with CUDA post-processing (vision-rt).
6. XFeat FP16/INT8 gives no TRT speedup at 1184² (robofans). The model is memory-bound, and FP16
   cost about 2 AUC@5.
7. Validate on MegaDepth-1500 against PyTorch, not by eyeballing demos (#100, #77).

## Could not verify

- qdLMF FlashAttention plugin speedups (none published).
- Hardware behind robofans' 16.7 ms and Raven-Z-CV's CPU/GPU tables.
- Resolution of the 700 ms Orin NX report in #77.
- Contents of the stschake LighterGlue ONNX fork (seen only through links).
- Per-module AirSLAM LightGlue timing.
- Any OpenVINO-specific or Arm-CPU (XNNPACK/ACL/KleidiAI) LightGlue numbers. fabio-sim supports the
  OpenVINO EP but publishes no CPU figures.
- No papers on a faster LightGlue specifically (FlashAttention kernels, pruning export) were found
  beyond the LightGlue paper itself. MambaGlue (arXiv 2502.00462) is a different architecture.
