# Inference runtimes per device (as of 2026-10-23)

Method: PyPI JSON API (<https://pypi.org/pypi/><pkg>/json) for versions, dates, wheel tags; wheel
contents inspected by downloading; vendor docs via fetch. "Unverified" marks what I could not
confirm.

## 1. Wheel matrix (PyPI, verified)

| Package | Latest (date) | cp313 | cp314 | Platforms / notes |
|---|---|---|---|---|
| onnxruntime | 1.31.0 (2026-10-09) | yes | yes (+cp314t) | manylinux_2_28 x86_64 and aarch64, macOS 14 arm64 only (no macOS x86_64), win. 21-24 MB |
| onnxruntime-gpu | 1.31.0 (2026-10-09) | yes | yes (+t) | manylinux_2_28 x86_64 (256 MB), manylinux_2_34 aarch64 (212 MB), win. Requires `nvidia-*~=13.0` via extras `[cuda,cudnn]` => CUDA 13 build |
| onnxruntime-openvino (legacy) | 1.24.1 (2026-02-26) | yes | NO | linux x86_64 + win only; deprecated by Intel in favor of plugin EP |
| onnxruntime-ep-openvino (plugin) | 1.8.0 (2026-10-16) | py3-none | py3-none | linux x86_64 + win only; does NOT bundle ORT, needs `onnxruntime>=1.23`; binary-only |
| openvino | 2026.4.1 (2026-10-01) | yes | yes (+t) | manylinux_2_28 x86_64 (59 MB), manylinux_2_35 aarch64 (30 MB), macOS arm64 (33 MB), win |
| tensorrt / tensorrt-cu13 / -cu12 | 11.4.0.106 (2026-10-08) | via bindings | via bindings | metas; default `tensorrt` = cu13 |
| tensorrt_cu13_bindings | 11.4.0.106 | yes | yes | manylinux_2_28 x86_64, manylinux_2_35 aarch64, win_amd64, win_arm64 (1.2 MB) |
| tensorrt_cu12_bindings | 11.4.0.106 | yes | yes | x86_64 + win only (no aarch64) |
| tensorrt_cu13_libs / cu12_libs | 11.4.0.106 | sdist only | sdist only | 15 KB sdist using `wheel_stub` that pulls the real wheel from <https://pypi.nvidia.com/> at build/install time (read pyproject.toml myself) |
| coremltools | 9.0 (2025-11-10) | yes | NO | macOS arm64/x86_64 and linux x86_64 (conversion only; no runtime on Linux). 2.8 MB. Newest release is ~11 months old |
| mlx | 0.32.4 (2026-10-14) | yes | yes | macOS 14/15/26 arm64, linux x86_64/aarch64 (CPU/CUDA variants via extra packages; unverified), win |
| ai-edge-litert (LiteRT) | 2.3.0 (2026-10-06) | yes | yes | manylinux_2_27 aarch64 (14 MB), x86_64 (22 MB), macOS arm64, win |
| ncnn | 1.0.20261020 (2026-10-20) | yes | yes | manylinux_2_24/2_28 aarch64, armv7l, x86_64, macOS arm64, many more (5-11 MB) |
| pnnx (PyTorch->ncnn converter) | 20261020 | py3-none | py3-none | linux aarch64/x86_64, macOS, win |
| tflite-runtime | 2.14.0 (2023) | dead | dead | replaced by ai-edge-litert |
| onnxruntime-coreml | 1.13.1 (2023) | dead | dead | CoreML EP is inside the plain macOS onnxruntime wheel |
| onnxruntime-trt-rtx | 1.23.2 (2025-12) | win_amd64 only | no | superseded by plugin `onnxruntime-ep-nv-tensorrt-rtx` (RTX only; not for datacenter GPUs per its docs) |

Python 3.14 blockers: coremltools (no cp314 wheel), legacy onnxruntime-openvino. Everything else
ships cp314.

## 2. Package exclusivity and uv

- `onnxruntime`, `onnxruntime-gpu`, legacy `onnxruntime-openvino` all install the same
  `onnxruntime/` module; installing two corrupts it. NVIDIA's TensorRT-RTX EP docs state this
  explicitly ("uninstall onnxruntime-gpu first, it ships a conflicting onnxruntime module")
  <https://nvidia.github.io/TensorRT-RTX-EP-ABI/installation.html>
- Plugin EPs (onnxruntime-ep-openvino, ep-nv-tensorrt-rtx, CUDA plugin EP) layer on the plain
  `onnxruntime` wheel: no module clash. Plugin direction is the current ORT design (v1.30/1.31
  notes: CoreML, CUDA, WebGPU plugin EPs;
  <https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0>).
- uv: extras that must never co-install are declared with
  `[tool.uv] conflicts = [[{extra="cpu"},{extra="gpu"}]]` (documented for extras and groups,
  <https://docs.astral.sh/uv/concepts/projects/config/#conflicting-dependencies>). `environments` /
  `required-environments` limit/require platforms in the lock. I did not find a `package`-level
  conflict form on that page. Markers work for hard splits, e.g.
  `onnxruntime-gpu; sys_platform=='linux' and platform_machine=='x86_64'` and
  `onnxruntime; sys_platform=='darwin'`; but a marker cannot express "CPU only vs GPU" on the same
  x86_64 Linux box, which needs an extra.
- Workspaces: one lockfile, one `requires-python`, "not suited for members with conflicting
  requirements"; advice is independent projects linked through `tool.uv.sources`
  (<https://docs.astral.sh/uv/concepts/projects/workspaces/>). Recommendation: one workspace with a
  core package (no runtime dep) plus one member/extra per backend, conflicts declared; a separate
  (non-workspace) project only for Jetson, which cannot share a lock with x86 CUDA (different CUDA
  userland, see section 6).
- I did not run `uv lock` against any of this (read-only task): the sdist-only TensorRT libs stub
  and its interaction with uv's universal resolver is unverified.

## 3. CPU

### x86 CPU

1. ONNX Runtime CPU (1.31): wheel for cp313/cp314, 24 MB, new x86 FP16 LayerNorm/RMSNorm kernels in
   1.30 (<https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0>). MLAS has AVX2/AVX512
   kernels. Safe default.
2. OpenVINO native (`openvino` 2026.4.1, cp313+cp314 x86_64, 59 MB): usually fastest on Intel CPUs
   (AMX/AVX512-BF16/INT8 VNNI; not benchmarked here, no measurement -> treat as expected, not
   verified). Runs on AMD too (AVX2/AVX512). Reads ONNX directly. Zero-copy: wrap numpy via
   `ov.Tensor(array, shared_memory=True)` (my recollection, not verified in this session).
3. ORT + OpenVINO plugin EP: Python 3.14-compatible (py3-none), layers on plain `onnxruntime`. ORT
   OpenVINO docs: CPU precision FP32 only, FP16 for GPU/NPU, INT8 via NNCF
   (<https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html>). Extra layer
   buys nothing over native OpenVINO unless you want one API. Not recommended: legacy
   onnxruntime-openvino (no cp314, deprecated).

### Arm CPU (Raspberry Pi class, aarch64 Linux)

1. ONNX Runtime CPU: aarch64 manylinux_2_28 cp313/cp314 (21 MB). 1.30/1.31 add Arm64 NEON/SVE
   kernels and KleidiAI (SVE2.1/SME2 FP16 GEMM) but those need SVE2/SME2 silicon; Pi 5 (Cortex-A76)
   lacks them, so Pi benefit is unverified (<https://onnxruntime.ai/docs/performance/kleidiai/>).
   ORT ACL EP is deprecated/slated for removal (1.31 notes).
2. OpenVINO native on Arm: aarch64 wheel exists for cp313/cp314 (manylinux_2_35, so needs glibc >=
   2.35: Ubuntu 22.04+/Debian 12 bookworm has 2.36, fine). Third-party sources say the Arm CPU
   plugin does FP16 on ARMv8.2+ (Pi 5) and run YOLO on Pi 5
   (<https://www.raspberrypi.com/news/run-ultralytics-yolo-on-raspberry-pi-with-openvino/>); I could
   not fetch official OpenVINO CPU docs (page too long for the fetcher) so Arm FP16 plugin behavior
   is unverified. Pi 4 (A72, no fp16 arithmetic) gets no FP16 gain.
3. LiteRT (ai-edge-litert 2.3.0): aarch64 cp313/cp314 wheel, XNNPACK default CPU delegate,
   Google/Raspberry Pi docs show Pi 5 use
   (<https://www.raspberrypi.com/news/get-started-with-litert-on-raspberry-pi/>). Needs
   PyTorch->tflite conversion (litert-torch converter, torch needed at export time only). XFeat's
   interpolate/grid ops and LighterGlue attention conversion risk is unverified.
4. ncnn 1.0.20261020: tiny wheels, hand-tuned NEON/fp16 arm kernels, converters via pnnx or ONNX.
   Conversion of transformers (LighterGlue) is the risk; unverified. Rank: ORT CPU (lowest friction,
   same ONNX as other targets) ~ OpenVINO (probable speed upside, verify on-device) > LiteRT > ncnn.
   Decide by on-device benchmark on Pi 5; I have no measurements.

CPU accuracy: FP16/INT8 knobs: ORT FP16 via model conversion (onnxconverter-common / ORT transformer
optimizer); INT8 via ORT quantization or NNCF. No XFeat-specific accuracy data was gathered;
keypoint heatmap/descriptor quantization accuracy needs measuring.

## 4. NVIDIA CUDA (desktop/server, x86_64 + SBSA aarch64)

1. TensorRT Python API (tensorrt-cu13 11.4.0.106, bindings cp313/cp314 x86_64 + aarch64 SBSA):
   fastest option, FP16/INT8/FP8 builder flags, engine serialization, runtime-only
   `tensorrt-lean`/`-dispatch` variants for deployment
   (<https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html>:
   supports Python 3.10-3.14, CUDA 13 default, `-cu12` suffix for CUDA 12; pip wheels exclude
   trtexec and headers). Zero-copy: engine takes raw device pointers (`context.set_tensor_address`)
   so pair with `cuda-python` 13.4.4 (pure py3-none wheel) or any DLPack-capable buffer owner; no
   PyTorch needed (standard TRT pattern, not tested here). Engines are per-GPU/per-TRT-version, so
   build on target or ship ONNX + cache. Install footprint: libs from pypi.nvidia.com via sdist stub
   (size not measured), plus cuda-python/nvidia-cuda-runtime. Caveat: `tensorrt_cu13_libs` is an
   sdist-only 15 KB stub (build-time download from <https://pypi.nvidia.com/>); fragile with uv
   lockfiles, mirrors, offline CI. Unverified how uv locks it.
2. ONNX Runtime CUDA EP (onnxruntime-gpu 1.31, CUDA 13 default since 1.27; CUDA 12 builds
   deprecated/renamed;
   <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>): 256 MB wheel +
   `[cuda,cudnn]` (cuDNN ~540 MB per nvidia-cudnn-cu13 wheel). Zero-copy via IOBinding / `OrtValue`
   on device, `user_compute_stream`, `prefer_nhwc`, `cudnn_conv_use_max_workspace` (important for
   FP16). `preload_dlls()` handles pip-installed NVIDIA libs. Easy, same ONNX as CPU, but slower
   than native TRT for a fixed-shape CNN.
3. ORT TensorRT EP: in the x86_64 1.31 wheel only as a 1 MB `libonnxruntime_providers_tensorrt.so`
   shim; I verified the aarch64 wheel has CUDA provider only, NO TensorRT provider. Needs libnvinfer
   on the path (from tensorrt libs). Docs only describe TRT 10.x and say "only CUDA 12 packages"
   (stale vs. 1.31 CUDA 13), so TRT11/CUDA13 compatibility of this EP is unverified. Features:
   `trt_fp16_enable`, `trt_int8_enable`, engine/timing cache, EPContext embedded engine (SD UNet
   session create 384s uncached -> 1.9s embedded)
   (<https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html>).
4. TensorRT-RTX EP plugin (`onnxruntime-ep-nv-tensorrt-rtx`): RTX Ampere+ only per docs; not for
   server/data-center SKUs as written. Optional later. Rank: TensorRT Python (speed) > ORT CUDA EP
   (simplicity/fallback, shares ONNX path) > ORT TRT EP (middle, version-matrix risk). Evidence for
   ranking is architectural (TRT does graph fusion + precision autotuning); no benchmark run.
   FP16/INT8: TRT FP16 mostly safe for CNN; LighterGlue attention softmax/LayerNorm may need FP32
   layers (typical, unverified for this model). INT8 needs calibration; descriptors/keypoint
   heatmaps are the sensitive part (unmeasured).

## 5. Apple Silicon

1. Core ML via coremltools 9.0: only convertible from PyTorch (ONNX frontend was removed long ago; I
   did not re-verify), no cp314 wheel (cp313 max; release notes say "Added Python 3.13 support",
   <https://github.com/apple/coremltools/releases>) and last release Nov 2025. Conversion is an
   offline export step; runtime Core ML inference from Python still goes through coremltools'
   `MLModel.predict` (numpy in/out => copies; CVPixelBuffer/MLMultiArray zero-copy needs Swift/ObjC,
   not Python). Best chance of ANE use (`compute_units=CPU_AND_NE`, ML Program, FP16 default
   precision). Constraint: conversion needs torch (export machine only), so the runtime venv can be
   torch-free but exporters cannot be on 3.14.
2. ONNX Runtime CoreML EP (in the stock macOS arm64 wheel, cp313/cp314, 21 MB):
   `ModelFormat=MLProgram`, `MLComputeUnits=CPUAndNeuralEngine`, `RequireStaticInputShapes=1` (docs:
   default NeuralNetwork format, ANE not guaranteed for whole model, op limitations such as constant
   Slice params, 2D pools only)
   (<https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html>). 1.30/1.31 add
   CoreML plugin-EP-factory + `PREFER_NPU` selection. No IOBinding documentation for CoreML.
   Partitioning means CPU fallback and copies at boundaries for unsupported ops (transformer part
   risk).
3. MLX 0.32.4 (cp313+cp314 macOS arm64): Metal GPU only, no ANE; unified memory gives cheap
   numpy/`mx.array` handoff (stated design; not tested). Requires re-implementing XFeat and
   LighterGlue in MLX or an ONNX-to-MLX path (none verified) => hand-written model code, large
   scope. Rank: ORT CoreML EP (cp314-ready, single ONNX path; first to try) vs Core ML via
   coremltools (best ANE odds, but py<=3.13 for the converter, extra export pipeline). Rank 1 for
   ship speed: ORT CoreML EP with MLProgram; rank 1 for raw ANE speed: coremltools; MLX third.
   Unresolved without a Mac benchmark. FP16 on ANE is mandatory (ANE is FP16 only; known minor
   accuracy shifts, unmeasured).

## 6. Jetson (later) - verified facts

- Current JetPack: 7.2.1 (Jetson Linux 39.2.1, 2026-08-11): CUDA 13.2.2, cuDNN 9.20.0, TensorRT
  10.16.2, Ubuntu 24.04, kernel 6.8, Container Toolkit 1.19 bundled; hardware Thor (T5000/T4000) and
  Orin family (<https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.1.md>). JetPack
  7.2 (2026-06-02): CUDA 13.2.1, TRT 10.16.2
  (<https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.md>). 7.1 (Linux 38.4): CUDA
  13.0.0, TRT 10.13.3.9 (<https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.1>).
  Orin joined JetPack 7 only at 7.2. JetPack 6.x (CUDA 12.x, Orin) is the older line.
- Thor is SBSA-aligned (JetPack 7 notes): the standard Arm SBSA CUDA 13 stack, so aarch64
  `manylinux_2_34/2_35` PyPI wheels (onnxruntime-gpu aarch64, tensorrt_cu13_bindings aarch64,
  nvidia-cudnn-cu13 aarch64) are plausible but I could not verify that they work on Thor/Orin iGPU
  (sm_110 for Thor; ORT CUDA provider kernels compiled for sm_110 unverified; a community repo
  claims no prebuilt aarch64 CUDA13 ORT GPU wheel suits Jetson:
  <https://github.com/straga/jetson-jp7-onnxruntime>).
- TensorRT version skew: PyPI is 11.4 while JetPack ships 10.16.2. Mixing pip TRT with system TRT is
  a risk. Unverified which to use; likely use the JetPack-provided libs/container.
- Containers: NVIDIA staff said no l4t-jetpack JetPack 7.x container exists and none planned;
  recommended CUDA base images and NGC images (nvcr.io/nvidia/pytorch, nvcr.io/nvidia/tensorrt at
  JetPack 7.0 time); Thor no longer needs `-igpu` tags
  (<https://forums.developer.nvidia.com/t/docker-container-jetpack-7-x/379812>; JetPack 7.0 notes
  via
  <https://forums.developer.nvidia.com/t/jetpack-7-0-jetson-linux-38-2-for-nvidia-jetson-thor-is-now-live/343127>).
  Runtime flow: NVIDIA Container Toolkit; CDI spec at /var/run/cdi/nvidia.yaml auto-generated by
  `nvidia-cdi-refresh` since toolkit 1.18; Docker shown with `--runtime=nvidia`
  (<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html>). The
  CDI page does not mention Jetson/Tegra, so Jetson-specific CDI usage is unverified. Forum:
  `default-runtime: nvidia` is not set in daemon.json by default after USB flash.
- Jetson wheels: Jetson AI Lab index (pypi.jetson-ai-lab.io) is described by NVIDIA skill text as
  canonical for onnxruntime-gpu/PyTorch/vLLM; only jp6 cu126/cu128 subtrees were visible; JP7/cu130
  and TensorRT presence unverified. NVIDIA's skill mirrors are secondhand.
- Not verified: which Python the JetPack 7.x base images ship (Ubuntu 24.04 system Python is 3.12;
  3.13/3.14 would come from uv-managed interpreters); whether the TensorRT 10.16.2 Python bindings
  in the JetPack image support 3.13/3.14.
- Implication: Jetson probably needs its own uv project (container with uv + managed Python
  3.13/3.14 + system TensorRT), not the x86 lock.

## 7. Recommended per-device ranking (summary)

| Device | 1 | 2 | 3 |
|---|---|---|---|
| x86 CPU | OpenVINO native (verify speed on-box) / ORT CPU as baseline | ORT CPU | ORT+OV plugin EP |
| Arm CPU (Pi) | ORT CPU (verify) / OpenVINO aarch64 | LiteRT | ncnn |
| NVIDIA CUDA | TensorRT Python (cu13) | ORT CUDA EP | ORT TRT EP |
| Apple | ORT CoreML EP (cp314-ready) / coremltools (best ANE, cp313 only) | the other | MLX |
| Jetson | JetPack-provided TensorRT in NVIDIA CUDA/NGC base container | ORT CUDA EP if a sm_110 aarch64 wheel proves out | - |

## 8. Could not verify

- Any benchmark or accuracy number (none run).
- OpenVINO Arm CPU plugin official FP16/INT8 docs (docs fetch truncated); OpenVINO zero-copy API
  details.
- Official LiteRT/XNNPACK FP16 docs; ncnn/LiteRT conversion success for XFeat and LighterGlue.
- uv lock behavior with sdist-only tensorrt libs stub and `conflicts`; run `uv lock` to confirm.
- ORT TensorRT EP compatibility with TensorRT 11 / CUDA 13; ORT 1.31 release-note packaging details
  (the notes are LLM-generated per the 1.27 page).
- Core ML zero-copy from Python; ONNX frontend status in coremltools 9.0; MLX Linux extras.
- Jetson items listed in section 6; official Jetson CDI instructions.
