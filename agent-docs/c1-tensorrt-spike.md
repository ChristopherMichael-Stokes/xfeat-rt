# C1 spike, part U3: TensorRT install and engine builds

Run on 2026-10-24, WSL2, RTX 4070 Ti (sm_89), driver CUDA UMD 13.4, system CUDA toolkit 13.3
at `/usr/local/cuda-13.3`, uv 0.13.5. Scratch project (not in the repo):
`/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/trt/`
(scripts `mk.py`, `build.py`, `run.py`, `batch.py`, `fp16b.py`). ONNX inputs came from
`.../scratchpad/c1/onnx/` (produced by another agent).

## Findings per unknown

### U3a: `uv lock` / `uv sync` for a torch-free TensorRT project

`pyproject.toml`: `requires-python = ">=3.13"`, dependencies `tensorrt-cu13`, `cuda-python`,
`numpy`. Dev group (added later): `onnx`, `onnxconverter-common`.

| Item | Result |
| --- | --- |
| `uv lock` | works, 0.28 s, 9 packages |
| `uv sync --python 3.13` (cold cache) | works, 54 s (stubs build, 4.4 GB of libs downloaded) |
| `uv sync --python 3.14.9` in a copy (warm cache) | works, 5.7 s, identical lock; `import tensorrt` and `Builder` creation work on 3.14.9 |
| `uv lock` default interpreter | picked `CPython 3.14.0+freethreaded` (the only local 3.14); lock content identical because the lock is keyed on `requires-python` only |
| `grep -ci torch uv.lock` | 0 (also 0 after adding `onnx` and `onnxconverter-common`) |

Resolved versions: tensorrt-cu13 11.4.0.106, tensorrt-cu13-bindings 11.4.0.106,
tensorrt-cu13-libs 11.4.0.106, cuda-python 13.4.4, cuda-bindings 13.4.4, cuda-core 1.2.2,
cuda-pathfinder 1.8.4, numpy 2.5.4; dev: onnx 1.23.2, protobuf 7.36.2, ml-dtypes 0.6.0,
onnxconverter-common 1.16.0, packaging 26.3.

Stub packages:

- `tensorrt-cu13` and `tensorrt-cu13-libs` are sdist-only in the lock (17,970 B and 15,508 B).
  `uv sync` printed `Building tensorrt-cu13==11.4.0.106` and `Building tensorrt-cu13-libs==11.4.0.106`.
- The libs stub does pull the real libraries at build time: `.venv/.../tensorrt_libs` is 4.4 GB
  (`libnvinfer.so.11` 715 MB, builder resources for sm75/80/86/89/90/100/120, plus `win_*`
  resource files on Linux), `.venv` total 4.5 GB.
- The real library wheel is not in `uv.lock` (no URL or hash for it): the lock pins only the
  15 KB sdist. The download from NVIDIA's index is not covered by the lock.
- Warm cache reuse works (second venv synced in 5.7 s).
- `tensorrt-cu13-bindings` is a normal 1.2 MB wheel.

### U3b: runtime environment on WSL2

- `import tensorrt` prints 11.4.0.106; `trt.Builder(logger)` creates on the GPU
  (`[MemUsageChange] Init CUDA`); `cuda.bindings.runtime` sees 1 device,
  `NVIDIA GeForce RTX 4070 Ti`.
- No `LD_LIBRARY_PATH` or other environment fix was needed (`LD_LIBRARY_PATH` was empty).
- `libcuda.so.1` comes from `/usr/lib/wsl/lib`. `cuda-python` located `libcudart.so.13` through
  `cuda.pathfinder` with `found_via = system-search`, at `/usr/local/cuda/targets/x86_64-linux/lib/`
  (also libnvrtc 13.3.33 from `/usr/local/cuda-13.3`). The venv contains no `libcudart`.
  Whether a machine without a system CUDA toolkit works was not tested. `cuda-bindings` has an
  extra `all` that adds `cuda-toolkit[...]==13.*` wheels.

### U3c: API differences in TensorRT 11 (checked from the installed package)

- `Builder.platform_has_fast_fp16` does not exist.
- `trt.BuilderFlag` has no `FP16`, `INT8` or `BF16`. The only network creation flag is
  `STRONGLY_TYPED`; `create_network(0)` still works.
- Consequence: FP16 is selected by the ONNX tensor types (strongly typed network), not by a
  builder flag. An FP32 ONNX built with `STRONGLY_TYPED` runs in FP32. An FP16 engine needs an
  FP16-typed ONNX.
- Engine plan: `builder.build_serialized_network` returns `IHostMemory` (`nbytes`, `dtype`;
  no `len`). Inference used `execute_async_v3` with `set_tensor_address` and cuda-python
  `cudaMalloc`/`cudaMemcpy`.

### U3d: tiny graph: TopK, GridSample cubic, MaxPool

Graph made with `onnx.helper` (opset 20, no torch): `MaxPool` 5x5 stride 1 pad 2 on 64x64,
`Reshape`, `TopK(K)`, and `GridSample(mode="cubic", padding_mode="border", align_corners=0)`.
Built with `STRONGLY_TYPED`, FP32 and FP16 typed models. Both ran on the GPU with finite
outputs of the expected shapes.

| K | FP32 | FP16 |
| --- | --- | --- |
| 1024 | build ok | build ok |
| 3839 | build ok | build ok |
| 3840 | build ok | build ok |
| 3841 | fail | fail |
| 4096 | fail | fail |

Error text for K=4096 (identical for K=3841):

```text
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error (ITopKLayer
    node_of_vals: node_of_vals: K exceeds the maximum value allowed (3840). In processCheck at
    /_src/optimizer/shapeof/graphShapeAnalyzer.cpp:1066 In analyzeShapes at
    /_src/optimizer/shapeof/graphShapeAnalyzer.cpp:2826)
```

The limit is K <= 3840 inclusive (the documented "K < 3840" is off by one). The largest K that
builds in a single TopK is 3840. Cubic GridSample parsed and built without warnings in both
precisions.

### U3e / U9: real XFeat and LighterGlue ONNX

Build code (`build.py`): `create_network(STRONGLY_TYPED)`, `OnnxParser.parse_from_file`,
default `BuilderConfig`, `build_serialized_network`. FP16 ONNX made with
`onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True, ...)`.
Model: 1280x960, batch 1. All timings are single builds on a cold timing cache.

FP32 (all parse with no unsupported-op errors):

| ONNX | Build s | Engine MB | Result |
| --- | --- | --- | --- |
| xfeat_1280x960_k2048 (plain TopK) | 51.7 | 28.91 | ok, runs |
| xfeat_1280x960_k2048_chunked | 16.4 | 28.96 | ok, runs |
| xfeat_1280x960_k2048_relresize | 13.0 | 21.91 | ok |
| xfeat_1280x960_k2048_chunked_relresize | 15.7 | 22.00 | ok |
| xfeat_1280x960_k2048_relresize_s2d | 11.5 | 21.90 | ok |
| xfeat_1280x960_k2048_chunked_relresize_s2d | 11.8 | 21.95 | ok |
| xfeat_1280x960_k4096 (plain TopK) | n/a | n/a | fail: TopK K exceeds 3840 |
| xfeat_1280x960_k4096_relresize | n/a | n/a | fail: same |
| xfeat_1280x960_k4096_relresize_s2d | n/a | n/a | fail: same |
| xfeat_1280x960_k4096_chunked | 12.4 | 29.25 | ok, runs |
| xfeat_1280x960_k4096_chunked_relresize | 12.2 | 22.04 | ok |
| xfeat_1280x960_k4096_chunked_relresize_s2d | 12.0 | 22.00 | ok, runs |
| xfeat_backbone_1280x960 | 28.8 | 21.98 | ok, runs |
| xfeat_backbone_1280x960_pixelops | 11.4 | 21.80 | ok |
| xfeat_backbone_1280x960_s2d | 20.6 | 21.80 | ok |
| lighterglue_k2048 | 8.0 | 5.32 | ok, runs |
| lighterglue_k4096 | 8.8 | 5.34 | ok, runs |
| stage_*: nms, topk_plain_k2048, topk_chunked_k2048/k4096, desc_k2048, rel_candidates_n16384, rel_dense_resize, rel_dense_gridsample | 0.5 to 1.9 | 0.02 to 9.84 | ok |
| stage_topk_plain_k4096 | n/a | n/a | fail: TopK K exceeds 3840 |
| probe_sdpa_opset20 | 3.0 | 0.06 | ok |
| probe_sdpa_opset23 (ONNX `Attention` op) | n/a | n/a | process segfault (exit -11, SIGSEGV) during build; the build log shows `Unused Input: q`, `Unused Input: k` warnings before the crash |

K=4096 failure text (plain TopK models):

```text
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error (ITopKLayer
    node_topk__1: node_topk__1: K exceeds the maximum value allowed (3840). In processCheck at
    /_src/optimizer/shapeof/graphShapeAnalyzer.cpp:1066 In analyzeShapes at
    /_src/optimizer/shapeof/graphShapeAnalyzer.cpp:2826)
```

Chunked TopK (three 2048-wide chunks plus merge, per the export metadata) builds at K=4096 and
runs. Largest plain K that builds is 3840 (tiny graph); no real XFeat model at K=3840 was
exported, so that figure comes from the tiny graph only.

FP16 (ONNX converted with onnxconverter-common):

| Model | Converter setting | Result |
| --- | --- | --- |
| Full XFeat models (k2048, k2048_chunked, k2048_relresize, k4096, k4096_chunked) | default op block list | fail at parse, see below |
| xfeat_1280x960_k2048, k2048_chunked, k2048_relresize, k4096_chunked | op block list extended with `Div, Mod, Floor, TopK, Gather, Greater, Where, Equal, And, Clip, Sub, Add, Mul, Expand, Slice, Concat, MaxPool` | build ok, 22.3 to 23.6 s, 46.2 / 46.2 / 41.3 / 46.3 MB, runs |
| xfeat_backbone_1280x960 | default | ok, 45.5 s, 19.10 MB (FP32: 21.98 MB), runs |
| backbone pixelops / s2d, k2048 and k4096 chunked_relresize (with and without s2d), k2048_relresize_s2d | extended block list | ok, 18.9 to 52.5 s, 41.2 to 41.5 MB |
| lighterglue_k2048, lighterglue_k4096 | default; two extended block lists tried on k2048 | fail at parse |
| stage_* models, probe_sdpa_opset20/23 | default | ok |

FP16 failure texts:

```text
[TRT] [E] ITensor::getDimensions: Error Code 4: API Usage Error (node_Div_544:
    ElementWiseOperation DIV must have same input types. But they are of types Float and Half. In
    validateTypes at /_src/optimizer/common/nodes/elementWiseNode.cpp:59)
[TRT] [E] ITensor::getDimensions: Error Code 4: API Usage Error (node_gt: ElementWiseOperation
    GREATER must have same input types. But they are of types Half and Float. In validateTypes at
    /_src/optimizer/common/nodes/elementWiseNode.cpp:59)
```

Cause found for the XFeat `Div`: the converter turns the constant divisor (`1280`, from
`index // width`) into FP16 but leaves the `Cast(to=1)` of the integer index in FP32. For
LighterGlue, `Greater` consumes `mscores0` (a graph output) through a converter-inserted
`Cast(to=10)` on input 0 and `Cast(to=1)` on input 1. These are converter output defects, not
TensorRT parse limits. No FP16 LighterGlue engine was produced; no manual fix was attempted.

Observations on FP16:

- The block list keeps the post-processing and elementwise ops in FP32 and gives FP16
  engines larger than FP32 (41 to 46 MB versus 22 to 29 MB). The cause was not investigated.
- The `stage_nms` FP16 engine is 17.23 MB versus 0.02 MB in FP32 (cause not investigated).
- Converter warnings: `-inf` truncated to `-10000.0`; `9.99e-13` and `2.57e-08` truncated to
  `1e-07`.
- Precision accuracy was not checked (out of scope).

Execution check with `run.py` on random inputs (uniform [0,1) image; zero-filled
non-float inputs):

| Engine | Outputs | Finite | Valid count |
| --- | --- | --- | --- |
| xfeat k2048 FP32, k2048_chunked FP32 | kpts [1,2048,2] f32, scores [1,2048], desc [1,2048,64], valid [1,2048] bool | yes | 2048 |
| xfeat k2048 FP16 (block list) | kpts f32, scores and desc f16 | yes | 2048 |
| xfeat k4096_chunked FP32, FP16 (block list), k4096_chunked_relresize_s2d FP32 | same with 4096 | yes | 4096 |
| xfeat_backbone FP16 | feats [1,64,120,160], prob [1,1,960,1280], reliability [1,1,120,160] f32 | yes | n/a |
| lighterglue k2048 / k4096 FP32 | m0 int64 [1,K], mscores0 f32 [1,K] | yes | m0 all -1 (all-invalid zero inputs, so uninformative) |

Plain and chunked K=2048 produced the same output ranges (kpts 0 to 1278; scores
0.00106 to 0.0205). Numerical cross-checks against the reference were not done.

## Commands

```bash
cd <scratch>/trt                     # pyproject.toml as described above
uv lock
uv sync --python 3.13
uv add --dev onnx onnxconverter-common
uv run python t2.py                  # import, Builder, cuda-python device query
uv run python mk.py w/t_fp32_3840.onnx 3840 fp32   # tiny graph, args: out K dtype
uv run python build.py w/t_fp32_3840.onnx w/t_fp32_3840.plan
uv run python batch.py <onnx basenames>             # FP32 + FP16 builds, log in logs/trt.log
uv run python fp16b.py <basename>                   # FP16 with extended block list
uv run python run.py plans/<name>.plan              # deserialize, run once, print outputs
```

## Summary of answers

| Question | Answer |
| --- | --- |
| Does a torch-free uv project lock and sync TensorRT 11.4 cu13 + cuda-python on 3.13 and 3.14? | Yes, no torch in the lock; 4.5 GB venv; libs are fetched by the stub build and are not hashed in `uv.lock` |
| Does it work on WSL2 without environment fixes? | Yes; `libcudart` came from the system CUDA 13.3 toolkit |
| Do cubic GridSample, TopK and MaxPool build in FP32 and FP16? | Yes |
| TopK limit | K <= 3840; K=3841 and 4096 fail with the error above |
| Do real XFeat and LighterGlue parse and build? | FP32: all do, except plain TopK at K=4096; chunked K=4096 builds. FP16: XFeat builds only with a custom converter block list; LighterGlue FP16 failed with onnxconverter-common |
| ONNX `Attention` (opset 23) | segfault during TensorRT 11.4 FP32 build |
