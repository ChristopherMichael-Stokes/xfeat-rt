# C1 spike: in-graph XFeat post-processing and static LighterGlue export (U1, U9 input)

Verdict: **go** for in-graph post-processing (NMS, TopK, descriptor sampling) and for the padded
static LighterGlue graph. Both export with the dynamo exporter at opset 20 and match the reference
on ORT CPU to within one tie-broken keypoint per image. Two graph rewrites are required for speed:
dense reliability sampling through `Resize` instead of a dense `GridSample`, and `SpaceToDepth` /
`DepthToSpace` instead of upstream's 6-D unfold/permute. TensorRT build of these files (U3, the
TensorRT half of U9) is not covered here.

## Setup

- torch 2.14.1+cu130, onnx 1.23.2, onnxscript 0.7.2 (both overlaid with `uv run --frozen --with`,
  the reference lock is unchanged), kornia 0.8.3, upstream XFeat `e92685f`.
- ORT 1.31.0 CPU EP in a torch-free scratch venv. AMD Ryzen 7 5800X3D (8 cores, 16 threads), WSL2.
- Reference: `ReferencePipeline`, `pixel_centre`, pruning off, CPU FP32, detection threshold
  0.05, match threshold 0.1.
- Inputs: the first 5 pairs of MegaDepth-1500 `pairs_calibrated.txt` (scene 0015 x1, 0022 x4),
  RGB channel mean / 255, bilinear resize to exactly 1280x960 (aspect not preserved), the same
  array fed to the reference and to ORT.
- Timing: 5 warm-up runs, then 20 to 50 runs, p50 of `perf_counter_ns` around `session.run`.
  **The box was shared**: load average 8 to 18 on 16 threads during every timing run, from
  processes outside this spike. Absolute wall times are inflated and vary up to ±20 % between
  runs; ratios inside one interleaved run and ORT per-op profiles are the reliable numbers.

## Graphs

XFeat: `image[1,1,960,1280]` gray in [0,1] -> backbone -> softmax heatmap (8x8 depth-to-space)
-> peaks = `(prob == MaxPool5x5(prob)) & (prob > 0.05)` -> score map `where(peaks, prob x
reliability_full, -1)` -> TopK(K) -> integer `%` and floor-div to (x, y) -> bicubic `GridSample`
of the L2-normalised descriptor map at the pixel-centre grid -> L2 norm. Outputs `kpts[1,K,2]`
float (x, y px), `scores[1,K]` (0 where invalid), `desc[1,K,64]`, `valid[1,K]` bool. Precedent:
optocycle/accelerated_features `deployment/export_xfeat_onnx.py` at
`f8baecb4d1e5e6a78481541097f11b458be21fb4` (same dense peak map, dense score map, TopK,
GridSample), read from source.

- `pixel_centre` makes upstream's `nearest` sample of the full-res heatmap the pixel itself, so the
  graph uses `prob` directly.
- Reliability at every pixel, two exact forms (both checked equal to the reference):
  - `gridsample`: dense bilinear `GridSample` on a constant 960x1280 grid (optocycle). Adds a
    9.8 MB initializer.
  - `resize` (`_relresize`): 1-px zero pad, bilinear x8 `Resize` (half-pixel), crop. The zero pad
    reproduces `GridSample`'s zeros padding at the border.
- `_s2d`: `_unfold2d(x, 8)` replaced by `pixel_unshuffle` (exported through a
  `custom_translation_table` to ONNX `SpaceToDepth`), heatmap reshuffle by `pixel_shuffle`
  (exports as `DepthToSpace`). Both checked bit-identical to upstream in torch. Precedent: BCDL
  XFeat port (pixel_unshuffle for `_unfold2d`, research 02). Without the translation table
  `pixel_unshuffle` lowers to Reshape/Transpose/Reshape.
- Chunked top-K (`_chunked`), flag `ChunkedTopK(chunk_size=65536, max_k_per_op=2048)`:
  - Stage 1 is fabio-sim/LightGlue-ONNX `lightglue_dynamo/models/raco.py::_chunked_topk` at
    `d12b4ba1632f558234e3f084e1f3d8bdf9147890`, read from source: pad with -inf, TopK in each
    disjoint chunk, TopK over the union. Exact, but both of its TopK nodes use the full K, so on
    its own it does not get under the TensorRT cap for K = 4096.
  - Stage 2 for K > `max_k_per_op`: K/2048 passes of TopK(2048) over the union, each pass setting
    the previous winners to -inf with `ScatterElements`, then concatenated. **No published
    precedent was found for this pass split.** It is exact whenever no 65536-pixel chunk holds more
    than 2048 of the global top-K.
  - K = 4096 chunked graph has TopK K values [2048, 2048, 2048]. K = 2048 chunked has
    [2048, 2048].
- Deviation from upstream: upstream treats a real detection at pixel (0,0) as padding (score -1).
  The graph keeps it. This did not occur on the 10 images.

LighterGlue: inputs `kpts{0,1}[1,K,2]` f32, `desc{0,1}[1,K,64]` f32, `valid{0,1}[1,K]` bool,
`image_size{0,1}[1,2]` f32 (w, h). Outputs `m0[1,K]` int64 (-1 = unmatched), `mscores0[1,K]` f32.
The kornia 0.8.3 CPU path is rewritten module by module on the loaded kornia weights: self-attention,
cross-attention with one shared similarity and two softmaxes, final assignment and mutual-NN filter.
Attention is explicit `MatMul`/`Softmax` with an additive key mask of -30000, which stays finite in
FP16 (no NaN rows) and underflows to exactly 0 after `exp`. Log-assignment and the max/argmax run
masked, and dustbins are not built (`filter_matches` never reads them). Pruning and early exit are
off. Upstream's per-layer pruning code is not in the graph.

## Export (U1)

| Graph | Result | Notes |
| ----- | ------ | ----- |
| XFeat, plain TopK, K = 2048 / 4096 | ok | 1 TopK node (K = 2048 / 4096) |
| XFeat, chunked, K = 2048 / 4096 | ok | TopK K ≤ 2048 in every node |
| XFeat `_relresize`, `_relresize_s2d`, every K and TopK mode | ok | 2.9 MB vs 12.7 MB |
| Backbone only (`_s2d` and upstream layout) | ok | fallback graph |
| LighterGlue, K = 2048 / 4096 | ok | 5.4 MB, 159 MatMul, 24 Softmax, `Gelu`, `LayerNormalization`, ArgMax |
| SDPA probe, opset 20 | ok | Reshape/Transpose/Mul/MatMul/Softmax |
| SDPA probe, opset 23 | ok | one ONNX `Attention` node; ORT 1.31 CPU runs it, max abs diff 2.4e-7 vs opset 20 |

- Nothing failed to export. Upstream `Tensor.unfold`, MaxPool equality, bicubic `GridSample`, TopK,
  ScatterElements and the bool masks all go through dynamo at opset 20 with no rewrite needed for
  correctness.
- Research 02 found no torchlib lowering of SDPA to ONNX `Attention`. torch 2.14.1 does emit
  `Attention` at opset 23 (unmasked probe only; the masked case is untested). The LighterGlue graphs
  use explicit attention at opset 20.
- LighterGlue `logsigmoid` exports as `Log(Sigmoid)`: -inf for very negative logits. This is a
  possible FP16 issue, untested.

## Accuracy vs reference (ORT CPU, 10 images, 5 pairs)

| Graph family (all variants identical) | K | Recall@1px min | Precision@1px min | Kpt mismatches / 10 images | Desc cosine min | Score abs diff max |
| --- | --- | --- | --- | --- | --- | --- |
| XFeat, every variant | 2048 | 0.99951 | 0.99951 | 1 | 0.9999993 | 7.6e-5 |
| XFeat, every variant | 4096 | 0.99976 | 0.99976 | 1 | 0.9999988 | 7.6e-5 |

- Each mismatch is one tie at the K-th position: pair3_0 at K = 2048 (ref 0.11951013 vs ORT
  0.11951215), and pair2_1 at K = 4096 (0.06183460 vs 0.06183922). FP32 reassociation reorders the
  last slot. Every other keypoint is at the same integer pixel.
- The chunked graphs give the same keypoint sets as plain TopK on 10/10 images at both K. The
  stage-2 exactness condition held on every image.
- pair4_0 has only 3607 NMS candidates at K = 4096, so the valid-mask path (fewer than K) is
  exercised.
- The torch versions of every static module match the reference exactly (identical sets, Jaccard
  1.0) before export.

| LighterGlue | K | Jaccard, ORT LG on reference features (min of 5) | Jaccard, ORT XFeat -> ORT LG vs reference, by coordinates (min of 5) | Matches per pair |
| --- | --- | --- | --- | --- |
| `lighterglue_k2048` | 2048 | 1.0000 | 0.9956 | 227 to 615 |
| `lighterglue_k4096` | 4096 | 1.0000 | 0.9975 | 500 to 1204 |

The end-to-end loss below 1.0 traces to the one tie-swapped keypoint (pair3 at K = 2048, pair2 at
K = 4096). All other pairs are 1.0.

## Timing (ORT CPU, 1280x960, batch 1)

### Whole graphs

Interleaved runs, pair0_0, ms. Two thread settings were measured: `intra_op_num_threads` = 0
(ORT default) and 8. The 8-thread numbers held up better under the background load.

| Graph | p50 default threads | p50 8 threads |
| --- | --- | --- |
| Backbone, upstream layout | 114.7 | 93.2 |
| Backbone `_s2d` | 78.9 | 50.1 |
| XFeat K = 2048, upstream layout + dense GridSample (first export) | 195.1 | 153.6 |
| XFeat K = 2048 `_relresize_s2d` | 87.2 | 58.3 |
| XFeat K = 4096 `_chunked_relresize_s2d` | 93.3 | 67.4 |
| LighterGlue K = 2048 | 180.6 | 170.5 |
| LighterGlue K = 4096 (10 runs) | 664.6 | 737.3 |

### ORT per-op kernel time inside the whole graph

From the ORT profiler, ms per run, 20 to 25 runs.

| Op | K = 2048, upstream layout, dense GridSample | K = 2048 `_relresize` | K = 2048 `_relresize_s2d` | K = 4096 `_chunked_relresize_s2d` |
| --- | --- | --- | --- | --- |
| GridSample, dense reliability | 75.4 | absent | absent | absent |
| Transpose (6-D unfold + heatmap permute) | 61.8 | 51.4 (unfold alone 43.0) | ≈0 | ≈0 |
| Conv (all 27) | 25.3 | 18.8 | 17.8 | 18.2 |
| Softmax (65-channel keypoint logits) | 19.6 | 14.5 | 13.6 | 14.1 |
| Resize (backbone, plus reliability x8 where present) | below top 12 | 3.1 | 2.9 | 2.9 |
| MaxPool 5x5 (NMS) | 2.5 | 1.7 | 1.7 | 1.7 |
| TopK | 1.5 | 1.3 | 1.3 | 1.6 |
| GridSample, bicubic descriptors | 0.9 | 0.7 | 0.8 | 1.4 |
| SpaceToDepth + DepthToSpace | absent | absent | 2.1 | 2.3 |
| Sum of all kernels | 203.6 | not summed | 50.7 | 53.2 |

### Stand-alone stage graphs

p50 ms, default threads, inputs are real backbone outputs of pair0_0.

| Stage | p50 |
| --- | --- |
| NMS: MaxPool 5x5 + Equal + Greater + And, 1.2 Mpx | 2.35 |
| Dense reliability, `GridSample` bilinear at 1.2 Mpx | 60.4 |
| Dense reliability, pad + `Resize` x8 + crop | 2.99 |
| Candidate-only reliability, `GridSample` at 16384 points | 0.83 |
| TopK plain K = 2048 / 4096 over 1.2 M | 1.36 / 1.84 |
| TopK chunked K = 2048 / 4096 | 1.27 / 6.17 |
| Descriptor bicubic `GridSample` + L2, K = 2048 / 4096 | 1.20 / 5.69 |

- **In-graph NMS costs about 2 ms**, against a 50 to 80 ms backbone. The plan's "5x5 max-pool
  about 55 ms" does not hold for ORT. The 55 to 75 ms cost in the first export was the dense
  bilinear `GridSample` (ORT CPU's `GridSample` is slow per point). `Resize` gives identical values
  in 3 ms.
- After the rewrites, in-graph post-processing (MaxPool, Resize, Where, TopK, descriptor sampling)
  is about 8 to 17 ms of a 58 to 67 ms XFeat graph at 8 threads.
- Candidate-only reliability sampling (0.8 ms at 16k points; the images had 3.6k to 19.7k NMS
  candidates) is cheaper than dense `Resize` (3 ms). It needs data-dependent candidate counts, or a
  fixed candidate cap that is not exact. The dense form is kept.
- The remaining large costs are backbone-side: Conv (18 ms) and the 65-channel softmax on 120x160
  (14 ms, probably the channel-axis layout in ORT).
- Chunked TopK has no benefit on ORT CPU (plain TopK is 1.4 to 1.8 ms). It exists for TensorRT.
- LighterGlue on ORT CPU: about 170 to 180 ms at K = 2048 and 660 to 740 ms at K = 4096, about
  4x per doubling. Research 01 reported 275 / 1263 ms for torch CPU, measured under different
  conditions and not rerun here. LighterGlue stays the dominant cost, 3x XFeat at K = 2048 and
  10x at 4096.

## Unknowns

- **U1** settled for export and ORT CPU. Both static graphs export at opset 20, match at set
  level (recall/precision ≥ 0.9995 at 1 px, cosine ≥ 0.999999, LG Jaccard 1.0 on identical inputs),
  and in-graph NMS costs about 2 ms vs a 50 to 115 ms backbone.
- **U9** input only. The chunked K = 4096 graph uses TopK K ≤ 2048 throughout and matches plain
  TopK on 10/10 images. Whether TensorRT builds it, and what it costs there, is for the TensorRT
  agent. The plain K = 4096 graph is included to confirm the cap.
- **New, not in the plan:**
  - upstream `_unfold2d` costs 43 ms on ORT CPU and is fixed by `SpaceToDepth`;
  - optocycle's dense `GridSample` costs 60 to 75 ms on ORT CPU and is fixed by `Resize`;
  - opset 23 exports SDPA as ONNX `Attention` (fused-attention route for C2/C3 with a masked
    probe still to do).

## Files for the TensorRT agent

Directory `/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/onnx/`
(scratch, not committed; regenerate with the scripts below). Every file carries `metadata_props`
(height, width, batch, grid, threshold, top_k, topk mode, chunk settings, opset).

| Use | File |
| --- | --- |
| XFeat K = 2048, plain TopK (recommended) | `/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/onnx/xfeat_1280x960_k2048_relresize_s2d.onnx` |
| XFeat K = 2048, chunked TopK | `/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/onnx/xfeat_1280x960_k2048_chunked_relresize_s2d.onnx` |
| XFeat K = 4096, chunked, all TopK ≤ 2048 (recommended) | `/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/onnx/xfeat_1280x960_k4096_chunked_relresize_s2d.onnx` |
| XFeat K = 4096, plain TopK(4096) (cap check) | `/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/onnx/xfeat_1280x960_k4096_relresize_s2d.onnx` |
| LighterGlue K = 2048 | `/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/onnx/lighterglue_k2048.onnx` |
| LighterGlue K = 4096 | `/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/onnx/lighterglue_k4096.onnx` |
| Fallback backbone only (`feats`, `prob`, `reliability`) | `/tmp/claude-1000/-home-chris-xfeat-dev-xfeat-rt/ec4ccc2e-76cf-504e-afb3-691238a05732/scratchpad/c1/onnx/xfeat_backbone_1280x960_s2d.onnx` |

Same directory, for comparison:

- `xfeat_1280x960_k{2048,4096}[_chunked].onnx` (upstream layout, dense GridSample);
- `*_relresize.onnx` without `_s2d`;
- `stage_*.onnx` (stage graphs);
- `probe_sdpa_opset{20,23}.onnx`.

Inputs for comparison:

- 10 images as `pair{i}_{v}.npy` in `.../scratchpad/c1/data/`;
- reference outputs as `ref_k{K}_pair{i}_{v}.npz` (kpts, scores, desc) and
  `ref_k{K}_pair{i}_matches.npy` in `.../scratchpad/c1/ref/`.

TensorRT points to check:

- bicubic `GridSample` in FP16;
- `SpaceToDepth` / `DepthToSpace`;
- `ScatterElements` with -inf;
- bool I/O;
- int64 `m0` / TopK indices;
- the -30000 mask in FP16;
- `Log(Sigmoid)` in FP16.

## Scripts

All in `.../scratchpad/c1/`. Throwaway.

| Script | Role |
| --- | --- |
| `export/static_models.py` | Static modules |
| `export/reference_outputs.py` | Reference outputs and torch equivalence |
| `export/export_all.py` | Main export |
| `export/export_pixelops.py` | `_s2d` export |
| `ortenv/verify.py` | ORT accuracy, stage timing and profiles (torch-free venv) |
| `ortenv/verify2.py` | ORT follow-up checks |
| `ortenv/verify3.py` | ORT checks of the `_s2d` graphs |
| `ortenv/timing_final.py` | Final interleaved timing |

Raw results are in `results/*.json` and the progress log in `logs/export.log`.

## Recommendation

- **Go**: XFeat as one graph with in-graph NMS, TopK, sampling and validity mask. Fold into
  `xfeat-export`:
  - the `resize` reliability form;
  - `SpaceToDepth` / `DepthToSpace` (custom translation table);
  - the `ChunkedTopK` flag (off by default on ORT, on for TensorRT at K ≥ 3840).
- **Go**: static padded LighterGlue with explicit masked attention at opset 20. The opset-23
  `Attention` route is worth a masked probe before C3 fusion work.
- Contract test: compare keypoints as sets. Allow at most one tie swap at the K-th score per image
  (score gap ≤ 1e-5 relative).
- **Fallback** if a backend cannot run the post-processing (for example a TensorRT failure on
  bicubic FP16 or on ScatterElements): `xfeat_backbone_1280x960_s2d.onnx` with NumPy/native
  MaxPool, TopK and sampling outside, as in the BCDL port and kornia/vision-rt. For TopK alone,
  the plain K ≤ 3839 graph keeps everything else in-graph.
- Timings need a rerun on an idle machine before they go into any published table.
