# 01 — XFeat + LighterGlue: method spec for export planning

## Bottom line

- **XFeat sparse = one static CNN plus about six post-processing ops.** The CNN needs H and W to be
  multiples of 32. Every post-processing op except two can be expressed with static shapes:
  `nonzero` in NMS and the final `scores > 0` filter. Replacing them with a dense score map, then
  `topk(K)`, then a validity mask gives a fixed-shape graph. That is my observation from reading the
  code. I have not tested whether its outputs match the reference.
- **LighterGlue is not a separate architecture.** It is kornia's `LightGlue` class run with a small
  config: d=96, 6 layers, 1 head, 64-D input. The weights were trained by the XFeat authors with
  GlueFactory and ship in the XFeat repo. That repo is Apache-2.0.
  - With LighterGlue's defaults, early exit is **off** (`depth_confidence=-1`).
  - Point pruning is **on** (`width_confidence=0.95`). Whether it actually runs depends on the
    device (details below). It changes outputs, not only speed.
  - With pruning off and both sides padded to K, the whole matcher, including mutual-NN
    thresholding, can be expressed with static shapes.
- **Measured here (Ryzen 7 5800X3D, PyTorch 2.14 CPU, 4 threads), 1280x960:**
  - XFeat CNN: about 50–56 ms for 5.24 GMAC.
  - NMS: about 55 ms. Almost all of that is PyTorch's slow 5x5 stride-1 `max_pool2d` on the 1.2 Mpx
    heatmap, not inherent cost.
  - LighterGlue (pruning off): about 26 ms at K=512, 73 ms at K=1024, 275 ms at K=2048, about 1.26 s
    at K=4096. It grows roughly quadratically in K.
- **No published source gives numbers at 1280x960.** Published numbers are VGA-on-CPU (XFeat paper)
  and RTX 3080 / SuperPoint-LightGlue (LightGlue paper).
- **Your earlier code (`core.tar.gz`) followed upstream XFeat closely**, at pinned commit `e92685f`,
  which is still upstream HEAD. The deliberate deviations:
  - fp16 backbone on CUDA and MPS
  - CUDA-graph capture of the backbone
  - resize to a fixed W×H instead of upstream's floor-to-32 resize
  - LightGlue pruning forced off
  - a custom padded, masked LightGlue forward, used with `torch.compile` on Jetson
  - top_k 512 (Jetson), 2048 (local MPS), 1024 (default) or 4096 (calibration), at 1280x960

Sources (verified, read in full or the parts cited):

- XFeat repo at commit e92685f57f8318b18725c5c8c0bd28c7fe188d9a (2025-01-15; `git log origin/HEAD`
  shows it is HEAD):
  <https://github.com/verlab/accelerated_features/tree/e92685f57f8318b18725c5c8c0bd28c7fe188d9a>. In
  the citations below `X:` is short for `.../blob/e92685f/`.
- XFeat paper: arXiv 2404.19174 (PDF text extracted with pypdf).
- LightGlue paper: arXiv 2306.13643.
- kornia 0.8.3, `kornia/feature/lightglue.py`. This is the version your code mentions. Upstream
  `requirements.txt:3` pins `kornia==0.7.2`, and I did **not** diff 0.7.2 against 0.8.3.
- Weights inspected with `torch.load(weights_only=True)`.

---

## 1. XFeat sparse pipeline (`detectAndCompute`, X:modules/xfeat.py:49-103)

Input: B×C×H×W float. With batch 1 at the 1280x960 target, H=960 and W=1280.

| # | Stage | Kind | Output shape | Notes / refs |
|---|---|---|---|---|
| 0 | `preprocess_tensor`: resize to `(H//32*32, W//32*32)`, bilinear, `align_corners=False`. Keeps `rh=H/_H`, `rw=W/_W` | pre | B×C×_H×_W | X:modules/xfeat.py:219-240. **No /255.** The numpy path feeds 0–255 floats; only `parse_input` (match_xfeat path, :396-403) divides by 255. InstanceNorm makes the input scale nearly irrelevant, apart from eps=1e-5. |
| 1 | `x.mean(dim=1)` (RGB→gray), then `InstanceNorm2d(1)` (no affine) | network | B×1×H×W | X:modules/model.py:133-136 |
| 2 | block1: 4 BasicLayers (Conv3x3 + BN(affine=False) + ReLU), 1→4→8(s2)→8→24(s2) | network | B×24×H/4×W/4 | :43-48. The most expensive part per pixel: full-res and half-res convs. |
| 3 | skip1: AvgPool4 + Conv1x1(1→24), added before block2 | network | B×24×H/4×W/4 | :40-41, :140 |
| 4 | block2 (24→24 ×2) | network | B×24×H/4×W/4 | :50-53 |
| 5 | block3: 24→64 (s2), 64→64, 1x1 | network | B×64×H/8×W/8 | :55-59 |
| 6 | block4: 64→64 (s2) ×3 | network | B×64×H/16×W/16 | :60-64 |
| 7 | block5: 64→128 (s2), 128→128 ×2, 1x1 to 64 | network | B×64×H/32×W/32 | :66-71. **This is why H and W must be multiples of 32.** |
| 8 | Fusion: x4 and x5 bilinearly upsampled (`F.interpolate`, default `align_corners=False`) to H/8; `x3+x4+x5`; 2 BasicLayers + Conv1x1 | network | **feats M: B×64×H/8×W/8** | :145-148 |
| 9 | heatmap_head: 1x1 convs + Sigmoid | network | **reliability R: B×1×H/8×W/8** | :79-84, :151 |
| 10 | keypoint_head on `_unfold2d(x_norm, 8)`: space-to-depth of the normalized image to 64 channels, then 1x1 convs | network | **logits K: B×65×H/8×W/8** | :87-92, :113-120, :152. `fine_matcher` (0.89 M params) is used only by XFeat* (semi-dense) and is not part of the sparse path. |
| 11 | `F.normalize(M, dim=1)` | post (exportable) | B×64×H/8×W/8 | xfeat.py:70 |
| 12 | `get_kpts_heatmap`: softmax over 65 channels, drop the dustbin, reshape 8×8 cells to full resolution (a PixelShuffle(8) equivalent) | post (exportable) | B×1×H×W | xfeat.py:242-247 |
| 13 | **NMS**: `MaxPool2d(5, stride 1, pad 2)`; `pos = (x==maxpool)&(x>the)`; `nonzero` | post, **dynamic N_cand** | B×N_cand×2 (int64, x,y on the full-res integer grid) | xfeat.py:249-263. the = `detection_threshold`, default 0.05 (:23). Per-batch Python loop and padding with zeros. |
| 14 | score = `nearest(heatmap, kp) * bilinear(R, kp)` via `grid_sample`; padded rows set to −1 | post | B×N_cand | xfeat.py:77-80. Sampling convention below. |
| 15 | `argsort(-scores)` over **all** candidates, gather first K | post (full sort) | B×K×2, B×K | xfeat.py:83-87. Not `topk`; argsort is not stable, so tie order is unspecified. |
| 16 | Descriptors: **bicubic** `grid_sample` of M at kp, then L2 normalize | post (exportable given fixed K) | B×K×64 | xfeat.py:90-93; X:modules/interpolator.py |
| 17 | Rescale kp by (rw, rh); `valid = scores > 0` | post, **dynamic ≤K** | N≤K | xfeat.py:96-103 |

**Sampling convention** (X:modules/interpolator.py):

- Grid normalization is `2*x/(W-1) - 1`, but `grid_sample` is called with `align_corners=False`. So
  a full-res pixel x samples source coordinate `x*W'/(W-1) - 0.5` in a map of width W'. That is a
  deliberate or accidental half-pixel-plus-scale skew, and an export must reproduce it exactly.
- **Verified edge effect** (harness run here): for nearest sampling of the full-res heatmap, the
  last column and last row map to `W-0.5` and `H-0.5`. Round-half-even sends these out of bounds, so
  the sample is 0, the score is 0, and keypoints in the last row or column are always discarded by
  `valid = scores > 0`. That is 2239 px at 1280x960 and 1119 px at VGA.

**Model size** (from the state_dict):

- `xfeat.pt` has 1.55 M float params in total, of which 0.89 M are `fine_matcher`. So the sparse
  backbone is about 0.66 M.
- SHA-256 `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`.
- BN is `affine=False` with running stats, so it can be folded into the convs.

**CNN compute** (`torch.utils.flop_counter`, measured here):
**1.31 GMAC at 640x480; 5.24 GMAC at 1280x960.** It scales linearly with H·W.

**Keypoints are integer pixel positions.** There is no sub-pixel refinement in sparse mode.

**Paper vs code:**

- The paper's score is `K_ij · R_ij` (§4 "XFeat inference"). The code implements this through the
  samplers above.
- The paper says XFeat* uses scales 0.65 and 1.3; the code uses 0.6 and 1.3 (xfeat.py:379). This is
  irrelevant to the sparse path.

### Data-dependent parts of XFeat (static-export blockers)

- `nonzero` in NMS: N_cand varies with image content.
  - Measured on upstream `assets/ref.png` upscaled: 7,433 at VGA and 25,028 at 1280x960.
  - Random noise gives 8,709 and 34,586.
  - So for textured images N_cand ≫ K and the output is exactly K. Blank or low-texture frames can
    give fewer than K.
- `valid = scores > 0` gives a variable-length output.
- The per-batch Python loop in NMS. Batching works in the CNN, but each image's post-processing is
  independent.
- A static rewrite is possible: score map = mask · heat · R-sampled-densely (with the same skewed
  grid), then `topk(K)` over H·W, then bicubic sampling, plus a validity mask. This is my
  observation, not taken from any source. Two caveats:
  - Tie ordering under `topk` vs `argsort` may differ, especially in fp16.
  - Dense sampling of R at 1.2 Mpx costs more than sampling only the candidates.

---

## 2. LighterGlue

### What it is (verified)

- `X:modules/lighterglue.py:1-56` is a wrapper. It overwrites the **class attribute**
  `LightGlue.default_conf` (a global side effect) with:
  - `input_dim 64`, `descriptor_dim 96`, `n_layers 6`, `num_heads 1`
  - `flash True`, `mp False`
  - `depth_confidence -1` (early exit off)
  - `width_confidence 0.95` (pruning on)
  - `filter_threshold 0.1`
- It then builds `kornia.feature.lightglue.LightGlue(None)`.
- It loads `weights/xfeat-lighterglue.pt` with `strict=False`. If the local file is missing, it
  downloads `https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt`,
  which is the unpinned `main` branch.
- kornia 0.8.3 also lists `"xfeat"` as a native feature type with the same config and the same URL
  (lightglue.py:459-468, :542-544).
- README (:207-218): "a lighter version of LightGlue … approximately three times faster than the
  original LightGlue", trained with GlueFactory. On MegaDepth-1500, AUC@5/10/20:
  - 640 px, 1300 kp: 0.444 / 0.610 / 0.746
  - 1024 px, 4096 kp: 0.564 / 0.710 / 0.819
  - I could not find the speedup measurement or the training dataset or recipe documented in the
    repo. **Unverified.**
- **License and provenance:**
  - XFeat repo LICENSE is Apache-2.0 and covers the in-repo `.pt` files. There is no separate
    weights license.
  - kornia is Apache-2.0.
  - The architecture is from cvg/LightGlue (paper 2306.13643). I did not fetch the cvg repo's
    LICENSE; I know of it as Apache-2.0 but have **not verified** that this session.
- **Checkpoint contents** (`xfeat-lighterglue.pt`, SHA-256 `766102df…654b8`, 291 tensors):
  - 169 `matcher.*` tensors, 1.126 M params. These are input_proj, posenc, transformers 0–5,
    log_assignment 0–5, and token_confidence 0–4.
  - 122 `extractor.model.net.*` tensors: a full XFeat copy. Its
    **conv weights are identical to `xfeat.pt`, but its BN running stats differ** (max relative diff
    0.70; `num_batches_tracked` 538,624 vs 360,944).
  - Upstream inference ignores the embedded extractor and uses `xfeat.pt`. Which stats LighterGlue
    was trained against is not documented. **Unverified.**
  - Loading reports no unexpected matcher keys. Only `confidence_thresholds` is missing; it is a
    computed buffer.

### Stages (kornia 0.8.3 `LightGlue._forward`, lightglue.py:612-783)

M and N are the keypoint counts per image (≤K); d=96; h=1 head; L=6.

| Stage | Kind | Shape | Ref |
|---|---|---|---|
| `normalize_keypoints`: `(kp - size/2) / (max(size)/2)` | pre | B×M×2 | :55-62. **Hard check** that all normalized kp are in [−1,1] (:626-627, `.item()` host sync). For 1280x960, y is in [−0.75, 0.75]. |
| input_proj Linear 64→96 | network | B×M×96 | :508-509, :696 |
| Positional encoding: `Wr` Linear(2→48, no bias), then cos/sin, then repeat_interleave → rotary freqs | network | 2×B×1×M×96 | :88-113. Computed once per pair and cached across layers. |
| ×6 TransformerLayer. **Self**: Wqkv 96→288, rotary on q and k, SDPA M×M, out_proj, FFN [cat 192 → 192 → LN → GELU → 96] residual | network | B×M×96 | :185-224 |
| ×6 TransformerLayer. **Cross** (bidirectional, shared qk): one sim matrix M×N, softmax over both directions, to_out, FFN | network | B×M×96, B×N×96 | :227-284 |
| (layers 0–4) TokenConfidence Linear 96→1 + sigmoid | network | B×M | :116-134. **Only used if depth_confidence>0, which is off for LighterGlue.** |
| (layers 0–4) Pruning: matchability = sigmoid(Linear 96→1); keep where `> 1-0.95 = 0.05` (token is None, so confidence plays no role); `index_select` | post, **dynamic** | M_i ≤ M | :722-737, :790-795 |
| Final MatchAssignment on the **last executed layer**: final_proj 96→96, each side scaled by d^-0.25, sim = M×N; z = matchability logits; `log_softmax` rows + `log_softmax` cols + logsigmoid(z0) + logsigmoid(z1)ᵀ; dustbin row/col = logsigmoid(−z) | network | B×(M+1)×(N+1) | :333-368, :741 |
| `filter_matches`: row and column argmax, mutual check, `exp(max) > filter_threshold` (0.1) | post (static: m0 B×M, scores B×M) | then `where(valid)` → S×2 dynamic | :374-390, :744-753 |

### Adaptive depth and width (paper vs code)

- **Paper §3.3 and App. B**:
  - Exit when more than α=95% of points have confidence c > λ_ℓ, where λ_ℓ = 0.8 + 0.1·e^(−4ℓ/L).
  - Prune a point when it is confident **and** σ < β.
  - Measured effect on MegaDepth: 1.86× speedup on easy pairs, 1.45× on average (Table 5).
- **kornia, with LighterGlue's defaults:**
  - Early exit is off.
  - Pruning keeps points with **matchability > 0.05 only**. Because `token` is None, the "confident"
    condition of Eq. 13 does not apply.
  - The pruning threshold is per device (`pruning_keypoint_thresholds`, :423-428, :810-814):
    - **CPU and MPS: −1, so pruning runs at every keypoint count.**
    - CUDA: 1024.
    - CUDA with SDPA available (any torch ≥2): **1536**.
    - **So the reference changes behavior by device.** CPU and MPS prune always; CUDA does not prune
      at K ≤ 1536.
  - Pruning removes points from matching altogether, so outputs change, not only speed.
  - **For a deterministic reference, pin `width_confidence=-1` (your earlier code did) or fix the
    rule explicitly.**
- **Precision**:
  - On CUDA with `flash=True`, kornia casts q, k and v to **fp16** for SDPA regardless of `mp=False`
    (:161-166, :266-268).
  - On CPU, attention is fp32.
  - So "the PyTorch reference" is already fp16-attention on CUDA.

### Static export of LighterGlue

Padding both sides to K and using masks (`pad_to_length`, `masked_forward`, :65-73, :316-330) gives
fixed shapes. The conditions:

- Pruning off; early exit is already off.
- The final head is `log_assignment[5]`.
- Matches are produced as fixed-shape `m0` / `mscores0` (B×K) plus a −1 sentinel.

Caveats:

- **kornia 0.8.3's own compiled path is broken for padding.** :739 slices `desc0[..., :m, :]` after
  every layer, while the masks and encodings stay at the padded length. This is what your earlier
  code worked around; the line confirms it.
- The two images can have different `image_size` values. These are runtime inputs, not export
  constants.

---

## 3. Where time goes

### Published measurements

- **XFeat paper, Table 1:** 27.1 ± 0.33 FPS for sparse XFeat at **VGA** on an Intel i5-1135G7 laptop
  CPU, vanilla PyTorch. It includes MNN matching of 4096 kp (§4 "Baselines"). XFeat* reaches 19.2
  FPS.
- **XFeat paper, Fig. 7 (i7-6700K CPU), read off the bar chart, approximate:**
  - sparse XFeat: detection ≈ 8 ms, description ≈ 35 ms, MNN matching ≈ 83 ms
  - The resolution and K behind Fig. 7 are not stated in the text I extracted. **Unverified.**
- **XFeat paper, App. C:** 1.8 FPS on an Orange Pi Zero 3 (Cortex-A53) at 480×360.
- **XFeat paper, Table 6:** XFeat* at 1.33 pairs/s at 1200 px on i7-6700K.
- **XFeat README:** about 1,400 FPS batched at VGA on an RTX 4090; >150 FPS at batch 1 on GPU at
  VGA. The measurement setup is not given.
- **LightGlue paper, App. D / Fig. 7 / Fig. 13:** RTX 3080, SuperPoint features at d=256, 9 layers,
  4 heads, so not LighterGlue. It shows runtime vs keypoint count (512–4096). Figure values were not
  extracted as numbers.
- **No source gives numbers at 1280x960, for LighterGlue on any hardware, or on Jetson, Raspberry Pi
  or MPS.**

### Measured here (indicative only)

- Setup: Ryzen 7 5800X3D, WSL2, torch 2.14.1+cpu, kornia 0.8.3, median of 15 runs.
- XFeat input is random noise. NMS was also run on a real image, with the same time.
- LighterGlue input is random kp and descriptors with pruning off, so all K points go through all 6
  layers.
- Harness:
  `/tmp/claude-1000/-home-chris-xfeat-dev/1902fc88-97d1-56bb-8dcc-d233cba4e78e/scratchpad/bench/stage_timing.py`
  and `.../bench/real_cands.py`.

| Stage | VGA, 1 the | VGA, 4 the | 1280x960, 1 the | 1280x960, 4 the |
|---|---|---|---|---|
| XFeat CNN | 29.2 ms | 11.8–16.8 | 127.6 | 49.3–55.8 |
| heatmap reshuffle | 0.5 | 0.1 | 1.8 | 0.5–0.8 |
| NMS (maxpool + compare + nonzero) | 14.5 | 14.3 | 57.8 | 56.6 (maxpool alone 53 ms) |
| score sampling | 0.1 | 0.1 | 0.2 | 0.2 |
| argsort (all cands) | 0.4 | 0.4 | 2.1 | 2.1 |
| bicubic desc + norm, K=1024 / 4096 | 0.8 / 3.2 | 0.7 / 2.6 | 0.9 / 3.3 | 0.7 / 4.6 |

| LighterGlue (no pruning) | K=512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| GMAC | 1.57 | 4.10 | 12.0 | 39.3 |
| CPU 1 thread | 55 ms | 185 | 685 | 3516 |
| CPU 4 threads | 26 ms | 73 | 275 | 1263 |

Reading:

- At 1280x960 the **CNN and NMS are of the same order on CPU**. The NMS cost is PyTorch's
  single-threaded 5×5 stride-1 maxpool over 1.2 Mpx, which looks like an artifact a custom or fused
  kernel would remove.
- LighterGlue cost is dominated by the K×K attention and assignment terms. It is about 4× per
  doubling of K at larger K. At K≥1024 it exceeds XFeat extraction.

### Your earlier Jetson Orin number

`core/cv/match.py:34` comment, unverified by me: LighterGlue with `torch.compile` reduce-overhead
went from 53 ms to 13 ms on Orin, with pad_to=512 per the Jetson config.

---

## 4. The user's earlier code (`/home/chris/xfeat_dev/core.tar.gz`)

Extracted to `/tmp/tmp.5bR8j0KoaA/core` (mktemp; read only, nothing executed).

**Files:**

- `core/cv/xfeat.py`: `XFeatExtractor`
- `core/cv/match.py`: `Matcher`
- Call sites:
  - `core/nav_v1/fpv_cv.py:177-188` (FPV extraction)
  - `core/nav_v1/satellite_localization.py:242-270` (satellite-crop extraction and matching)
  - `core/calibration/rig.py:163-175`

### Entry points and weights

- `torch.hub.load("verlab/accelerated_features:e92685f…", "XFeat", pretrained=True, top_k, detection_threshold)`
  (xfeat.py:15-16, :133-140). Weights resolve as follows:
  - **xfeat.pt** comes from hubconf's URL `raw/main/weights/xfeat.pt` (X:hubconf.py:255). The code
    is pinned by SHA but **these weights are not**. Today main == e92685f, so the bytes match.
  - Optional `weights_path` loads a local state_dict.
  - **LighterGlue** comes from `modules.lighterglue.LighterGlue()` imported from the hub-cache repo
    dir (match.py:144-155). It uses the repo-local `weights/xfeat-lighterglue.pt`, which is
    SHA-pinned via the checkout.
- **XFeat is run by calling `xf.net` directly** (not `detectAndCompute`), then re-implementing the
  post-processing in `_extract_tensors` (xfeat.py:211-236).
  - The re-implementation is line-for-line the upstream algorithm: `get_kpts_heatmap`,
    `NMS(kernel 5)`, nearest×bilinear score, −1 on zero rows, argsort, top_k, bicubic, L2,
    `scores>0`.
  - It uses upstream's own `NMS`, `get_kpts_heatmap` and interpolator classes.

### Preprocessing

`_preprocess`, xfeat.py:187-209:

- Takes a uint8 gray (H,W) or RGB (H,W,3) frame; float in 0–255 (no /255, same as upstream).
- RGB is averaged to gray **before** resizing. Upstream averages inside the net after resizing; the
  result is near-identical but not bit-exact.
- Bilinear resize (`align_corners=False`) to a **configured fixed W×H** that must be a multiple of
  32, else `ValueError`. Upstream floors to a multiple of 32 instead.
- Cast to fp16 if enabled; optional channels_last.
- Keypoints are scaled back to source pixels by `src/W`, `src/H` (:235), equivalent to upstream's rw
  and rh.

### Parameters actually used

| Item | Value |
|---|---|
| `detection_threshold` | 0.05 (default) |
| NMS kernel | 5 |
| `min_conf` | 0.1 |
| MNN fallback `min_cossim` | 0.82 |

| Config | Device | Frame / crop | top_k | LighterGlue compile |
|---|---|---|---|---|
| `configs/nav_v1_jetson.toml:179-217` | cuda | 1280×960 frames and satellite crops | 512 | `matcher_compile=true`, `pad_to=512` |
| `configs/nav_v1_local.toml:231-284` | mps | 1280×960 | 2048 | no |
| `configs/nav_v1_replay_lockstep.toml:228-278` | mps | 1280×960 | 512 | no |
| `configs/calibration/video_loc.toml:4-7` | — | 1280×704 | 4096 | — |
| Code defaults | — | 640×480 (`XFeatConfig`) | 512 in `XFeatConfig`, 1024 in the node params | — |

- A stray key `extractor_top_k` appears beside `extractor_topk` in the toml files. Pydantic's
  default is `extra=ignore`, so it is ignored. Harmless.

### Deviations from upstream

- **Precision:** fp16 backbone by default on CUDA, and **also on MPS** (configs set
  `torch_fp16=true` explicitly). Post-processing is cast back to fp32 (xfeat.py:217-219). Upstream
  is fp32 throughout, except kornia's fp16 SDPA on CUDA.
- **CUDA graph** capture of the backbone at fixed shape (:162-185). `cudnn.benchmark=True`.
- **`torch.set_num_threads(1)`** globally (:129), which matters for CPU speed.
- **LighterGlue pruning forced off:** `width_confidence = -1` (match.py:157-162). The stated reason
  is that data-dependent shapes broke MPS graph caching and cudagraphs.
  - The code comment says CUDA prunes above 1024. kornia actually uses **1536** when SDPA is
    available, and CPU/MPS prune at any count.
- **Padded static-length LighterGlue** (`_padded_lightglue`, match.py:178-221), used when
  `matcher_compile`:
  - Pads to `pad_to` with masks.
  - Runs the 6 `masked_forward` layers compiled with `torch.compile(mode="reduce-overhead")`.
  - Applies the last head `log_assignment[-1]` to the real rows only, then `filter_matches`.
  - Reason: kornia 0.8.3's slicing bug (§2).
  - This matches kornia's unpadded output when pruning and early exit are off (same head, masked
    attention), up to fp16 SDPA rounding on CUDA. I did not test that equivalence.
- Keypoints and descriptors stay on device between extractor and matcher; the remote side's features
  arrive as numpy.

---

## 5. Unverified or open

- Hardware and resolution behind XFeat Fig. 7, and the LighterGlue "≈3× faster" claim.
- LighterGlue training data, recipe, and which XFeat BN stats it was trained against.
- cvg/LightGlue LICENSE text (not fetched).
- Differences between kornia 0.7.2 (upstream's pin) and 0.8.3 `LightGlue`.
- Whether ONNX, TensorRT, CoreML or ArmNN support bicubic `grid_sample` with these exact semantics:
  PyTorch cubic A=−0.75, zeros padding, `align_corners=False`. Not checked; that belongs to a
  separate runtime-support question.
- No measurements on GPU, MPS, Jetson or Raspberry Pi were taken here.
