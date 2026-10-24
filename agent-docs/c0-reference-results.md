# C0 results: reference pipeline

Checkpoint C0 of `implementation-plan.md`. Gate 0 passes.

## Setup

- XFeat: `verlab/accelerated_features` at `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`, loaded through
  its `hubconf.py` with `torch.hub` (the repository has no `pyproject.toml`). Weights come from
  the pinned checkout's `weights/`, checked by SHA-256.
- LighterGlue: upstream `modules/lighterglue.py` on kornia 0.8.3.
- Scoring: GlueFactory `2d17e3b3bd7d30f0c828d4c4d3eac4ecefbf283d`, `MegaDepth1500Pipeline`, poselib,
  RANSAC threshold sweep 0.5 to 3.0 px with the best reported (GlueFactory `ransac_th: -1`).
- 640 px long side, K = 1300, detection threshold 0.05, match threshold 0.1, all 1500 pairs.
- Keypoints written in COLMAP convention (+0.5 px) before rescaling to the original image.
- torch 2.14.1+cu130, numpy 2.3.5, RTX 4070 Ti, WSL2.
- Command: `uv run --frozen xfeat-megadepth1500 --device <cpu|cuda> --grid <grid> --[no-]pruning
  --output-dir uncommitted/megadepth1500/<run>` from `reference/`.

## Gate 0 and grid convention (CUDA)

AUC of relative pose error at 5/10/20 degrees.

| Config                        | AUC@5  | AUC@10 | AUC@20 | Matches/pair |
| ----------------------------- | ------ | ------ | ------ | ------------ |
| README (published)            | 0.444  | 0.610  | 0.746  |              |
| `upstream_exact`              | 0.4467 | 0.6147 | 0.7489 | 397          |
| `pixel_centre` (default)      | 0.4441 | 0.6130 | 0.7476 | 398          |

- `upstream_exact` reproduces the README within +0.005 on all three thresholds.
- `pixel_centre` (edge-row and skew fix) is within -0.003 of `upstream_exact`. No confidence
  interval has been computed, so the delta is not shown to be different from zero.
- Cross-check: an independent prototype's predictions (upstream quirks, keypoint offset 0 instead
  of +0.5) scored 0.4456/0.6136/0.7480 with the same scorer.

## Pruning

- kornia prunes only when an image has more keypoints than a device threshold
  (`kornia/feature/lightglue.py:423-427, 722`): 1536 on CUDA with flash attention, 1024 on CUDA
  without, always on CPU. At K = 1300 on CUDA pruning never runs; on and off gave identical
  output (same AUC to 4 decimals, same match count).
- Pruning is measured on CPU, where it is active, and matters on CUDA from K > 1536, which covers
  the plan's K = 2048 and 4096 targets.

| Config (CPU)                  | AUC@5  | AUC@10 | AUC@20 | Matches/pair | Match ms/pair |
| ----------------------------- | ------ | ------ | ------ | ------------ | ------------- |
| `pixel_centre`, pruning on    | 0.4365 | 0.6057 | 0.7415 | 396          | 171           |
| `pixel_centre`, pruning off   | 0.4458 | 0.6135 | 0.7485 | 398          | 207           |

- Pruning costs 0.009/0.008/0.007 AUC at 5/10/20 degrees for about 17% less LighterGlue time
  at K = 1300. Timings are rough: both runs shared the CPU, with no warm-up or synchronisation.
- CPU pruning-off agrees with CUDA `pixel_centre` within 0.002 AUC.

## Pinned-environment workarounds

Each is in `reference/pyproject.toml` or `reference/prek.toml` with its reason:

- `numpy<2.4`: GlueFactory computes AUC with `np.trapz`, removed in numpy 2.4.0.
- `setuptools<81`: GlueFactory imports `pkg_resources`; uv-audit ignores two setuptools advisories
  about sdist builds on macOS and legacy wheel installs on Windows.
- `scikit-learn>=1.5`: GlueFactory pins `~=1.3.0`, which has no cp313 wheels.
- `opencv-python` replaced by `opencv-python-headless`.
- GlueFactory reads a relative predictions path against its `DATA_PATH`; the evaluation module
  passes absolute paths.
