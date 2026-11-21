# AGENTS.md: xfeat-rt

`ARCHITECTURE.md` holds what the system is: run modes, layout, the contract between workspaces and
its invariants. Read it before changing code.

## Rules

- Check the affected `uv.lock` after adding a dependency: `runtime/` and `backends/` never pull in
  PyTorch, even transitively.
- Exported models are regenerated from `reference`, never committed.
- Existing XFeat/LighterGlue code outside this repo is reference material only. Read it to
  understand behavior; never copy it in.
- Python tools run through `uv run` or `uvx`, never bare: the uv tool directory is not always on
  `PATH`.

## Agent docs

- `agent-docs/` holds session briefs, plans, results and research reports for agents. Human-facing
  docs (`README.md`, `ARCHITECTURE.md`, any `docs/`) never link into it, and agent material never
  goes in them.
- `agent-docs/STATUS.md` holds current state in about 40 lines: what is done, what is in progress,
  and which doc holds each line of work. A fresh session starts there.
- One plan and one results doc per line of work (the TensorRT backend, batch processing, …),
  updated in place. Never a new file per checkpoint or session. Research reports are one file per
  question.
- Docs, files and headings name stages by what they are ("the TensorRT backend stage"), never by
  codes (stage letters and numbers, unknown or fork numbers).
- A brief is self-contained: a fresh session runs it with no access to the conversation that wrote
  it, in at most three reads.
- Ideas raised mid-task go into `TODO.md` as one unchecked line under the section they belong to.

## Commands

Run from the repo root unless a workspace is named.

| What                         | Command                                                |
| ---------------------------- | ------------------------------------------------------ |
| Install prek for the git hook (once) | `uv tool install prek@0.5.7`                   |
| Sync workspaces, the machine's backends, hooks | `make setup` (`ort` on macOS arm64; `ort tensorrt` on Linux x86_64/aarch64; override with `BACKENDS="..."`) |
| All checks                   | `make check`                                           |
| Checks for one workspace     | `uvx prek@0.5.7 run --all-files runtime/`              |
| Update hook revisions        | `make update-hooks` (skips releases under 7 days old)  |
| Types in one workspace       | `uv run --frozen ty check` from `reference/` or `runtime/` |
| Add a dependency to a member | `uv add --package <dist-name> <dep>` from the workspace |
| All tests                    | `make test`                                            |
| Reference tests (incl. contract test) | `uv run --frozen pytest packages` from `reference/` |
| Export models                | `make export TOP_K=2048` (FP16 chunked TopK graphs into `reference/uncommitted/models/`; `--target native`: Core ML on Apple silicon, TensorRT elsewhere; `TARGET=tensorrt` or `TARGET=coreml` to pick); `xfeat-export --help` from `reference/` for FP32 or plain TopK |
| CPU latency                  | `uv run --frozen xfeat-ort-bench --help` from `backends/ort/` |
| Core ML latency (Apple silicon) | `uv run --frozen xfeat-ort-bench --provider coreml --xfeat <..._coreml.onnx> --lighterglue <..._coreml.onnx> --out-dir <dir>` from `backends/ort/` (`--xfeat-units`, `--lighterglue-units`, `--cache-dir`) |
| Core ML predictions          | `uv run --frozen xfeat-ort-predict --provider coreml --xfeat <..._coreml.onnx> --lighterglue <..._coreml.onnx> --manifest <pairs.jsonl> --resize letterbox --out-dir <dir>` from `backends/ort/` |
| CUDA latency                 | `uv run --frozen xfeat-trt-bench --help` from `backends/tensorrt/` (export with `--chunked-topk`; plain TopK costs ~5 ms on TensorRT) |
| Batch ladder export          | `xfeat-export --target tensorrt --chunked-topk --precision fp16 --batch 1 2 4 8 16 32 --out-dir <dir>` from `reference/`; one resolution, K and precision per directory |
| CUDA batch-size calibration  | `uv run --frozen xfeat-trt-calibrate --models <ladder dir>` from `backends/tensorrt/` (stored per device, driver, TensorRT, build options and ONNX hashes; a second call reads it) |
| CUDA inference-only pair throughput (development only) | `uv run --frozen python scripts/bench_pairs.py --models <ladder dir> --manifest <pairs.jsonl> --resize letterbox --input-store uncommitted/inputs --out-dir <dir>` from `backends/tensorrt/` (model inputs persisted by `scripts/input_store.py`; times model calls only: per pair at batch 1 with and without CUDA graphs, and every XFeat batch size of the ladder) |
| CUDA batch predictions       | `uv run --frozen xfeat-trt-predict-batch --models <ladder dir> ...` from `backends/tensorrt/` (`--help`; calibrates on a miss) |
| MegaDepth-1500 AUC           | `uv run --frozen xfeat-megadepth1500 --grid pixel_centre --no-pruning --output-dir uncommitted/megadepth1500/<run>` from `reference/` (`--help` for device, K, resize) |

## Checks

- prek is the whole CI apart from builds and tests. A new tool or file type in the repo gets its
  hook in `prek.toml` in the same change.
- Root `prek.toml` covers file hygiene, secrets (gitleaks), spelling (typos), ruff, notebook
  stripping (nbstripout), `pyproject.toml` validation, TOML (tombi), YAML (yamllint), Markdown
  (rumdl), Makefiles (mbake, checkmake) and shell (shellcheck, shfmt).
- Each workspace's `prek.toml` adds `uv-lock`, `uv-audit` and ty, run inside that workspace.
- checkmake requires `setup` and `check` targets in every Makefile (`checkmake.ini`).
- ruff selects every rule; the ignores in `ruff.toml` each carry their reason. ty raises every rule
  to error and does not honor `# type: ignore`.
- Done means `make check` passes and the behavior has been exercised, not only tested.

## Notebooks

Committed notebooks are clean (nbstripout enforces it) and run from their inputs. Executed runs
worth keeping go to a gitignored `uncommitted/` directory beside the notebook, under a timestamped
name with the outputs they wrote.
