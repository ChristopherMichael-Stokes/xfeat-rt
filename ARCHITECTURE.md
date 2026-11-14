# ARCHITECTURE.md: xfeat-rt

XFeat feature extraction plus LighterGlue matching, made as fast as possible on CPU, CUDA GPUs and
Apple MPS. A PyTorch reference defines correct behavior and exports the models; the runtime
backends and the pipeline run the exported models without PyTorch.

## Run modes

Two run modes live side by side and may use different exports and code paths:

- **Real-time:** batch 1, lowest latency per pair.
- **Batch processing:** highest throughput (items per second); latency per item does not matter.
  The batch size is picked per machine by measurement, not fixed.

Pipelining around the models (thread pools, prefetching, overlapping I/O with inference) belongs
to deployment code, not this repo. The repo provides models that take batches and serial tools
built on them.

Throughput benchmarks time the model code. Benchmark datasets (MegaDepth-1500 and the like) may be
preprocessed once and the model inputs persisted in a gitignored `uncommitted/` directory.

## Layout

| Path         | uv workspace | Holds                                                     |
| ------------ | ------------ | --------------------------------------------------------- |
| `reference/` | yes          | PyTorch reference implementation and model export         |
| `runtime/`   | yes          | `xfeat-core` (metadata, result types, glue), `xfeat-pairs` (pair manifest, preprocessing, prediction directory) and `xfeat-bench` (timing harness); no PyTorch |
| `backends/<name>/` | no, one uv project each | One inference backend per directory; own lock, `.venv` and Python pin; never installed together |
| `agent-docs/` | no          | Docs written by and for agents                            |
| root         | no           | Shared checks (`prek.toml`), `ruff.toml`, `Makefile`      |

- Each workspace has its own `uv.lock` and `.venv`. Members live at
  `<workspace>/packages/<dist-name>/src/<module>/` and are picked up by the `packages/*` glob.
- Backends consume `runtime/` packages by path in development and as built wheels elsewhere.

## Contract between workspaces

The workspaces meet through exported model files and prediction directories. The contract test
exports with `reference`, loads with a backend and reads the backend's predictions back.
`xfeat-evaluation` consumes `runtime/packages/xfeat-pairs` by path (pure Python, no torch) to read
predictions and manifests; `reference/` pins `numpy<2.4` for GlueFactory, which overrides the
runtime packages' own numpy floor there.

## Invariants

- **`runtime/` and `backends/` never depend on PyTorch**, directly or transitively.
- **Exported models are build outputs**, regenerated from `reference` and never committed
  (`.gitignore` covers the usual formats).
- **Python:** 3.13 is the baseline (`.python-version`). `requires-python` stays open-ended
  (`>=3.13`) with no upper bound: the aim is to build on the newest Python every backend supports.
