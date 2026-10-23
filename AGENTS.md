# AGENTS.md: xfeat-rt

XFeat feature extraction plus LighterGlue matching, made as fast as possible on CPU, CUDA GPUs and
Apple MPS. A PyTorch reference defines correct behavior and exports the models; the runtime
backends and the pipeline run the exported models without PyTorch.

## Layout

| Path         | uv workspace | Holds                                                     |
| ------------ | ------------ | --------------------------------------------------------- |
| `reference/` | yes          | PyTorch reference implementation and model export         |
| `runtime/`   | yes          | Inference backends and pipeline/glue code; no PyTorch     |
| root         | no           | Shared checks (`prek.toml`), `ruff.toml`, `Makefile`      |

- Each workspace has its own `uv.lock` and `.venv`. Members live at
  `<workspace>/packages/<dist-name>/src/<module>/` and are picked up by the `packages/*` glob.
- The workspaces meet only through exported model files. Test that contract by exporting with
  `reference` and loading with `runtime`.

## Hard constraints

- **`runtime/` never depends on PyTorch**, directly or transitively. Check `runtime/uv.lock` after
  adding a dependency.
- **Exported models are build outputs.** Regenerate them from `reference`; never commit them
  (`.gitignore` covers the usual formats).
- **Existing XFeat/LighterGlue code outside this repo is reference material only.** Read it to
  understand behavior; never copy it in.
- **Python:** 3.13 is the baseline (`.python-version`). `requires-python` stays open-ended
  (`>=3.13`) with no upper bound: the aim is to build on the newest Python every backend supports.

## Commands

Run from the repo root unless a workspace is named.

| What                         | Command                                                |
| ---------------------------- | ------------------------------------------------------ |
| Install prek (once)          | `uv tool install prek@0.5.7`                           |
| Sync both workspaces, hooks  | `make setup`                                           |
| All checks                   | `make check`                                           |
| Checks for one workspace     | `prek run --all-files runtime/`                        |
| Update hook revisions        | `make update-hooks` (skips releases under 7 days old)  |
| Types in one workspace       | `uv run --frozen ty check` from `reference/` or `runtime/` |
| Add a dependency to a member | `uv add --package <dist-name> <dep>` from the workspace |

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
