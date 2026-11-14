.PHONY: setup check test export update-hooks

# uv tool directory is not always on PATH
PREK := uvx prek@0.5.7

setup:
	uv sync --project reference --all-packages --locked
	uv sync --project runtime --all-packages --locked
	uv sync --project backends/ort --locked
	uv sync --project backends/tensorrt --locked
	$(PREK) install

check:
	$(PREK) run --all-files

test:
	cd reference && uv run --frozen pytest packages
	cd runtime && uv run --frozen pytest packages
	cd backends/tensorrt && uv run --frozen pytest tests

# TODO: separate export targets per backend. This target writes the chunked TopK graph that
# TensorRT needs; ORT consumes the same files and has not been measured on them.
export:
	cd reference && uv run --frozen xfeat-export --top-k $(TOP_K) --chunked-topk --out-dir uncommitted/models

update-hooks:
	$(PREK) update --cooldown-days 7
