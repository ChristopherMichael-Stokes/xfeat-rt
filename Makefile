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

export:
	cd reference && uv run --frozen xfeat-export --top-k $(TOP_K) --out-dir uncommitted/models

update-hooks:
	$(PREK) update --cooldown-days 7
