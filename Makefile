.PHONY: setup check test export update-hooks

setup:
	uv sync --project reference --all-packages --locked
	uv sync --project runtime --all-packages --locked
	uv sync --project backends/ort --locked
	prek install

check:
	prek run --all-files

test:
	cd reference && uv run --frozen pytest packages
	cd runtime && uv run --frozen pytest packages

export:
	cd reference && uv run --frozen xfeat-export --top-k $(TOP_K) --out-dir uncommitted/models

update-hooks:
	prek update --cooldown-days 7
