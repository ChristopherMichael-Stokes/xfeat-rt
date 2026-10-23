.PHONY: setup check update-hooks

setup:
	uv sync --project reference --all-packages --locked
	uv sync --project runtime --all-packages --locked
	prek install

check:
	prek run --all-files

update-hooks:
	prek update --cooldown-days 7
