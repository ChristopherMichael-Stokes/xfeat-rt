.PHONY: setup check test export update-hooks

# uv tool directory is not always on PATH
PREK := uvx prek@0.5.7

PLATFORM := $(shell uname -s)-$(shell uname -m)

# TensorRT has no macOS wheels; its lock carries manylinux x86_64 and aarch64 wheels
ifeq ($(PLATFORM),Darwin-arm64)
BACKENDS ?= ort
else ifeq ($(PLATFORM),Linux-x86_64)
BACKENDS ?= ort tensorrt
else ifeq ($(PLATFORM),Linux-aarch64)
BACKENDS ?= ort tensorrt
else
$(error unsupported platform $(PLATFORM); set BACKENDS to the backends to sync)
endif

TARGET ?= native

setup:
	uv sync --project reference --all-packages --locked
	uv sync --project runtime --all-packages --locked
	for backend in $(BACKENDS); do uv sync --project backends/$$backend --locked || exit 1; done
	$(PREK) install

check:
	$(PREK) run --all-files

test:
	cd reference && uv run --frozen pytest packages
	cd runtime && uv run --frozen pytest packages
	for backend in $(BACKENDS); do [ ! -d backends/$$backend/tests ] || (cd backends/$$backend && uv run --frozen pytest tests) || exit 1; done

export:
	cd reference && uv run --frozen xfeat-export --target $(TARGET) --top-k $(TOP_K) --chunked-topk --precision fp16 --out-dir uncommitted/models

update-hooks:
	$(PREK) update --cooldown-days 7
