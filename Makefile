.PHONY: setup check test export update-hooks

# uv tool directory is not always on PATH
PREK := uvx prek@0.5.7

PLATFORM := $(shell uname -s)-$(shell uname -m)

# TensorRT has no macOS wheels; reference (export, evaluation) has no Linux aarch64 lock
ifeq ($(PLATFORM),Darwin-arm64)
WORKSPACES ?= reference runtime
BACKENDS ?= ort
else ifeq ($(PLATFORM),Linux-x86_64)
WORKSPACES ?= reference runtime
BACKENDS ?= ort tensorrt
else ifeq ($(PLATFORM),Linux-aarch64)
WORKSPACES ?= runtime
BACKENDS ?= ort tensorrt
else
$(error unsupported platform $(PLATFORM); set WORKSPACES and BACKENDS to the projects to sync)
endif

TARGET ?= native

setup:
	for workspace in $(WORKSPACES); do uv sync --project $$workspace --all-packages --locked || exit 1; done
	for backend in $(BACKENDS); do uv sync --project backends/$$backend --locked || exit 1; done
	$(PREK) install

check:
	$(PREK) run --all-files

test:
	for workspace in $(WORKSPACES); do (cd $$workspace && uv run --frozen pytest packages) || exit 1; done
	for backend in $(BACKENDS); do [ ! -d backends/$$backend/tests ] || (cd backends/$$backend && uv run --frozen pytest tests) || exit 1; done

export:
	cd reference && uv run --frozen xfeat-export --target $(TARGET) --top-k $(TOP_K) --chunked-topk --precision fp16 --out-dir uncommitted/models

update-hooks:
	$(PREK) update --cooldown-days 7
