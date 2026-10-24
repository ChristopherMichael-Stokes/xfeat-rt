"""Building, caching and loading TensorRT engines from the exported ONNX graphs.

Engines are tied to the TensorRT version and the GPU, so the cache key covers the ONNX bytes, the
TensorRT version, the compute capability, the device name and the builder options. A plan that
fails to deserialize is rebuilt.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import onnx
import tensorrt as trt
from cuda.core import Device

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "xfeat-rt" / "tensorrt"

MAX_OPTIMIZATION_LEVEL = 5
DEFAULT_OPTIMIZATION_LEVEL = 3
"""TensorRT's own default; levels 0 to 5 trade build time for faster engines."""


class EngineBuildError(RuntimeError):
    """The ONNX file could not be parsed or the network could not be built."""


@dataclass(frozen=True)
class BuildOptions:
    """Builder settings that change the engine; all are part of the cache key."""

    tf32: bool = False
    """Allow TF32 tensor-core math in FP32 layers: faster, about 1e-3 relative error per op."""
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL

    def __post_init__(self) -> None:
        """Reject levels TensorRT does not define."""
        if not 0 <= self.optimization_level <= MAX_OPTIMIZATION_LEVEL:
            msg = f"optimization_level must be in 0..{MAX_OPTIMIZATION_LEVEL}"
            raise ValueError(msg)


@dataclass(frozen=True)
class LoadedEngine:
    """A deserialized engine with the ONNX `metadata_props` it was built from."""

    engine: trt.ICudaEngine
    metadata: dict[str, str]
    path: Path
    built: bool


# TensorRT keeps the first logger registered in the process and ignores later ones.
_LOGGER = trt.Logger(trt.Logger.WARNING)


def read_onnx_metadata(onnx_path: Path) -> dict[str, str]:
    """The raw `metadata_props` string map of `onnx_path`."""
    model = onnx.load(onnx_path, load_external_data=False)
    return {entry.key: entry.value for entry in model.metadata_props}


def engine_cache_key(onnx_path: Path, device: Device, options: BuildOptions) -> str:
    """Hex digest identifying the engine built from `onnx_path` on `device`."""
    digest = hashlib.sha256(onnx_path.read_bytes())
    parts = (
        trt.__version__,
        str(device.arch),
        device.name,
        f"optimization_level={options.optimization_level}",
        f"tf32={options.tf32}",
    )
    digest.update("\0".join(parts).encode())
    return digest.hexdigest()


def _timing_cache_path(cache_dir: Path, device: Device) -> Path:
    return cache_dir / f"timing-trt{trt.__version__}-sm{device.arch}.cache"


def _write_atomically(path: Path, data: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    partial.replace(path)


def build_engine(onnx_path: Path, timing_cache_path: Path, options: BuildOptions) -> bytes:
    """Serialized engine for `onnx_path`, reading and updating the timing cache file.

    The network is strongly typed, so the precision is the one of the ONNX tensors; TF32 applies
    to the FP32 layers only when `options.tf32` is set.
    """
    builder = trt.Builder(_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, _LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        msg = f"TensorRT cannot parse {onnx_path}: {errors}"
        raise EngineBuildError(msg)

    config = builder.create_builder_config()
    config.builder_optimization_level = options.optimization_level
    if not options.tf32:  # TensorRT enables TF32 by default
        config.clear_flag(trt.BuilderFlag.TF32)
    previous = timing_cache_path.read_bytes() if timing_cache_path.exists() else b""
    timing_cache = config.create_timing_cache(previous)
    if not config.set_timing_cache(timing_cache, False):  # noqa: FBT003  # TensorRT positional flag: ignore_mismatch
        timing_cache = config.create_timing_cache(b"")
        config.set_timing_cache(timing_cache, False)  # noqa: FBT003

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        msg = f"TensorRT failed to build an engine from {onnx_path}"
        raise EngineBuildError(msg)
    _write_atomically(timing_cache_path, bytes(timing_cache.serialize()))
    return bytes(serialized)


def load_or_build_engine(
    onnx_path: Path,
    device: Device,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    options: BuildOptions = BuildOptions(),  # noqa: B008  # frozen, safe default
) -> LoadedEngine:
    """Engine for `onnx_path` from the cache, built and stored there on a miss."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{onnx_path.stem}-{engine_cache_key(onnx_path, device, options)[:16]}"
    plan_path = cache_dir / f"{stem}.engine"
    metadata_path = cache_dir / f"{stem}.metadata.json"
    runtime = trt.Runtime(_LOGGER)

    if plan_path.exists() and metadata_path.exists():
        engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if engine is not None:
            metadata = json.loads(metadata_path.read_text())
            return LoadedEngine(engine, metadata, plan_path, built=False)

    plan = build_engine(onnx_path, _timing_cache_path(cache_dir, device), options)
    metadata = read_onnx_metadata(onnx_path)
    _write_atomically(plan_path, plan)
    _write_atomically(metadata_path, json.dumps(metadata).encode())
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        msg = f"TensorRT cannot deserialize the engine it just built from {onnx_path}"
        raise EngineBuildError(msg)
    return LoadedEngine(engine, metadata, plan_path, built=True)
