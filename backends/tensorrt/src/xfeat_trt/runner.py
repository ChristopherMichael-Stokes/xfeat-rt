"""Execution of a deserialized engine with persistent device and pinned host buffers."""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import tensorrt as trt
from cuda.core import Device, PinnedMemoryResource

NUMPY_DTYPES: dict[trt.DataType, type[np.generic]] = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT64: np.int64,
    trt.DataType.BOOL: np.bool_,
}


class EngineRunner:
    """Owns the stream, execution context and I/O buffers of one static-shape engine.

    Inputs are cast to the engine's tensor dtype on upload; outputs keep the engine's dtype.
    """

    def __init__(self, engine: trt.ICudaEngine, device: Device) -> None:
        """Allocate buffers for every I/O tensor of `engine` on `device`."""
        device.set_current()
        self._stream = device.create_stream()
        self._context = engine.create_execution_context()
        self._graph = None
        pinned = PinnedMemoryResource()
        self._device_buffers = {}
        self._host_buffers = {}
        self.host: dict[str, npt.NDArray[np.generic]] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            shape = tuple(engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                msg = (
                    f"tensor {name} has a dynamic shape {shape}; only static engines are supported"
                )
                raise ValueError(msg)
            dtype = np.dtype(NUMPY_DTYPES[engine.get_tensor_dtype(name)])
            size = int(np.prod(shape)) * dtype.itemsize
            self._device_buffers[name] = device.allocate(size, stream=self._stream)
            self._host_buffers[name] = pinned.allocate(size, stream=self._stream)
            self.host[name] = np.from_dlpack(self._host_buffers[name]).view(dtype).reshape(shape)
            self._context.set_tensor_address(name, int(self._device_buffers[name].handle))
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_input else self.outputs).append(name)
        self._stream.sync()

    def upload(self, inputs: Mapping[str, npt.ArrayLike]) -> None:
        """Copy `inputs` (one array per input tensor) to the device."""
        if set(inputs) != set(self.inputs):
            msg = f"inputs {sorted(inputs)} do not match the engine's {sorted(self.inputs)}"
            raise ValueError(msg)
        for name, value in inputs.items():
            self.host[name][...] = value
            self._device_buffers[name].copy_from(self._host_buffers[name], stream=self._stream)

    def execute(self) -> None:
        """Run the engine on the uploaded inputs and wait for it to finish."""
        if self._graph is not None:
            self._graph.launch(self._stream)
        elif not self._context.execute_async_v3(self._stream.handle):
            msg = "execute_async_v3 failed"
            raise RuntimeError(msg)
        self._stream.sync()

    def download(self) -> dict[str, npt.NDArray[np.generic]]:
        """Copies of every output tensor."""
        for name in self.outputs:
            self._host_buffers[name].copy_from(self._device_buffers[name], stream=self._stream)
        self._stream.sync()
        return {name: self.host[name].copy() for name in self.outputs}

    def run(self, inputs: Mapping[str, npt.ArrayLike]) -> dict[str, npt.NDArray[np.generic]]:
        """Upload `inputs`, execute, download the outputs."""
        self.upload(inputs)
        self.execute()
        return self.download()

    def capture_graph(self) -> None:
        """Replace later `execute` calls by a CUDA graph launch of the engine."""
        if not self._context.execute_async_v3(self._stream.handle):
            msg = "warm-up execute_async_v3 failed"
            raise RuntimeError(msg)
        self._stream.sync()
        builder = self._stream.create_graph_builder()
        builder.begin_building()
        if not self._context.execute_async_v3(builder.stream.handle):
            msg = "execute_async_v3 failed during graph capture"
            raise RuntimeError(msg)
        builder.end_building()
        graph = builder.complete()
        graph.upload(self._stream)
        self._graph = graph
