from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CudaGraphRunResult:
    output: torch.Tensor | None
    shape_key: dict[str, Any] | None
    capture_success: bool
    replayed: bool
    fallback_reason: str | None = None
    capture_wall_ms: float | None = None


@dataclass
class _GraphEntry:
    signature: tuple[Any, ...]
    shape_key: dict[str, Any]
    static_inputs: dict[str, Any]
    graph: Any
    output: torch.Tensor
    capture_wall_ms: float
    replay_count: int = 0


class ActionBodyCudaGraphManager:
    """CUDA Graph wrapper for FastWAM's action-body MoT call."""

    def __init__(self, *, warmup_iters: int = 1) -> None:
        self.warmup_iters = max(0, int(warmup_iters))
        self._lock = threading.Lock()
        self._entry: _GraphEntry | None = None

    def reset(self) -> None:
        with self._lock:
            self._entry = None

    def run(
        self,
        fn: Callable[..., torch.Tensor],
        *,
        shape_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CudaGraphRunResult:
        shape_key = _shape_key(kwargs, metadata=shape_metadata or {})
        fallback_reason = _static_cuda_fallback_reason(kwargs)
        if fallback_reason is not None:
            return CudaGraphRunResult(
                output=None,
                shape_key=shape_key,
                capture_success=False,
                replayed=False,
                fallback_reason=fallback_reason,
            )

        signature = _shape_signature(shape_key)
        with self._lock:
            entry = self._entry
            if entry is None or entry.signature != signature:
                return self._capture(fn, kwargs=kwargs, shape_key=shape_key, signature=signature)
            try:
                _copy_tree(entry.static_inputs, kwargs)
                entry.graph.replay()
            except Exception as exc:  # pragma: no cover - requires CUDA failure path.
                self._entry = None
                return CudaGraphRunResult(
                    output=None,
                    shape_key=shape_key,
                    capture_success=False,
                    replayed=False,
                    fallback_reason=f"replay_failed:{type(exc).__name__}",
                )
            entry.replay_count += 1
            return CudaGraphRunResult(
                output=entry.output,
                shape_key=shape_key,
                capture_success=True,
                replayed=True,
                capture_wall_ms=entry.capture_wall_ms,
            )

    def _capture(
        self,
        fn: Callable[..., torch.Tensor],
        *,
        kwargs: dict[str, Any],
        shape_key: dict[str, Any],
        signature: tuple[Any, ...],
    ) -> CudaGraphRunResult:
        start = time.perf_counter()
        try:
            static_inputs = _clone_tree(kwargs)
            device = _first_cuda_device(static_inputs)
            if device is None:
                return CudaGraphRunResult(
                    output=None,
                    shape_key=shape_key,
                    capture_success=False,
                    replayed=False,
                    fallback_reason="cuda_device_unavailable",
                )

            with torch.cuda.device(device), torch.no_grad():
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(self.warmup_iters):
                        fn(**static_inputs)
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fn(**static_inputs)
                torch.cuda.synchronize(device)
        except Exception as exc:
            self._entry = None
            return CudaGraphRunResult(
                output=None,
                shape_key=shape_key,
                capture_success=False,
                replayed=False,
                fallback_reason=f"capture_failed:{type(exc).__name__}",
            )

        capture_wall_ms = (time.perf_counter() - start) * 1000
        self._entry = _GraphEntry(
            signature=signature,
            shape_key=shape_key,
            static_inputs=static_inputs,
            graph=graph,
            output=output,
            capture_wall_ms=capture_wall_ms,
        )
        return CudaGraphRunResult(
            output=output,
            shape_key=shape_key,
            capture_success=True,
            replayed=False,
            capture_wall_ms=capture_wall_ms,
        )


def _static_cuda_fallback_reason(value: Any) -> str | None:
    tensors = list(_iter_tensors(value))
    if not tensors:
        return "no_tensor_inputs"
    if not bool(torch.cuda.is_available()):
        return "cuda_unavailable"
    devices = {tensor.device for tensor in tensors}
    if any(device.type != "cuda" for device in devices):
        return "non_cuda_tensor"
    if len(devices) > 1:
        return "multiple_cuda_devices"
    return None


def _first_cuda_device(value: Any) -> torch.device | None:
    for tensor in _iter_tensors(value):
        if tensor.device.type == "cuda":
            return tensor.device
    return None


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _iter_tensors(item)


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        cloned = torch.empty_like(value, memory_format=torch.contiguous_format)
        cloned.copy_(value)
        return cloned
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _copy_tree(destination: Any, source: Any) -> None:
    if isinstance(destination, torch.Tensor) and isinstance(source, torch.Tensor):
        if (
            tuple(destination.shape) != tuple(source.shape)
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            raise ValueError("CUDA graph static tensor shape/dtype/device changed")
        destination.copy_(source, non_blocking=True)
        return
    if isinstance(destination, dict) and isinstance(source, dict):
        if set(destination) != set(source):
            raise ValueError("CUDA graph static dict keys changed")
        for key in destination:
            _copy_tree(destination[key], source[key])
        return
    if isinstance(destination, list) and isinstance(source, list):
        if len(destination) != len(source):
            raise ValueError("CUDA graph static list length changed")
        for left, right in zip(destination, source):
            _copy_tree(left, right)
        return
    if isinstance(destination, tuple) and isinstance(source, tuple):
        if len(destination) != len(source):
            raise ValueError("CUDA graph static tuple length changed")
        for left, right in zip(destination, source):
            _copy_tree(left, right)
        return
    if destination != source:
        raise ValueError("CUDA graph static non-tensor value changed")


def _shape_key(value: Any, *, metadata: dict[str, Any]) -> dict[str, Any]:
    key = {
        "metadata": _jsonable(metadata),
        "inputs": _shape_descriptor(value),
    }
    first_device = _first_cuda_device(value)
    if first_device is not None:
        key["device"] = str(first_device)
    return key


def _shape_descriptor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): _shape_descriptor(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_shape_descriptor(item) for item in value]
    if isinstance(value, tuple):
        return [_shape_descriptor(item) for item in value]
    return _jsonable(value)


def _shape_signature(shape_key: dict[str, Any]) -> tuple[Any, ...]:
    return _freeze_jsonable(shape_key)


def _freeze_jsonable(value: Any) -> tuple[Any, ...] | str | int | float | bool | None:
    if isinstance(value, dict):
        return tuple((str(key), _freeze_jsonable(item)) for key, item in sorted(value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_jsonable(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.dtype | torch.device):
        return str(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
