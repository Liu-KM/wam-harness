from __future__ import annotations

from typing import Any, Protocol

from wam_harness.backends.native_support.optimizations import (
    native_optimization_status_dicts,
)
from wam_harness.core.types import Manifest, OptimizationProfile


class NativeContractProcessor(Protocol):
    def modality_limits(self) -> dict[str, object]: ...


class NativeContractBackend(Protocol):
    def native_runtime_mode(self) -> str | None: ...

    def native_runtime_loader_name(self) -> str | None: ...

    def native_model_adapter_name(self) -> str | None: ...

    def optimization_status_overrides(self) -> dict[str, dict[str, object]]: ...


def native_runtime_contract_payload(
    manifest: Manifest,
    profiles: list[OptimizationProfile],
    *,
    processor: object | None = None,
    backend: object | None = None,
) -> dict[str, object] | None:
    """Describe the native runtime contract recorded before backend load."""

    mode = str(manifest.backend.get("mode", ""))
    payload: dict[str, object] = {
        "native": True,
        "backend": manifest.backend_name,
        "processor": manifest.processor_name,
        "workload": manifest.workload_name,
        "mode": mode,
        "supported_optimizations": manifest.supported_optimizations,
        "optimization_profile_status": native_optimization_status_dicts(
            manifest,
            profiles,
            applied_statuses=_backend_optimization_status_overrides(backend),
        ),
        "deployment": dict(manifest.deployment),
        "backend_config_keys": sorted(
            str(key)
            for key in _mapping_or_empty(manifest.backend.get("config")).keys()
        ),
    }
    runtime_mode = _backend_runtime_mode(backend)
    model_adapter = _backend_model_adapter_name(backend)
    runtime_loader = _backend_runtime_loader_name(backend)
    if runtime_mode is not None:
        payload["runtime_mode"] = runtime_mode
    if runtime_loader is not None:
        payload["runtime_loader"] = runtime_loader
    if model_adapter is not None:
        payload["model_adapter"] = model_adapter
    modality = _processor_modality(processor)
    if modality is not None:
        payload["processor_modality"] = modality
    return payload


def _processor_modality(processor: object | None) -> dict[str, object] | None:
    if processor is None or not hasattr(processor, "modality_limits"):
        return None
    modality = _as_contract_processor(processor).modality_limits()
    return dict(modality) if isinstance(modality, dict) else None


def _as_contract_processor(processor: object) -> NativeContractProcessor:
    return processor  # type: ignore[return-value]


def _backend_model_adapter_name(backend: object | None) -> str | None:
    if backend is None or not hasattr(backend, "native_model_adapter_name"):
        return None
    name = _as_contract_backend(backend).native_model_adapter_name()
    return str(name) if name is not None else None


def _backend_runtime_loader_name(backend: object | None) -> str | None:
    if backend is None or not hasattr(backend, "native_runtime_loader_name"):
        return None
    name = _as_contract_backend(backend).native_runtime_loader_name()
    return str(name) if name is not None else None


def _backend_runtime_mode(backend: object | None) -> str | None:
    if backend is None or not hasattr(backend, "native_runtime_mode"):
        return None
    mode = _as_contract_backend(backend).native_runtime_mode()
    return str(mode) if mode is not None else None


def _backend_optimization_status_overrides(
    backend: object | None,
) -> dict[str, dict[str, object]]:
    if backend is None or not hasattr(backend, "optimization_status_overrides"):
        return {}
    value = _as_contract_backend(backend).optimization_status_overrides()
    return dict(value) if isinstance(value, dict) else {}


def _as_contract_backend(backend: object) -> NativeContractBackend:
    return backend  # type: ignore[return-value]


def _mapping_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
