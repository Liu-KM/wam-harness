from __future__ import annotations

from pathlib import Path

from eazywam.core.manifest import manifest_from_dict
from eazywam.core.runtime import (
    RuntimeOptions,
    RuntimePlan,
    RuntimeResolutionError,
    RuntimeSpec,
    validate_runtime_spec,
)
from eazywam.core.types import Manifest


class NativeRuntimeError(RuntimeResolutionError):
    """Raised when a reference model entry cannot be mapped to a native runtime."""


def native_backend_name(manifest: Manifest) -> str | None:
    config = manifest.backend.get("config", {})
    if not isinstance(config, dict):
        return None
    name = config.get("native_backend")
    return str(name) if name is not None else None


def native_runtime_resolver(
    manifest: Manifest,
    spec: RuntimeSpec,
    options: RuntimeOptions,
) -> RuntimePlan | None:
    validate_runtime_spec(spec)
    backend_name = native_backend_name(manifest)
    if backend_name is None:
        return None
    runtime_manifest = native_runtime_manifest(
        manifest,
        mode=spec.mode,
        workload_name=spec.workload_name,
        workload_config=spec.workload_config,
        upstream_dir=options.upstream_dir,
        cache_dir=options.cache_dir,
        backend_overrides=options.backend_overrides,
    )
    return RuntimePlan(
        reference_manifest=manifest,
        manifest=runtime_manifest,
        mode=spec.mode,
        workload_name=spec.workload_name,
        mapped_backend=backend_name,
        transformed=True,
    )


def native_runtime_manifest(
    manifest: Manifest,
    *,
    mode: str,
    workload_name: str,
    workload_config: dict[str, object] | None = None,
    upstream_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    backend_overrides: dict[str, str] | None = None,
) -> Manifest:
    reference_backend_config = dict(manifest.backend.get("config", {}))
    backend_name = reference_backend_config.pop("native_backend", None)
    if backend_name is None:
        raise NativeRuntimeError(
            f"{manifest.id} does not declare backend.config.native_backend for native runtime"
        )

    backend_config = reference_backend_config
    if upstream_dir is not None:
        backend_config["upstream_dir"] = str(upstream_dir)
    if cache_dir is not None:
        backend_config["cache_dir"] = str(cache_dir)
    backend_config.update(backend_overrides or {})

    data = manifest.to_dict()
    data["backend"] = {
        "name": str(backend_name),
        "mode": mode,
        "config": backend_config,
    }
    data["workload"] = {
        "name": workload_name,
        "config": dict(workload_config or {}),
    }
    return manifest_from_dict(data)
