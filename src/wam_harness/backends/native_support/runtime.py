from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from wam_harness.core.manifest import manifest_from_dict
from wam_harness.core.types import Manifest


class NativeRuntimeError(RuntimeError):
    """Raised when a reference model entry cannot be mapped to a native runtime."""


@dataclass(frozen=True)
class NativeRuntimeSpec:
    mode: str
    workload_name: str
    workload_config: dict[str, object] = field(default_factory=dict)
    require_native_backend: bool = False


@dataclass(frozen=True)
class NativeRuntimePlan:
    reference_manifest: Manifest
    manifest: Manifest
    mode: str
    workload_name: str
    native_backend: str | None
    native_migration: bool


NATIVE_RUN_SPEC = NativeRuntimeSpec(
    mode="native_run",
    workload_name="processor_smoke",
    workload_config={
        "synthetic_observation": True,
        "episode_length": 1,
    },
)
NATIVE_INPUT_RUN_SPEC = NativeRuntimeSpec(
    mode="native_run",
    workload_name="single_observation",
    workload_config={
        "external_observation": True,
        "episode_length": 1,
    },
)
NATIVE_SMOKE_SPEC = NativeRuntimeSpec(
    mode="native_smoke",
    workload_name="native_smoke",
    workload_config={"synthetic_observation": True},
    require_native_backend=True,
)
NATIVE_SERVE_SPEC = NativeRuntimeSpec(
    mode="native_serve",
    workload_name="serve",
    workload_config={"external_observation": True},
)
NATIVE_PREPARE_SPEC = NativeRuntimeSpec(
    mode="native_prepare",
    workload_name="native_prepare",
)
NATIVE_DOCTOR_SPEC = NativeRuntimeSpec(
    mode="native_doctor",
    workload_name="native_doctor",
)


def native_backend_name(manifest: Manifest) -> str | None:
    config = manifest.backend.get("config", {})
    if not isinstance(config, dict):
        return None
    name = config.get("native_backend")
    return str(name) if name is not None else None


def resolve_native_runtime(
    manifest: Manifest,
    spec: NativeRuntimeSpec,
    *,
    upstream_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    backend_overrides: dict[str, str] | None = None,
) -> NativeRuntimePlan:
    _validate_spec(spec)
    backend_name = native_backend_name(manifest)
    if backend_name is None:
        if spec.require_native_backend:
            raise NativeRuntimeError(
                f"{manifest.id} does not declare backend.config.native_backend for native runtime"
            )
        return NativeRuntimePlan(
            reference_manifest=manifest,
            manifest=manifest,
            mode=str(manifest.backend.get("mode", spec.mode)),
            workload_name=manifest.workload_name,
            native_backend=None,
            native_migration=False,
        )

    runtime_manifest = native_runtime_manifest(
        manifest,
        mode=spec.mode,
        workload_name=spec.workload_name,
        workload_config=spec.workload_config,
        upstream_dir=upstream_dir,
        cache_dir=cache_dir,
        backend_overrides=backend_overrides,
    )
    return NativeRuntimePlan(
        reference_manifest=manifest,
        manifest=runtime_manifest,
        mode=spec.mode,
        workload_name=spec.workload_name,
        native_backend=backend_name,
        native_migration=True,
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


def _validate_spec(spec: NativeRuntimeSpec) -> None:
    if not spec.mode:
        raise NativeRuntimeError("native runtime spec mode must not be empty")
    if not spec.workload_name:
        raise NativeRuntimeError("native runtime spec workload_name must not be empty")
