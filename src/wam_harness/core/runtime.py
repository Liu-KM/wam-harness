from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from wam_harness.core.types import Manifest


class RuntimeResolutionError(RuntimeError):
    """Raised when a model entry cannot be mapped to the requested runtime."""


@dataclass(frozen=True)
class RuntimeSpec:
    mode: str
    workload_name: str
    workload_config: dict[str, object] = field(default_factory=dict)
    require_backend_mapping: bool = False


@dataclass(frozen=True)
class RuntimeOptions:
    upstream_dir: str | Path | None = None
    cache_dir: str | Path | None = None
    backend_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimePlan:
    reference_manifest: Manifest
    manifest: Manifest
    mode: str
    workload_name: str
    mapped_backend: str | None
    transformed: bool


RuntimeResolver = Callable[[Manifest, RuntimeSpec, RuntimeOptions], RuntimePlan | None]


RUN_SPEC = RuntimeSpec(
    mode="run",
    workload_name="processor_smoke",
    workload_config={
        "synthetic_observation": True,
        "episode_length": 1,
    },
)
INPUT_RUN_SPEC = RuntimeSpec(
    mode="run",
    workload_name="single_observation",
    workload_config={
        "external_observation": True,
        "episode_length": 1,
    },
)
SERVE_SPEC = RuntimeSpec(
    mode="serve",
    workload_name="serve",
    workload_config={"external_observation": True},
)
PREPARE_SPEC = RuntimeSpec(
    mode="prepare",
    workload_name="prepare",
)
DOCTOR_SPEC = RuntimeSpec(
    mode="doctor",
    workload_name="doctor",
)


def default_runtime_plan(manifest: Manifest, spec: RuntimeSpec) -> RuntimePlan:
    return RuntimePlan(
        reference_manifest=manifest,
        manifest=manifest,
        mode=str(manifest.backend.get("mode", spec.mode)),
        workload_name=manifest.workload_name,
        mapped_backend=None,
        transformed=False,
    )


def validate_runtime_spec(spec: RuntimeSpec) -> None:
    if not spec.mode:
        raise RuntimeResolutionError("runtime spec mode must not be empty")
    if not spec.workload_name:
        raise RuntimeResolutionError("runtime spec workload_name must not be empty")
