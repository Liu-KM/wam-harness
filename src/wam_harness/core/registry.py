from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from wam_harness.core.manifest import load_builtin_manifest
from wam_harness.core.runtime import (
    RuntimeOptions,
    RuntimePlan,
    RuntimeResolutionError,
    RuntimeResolver,
    RuntimeSpec,
    default_runtime_plan,
    validate_runtime_spec,
)
from wam_harness.core.types import (
    InferenceRequest,
    InferenceResult,
    Manifest,
    Observation,
    OptimizationProfile,
    RuntimeInfo,
)


class RegistryError(LookupError):
    """Raised when a registry key is unknown or incompatible."""


class Backend(Protocol):
    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def reset(self) -> None: ...

    def infer(self, request: InferenceRequest) -> InferenceResult: ...

    def runtime_info(self) -> RuntimeInfo: ...

    def close(self) -> None: ...


class Processor(Protocol):
    def to_model_inputs(self, observation: Observation) -> object: ...

    def to_harness_result(self, raw_output: object) -> InferenceResult: ...

    def modality_limits(self) -> dict[str, object]: ...

    def smoke_observation(self) -> Observation: ...


BackendFactory = Callable[[Manifest, list[OptimizationProfile]], Backend]
ProcessorFactory = Callable[[Manifest], Processor]
WorkloadFactory = Callable[[Manifest], object]


@dataclass
class Registry:
    backends: dict[str, BackendFactory] = field(default_factory=dict)
    processors: dict[str, ProcessorFactory] = field(default_factory=dict)
    workloads: dict[str, WorkloadFactory] = field(default_factory=dict)
    optimization_defaults: dict[str, dict[str, object]] = field(default_factory=dict)
    runtime_resolvers: list[RuntimeResolver] = field(default_factory=list)

    def register_backend(self, name: str, factory: BackendFactory) -> None:
        self.backends[name] = factory

    def register_processor(self, name: str, factory: ProcessorFactory) -> None:
        self.processors[name] = factory

    def register_workload(self, name: str, factory: WorkloadFactory) -> None:
        self.workloads[name] = factory

    def register_optimization(self, name: str, defaults: dict[str, object] | None = None) -> None:
        self.optimization_defaults[name] = defaults or {}

    def register_runtime_resolver(self, resolver: RuntimeResolver) -> None:
        self.runtime_resolvers.append(resolver)

    def load_manifest(self, model_id: str) -> Manifest:
        return load_builtin_manifest(model_id)

    def resolve_runtime(
        self,
        manifest: Manifest,
        spec: RuntimeSpec,
        *,
        upstream_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        backend_overrides: dict[str, str] | None = None,
    ) -> RuntimePlan:
        validate_runtime_spec(spec)
        options = RuntimeOptions(
            upstream_dir=upstream_dir,
            cache_dir=cache_dir,
            backend_overrides=backend_overrides or {},
        )
        for resolver in self.runtime_resolvers:
            plan = resolver(manifest, spec, options)
            if plan is not None:
                return plan
        if spec.require_backend_mapping:
            raise RuntimeResolutionError(
                f"{manifest.id} does not declare a backend mapping for {spec.mode}"
            )
        return default_runtime_plan(manifest, spec)

    def build_optimization_profiles(
        self, manifest: Manifest, enabled_names: list[str]
    ) -> list[OptimizationProfile]:
        profiles: list[OptimizationProfile] = []
        supported = set(manifest.supported_optimizations)
        manifest_profiles = manifest.optimizations.get("profiles", {})
        for name in enabled_names:
            if name not in supported:
                supported_text = ", ".join(sorted(supported)) or "<none>"
                raise RegistryError(
                    f"optimization profile '{name}' is not supported by {manifest.id}; "
                    f"supported profiles: {supported_text}"
                )
            defaults = dict(self.optimization_defaults.get(name, {}))
            profile_defaults = manifest_profiles.get(name, {})
            if isinstance(profile_defaults, dict):
                defaults.update(profile_defaults.get("params", {}))
            profiles.append(OptimizationProfile(name=name, enabled=True, params=defaults))
        return profiles

    def create_backend(
        self, manifest: Manifest, profiles: list[OptimizationProfile]
    ) -> Backend:
        try:
            factory = self.backends[manifest.backend_name]
        except KeyError as exc:
            raise RegistryError(f"unknown backend: {manifest.backend_name}") from exc
        return factory(manifest, profiles)

    def create_processor(self, manifest: Manifest) -> Processor:
        try:
            factory = self.processors[manifest.processor_name]
        except KeyError as exc:
            raise RegistryError(f"unknown processor: {manifest.processor_name}") from exc
        return factory(manifest)

    def create_workload(self, manifest: Manifest) -> object:
        try:
            factory = self.workloads[manifest.workload_name]
        except KeyError as exc:
            raise RegistryError(f"unknown workload: {manifest.workload_name}") from exc
        return factory(manifest)


def default_registry() -> Registry:
    from wam_harness.defaults import default_registry as build_default_registry

    return build_default_registry()
