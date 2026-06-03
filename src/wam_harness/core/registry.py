from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from wam_harness.core.manifest import load_builtin_manifest
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

    def register_backend(self, name: str, factory: BackendFactory) -> None:
        self.backends[name] = factory

    def register_processor(self, name: str, factory: ProcessorFactory) -> None:
        self.processors[name] = factory

    def register_workload(self, name: str, factory: WorkloadFactory) -> None:
        self.workloads[name] = factory

    def register_optimization(self, name: str, defaults: dict[str, object] | None = None) -> None:
        self.optimization_defaults[name] = defaults or {}

    def load_manifest(self, model_id: str) -> Manifest:
        return load_builtin_manifest(model_id)

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
    from wam_harness.backends.cosmos_policy import CosmosPolicyBackend
    from wam_harness.backends.dreamzero import DreamZeroBackend
    from wam_harness.backends.fastwam import FastWAMBackend
    from wam_harness.backends.fake import FakeBackend
    from wam_harness.processors.cosmos_policy_libero import CosmosPolicyLiberoProcessor
    from wam_harness.processors.dreamzero_droid import DreamZeroDroidProcessor
    from wam_harness.processors.fastwam_libero import FastWAMLiberoProcessor
    from wam_harness.processors.passthrough import PassthroughProcessor
    from wam_harness.workloads.open_loop import OpenLoopWorkload

    registry = Registry()
    registry.register_backend("fake", FakeBackend)
    registry.register_backend("cosmos_policy", CosmosPolicyBackend)
    registry.register_backend("dreamzero", DreamZeroBackend)
    registry.register_backend("fastwam", FastWAMBackend)
    registry.register_processor("passthrough", PassthroughProcessor.from_manifest)
    registry.register_processor("fastwam_libero", FastWAMLiberoProcessor.from_manifest)
    registry.register_processor(
        "cosmos_policy_libero",
        CosmosPolicyLiberoProcessor.from_manifest,
    )
    registry.register_processor("dreamzero_droid", DreamZeroDroidProcessor.from_manifest)
    registry.register_workload("open_loop", OpenLoopWorkload.from_manifest)
    registry.register_optimization("fake_cache", {"cache_scope": "replan"})
    registry.register_optimization("action_chunk_scheduling", {})
    registry.register_optimization("dit_cache", {})
    registry.register_optimization("jpeg_observation_compression", {})
    registry.register_optimization("parallel_inference", {})
    return registry
