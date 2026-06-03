from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]


def _clean_dict(data: JsonDict) -> JsonDict:
    return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class OptimizationProfile:
    name: str
    enabled: bool = True
    params: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "params": self.params,
        }


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    id: str
    display_name: str
    source: JsonDict
    assets: JsonDict
    backend: JsonDict
    processor: JsonDict
    workload: JsonDict
    defaults: JsonDict
    optimizations: JsonDict
    deployment: JsonDict = field(default_factory=dict)
    eval: JsonDict = field(default_factory=dict)
    known_gaps: list[str] = field(default_factory=list)

    @property
    def backend_name(self) -> str:
        return str(self.backend["name"])

    @property
    def processor_name(self) -> str:
        return str(self.processor["name"])

    @property
    def workload_name(self) -> str:
        return str(self.workload["name"])

    @property
    def source_repo(self) -> str | None:
        repo = self.source.get("repo")
        return str(repo) if repo is not None else None

    @property
    def supported_optimizations(self) -> list[str]:
        supported = self.optimizations.get("supported", [])
        return [str(item) for item in supported]

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "display_name": self.display_name,
            "source": self.source,
            "assets": self.assets,
            "backend": self.backend,
            "processor": self.processor,
            "workload": self.workload,
            "defaults": self.defaults,
            "optimizations": self.optimizations,
            "deployment": self.deployment,
            "eval": self.eval,
            "known_gaps": self.known_gaps,
        }


@dataclass(frozen=True)
class Observation:
    images: JsonDict
    prompt: str
    state: JsonDict = field(default_factory=dict)
    history: list[JsonDict] = field(default_factory=list)
    session: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "images": self.images,
            "prompt": self.prompt,
            "state": self.state,
            "history": self.history,
            "session": self.session,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class InferenceRequest:
    observation: Observation
    action_horizon: int
    replan_steps: int
    optimization_profiles: list[OptimizationProfile] = field(default_factory=list)
    reset: bool = False
    runtime_options: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "observation": self.observation.to_dict(),
            "action_horizon": self.action_horizon,
            "replan_steps": self.replan_steps,
            "optimization_profiles": [profile.to_dict() for profile in self.optimization_profiles],
            "reset": self.reset,
            "runtime_options": self.runtime_options,
        }


@dataclass(frozen=True)
class ActionChunk:
    actions: list[list[float]]

    @property
    def horizon(self) -> int:
        return len(self.actions)

    @property
    def action_dim(self) -> int:
        if not self.actions:
            return 0
        return len(self.actions[0])

    def to_dict(self) -> JsonDict:
        return {
            "actions": self.actions,
            "shape": [self.horizon, self.action_dim],
        }


@dataclass(frozen=True)
class RuntimeInfo:
    manifest_id: str
    model_name: str
    backend: str
    processor: str
    source_repo: str | None
    mode: str
    device: str
    dtype: str
    optimization_profiles: list[OptimizationProfile] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return _clean_dict(
            {
                "manifest_id": self.manifest_id,
                "model_name": self.model_name,
                "backend": self.backend,
                "processor": self.processor,
                "source_repo": self.source_repo,
                "mode": self.mode,
                "device": self.device,
                "dtype": self.dtype,
                "optimization_profiles": [
                    profile.to_dict() for profile in self.optimization_profiles
                ],
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class InferenceResult:
    action_chunk: ActionChunk
    warnings: list[str] = field(default_factory=list)
    backend_metadata: JsonDict = field(default_factory=dict)
    timing: JsonDict = field(default_factory=dict)
    memory: JsonDict = field(default_factory=dict)
    future_frames: JsonDict | None = None
    value: JsonDict | float | int | None = None

    def to_dict(self) -> JsonDict:
        return _clean_dict(
            {
                "action_chunk": self.action_chunk.to_dict(),
                "future_frames": self.future_frames,
                "value": self.value,
                "warnings": self.warnings,
                "backend_metadata": self.backend_metadata,
                "timing": self.timing,
                "memory": self.memory,
            }
        )


@dataclass(frozen=True)
class TraceEvent:
    schema_version: int
    event: str
    timestamp: str
    run_id: str
    payload: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "event": self.event,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            **self.payload,
        }
