from __future__ import annotations

from wam_harness.core.types import (
    ActionChunk,
    InferenceRequest,
    InferenceResult,
    Manifest,
    OptimizationProfile,
    RuntimeInfo,
)


class FakeBackend:
    def __init__(self, manifest: Manifest, profiles: list[OptimizationProfile]) -> None:
        self.manifest = manifest
        self.profiles = profiles
        self.loaded = False
        self.warmed = False
        self.reset_count = 0
        config = manifest.backend.get("config", {})
        self.action_dim = int(config.get("action_dim", 4))

    def load(self) -> None:
        self.loaded = True

    def warmup(self) -> None:
        if not self.loaded:
            raise RuntimeError("fake backend must be loaded before warmup")
        self.warmed = True

    def reset(self) -> None:
        if not self.loaded:
            raise RuntimeError("fake backend must be loaded before reset")
        self.reset_count += 1

    def close(self) -> None:
        self.warmed = False
        self.loaded = False

    def runtime_info(self) -> RuntimeInfo:
        defaults = self.manifest.defaults
        return RuntimeInfo(
            manifest_id=self.manifest.id,
            model_name=self.manifest.display_name,
            backend=self.manifest.backend_name,
            processor=self.manifest.processor_name,
            source_repo=self.manifest.source_repo,
            mode=str(self.manifest.backend.get("mode", "fake")),
            device=str(defaults.get("device", "cpu")),
            dtype=str(defaults.get("dtype", "fp32")),
            optimization_profiles=self.profiles,
            metadata={"fake": True},
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        if not self.warmed:
            raise RuntimeError("fake backend must be warmed before inference")

        step_id = int(request.observation.session.get("step_id", 0))
        episode_id = int(request.observation.session.get("episode_id", 0))
        fake_cache_enabled = any(profile.name == "fake_cache" for profile in self.profiles)
        actions: list[list[float]] = []
        for horizon_index in range(request.action_horizon):
            row = []
            for dim in range(self.action_dim):
                value = episode_id * 1000 + step_id * 100 + horizon_index * 10 + dim
                row.append(float(value) / 1000.0)
            actions.append(row)

        model_ms = 0.05 if fake_cache_enabled else 0.1
        return InferenceResult(
            action_chunk=ActionChunk(actions=actions),
            backend_metadata={
                "fake_cache_enabled": fake_cache_enabled,
                "reset_count": self.reset_count,
            },
            timing={
                "preprocess_ms": 0.01,
                "model_ms": model_ms,
                "postprocess_ms": 0.01,
                "total_ms": 0.02 + model_ms,
            },
            memory={},
        )
