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
        self.optimization_statuses: dict[str, dict[str, object]] = {}

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

    def runtime_contract(self, *, processor: object | None = None) -> dict[str, object] | None:
        return None

    def preflight(self) -> None:
        return None

    def action_contract_enabled(self) -> bool:
        return False

    def apply_optimization_profiles(
        self,
        profiles: list[OptimizationProfile],
    ) -> list[dict[str, object]]:
        statuses: list[dict[str, object]] = []
        supported = set(self.manifest.supported_optimizations)
        for profile in profiles:
            state = "requested"
            hook = None
            reason = None
            if profile.name not in supported:
                state = "unsupported_by_manifest"
                reason = "not_declared_in_manifest"
            elif not profile.enabled:
                state = "disabled"
            elif profile.name == "fake_cache":
                state = "applied"
                hook = "fake_backend_latency_model"
            elif profile.name == "action_chunk_scheduling":
                state = "applied"
                hook = "open_loop_replan_schedule"
            else:
                reason = "no_fake_backend_hook"
            status = {
                "name": profile.name,
                "enabled": profile.enabled,
                "params": dict(profile.params),
                "declared_supported": profile.name in supported,
                "state": state,
                "target": "fake_backend",
            }
            if hook is not None:
                status["hook"] = hook
            if reason is not None:
                status["reason"] = reason
            statuses.append(status)
            self.optimization_statuses[profile.name] = status
        return statuses

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
