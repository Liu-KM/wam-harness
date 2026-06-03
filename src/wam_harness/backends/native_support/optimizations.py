from __future__ import annotations

from dataclasses import dataclass

from wam_harness.core.types import Manifest, OptimizationProfile


@dataclass(frozen=True)
class NativeOptimizationStatus:
    name: str
    enabled: bool
    params: dict[str, object]
    declared_supported: bool
    scope: str | None
    target: str
    state: str

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "enabled": self.enabled,
            "params": self.params,
            "declared_supported": self.declared_supported,
            "target": self.target,
            "state": self.state,
        }
        if self.scope is not None:
            payload["scope"] = self.scope
        return payload


def native_optimization_statuses(
    manifest: Manifest,
    profiles: list[OptimizationProfile],
) -> list[NativeOptimizationStatus]:
    supported = set(manifest.supported_optimizations)
    return [
        NativeOptimizationStatus(
            name=profile.name,
            enabled=profile.enabled,
            params=dict(profile.params),
            declared_supported=profile.name in supported,
            scope=_profile_scope(manifest, profile.name),
            target=_profile_target(_profile_scope(manifest, profile.name)),
            state=_profile_state(profile, profile.name in supported),
        )
        for profile in profiles
    ]


def native_optimization_status_dicts(
    manifest: Manifest,
    profiles: list[OptimizationProfile],
) -> list[dict[str, object]]:
    return [status.to_dict() for status in native_optimization_statuses(manifest, profiles)]


def _profile_scope(manifest: Manifest, name: str) -> str | None:
    profiles = manifest.optimizations.get("profiles", {})
    if not isinstance(profiles, dict):
        return None
    raw = profiles.get(name)
    if not isinstance(raw, dict):
        return None
    scope = raw.get("scope")
    return str(scope) if scope is not None else None


def _profile_target(scope: str | None) -> str:
    if scope in {"replan", "simulator_eval", "workload"}:
        return "workload"
    if scope in {"multi_gpu", "deployment"}:
        return "deployment"
    return "native_backend"


def _profile_state(profile: OptimizationProfile, declared_supported: bool) -> str:
    if not declared_supported:
        return "unsupported_by_manifest"
    if not profile.enabled:
        return "disabled"
    return "requested"
