from __future__ import annotations

from typing import Protocol

from wam_harness.core.preflight import PreflightReport
from wam_harness.core.types import OptimizationProfile


class RuntimeContractBackend(Protocol):
    def runtime_contract(self, *, processor: object | None = None) -> dict[str, object] | None: ...


class PreflightBackend(Protocol):
    def preflight(self) -> PreflightReport | None: ...


class ActionContractBackend(Protocol):
    def action_contract_enabled(self) -> bool: ...


class ProcessorAttachBackend(Protocol):
    def attach_processor(self, processor: object) -> None: ...


class OptimizationPlanBackend(Protocol):
    def plan_optimization_profiles(
        self,
        profiles: list[OptimizationProfile],
    ) -> list[dict[str, object]] | None: ...


class OptimizationApplyBackend(Protocol):
    def apply_loaded_optimization_profiles(
        self,
        profiles: list[OptimizationProfile],
    ) -> list[dict[str, object]] | None: ...


def runtime_contract_payload(
    backend: object,
    *,
    processor: object | None = None,
) -> dict[str, object] | None:
    method = getattr(backend, "runtime_contract", None)
    if not callable(method):
        return None
    payload = method(processor=processor)
    return dict(payload) if isinstance(payload, dict) else None


def preflight_report(backend: object) -> PreflightReport | None:
    method = getattr(backend, "preflight", None)
    if not callable(method):
        return None
    report = method()
    if report is None:
        return None
    if isinstance(report, PreflightReport):
        return report
    if hasattr(report, "to_dict"):
        payload = report.to_dict()
        if isinstance(payload, dict):
            return PreflightReport(
                status=str(payload.get("status", "unknown")),
                payload=dict(payload),
            )
    if isinstance(report, dict):
        return PreflightReport(
            status=str(report.get("status", "unknown")),
            payload=dict(report),
        )
    return None


def action_contract_enabled(backend: object) -> bool:
    method = getattr(backend, "action_contract_enabled", None)
    if not callable(method):
        return False
    return bool(method())


def attach_processor(backend: object, processor: object | None) -> None:
    if processor is None:
        return
    method = getattr(backend, "attach_processor", None)
    if callable(method):
        method(processor)


def plan_optimization_profiles(
    backend: object,
    profiles: list[OptimizationProfile],
) -> list[dict[str, object]]:
    return _optimization_statuses(backend, profiles, method_name="plan_optimization_profiles")


def apply_loaded_optimization_profiles(
    backend: object,
    profiles: list[OptimizationProfile],
) -> list[dict[str, object]]:
    return _optimization_statuses(
        backend,
        profiles,
        method_name="apply_loaded_optimization_profiles",
    )


def _optimization_statuses(
    backend: object,
    profiles: list[OptimizationProfile],
    *,
    method_name: str,
) -> list[dict[str, object]]:
    if not profiles:
        return []
    method = getattr(backend, method_name, None)
    if not callable(method):
        return []
    statuses = method(profiles)
    if not isinstance(statuses, list):
        return []
    normalized: list[dict[str, object]] = []
    for status in statuses:
        if isinstance(status, dict):
            normalized.append(dict(status))
    return normalized
