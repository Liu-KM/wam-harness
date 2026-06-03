from __future__ import annotations

from typing import Protocol

from wam_harness.core.preflight import PreflightReport


class RuntimeContractBackend(Protocol):
    def runtime_contract(self, *, processor: object | None = None) -> dict[str, object] | None: ...


class PreflightBackend(Protocol):
    def preflight(self) -> PreflightReport | None: ...


class ActionContractBackend(Protocol):
    def action_contract_enabled(self) -> bool: ...


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
