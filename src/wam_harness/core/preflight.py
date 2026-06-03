from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PreflightError(RuntimeError):
    """Raised when a backend reports it cannot run before load."""


@dataclass(frozen=True)
class PreflightReport:
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
    message: str | None = None

    def to_trace_payload(self) -> dict[str, Any]:
        return {"status": self.status, **self.payload}


def assert_preflight(
    report: PreflightReport | None,
    *,
    require_ready: bool = False,
) -> None:
    if report is None:
        return
    if report.status == "blocked":
        raise PreflightError(_preflight_message(report, "blocked"))
    if require_ready and report.status != "ready":
        raise PreflightError(_preflight_message(report, report.status))


def _preflight_message(report: PreflightReport, status: str) -> str:
    if report.message:
        return report.message
    payload = report.payload
    backend = payload.get("backend", "backend")
    details = []
    upstream = payload.get("upstream", {})
    if isinstance(upstream, dict):
        upstream_status = upstream.get("status")
        if upstream_status and upstream_status != "present":
            details.append(f"upstream={upstream_status}")
        missing_paths = _as_list(upstream.get("missing_paths"))
        if missing_paths:
            details.append(f"missing_upstream_paths={', '.join(missing_paths)}")
        if upstream.get("commit_status") == "mismatch":
            details.append(
                "upstream_commit=mismatch"
                f" (expected {upstream.get('expected_commit')}, "
                f"got {upstream.get('selected_commit')})"
            )
    missing_required = _as_list(payload.get("missing_required_assets"))
    missing_runtime = _as_list(payload.get("missing_runtime_assets"))
    missing_python = _as_list(payload.get("missing_python_modules"))
    if missing_required:
        details.append(f"missing_required_assets={', '.join(missing_required)}")
    if missing_runtime:
        details.append(f"missing_runtime_assets={', '.join(missing_runtime)}")
    if missing_python:
        details.append(f"missing_python_modules={', '.join(missing_python)}")
    suffix = f": {'; '.join(details)}" if details else ""
    return f"{backend} preflight is {status}{suffix}"


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
