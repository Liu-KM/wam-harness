from __future__ import annotations

from typing import Any, Protocol


class NativePreflightError(RuntimeError):
    """Raised when a native backend is known to be un-runnable before load."""


class NativeReadinessInspectable(Protocol):
    def native_readiness(self) -> object: ...


def native_readiness_payload(backend: object) -> dict[str, Any] | None:
    if not hasattr(backend, "native_readiness"):
        return None
    readiness = _as_native_readiness_inspectable(backend).native_readiness()
    if hasattr(readiness, "to_dict"):
        payload = readiness.to_dict()
        return dict(payload) if isinstance(payload, dict) else None
    return None


def assert_native_preflight(
    payload: dict[str, Any] | None,
    *,
    require_ready: bool = False,
) -> None:
    """Fail early when native readiness already proves load should not run."""

    if payload is None:
        return
    status = str(payload.get("status", "unknown"))
    if status == "blocked":
        raise NativePreflightError(_preflight_message(payload, "blocked"))
    if require_ready and status != "ready":
        raise NativePreflightError(_preflight_message(payload, status))


def _as_native_readiness_inspectable(backend: object) -> NativeReadinessInspectable:
    return backend  # type: ignore[return-value]


def _preflight_message(payload: dict[str, Any], status: str) -> str:
    backend = payload.get("backend", "native")
    missing_required = _as_list(payload.get("missing_required_assets"))
    missing_runtime = _as_list(payload.get("missing_runtime_assets"))
    missing_python = _as_list(payload.get("missing_python_modules"))
    upstream = payload.get("upstream", {})
    upstream_status = None
    commit_status = None
    expected_commit = None
    selected_commit = None
    missing_paths: list[str] = []
    if isinstance(upstream, dict):
        upstream_status = upstream.get("status")
        commit_status = upstream.get("commit_status")
        expected_commit = upstream.get("expected_commit")
        selected_commit = upstream.get("selected_commit")
        missing_paths = _as_list(upstream.get("missing_paths"))

    details = []
    if upstream_status and upstream_status != "present":
        details.append(f"upstream={upstream_status}")
    if missing_paths:
        details.append(f"missing_upstream_paths={', '.join(missing_paths)}")
    if commit_status == "mismatch":
        details.append(
            "upstream_commit=mismatch"
            f" (expected {expected_commit}, got {selected_commit})"
        )
    if missing_required:
        details.append(f"missing_required_assets={', '.join(missing_required)}")
    if missing_runtime:
        details.append(f"missing_runtime_assets={', '.join(missing_runtime)}")
    if missing_python:
        details.append(f"missing_python_modules={', '.join(missing_python)}")
    suffix = f": {'; '.join(details)}" if details else ""
    return f"{backend} native readiness is {status}{suffix}"


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
