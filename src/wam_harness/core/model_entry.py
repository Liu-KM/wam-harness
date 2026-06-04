from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from wam_harness.core._utils import (
    csv_text,
    default_cache_dir,
    optional_int as _optional_int,
    ordered_unique as _ordered_unique,
)
from wam_harness.core.assets import AssetDownloader, AssetError, HuggingFaceAssetDownloader
from wam_harness.core.backend_capabilities import preflight_report
from wam_harness.core.registry import Registry, RegistryError, default_registry
from wam_harness.core.runtime import DOCTOR_SPEC, PREPARE_SPEC
from wam_harness.core.types import Manifest
from wam_harness.model_entry_labels import (
    model_deployment_label,
    model_runtime_label,
)


@dataclass(frozen=True)
class AssetStatus:
    name: str
    uri: str | None
    expected_path: Path | None
    status: str
    size_bytes: int | None = None
    downloaded: bool = False
    message: str | None = None
    required: bool = False
    runtime: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "uri": self.uri,
            "expected_path": str(self.expected_path) if self.expected_path is not None else None,
            "status": self.status,
            "size_bytes": self.size_bytes,
            "downloaded": self.downloaded,
            "message": self.message,
            "required": self.required,
            "runtime": self.runtime,
        }


@dataclass(frozen=True)
class PrepareSummary:
    model_id: str
    cache_dir: Path
    assets: list[AssetStatus]
    status: str
    download: bool = False
    selected_assets: list[str] | None = None


@dataclass(frozen=True)
class DoctorSummary:
    cache_dir: Path
    cache_status: str
    runtime_setup: str
    status: str
    model_id: str | None = None
    runtime: str | None = None
    deployment: str | None = None
    gpu: str | None = None
    assets: list[AssetStatus] = field(default_factory=list)
    backend_lines: list[str] = field(default_factory=list)
    backend: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "cache_dir": str(self.cache_dir),
            "cache_status": self.cache_status,
            "runtime_setup": self.runtime_setup,
            "model_id": self.model_id,
            "runtime": self.runtime,
            "deployment": self.deployment,
            "gpu": self.gpu,
            "assets": [asset.to_dict() for asset in self.assets],
            "backend": self.backend,
        }


def load_model_entry(model_id: str, registry: Registry | None = None) -> Manifest:
    return _registry_or_default(registry).load_manifest(model_id)


def load_model_entries(registry: Registry | None = None) -> list[Manifest]:
    model_registry = _registry_or_default(registry)
    return [
        model_registry.load_manifest(model_id)
        for model_id in model_registry.list_model_ids()
    ]


def doctor_model_entry(
    model_id: str | None = None,
    cache_dir: str | Path | None = None,
    upstream_dir: str | Path | None = None,
    registry: Registry | None = None,
) -> DoctorSummary:
    model_registry = _registry_or_default(registry)
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache_status = _cache_check_label(cache)
    runtime_setup = "not modified"
    status = "ok"
    if model_id is None:
        return DoctorSummary(
            cache_dir=cache,
            cache_status=cache_status,
            runtime_setup=runtime_setup,
            status=status,
        )

    entry = model_registry.load_manifest(model_id)
    gpu_status = None
    if _requires_cuda(entry):
        gpu_status = "found" if _gpu_visible() else "missing"
        if gpu_status == "missing":
            status = "warning"

    assets = asset_statuses(entry, cache, registry=model_registry)
    if any(asset.status == "missing" for asset in assets):
        status = "warning"

    backend_lines, backend_status, backend_payload = _backend_doctor_lines(
        entry,
        cache_dir=cache,
        upstream_dir=upstream_dir,
        registry=model_registry,
    )
    if backend_status == "blocked":
        status = "blocked"
    elif backend_status != "ok" and status == "ok":
        status = "warning"

    return DoctorSummary(
        cache_dir=cache,
        cache_status=cache_status,
        runtime_setup=runtime_setup,
        status=status,
        model_id=entry.id,
        runtime=model_runtime_label(entry),
        deployment=model_deployment_label(entry),
        gpu=gpu_status,
        assets=assets,
        backend_lines=backend_lines,
        backend=backend_payload,
    )


def prepare_model_entry(
    model_id: str,
    cache_dir: str | Path | None = None,
    *,
    download: bool = False,
    selected_assets: list[str] | None = None,
    downloader: AssetDownloader | None = None,
    registry: Registry | None = None,
) -> PrepareSummary:
    model_registry = _registry_or_default(registry)
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    entry = model_registry.load_manifest(model_id)
    selected = _validate_selected_assets(entry, selected_assets)
    downloaded_assets: set[str] = set()
    asset_messages: dict[str, str] = {}
    if download:
        downloaded_assets, asset_messages = _download_missing_assets(
            entry,
            cache,
            selected_assets=selected,
            downloader=downloader or HuggingFaceAssetDownloader(),
        )
    assets = asset_statuses(
        entry,
        cache,
        downloaded_assets=downloaded_assets,
        messages=asset_messages,
        registry=model_registry,
    )
    scoped_assets = [asset for asset in assets if selected is None or asset.name in selected]
    status = "ok"
    if any(asset.status in {"missing", "error"} for asset in scoped_assets):
        status = "incomplete"
    return PrepareSummary(
        model_id=entry.id,
        cache_dir=cache,
        assets=assets,
        status=status,
        download=download,
        selected_assets=selected,
    )


def asset_statuses(
    entry: Manifest,
    cache_dir: Path,
    *,
    downloaded_assets: set[str] | None = None,
    messages: dict[str, str] | None = None,
    registry: Registry | None = None,
) -> list[AssetStatus]:
    statuses: list[AssetStatus] = []
    downloaded = downloaded_assets or set()
    asset_messages = messages or {}
    backend_roles = _backend_asset_roles(entry, cache_dir, registry=registry)
    for name, raw in entry.assets.items():
        if not isinstance(raw, dict):
            continue
        asset_name = str(name)
        uri = raw.get("uri")
        local_path = raw.get("local_path")
        size_bytes = _optional_int(raw.get("size_bytes"))
        expected_path = _expected_asset_path(local_path, cache_dir)
        status = "unknown"
        if expected_path is not None:
            status = "present" if expected_path.exists() else "missing"
        statuses.append(
            AssetStatus(
                name=asset_name,
                uri=str(uri) if uri is not None else None,
                expected_path=expected_path,
                status=status,
                size_bytes=size_bytes,
                downloaded=asset_name in downloaded and status == "present",
                message=asset_messages.get(asset_name),
                required=asset_name in backend_roles["required"],
                runtime=asset_name in backend_roles["runtime"],
            )
        )
    return statuses


def _download_missing_assets(
    entry: Manifest,
    cache_dir: Path,
    *,
    selected_assets: list[str] | None,
    downloader: AssetDownloader,
) -> tuple[set[str], dict[str, str]]:
    downloaded: set[str] = set()
    messages: dict[str, str] = {}
    for asset in asset_statuses(entry, cache_dir):
        if selected_assets is not None and asset.name not in selected_assets:
            continue
        if asset.status != "missing":
            continue
        if asset.uri is None:
            messages[asset.name] = "no source URI declared"
            continue
        if asset.expected_path is None:
            messages[asset.name] = "no local path declared"
            continue
        try:
            downloader.download(asset.uri, asset.expected_path)
            if asset.expected_path.exists():
                downloaded.add(asset.name)
            else:
                messages[asset.name] = "download completed but expected path is still missing"
        except AssetError as exc:
            messages[asset.name] = str(exc)
        except Exception as exc:
            messages[asset.name] = f"failed to prepare from {asset.uri}: {exc}"
    return downloaded, messages


def _validate_selected_assets(entry: Manifest, selected_assets: list[str] | None) -> list[str] | None:
    if not selected_assets:
        return None
    known = set(entry.assets)
    selected = []
    unknown = []
    for name in selected_assets:
        value = str(name)
        if value not in known:
            unknown.append(value)
            continue
        if value not in selected:
            selected.append(value)
    if unknown:
        known_text = ", ".join(sorted(known)) or "<none>"
        raise ValueError(
            f"unknown asset(s) for {entry.id}: {', '.join(unknown)}; known assets: {known_text}"
        )
    return selected


def _expected_asset_path(local_path: object, cache_dir: Path) -> Path | None:
    if local_path is None:
        return None
    path = Path(str(local_path))
    if path.is_absolute():
        return path
    return cache_dir / path


def _backend_asset_roles(
    entry: Manifest,
    cache_dir: Path,
    *,
    registry: Registry | None = None,
) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {"required": set(), "runtime": set()}
    model_registry = _registry_or_default(registry)
    try:
        manifest = model_registry.resolve_runtime(
            entry,
            PREPARE_SPEC,
            cache_dir=cache_dir,
        ).manifest
        backend = model_registry.create_backend(manifest, [])
    except (RegistryError, RuntimeError, ValueError):
        return roles
    requirements = _backend_requirements(backend)
    if requirements is None:
        return roles
    roles["required"].update(_as_str_list(getattr(requirements, "required_assets", [])))
    roles["runtime"].update(_as_str_list(getattr(requirements, "runtime_assets", [])))
    return roles


def _backend_doctor_lines(
    entry: Manifest,
    *,
    cache_dir: str | Path | None = None,
    upstream_dir: str | Path | None = None,
    registry: Registry | None = None,
) -> tuple[list[str], str, dict[str, object] | None]:
    model_registry = _registry_or_default(registry)
    try:
        manifest = model_registry.resolve_runtime(
            entry,
            DOCTOR_SPEC,
            cache_dir=cache_dir,
            upstream_dir=upstream_dir,
        ).manifest
        backend = model_registry.create_backend(manifest, [])
    except (RegistryError, RuntimeError, ValueError) as exc:
        return (
            [f"Backend requirements: unavailable ({exc})"],
            "warning",
            {"declared": True, "status": "unavailable", "message": str(exc)},
        )

    requirements = _backend_requirements(backend)
    report = preflight_report(backend)
    if requirements is None or report is None:
        return (["Backend requirements: none declared"], "ok", {"declared": False})

    backend_payload = report.to_trace_payload()
    backend_payload["declared"] = True
    upstream = getattr(requirements, "upstream", None)
    next_steps = _backend_next_steps(
        entry,
        report.to_trace_payload(),
        requirements,
        cache_dir=Path(cache_dir) if cache_dir is not None else default_cache_dir(),
    )
    backend_payload["next_steps"] = next_steps
    lines = [
        f"Backend target: {getattr(requirements, 'backend', 'unknown')} ({getattr(requirements, 'label', 'unknown')})",
        f"Backend runtime mode: {getattr(requirements, 'runtime_mode', None) or 'none'}",
        f"Backend runtime loader: {getattr(requirements, 'runtime_loader', None) or 'none'}",
        f"Backend model adapter: {getattr(requirements, 'model_adapter', None) or 'none'}",
        f"Backend readiness: {report.status}",
        f"Backend required assets: {csv_text(getattr(requirements, 'required_assets', []), default='none')}",
    ]
    runtime_assets = _as_str_list(getattr(requirements, "runtime_assets", []))
    if runtime_assets:
        lines.append(
            f"Backend runtime assets: {csv_text(runtime_assets, default='none')}"
        )
    required_python = _as_str_list(getattr(requirements, "required_python_modules", []))
    if required_python:
        lines.append(
            "Backend required Python modules: "
            f"{csv_text(required_python, default='none')}"
        )
    missing_required = _as_str_list(backend_payload.get("missing_required_assets"))
    missing_runtime = _as_str_list(backend_payload.get("missing_runtime_assets"))
    missing_python = _as_str_list(backend_payload.get("missing_python_modules"))
    if missing_required:
        lines.append(
            "Backend missing required assets: "
            f"{csv_text(missing_required, default='none')}"
        )
    if missing_runtime:
        lines.append(
            "Backend missing runtime assets: "
            f"{csv_text(missing_runtime, default='none')}"
        )
    if missing_python:
        lines.append(
            "Backend missing Python modules: "
            f"{csv_text(missing_python, default='none')}"
        )
    if upstream is not None:
        selected = getattr(upstream, "selected", None)
        lines.extend(
            [
                (
                    "Upstream repo: "
                    f"{getattr(upstream, 'status', 'unknown')}"
                    f"{f' ({selected})' if selected else ''}"
                ),
                f"Upstream env: {getattr(upstream, 'env_var', 'unknown')}",
            ]
        )
        if getattr(upstream, "default_dir", None) is not None:
            lines.append(f"Upstream default: {upstream.default_dir}")
        if getattr(upstream, "expected_commit", None) is not None:
            lines.append(f"Upstream expected commit: {upstream.expected_commit}")
        if getattr(upstream, "selected_commit", None) is not None:
            lines.append(f"Upstream selected commit: {upstream.selected_commit}")
        if getattr(upstream, "commit_status", None) is not None:
            lines.append(f"Upstream commit status: {upstream.commit_status}")
        required_paths = _as_str_list(getattr(upstream, "required_paths", []))
        missing_paths = _as_str_list(getattr(upstream, "missing_paths", []))
        candidates = _as_str_list(getattr(upstream, "candidates", []))
        if required_paths:
            lines.append(f"Upstream required paths: {csv_text(required_paths, default='none')}")
        if missing_paths:
            lines.append(f"Upstream missing paths: {csv_text(missing_paths, default='none')}")
        if candidates:
            lines.append(f"Upstream checked: {csv_text(candidates, default='none')}")
    if next_steps:
        lines.append("Backend next steps:")
        lines.extend(f"- {step}" for step in next_steps)

    if report.status == "ready":
        backend_status = "ok"
    elif report.status == "blocked":
        backend_status = "blocked"
    else:
        backend_status = "warning"
    return (lines, backend_status, backend_payload)


def _backend_next_steps(
    entry: Manifest,
    payload: dict[str, object],
    requirements: object,
    *,
    cache_dir: Path,
) -> list[str]:
    steps: list[str] = []
    upstream = getattr(requirements, "upstream", None)
    if upstream is not None:
        upstream_status = getattr(upstream, "status", "unknown")
        if upstream_status != "present":
            steps.append(
                f"Set {getattr(upstream, 'env_var', '<env>')}=<repo> "
                f"or pass --upstream-dir <repo> with required paths: "
                f"{csv_text(getattr(upstream, 'required_paths', []), default='none')}."
            )
        elif getattr(upstream, "commit_status", None) == "mismatch":
            steps.append(
                "Use the expected upstream checkout "
                f"{getattr(upstream, 'expected_commit', None)} or record the tested commit explicitly."
            )

    missing_assets = _ordered_unique(
        [
            *_as_str_list(payload.get("missing_required_assets")),
            *_as_str_list(payload.get("missing_runtime_assets")),
        ]
    )
    if missing_assets:
        asset_args = " ".join(f"--asset {name}" for name in missing_assets)
        steps.append(
            f"Prepare missing backend assets: wam prepare {entry.id} "
            f"--cache-dir {cache_dir} --download {asset_args}."
        )

    missing_python = _as_str_list(payload.get("missing_python_modules"))
    if missing_python:
        steps.append(
            "Run inside the backend container or install backend dependencies: "
            f"{csv_text(missing_python, default='none')}."
        )

    return steps


def _backend_requirements(backend: object) -> object | None:
    method = getattr(backend, "backend_requirements", None)
    if not callable(method):
        return None
    return method()


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return []


def _requires_cuda(entry: Manifest) -> bool:
    return str(entry.defaults.get("device", "")).startswith("cuda")


def _gpu_visible() -> bool:
    return shutil.which("nvidia-smi") is not None or bool(os.environ.get("CUDA_VISIBLE_DEVICES"))


def _cache_check_label(path: Path) -> str:
    if path.exists():
        return "writable" if os.access(path, os.W_OK) else "not writable"

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return "creatable" if os.access(parent, os.W_OK) else "not creatable"


def _registry_or_default(registry: Registry | None) -> Registry:
    return registry or default_registry()
