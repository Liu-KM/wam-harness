from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from wam_harness.backends.native import NativeReadiness, NativeRequirements
from wam_harness.core._utils import (
    csv_text,
    default_cache_dir,
    optional_int as _optional_int,
    ordered_unique as _ordered_unique,
)
from wam_harness.core.assets import AssetDownloader, AssetError, HuggingFaceAssetDownloader
from wam_harness.core.manifest import list_builtin_manifest_ids, load_builtin_manifest
from wam_harness.backends.native_support.runtime import (
    NATIVE_DOCTOR_SPEC,
    NATIVE_PREPARE_SPEC,
    native_backend_name,
    resolve_native_runtime,
)
from wam_harness.core.model_entry_labels import (
    model_deployment_label,
    model_runtime_label,
)
from wam_harness.core.registry import RegistryError, default_registry
from wam_harness.core.types import Manifest


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
    native_lines: list[str] = field(default_factory=list)
    native: dict[str, object] | None = None

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
            "native": self.native,
        }


class NativeInspectableBackend(Protocol):
    def native_requirements(self) -> NativeRequirements: ...
    def native_readiness(self) -> NativeReadiness: ...


def load_model_entries() -> list[Manifest]:
    return [load_builtin_manifest(model_id) for model_id in list_builtin_manifest_ids()]


def doctor_model_entry(
    model_id: str | None = None,
    cache_dir: str | Path | None = None,
    upstream_dir: str | Path | None = None,
) -> DoctorSummary:
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

    entry = load_builtin_manifest(model_id)
    gpu_status = None
    if _requires_cuda(entry):
        gpu_status = "found" if _gpu_visible() else "missing"
        if gpu_status == "missing":
            status = "warning"

    assets = asset_statuses(entry, cache)
    if any(asset.status == "missing" for asset in assets):
        status = "warning"

    native_lines, native_status, native_payload = _native_backend_doctor_lines(
        entry,
        cache_dir=cache,
        upstream_dir=upstream_dir,
    )
    if native_status == "blocked":
        status = "blocked"
    elif native_status != "ok" and status == "ok":
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
        native_lines=native_lines,
        native=native_payload,
    )


def prepare_model_entry(
    model_id: str,
    cache_dir: str | Path | None = None,
    *,
    download: bool = False,
    selected_assets: list[str] | None = None,
    downloader: AssetDownloader | None = None,
) -> PrepareSummary:
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    entry = load_builtin_manifest(model_id)
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
) -> list[AssetStatus]:
    statuses: list[AssetStatus] = []
    downloaded = downloaded_assets or set()
    asset_messages = messages or {}
    native_roles = _native_asset_roles(entry, cache_dir)
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
                required=asset_name in native_roles["required"],
                runtime=asset_name in native_roles["runtime"],
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


def _native_asset_roles(entry: Manifest, cache_dir: Path) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {"required": set(), "runtime": set()}
    if native_backend_name(entry) is None and entry.backend_name == "external_eval":
        return roles
    try:
        manifest = resolve_native_runtime(
            entry,
            NATIVE_PREPARE_SPEC,
            cache_dir=cache_dir,
        ).manifest
        backend = default_registry().create_backend(manifest, [])
    except (RegistryError, RuntimeError, ValueError):
        return roles
    if not hasattr(backend, "native_requirements"):
        return roles
    requirements = _as_native_inspectable(backend).native_requirements()
    roles["required"].update(requirements.required_assets)
    roles["runtime"].update(requirements.runtime_assets)
    return roles


def _native_backend_doctor_lines(
    entry: Manifest,
    *,
    cache_dir: str | Path | None = None,
    upstream_dir: str | Path | None = None,
) -> tuple[list[str], str, dict[str, object] | None]:
    if native_backend_name(entry) is None and entry.backend_name == "external_eval":
        return (["Native backend: none declared"], "ok", {"declared": False})

    try:
        manifest = resolve_native_runtime(
            entry,
            NATIVE_DOCTOR_SPEC,
            cache_dir=cache_dir,
            upstream_dir=upstream_dir,
        ).manifest
        backend = default_registry().create_backend(manifest, [])
    except (RegistryError, RuntimeError, ValueError) as exc:
        return (
            [f"Native backend: unavailable ({exc})"],
            "warning",
            {"declared": True, "status": "unavailable", "message": str(exc)},
        )

    if not hasattr(backend, "native_requirements"):
        return (["Native backend: none declared"], "ok", {"declared": False})

    readiness = _as_native_inspectable(backend).native_readiness()
    native_payload = readiness.to_dict()
    native_payload["declared"] = True
    requirements = readiness.requirements
    upstream = requirements.upstream
    next_steps = _native_next_steps(
        entry,
        readiness,
        cache_dir=Path(cache_dir) if cache_dir is not None else default_cache_dir(),
    )
    native_payload["next_steps"] = next_steps
    lines = [
        f"Native backend: {requirements.backend} ({requirements.label})",
        f"Native runtime mode: {requirements.runtime_mode or 'none'}",
        f"Native runtime loader: {requirements.runtime_loader or 'none'}",
        f"Native model adapter: {requirements.model_adapter or 'none'}",
        f"Native readiness: {readiness.status}",
        f"Native required assets: {csv_text(requirements.required_assets, default='none')}",
    ]
    if requirements.runtime_assets:
        lines.append(
            f"Native runtime assets: {csv_text(requirements.runtime_assets, default='none')}"
        )
    if requirements.required_python_modules:
        lines.append(
            "Native required Python modules: "
            f"{csv_text(requirements.required_python_modules, default='none')}"
        )
    if readiness.missing_required_assets:
        lines.append(
            "Native missing required assets: "
            f"{csv_text(readiness.missing_required_assets, default='none')}"
        )
    if readiness.missing_runtime_assets:
        lines.append(
            "Native missing runtime assets: "
            f"{csv_text(readiness.missing_runtime_assets, default='none')}"
        )
    if readiness.missing_python_modules:
        lines.append(
            "Native missing Python modules: "
            f"{csv_text(readiness.missing_python_modules, default='none')}"
        )
    lines.extend(
        [
            (
                "Upstream repo: "
                f"{upstream.status}"
                f"{f' ({upstream.selected})' if upstream.selected else ''}"
            ),
            f"Upstream env: {upstream.env_var}",
        ]
    )
    if upstream.default_dir is not None:
        lines.append(f"Upstream default: {upstream.default_dir}")
    if upstream.expected_commit is not None:
        lines.append(f"Upstream expected commit: {upstream.expected_commit}")
    if upstream.selected_commit is not None:
        lines.append(f"Upstream selected commit: {upstream.selected_commit}")
    if upstream.commit_status is not None:
        lines.append(f"Upstream commit status: {upstream.commit_status}")
    if upstream.required_paths:
        lines.append(f"Upstream required paths: {csv_text(upstream.required_paths, default='none')}")
    if upstream.missing_paths:
        lines.append(f"Upstream missing paths: {csv_text(upstream.missing_paths, default='none')}")
    if upstream.candidates:
        lines.append(f"Upstream checked: {csv_text(upstream.candidates, default='none')}")
    if next_steps:
        lines.append("Native next steps:")
        lines.extend(f"- {step}" for step in next_steps)

    if readiness.status == "ready":
        native_status = "ok"
    elif readiness.status == "blocked":
        native_status = "blocked"
    else:
        native_status = "warning"
    return (lines, native_status, native_payload)


def _native_next_steps(
    entry: Manifest,
    readiness: NativeReadiness,
    *,
    cache_dir: Path,
) -> list[str]:
    steps: list[str] = []
    upstream = readiness.requirements.upstream
    if upstream.status != "present":
        steps.append(
            f"Set {upstream.env_var}=<repo> or pass --upstream-dir <repo> "
            f"with required paths: {csv_text(upstream.required_paths, default='none')}."
        )
    elif upstream.commit_status == "mismatch":
        steps.append(
            "Use the expected upstream checkout "
            f"{upstream.expected_commit} or record the tested commit explicitly."
        )

    missing_assets = _ordered_unique(
        [*readiness.missing_required_assets, *readiness.missing_runtime_assets]
    )
    if missing_assets:
        asset_args = " ".join(f"--asset {name}" for name in missing_assets)
        steps.append(
            f"Prepare missing native assets: wam prepare {entry.id} "
            f"--cache-dir {cache_dir} --download {asset_args}."
        )

    if readiness.missing_python_modules:
        steps.append(
            "Run inside the backend container or install native dependencies: "
            f"{csv_text(readiness.missing_python_modules, default='none')}."
        )

    if readiness.status in {"ready", "warning"}:
        upstream_arg = f" --upstream-dir {upstream.selected}" if upstream.selected else ""
        steps.append(
            f"Validate the product path: wam native-smoke {entry.id} "
            f"--cache-dir {cache_dir}{upstream_arg} --require-ready."
        )
    return steps


def _as_native_inspectable(backend: object) -> NativeInspectableBackend:
    return backend  # type: ignore[return-value]


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
