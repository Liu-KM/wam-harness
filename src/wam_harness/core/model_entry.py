from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from wam_harness.backends.native import NativeReadiness, NativeRequirements
from wam_harness.core.assets import AssetDownloader, AssetError, HuggingFaceAssetDownloader
from wam_harness.core.manifest import list_builtin_manifest_ids, load_builtin_manifest
from wam_harness.core.native_runtime import (
    NATIVE_DOCTOR_SPEC,
    NATIVE_PREPARE_SPEC,
    native_backend_name,
    resolve_native_runtime,
)
from wam_harness.core.registry import RegistryError, default_registry
from wam_harness.core.types import Manifest


def default_cache_dir() -> Path:
    return Path(os.environ.get("WAM_CACHE_DIR", str(Path.home() / ".cache" / "wam")))


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


def render_model_list() -> str:
    entries = load_model_entries()
    id_width = max([len("MODEL ID"), *(len(entry.id) for entry in entries)])
    task_width = max([len("TASK"), *(len(_task_label(entry)) for entry in entries)])
    lines = [f"{'MODEL ID':<{id_width}}  {'TASK':<{task_width}}  RUNTIME"]
    for entry in entries:
        lines.append(f"{entry.id:<{id_width}}  {_task_label(entry):<{task_width}}  {_runtime_label(entry)}")
    return "\n".join(lines)


def render_model_info(model_id: str) -> str:
    entry = load_builtin_manifest(model_id)
    source = entry.source_repo or "unknown"
    lines = [
        f"Model: {entry.id}",
        f"Name: {entry.display_name}",
        f"Task: {_task_label(entry)}",
        f"Source: {source}",
        f"Inputs: {_input_label(entry)}",
        f"Outputs: {_output_label(entry)}",
        f"Runtime: {_runtime_label(entry)}",
        f"Deployment: {_deployment_label(entry)}",
        f"Assets: {_assets_label(entry)}",
        f"Supported opts: {_supported_opts_label(entry)}",
    ]
    if entry.known_gaps:
        lines.append("Known gaps:")
        lines.extend(f"- {gap}" for gap in entry.known_gaps)
    return "\n".join(lines)


def render_doctor(
    model_id: str | None = None,
    cache_dir: str | Path | None = None,
    upstream_dir: str | Path | None = None,
) -> str:
    summary = doctor_model_entry(
        model_id=model_id,
        cache_dir=cache_dir,
        upstream_dir=upstream_dir,
    )
    lines = [
        "WAM doctor",
        f"Cache directory: {summary.cache_dir} ({summary.cache_status})",
        f"Runtime setup: {summary.runtime_setup}",
    ]

    if summary.model_id is not None:
        lines.append(f"Model: {summary.model_id}")
        if summary.runtime is not None:
            lines.append(f"Runtime: {summary.runtime}")
        if summary.deployment is not None:
            lines.append(f"Deployment: {summary.deployment}")
        if summary.gpu is not None:
            lines.append(f"GPU: {summary.gpu}")

        if summary.assets:
            lines.append("Assets:")
            for asset in summary.assets:
                lines.append(f"- {asset.name}: {asset.status} ({asset.expected_path})")
        else:
            lines.append("Assets: none declared")

        lines.extend(summary.native_lines)

    lines.append(f"Status: {summary.status}")
    return "\n".join(lines)


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
        runtime=_runtime_label(entry),
        deployment=_deployment_label(entry),
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


def render_prepare(summary: PrepareSummary) -> str:
    lines = [
        f"Model: {summary.model_id}",
        f"Cache directory: {summary.cache_dir}",
        "Runtime setup: not modified",
    ]
    if summary.selected_assets is not None:
        lines.append(f"Selected assets: {_csv(summary.selected_assets, default='none')}")
    if summary.download:
        lines.append("Download: enabled")
    if summary.assets:
        lines.append("Assets:")
        for asset in summary.assets:
            line = f"- {asset.name}: {asset.status} ({asset.expected_path})"
            roles = _asset_role_labels(asset)
            if roles:
                line += f" [{','.join(roles)}]"
            if asset.size_bytes is not None:
                line += f" [{_format_size(asset.size_bytes)}]"
            if asset.downloaded:
                line += " [downloaded]"
            if asset.message:
                line += f" - {asset.message}"
            lines.append(line)
    else:
        lines.append("Assets: none declared")
    lines.append(f"Status: {summary.status}")
    return "\n".join(lines)


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


def _asset_role_labels(asset: AssetStatus) -> list[str]:
    labels = []
    if asset.required:
        labels.append("required")
    if asset.runtime:
        labels.append("runtime")
    return labels


def _task_label(entry: Manifest) -> str:
    eval_config = entry.eval
    if eval_config:
        simulator = eval_config.get("simulator") or entry.workload.get("config", {}).get("simulator")
        suite = eval_config.get("suite") or entry.workload.get("config", {}).get("suite")
        if simulator and suite:
            return f"{simulator} {suite}"
        if simulator:
            return str(simulator)
    return str(entry.workload_name)


def _runtime_label(entry: Manifest) -> str:
    device = str(entry.defaults.get("device", "unknown"))
    mode = str(entry.backend.get("mode", entry.backend_name))
    if device.startswith("cuda"):
        return f"GPU container recommended ({mode})"
    return f"CPU ok ({mode})"


def _deployment_label(entry: Manifest) -> str:
    deployment = entry.deployment
    if not deployment:
        return "native"
    reference = str(deployment.get("reference_path", "none"))
    product = str(deployment.get("product_path", "unknown"))
    native = deployment.get("native_backend")
    stage = str(deployment.get("native_stage", "unknown"))
    verified = bool(deployment.get("native_verified", False))
    parity = bool(deployment.get("parity_verified", False))
    parts = [f"product={product}", f"reference={reference}"]
    if native is not None:
        parts.append(f"native={native} ({stage})")
    parts.append(f"native_verified={str(verified).lower()}")
    parts.append(f"parity_verified={str(parity).lower()}")
    next_gate = deployment.get("next_gate")
    if next_gate is not None:
        parts.append(f"next={next_gate}")
    return "; ".join(parts)


def _input_label(entry: Manifest) -> str:
    observation = entry.processor.get("observation", {})
    if not isinstance(observation, dict):
        return "unknown"
    images = _csv(observation.get("image_views"), default="none")
    state = str(observation.get("state", "none"))
    prompt = str(observation.get("prompt", "none"))
    return f"images={images}; state={state}; prompt={prompt}"


def _output_label(entry: Manifest) -> str:
    action = entry.processor.get("action", {})
    if not isinstance(action, dict):
        return "action chunks"
    horizon = action.get("horizon")
    dim = action.get("dim")
    parts = ["action chunks"]
    if horizon is not None:
        parts.append(f"horizon={horizon}")
    if dim is not None:
        parts.append(f"dim={dim}")
    return "; ".join(parts)


def _assets_label(entry: Manifest) -> str:
    if not entry.assets:
        return "none declared"
    return ", ".join(str(name) for name in entry.assets)


def _supported_opts_label(entry: Manifest) -> str:
    if not entry.supported_optimizations:
        return "none"
    return ", ".join(entry.supported_optimizations)


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
        f"Native required assets: {_csv(requirements.required_assets, default='none')}",
    ]
    if requirements.runtime_assets:
        lines.append(
            f"Native runtime assets: {_csv(requirements.runtime_assets, default='none')}"
        )
    if requirements.required_python_modules:
        lines.append(
            "Native required Python modules: "
            f"{_csv(requirements.required_python_modules, default='none')}"
        )
    if readiness.missing_required_assets:
        lines.append(
            "Native missing required assets: "
            f"{_csv(readiness.missing_required_assets, default='none')}"
        )
    if readiness.missing_runtime_assets:
        lines.append(
            "Native missing runtime assets: "
            f"{_csv(readiness.missing_runtime_assets, default='none')}"
        )
    if readiness.missing_python_modules:
        lines.append(
            "Native missing Python modules: "
            f"{_csv(readiness.missing_python_modules, default='none')}"
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
        lines.append(f"Upstream required paths: {_csv(upstream.required_paths, default='none')}")
    if upstream.missing_paths:
        lines.append(f"Upstream missing paths: {_csv(upstream.missing_paths, default='none')}")
    if upstream.candidates:
        lines.append(f"Upstream checked: {_csv(upstream.candidates, default='none')}")
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
            f"with required paths: {_csv(upstream.required_paths, default='none')}."
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
            f"{_csv(readiness.missing_python_modules, default='none')}."
        )

    if readiness.status in {"ready", "warning"}:
        upstream_arg = f" --upstream-dir {upstream.selected}" if upstream.selected else ""
        steps.append(
            f"Validate the product path: wam native-smoke {entry.id} "
            f"--cache-dir {cache_dir}{upstream_arg} --require-ready."
        )
    return steps


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _as_native_inspectable(backend: object) -> NativeInspectableBackend:
    return backend  # type: ignore[return-value]


def _csv(value: object, default: str) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if value is None:
        return default
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _format_size(size_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(size_bytes)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.1f} {unit}"


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
