from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Protocol

from wam_harness.core._utils import (
    default_cache_dir,
    optional_float,
    optional_int,
    ordered_unique,
)
from wam_harness.backends.native_support.contract import native_runtime_contract_payload
from wam_harness.backends.native_support.optimizations import (
    native_optimization_status_dicts,
)
from wam_harness.core.preflight import PreflightReport
from wam_harness.core.types import (
    InferenceRequest,
    InferenceResult,
    Manifest,
    OptimizationProfile,
    RuntimeInfo,
)


class NativeBackendError(RuntimeError):
    """Raised when a native backend cannot load or run."""


class NativeProcessor(Protocol):
    def to_model_inputs(self, observation: object) -> object: ...

    def to_harness_result(self, raw_output: object) -> InferenceResult: ...


@dataclass(frozen=True)
class NativeModelCall:
    """Result from the backend-native model/server call before postprocessing."""

    raw_output: object
    timing_key: str = "model_ms"
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class NativeModelAdapter:
    """Backend-owned adapter around a loaded native model or resident server.

    Native backends own source/asset discovery and lifecycle. Adapters own the
    loaded runtime object call boundary so future optimization hooks can wrap a
    model/server consistently without copying an upstream official script.
    """

    name = "native_model_adapter"

    def require_ready(self) -> None:
        return

    def warmup(self) -> None:
        return

    def reset(self) -> None:
        return

    def close(self) -> None:
        return

    def model_timing_key(self) -> str:
        return "model_ms"

    def runtime_metadata(self) -> dict[str, object]:
        return {"model_adapter": self.name}

    def inference_metadata(self) -> dict[str, object]:
        return {"model_adapter": self.name}

    def inference_warnings(self) -> list[str]:
        return []

    def infer(
        self,
        request: InferenceRequest,
        model_inputs: object,
    ) -> object | NativeModelCall:
        raise NotImplementedError(f"{self.name} must implement infer()")


class NativeRuntimeLoader:
    """Backend-owned loader for native model/server runtime construction.

    Runtime loaders own heavy upstream imports, config composition, checkpoint
    loading, dataset-stat loading, or resident-server startup. Adapters own the
    actual inference call after the runtime exists.
    """

    name = "native_runtime_loader"
    runtime_mode: str | None = None

    def load(self, *args: Any, **kwargs: Any) -> object:
        raise NotImplementedError(f"{self.name} must implement load()")


@dataclass(frozen=True)
class NativeUpstreamStatus:
    env_var: str
    default_dir: str | None
    candidates: list[str]
    selected: str | None
    required_paths: list[str]
    missing_paths: list[str]
    status: str
    expected_commit: str | None = None
    selected_commit: str | None = None
    commit_status: str | None = None


@dataclass(frozen=True)
class NativePythonModuleStatus:
    name: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
        }


@dataclass(frozen=True)
class NativeRequirements:
    backend: str
    label: str
    runtime_mode: str | None
    runtime_loader: str | None
    model_adapter: str | None
    required_assets: list[str]
    runtime_assets: list[str]
    required_python_modules: list[str]
    upstream: NativeUpstreamStatus


@dataclass(frozen=True)
class NativeAssetStatus:
    name: str
    status: str
    path: str | None
    required: bool
    runtime: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "path": self.path,
            "required": self.required,
            "runtime": self.runtime,
        }


@dataclass(frozen=True)
class NativeReadiness:
    requirements: NativeRequirements
    assets: list[NativeAssetStatus]
    python_modules: list[NativePythonModuleStatus]
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "backend": self.requirements.backend,
            "label": self.requirements.label,
            "runtime_mode": self.requirements.runtime_mode,
            "runtime_loader": self.requirements.runtime_loader,
            "model_adapter": self.requirements.model_adapter,
            "required_assets": self.requirements.required_assets,
            "runtime_assets": self.requirements.runtime_assets,
            "required_python_modules": self.requirements.required_python_modules,
            "missing_required_assets": self.missing_required_assets,
            "missing_runtime_assets": self.missing_runtime_assets,
            "missing_python_modules": self.missing_python_modules,
            "upstream": {
                "status": self.requirements.upstream.status,
                "env_var": self.requirements.upstream.env_var,
                "default_dir": self.requirements.upstream.default_dir,
                "selected": self.requirements.upstream.selected,
                "expected_commit": self.requirements.upstream.expected_commit,
                "selected_commit": self.requirements.upstream.selected_commit,
                "commit_status": self.requirements.upstream.commit_status,
                "required_paths": self.requirements.upstream.required_paths,
                "missing_paths": self.requirements.upstream.missing_paths,
                "candidates": self.requirements.upstream.candidates,
            },
            "assets": [asset.to_dict() for asset in self.assets],
            "python_modules": [module.to_dict() for module in self.python_modules],
        }

    @property
    def missing_required_assets(self) -> list[str]:
        return [
            asset.name
            for asset in self.assets
            if asset.required and asset.status != "present"
        ]

    @property
    def missing_runtime_assets(self) -> list[str]:
        return [
            asset.name
            for asset in self.assets
            if asset.runtime and not asset.required and asset.status != "present"
        ]

    @property
    def missing_python_modules(self) -> list[str]:
        return [
            module.name
            for module in self.python_modules
            if module.status != "present"
        ]


class NativeBackendBase:
    """Shared support for native WAM backends.

    This class is deliberately small. It owns repo/asset discovery and runtime
    metadata, while model loading and observation conversion stay in concrete
    backends and processors.
    """

    error_cls: type[NativeBackendError] = NativeBackendError
    default_upstream_env = ""
    required_upstream_paths: tuple[str, ...] = ()
    required_asset_names: tuple[str, ...] = ()
    runtime_asset_names: tuple[str, ...] = ()
    required_python_modules: tuple[str, ...] = ()
    runtime_mode: str | None = None
    runtime_loader_name: str | None = None
    model_adapter_name: str | None = None
    optimization_hooks: ClassVar[dict[str, str]] = {
        "action_chunk_scheduling": "action_chunk_contract",
    }

    def __init__(
        self,
        manifest: Manifest,
        profiles: list[OptimizationProfile],
        *,
        backend_label: str,
    ) -> None:
        self.manifest = manifest
        self.profiles = profiles
        self.backend_label = backend_label
        self.config = dict(manifest.backend.get("config", {}))
        self.device = str(manifest.defaults.get("device", "cuda"))
        self.dtype = str(manifest.defaults.get("dtype", "bf16"))
        self.loaded = False
        self.warmed = False
        self.upstream_repo: Path | None = None
        self.runtime_loader: NativeRuntimeLoader | None = None
        self.model_adapter: NativeModelAdapter | None = None
        self.processor: NativeProcessor | None = None
        self.optimization_statuses: dict[str, dict[str, object]] = {}

    def runtime_info(self, metadata: dict[str, object] | None = None) -> RuntimeInfo:
        payload: dict[str, object] = {
            "native": True,
            "loaded": self.loaded,
            "upstream_repo": str(self.upstream_repo) if self.upstream_repo else None,
            "cache_dir": str(self.cache_dir()),
            "native_optimization_plan": native_optimization_status_dicts(
                self.manifest,
                self.profiles,
                applied_statuses=self.optimization_status_overrides(),
            ),
        }
        runtime_mode = self.native_runtime_mode()
        if runtime_mode is not None:
            payload["runtime_mode"] = runtime_mode
        loader_name = self.native_runtime_loader_name()
        if loader_name is not None:
            payload["runtime_loader"] = loader_name
        adapter = self.native_model_adapter(required=False)
        if adapter is not None:
            payload.update(adapter.runtime_metadata())
        else:
            adapter_name = self.native_model_adapter_name()
            if adapter_name is not None:
                payload["model_adapter"] = adapter_name
        if metadata:
            payload.update(metadata)
        return RuntimeInfo(
            manifest_id=self.manifest.id,
            model_name=self.manifest.display_name,
            backend=self.manifest.backend_name,
            processor=self.manifest.processor_name,
            source_repo=self.manifest.source_repo,
            mode=str(self.manifest.backend.get("mode", "native")),
            device=self.device,
            dtype=self.dtype,
            optimization_profiles=self.profiles,
            metadata=payload,
        )

    def native_requirements(self) -> NativeRequirements:
        return NativeRequirements(
            backend=self.manifest.backend_name,
            label=self.backend_label,
            runtime_mode=self.native_runtime_mode(),
            runtime_loader=self.native_runtime_loader_name(),
            model_adapter=self.native_model_adapter_name(),
            required_assets=list(self.native_required_asset_names()),
            runtime_assets=list(self.native_runtime_asset_names()),
            required_python_modules=list(self.native_required_python_modules()),
            upstream=self.inspect_upstream_repo(),
        )

    def backend_requirements(self) -> NativeRequirements:
        return self.native_requirements()

    def runtime_contract(self, *, processor: object | None = None) -> dict[str, object] | None:
        return native_runtime_contract_payload(
            self.manifest,
            self.profiles,
            processor=processor,
            backend=self,
        )

    def preflight(self) -> PreflightReport:
        readiness = self.native_readiness()
        return PreflightReport(
            status=readiness.status,
            payload=readiness.to_dict(),
        )

    def action_contract_enabled(self) -> bool:
        return True

    def attach_processor(self, processor: object) -> None:
        self.processor = processor  # type: ignore[assignment]

    def apply_optimization_profiles(
        self,
        profiles: list[OptimizationProfile],
    ) -> list[dict[str, object]]:
        self.optimization_statuses = {
            profile.name: self._apply_optimization_profile(profile)
            for profile in profiles
        }
        return native_optimization_status_dicts(
            self.manifest,
            profiles,
            applied_statuses=self.optimization_status_overrides(),
        )

    def optimization_status_overrides(self) -> dict[str, dict[str, object]]:
        return dict(self.optimization_statuses)

    def native_required_upstream_paths(self) -> tuple[str, ...]:
        return tuple(self.required_upstream_paths)

    def native_required_asset_names(self) -> tuple[str, ...]:
        return tuple(self.required_asset_names)

    def native_runtime_asset_names(self) -> tuple[str, ...]:
        return tuple(self.runtime_asset_names)

    def native_required_python_modules(self) -> tuple[str, ...]:
        return tuple(self.required_python_modules)

    def native_model_adapter_name(self) -> str | None:
        if self.model_adapter_name is not None:
            return self.model_adapter_name
        adapter = self.native_model_adapter(required=False)
        return adapter.name if adapter is not None else None

    def native_runtime_loader_name(self) -> str | None:
        if self.runtime_loader_name is not None:
            return self.runtime_loader_name
        loader = self.runtime_loader
        name = getattr(loader, "name", None)
        return str(name) if name is not None else None

    def native_runtime_mode(self) -> str | None:
        if self.runtime_mode is not None:
            return self.runtime_mode
        loader = self.runtime_loader
        mode = getattr(loader, "runtime_mode", None)
        return str(mode) if mode is not None else None

    def native_optimization_hooks(self) -> dict[str, str]:
        return dict(self.optimization_hooks)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run one native product inference through the shared harness spine."""

        self.require_warmed()
        self.require_inference_ready()
        return self.infer_with_processor(
            request,
            self.native_processor(),
            lambda model_inputs: self._infer_model(request, model_inputs),
            model_timing_key=self.native_model_timing_key(),
            metadata=self.native_inference_metadata(),
            warnings=self.native_inference_warnings(),
        )

    def native_processor(self) -> NativeProcessor:
        if self.processor is None:
            raise self.error_cls(
                f"{self.backend_label} backend does not expose a native processor"
            )
        return self.processor

    def native_model_adapter(self, *, required: bool = True) -> NativeModelAdapter | None:
        adapter = self.model_adapter
        if adapter is not None:
            return adapter
        if not required:
            return None
        raise self.error_cls(
            f"{self.backend_label} backend does not expose a native model adapter"
        )

    def require_inference_ready(self) -> None:
        adapter = self.native_model_adapter(required=False)
        if adapter is not None:
            adapter.require_ready()

    def native_model_timing_key(self) -> str:
        adapter = self.native_model_adapter(required=False)
        if adapter is None:
            return "model_ms"
        return adapter.model_timing_key()

    def native_inference_metadata(self) -> dict[str, object]:
        adapter = self.native_model_adapter(required=False)
        if adapter is None:
            return {}
        return adapter.inference_metadata()

    def native_inference_warnings(self) -> list[str]:
        adapter = self.native_model_adapter(required=False)
        if adapter is None:
            return []
        return adapter.inference_warnings()

    def _infer_model(
        self,
        request: InferenceRequest,
        model_inputs: object,
    ) -> object | NativeModelCall:
        adapter = self.native_model_adapter(required=False)
        if adapter is not None:
            return adapter.infer(request, model_inputs)
        raise NotImplementedError(
            f"{self.backend_label} backend must implement _infer_model() "
            "or set self.model_adapter during load()."
        )

    def native_readiness(self) -> NativeReadiness:
        requirements = self.native_requirements()
        assets = self.inspect_native_assets(requirements)
        python_modules = self.inspect_python_modules(requirements)
        missing_required = [
            asset.name for asset in assets if asset.required and asset.status != "present"
        ]
        missing_runtime = [
            asset.name
            for asset in assets
            if asset.runtime and not asset.required and asset.status != "present"
        ]
        missing_python = [
            module.name for module in python_modules if module.status != "present"
        ]
        if requirements.upstream.status != "present" or missing_required or missing_python:
            status = "blocked"
        elif requirements.upstream.commit_status == "mismatch" or missing_runtime:
            status = "warning"
        else:
            status = "ready"
        return NativeReadiness(
            requirements=requirements,
            assets=assets,
            python_modules=python_modules,
            status=status,
        )

    def infer_with_processor(
        self,
        request: InferenceRequest,
        processor: NativeProcessor,
        call_model: Callable[[object], object | NativeModelCall],
        *,
        model_timing_key: str = "model_ms",
        metadata: dict[str, object] | None = None,
        warnings: list[str] | None = None,
    ) -> InferenceResult:
        """Run the common native inference spine.

        Concrete backends own the native model/server call. The harness owns the
        surrounding processor conversion, timing keys, native metadata, and
        warning merge so these details do not drift across model integrations.
        """

        preprocess_start = time.perf_counter()
        model_inputs = processor.to_model_inputs(request.observation)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

        model_start = time.perf_counter()
        call_result = call_model(model_inputs)
        model_ms = (time.perf_counter() - model_start) * 1000

        timing_key = model_timing_key
        raw_output = call_result
        call_metadata: dict[str, object] = {}
        call_warnings: list[str] = []
        if isinstance(call_result, NativeModelCall):
            raw_output = call_result.raw_output
            timing_key = call_result.timing_key or model_timing_key
            call_metadata = dict(call_result.metadata)
            call_warnings = list(call_result.warnings)

        postprocess_start = time.perf_counter()
        result = processor.to_harness_result(raw_output)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000

        timing = {
            "preprocess_ms": preprocess_ms,
            timing_key: model_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": preprocess_ms + model_ms + postprocess_ms,
        }
        merged_metadata = {
            **result.backend_metadata,
            "native_backend": True,
            **(metadata or {}),
            **call_metadata,
        }
        merged_warnings = [
            *result.warnings,
            *(warnings or []),
            *call_warnings,
        ]
        return replace(
            result,
            timing=timing,
            backend_metadata=merged_metadata,
            warnings=merged_warnings,
        )

    def inspect_native_assets(
        self,
        requirements: NativeRequirements | None = None,
    ) -> list[NativeAssetStatus]:
        resolved = requirements or self.native_requirements()
        names = ordered_unique(
            [*resolved.required_assets, *resolved.runtime_assets]
        )
        required = set(resolved.required_assets)
        runtime = set(resolved.runtime_assets)
        statuses: list[NativeAssetStatus] = []
        for name in names:
            raw = self.manifest.assets.get(name)
            declared = isinstance(raw, dict) and bool(raw.get("local_path"))
            path = self.optional_asset_path(name) if declared else None
            status = "undeclared"
            if path is not None:
                status = "present" if path.exists() else "missing"
            statuses.append(
                NativeAssetStatus(
                    name=name,
                    status=status,
                    path=str(path) if path is not None else None,
                    required=name in required,
                    runtime=name in runtime,
                )
            )
        return statuses

    def inspect_python_modules(
        self,
        requirements: NativeRequirements | None = None,
    ) -> list[NativePythonModuleStatus]:
        resolved = requirements or self.native_requirements()
        statuses: list[NativePythonModuleStatus] = []
        for name in resolved.required_python_modules:
            statuses.append(
                NativePythonModuleStatus(
                    name=name,
                    status="present" if _module_present(name) else "missing",
                )
            )
        return statuses

    def inspect_upstream_repo(
        self,
        *,
        default_env: str | None = None,
        required_paths: list[str] | tuple[str, ...] | None = None,
    ) -> NativeUpstreamStatus:
        env_name = self.upstream_env_name(default_env=default_env)
        default_dir = self.upstream_default_dir()
        candidates = self.upstream_candidates(env_name=env_name, default_dir=default_dir)
        required = list(
            required_paths
            if required_paths is not None
            else self.native_required_upstream_paths()
        )
        expected_commit = self.expected_upstream_commit()

        if not candidates:
            return NativeUpstreamStatus(
                env_var=env_name,
                default_dir=str(default_dir) if default_dir is not None else None,
                candidates=[],
                selected=None,
                required_paths=required,
                missing_paths=[],
                status="missing",
                expected_commit=expected_commit,
            )

        first_existing_invalid: tuple[Path, list[str]] | None = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            resolved = candidate.resolve()
            missing = [path for path in required if not (resolved / path).exists()]
            if not missing:
                selected_commit = self.selected_upstream_commit(resolved)
                return NativeUpstreamStatus(
                    env_var=env_name,
                    default_dir=str(default_dir) if default_dir is not None else None,
                    candidates=[str(path) for path in candidates],
                    selected=str(resolved),
                    required_paths=required,
                    missing_paths=[],
                    status="present",
                    expected_commit=expected_commit,
                    selected_commit=selected_commit,
                    commit_status=_commit_status(expected_commit, selected_commit),
                )
            if first_existing_invalid is None:
                first_existing_invalid = (resolved, missing)

        if first_existing_invalid is not None:
            selected, missing = first_existing_invalid
            return NativeUpstreamStatus(
                env_var=env_name,
                default_dir=str(default_dir) if default_dir is not None else None,
                candidates=[str(path) for path in candidates],
                selected=str(selected),
                required_paths=required,
                missing_paths=missing,
                status="invalid",
                expected_commit=expected_commit,
                selected_commit=self.selected_upstream_commit(selected),
            )

        return NativeUpstreamStatus(
            env_var=env_name,
            default_dir=str(default_dir) if default_dir is not None else None,
            candidates=[str(path) for path in candidates],
            selected=None,
            required_paths=required,
            missing_paths=[],
            status="missing",
            expected_commit=expected_commit,
        )

    def resolve_upstream_repo(
        self,
        *,
        default_env: str | None = None,
        required_paths: list[str] | tuple[str, ...] | None = None,
    ) -> Path:
        status = self.inspect_upstream_repo(
            default_env=default_env,
            required_paths=required_paths,
        )
        if status.status == "present" and status.selected is not None:
            self.upstream_repo = Path(status.selected)
            return self.upstream_repo

        if status.status == "invalid" and status.selected is not None:
            raise self.error_cls(
                f"{self.backend_label} upstream repo is invalid: {status.selected}; "
                f"missing required paths: {', '.join(status.missing_paths)}."
            )

        tried = ", ".join(status.candidates) or "<none>"
        raise self.error_cls(
            f"{self.backend_label} upstream repo not found for native backend. "
            f"Set {status.env_var}=/path/to/repo or backend.config.upstream_dir. Tried: {tried}."
        )

    def upstream_env_name(self, *, default_env: str | None = None) -> str:
        fallback = default_env if default_env is not None else self.default_upstream_env
        return str(self.upstream_config().get("local_env", fallback))

    def upstream_default_dir(self) -> Path | None:
        default_dir = self.upstream_config().get("default_dir")
        if default_dir is None:
            return None
        return Path(str(default_dir)).expanduser()

    def upstream_candidates(self, *, env_name: str, default_dir: Path | None) -> list[Path]:
        candidates: list[Path] = []
        explicit = self.config.get("upstream_dir")
        if explicit:
            candidates.append(Path(str(explicit)).expanduser())

        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())

        if default_dir is not None:
            candidates.append(default_dir)

        return candidates

    def expected_upstream_commit(self) -> str | None:
        commit = self.upstream_config().get("commit")
        return str(commit) if commit is not None else None

    def selected_upstream_commit(self, repo: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        value = result.stdout.strip()
        return value or None

    def resolve_required_asset(self, name: str) -> Path:
        raw = self.manifest.assets.get(name)
        if not isinstance(raw, dict) or not raw.get("local_path"):
            raise self.error_cls(f"{self.backend_label} manifest is missing asset: {name}")
        path = self.resolve_asset_path(str(raw["local_path"]))
        if not path.exists():
            raise self.error_cls(
                f"{self.backend_label} required asset '{name}' is missing at {path}. "
                "Run `wam prepare <model-id>` to inspect expected assets, or mount the "
                "checkpoint/cache path inside the backend container."
            )
        return path.resolve()

    def optional_asset_path(self, name: str) -> Path | None:
        raw = self.manifest.assets.get(name)
        if not isinstance(raw, dict) or not raw.get("local_path"):
            return None
        return self.resolve_asset_path(str(raw["local_path"]))

    def resolve_asset_path(self, local_path: str) -> Path:
        path = Path(local_path).expanduser()
        if path.is_absolute():
            return path
        if self.config.get("cache_dir"):
            return self.cache_dir() / path
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        return self.cache_dir() / path

    def add_upstream_paths(self, *paths: Path) -> None:
        for path in reversed(paths):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

    def set_runtime_env_defaults(self, values: dict[str, object | None]) -> None:
        for key, value in values.items():
            if value is not None:
                os.environ.setdefault(str(key), str(value))

    def eval_defaults(self) -> dict[str, Any]:
        defaults = self.manifest.eval.get("defaults", {})
        return dict(defaults) if isinstance(defaults, dict) else {}

    def optimization_context(self, name: str) -> dict[str, Any]:
        contexts = self.manifest.eval.get("optimization_context", {})
        if not isinstance(contexts, dict):
            return {}
        raw = contexts.get(name, {})
        return dict(raw) if isinstance(raw, dict) else {}

    def profile_enabled(self, name: str) -> bool:
        return any(profile.name == name and profile.enabled for profile in self.profiles)

    def profile_params(self, name: str) -> dict[str, object]:
        for profile in self.profiles:
            if profile.name == name:
                return dict(profile.params)
        return {}

    def profile_settings(self, name: str) -> dict[str, object]:
        settings: dict[str, object] = self.optimization_context(name)
        settings.update(self.profile_params(name))
        return settings

    def _apply_optimization_profile(
        self,
        profile: OptimizationProfile,
    ) -> dict[str, object]:
        if profile.name not in self.manifest.supported_optimizations:
            return {
                "state": "unsupported_by_manifest",
                "reason": "not_declared_in_manifest",
            }
        if not profile.enabled:
            return {"state": "disabled"}

        hook = self.native_optimization_hooks().get(profile.name)
        if hook is not None:
            return {"state": "applied", "hook": hook}

        return {
            "state": "requested",
            "reason": "no_backend_hook",
        }

    def upstream_config(self) -> dict[str, Any]:
        upstream = self.manifest.eval.get("upstream", {})
        return dict(upstream) if isinstance(upstream, dict) else {}

    def cache_dir(self) -> Path:
        configured = self.config.get("cache_dir")
        if configured:
            return Path(str(configured)).expanduser()
        return default_cache_dir()

    def require_loaded(self) -> None:
        if not self.loaded:
            raise self.error_cls(f"{self.backend_label} backend must be loaded first")

    def require_warmed(self) -> None:
        if not self.warmed:
            raise self.error_cls(f"{self.backend_label} backend must be warmed before inference")

    def close(self) -> None:
        if self.model_adapter is not None:
            self.model_adapter.close()
            self.model_adapter = None
        self.warmed = False
        self.loaded = False

    def optional_int(self, value: Any) -> int | None:
        return optional_int(value)

    def optional_float(self, value: Any) -> float | None:
        return optional_float(value)

    def no_grad(self) -> object:
        torch = importlib.import_module("torch")
        return torch.no_grad()


def _commit_status(expected: str | None, selected: str | None) -> str | None:
    if expected is None:
        return None
    if selected is None:
        return "unknown"
    expected_key = expected.strip().lower()
    selected_key = selected.strip().lower()
    if selected_key.startswith(expected_key) or expected_key.startswith(selected_key):
        return "match"
    return "mismatch"


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
