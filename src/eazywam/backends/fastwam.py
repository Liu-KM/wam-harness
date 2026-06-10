from __future__ import annotations

import importlib
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar

from eazywam.backends.native import (
    NativeBackendBase,
    NativeBackendError,
    NativeUpstreamStatus,
    NativeModelAdapter,
    NativeModelCall,
    NativeRuntimeLoader,
)
from eazywam.core._utils import (
    optional_float as _optional_float,
    optional_int as _optional_int,
)
from eazywam.core.types import (
    InferenceRequest,
    Manifest,
    OptimizationProfile,
)


class FastWAMNativeBackendError(NativeBackendError):
    """Raised when the native FastWAM path cannot be loaded."""


@dataclass(frozen=True)
class FastWAMRuntimeBundle:
    """Loaded FastWAM runtime pieces before harness processor/adapter binding."""

    model: Any
    cfg: Any
    upstream_processor: Any
    prompt_template: str
    device: str
    dtype: Any
    checkpoint_path: Path
    dataset_stats_path: Path


@dataclass(frozen=True)
class FastWAMRuntimeImports:
    """Heavy upstream modules needed to construct the FastWAM runtime."""

    torch: Any
    hydra: Any
    global_hydra: Any
    hydra_utils: Any
    fastwam_normalizer: Any
    robot_video_dataset: Any


class FastWAMRuntimeLoader(NativeRuntimeLoader):
    """Build a loaded FastWAM runtime without running an official eval script."""

    name = "fastwam_runtime_loader"
    runtime_mode = "in_process"

    def __init__(self, backend: FastWAMBackend) -> None:
        self.backend = backend

    def load(
        self,
        *,
        config_dir: Path,
        checkpoint_path: Path,
        dataset_stats_path: Path,
    ) -> FastWAMRuntimeBundle:
        modules = self._import_runtime_modules()
        self.backend._register_fastwam_config_resolvers()
        cfg = self.backend._compose_fastwam_config(
            config_dir,
            modules.hydra,
            modules.global_hydra,
        )
        self.backend._validate_visualize_future_video_config(cfg)
        device = self.backend._resolve_eval_device(modules.torch, cfg)
        model_dtype = self.backend._mixed_precision_to_torch_dtype(
            modules.torch,
            str(_get_config_value(cfg, "mixed_precision", self.backend.dtype)),
        )
        model = modules.hydra_utils.instantiate(
            cfg.model,
            model_dtype=model_dtype,
            device=device,
        )
        model.load_checkpoint(str(checkpoint_path))
        model = model.to(device).eval()

        dataset_stats = modules.fastwam_normalizer.load_dataset_stats_from_json(
            str(dataset_stats_path)
        )
        upstream_processor = modules.hydra_utils.instantiate(cfg.data.train.processor).eval()
        upstream_processor.set_normalizer_from_stats(dataset_stats)

        return FastWAMRuntimeBundle(
            model=model,
            cfg=cfg,
            upstream_processor=upstream_processor,
            prompt_template=str(modules.robot_video_dataset.DEFAULT_PROMPT),
            device=device,
            dtype=getattr(model, "torch_dtype", model_dtype),
            checkpoint_path=checkpoint_path,
            dataset_stats_path=dataset_stats_path,
        )

    def _import_runtime_modules(self) -> FastWAMRuntimeImports:
        try:
            return FastWAMRuntimeImports(
                torch=importlib.import_module("torch"),
                hydra=importlib.import_module("hydra"),
                global_hydra=importlib.import_module("hydra.core.global_hydra"),
                hydra_utils=importlib.import_module("hydra.utils"),
                fastwam_normalizer=importlib.import_module(
                    "fastwam.datasets.lerobot.utils.normalizer"
                ),
                robot_video_dataset=importlib.import_module(
                    "fastwam.datasets.lerobot.robot_video_dataset"
                ),
            )
        except ModuleNotFoundError as exc:
            raise self.backend.error_cls(
                "FastWAM native backend dependencies are not importable. "
                "Run inside a FastWAM-compatible container or install the "
                "self-managed FastWAM runtime environment."
            ) from exc


class FastWAMModelAdapter(NativeModelAdapter):
    """Native adapter around a loaded FastWAM model object."""

    name = "fastwam_model"

    def __init__(
        self,
        *,
        model: Any,
        cfg: Any,
        checkpoint_path: Path | None,
        dataset_stats_path: Path | None,
        config: dict[str, Any],
        dit_cache_params: dict[str, object],
        no_grad_factory: Callable[[], object],
        error_cls: type[FastWAMNativeBackendError],
        cuda_graph_params: dict[str, object] | None = None,
        cuda_graph_enabled: bool = False,
        torch_compile_params: dict[str, object] | None = None,
        torch_compile_enabled: bool = False,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.dataset_stats_path = dataset_stats_path
        self.config = config
        self.dit_cache_params = dit_cache_params
        self.cuda_graph_params = cuda_graph_params or {}
        self.cuda_graph_enabled = bool(cuda_graph_enabled)
        self.torch_compile_params = torch_compile_params or {}
        self.torch_compile_enabled = bool(torch_compile_enabled)
        self.no_grad_factory = no_grad_factory
        self.error_cls = error_cls

    def require_ready(self) -> None:
        if self.model is None:
            raise self.error_cls("FastWAM model adapter is not loaded")

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "model_adapter": self.name,
            "model_class": type(self.model).__name__ if self.model is not None else None,
        }

    def inference_metadata(self) -> dict[str, object]:
        return {
            "model_adapter": self.name,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "dataset_stats_path": (
                str(self.dataset_stats_path) if self.dataset_stats_path else None
            ),
        }

    def infer(self, request: InferenceRequest, model_inputs: object) -> NativeModelCall:
        self.require_ready()
        if not isinstance(model_inputs, dict):
            raise self.error_cls("FastWAM processor must return a mapping of model inputs")

        visualize_future_video = self._visualize_future_video()
        infer_kwargs = {
            "prompt": model_inputs["prompt"],
            "input_image": model_inputs["input_image"],
            "action_horizon": int(request.action_horizon),
            "negative_prompt": str(self._evaluation_value("negative_prompt", "")),
            "text_cfg_scale": float(self._evaluation_value("text_cfg_scale", 1.0)),
            "num_inference_steps": self._num_inference_steps(request),
            "proprio": model_inputs["proprio"],
            "sigma_shift": _optional_float(self._evaluation_value("sigma_shift", None)),
            "seed": _optional_int(self._config_value("seed", None)),
            "rand_device": str(self._evaluation_value("rand_device", "cpu")),
            "tiled": bool(self._evaluation_value("tiled", False)),
            "cache_mode": self._dit_cache_mode(request),
        }
        cuda_graph_mode = self._cuda_graph_mode(request)
        if self._infer_action_accepts("cuda_graph_mode"):
            infer_kwargs["cuda_graph_mode"] = cuda_graph_mode
        torch_compile_mode = self._torch_compile_mode(request)
        if self._infer_action_accepts("torch_compile_mode"):
            infer_kwargs["torch_compile_mode"] = torch_compile_mode
        if (
            visualize_future_video
            or "num_video_frames" in inspect.signature(self.model.infer_action).parameters
        ):
            infer_kwargs["num_video_frames"] = self._num_video_frames()

        with self.no_grad_factory():
            if visualize_future_video:
                if not hasattr(self.model, "infer_joint"):
                    raise self.error_cls(
                        "FastWAM EVALUATION.visualize_future_video=true requires "
                        "model.infer_joint, but the loaded model does not provide it."
                    )
                joint_kwargs = dict(infer_kwargs)
                joint_kwargs.pop("cache_mode", None)
                raw_output = self.model.infer_joint(**joint_kwargs)
                return NativeModelCall(
                    raw_output=raw_output,
                    metadata={
                        "fastwam_call": "infer_joint",
                        "future_video_present": _future_video_present(raw_output),
                        "num_video_frames": infer_kwargs.get("num_video_frames"),
                    },
                )
            raw_output = self.model.infer_action(**infer_kwargs)
            metadata = {
                "fastwam_call": "infer_action",
                "num_video_frames": infer_kwargs.get("num_video_frames"),
                "cuda_graph_enabled": cuda_graph_mode != "off",
                "cuda_graph_mode": cuda_graph_mode,
                "cuda_graph_hook": "fastwam_cuda_graph_action_body",
                "torch_compile_enabled": torch_compile_mode != "off",
                "torch_compile_mode": torch_compile_mode,
                "torch_compile_hook": "fastwam_torch_compile_action_body",
            }
            if isinstance(raw_output, dict) and isinstance(raw_output.get("metadata"), dict):
                metadata.update(raw_output["metadata"])
            return NativeModelCall(
                raw_output=raw_output,
                metadata=metadata,
            )

    def close(self) -> None:
        self.model = None
        self.cfg = None

    def _config_value(self, key: str, default: Any) -> Any:
        if self.cfg is None:
            return default
        return _get_config_value(self.cfg, key, default)

    def _evaluation_value(self, key: str, default: Any) -> Any:
        if self.cfg is None:
            return default
        evaluation = _get_config_value(self.cfg, "EVALUATION", {})
        return _get_config_value(evaluation, key, default)

    def _visualize_future_video(self) -> bool:
        return bool(self._evaluation_value("visualize_future_video", False))

    def _num_inference_steps(self, request: InferenceRequest) -> int:
        if "num_inference_steps" in request.runtime_options:
            return int(request.runtime_options["num_inference_steps"])
        configured = self._evaluation_value("num_inference_steps", None)
        if configured is not None:
            return int(configured)
        return int(self._config_value("eval_num_inference_steps", 20))

    def _num_video_frames(self) -> int:
        if self.cfg is None:
            return 1
        num_frames = int(self.cfg.data.train.num_frames)
        action_video_freq_ratio = int(self.cfg.data.train.action_video_freq_ratio)
        return (num_frames - 1) // action_video_freq_ratio + 1

    def _dit_cache_mode(self, request: InferenceRequest) -> str:
        configured = request.runtime_options.get("dit_cache_mode")
        if configured is None:
            configured = self.config.get("dit_cache_mode")
        if configured is None:
            configured = self._dit_cache_profile_mode()
        mode = str(configured or "video_kv")
        if mode not in {"video_kv", "recompute"}:
            raise self.error_cls(
                "FastWAM dit_cache mode must be one of: video_kv, recompute; "
                f"got {mode!r}."
            )
        return mode

    def _dit_cache_profile_mode(self) -> str:
        if self.dit_cache_params.get("mode") is not None:
            return str(self.dit_cache_params["mode"])
        return "video_kv"

    def _cuda_graph_mode(self, request: InferenceRequest) -> str:
        configured = request.runtime_options.get("cuda_graph_mode")
        if configured is None:
            configured = self.config.get("cuda_graph_mode")
        if configured is None and self.cuda_graph_enabled:
            configured = self.cuda_graph_params.get("mode", "auto")
        mode = _normalize_cuda_graph_mode(configured)
        if mode not in {"off", "auto"}:
            raise self.error_cls(
                "FastWAM cuda_graph mode must be one of: off, auto; "
                f"got {mode!r}."
            )
        return mode

    def _torch_compile_mode(self, request: InferenceRequest) -> str:
        configured = request.runtime_options.get("torch_compile_mode")
        if configured is None:
            configured = self.config.get("torch_compile_mode")
        if configured is None and self.torch_compile_enabled:
            configured = self.torch_compile_params.get("mode", "auto")
        mode = _normalize_torch_compile_mode(configured)
        if mode not in {"off", "auto", "default", "reduce-overhead", "max-autotune"}:
            raise self.error_cls(
                "FastWAM torch_compile mode must be one of: off, auto, default, "
                f"reduce-overhead, max-autotune; got {mode!r}."
            )
        return mode

    def _infer_action_accepts(self, name: str) -> bool:
        try:
            signature = inspect.signature(self.model.infer_action)
        except (TypeError, ValueError):
            return False
        return name in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )


class FastWAMBackend(NativeBackendBase):
    """Native FastWAM backend.

    This backend imports the vendored FastWAM runtime only inside ``load`` so
    the core harness remains lightweight outside a FastWAM container.
    """

    error_cls = FastWAMNativeBackendError
    default_upstream_env = "WAM_FASTWAM_REPO"
    required_upstream_paths = ()
    required_asset_names = ("checkpoint", "dataset_stats")
    runtime_asset_names = (
        "checkpoint",
        "dataset_stats",
        "wan22_vae",
        "wan22_t5_encoder",
        "wan21_tokenizer_spiece",
        "wan21_tokenizer_json",
        "wan21_tokenizer_config",
        "wan21_special_tokens_map",
    )
    required_python_modules = (
        "torch",
        "hydra",
        "hydra.core.global_hydra",
        "hydra.utils",
        "omegaconf",
        "numpy",
        "PIL.Image",
        "einops",
    )
    model_adapter_name = FastWAMModelAdapter.name
    optimization_hooks: ClassVar[dict[str, str]] = {
        **NativeBackendBase.optimization_hooks,
        "dit_cache": "fastwam_video_kv_cache",
        "cuda_graph": "fastwam_cuda_graph_action_body",
        "torch_compile": "fastwam_torch_compile_action_body",
    }
    loaded_optimization_hooks: ClassVar[dict[str, str]] = {
        "dit_cache": "fastwam_video_kv_cache",
        "cuda_graph": "fastwam_cuda_graph_action_body",
        "torch_compile": "fastwam_torch_compile_action_body",
    }

    def __init__(self, manifest: Manifest, profiles: list[OptimizationProfile]) -> None:
        super().__init__(manifest, profiles, backend_label="FastWAM")
        self.model: Any | None = None
        self.cfg: Any | None = None
        self.checkpoint_path = None
        self.dataset_stats_path = None
        self.runtime_loader = FastWAMRuntimeLoader(self)

    def load(self) -> None:
        config_dir = self._fastwam_config_dir()
        checkpoint_path = self.resolve_required_asset("checkpoint")
        dataset_stats_path = self.resolve_required_asset("dataset_stats")
        self.checkpoint_path = checkpoint_path
        self.dataset_stats_path = dataset_stats_path
        self._apply_runtime_env()

        runtime = self.runtime_loader.load(
            config_dir=config_dir,
            checkpoint_path=checkpoint_path,
            dataset_stats_path=dataset_stats_path,
        )

        processor = self.native_processor()
        bind_runtime = getattr(processor, "bind_runtime", None)
        if not callable(bind_runtime):
            raise self.error_cls(
                "FastWAM native backend requires a FastWAM-compatible processor "
                "attached by the invocation layer before load()."
            )
        bind_runtime(
            upstream_processor=runtime.upstream_processor,
            cfg=runtime.cfg,
            device=runtime.device,
            dtype=runtime.dtype,
            prompt_template=runtime.prompt_template,
        )
        self.device = runtime.device
        self.model = runtime.model
        self.cfg = runtime.cfg
        self.model_adapter = self._create_model_adapter(
            model=runtime.model,
            cfg=runtime.cfg,
        )
        self.upstream_repo = config_dir
        self.loaded = True

    def warmup(self) -> None:
        self.require_loaded()
        self._fastwam_adapter().warmup()
        self.warmed = True

    def reset(self) -> None:
        self.require_loaded()
        self._fastwam_adapter().reset()

    def close(self) -> None:
        self.model = None
        self.cfg = None
        super().close()

    def _create_model_adapter(self, *, model: Any, cfg: Any) -> FastWAMModelAdapter:
        return FastWAMModelAdapter(
            model=model,
            cfg=cfg,
            checkpoint_path=self.checkpoint_path,
            dataset_stats_path=self.dataset_stats_path,
            config=dict(self.config),
            dit_cache_params=self.profile_settings("dit_cache"),
            cuda_graph_params=self.profile_settings("cuda_graph"),
            cuda_graph_enabled=self.profile_enabled("cuda_graph"),
            torch_compile_params=self.profile_settings("torch_compile"),
            torch_compile_enabled=self.profile_enabled("torch_compile"),
            no_grad_factory=self.no_grad,
            error_cls=self.error_cls,
        )

    def native_model_adapter(self, *, required: bool = True) -> NativeModelAdapter | None:
        if isinstance(self.model_adapter, FastWAMModelAdapter):
            return self.model_adapter
        if self.model is not None and self.cfg is not None:
            self.model_adapter = self._create_model_adapter(model=self.model, cfg=self.cfg)
            return self.model_adapter
        return super().native_model_adapter(required=required)

    def _fastwam_adapter(self) -> FastWAMModelAdapter:
        adapter = self.native_model_adapter(required=True)
        if isinstance(adapter, FastWAMModelAdapter):
            return adapter
        raise self.error_cls("FastWAM model is not loaded")

    def _apply_loaded_optimization_profile(
        self,
        profile: OptimizationProfile,
        planned_status: dict[str, object] | None,
    ) -> dict[str, object]:
        status = super()._apply_loaded_optimization_profile(profile, planned_status)
        if status.get("state") != "applied":
            return status

        if profile.name == "cuda_graph":
            if self._cuda_graph_hook_available():
                return status
            return {
                **status,
                "state": "fallback",
                "hook": "fastwam_cuda_graph_action_body",
                "reason": "cuda_graph_hook_unavailable",
            }

        if profile.name == "torch_compile":
            if self._torch_compile_hook_available():
                return status
            return {
                **status,
                "state": "fallback",
                "hook": "fastwam_torch_compile_action_body",
                "reason": "torch_compile_hook_unavailable",
            }

        if profile.name != "dit_cache" or self._dit_cache_hook_available():
            return status
        return {
            **status,
            "state": "fallback",
            "hook": "fastwam_video_kv_cache",
            "reason": "cache_hook_unavailable",
        }

    def _dit_cache_hook_available(self) -> bool:
        model = self.model
        if model is None:
            return False
        mot = getattr(model, "mot", None)
        infer_action = getattr(model, "infer_action", None)
        if not callable(infer_action):
            return False
        try:
            has_cache_mode = "cache_mode" in inspect.signature(infer_action).parameters
        except (TypeError, ValueError):
            has_cache_mode = False
        return (
            has_cache_mode
            and callable(getattr(mot, "prefill_video_cache", None))
            and callable(getattr(mot, "forward_action_with_video_cache", None))
        )

    def _cuda_graph_hook_available(self) -> bool:
        if not self._dit_cache_hook_available():
            return False
        infer_action = getattr(self.model, "infer_action", None)
        if not callable(infer_action):
            return False
        try:
            signature = inspect.signature(infer_action)
        except (TypeError, ValueError):
            return False
        return "cuda_graph_mode" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _torch_compile_hook_available(self) -> bool:
        if not self._dit_cache_hook_available():
            return False
        infer_action = getattr(self.model, "infer_action", None)
        if not callable(infer_action):
            return False
        try:
            signature = inspect.signature(infer_action)
        except (TypeError, ValueError):
            return False
        return "torch_compile_mode" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def inspect_upstream_repo(
        self,
        *,
        default_env: str | None = None,
        required_paths: list[str] | tuple[str, ...] | None = None,
    ) -> NativeUpstreamStatus:
        if self.config.get("upstream_dir") or self.upstream_candidates(
            env_name=self.upstream_env_name(default_env=default_env),
            default_dir=None,
        ):
            return super().inspect_upstream_repo(
                default_env=default_env,
                required_paths=required_paths or self.native_required_upstream_paths(),
            )
        config_dir = self._vendored_config_dir()
        return NativeUpstreamStatus(
            env_var=self.upstream_env_name(default_env=default_env),
            default_dir=None,
            candidates=[str(config_dir)],
            selected=str(config_dir),
            required_paths=[],
            missing_paths=[],
            status="present",
            expected_commit="45d8e14",
            selected_commit="vendored:45d8e1458921d83f8ad6cf9ce993d371208dabd0",
            commit_status="vendored",
        )

    def native_required_upstream_paths(self) -> tuple[str, ...]:
        env_name = self.upstream_env_name(default_env=None)
        if not (self.config.get("upstream_dir") or os.environ.get(env_name)):
            return ()
        config_name = str(self.config.get("config_name", "sim_libero"))
        if not config_name.endswith(".yaml"):
            config_name = f"{config_name}.yaml"
        paths = [
            f"configs/{config_name}",
            "configs/train.yaml",
            *self._hydra_config_group_paths(),
        ]
        return tuple(dict.fromkeys(paths))

    def _compose_fastwam_config(self, config_dir: Path, hydra: Any, global_hydra: Any) -> Any:
        config_name = str(self.config.get("config_name", "sim_libero"))
        overrides = self._hydra_overrides()
        instance = global_hydra.GlobalHydra.instance()
        if instance.is_initialized():
            instance.clear()
        with hydra.initialize_config_dir(
            config_dir=str(config_dir),
            version_base="1.3",
        ):
            return hydra.compose(config_name=config_name, overrides=overrides)

    def _fastwam_config_dir(self) -> Path:
        explicit = self.config.get("upstream_dir")
        if explicit:
            config_dir = Path(str(explicit)).expanduser() / "configs"
            if not config_dir.exists():
                raise self.error_cls(
                    f"FastWAM config directory not found at {config_dir}. "
                    "Omit --upstream-dir to use the vendored EazyWAM runtime, "
                    "or pass a FastWAM checkout that contains configs/."
                )
            return config_dir
        return self._vendored_config_dir()

    def _vendored_config_dir(self) -> Path:
        return Path(str(resources.files("fastwam").joinpath("configs"))).resolve()

    def _register_fastwam_config_resolvers(self) -> None:
        try:
            resolvers = importlib.import_module("fastwam.utils.config_resolvers")
        except ModuleNotFoundError as exc:
            raise self.error_cls(
                "FastWAM config resolver module is not importable. "
                "The native backend needs fastwam.utils.config_resolvers before "
                "Hydra composition."
            ) from exc
        register = getattr(resolvers, "register_default_resolvers", None)
        if not callable(register):
            raise self.error_cls(
                "FastWAM config resolver module does not expose register_default_resolvers()."
            )
        register()

    def _hydra_overrides(self) -> list[str]:
        eval_defaults = self.eval_defaults()
        task = self.config.get("task") or eval_defaults.get("task")
        overrides = []
        if task:
            overrides.append(f"task={task}")
        if self.checkpoint_path is not None:
            overrides.append(f"ckpt={self.checkpoint_path}")
        if self.dataset_stats_path is not None:
            overrides.append(f"EVALUATION.dataset_stats_path={self.dataset_stats_path}")
        redirect_common_files = self.config.get(
            "redirect_common_files",
            eval_defaults.get("redirect_common_files", None),
        )
        if redirect_common_files is not None:
            overrides.append(f"model.redirect_common_files={redirect_common_files}")
        extra = self.config.get("hydra_overrides", [])
        if isinstance(extra, list):
            overrides.extend(str(item) for item in extra)
        return overrides

    def _hydra_config_group_paths(self) -> list[str]:
        task = self._task_name()
        if not task:
            return [
                "configs/data/libero_2cam.yaml",
                "configs/model/fastwam.yaml",
            ]

        paths = [f"configs/task/{task}.yaml"]
        if "robotwin" in task:
            paths.append("configs/data/robotwin.yaml")
        else:
            paths.append("configs/data/libero_2cam.yaml")

        if "_idm_" in task:
            paths.append("configs/model/fastwam_idm.yaml")
        elif "_joint_" in task:
            paths.append("configs/model/fastwam_joint.yaml")
        else:
            paths.append("configs/model/fastwam.yaml")
        return paths

    def _task_name(self) -> str | None:
        task = self.config.get("task") or self.eval_defaults().get("task")
        return str(task) if task else None

    def _apply_runtime_env(self) -> None:
        eval_defaults = self.eval_defaults()
        env_defaults = {
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "WANDB_MODE": "offline",
        }
        diffsynth_base = self._diffsynth_model_base_path()
        if diffsynth_base is not None:
            env_defaults["DIFFSYNTH_MODEL_BASE_PATH"] = str(diffsynth_base)
        diffsynth_source = eval_defaults.get("diffsynth_download_source")
        if diffsynth_source is not None:
            env_defaults["DIFFSYNTH_DOWNLOAD_SOURCE"] = str(diffsynth_source)
        self.set_runtime_env_defaults(env_defaults)

    def _diffsynth_model_base_path(self) -> Path | str | None:
        model_base = self.optional_asset_path("model_base")
        if model_base is None:
            model_base = self.optional_asset_path("wan22_vae")
        if model_base is not None:
            # DiffSynth expects the cache root, while the Wamfile points at a
            # specific HF repo directory or file such as
            # Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth.
            parts = model_base.parts
            if "Wan-AI" in parts:
                index = parts.index("Wan-AI")
                return Path(*parts[:index]) if index else Path(".")
            return model_base.parent
        return self.eval_defaults().get("diffsynth_model_base_path")

    def _config_value(self, key: str, default: Any) -> Any:
        if self.cfg is None:
            return default
        return _get_config_value(self.cfg, key, default)

    def _evaluation_value(self, key: str, default: Any) -> Any:
        if self.cfg is None:
            return default
        evaluation = _get_config_value(self.cfg, "EVALUATION", {})
        return _get_config_value(evaluation, key, default)

    def _resolve_eval_device(self, torch: Any, cfg: Any) -> str:
        evaluation = _get_config_value(cfg, "EVALUATION", {})
        eval_device = _get_config_value(evaluation, "device", None)
        if eval_device is not None:
            return str(eval_device)
        if self.device:
            return self.device
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"

    def _validate_visualize_future_video_config(self, cfg: Any) -> None:
        evaluation = _get_config_value(cfg, "EVALUATION", {})
        if not bool(_get_config_value(evaluation, "visualize_future_video", False)):
            return
        model_cfg = _get_config_value(cfg, "model", {})
        video_dit_config = _get_config_value(model_cfg, "video_dit_config", {})
        action_conditioned = _get_config_value(video_dit_config, "action_conditioned", None)
        if action_conditioned is not False:
            raise self.error_cls(
                "FastWAM EVALUATION.visualize_future_video=true requires "
                "model.video_dit_config.action_conditioned=false."
            )

    def _mixed_precision_to_torch_dtype(self, torch: Any, mixed_precision: str) -> Any:
        key = mixed_precision.strip().lower()
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16"}:
            return torch.float16
        if key in {"no", "fp32", "float32"}:
            return torch.float32
        raise self.error_cls(
            f"Unsupported FastWAM dtype/mixed_precision: {mixed_precision}"
        )


def _get_config_value(config: object, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _normalize_cuda_graph_mode(value: object) -> str:
    if value is True:
        return "auto"
    if value is False or value is None:
        return "off"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "auto"}:
        return "auto"
    if text in {"0", "false", "no", "off", "none"}:
        return "off"
    return text


def _normalize_torch_compile_mode(value: object) -> str:
    if value is True:
        return "auto"
    if value is False or value is None:
        return "off"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "auto"}:
        return "auto"
    if text in {"0", "false", "no", "off", "none"}:
        return "off"
    return text


def _future_video_present(raw_output: object) -> bool:
    if not isinstance(raw_output, dict):
        return False
    video = raw_output.get("video")
    return isinstance(video, list) and len(video) > 0
