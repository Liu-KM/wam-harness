from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from wam_harness.backends.native import (
    NativeBackendBase,
    NativeBackendError,
    NativeModelAdapter,
    NativeRuntimeLoader,
)
from wam_harness.core.types import InferenceRequest, Manifest, OptimizationProfile
from wam_harness.processors.cosmos_policy_libero import CosmosPolicyLiberoProcessor


class CosmosPolicyNativeBackendError(NativeBackendError):
    """Raised when the native Cosmos-Policy path cannot be loaded."""


@dataclass
class CosmosPolicyEvalConfig:
    """Minimal native eval config for Cosmos-Policy model/action utilities."""

    suite: str = "libero"
    config: str = ""
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 16
    task_suite_name: str = "libero_10"
    randomize_seed: bool = False
    seed: int = 195
    use_variance_scale: bool = False
    deterministic: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    use_jpeg_compression: bool = True
    flip_images: bool = True
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    use_parallel_inference: bool = False
    num_queries_best_of_n: int = 1
    unnorm_key: str | None = None


@dataclass(frozen=True)
class CosmosPolicyRuntimeBundle:
    """Loaded Cosmos-Policy runtime pieces before adapter binding."""

    cfg: Any
    model: Any
    cosmos_config: Any
    dataset_stats: dict[str, Any]
    cosmos_utils: Any


class CosmosPolicyRuntimeLoader(NativeRuntimeLoader):
    """Build a Cosmos-Policy runtime without importing the official evaluator."""

    name = "cosmos_policy_runtime_loader"
    runtime_mode = "in_process"

    def __init__(self, backend: CosmosPolicyBackend) -> None:
        self.backend = backend

    def load(self) -> CosmosPolicyRuntimeBundle:
        cosmos_utils = self._import_runtime_modules()
        cfg = self.backend._build_eval_config(CosmosPolicyEvalConfig)
        cosmos_utils.init_t5_text_embeddings_cache(str(self.backend.text_embeddings_path))
        dataset_stats = cosmos_utils.load_dataset_stats(str(self.backend.dataset_stats_path))
        model, cosmos_config = cosmos_utils.get_model(cfg)
        self.backend._validate_loaded_runtime(cfg, model, cosmos_config)
        return CosmosPolicyRuntimeBundle(
            cfg=cfg,
            model=model,
            cosmos_config=cosmos_config,
            dataset_stats=dataset_stats,
            cosmos_utils=cosmos_utils,
        )

    def _import_runtime_modules(self) -> Any:
        try:
            return importlib.import_module("cosmos_policy.experiments.robot.cosmos_utils")
        except ModuleNotFoundError as exc:
            raise self.backend.error_cls(
                "Cosmos-Policy native backend dependencies are not importable. "
                "Run inside a Cosmos-Policy-compatible container and set "
                "WAM_COSMOS_POLICY_REPO=/path/to/cosmos-policy."
            ) from exc


class CosmosPolicyModelAdapter(NativeModelAdapter):
    """Native adapter around a loaded Cosmos-Policy model object."""

    name = "cosmos_policy_model"

    def __init__(
        self,
        *,
        cfg: Any,
        model: Any,
        dataset_stats: dict[str, Any] | None,
        cosmos_utils: Any,
        checkpoint_path: object | None,
        dataset_stats_path: object | None,
        text_embeddings_path: object | None,
        error_cls: type[CosmosPolicyNativeBackendError],
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.dataset_stats = dataset_stats
        self.cosmos_utils = cosmos_utils
        self.checkpoint_path = checkpoint_path
        self.dataset_stats_path = dataset_stats_path
        self.text_embeddings_path = text_embeddings_path
        self.error_cls = error_cls

    def require_ready(self) -> None:
        if self.model is None or self.dataset_stats is None or self.cosmos_utils is None:
            raise self.error_cls("Cosmos-Policy model adapter is not loaded")

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
            "text_embeddings_path": (
                str(self.text_embeddings_path) if self.text_embeddings_path else None
            ),
        }

    def infer(self, request: InferenceRequest, model_inputs: object) -> object:
        self.require_ready()
        if not isinstance(model_inputs, dict):
            raise self.error_cls("Cosmos-Policy processor must return a mapping of model inputs")
        return self.cosmos_utils.get_action(
            self.cfg,
            self.model,
            self.dataset_stats,
            model_inputs["observation"],
            model_inputs["prompt"],
            seed=self._seed(request),
            randomize_seed=bool(self._cfg_value("randomize_seed", False)),
            num_denoising_steps_action=self._num_denoising_steps(request),
            generate_future_state_and_value_in_parallel=True,
        )

    def close(self) -> None:
        self.cfg = None
        self.model = None
        self.dataset_stats = None
        self.cosmos_utils = None

    def _cfg_value(self, key: str, default: Any) -> Any:
        if self.cfg is None:
            return default
        return getattr(self.cfg, key, default)

    def _seed(self, request: InferenceRequest) -> int:
        if "seed" in request.runtime_options:
            return int(request.runtime_options["seed"])
        return int(self._cfg_value("seed", 195))

    def _num_denoising_steps(self, request: InferenceRequest) -> int:
        if "num_denoising_steps_action" in request.runtime_options:
            return int(request.runtime_options["num_denoising_steps_action"])
        return int(self._cfg_value("num_denoising_steps_action", 5))


class CosmosPolicyBackend(NativeBackendBase):
    error_cls = CosmosPolicyNativeBackendError
    default_upstream_env = "WAM_COSMOS_POLICY_REPO"
    required_upstream_paths = (
        "cosmos_policy/experiments/robot/cosmos_utils.py",
    )
    required_asset_names = ("checkpoint", "dataset_stats", "text_embeddings")
    runtime_asset_names = (
        "checkpoint",
        "dataset_stats",
        "text_embeddings",
        "tokenizer",
    )
    required_python_modules = ("numpy", "torch")
    model_adapter_name = CosmosPolicyModelAdapter.name

    def __init__(self, manifest: Manifest, profiles: list[OptimizationProfile]) -> None:
        super().__init__(manifest, profiles, backend_label="Cosmos-Policy")
        self.processor = CosmosPolicyLiberoProcessor.from_manifest(manifest)
        self.model: Any | None = None
        self.dataset_stats: dict[str, Any] | None = None
        self.cosmos_utils: Any | None = None
        self.cfg: Any | None = None
        self.checkpoint_path = None
        self.dataset_stats_path = None
        self.text_embeddings_path = None
        self.runtime_loader = CosmosPolicyRuntimeLoader(self)

    def native_required_upstream_paths(self) -> tuple[str, ...]:
        defaults = self.eval_defaults()
        config_file = str(
            self.config.get(
                "config_file",
                defaults.get("config_file", "cosmos_policy/config/config.py"),
            )
        )
        paths = [
            *super().native_required_upstream_paths(),
            config_file,
        ]
        return tuple(dict.fromkeys(path for path in paths if path))

    def load(self) -> None:
        repo = self.resolve_upstream_repo()
        checkpoint_path = self.resolve_required_asset("checkpoint")
        dataset_stats_path = self.resolve_required_asset("dataset_stats")
        text_embeddings_path = self.resolve_required_asset("text_embeddings")
        self.checkpoint_path = checkpoint_path
        self.dataset_stats_path = dataset_stats_path
        self.text_embeddings_path = text_embeddings_path
        self.set_runtime_env_defaults(
            {
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                "WANDB_MODE": "offline",
            }
        )
        self.add_upstream_paths(repo)

        runtime = self.runtime_loader.load()

        self.cosmos_utils = runtime.cosmos_utils
        self.dataset_stats = runtime.dataset_stats
        self.model = runtime.model
        self.cfg = runtime.cfg
        self.model_adapter = self._create_model_adapter(
            cfg=runtime.cfg,
            model=runtime.model,
            dataset_stats=runtime.dataset_stats,
            cosmos_utils=runtime.cosmos_utils,
        )
        self.upstream_repo = repo
        self.loaded = True

    def warmup(self) -> None:
        self.require_loaded()
        self.native_model_adapter(required=True).warmup()
        self.warmed = True

    def reset(self) -> None:
        self.require_loaded()
        self.native_model_adapter(required=True).reset()

    def close(self) -> None:
        self.model = None
        self.dataset_stats = None
        self.cosmos_utils = None
        self.cfg = None
        super().close()

    def _create_model_adapter(
        self,
        *,
        cfg: Any,
        model: Any,
        dataset_stats: dict[str, Any],
        cosmos_utils: Any,
    ) -> CosmosPolicyModelAdapter:
        return CosmosPolicyModelAdapter(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            cosmos_utils=cosmos_utils,
            checkpoint_path=self.checkpoint_path,
            dataset_stats_path=self.dataset_stats_path,
            text_embeddings_path=self.text_embeddings_path,
            error_cls=self.error_cls,
        )

    def _build_eval_config(self, config_cls: type[Any] = CosmosPolicyEvalConfig) -> Any:
        defaults = self.eval_defaults()
        jpeg_enabled = self.profile_enabled("jpeg_observation_compression")
        jpeg_settings = (
            self.profile_settings("jpeg_observation_compression") if jpeg_enabled else {}
        )
        parallel_enabled = self.profile_enabled("parallel_inference")
        parallel_settings = (
            self.profile_settings("parallel_inference") if parallel_enabled else {}
        )
        cfg = config_cls()
        values: dict[str, Any] = {
            "suite": "libero",
            "config": self.config.get(
                "config",
                defaults.get("config", "cosmos_predict2_2b_480p_libero__inference_only"),
            ),
            "ckpt_path": str(self.checkpoint_path),
            "config_file": self.config.get(
                "config_file",
                defaults.get("config_file", "cosmos_policy/config/config.py"),
            ),
            "use_wrist_image": True,
            "use_proprio": True,
            "normalize_proprio": True,
            "unnormalize_actions": True,
            "dataset_stats_path": str(self.dataset_stats_path),
            "t5_text_embeddings_path": str(self.text_embeddings_path),
            "trained_with_image_aug": True,
            "chunk_size": int(
                self.manifest.defaults.get(
                    "action_horizon",
                    defaults.get("chunk_size", 16),
                )
            ),
            "num_open_loop_steps": int(
                self.manifest.defaults.get(
                    "replan_steps",
                    defaults.get("num_open_loop_steps", 16),
                )
            ),
            "task_suite_name": str(defaults.get("task_suite_name", "libero_10")),
            "randomize_seed": str(defaults.get("randomize_seed", "False")) == "True",
            "seed": int(defaults.get("seed", 195)),
            "use_variance_scale": str(defaults.get("use_variance_scale", "False")) == "True",
            "deterministic": str(defaults.get("deterministic", "True")) == "True",
            "ar_future_prediction": str(defaults.get("ar_future_prediction", "False")) == "True",
            "ar_value_prediction": str(defaults.get("ar_value_prediction", "False")) == "True",
            "use_jpeg_compression": _bool_value(
                jpeg_settings.get(
                    "use_jpeg_compression",
                    defaults.get("use_jpeg_compression", "True"),
                )
            ),
            "flip_images": str(defaults.get("flip_images", "True")) == "True",
            "num_denoising_steps_action": int(defaults.get("num_denoising_steps_action", 5)),
            "num_denoising_steps_future_state": int(
                defaults.get("num_denoising_steps_future_state", 1)
            ),
            "num_denoising_steps_value": int(defaults.get("num_denoising_steps_value", 1)),
            "use_parallel_inference": parallel_enabled
            and _bool_value(parallel_settings.get("use_parallel_inference", "True")),
            "num_queries_best_of_n": int(
                parallel_settings.get(
                    "num_queries_best_of_n",
                    defaults.get("num_queries_best_of_n", 1),
                )
                if parallel_enabled
                else defaults.get("num_queries_best_of_n", 1)
            ),
        }
        for key, value in values.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        for key, value in dict(self.config.get("cfg_overrides", {})).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def _validate_loaded_runtime(self, cfg: Any, model: Any, cosmos_config: Any) -> None:
        model_chunk_size = cosmos_config.dataloader_train.dataset.chunk_size
        if int(cfg.chunk_size) != int(model_chunk_size):
            raise self.error_cls(
                "Cosmos-Policy chunk size mismatch: "
                f"manifest/config={cfg.chunk_size}, model={model_chunk_size}."
            )
        self._check_unnorm_key(cfg, model)

    def _check_unnorm_key(self, cfg: Any, model: Any) -> None:
        unnorm_key = cfg.task_suite_name
        norm_stats = getattr(model, "norm_stats", {})
        if not norm_stats:
            return
        if unnorm_key not in norm_stats and f"{unnorm_key}_no_noops" in norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        if unnorm_key not in norm_stats:
            return
        cfg.unnorm_key = unnorm_key


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"
