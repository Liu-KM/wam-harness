from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from eazywam.backends.fastwam import FastWAMModelAdapter, FastWAMNativeBackendError
from eazywam.backends.fastwam import FastWAMRuntimeBundle
from eazywam.core.manifest import load_builtin_manifest, manifest_from_dict
from eazywam.core.registry import default_registry
from eazywam.core.types import ActionChunk, InferenceRequest, InferenceResult, Observation
from eazywam.processors.fastwam_libero import FastWAMProcessorError, _future_frames_summary


def test_default_registry_exposes_fastwam_native_backend() -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)

    backend = registry.create_backend(manifest, [])

    info = backend.runtime_info()
    assert info.backend == "fastwam"
    assert info.processor == "fastwam_libero"
    assert info.mode == "native"
    assert info.metadata["native"] is True


def test_fastwam_manifests_build_dit_cache_profile() -> None:
    registry = default_registry()

    for model_id in ("fastwam-libero", "fastwam-robotwin"):
        manifest = registry.load_manifest(model_id)
        profiles = registry.build_optimization_profiles(manifest, ["dit_cache"])

        assert "dit_cache" in manifest.supported_optimizations
        assert profiles[0].name == "dit_cache"
        assert profiles[0].params == {"mode": "video_kv"}


def test_fastwam_manifests_build_cuda_graph_profile() -> None:
    registry = default_registry()

    for model_id in ("fastwam-libero", "fastwam-robotwin"):
        manifest = registry.load_manifest(model_id)
        profiles = registry.build_optimization_profiles(manifest, ["cuda_graph"])

        assert "cuda_graph" in manifest.supported_optimizations
        assert profiles[0].name == "cuda_graph"
        assert profiles[0].params == {"mode": "auto", "capture": "action_body"}


def test_fastwam_manifests_build_torch_compile_profile() -> None:
    registry = default_registry()

    for model_id in ("fastwam-libero", "fastwam-robotwin"):
        manifest = registry.load_manifest(model_id)
        profiles = registry.build_optimization_profiles(manifest, ["torch_compile"])

        assert "torch_compile" in manifest.supported_optimizations
        assert profiles[0].name == "torch_compile"
        assert profiles[0].params == {"mode": "auto", "target": "action_body"}


def test_default_registry_exposes_fastwam_libero_processor() -> None:
    registry = default_registry()
    manifest = load_builtin_manifest("fastwam-libero")

    processor = registry.create_processor(manifest)

    limits = processor.modality_limits()
    assert limits["processor"] == "fastwam_libero"
    assert limits["images"] == ["primary", "wrist"]
    assert limits["state"] == "proprio"
    smoke = processor.smoke_observation()
    assert {"primary", "wrist"} <= set(smoke.images)
    assert {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"} <= set(smoke.state)


def test_fastwam_processor_requires_runtime_binding_before_inference() -> None:
    registry = default_registry()
    manifest = load_builtin_manifest("fastwam-libero")
    processor = registry.create_processor(manifest)

    with pytest.raises(FastWAMProcessorError, match="not bound to an upstream FastWAM runtime"):
        processor.to_model_inputs(Observation(images={}, prompt="open the drawer"))


def test_fastwam_processor_summarizes_future_frames_without_embedding_video() -> None:
    assert _future_frames_summary({"video": ["future-a", "future-b"]}) == {
        "present": True,
        "count": 2,
        "format": "frame_list",
        "frame_types": ["str"],
    }


def test_fastwam_native_backend_fails_clearly_for_bad_explicit_upstream_dir(tmp_path) -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"upstream_dir": str(tmp_path / "missing-fastwam")},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    with pytest.raises(FastWAMNativeBackendError, match="FastWAM config directory not found"):
        backend.load()


def test_fastwam_native_required_paths_are_empty_for_vendored_runtime() -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    required_paths = backend.native_required_upstream_paths()

    assert required_paths == ()


def test_fastwam_native_required_paths_for_explicit_upstream_are_config_only(tmp_path) -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"upstream_dir": str(tmp_path / "FastWAM")},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    required_paths = backend.native_required_upstream_paths()

    assert "experiments/libero/eval_libero_single.py" not in required_paths
    assert "experiments/libero/run_libero_manager.py" not in required_paths
    assert "src/fastwam/runtime.py" not in required_paths
    assert set(required_paths) >= {
        "configs/train.yaml",
        "configs/task/libero_uncond_2cam224_1e-4.yaml",
        "configs/data/libero_2cam.yaml",
        "configs/model/fastwam.yaml",
    }


def test_fastwam_native_backend_load_binds_runtime_loader_bundle(tmp_path) -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    repo = tmp_path / "FastWAM"
    cache = tmp_path / "cache"
    checkpoint = cache / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt"
    dataset_stats = (
        cache
        / "checkpoints"
        / "fastwam_release"
        / "libero_uncond_2cam224_dataset_stats.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    dataset_stats.write_text("{}", encoding="utf-8")
    data["assets"]["checkpoint"]["local_path"] = str(checkpoint)
    data["assets"]["dataset_stats"]["local_path"] = str(dataset_stats)
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    processor = registry.create_processor(manifest)
    backend.attach_processor(processor)
    _write_fastwam_required_paths(repo, backend.native_required_upstream_paths())
    runtime_loader = _FakeFastWAMRuntimeLoader()
    backend.runtime_loader = runtime_loader

    backend.load()

    assert runtime_loader.config_dir == (repo / "configs").resolve()
    assert runtime_loader.checkpoint_path == checkpoint.resolve()
    assert runtime_loader.dataset_stats_path == dataset_stats.resolve()
    assert backend.loaded is True
    assert backend.upstream_repo == (repo / "configs")
    assert backend.model is runtime_loader.bundle.model
    assert backend.cfg is runtime_loader.bundle.cfg
    assert backend.device == "cpu"
    assert isinstance(backend.model_adapter, FastWAMModelAdapter)
    assert backend.processor.upstream_processor is runtime_loader.bundle.upstream_processor
    assert backend.processor.prompt_template == "template: {task}"


def test_fastwam_native_backend_derives_diffsynth_root_from_wan_file_asset(tmp_path) -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    wan22_vae = (
        tmp_path
        / "diffsynth-models"
        / "Wan-AI"
        / "Wan2.2-TI2V-5B"
        / "Wan2.2_VAE.pth"
    )
    wan22_vae.parent.mkdir(parents=True)
    wan22_vae.write_text("asset\n", encoding="utf-8")
    data["assets"]["wan22_vae"]["local_path"] = str(wan22_vae)
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)

    backend = registry.create_backend(manifest, [])

    assert backend._diffsynth_model_base_path() == tmp_path / "diffsynth-models"


def test_fastwam_native_backend_derives_diffsynth_root_from_custom_wan_asset_path() -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["assets"]["wan22_vae"]["local_path"] = (
        "/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    )
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)

    backend = registry.create_backend(manifest, [])

    assert backend._diffsynth_model_base_path() == Path("/models")


def test_fastwam_native_backend_prefers_configured_cache_for_missing_wan_file(
    tmp_path,
) -> None:
    registry = default_registry()
    cache_dir = tmp_path / "cache"
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"cache_dir": str(cache_dir)},
    }
    manifest = manifest_from_dict(data)

    backend = registry.create_backend(manifest, [])

    assert backend._diffsynth_model_base_path() == cache_dir / "diffsynth-models"


def test_fastwam_native_backend_matches_official_infer_action_kwargs() -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    model = _ModelRequiringNumVideoFrames()
    backend.model = model
    backend.processor = _BoundFastWAMProcessor()
    backend.cfg = _fastwam_cfg(seed=123)
    backend.loaded = True
    backend.warmed = True
    backend.no_grad = lambda: nullcontext()

    result = backend.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=32,
            replan_steps=10,
            runtime_options={"num_inference_steps": 7},
        )
    )

    assert [result.action_chunk.horizon, result.action_chunk.action_dim] == [1, 7]
    assert model.kwargs["action_horizon"] == 32
    assert model.kwargs["num_video_frames"] == 9
    assert model.kwargs["negative_prompt"] == "avoid blur"
    assert model.kwargs["num_inference_steps"] == 7
    assert model.kwargs["cache_mode"] == "video_kv"
    assert model.kwargs["seed"] == 123
    assert isinstance(backend.model_adapter, FastWAMModelAdapter)
    assert result.backend_metadata["model_adapter"] == "fastwam_model"
    assert result.backend_metadata["fastwam_call"] == "infer_action"


def test_fastwam_model_adapter_passes_profile_cache_mode_and_metadata() -> None:
    adapter = FastWAMModelAdapter(
        model=_ModelRequiringNumVideoFrames(),
        cfg=_fastwam_cfg(),
        checkpoint_path=None,
        dataset_stats_path=None,
        config={},
        dit_cache_params={"mode": "recompute"},
        no_grad_factory=lambda: nullcontext(),
        error_cls=FastWAMNativeBackendError,
    )

    call = adapter.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=32,
            replan_steps=10,
        ),
        {"prompt": "prompt", "input_image": "image", "proprio": "proprio"},
    )

    assert adapter.model.kwargs["cache_mode"] == "recompute"
    assert call.metadata["fastwam_call"] == "infer_action"
    assert call.metadata["dit_cache_enabled"] is False
    assert call.metadata["dit_cache_mode"] == "recompute"
    assert call.metadata["dit_cache_hook"] == "fastwam_video_kv_cache"
    assert call.metadata["num_inference_steps"] == 10
    assert call.metadata["video_seq_len"] == 2
    assert call.metadata["action_seq_len"] == 32
    assert call.metadata["cache_layers"] == 0
    assert call.metadata["cache_prefill_wall_ms"] is None
    assert call.metadata["denoise_wall_ms"] == 1.0
    assert call.metadata["cache_bytes"] is None


def test_fastwam_model_adapter_runtime_options_switch_cached_and_recompute_modes() -> None:
    adapter = FastWAMModelAdapter(
        model=_ModelRequiringNumVideoFrames(),
        cfg=_fastwam_cfg(),
        checkpoint_path=None,
        dataset_stats_path=None,
        config={},
        dit_cache_params={"mode": "video_kv"},
        no_grad_factory=lambda: nullcontext(),
        error_cls=FastWAMNativeBackendError,
    )

    cached = adapter.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=32,
            replan_steps=10,
        ),
        {"prompt": "prompt", "input_image": "image", "proprio": "proprio"},
    )
    recompute = adapter.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=32,
            replan_steps=10,
            runtime_options={"dit_cache_mode": "recompute"},
        ),
        {"prompt": "prompt", "input_image": "image", "proprio": "proprio"},
    )

    assert cached.metadata["dit_cache_enabled"] is True
    assert cached.metadata["dit_cache_mode"] == "video_kv"
    assert cached.metadata["cache_layers"] == 1
    assert cached.metadata["cache_prefill_wall_ms"] == 0.5
    assert cached.metadata["cache_bytes"] == 128
    assert recompute.metadata["dit_cache_enabled"] is False
    assert recompute.metadata["dit_cache_mode"] == "recompute"
    assert recompute.metadata["cache_layers"] == 0


def test_fastwam_model_adapter_passes_cuda_graph_profile_mode_and_metadata() -> None:
    adapter = FastWAMModelAdapter(
        model=_ModelRequiringNumVideoFrames(),
        cfg=_fastwam_cfg(),
        checkpoint_path=None,
        dataset_stats_path=None,
        config={},
        dit_cache_params={"mode": "video_kv"},
        no_grad_factory=lambda: nullcontext(),
        error_cls=FastWAMNativeBackendError,
        cuda_graph_params={"mode": "auto", "capture": "action_body"},
        cuda_graph_enabled=True,
    )

    call = adapter.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=32,
            replan_steps=10,
        ),
        {"prompt": "prompt", "input_image": "image", "proprio": "proprio"},
    )

    assert adapter.model.kwargs["cache_mode"] == "video_kv"
    assert adapter.model.kwargs["cuda_graph_mode"] == "auto"
    assert call.metadata["cuda_graph_enabled"] is True
    assert call.metadata["cuda_graph_mode"] == "auto"
    assert call.metadata["cuda_graph_hook"] == "fastwam_cuda_graph_action_body"


def test_fastwam_model_adapter_runtime_options_disable_cuda_graph() -> None:
    adapter = FastWAMModelAdapter(
        model=_ModelRequiringNumVideoFrames(),
        cfg=_fastwam_cfg(),
        checkpoint_path=None,
        dataset_stats_path=None,
        config={},
        dit_cache_params={"mode": "video_kv"},
        no_grad_factory=lambda: nullcontext(),
        error_cls=FastWAMNativeBackendError,
        cuda_graph_params={"mode": "auto"},
        cuda_graph_enabled=True,
    )

    call = adapter.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=32,
            replan_steps=10,
            runtime_options={"cuda_graph_mode": "off"},
        ),
        {"prompt": "prompt", "input_image": "image", "proprio": "proprio"},
    )

    assert adapter.model.kwargs["cuda_graph_mode"] == "off"
    assert call.metadata["cuda_graph_enabled"] is False
    assert call.metadata["cuda_graph_mode"] == "off"


def test_fastwam_model_adapter_passes_torch_compile_profile_mode_and_metadata() -> None:
    adapter = FastWAMModelAdapter(
        model=_ModelRequiringNumVideoFrames(),
        cfg=_fastwam_cfg(),
        checkpoint_path=None,
        dataset_stats_path=None,
        config={},
        dit_cache_params={"mode": "video_kv"},
        no_grad_factory=lambda: nullcontext(),
        error_cls=FastWAMNativeBackendError,
        torch_compile_params={"mode": "auto", "target": "action_body"},
        torch_compile_enabled=True,
    )

    call = adapter.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=32,
            replan_steps=10,
        ),
        {"prompt": "prompt", "input_image": "image", "proprio": "proprio"},
    )

    assert adapter.model.kwargs["torch_compile_mode"] == "auto"
    assert call.metadata["torch_compile_enabled"] is True
    assert call.metadata["torch_compile_mode"] == "auto"
    assert call.metadata["torch_compile_hook"] == "fastwam_torch_compile_action_body"


def test_fastwam_model_adapter_runtime_options_disable_torch_compile() -> None:
    adapter = FastWAMModelAdapter(
        model=_ModelRequiringNumVideoFrames(),
        cfg=_fastwam_cfg(),
        checkpoint_path=None,
        dataset_stats_path=None,
        config={},
        dit_cache_params={"mode": "video_kv"},
        no_grad_factory=lambda: nullcontext(),
        error_cls=FastWAMNativeBackendError,
        torch_compile_params={"mode": "auto"},
        torch_compile_enabled=True,
    )

    call = adapter.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=32,
            replan_steps=10,
            runtime_options={"torch_compile_mode": "off"},
        ),
        {"prompt": "prompt", "input_image": "image", "proprio": "proprio"},
    )

    assert adapter.model.kwargs["torch_compile_mode"] == "off"
    assert call.metadata["torch_compile_enabled"] is False
    assert call.metadata["torch_compile_mode"] == "off"


def test_fastwam_model_adapter_rejects_invalid_cache_mode() -> None:
    adapter = FastWAMModelAdapter(
        model=_ModelRequiringNumVideoFrames(),
        cfg=_fastwam_cfg(),
        checkpoint_path=None,
        dataset_stats_path=None,
        config={},
        dit_cache_params={"mode": "bad"},
        no_grad_factory=lambda: nullcontext(),
        error_cls=FastWAMNativeBackendError,
    )

    with pytest.raises(FastWAMNativeBackendError, match="dit_cache mode"):
        adapter.infer(
            InferenceRequest(
                observation=Observation(images={}, prompt="open the drawer"),
                action_horizon=32,
                replan_steps=10,
            ),
            {"prompt": "prompt", "input_image": "image", "proprio": "proprio"},
        )


def test_fastwam_native_backend_uses_infer_joint_for_future_video() -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    model = _ModelWithInferJoint()
    backend.model = model
    backend.processor = _BoundFastWAMProcessor()
    backend.cfg = _fastwam_cfg(visualize_future_video=True)
    backend.loaded = True
    backend.warmed = True
    backend.no_grad = lambda: nullcontext()

    result = backend.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=32,
            replan_steps=10,
        )
    )

    assert model.joint_kwargs is not None
    assert model.action_called is False
    assert model.joint_kwargs["action_horizon"] == 32
    assert model.joint_kwargs["num_video_frames"] == 9
    assert "cache_mode" not in model.joint_kwargs
    assert result.backend_metadata["fastwam_call"] == "infer_joint"
    assert result.backend_metadata["future_video_present"] is True
    assert result.future_frames == {
        "present": True,
        "count": 1,
        "format": "frame_list",
        "frame_types": ["str"],
    }
    assert isinstance(backend.model_adapter, FastWAMModelAdapter)


def test_fastwam_model_adapter_close_releases_model_reference() -> None:
    adapter = FastWAMModelAdapter(
        model=_ModelRequiringNumVideoFrames(),
        cfg=_fastwam_cfg(),
        checkpoint_path=None,
        dataset_stats_path=None,
        config={},
        dit_cache_params={},
        no_grad_factory=lambda: nullcontext(),
        error_cls=FastWAMNativeBackendError,
    )

    assert adapter.runtime_metadata()["model_adapter"] == "fastwam_model"
    assert adapter.runtime_metadata()["model_class"] == "_ModelRequiringNumVideoFrames"

    adapter.close()

    with pytest.raises(FastWAMNativeBackendError, match="adapter is not loaded"):
        adapter.require_ready()


def test_fastwam_native_backend_rejects_invalid_future_video_config() -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    cfg = _fastwam_cfg(visualize_future_video=True)
    cfg.model = SimpleNamespace(video_dit_config={"action_conditioned": True})

    with pytest.raises(
        FastWAMNativeBackendError,
        match="visualize_future_video=true requires",
    ):
        backend._validate_visualize_future_video_config(cfg)


def test_fastwam_dit_cache_profile_status_planned_fallback_and_applied() -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)
    profiles = registry.build_optimization_profiles(manifest, ["dit_cache"])
    backend = registry.create_backend(manifest, profiles)

    planned = backend.plan_optimization_profiles(profiles)
    fallback = backend.apply_loaded_optimization_profiles(profiles)
    backend.model = _ModelWithFastWAMCacheHook()
    backend.loaded = True
    applied = backend.apply_loaded_optimization_profiles(profiles)

    assert planned[0]["state"] == "planned"
    assert planned[0]["hook"] == "fastwam_video_kv_cache"
    assert fallback[0]["state"] == "fallback"
    assert fallback[0]["reason"] == "backend_not_loaded"
    assert applied[0]["state"] == "applied"
    assert applied[0]["hook"] == "fastwam_video_kv_cache"


def test_fastwam_cuda_graph_profile_status_planned_fallback_and_applied() -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)
    profiles = registry.build_optimization_profiles(manifest, ["cuda_graph"])
    backend = registry.create_backend(manifest, profiles)

    planned = backend.plan_optimization_profiles(profiles)
    fallback = backend.apply_loaded_optimization_profiles(profiles)
    backend.model = _ModelWithFastWAMCudaGraphHook()
    backend.loaded = True
    applied = backend.apply_loaded_optimization_profiles(profiles)

    assert planned[0]["state"] == "planned"
    assert planned[0]["hook"] == "fastwam_cuda_graph_action_body"
    assert fallback[0]["state"] == "fallback"
    assert fallback[0]["reason"] == "backend_not_loaded"
    assert applied[0]["state"] == "applied"
    assert applied[0]["hook"] == "fastwam_cuda_graph_action_body"


def test_fastwam_torch_compile_profile_status_planned_fallback_and_applied() -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {"name": "fastwam", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)
    profiles = registry.build_optimization_profiles(manifest, ["torch_compile"])
    backend = registry.create_backend(manifest, profiles)

    planned = backend.plan_optimization_profiles(profiles)
    fallback = backend.apply_loaded_optimization_profiles(profiles)
    backend.model = _ModelWithFastWAMTorchCompileHook()
    backend.loaded = True
    applied = backend.apply_loaded_optimization_profiles(profiles)

    assert planned[0]["state"] == "planned"
    assert planned[0]["hook"] == "fastwam_torch_compile_action_body"
    assert fallback[0]["state"] == "fallback"
    assert fallback[0]["reason"] == "backend_not_loaded"
    assert applied[0]["state"] == "applied"
    assert applied[0]["hook"] == "fastwam_torch_compile_action_body"


class _ModelRequiringNumVideoFrames:
    kwargs = None

    def infer_action(
        self,
        *,
        prompt,
        input_image,
        action_horizon,
        num_video_frames,
        **kwargs,
    ):
        self.kwargs = {
            "prompt": prompt,
            "input_image": input_image,
            "action_horizon": action_horizon,
            "num_video_frames": num_video_frames,
            **kwargs,
        }
        cache_mode = str(kwargs.get("cache_mode", "video_kv"))
        cuda_graph_mode = str(kwargs.get("cuda_graph_mode", "off"))
        torch_compile_mode = str(kwargs.get("torch_compile_mode", "off"))
        return {
            "action": [[0.0] * 7],
            "metadata": {
                "dit_cache_enabled": cache_mode == "video_kv",
                "dit_cache_mode": cache_mode,
                "dit_cache_hook": "fastwam_video_kv_cache",
                "num_inference_steps": int(kwargs.get("num_inference_steps", 0)),
                "video_seq_len": 2,
                "action_seq_len": action_horizon,
                "cache_layers": 1 if cache_mode == "video_kv" else 0,
                "cache_prefill_wall_ms": 0.5 if cache_mode == "video_kv" else None,
                "denoise_wall_ms": 1.0,
                "cache_bytes": 128 if cache_mode == "video_kv" else None,
                "cuda_graph_enabled": cuda_graph_mode != "off",
                "cuda_graph_mode": cuda_graph_mode,
                "cuda_graph_hook": "fastwam_cuda_graph_action_body",
                "cuda_graph_capture_success": False,
                "cuda_graph_replay_count": 0,
                "cuda_graph_fallback_reason": None,
                "cuda_graph_shape_key": None,
                "cuda_graph_capture_wall_ms": None,
                "torch_compile_enabled": torch_compile_mode != "off",
                "torch_compile_mode": torch_compile_mode,
                "torch_compile_hook": "fastwam_torch_compile_action_body",
                "torch_compile_success": False,
                "torch_compile_fallback_reason": None,
                "torch_compile_wall_ms": None,
            },
        }

    def to(self, device):
        return self

    def eval(self):
        return self


class _FastWAMCacheHookMot:
    def prefill_video_cache(self):
        return None

    def forward_action_with_video_cache(self):
        return None


class _ModelWithFastWAMCacheHook:
    mot = _FastWAMCacheHookMot()

    def infer_action(self, *, cache_mode="video_kv"):
        return {"action": [[0.0] * 7]}


class _ModelWithFastWAMCudaGraphHook:
    mot = _FastWAMCacheHookMot()

    def infer_action(self, *, cache_mode="video_kv", cuda_graph_mode="auto"):
        return {"action": [[0.0] * 7]}


class _ModelWithFastWAMTorchCompileHook:
    mot = _FastWAMCacheHookMot()

    def infer_action(self, *, cache_mode="video_kv", torch_compile_mode="auto"):
        return {"action": [[0.0] * 7]}


class _ModelWithInferJoint:
    joint_kwargs = None
    action_called = False

    def infer_action(self, **kwargs):
        self.action_called = True
        return {"action": [[0.0] * 7]}

    def infer_joint(self, **kwargs):
        self.joint_kwargs = dict(kwargs)
        return {"action": [[0.0] * 7], "video": ["future"]}


class _BoundFastWAMProcessor:
    def to_model_inputs(self, observation):
        return {
            "prompt": f"prompt: {observation.prompt}",
            "input_image": "image",
            "proprio": "proprio",
        }

    def to_harness_result(self, raw_output):
        return InferenceResult(
            action_chunk=ActionChunk(actions=[[0.0] * 7]),
            backend_metadata={"raw_keys": sorted(raw_output)},
            future_frames=_future_frames_summary(raw_output),
        )


def _fastwam_cfg(*, visualize_future_video: bool = False, seed=None):
    cfg = {
        "EVALUATION": {
            "negative_prompt": "avoid blur",
            "text_cfg_scale": 1.0,
            "num_inference_steps": 10,
            "rand_device": "cpu",
            "tiled": False,
            "visualize_future_video": visualize_future_video,
        },
        "seed": seed,
        "eval_num_inference_steps": 10,
    }
    cfg = _AttrDict(cfg)
    cfg.data = SimpleNamespace(
        train=SimpleNamespace(num_frames=33, action_video_freq_ratio=4)
    )
    cfg.model = SimpleNamespace(video_dit_config={"action_conditioned": False})
    return cfg


class _AttrDict(dict):
    pass


class _FakeFastWAMRuntimeLoader:
    def __init__(self) -> None:
        self.config_dir = None
        self.checkpoint_path = None
        self.dataset_stats_path = None
        self.bundle = FastWAMRuntimeBundle(
            model=_ModelRequiringNumVideoFrames(),
            cfg=_fastwam_cfg(),
            upstream_processor=object(),
            prompt_template="template: {task}",
            device="cpu",
            dtype="fp32",
            checkpoint_path=Path("checkpoint"),
            dataset_stats_path=Path("dataset_stats"),
        )

    def load(self, *, config_dir, checkpoint_path, dataset_stats_path):
        self.config_dir = config_dir.resolve()
        self.checkpoint_path = checkpoint_path
        self.dataset_stats_path = dataset_stats_path
        self.bundle = FastWAMRuntimeBundle(
            model=self.bundle.model,
            cfg=self.bundle.cfg,
            upstream_processor=self.bundle.upstream_processor,
            prompt_template=self.bundle.prompt_template,
            device=self.bundle.device,
            dtype=self.bundle.dtype,
            checkpoint_path=checkpoint_path,
            dataset_stats_path=dataset_stats_path,
        )
        return self.bundle


def _write_fastwam_required_paths(repo, required_paths) -> None:
    for relative in required_paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# smoke\n", encoding="utf-8")
