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
    assert model.kwargs["seed"] == 123
    assert isinstance(backend.model_adapter, FastWAMModelAdapter)
    assert result.backend_metadata["model_adapter"] == "fastwam_model"
    assert result.backend_metadata["fastwam_call"] == "infer_action"


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
        return {"action": [[0.0] * 7]}

    def to(self, device):
        return self

    def eval(self):
        return self


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
