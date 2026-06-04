import subprocess
from types import SimpleNamespace

import pytest

from wam_harness.backends.cosmos_policy import (
    CosmosPolicyBackend,
    CosmosPolicyRuntimeBundle,
    CosmosPolicyModelAdapter,
    CosmosPolicyNativeBackendError,
)
from wam_harness.backends.dreamzero import (
    DreamZeroBackend,
    DreamZeroNativeBackendError,
    DreamZeroPolicyServerAdapter,
    DreamZeroPolicyServerRuntimeBundle,
)
from wam_harness.backends.fastwam import FastWAMBackend
from wam_harness.backends.native import (
    NativeBackendBase,
    NativeModelAdapter,
    NativeModelCall,
    NativeRuntimeLoader,
)
from wam_harness.core.manifest import load_builtin_manifest, manifest_from_dict
from wam_harness.core.registry import default_registry
from wam_harness.core.types import ActionChunk, InferenceRequest, InferenceResult, Observation


def _native_manifest(model_id: str, backend_name: str, tmp_path):
    data = load_builtin_manifest(model_id).to_dict()
    data["backend"] = {
        "name": backend_name,
        "mode": "native",
        "config": {"upstream_dir": str(tmp_path / f"missing-{backend_name}")},
    }
    return manifest_from_dict(data)


def _write_fastwam_required_paths(repo) -> None:
    paths = [
        "src/fastwam/runtime.py",
        "src/fastwam/utils/config_resolvers.py",
        "src/fastwam/datasets/lerobot/robot_video_dataset.py",
        "src/fastwam/datasets/lerobot/utils/normalizer.py",
        "configs/sim_libero.yaml",
        "configs/train.yaml",
        "configs/data/libero_2cam.yaml",
        "configs/model/fastwam.yaml",
        "configs/task/libero_uncond_2cam224_1e-4.yaml",
    ]
    for relative in paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# smoke\n", encoding="utf-8")


def _write_cosmos_required_paths(repo) -> None:
    paths = [
        "cosmos_policy/experiments/robot/cosmos_utils.py",
        "cosmos_policy/config/config.py",
    ]
    for relative in paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# smoke\n", encoding="utf-8")


def test_default_registry_exposes_cosmos_policy_native_backend(tmp_path) -> None:
    registry = default_registry()
    manifest = _native_manifest("cosmos-policy-libero", "cosmos_policy", tmp_path)

    backend = registry.create_backend(manifest, [])
    processor = registry.create_processor(manifest)

    assert backend.runtime_info().backend == "cosmos_policy"
    assert processor.modality_limits()["processor"] == "cosmos_policy_libero"
    smoke = processor.smoke_observation()
    assert {"primary", "wrist"} <= set(smoke.images)


def test_native_backend_base_owns_inference_spine() -> None:
    data = load_builtin_manifest("fake-open-loop").to_dict()
    data["backend"] = {"name": "spine", "mode": "native", "config": {}}
    manifest = manifest_from_dict(data)
    backend = _SpineBackend(manifest, [])

    result = backend.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="smoke"),
            action_horizon=2,
            replan_steps=1,
        )
    )

    assert backend.seen_model_inputs == {"prompt": "smoke"}
    assert result.action_chunk.actions == [[1.0, 2.0], [3.0, 4.0]]
    assert set(result.timing) == {"preprocess_ms", "model_ms", "postprocess_ms", "total_ms"}
    assert result.backend_metadata == {
        "processor": "spine",
        "native_backend": True,
        "backend_meta": "shared",
        "call_meta": "native-call",
    }
    assert result.warnings == [
        "processor-warning",
        "backend-warning",
        "call-warning",
    ]


def test_native_model_adapter_defines_runtime_boundary() -> None:
    adapter = NativeModelAdapter()

    assert adapter.model_timing_key() == "model_ms"
    assert adapter.runtime_metadata() == {"model_adapter": "native_model_adapter"}
    assert adapter.inference_metadata() == {"model_adapter": "native_model_adapter"}
    assert adapter.inference_warnings() == []
    with pytest.raises(NotImplementedError, match="must implement infer"):
        adapter.infer(
            InferenceRequest(
                observation=Observation(images={}, prompt="smoke"),
                action_horizon=1,
                replan_steps=1,
            ),
            {},
        )


def test_native_runtime_loader_defines_runtime_boundary() -> None:
    loader = NativeRuntimeLoader()

    assert loader.name == "native_runtime_loader"
    assert loader.runtime_mode is None
    with pytest.raises(NotImplementedError, match="must implement load"):
        loader.load()


def test_real_native_backends_do_not_override_shared_infer_spine() -> None:
    assert FastWAMBackend.infer is NativeBackendBase.infer
    assert CosmosPolicyBackend.infer is NativeBackendBase.infer
    assert DreamZeroBackend.infer is NativeBackendBase.infer
    assert FastWAMBackend._infer_model is NativeBackendBase._infer_model
    assert CosmosPolicyBackend._infer_model is NativeBackendBase._infer_model
    assert DreamZeroBackend._infer_model is NativeBackendBase._infer_model


def test_cosmos_policy_native_backend_declares_actual_imported_upstream_files(
    tmp_path,
) -> None:
    registry = default_registry()
    repo = tmp_path / "cosmos-policy"
    _write_cosmos_required_paths(repo)
    data = load_builtin_manifest("cosmos-policy-libero").to_dict()
    data["backend"] = {
        "name": "cosmos_policy",
        "mode": "native",
        "config": {"upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    requirements = backend.native_requirements()

    assert requirements.runtime_mode == "in_process"
    assert requirements.runtime_loader == "cosmos_policy_runtime_loader"
    assert requirements.model_adapter == "cosmos_policy_model"
    assert requirements.upstream.status == "present"
    assert requirements.upstream.required_paths == [
        "cosmos_policy/experiments/robot/cosmos_utils.py",
        "cosmos_policy/config/config.py",
    ]
    assert "cosmos_policy/experiments/robot/libero/run_libero_eval.py" not in (
        requirements.upstream.required_paths
    )


def test_dreamzero_native_backend_checks_configured_server_module_path(
    tmp_path,
) -> None:
    registry = default_registry()
    repo = tmp_path / "dreamzero"
    server_module = repo / "custom" / "policy_server.py"
    server_module.parent.mkdir(parents=True)
    server_module.write_text("# smoke\n", encoding="utf-8")
    data = load_builtin_manifest("dreamzero-droid-sim").to_dict()
    data["backend"] = {
        "name": "dreamzero",
        "mode": "native",
        "config": {
            "upstream_dir": str(repo),
            "server_module": "custom.policy_server",
        },
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    requirements = backend.native_requirements()

    assert requirements.upstream.status == "present"
    assert requirements.upstream.required_paths == ["custom/policy_server.py"]
    assert requirements.runtime_mode == "resident_server"
    assert requirements.runtime_loader == "dreamzero_policy_server_runtime_loader"


def test_dreamzero_native_backend_default_path_is_policy_server_not_sim_eval(
    tmp_path,
) -> None:
    registry = default_registry()
    repo = tmp_path / "dreamzero"
    server_module = repo / "eval_utils" / "serve_dreamzero_wan22.py"
    server_module.parent.mkdir(parents=True)
    server_module.write_text("# smoke\n", encoding="utf-8")
    data = load_builtin_manifest("dreamzero-droid-sim").to_dict()
    data["backend"] = {
        "name": "dreamzero",
        "mode": "native",
        "config": {"upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    requirements = backend.native_requirements()

    assert requirements.upstream.status == "present"
    assert requirements.runtime_mode == "resident_server"
    assert requirements.runtime_loader == "dreamzero_policy_server_runtime_loader"
    assert requirements.upstream.required_paths == ["eval_utils/serve_dreamzero_wan22.py"]
    assert "eval_utils/run_sim_eval.py" not in requirements.upstream.required_paths


def test_native_backend_declares_and_checks_upstream_requirements(tmp_path) -> None:
    registry = default_registry()
    repo = tmp_path / "FastWAM"
    _write_fastwam_required_paths(repo)
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"cache_dir": str(tmp_path / "cache"), "upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    requirements = backend.native_requirements()
    readiness = backend.native_readiness()

    assert requirements.backend == "fastwam"
    assert requirements.runtime_mode == "in_process"
    assert requirements.runtime_loader == "fastwam_runtime_loader"
    assert requirements.required_assets == ["checkpoint", "dataset_stats"]
    assert requirements.runtime_assets == [
        "checkpoint",
        "dataset_stats",
        "model_base",
        "tokenizer_components",
    ]
    assert requirements.required_python_modules == [
        "torch",
        "hydra",
        "hydra.core.global_hydra",
        "hydra.utils",
        "omegaconf",
        "numpy",
        "PIL.Image",
        "einops",
    ]
    assert requirements.upstream.status == "present"
    assert requirements.upstream.selected == str(repo.resolve())
    assert requirements.upstream.required_paths == [
        "src/fastwam/runtime.py",
        "src/fastwam/utils/config_resolvers.py",
        "src/fastwam/datasets/lerobot/robot_video_dataset.py",
        "src/fastwam/datasets/lerobot/utils/normalizer.py",
        "configs/sim_libero.yaml",
        "configs/train.yaml",
        "configs/task/libero_uncond_2cam224_1e-4.yaml",
        "configs/data/libero_2cam.yaml",
        "configs/model/fastwam.yaml",
    ]
    assert requirements.upstream.expected_commit == "45d8e14"
    assert requirements.upstream.commit_status == "unknown"
    assert readiness.status == "blocked"
    assert readiness.missing_required_assets == ["checkpoint", "dataset_stats"]
    assert readiness.missing_runtime_assets == ["model_base", "tokenizer_components"]
    assert "torch" in readiness.missing_python_modules


def test_fastwam_native_backend_derives_hydra_paths_from_task_name(
    tmp_path,
) -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {
            "upstream_dir": str(tmp_path / "FastWAM"),
            "task": "robotwin_idm_3cam_384_1e-4",
        },
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    assert set(backend.native_required_upstream_paths()) >= {
        "configs/train.yaml",
        "configs/task/robotwin_idm_3cam_384_1e-4.yaml",
        "configs/data/robotwin.yaml",
        "configs/model/fastwam_idm.yaml",
    }


def test_native_backend_readiness_warns_when_only_runtime_assets_are_missing(
    tmp_path,
) -> None:
    registry = default_registry()
    repo = tmp_path / "FastWAM"
    _write_fastwam_required_paths(repo)
    cache_dir = tmp_path / "cache"
    checkpoint = cache_dir / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt"
    dataset_stats = (
        cache_dir
        / "checkpoints"
        / "fastwam_release"
        / "libero_uncond_2cam224_dataset_stats.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    dataset_stats.write_text("{}", encoding="utf-8")
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"cache_dir": str(cache_dir), "upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    backend.required_python_modules = ()

    readiness = backend.native_readiness()

    assert readiness.status == "warning"
    assert readiness.to_dict()["runtime_mode"] == "in_process"
    assert readiness.to_dict()["runtime_loader"] == "fastwam_runtime_loader"
    assert readiness.missing_required_assets == []
    assert readiness.missing_runtime_assets == ["model_base", "tokenizer_components"]


def test_native_backend_readiness_is_ready_when_fastwam_runtime_assets_are_in_cache(
    tmp_path,
) -> None:
    registry = default_registry()
    repo = tmp_path / "FastWAM"
    _write_fastwam_required_paths(repo)
    cache_dir = tmp_path / "cache"
    paths = [
        cache_dir / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt",
        cache_dir
        / "checkpoints"
        / "fastwam_release"
        / "libero_uncond_2cam224_dataset_stats.json",
        cache_dir / "diffsynth-models" / "Wan-AI" / "Wan2.2-TI2V-5B",
        cache_dir / "diffsynth-models" / "Wan-AI" / "Wan2.1-T2V-1.3B",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset\n", encoding="utf-8")
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"cache_dir": str(cache_dir), "upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    backend.required_python_modules = ()

    readiness = backend.native_readiness()

    assert readiness.status == "ready"
    assert readiness.missing_required_assets == []
    assert readiness.missing_runtime_assets == []


def test_native_backend_readiness_warns_on_upstream_commit_mismatch(
    tmp_path,
) -> None:
    registry = default_registry()
    repo = tmp_path / "FastWAM"
    _write_fastwam_required_paths(repo)
    cache_dir = tmp_path / "cache"
    paths = [
        cache_dir / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt",
        cache_dir
        / "checkpoints"
        / "fastwam_release"
        / "libero_uncond_2cam224_dataset_stats.json",
        cache_dir / "diffsynth-models" / "Wan-AI" / "Wan2.2-TI2V-5B",
        cache_dir / "diffsynth-models" / "Wan-AI" / "Wan2.1-T2V-1.3B",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset\n", encoding="utf-8")
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"cache_dir": str(cache_dir), "upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    backend.required_python_modules = ()
    backend.selected_upstream_commit = lambda _repo: "0000000000000000000000000000000000000000"

    readiness = backend.native_readiness()

    assert readiness.status == "warning"
    assert readiness.requirements.upstream.commit_status == "mismatch"
    assert readiness.requirements.upstream.expected_commit == "45d8e14"
    assert readiness.requirements.upstream.selected_commit.startswith("000000")


def test_native_backend_readiness_blocks_when_python_modules_are_missing(
    tmp_path,
) -> None:
    registry = default_registry()
    repo = tmp_path / "FastWAM"
    _write_fastwam_required_paths(repo)
    cache_dir = tmp_path / "cache"
    paths = [
        cache_dir / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt",
        cache_dir
        / "checkpoints"
        / "fastwam_release"
        / "libero_uncond_2cam224_dataset_stats.json",
        cache_dir / "diffsynth-models" / "Wan-AI" / "Wan2.2-TI2V-5B",
        cache_dir / "diffsynth-models" / "Wan-AI" / "Wan2.1-T2V-1.3B",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset\n", encoding="utf-8")
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"cache_dir": str(cache_dir), "upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    backend.required_python_modules = ("definitely_missing_wam_module",)

    readiness = backend.native_readiness()

    assert readiness.status == "blocked"
    assert readiness.missing_required_assets == []
    assert readiness.missing_runtime_assets == []
    assert readiness.missing_python_modules == ["definitely_missing_wam_module"]
    assert readiness.to_dict()["missing_python_modules"] == [
        "definitely_missing_wam_module"
    ]


def test_other_native_backends_declare_runtime_assets(tmp_path) -> None:
    registry = default_registry()
    cosmos = registry.create_backend(
        _native_manifest("cosmos-policy-libero", "cosmos_policy", tmp_path),
        [],
    )
    dreamzero = registry.create_backend(
        _native_manifest("dreamzero-droid-sim", "dreamzero", tmp_path),
        [],
    )

    assert cosmos.native_requirements().runtime_assets == [
        "checkpoint",
        "dataset_stats",
        "text_embeddings",
        "tokenizer",
    ]
    assert dreamzero.native_requirements().runtime_assets == ["checkpoint"]
    assert cosmos.native_requirements().runtime_mode == "in_process"
    assert cosmos.native_requirements().runtime_loader == "cosmos_policy_runtime_loader"
    assert dreamzero.native_requirements().runtime_mode == "resident_server"
    assert dreamzero.native_requirements().runtime_loader == (
        "dreamzero_policy_server_runtime_loader"
    )
    assert cosmos.native_requirements().model_adapter == "cosmos_policy_model"
    assert dreamzero.native_requirements().model_adapter == "dreamzero_policy_server"


class _SpineBackend(NativeBackendBase):
    def __init__(self, manifest, profiles) -> None:
        super().__init__(manifest, profiles, backend_label="Spine")
        self.processor = _SpineProcessor()
        self.loaded = True
        self.warmed = True
        self.seen_model_inputs = None

    def native_inference_metadata(self) -> dict[str, object]:
        return {"backend_meta": "shared"}

    def native_inference_warnings(self) -> list[str]:
        return ["backend-warning"]

    def _infer_model(self, request: InferenceRequest, model_inputs: object) -> NativeModelCall:
        self.seen_model_inputs = model_inputs
        return NativeModelCall(
            raw_output={"actions": [[1.0, 2.0], [3.0, 4.0]]},
            metadata={"call_meta": "native-call"},
            warnings=["call-warning"],
        )


class _SpineProcessor:
    def to_model_inputs(self, observation: Observation) -> dict[str, object]:
        return {"prompt": observation.prompt}

    def to_harness_result(self, raw_output: object) -> InferenceResult:
        if not isinstance(raw_output, dict):
            raise TypeError("expected raw output mapping")
        return InferenceResult(
            action_chunk=ActionChunk(actions=raw_output["actions"]),
            backend_metadata={"processor": "spine"},
            warnings=["processor-warning"],
        )


def test_native_backend_resolves_assets_from_configured_cache_dir(tmp_path) -> None:
    registry = default_registry()
    cache_dir = tmp_path / "cache"
    checkpoint = cache_dir / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"cache_dir": str(cache_dir)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    assert backend.cache_dir() == cache_dir
    assert backend.runtime_info().metadata["cache_dir"] == str(cache_dir)
    assert backend.runtime_info().metadata["runtime_mode"] == "in_process"
    assert backend.runtime_info().metadata["runtime_loader"] == "fastwam_runtime_loader"
    assert backend.runtime_info().metadata["model_adapter"] == "fastwam_model"
    assert backend.resolve_required_asset("checkpoint") == checkpoint.resolve()


def test_native_backend_runtime_info_records_optimization_plan(tmp_path) -> None:
    registry = default_registry()
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {
        "name": "fastwam",
        "mode": "native",
        "config": {"upstream_dir": str(tmp_path / "FastWAM")},
    }
    manifest = manifest_from_dict(data)
    profiles = registry.build_optimization_profiles(manifest, ["action_chunk_scheduling"])
    backend = registry.create_backend(manifest, profiles)

    statuses = backend.apply_optimization_profiles(profiles)
    plan = backend.runtime_info().metadata["native_optimization_plan"]

    assert statuses == [
        {
            "name": "action_chunk_scheduling",
            "enabled": True,
            "params": {},
            "declared_supported": True,
            "scope": "simulator_eval",
            "target": "workload",
            "state": "applied",
            "hook": "action_chunk_contract",
        }
    ]
    assert plan == [
        {
            "name": "action_chunk_scheduling",
            "enabled": True,
            "params": {},
            "declared_supported": True,
            "scope": "simulator_eval",
            "target": "workload",
            "state": "applied",
            "hook": "action_chunk_contract",
        }
    ]


def test_native_backend_base_wraps_processor_model_call_with_trace_metadata() -> None:
    manifest = load_builtin_manifest("fake-open-loop")
    backend = NativeBackendBase(manifest, [], backend_label="Unit")
    processor = _UnitProcessor()

    result = backend.infer_with_processor(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=2,
            replan_steps=1,
        ),
        processor,
        lambda model_inputs: NativeModelCall(
            raw_output={"actions": [[1.0, 2.0]], "prompt": model_inputs["prompt"]},
            timing_key="server_ms",
            metadata={"adapter": "unit"},
            warnings=["native warning"],
        ),
        metadata={"checkpoint_path": "/tmp/checkpoint"},
        warnings=["backend warning"],
    )

    assert result.action_chunk.actions == [[1.0, 2.0]]
    assert set(result.timing) == {"preprocess_ms", "server_ms", "postprocess_ms", "total_ms"}
    assert result.backend_metadata == {
        "processor": "unit",
        "native_backend": True,
        "checkpoint_path": "/tmp/checkpoint",
        "adapter": "unit",
    }
    assert result.warnings == ["processor warning", "backend warning", "native warning"]


def test_native_backend_base_uses_model_adapter_by_default() -> None:
    manifest = load_builtin_manifest("fake-open-loop")
    backend = _AdapterOnlyBackend(manifest, [])

    result = backend.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=2,
            replan_steps=1,
        )
    )

    assert backend.model_adapter.seen_model_inputs == {"prompt": "open the drawer"}
    assert result.action_chunk.actions == [[5.0, 6.0]]
    assert set(result.timing) == {"preprocess_ms", "adapter_ms", "postprocess_ms", "total_ms"}
    assert result.backend_metadata == {
        "processor": "unit",
        "native_backend": True,
        "model_adapter": "unit_adapter",
        "adapter_phase": "inference",
        "call_meta": "adapter-call",
    }
    assert result.warnings == ["processor warning", "adapter warning", "call warning"]
    assert backend.runtime_info().metadata["model_adapter"] == "unit_adapter"


def test_cosmos_policy_model_adapter_calls_upstream_action_function() -> None:
    cosmos_utils = _CosmosUtils()
    cfg = SimpleNamespace(seed=123, randomize_seed=True, num_denoising_steps_action=4)
    adapter = CosmosPolicyModelAdapter(
        cfg=cfg,
        model="model",
        dataset_stats={"action": "stats"},
        cosmos_utils=cosmos_utils,
        checkpoint_path="/tmp/checkpoint",
        dataset_stats_path="/tmp/stats.json",
        text_embeddings_path="/tmp/text.pkl",
        error_cls=CosmosPolicyNativeBackendError,
    )

    output = adapter.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="open the drawer"),
            action_horizon=2,
            replan_steps=1,
            runtime_options={"seed": 9, "num_denoising_steps_action": 8},
        ),
        {"observation": "obs", "prompt": "prompt"},
    )

    assert output == {"actions": [[1.0, 2.0]], "value": [0.5]}
    assert cosmos_utils.call == {
        "cfg": cfg,
        "model": "model",
        "dataset_stats": {"action": "stats"},
        "observation": "obs",
        "prompt": "prompt",
        "seed": 9,
        "randomize_seed": True,
        "num_denoising_steps_action": 8,
        "generate_future_state_and_value_in_parallel": True,
    }
    assert adapter.inference_metadata() == {
        "model_adapter": "cosmos_policy_model",
        "checkpoint_path": "/tmp/checkpoint",
        "dataset_stats_path": "/tmp/stats.json",
        "text_embeddings_path": "/tmp/text.pkl",
    }

    adapter.close()

    with pytest.raises(CosmosPolicyNativeBackendError, match="adapter is not loaded"):
        adapter.require_ready()


def test_dreamzero_policy_server_adapter_reports_server_timing_and_resets() -> None:
    client = _DreamZeroClient()
    adapter = DreamZeroPolicyServerAdapter(
        client=client,
        server_metadata={"model": "dreamzero"},
        checkpoint_path="/tmp/checkpoint",
        error_cls=DreamZeroNativeBackendError,
    )

    call = adapter.infer(
        InferenceRequest(
            observation=Observation(images={}, prompt="pick"),
            action_horizon=1,
            replan_steps=1,
        ),
        {"payload": "obs"},
    )
    adapter.reset()

    assert adapter.model_timing_key() == "server_ms"
    assert call.timing_key == "server_ms"
    assert call.raw_output == {"actions": [[0.1, 0.2]], "score": 0.9}
    assert call.metadata == {"server_metadata": {"model": "dreamzero"}}
    assert client.infer_payloads == [{"payload": "obs"}]
    assert client.reset_payloads == [{}]
    assert adapter.inference_metadata() == {
        "model_adapter": "dreamzero_policy_server",
        "transport": "websocket",
        "server_metadata": {"model": "dreamzero"},
        "checkpoint_path": "/tmp/checkpoint",
    }

    adapter.close()

    assert client.closed is True
    with pytest.raises(DreamZeroNativeBackendError, match="adapter is not connected"):
        adapter.require_ready()


def test_cosmos_policy_native_backend_applies_parallel_profile_to_config(
    tmp_path,
) -> None:
    registry = default_registry()
    manifest = _native_manifest("cosmos-policy-libero", "cosmos_policy", tmp_path)
    profiles = registry.build_optimization_profiles(manifest, ["parallel_inference"])
    backend = registry.create_backend(manifest, profiles)
    backend.checkpoint_path = tmp_path / "checkpoint"
    backend.dataset_stats_path = tmp_path / "dataset_stats.json"
    backend.text_embeddings_path = tmp_path / "text_embeddings.pkl"

    cfg = backend._build_eval_config(_CosmosPolicyConfig)

    assert cfg.use_parallel_inference is True
    assert cfg.num_queries_best_of_n == 4


def test_cosmos_policy_native_backend_keeps_parallel_profile_off_by_default(
    tmp_path,
) -> None:
    registry = default_registry()
    manifest = _native_manifest("cosmos-policy-libero", "cosmos_policy", tmp_path)
    backend = registry.create_backend(manifest, [])
    backend.checkpoint_path = tmp_path / "checkpoint"
    backend.dataset_stats_path = tmp_path / "dataset_stats.json"
    backend.text_embeddings_path = tmp_path / "text_embeddings.pkl"

    cfg = backend._build_eval_config(_CosmosPolicyConfig)

    assert cfg.use_parallel_inference is False
    assert cfg.num_queries_best_of_n == 1


def test_cosmos_policy_native_backend_selects_no_noops_unnorm_key(tmp_path) -> None:
    registry = default_registry()
    manifest = _native_manifest("cosmos-policy-libero", "cosmos_policy", tmp_path)
    backend = registry.create_backend(manifest, [])
    cfg = _CosmosPolicyConfig()
    cfg.task_suite_name = "libero_10"
    model = SimpleNamespace(norm_stats={"libero_10_no_noops": {}})

    backend._check_unnorm_key(cfg, model)

    assert cfg.unnorm_key == "libero_10_no_noops"


def test_cosmos_policy_native_backend_allows_missing_model_norm_stats(tmp_path) -> None:
    registry = default_registry()
    manifest = _native_manifest("cosmos-policy-libero", "cosmos_policy", tmp_path)
    backend = registry.create_backend(manifest, [])
    cfg = _CosmosPolicyConfig()
    cfg.task_suite_name = "libero_10"
    model = SimpleNamespace(norm_stats={})

    backend._check_unnorm_key(cfg, model)

    assert getattr(cfg, "unnorm_key", None) is None


def test_dreamzero_native_backend_applies_dit_cache_profile_args(tmp_path) -> None:
    registry = default_registry()
    manifest = _native_manifest("dreamzero-droid-sim", "dreamzero", tmp_path)
    profiles = registry.build_optimization_profiles(manifest, ["dit_cache"])
    backend = registry.create_backend(manifest, profiles)

    assert backend._profile_server_args() == ["--enable-dit-cache"]


def test_default_registry_exposes_dreamzero_native_backend(tmp_path) -> None:
    registry = default_registry()
    manifest = _native_manifest("dreamzero-droid-sim", "dreamzero", tmp_path)

    backend = registry.create_backend(manifest, [])
    processor = registry.create_processor(manifest)

    info = backend.runtime_info()
    assert info.backend == "dreamzero"
    assert info.metadata["transport"] == "websocket"
    assert processor.modality_limits()["processor"] == "dreamzero_droid"
    smoke = processor.smoke_observation()
    assert {"right", "left", "wrist"} <= set(smoke.images)


def test_cosmos_policy_native_backend_fails_clearly_without_upstream_repo(tmp_path) -> None:
    registry = default_registry()
    manifest = _native_manifest("cosmos-policy-libero", "cosmos_policy", tmp_path)
    backend = registry.create_backend(manifest, [])

    with pytest.raises(CosmosPolicyNativeBackendError, match="Cosmos-Policy upstream repo not found"):
        backend.load()


def test_cosmos_policy_backend_load_binds_runtime_loader_bundle(tmp_path) -> None:
    registry = default_registry()
    repo = tmp_path / "cosmos-policy"
    _write_cosmos_required_paths(repo)
    data = load_builtin_manifest("cosmos-policy-libero").to_dict()
    cache = tmp_path / "cache"
    checkpoint = cache / "checkpoints" / "cosmos_policy" / "checkpoint.pt"
    dataset_stats = cache / "checkpoints" / "cosmos_policy" / "dataset_stats.json"
    text_embeddings = cache / "checkpoints" / "cosmos_policy" / "text_embeddings.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    dataset_stats.write_text("{}", encoding="utf-8")
    text_embeddings.write_bytes(b"text embeddings")
    data["assets"]["checkpoint"]["local_path"] = str(checkpoint)
    data["assets"]["dataset_stats"]["local_path"] = str(dataset_stats)
    data["assets"]["text_embeddings"]["local_path"] = str(text_embeddings)
    data["backend"] = {
        "name": "cosmos_policy",
        "mode": "native",
        "config": {"upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    runtime_loader = _FakeCosmosRuntimeLoader()
    backend.runtime_loader = runtime_loader

    backend.load()

    assert runtime_loader.loaded is True
    assert backend.loaded is True
    assert backend.upstream_repo == repo.resolve()
    assert backend.model is runtime_loader.bundle.model
    assert backend.cfg is runtime_loader.bundle.cfg
    assert backend.dataset_stats is runtime_loader.bundle.dataset_stats
    assert backend.cosmos_utils is runtime_loader.bundle.cosmos_utils
    assert isinstance(backend.model_adapter, CosmosPolicyModelAdapter)


def test_dreamzero_backend_load_binds_runtime_loader_bundle(tmp_path) -> None:
    registry = default_registry()
    repo = tmp_path / "dreamzero"
    server_module = repo / "eval_utils" / "serve_dreamzero_wan22.py"
    server_module.parent.mkdir(parents=True)
    server_module.write_text("# smoke\n", encoding="utf-8")
    data = load_builtin_manifest("dreamzero-droid-sim").to_dict()
    checkpoint = tmp_path / "cache" / "dreamzero" / "checkpoint"
    checkpoint.mkdir(parents=True)
    data["assets"]["checkpoint"]["local_path"] = str(checkpoint)
    data["backend"] = {
        "name": "dreamzero",
        "mode": "native",
        "config": {"upstream_dir": str(repo)},
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    runtime_loader = _FakeDreamZeroRuntimeLoader()
    backend.runtime_loader = runtime_loader

    backend.load()

    assert runtime_loader.repo == repo.resolve()
    assert runtime_loader.checkpoint_path == checkpoint.resolve()
    assert backend.loaded is True
    assert backend.upstream_repo == repo.resolve()
    assert backend.client is runtime_loader.bundle.client
    assert backend.server_process is runtime_loader.bundle.server_process
    assert backend.server_metadata == {"model": "dreamzero", "runtime": "test"}
    assert isinstance(backend.model_adapter, DreamZeroPolicyServerAdapter)
    assert backend.runtime_info().metadata["server_started_by_harness"] is True

    backend.close()

    assert runtime_loader.bundle.server_process.terminated is True
    assert runtime_loader.bundle.client.closed is True


def test_dreamzero_native_backend_fails_clearly_without_upstream_repo(tmp_path) -> None:
    registry = default_registry()
    manifest = _native_manifest("dreamzero-droid-sim", "dreamzero", tmp_path)
    backend = registry.create_backend(manifest, [])

    with pytest.raises(DreamZeroNativeBackendError, match="DreamZero upstream repo not found"):
        backend.load()


def test_dreamzero_native_backend_renders_runtime_path_templates(tmp_path) -> None:
    registry = default_registry()
    manifest = _native_manifest("dreamzero-droid-sim", "dreamzero", tmp_path)
    backend = registry.create_backend(manifest, [])
    repo = tmp_path / "dreamzero"

    rendered = backend._render_runtime_template(
        "{upstream_dir}/.venv/bin/python:{cache_dir}/models",
        repo=repo,
    )

    assert rendered == f"{repo}/.venv/bin/python:{backend.cache_dir()}/models"


def test_dreamzero_start_server_captures_stdout_to_log(tmp_path, monkeypatch) -> None:
    for env_name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TORCH_COMPILE_DISABLE"):
        monkeypatch.delenv(env_name, raising=False)
    registry = default_registry()
    repo = tmp_path / "dreamzero"
    server_module = repo / "eval_utils" / "serve_dreamzero_wan22.py"
    server_module.parent.mkdir(parents=True)
    server_module.write_text("# smoke\n", encoding="utf-8")
    checkpoint = tmp_path / "cache" / "dreamzero" / "checkpoint"
    checkpoint.mkdir(parents=True)
    server_log = tmp_path / "logs" / "policy-server.log"
    data = load_builtin_manifest("dreamzero-droid-sim").to_dict()
    data["assets"]["checkpoint"]["local_path"] = str(checkpoint)
    data["backend"] = {
        "name": "dreamzero",
        "mode": "native",
        "config": {
            "cache_dir": str(tmp_path / "cache"),
            "upstream_dir": str(repo),
            "server_log_path": str(server_log),
            "master_port": "23456",
        },
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])
    seen: dict[str, object] = {}

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        seen["stderr"] = kwargs["stderr"]
        seen["env"] = kwargs["env"]
        kwargs["stdout"].write("server boot log\n")
        kwargs["stdout"].flush()
        return _DreamZeroProcess()

    monkeypatch.setattr("wam_harness.backends.dreamzero.subprocess.Popen", fake_popen)

    process = backend._start_server(repo, checkpoint)

    assert seen["argv"] == [
        f"{repo}/.venv/bin/python",
        "-m",
        "wam_harness.compat.dreamzero_eval.serve_dreamzero_no_compile",
        "--model_path",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        "6000",
    ]
    assert seen["stderr"] is subprocess.STDOUT
    assert backend.server_log_path == server_log
    assert seen["env"]["HF_HOME"] == str(tmp_path / "cache" / "huggingface")
    assert seen["env"]["HF_HUB_CACHE"] == str(tmp_path / "cache" / "huggingface" / "hub")
    assert seen["env"]["HF_XET_CACHE"] == str(tmp_path / "cache" / "huggingface" / "xet")
    assert seen["env"]["TORCH_COMPILE_DISABLE"] == "1"
    assert seen["env"]["MASTER_ADDR"] == "127.0.0.1"
    assert seen["env"]["MASTER_PORT"] == "23456"
    assert seen["env"]["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "cache" / "torchinductor")
    assert seen["env"]["TRITON_CACHE_DIR"] == str(tmp_path / "cache" / "triton")
    assert seen["env"]["WAM_CACHE_DIR"] == str(tmp_path / "cache")
    assert seen["env"]["WAM_DREAMZERO_SKIP_WEIGHT_INIT"] == "1"
    assert "DreamZero policy server command" in server_log.read_text(encoding="utf-8")
    assert "TORCH_COMPILE_DISABLE=1" in server_log.read_text(encoding="utf-8")
    assert "MASTER_PORT=23456" in server_log.read_text(encoding="utf-8")
    assert "WAM_DREAMZERO_SKIP_WEIGHT_INIT=1" in server_log.read_text(encoding="utf-8")
    assert "server boot log" in server_log.read_text(encoding="utf-8")

    backend._stop_server_process(process)

    assert backend.server_log_handle is None


def test_dreamzero_uses_wamfile_server_startup_default(tmp_path) -> None:
    registry = default_registry()
    manifest = _native_manifest("dreamzero-droid-sim", "dreamzero", tmp_path)
    backend = registry.create_backend(manifest, [])

    assert backend._server_startup_seconds() == 1200


def test_dreamzero_backend_override_controls_server_startup_default(tmp_path) -> None:
    registry = default_registry()
    data = load_builtin_manifest("dreamzero-droid-sim").to_dict()
    data["backend"] = {
        "name": "dreamzero",
        "mode": "native",
        "config": {
            "upstream_dir": str(tmp_path / "dreamzero"),
            "server_startup_seconds": "42",
        },
    }
    manifest = manifest_from_dict(data)
    backend = registry.create_backend(manifest, [])

    assert backend._server_startup_seconds() == 42


class _UnitProcessor:
    def to_model_inputs(self, observation: Observation) -> dict[str, object]:
        return {"prompt": observation.prompt}

    def to_harness_result(self, raw_output: object) -> InferenceResult:
        actions = raw_output["actions"] if isinstance(raw_output, dict) else raw_output
        return InferenceResult(
            action_chunk=ActionChunk(actions=actions),
            warnings=["processor warning"],
            backend_metadata={"processor": "unit"},
        )


class _AdapterOnlyBackend(NativeBackendBase):
    def __init__(self, manifest, profiles) -> None:
        super().__init__(manifest, profiles, backend_label="AdapterOnly")
        self.processor = _UnitProcessor()
        self.model_adapter = _UnitAdapter()
        self.loaded = True
        self.warmed = True


class _UnitAdapter(NativeModelAdapter):
    name = "unit_adapter"

    def __init__(self) -> None:
        self.seen_model_inputs = None

    def model_timing_key(self) -> str:
        return "adapter_ms"

    def runtime_metadata(self) -> dict[str, object]:
        return {"model_adapter": self.name, "adapter_phase": "runtime"}

    def inference_metadata(self) -> dict[str, object]:
        return {"model_adapter": self.name, "adapter_phase": "inference"}

    def inference_warnings(self) -> list[str]:
        return ["adapter warning"]

    def infer(
        self,
        request: InferenceRequest,
        model_inputs: object,
    ) -> NativeModelCall:
        self.seen_model_inputs = model_inputs
        return NativeModelCall(
            raw_output={"actions": [[5.0, 6.0]], "horizon": request.action_horizon},
            timing_key="adapter_ms",
            metadata={"call_meta": "adapter-call"},
            warnings=["call warning"],
        )


class _CosmosUtils:
    def __init__(self) -> None:
        self.call = None

    def get_action(
        self,
        cfg,
        model,
        dataset_stats,
        observation,
        prompt,
        **kwargs,
    ):
        self.call = {
            "cfg": cfg,
            "model": model,
            "dataset_stats": dataset_stats,
            "observation": observation,
            "prompt": prompt,
            **kwargs,
        }
        return {"actions": [[1.0, 2.0]], "value": [0.5]}


class _DreamZeroClient:
    def __init__(self) -> None:
        self.infer_payloads = []
        self.reset_payloads = []
        self.closed = False

    def infer(self, payload):
        self.infer_payloads.append(payload)
        return {"actions": [[0.1, 0.2]], "score": 0.9}

    def reset(self, payload):
        self.reset_payloads.append(payload)

    def close(self):
        self.closed = True


class _CosmosPolicyConfig:
    suite = None
    config = None
    ckpt_path = None
    config_file = None
    use_wrist_image = None
    use_proprio = None
    normalize_proprio = None
    unnormalize_actions = None
    dataset_stats_path = None
    t5_text_embeddings_path = None
    trained_with_image_aug = None
    chunk_size = None
    num_open_loop_steps = None
    task_suite_name = None
    randomize_seed = None
    seed = None
    use_variance_scale = None
    deterministic = None
    ar_future_prediction = None
    ar_value_prediction = None
    use_jpeg_compression = None
    flip_images = None
    num_denoising_steps_action = None
    num_denoising_steps_future_state = None
    num_denoising_steps_value = None
    use_parallel_inference = None
    num_queries_best_of_n = None


class _FakeCosmosRuntimeLoader:
    def __init__(self) -> None:
        self.loaded = False
        self.bundle = CosmosPolicyRuntimeBundle(
            cfg=_CosmosPolicyConfig(),
            model=_CosmosModel(),
            cosmos_config=SimpleNamespace(
                dataloader_train=SimpleNamespace(dataset=SimpleNamespace(chunk_size=16))
            ),
            dataset_stats={"actions_min": [0.0], "actions_max": [1.0]},
            cosmos_utils=_CosmosUtils(),
        )

    def load(self):
        self.loaded = True
        return self.bundle


class _CosmosModel:
    norm_stats = {"libero_10": {}}


class _FakeDreamZeroRuntimeLoader:
    def __init__(self) -> None:
        self.repo = None
        self.checkpoint_path = None
        self.bundle = DreamZeroPolicyServerRuntimeBundle(
            client=_DreamZeroClient(),
            server_process=_DreamZeroProcess(),
            server_metadata={"model": "dreamzero", "runtime": "test"},
        )

    def load(self, *, repo, checkpoint_path):
        self.repo = repo
        self.checkpoint_path = checkpoint_path
        return self.bundle


class _DreamZeroProcess:
    returncode = None

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self) -> None:
        self.killed = True
