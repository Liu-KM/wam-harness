import json

import pytest

from eazywam.backends.fastwam import FastWAMBackend
from eazywam.cli import build_parser, main
from eazywam.core.action_contract import ActionContractError
from eazywam.core.manifest import load_builtin_manifest
from eazywam.backends.native_support.smoke import (
    NativeSmokeRunner,
    NativeSmokeRunnerError,
    native_smoke_manifest,
)
from eazywam.backends.native_support.runtime import native_runtime_resolver
from eazywam.core.preflight import PreflightError
from eazywam.core.registry import Registry, default_registry
from eazywam.core.types import (
    ActionChunk,
    InferenceRequest,
    InferenceResult,
    Manifest,
    Observation,
    OptimizationProfile,
    RuntimeInfo,
)


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_fastwam_required_paths(repo) -> None:
    for relative in [
        "configs/sim_libero.yaml",
        "configs/train.yaml",
        "configs/task/libero_uncond_2cam224_1e-4.yaml",
        "configs/data/libero_2cam.yaml",
        "configs/model/fastwam.yaml",
    ]:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# smoke\n", encoding="utf-8")


def test_native_smoke_manifest_maps_reference_entry_to_native_backend(tmp_path) -> None:
    reference = load_builtin_manifest("cosmos-policy-libero")

    manifest = native_smoke_manifest(
        reference,
        upstream_dir=tmp_path / "repo",
        cache_dir=tmp_path / "cache",
        backend_overrides={"foo": "bar"},
    )

    assert reference.backend_name == "external_eval"
    assert reference.backend["config"]["native_backend"] == "cosmos_policy"
    assert manifest.backend_name == "cosmos_policy"
    assert manifest.backend["mode"] == "native_smoke"
    assert manifest.backend["config"]["upstream_dir"] == str(tmp_path / "repo")
    assert manifest.backend["config"]["cache_dir"] == str(tmp_path / "cache")
    assert manifest.backend["config"]["foo"] == "bar"
    assert "native_backend" not in manifest.backend["config"]
    assert manifest.workload_name == "native_smoke"


def test_native_smoke_rejects_model_without_native_backend() -> None:
    manifest = load_builtin_manifest("fake-open-loop")

    with pytest.raises(NativeSmokeRunnerError, match="backend.config.native_backend"):
        native_smoke_manifest(manifest)


def test_processors_provide_native_smoke_observations() -> None:
    registry = default_registry()
    fastwam = registry.create_processor(
        native_smoke_manifest(load_builtin_manifest("fastwam-libero"))
    ).smoke_observation()
    fastwam_robotwin = registry.create_processor(
        native_smoke_manifest(load_builtin_manifest("fastwam-robotwin"))
    ).smoke_observation()
    cosmos = registry.create_processor(
        native_smoke_manifest(load_builtin_manifest("cosmos-policy-libero"))
    ).smoke_observation()
    dreamzero = registry.create_processor(
        native_smoke_manifest(load_builtin_manifest("dreamzero-droid-sim"))
    ).smoke_observation()

    assert {"primary", "wrist"} <= set(fastwam.images)
    assert {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"} <= set(fastwam.state)
    assert {"head", "left_wrist", "right_wrist"} <= set(fastwam_robotwin.images)
    assert "joint_action" in fastwam_robotwin.state
    assert {"primary", "wrist"} <= set(cosmos.images)
    assert {"robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat"} <= set(cosmos.state)
    assert {"right", "left", "wrist"} <= set(dreamzero.images)
    assert {"joint_position", "gripper_position"} <= set(dreamzero.state)


def test_native_smoke_runner_fails_clearly_without_upstream_repo(tmp_path) -> None:
    runner = NativeSmokeRunner()

    with pytest.raises(PreflightError, match="fastwam preflight is blocked"):
        runner.run("fastwam-libero", trace_dir=tmp_path, upstream_dir=tmp_path / "missing")

    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = read_events(trace_paths[0])
    event_names = [event["event"] for event in events]
    assert event_names[0] == "run_start"
    assert "optimization_profile_status" in event_names
    assert "runtime_contract" in event_names
    assert "preflight" in event_names
    assert event_names[-2:] == ["error", "run_end"]
    contract = [event for event in events if event["event"] == "runtime_contract"][0]
    readiness = [event for event in events if event["event"] == "preflight"][0]
    assert contract["mode"] == "native_smoke"
    assert contract["backend"] == "fastwam"
    assert contract["runtime_mode"] == "in_process"
    assert contract["runtime_loader"] == "fastwam_runtime_loader"
    assert contract["model_adapter"] == "fastwam_model"
    assert contract["processor_modality"]["processor"] == "fastwam_libero"
    assert readiness["status"] == "blocked"
    assert readiness["runtime_mode"] == "in_process"
    assert readiness["runtime_loader"] == "fastwam_runtime_loader"
    assert readiness["model_adapter"] == "fastwam_model"
    assert readiness["upstream"]["status"] == "missing"
    assert "checkpoint" in readiness["missing_required_assets"]
    error = [event for event in events if event["event"] == "error"][0]
    assert error["stage"] == "preflight"
    assert error["recoverable"] is True
    assert error["trace_path"] == str(trace_paths[0])


def test_cli_native_smoke_routes_to_native_backend(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "native-smoke",
            "fastwam-libero",
            "--trace-dir",
            str(tmp_path),
            "--upstream-dir",
            str(tmp_path / "missing"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: fastwam preflight is blocked" in captured.err
    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    assert f"trace: {trace_paths[0]}" in captured.err


def test_native_smoke_require_ready_rejects_runtime_asset_warning(tmp_path) -> None:
    registry = default_registry()

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> FastWAMBackend:
        backend = FastWAMBackend(manifest, profiles)
        backend.required_python_modules = ()
        return backend

    registry.register_backend("fastwam", factory)
    repo = tmp_path / "FastWAM"
    write_fastwam_required_paths(repo)
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

    with pytest.raises(PreflightError, match="fastwam preflight is warning"):
        NativeSmokeRunner(registry=registry).run(
            "fastwam-libero",
            trace_dir=tmp_path / "runs",
            upstream_dir=repo,
            cache_dir=cache_dir,
            require_ready=True,
        )

    trace_paths = list((tmp_path / "runs").glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = read_events(trace_paths[0])
    contract = [event for event in events if event["event"] == "runtime_contract"][0]
    readiness = [event for event in events if event["event"] == "preflight"][0]
    assert contract["event"] == "runtime_contract"
    assert readiness["event"] == "preflight"
    assert readiness["status"] == "warning"
    assert readiness["runtime_mode"] == "in_process"
    assert readiness["runtime_loader"] == "fastwam_runtime_loader"
    assert readiness["missing_required_assets"] == []
    assert readiness["missing_runtime_assets"] == [
        "wan22_vae",
        "wan22_t5_encoder",
        "wan21_tokenizer_spiece",
        "wan21_tokenizer_json",
        "wan21_tokenizer_config",
        "wan21_special_tokens_map",
    ]
    error = [event for event in events if event["event"] == "error"][0]
    assert error["stage"] == "preflight"


def test_native_smoke_rejects_action_contract_mismatch(tmp_path) -> None:
    registry = Registry()
    registry.register_runtime_resolver(native_runtime_resolver)
    registry.register_backend(
        "fastwam",
        lambda manifest, profiles: _BadActionShapeBackend(manifest, profiles),
    )
    registry.register_processor("fastwam_libero", lambda manifest: _SmokeProcessor())

    with pytest.raises(ActionContractError, match="action horizon mismatch"):
        NativeSmokeRunner(registry=registry).run("fastwam-libero", trace_dir=tmp_path)

    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = read_events(trace_paths[0])
    assert events[-2]["event"] == "error"
    assert events[-2]["stage"] == "action_contract"
    assert events[-2]["error_type"] == "ActionContractError"
    assert events[-1]["event"] == "run_end"
    assert events[-1]["status"] == "error"


def test_native_smoke_traces_future_and_value_outputs(tmp_path) -> None:
    registry = Registry()
    registry.register_runtime_resolver(native_runtime_resolver)
    registry.register_backend(
        "fastwam",
        lambda manifest, profiles: _FutureValueNativeBackend(manifest, profiles),
    )
    registry.register_processor("fastwam_libero", lambda manifest: _SmokeProcessor())

    summary = NativeSmokeRunner(registry=registry).run(
        "fastwam-libero",
        trace_dir=tmp_path,
        action_horizon=32,
        replan_steps=10,
    )

    events = read_events(summary.trace_path)
    inference_end = [event for event in events if event["event"] == "inference_end"][0]
    assert inference_end["future_frames"] == {
        "present": True,
        "count": 4,
        "artifact_path": "future/native-smoke.json",
    }
    assert inference_end["value"] == {"score": 0.5}


def test_cli_native_smoke_accepts_cache_dir_and_backend_overrides() -> None:
    args = build_parser().parse_args(
        [
            "native-smoke",
            "fastwam-libero",
            "--cache-dir",
            "/cache/wam",
            "--upstream-dir",
            "/repo/FastWAM",
            "--backend-set",
            "task=libero",
            "--require-ready",
        ]
    )

    assert args.command == "native-smoke"
    assert args.cache_dir == "/cache/wam"
    assert args.upstream_dir == "/repo/FastWAM"
    assert args.backend_set == ["task=libero"]
    assert args.require_ready is True


def test_cli_native_smoke_help_lists_command(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["native-smoke", "--help"])

    captured = capsys.readouterr()
    assert "Run one synthetic observation through a native backend migration path" in captured.out


class _BadActionShapeBackend:
    def __init__(self, manifest: Manifest, profiles: list[OptimizationProfile]) -> None:
        self.manifest = manifest
        self.profiles = profiles

    def load(self) -> None:
        return

    def warmup(self) -> None:
        return

    def reset(self) -> None:
        return

    def infer(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(action_chunk=ActionChunk(actions=[[0.0] * 7]))

    def runtime_info(self) -> RuntimeInfo:
        return RuntimeInfo(
            manifest_id=self.manifest.id,
            model_name=self.manifest.display_name,
            backend=self.manifest.backend_name,
            processor=self.manifest.processor_name,
            source_repo=self.manifest.source_repo,
            mode=str(self.manifest.backend.get("mode", "native_smoke")),
            device="cpu",
            dtype="fp32",
            optimization_profiles=self.profiles,
        )

    def close(self) -> None:
        return


class _FutureValueNativeBackend(_BadActionShapeBackend):
    def infer(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            action_chunk=ActionChunk(actions=[[0.0] * 7 for _ in range(request.action_horizon)]),
            future_frames={
                "present": True,
                "count": 4,
                "artifact_path": "future/native-smoke.json",
            },
            value={"score": 0.5},
        )


class _SmokeProcessor:
    def to_model_inputs(self, observation: Observation) -> object:
        return observation

    def to_harness_result(self, raw_output: object) -> InferenceResult:
        if not isinstance(raw_output, InferenceResult):
            raise TypeError("expected InferenceResult")
        return raw_output

    def modality_limits(self) -> dict[str, object]:
        return {}

    def smoke_observation(self) -> Observation:
        return Observation(images={"primary": [[[0, 0, 0]]]}, prompt="smoke")
