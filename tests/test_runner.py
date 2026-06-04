import json

import pytest

from wam_harness.backends.native_support.runtime import native_runtime_resolver
from wam_harness.core.preflight import PreflightError
from wam_harness.core.registry import Registry
from wam_harness.core.runner import RunInputRequiredError, Runner
from wam_harness.core.types import (
    ActionChunk,
    InferenceRequest,
    InferenceResult,
    Manifest,
    Observation,
    OptimizationProfile,
    RuntimeInfo,
)
from wam_harness.processors.passthrough import PassthroughProcessor


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_runner_writes_trace_with_optimization_profile(tmp_path) -> None:
    summary = Runner().run(
        "fake-open-loop",
        enabled_opts=["fake_cache"],
        trace_dir=tmp_path,
    )

    events = read_events(summary.trace_path)
    event_names = [event["event"] for event in events]

    assert summary.steps == 6
    assert summary.model_calls == 3
    assert event_names[0] == "run_start"
    assert "backend_load" in event_names
    assert "inference_end" in event_names
    assert event_names[-1] == "run_end"
    assert events[0]["optimization_profiles"][0]["name"] == "fake_cache"
    optimization = [
        event for event in events if event["event"] == "optimization_profile_status"
    ][0]
    assert optimization["profiles"][0]["name"] == "fake_cache"
    assert optimization["profiles"][0]["state"] == "applied"
    assert optimization["profiles"][0]["hook"] == "fake_backend_latency_model"
    inference_events = [event for event in events if event["event"] == "inference_end"]
    assert inference_events[0]["action_summary"]["shape"] == [3, 4]
    assert inference_events[0]["action_summary"]["finite"] is True
    assert events[-1]["status"] == "ok"


def test_runner_replans_at_requested_interval(tmp_path) -> None:
    summary = Runner().run(
        "fake-open-loop",
        trace_dir=tmp_path,
        episode_length=5,
        action_horizon=3,
        replan_steps=2,
    )

    events = read_events(summary.trace_path)
    replan_events = [event for event in events if event["event"] == "replan_start"]

    assert len(replan_events) == 3


def test_runner_traces_future_and_value_outputs(tmp_path) -> None:
    registry = Registry()
    registry.register_backend(
        "fake",
        lambda manifest, profiles: FutureValueBackend(manifest, profiles),
    )
    registry.register_processor("passthrough", PassthroughProcessor.from_manifest)
    registry.register_workload("open_loop", lambda manifest: OneStepWorkload())

    summary = Runner(registry=registry).run(
        "fake-open-loop",
        trace_dir=tmp_path,
        episode_length=1,
        action_horizon=1,
        replan_steps=1,
    )

    events = read_events(summary.trace_path)
    inference_end = [event for event in events if event["event"] == "inference_end"][0]
    assert inference_end["future_frames"] == {
        "present": True,
        "count": 2,
        "artifact_path": "future/frames.json",
    }
    assert inference_end["value"] == {"score": 0.75}


def test_runner_invocation_attaches_registry_processor(tmp_path) -> None:
    registry = Registry()
    created_backends: list[ProcessorAttachBackend] = []
    created_processors: list[PassthroughProcessor] = []

    def backend_factory(
        manifest: Manifest,
        profiles: list[OptimizationProfile],
    ) -> ProcessorAttachBackend:
        backend = ProcessorAttachBackend(manifest, profiles)
        created_backends.append(backend)
        return backend

    def processor_factory(manifest: Manifest) -> PassthroughProcessor:
        processor = PassthroughProcessor.from_manifest(manifest)
        created_processors.append(processor)
        return processor

    registry.register_backend("fake", backend_factory)
    registry.register_processor("passthrough", processor_factory)
    registry.register_workload("open_loop", lambda manifest: OneStepWorkload())

    Runner(registry=registry).run(
        "fake-open-loop",
        trace_dir=tmp_path,
        episode_length=1,
        action_horizon=1,
        replan_steps=1,
    )

    assert created_backends[0].processor is created_processors[0]


class NativeRunBackend:
    def __init__(self, manifest: Manifest, profiles: list[OptimizationProfile]) -> None:
        self.manifest = manifest
        self.profiles = profiles
        self.last_request: InferenceRequest | None = None
        self.closed = False

    def load(self) -> None:
        return

    def warmup(self) -> None:
        return

    def reset(self) -> None:
        return

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.last_request = request
        action_dim = int(self.manifest.processor.get("action", {}).get("dim") or 1)
        return InferenceResult(
            action_chunk=ActionChunk(
                actions=[
                    [float(col) for col in range(action_dim)]
                    for _ in range(request.action_horizon)
                ]
            )
        )

    def runtime_info(self) -> RuntimeInfo:
        return RuntimeInfo(
            manifest_id=self.manifest.id,
            model_name=self.manifest.display_name,
            backend=self.manifest.backend_name,
            processor=self.manifest.processor_name,
            source_repo=self.manifest.source_repo,
            mode=str(self.manifest.backend.get("mode", "native_run")),
            device="cpu",
            dtype="fp32",
            optimization_profiles=self.profiles,
        )

    def runtime_contract(self, *, processor: object | None = None) -> dict[str, object]:
        return {
            "backend": self.manifest.backend_name,
            "processor": self.manifest.processor_name,
            "workload": self.manifest.workload_name,
            "mode": str(self.manifest.backend.get("mode", "run")),
            "model_adapter": self.native_model_adapter_name(),
            "supported_optimizations": self.manifest.supported_optimizations,
            "optimization_profile_status": [
                {
                    "name": profile.name,
                    "enabled": profile.enabled,
                    "params": dict(profile.params),
                    "declared_supported": profile.name
                    in self.manifest.supported_optimizations,
                    "scope": "simulator_eval",
                    "target": "workload",
                    "state": "requested",
                }
                for profile in self.profiles
            ],
        }

    def action_contract_enabled(self) -> bool:
        return True

    def native_model_adapter_name(self) -> str:
        return "test_native_run_adapter"

    def close(self) -> None:
        self.closed = True


class NativeRunProcessor:
    def to_model_inputs(self, observation: Observation) -> object:
        return observation

    def to_harness_result(self, raw_output: object) -> InferenceResult:
        if not isinstance(raw_output, InferenceResult):
            raise TypeError("expected InferenceResult")
        return raw_output

    def modality_limits(self) -> dict[str, object]:
        return {}

    def smoke_observation(self) -> Observation:
        return Observation(
            images={"primary": [[[0, 0, 0]]], "wrist": [[[1, 1, 1]]]},
            state={"proprio": [0.0]},
            prompt="native run smoke",
            session={"session_id": "processor-smoke"},
        )


class FutureValueBackend:
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
        return InferenceResult(
            action_chunk=ActionChunk(actions=[[0.0] * 4]),
            future_frames={
                "present": True,
                "count": 2,
                "artifact_path": "future/frames.json",
            },
            value={"score": 0.75},
        )

    def runtime_info(self) -> RuntimeInfo:
        return RuntimeInfo(
            manifest_id=self.manifest.id,
            model_name=self.manifest.display_name,
            backend=self.manifest.backend_name,
            processor=self.manifest.processor_name,
            source_repo=self.manifest.source_repo,
            mode=str(self.manifest.backend.get("mode", "fake")),
            device="cpu",
            dtype="fp32",
            optimization_profiles=self.profiles,
        )

    def close(self) -> None:
        return


class ProcessorAttachBackend(FutureValueBackend):
    def __init__(self, manifest: Manifest, profiles: list[OptimizationProfile]) -> None:
        super().__init__(manifest, profiles)
        self.processor: object | None = None

    def attach_processor(self, processor: object) -> None:
        self.processor = processor


class OneStepWorkload:
    episode_id = 0
    step_id = 0
    steps_since_replan = 0
    episode_length = 1

    @property
    def done(self) -> bool:
        return self.step_id >= self.episode_length

    def reset(self) -> None:
        self.step_id = 0
        self.steps_since_replan = 0

    def observation(self) -> Observation:
        return Observation(images={"primary": [[[0, 0, 0]]]}, prompt="smoke")

    def mark_replan(self) -> None:
        self.steps_since_replan = 0

    def step(self, action: list[float]) -> None:
        self.step_id += 1
        self.steps_since_replan += 1


def test_runner_requires_input_for_native_reference_entry(tmp_path) -> None:
    with pytest.raises(RunInputRequiredError, match="needs an observation input"):
        Runner().run(
            "fastwam-libero",
            trace_dir=tmp_path,
            upstream_dir=tmp_path / "FastWAM",
            cache_dir=tmp_path / "cache",
        )


def test_runner_maps_reference_entry_to_native_input_observation(tmp_path) -> None:
    registry = Registry()
    registry.register_runtime_resolver(native_runtime_resolver)
    created: list[NativeRunBackend] = []

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> NativeRunBackend:
        backend = NativeRunBackend(manifest, profiles)
        created.append(backend)
        return backend

    registry.register_backend("fastwam", factory)
    registry.register_processor("fastwam_libero", lambda manifest: NativeRunProcessor())

    summary = Runner(registry=registry).run(
        "fastwam-libero",
        enabled_opts=["action_chunk_scheduling"],
        trace_dir=tmp_path,
        upstream_dir=tmp_path / "FastWAM",
        cache_dir=tmp_path / "cache",
        observation=Observation(
            images={"primary": [[[9, 9, 9]]], "wrist": [[[8, 8, 8]]]},
            state={"proprio": [1.0]},
            prompt="external observation",
            session={"session_id": "input"},
        ),
    )

    assert summary.steps == 1
    assert summary.model_calls == 1
    assert summary.result is not None
    assert summary.result.action_chunk.horizon == 32
    assert created[0].closed is True
    assert created[0].manifest.backend_name == "fastwam"
    assert created[0].manifest.backend["mode"] == "run"
    assert created[0].manifest.backend["config"]["upstream_dir"] == str(tmp_path / "FastWAM")
    assert created[0].manifest.backend["config"]["cache_dir"] == str(tmp_path / "cache")
    assert created[0].manifest.workload_name == "single_observation"
    assert created[0].last_request is not None
    assert created[0].last_request.observation.prompt == "external observation"
    assert created[0].last_request.action_horizon == 32
    assert created[0].last_request.replan_steps == 10

    events = read_events(summary.trace_path)
    event_names = [event["event"] for event in events]
    assert "backend_load_start" in event_names
    assert events[0]["mode"] == "run"
    assert events[0]["synthetic_observation"] is False
    contract = [event for event in events if event["event"] == "runtime_contract"][0]
    assert contract["mode"] == "run"
    assert contract["backend"] == "fastwam"
    assert contract["processor"] == "fastwam_libero"
    assert contract["workload"] == "single_observation"
    assert contract["model_adapter"] == "test_native_run_adapter"
    assert contract["supported_optimizations"] == ["action_chunk_scheduling"]
    assert contract["optimization_profile_status"] == [
        {
            "name": "action_chunk_scheduling",
            "enabled": True,
            "params": {},
            "declared_supported": True,
            "scope": "simulator_eval",
            "target": "workload",
            "state": "requested",
        }
    ]
    replan = [event for event in events if event["event"] == "replan_start"][0]
    assert replan["observation_summary"]["image_keys"] == ["primary", "wrist"]
    assert replan["observation_summary"]["prompt"] == "external observation"
    assert "image_shapes" not in replan


def test_runner_traces_preflight_before_backend_load_failure(tmp_path) -> None:
    with pytest.raises(PreflightError, match="fastwam preflight is blocked"):
        Runner().run(
            "fastwam-libero",
            trace_dir=tmp_path,
            upstream_dir=tmp_path / "missing-fastwam",
            observation=Observation(images={"primary": []}, prompt="smoke"),
        )

    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = read_events(trace_paths[0])
    names = [event["event"] for event in events]
    assert names[:4] == ["run_start", "runtime_contract", "preflight", "error"]
    contract = events[1]
    readiness = events[2]
    assert contract["backend"] == "fastwam"
    assert contract["mode"] == "run"
    assert contract["runtime_mode"] == "in_process"
    assert contract["runtime_loader"] == "fastwam_runtime_loader"
    assert contract["model_adapter"] == "fastwam_model"
    assert readiness["status"] == "blocked"
    assert readiness["backend"] == "fastwam"
    assert readiness["runtime_mode"] == "in_process"
    assert readiness["runtime_loader"] == "fastwam_runtime_loader"
    assert readiness["model_adapter"] == "fastwam_model"
    assert readiness["upstream"]["status"] == "missing"
    assert events[3]["stage"] == "preflight"
    assert events[3]["recoverable"] is True
