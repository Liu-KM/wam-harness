from __future__ import annotations

import json

import pytest

from wam_harness.core.native_readiness import NativePreflightError
from wam_harness.core.registry import Registry
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
from wam_harness.serve import ServeApp, _observation_from_payload


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CapturingBackend:
    def __init__(
        self,
        manifest: Manifest,
        profiles: list[OptimizationProfile],
        *,
        contract_shape: bool = False,
        future_value: bool = False,
    ) -> None:
        self.manifest = manifest
        self.profiles = profiles
        self.last_request: InferenceRequest | None = None
        self.contract_shape = contract_shape
        self.future_value = future_value

    def load(self) -> None:
        return

    def warmup(self) -> None:
        return

    def reset(self) -> None:
        return

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.last_request = request
        future_frames = None
        value = None
        if self.future_value:
            future_frames = {
                "present": True,
                "count": 3,
                "artifact_path": "future/serve.json",
            }
            value = {"score": 0.25}
        if self.contract_shape:
            action_dim = int(self.manifest.processor.get("action", {}).get("dim") or 1)
            return InferenceResult(
                action_chunk=ActionChunk(
                    actions=[
                        [float(col) for col in range(action_dim)]
                        for _ in range(request.action_horizon)
                    ]
                ),
                future_frames=future_frames,
                value=value,
            )
        return InferenceResult(
            action_chunk=ActionChunk(actions=[[1.0, 2.0]]),
            future_frames=future_frames,
            value=value,
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

    def native_model_adapter_name(self) -> str:
        return "test_serve_adapter"

    def close(self) -> None:
        return


def _serve_registry() -> tuple[Registry, list[CapturingBackend]]:
    registry = Registry()
    created: list[CapturingBackend] = []

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> CapturingBackend:
        backend = CapturingBackend(manifest, profiles)
        created.append(backend)
        return backend

    registry.register_backend("fake", factory)
    registry.register_processor("passthrough", lambda manifest: PassthroughProcessor.from_manifest(manifest))
    registry.register_optimization("fake_cache", {"cache_scope": "replan"})
    return registry, created


def test_serve_infer_uses_request_observation(tmp_path) -> None:
    registry, created = _serve_registry()
    app = ServeApp("fake-open-loop", registry=registry, trace_dir=tmp_path)

    result = app.infer_once(
        {
            "observation": {
                "images": {"primary": [[[1, 2, 3]]]},
                "state": {"proprio": [0.1, 0.2]},
                "prompt": "pick up the block",
                "session": {"episode_id": 7, "step_id": 2},
            },
            "action_horizon": 5,
            "replan_steps": 3,
            "runtime_options": {"seed": 123},
        }
    )

    request = created[0].last_request
    assert result["action_chunk"]["shape"] == [1, 2]
    assert request is not None
    assert request.observation.prompt == "pick up the block"
    assert request.observation.session["episode_id"] == 7
    assert request.action_horizon == 5
    assert request.replan_steps == 3
    assert request.runtime_options["seed"] == 123
    app.close()
    events = read_events(app.trace_path)
    names = [event["event"] for event in events]
    assert names == [
        "serve_start",
        "backend_load_start",
        "backend_load",
        "backend_warmup",
        "reset",
        "serve_ready",
        "serve_request_start",
        "serve_request_end",
        "backend_close",
    ]
    request_end = [event for event in events if event["event"] == "serve_request_end"][0]
    assert request_end["action_chunk_shape"] == [1, 2]
    assert request_end["action_summary"]["shape"] == [1, 2]
    assert request_end["action_summary"]["mean"] == 1.5


def test_serve_infer_without_observation_requires_payload_by_default(tmp_path) -> None:
    registry, _created = _serve_registry()
    app = ServeApp("fake-open-loop", registry=registry, trace_dir=tmp_path)

    with pytest.raises(ValueError, match="observation.images"):
        app.infer_once({})

    assert app.health["accepts_synthetic_observation"] is False
    assert app.health["trace_path"] == str(app.trace_path)
    app.close()
    events = read_events(app.trace_path)
    request_end = [event for event in events if event["event"] == "serve_request_end"][0]
    assert request_end["status"] == "error"


def test_serve_smoke_mode_allows_processor_smoke_observation(tmp_path) -> None:
    registry, created = _serve_registry()
    app = ServeApp(
        "fake-open-loop",
        registry=registry,
        trace_dir=tmp_path,
        allow_synthetic_observation=True,
    )

    app.infer_once({})

    request = created[0].last_request
    assert request is not None
    assert request.observation.images["primary"]
    assert request.observation.prompt == "move the fake end effector to the target"
    assert app.health["accepts_synthetic_observation"] is True
    assert app.health["trace_path"] == str(app.trace_path)
    app.close()
    events = read_events(app.trace_path)
    request_end = [event for event in events if event["event"] == "serve_request_end"][0]
    assert request_end["status"] == "ok"
    assert request_end["synthetic_observation"] is True


def test_serve_returns_and_traces_future_and_value_outputs(tmp_path) -> None:
    registry = Registry()
    created: list[CapturingBackend] = []

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> CapturingBackend:
        backend = CapturingBackend(manifest, profiles, future_value=True)
        created.append(backend)
        return backend

    registry.register_backend("fake", factory)
    registry.register_processor("passthrough", lambda manifest: PassthroughProcessor.from_manifest(manifest))
    app = ServeApp("fake-open-loop", registry=registry, trace_dir=tmp_path)

    result = app.infer_once(
        {
            "observation": {
                "images": {"primary": [[[0, 0, 0]]]},
                "prompt": "serve future smoke",
            }
        }
    )

    assert result["future_frames"] == {
        "present": True,
        "count": 3,
        "artifact_path": "future/serve.json",
    }
    assert result["value"] == {"score": 0.25}
    app.close()
    events = read_events(app.trace_path)
    request_end = [event for event in events if event["event"] == "serve_request_end"][0]
    assert request_end["future_frames"] == result["future_frames"]
    assert request_end["value"] == result["value"]


def test_serve_maps_reference_entry_to_native_backend(tmp_path) -> None:
    registry = Registry()
    created: list[CapturingBackend] = []

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> CapturingBackend:
        backend = CapturingBackend(manifest, profiles, contract_shape=True)
        created.append(backend)
        return backend

    registry.register_backend("fastwam", factory)
    registry.register_processor("fastwam_libero", lambda manifest: PassthroughProcessor.from_manifest(manifest))

    app = ServeApp(
        "fastwam-libero",
        registry=registry,
        trace_dir=tmp_path,
        upstream_dir=tmp_path / "FastWAM",
        cache_dir=tmp_path / "cache",
        backend_overrides={"task": "libero_10"},
    )

    assert app.manifest.backend_name == "fastwam"
    assert app.manifest.backend["mode"] == "native_serve"
    assert app.manifest.backend["config"]["upstream_dir"] == str(tmp_path / "FastWAM")
    assert app.manifest.backend["config"]["cache_dir"] == str(tmp_path / "cache")
    assert app.manifest.backend["config"]["task"] == "libero_10"
    assert app.manifest.workload_name == "serve"
    assert "native_backend" not in app.manifest.backend["config"]
    assert created[0].manifest.backend_name == "fastwam"

    app.infer_once(
        {
            "observation": {
                "images": {"primary": [[[1, 1, 1]]], "wrist": [[[2, 2, 2]]]},
                "state": {"proprio": [0.0]},
                "prompt": "native serve request",
            }
        }
    )
    assert created[0].last_request is not None
    assert created[0].last_request.action_horizon == 32
    assert created[0].last_request.replan_steps == 10
    app.close()
    events = read_events(app.trace_path)
    contract = [event for event in events if event["event"] == "native_runtime_contract"][0]
    assert contract["mode"] == "native_serve"
    assert contract["backend"] == "fastwam"
    assert contract["workload"] == "serve"
    assert contract["model_adapter"] == "test_serve_adapter"
    assert contract["processor_modality"]["processor"] == "fastwam_libero"


def test_serve_traces_native_readiness_before_native_load_failure(tmp_path) -> None:
    with pytest.raises(NativePreflightError, match="fastwam native readiness is blocked"):
        ServeApp(
            "fastwam-libero",
            trace_dir=tmp_path,
            upstream_dir=tmp_path / "missing-fastwam",
        )

    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = read_events(trace_paths[0])
    names = [event["event"] for event in events]
    assert names[:4] == [
        "serve_start",
        "native_runtime_contract",
        "native_readiness",
        "error",
    ]
    contract = events[1]
    readiness = events[2]
    assert contract["mode"] == "native_serve"
    assert contract["backend"] == "fastwam"
    assert contract["runtime_mode"] == "in_process"
    assert contract["runtime_loader"] == "fastwam_runtime_loader"
    assert readiness["status"] == "blocked"
    assert readiness["backend"] == "fastwam"
    assert readiness["runtime_mode"] == "in_process"
    assert readiness["runtime_loader"] == "fastwam_runtime_loader"
    assert readiness["upstream"]["status"] == "missing"
    assert events[3]["stage"] == "native_preflight"
    assert events[3]["recoverable"] is True
    assert events[-1]["event"] == "backend_close"


def test_serve_traces_bad_request(tmp_path) -> None:
    registry, _created = _serve_registry()
    app = ServeApp("fake-open-loop", registry=registry, trace_dir=tmp_path)

    with pytest.raises(ValueError, match="observation.images"):
        app.infer_once({"observation": {"images": []}})

    app.close()
    events = read_events(app.trace_path)
    names = [event["event"] for event in events]
    assert "error" in names
    assert events[-2]["event"] == "serve_request_end"
    assert events[-2]["status"] == "error"


def test_observation_payload_validation() -> None:
    with pytest.raises(ValueError, match="observation.images"):
        _observation_from_payload({"observation": {"images": []}})

    with pytest.raises(ValueError, match="history entries"):
        _observation_from_payload({"observation": {"images": {}, "history": [1]}})


def test_observation_payload_accepts_direct_observation_shape() -> None:
    observation = _observation_from_payload({"images": {"primary": []}, "prompt": "direct"})

    assert isinstance(observation, Observation)
    assert observation.prompt == "direct"
