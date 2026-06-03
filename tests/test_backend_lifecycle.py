from __future__ import annotations

import json

import pytest

from wam_harness.backends.native_support.smoke import NativeSmokeRunner
from wam_harness.core.registry import Registry, RegistryError
from wam_harness.core.runner import Runner
from wam_harness.core.types import (
    ActionChunk,
    InferenceRequest,
    InferenceResult,
    Manifest,
    Observation,
    OptimizationProfile,
    RuntimeInfo,
)
from wam_harness.serve import ServeApp, WamHTTPServer
from wam_harness.workloads.open_loop import OpenLoopWorkload


class TrackingBackend:
    def __init__(
        self,
        manifest: Manifest,
        profiles: list[OptimizationProfile],
        *,
        fail_load: bool = False,
        fail_infer: bool = False,
        fail_warmup: bool = False,
        fail_reset: bool = False,
    ) -> None:
        self.manifest = manifest
        self.profiles = profiles
        self.fail_load = fail_load
        self.fail_infer = fail_infer
        self.fail_warmup = fail_warmup
        self.fail_reset = fail_reset
        self.closed = False
        self.loaded = False
        self.warmed = False

    def load(self) -> None:
        if self.fail_load:
            raise RuntimeError("load failed")
        self.loaded = True

    def warmup(self) -> None:
        if self.fail_warmup:
            raise RuntimeError("warmup failed")
        self.warmed = True

    def reset(self) -> None:
        if self.fail_reset:
            raise RuntimeError("reset failed")
        return

    def infer(self, request: InferenceRequest) -> InferenceResult:
        if self.fail_infer:
            raise RuntimeError("infer failed")
        action_dim = 4
        if str(self.manifest.backend.get("mode", "")).startswith("native_"):
            action_dim = int(self.manifest.processor.get("action", {}).get("dim") or 1)
        return InferenceResult(
            action_chunk=ActionChunk(
                actions=[[0.0] * action_dim for _ in range(request.action_horizon)]
            )
        )

    def runtime_info(self) -> RuntimeInfo:
        defaults = self.manifest.defaults
        return RuntimeInfo(
            manifest_id=self.manifest.id,
            model_name=self.manifest.display_name,
            backend=self.manifest.backend_name,
            processor=self.manifest.processor_name,
            source_repo=self.manifest.source_repo,
            mode=str(self.manifest.backend.get("mode", "fake")),
            device=str(defaults.get("device", "cpu")),
            dtype=str(defaults.get("dtype", "fp32")),
            optimization_profiles=self.profiles,
            metadata={},
        )

    def close(self) -> None:
        self.closed = True


class SmokeOnlyProcessor:
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
            images={"primary": [[[0, 0, 0]]]},
            state={"proprio": [0.0]},
            prompt="smoke",
            session={"episode_id": 0},
        )


def _fake_registry(backend: TrackingBackend | None = None) -> tuple[Registry, list[TrackingBackend]]:
    registry = Registry()
    created: list[TrackingBackend] = []

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> TrackingBackend:
        instance = backend or TrackingBackend(manifest, profiles)
        created.append(instance)
        return instance

    registry.register_backend("fake", factory)
    registry.register_processor("passthrough", lambda manifest: SmokeOnlyProcessor())
    registry.register_workload("open_loop", lambda manifest: OpenLoopWorkload.from_manifest(manifest))
    return registry, created


def test_runner_closes_backend_after_success(tmp_path) -> None:
    registry, created = _fake_registry()

    Runner(registry=registry).run("fake-open-loop", trace_dir=tmp_path, episode_length=1)

    assert created[0].closed is True


def test_runner_closes_backend_after_error(tmp_path) -> None:
    backend = TrackingBackend.__new__(TrackingBackend)
    registry, created = _fake_registry()

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> TrackingBackend:
        TrackingBackend.__init__(backend, manifest, profiles, fail_infer=True)
        created.append(backend)
        return backend

    registry.register_backend("fake", factory)

    with pytest.raises(RuntimeError, match="infer failed"):
        Runner(registry=registry).run("fake-open-loop", trace_dir=tmp_path, episode_length=1)

    assert backend.closed is True


def test_runner_traces_backend_load_failure(tmp_path) -> None:
    backend = TrackingBackend.__new__(TrackingBackend)
    registry, created = _fake_registry()

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> TrackingBackend:
        TrackingBackend.__init__(backend, manifest, profiles, fail_load=True)
        created.append(backend)
        return backend

    registry.register_backend("fake", factory)

    with pytest.raises(RuntimeError, match="load failed"):
        Runner(registry=registry).run("fake-open-loop", trace_dir=tmp_path, episode_length=1)

    assert backend.closed is True
    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = [
        json.loads(line)
        for line in trace_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_start",
        "backend_load_start",
        "error",
        "run_end",
    ]
    assert events[-1]["status"] == "error"


def test_native_smoke_closes_backend_after_success(tmp_path) -> None:
    registry = Registry()
    created: list[TrackingBackend] = []

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> TrackingBackend:
        backend = TrackingBackend(manifest, profiles)
        created.append(backend)
        return backend

    registry.register_backend("fastwam", factory)
    registry.register_processor("fastwam_libero", lambda manifest: SmokeOnlyProcessor())

    NativeSmokeRunner(registry=registry).run("fastwam-libero", trace_dir=tmp_path)

    assert created[0].closed is True


def test_native_smoke_traces_backend_load_failure(tmp_path) -> None:
    registry = Registry()
    created: list[TrackingBackend] = []

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> TrackingBackend:
        backend = TrackingBackend(manifest, profiles, fail_load=True)
        created.append(backend)
        return backend

    registry.register_backend("fastwam", factory)
    registry.register_processor("fastwam_libero", lambda manifest: SmokeOnlyProcessor())

    with pytest.raises(RuntimeError, match="load failed"):
        NativeSmokeRunner(registry=registry).run("fastwam-libero", trace_dir=tmp_path)

    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = [
        json.loads(line)
        for line in trace_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_start",
        "processor_smoke_observation",
        "backend_load_start",
        "error",
        "run_end",
    ]
    assert events[-2]["stage"] == "backend_load"
    assert events[-1]["status"] == "error"
    assert created[0].closed is True


@pytest.mark.parametrize(
    ("failure_kwargs", "expected_stage", "message"),
    [
        ({"fail_warmup": True}, "backend_warmup", "warmup failed"),
        ({"fail_reset": True}, "backend_reset", "reset failed"),
        ({"fail_infer": True}, "inference", "infer failed"),
    ],
)
def test_native_smoke_traces_lifecycle_failure_stage(
    tmp_path,
    failure_kwargs,
    expected_stage: str,
    message: str,
) -> None:
    registry = Registry()

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> TrackingBackend:
        return TrackingBackend(manifest, profiles, **failure_kwargs)

    registry.register_backend("fastwam", factory)
    registry.register_processor("fastwam_libero", lambda manifest: SmokeOnlyProcessor())

    with pytest.raises(RuntimeError, match=message):
        NativeSmokeRunner(registry=registry).run("fastwam-libero", trace_dir=tmp_path)

    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = [
        json.loads(line)
        for line in trace_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    assert events[-2]["event"] == "error"
    assert events[-2]["stage"] == expected_stage
    assert events[-2]["trace_path"] == str(trace_paths[0])
    assert events[-1]["event"] == "run_end"
    assert events[-1]["status"] == "error"


def test_serve_app_closes_backend_on_init_failure(tmp_path) -> None:
    backend = TrackingBackend.__new__(TrackingBackend)
    registry, created = _fake_registry()

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> TrackingBackend:
        TrackingBackend.__init__(backend, manifest, profiles, fail_warmup=True)
        created.append(backend)
        return backend

    registry.register_backend("fake", factory)

    with pytest.raises(RuntimeError, match="warmup failed"):
        ServeApp("fake-open-loop", registry=registry, trace_dir=tmp_path)

    assert backend.closed is True
    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = [
        json.loads(line)
        for line in trace_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "serve_start",
        "backend_load_start",
        "backend_load",
        "error",
        "backend_close",
    ]
    assert events[-2]["stage"] == "serve_start"


def test_serve_app_closes_backend_when_processor_creation_fails(tmp_path) -> None:
    backend = TrackingBackend.__new__(TrackingBackend)
    registry, created = _fake_registry()

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> TrackingBackend:
        TrackingBackend.__init__(backend, manifest, profiles)
        created.append(backend)
        return backend

    registry.register_backend("fake", factory)
    registry.processors.clear()

    with pytest.raises(RegistryError, match="unknown processor"):
        ServeApp("fake-open-loop", registry=registry, trace_dir=tmp_path)

    assert backend.closed is True
    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = [
        json.loads(line)
        for line in trace_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == ["serve_start", "error", "backend_close"]


def test_http_server_close_closes_backend() -> None:
    registry, created = _fake_registry()
    app = ServeApp("fake-open-loop", registry=registry)
    server = WamHTTPServer(("127.0.0.1", 0), app)

    server.server_close()

    assert created[0].closed is True
