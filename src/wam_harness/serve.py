from __future__ import annotations

import json
import threading
import time
import uuid
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from wam_harness.core.action_contract import ActionContractError
from wam_harness.core.inference_trace import inference_result_payload
from wam_harness.core.memory import memory_snapshot
from wam_harness.backends.native_support.contract import native_runtime_contract_payload
from wam_harness.backends.native_support.readiness import (
    NativePreflightError,
    assert_native_preflight,
    native_readiness_payload,
)
from wam_harness.backends.native_support.runtime import (
    NATIVE_SERVE_SPEC,
    resolve_native_runtime,
)
from wam_harness.core.observation_io import (
    dict_or_empty as _dict_or_empty,
    observation_from_payload as _observation_from_payload,
)
from wam_harness.core.registry import Registry, default_registry
from wam_harness.core.tracing import TraceWriter
from wam_harness.core.types import InferenceRequest, OptimizationProfile


class ServeApp:
    def __init__(
        self,
        model_id: str,
        enabled_opts: list[str] | None = None,
        registry: Registry | None = None,
        trace_dir: str | Path | None = None,
        upstream_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        backend_overrides: dict[str, str] | None = None,
        allow_synthetic_observation: bool = False,
    ) -> None:
        self.registry = registry or default_registry()
        self.run_id = uuid.uuid4().hex[:12]
        self.output_dir = (Path(trace_dir) / self.run_id) if trace_dir is not None else Path("runs") / self.run_id
        self.trace_path = self.output_dir / "trace.jsonl"
        self.trace: TraceWriter | None = None
        self.closed = False
        self.allow_synthetic_observation = allow_synthetic_observation
        runtime_plan = resolve_native_runtime(
            self.registry.load_manifest(model_id),
            NATIVE_SERVE_SPEC,
            upstream_dir=upstream_dir,
            cache_dir=cache_dir,
            backend_overrides=backend_overrides or {},
        )
        self.manifest = runtime_plan.manifest
        self.profiles = self.registry.build_optimization_profiles(
            self.manifest, enabled_opts or []
        )
        self.backend = self.registry.create_backend(self.manifest, self.profiles)
        self.trace = TraceWriter(self.trace_path, self.run_id, self.backend.runtime_info())
        self.trace.write(
            "serve_start",
            status="starting",
            output_dir=str(self.output_dir),
            optimization_profiles=[profile.to_dict() for profile in self.profiles],
            manifest_defaults=self.manifest.defaults,
            known_gaps=self.manifest.known_gaps,
        )
        try:
            self.processor = self.registry.create_processor(self.manifest)
            contract = native_runtime_contract_payload(
                self.manifest,
                self.profiles,
                processor=self.processor,
                backend=self.backend,
            )
            if contract is not None:
                self._trace("native_runtime_contract", **contract)
            readiness = native_readiness_payload(self.backend)
            if readiness is not None:
                self._trace("native_readiness", **readiness)
            assert_native_preflight(readiness)
            load_start = time.perf_counter()
            self._trace("backend_load_start")
            self.backend.load()
            self.trace.set_runtime_info(self.backend.runtime_info())
            self._trace(
                "backend_load",
                timing={"total_ms": (time.perf_counter() - load_start) * 1000},
                memory=memory_snapshot(),
            )
            warmup_start = time.perf_counter()
            self.backend.warmup()
            self._trace(
                "backend_warmup",
                timing={"total_ms": (time.perf_counter() - warmup_start) * 1000},
                memory=memory_snapshot(),
            )
            self.backend.reset()
            self._trace("reset")
            self._trace("serve_ready", status="ok")
        except Exception as exc:
            self._trace(
                "error",
                stage="native_preflight"
                if isinstance(exc, NativePreflightError)
                else "serve_start",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=isinstance(exc, NativePreflightError),
                backend=self.manifest.backend_name,
            )
            self.close()
            raise

    @property
    def runtime_info(self) -> dict[str, Any]:
        return self.backend.runtime_info().to_dict()

    @property
    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "run_id": self.run_id,
            "trace_path": str(self.trace_path),
            "runtime_info": self.runtime_info,
            "accepts_synthetic_observation": self.allow_synthetic_observation,
            "endpoints": {
                "health": "GET /health",
                "infer": "POST /infer",
            },
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.backend.close()
        finally:
            if self.trace is not None:
                self.trace.write("backend_close", trace_path=str(self.trace_path))
                self.trace.close()
                self.trace = None

    def infer_once(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        self._trace(
            "serve_request_start",
            request_id=request_id,
            payload_keys=sorted(payload.keys()),
        )
        try:
            observation = _observation_from_payload(payload)
            synthetic_observation = False
            if observation is None:
                if not self.allow_synthetic_observation:
                    raise ValueError("request body must contain observation.images")
                observation = self.processor.smoke_observation()
                synthetic_observation = True
            defaults = self.manifest.defaults
            action_horizon = _request_int(
                payload,
                "action_horizon",
                defaults.get("action_horizon") or _default_action_horizon(self.manifest),
            )
            request = InferenceRequest(
                observation=observation,
                action_horizon=action_horizon,
                replan_steps=_request_int(
                    payload,
                    "replan_steps",
                    defaults.get("replan_steps") or action_horizon,
                ),
                optimization_profiles=[
                    profile
                    for profile in self.profiles
                    if isinstance(profile, OptimizationProfile)
                ],
                reset=bool(payload.get("reset", False)),
                runtime_options=_dict_or_empty(payload.get("runtime_options"), "runtime_options"),
            )
            if request.reset:
                self.backend.reset()
            result = self.backend.infer(request)
            self._trace(
                "serve_request_end",
                request_id=request_id,
                status="ok",
                action_horizon=request.action_horizon,
                replan_steps=request.replan_steps,
                synthetic_observation=synthetic_observation,
                **inference_result_payload(
                    self.manifest,
                    result,
                    expected_horizon=request.action_horizon,
                    wall_ms=(time.perf_counter() - start) * 1000,
                ),
            )
            return result.to_dict()
        except Exception as exc:
            self._trace(
                "error",
                stage="action_contract"
                if isinstance(exc, ActionContractError)
                else "serve_request",
                request_id=request_id,
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=isinstance(exc, ValueError)
                and not isinstance(exc, ActionContractError),
                backend=self.manifest.backend_name,
            )
            self._trace(
                "serve_request_end",
                request_id=request_id,
                status="error",
                timing={"wall_ms": (time.perf_counter() - start) * 1000},
                warnings=[str(exc)],
            )
            raise

    def _trace(self, event: str, **payload: Any) -> None:
        if self.trace is not None:
            self.trace.write(event, **payload)


def make_handler(app: ServeApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json(200, app.health)
                return
            self._write_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path == "/infer":
                try:
                    payload = self._read_json()
                    self._write_json(200, app.infer_once(payload))
                except ValueError as exc:
                    self._write_json(400, {"error": "bad_request", "message": str(exc)})
                return
            self._write_json(404, {"error": "not_found"})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0") or 0)
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("request body must be valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


class WamHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], app: ServeApp) -> None:
        self.app = app
        super().__init__(server_address, make_handler(app))

    def server_close(self) -> None:
        try:
            self.app.close()
        finally:
            super().server_close()


def serve(
    model_id: str,
    enabled_opts: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    trace_dir: str | Path | None = None,
    upstream_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    backend_overrides: dict[str, str] | None = None,
    allow_synthetic_observation: bool = False,
) -> ThreadingHTTPServer:
    app = ServeApp(
        model_id=model_id,
        enabled_opts=enabled_opts,
        trace_dir=trace_dir,
        upstream_dir=upstream_dir,
        cache_dir=cache_dir,
        backend_overrides=backend_overrides,
        allow_synthetic_observation=allow_synthetic_observation,
    )
    return WamHTTPServer((host, port), app)


def smoke_serve(
    model_id: str,
    enabled_opts: list[str] | None = None,
    trace_dir: str | Path | None = None,
    upstream_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    backend_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    server = serve(
        model_id=model_id,
        enabled_opts=enabled_opts,
        port=0,
        trace_dir=trace_dir,
        upstream_dir=upstream_dir,
        cache_dir=cache_dir,
        backend_overrides=backend_overrides,
        allow_synthetic_observation=True,
    )
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        request = urllib.request.Request(
            f"http://{host}:{port}/infer",
            data=b"{}",
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            inference = json.loads(response.read().decode("utf-8"))
        return {"health": health, "inference": inference}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request_int(payload: dict[str, Any], key: str, default: object) -> int:
    value = payload.get(key)
    if value is None:
        value = default
    return int(value)


def _default_action_horizon(manifest: object) -> int:
    processor = getattr(manifest, "processor", {})
    if isinstance(processor, dict):
        action = processor.get("action", {})
        if isinstance(action, dict) and action.get("horizon") is not None:
            return int(action["horizon"])
    return 1
