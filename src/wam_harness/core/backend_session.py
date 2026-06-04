from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from wam_harness.core.backend_capabilities import (
    action_contract_enabled,
    apply_loaded_optimization_profiles,
    plan_optimization_profiles,
    preflight_report,
    runtime_contract_payload,
)
from wam_harness.core.inference_trace import inference_result_payload
from wam_harness.core.memory import memory_snapshot
from wam_harness.core.preflight import assert_preflight
from wam_harness.core.registry import Backend, Processor
from wam_harness.core.tracing import TraceWriter
from wam_harness.core.types import (
    InferenceRequest,
    InferenceResult,
    Manifest,
    OptimizationProfile,
    RuntimeInfo,
)


@dataclass
class BackendSession:
    """Own the common backend lifecycle shared by run, serve, and smoke paths."""

    manifest: Manifest
    profiles: list[OptimizationProfile]
    backend: Backend
    processor: Processor | None
    trace: TraceWriter
    closed: bool = False

    @property
    def runtime_info(self) -> RuntimeInfo:
        return self.backend.runtime_info()

    def start(
        self,
        *,
        require_ready: bool = False,
        stage_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._set_stage(stage_callback, "optimization_plan")
        self.plan_optimizations()
        self._set_stage(stage_callback, "runtime_contract")
        self.emit_runtime_contract()
        self._set_stage(stage_callback, "preflight")
        self.emit_preflight(require_ready=require_ready)
        self._set_stage(stage_callback, "backend_load")
        self.load_backend()
        self._set_stage(stage_callback, "optimization_apply")
        self.apply_loaded_optimizations()
        self._set_stage(stage_callback, "backend_warmup")
        self.warmup_backend()
        self._set_stage(stage_callback, "backend_reset")
        self.reset_backend()

    def _set_stage(
        self,
        stage_callback: Callable[[str], None] | None,
        stage: str,
    ) -> None:
        if stage_callback is not None:
            stage_callback(stage)

    def plan_optimizations(self) -> None:
        statuses = plan_optimization_profiles(self.backend, self.profiles)
        if statuses:
            self.trace.set_runtime_info(self.backend.runtime_info())
            self.trace.write(
                "optimization_profile_status",
                stage="plan",
                profiles=statuses,
            )

    def apply_loaded_optimizations(self) -> None:
        statuses = apply_loaded_optimization_profiles(self.backend, self.profiles)
        if statuses:
            self.trace.set_runtime_info(self.backend.runtime_info())
            self.trace.write(
                "optimization_profile_status",
                stage="post_load",
                profiles=statuses,
            )

    def emit_runtime_contract(self) -> None:
        contract = runtime_contract_payload(
            self.backend,
            processor=self.processor,
        )
        if contract is not None:
            self.trace.write("runtime_contract", **contract)

    def emit_preflight(self, *, require_ready: bool = False) -> None:
        report = preflight_report(self.backend)
        if report is not None:
            self.trace.write("preflight", **report.to_trace_payload())
        assert_preflight(report, require_ready=require_ready)

    def load_backend(self) -> None:
        load_start = time.perf_counter()
        self.trace.write("backend_load_start")
        self.backend.load()
        runtime_info = self.backend.runtime_info()
        self.trace.set_runtime_info(runtime_info)
        self.trace.write(
            "backend_load",
            timing={"total_ms": (time.perf_counter() - load_start) * 1000},
            memory=memory_snapshot(),
        )

    def warmup_backend(self) -> None:
        start = time.perf_counter()
        self.backend.warmup()
        self.trace.write(
            "backend_warmup",
            timing={"total_ms": (time.perf_counter() - start) * 1000},
            memory=memory_snapshot(),
        )

    def reset_backend(self) -> None:
        self.backend.reset()
        self.trace.write("reset")

    def infer_and_trace(
        self,
        request: InferenceRequest,
        *,
        event: str,
        expected_horizon: int,
        started_at: float | None = None,
        payload: dict[str, Any] | None = None,
        validate_action_contract: bool | None = None,
    ) -> InferenceResult:
        start = started_at if started_at is not None else time.perf_counter()
        result = self.backend.infer(request)
        should_validate = (
            action_contract_enabled(self.backend)
            if validate_action_contract is None
            else validate_action_contract
        )
        self.trace.write(
            event,
            **(payload or {}),
            **inference_result_payload(
                self.manifest,
                result,
                expected_horizon=expected_horizon,
                wall_ms=(time.perf_counter() - start) * 1000,
                validate_action_contract=should_validate,
            ),
        )
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.backend.close()

    def __enter__(self) -> BackendSession:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
