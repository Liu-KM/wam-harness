from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eazywam.core.backend_capabilities import attach_processor
from eazywam.core.backend_session import BackendSession
from eazywam.core.registry import Backend, Processor, Registry
from eazywam.core.runtime import RuntimePlan, RuntimeSpec
from eazywam.core.tracing import TraceWriter
from eazywam.core.types import Manifest, OptimizationProfile, RuntimeInfo


@dataclass
class Invocation:
    """Common model invocation spine shared by run, serve, and smoke entrypoints.

    The invocation owns product-level assembly: model id resolution, runtime
    mapping, optimization profile construction, backend/processor creation,
    processor attachment, trace opening, and backend session construction.
    Entry points still own their user-visible semantics such as run loops, HTTP,
    or external simulator process execution.
    """

    model_id: str
    runtime_plan: RuntimePlan
    manifest: Manifest
    profiles: list[OptimizationProfile]
    backend: Backend
    processor: Processor
    run_id: str
    output_dir: Path
    trace_path: Path
    trace: TraceWriter
    session: BackendSession
    closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        registry: Registry,
        model_id: str,
        spec: RuntimeSpec,
        enabled_opts: list[str] | None = None,
        trace_dir: str | Path | None = None,
        upstream_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        backend_overrides: dict[str, str] | None = None,
    ) -> Invocation:
        reference_manifest = registry.load_manifest(model_id)
        runtime_plan = registry.resolve_runtime(
            reference_manifest,
            spec,
            upstream_dir=upstream_dir,
            cache_dir=cache_dir,
            backend_overrides=backend_overrides or {},
        )
        return cls.from_runtime_plan(
            registry=registry,
            model_id=model_id,
            runtime_plan=runtime_plan,
            enabled_opts=enabled_opts,
            trace_dir=trace_dir,
        )

    @classmethod
    def from_runtime_plan(
        cls,
        *,
        registry: Registry,
        model_id: str,
        runtime_plan: RuntimePlan,
        enabled_opts: list[str] | None = None,
        trace_dir: str | Path | None = None,
    ) -> Invocation:
        manifest = runtime_plan.manifest
        profiles = registry.build_optimization_profiles(
            manifest,
            enabled_opts or [],
            include_defaults=True,
        )
        backend = registry.create_backend(manifest, profiles)
        trace: TraceWriter | None = None
        try:
            processor = registry.create_processor(manifest)
            attach_processor(backend, processor)

            run_id = uuid.uuid4().hex[:12]
            output_dir = (
                Path(trace_dir) / run_id if trace_dir is not None else Path("runs") / run_id
            )
            trace_path = output_dir / "trace.jsonl"
            trace = TraceWriter(trace_path, run_id, backend.runtime_info())
            session = BackendSession(
                manifest=manifest,
                profiles=profiles,
                backend=backend,
                processor=processor,
                trace=trace,
            )
        except Exception:
            if trace is not None:
                trace.close()
            backend.close()
            raise

        return cls(
            model_id=model_id,
            runtime_plan=runtime_plan,
            manifest=manifest,
            profiles=profiles,
            backend=backend,
            processor=processor,
            run_id=run_id,
            output_dir=output_dir,
            trace_path=trace_path,
            trace=trace,
            session=session,
        )

    @property
    def runtime_info(self) -> RuntimeInfo:
        return self.session.runtime_info

    def write_start(self, event: str, **payload: Any) -> None:
        self.trace.write(
            event,
            output_dir=str(self.output_dir),
            optimization_profiles=[profile.to_dict() for profile in self.profiles],
            manifest_defaults=self.manifest.defaults,
            known_gaps=self.manifest.known_gaps,
            **payload,
        )

    def write_error(
        self,
        *,
        exc: Exception,
        stage: str,
        recoverable: bool = False,
        **payload: Any,
    ) -> None:
        self.trace.write(
            "error",
            stage=stage,
            error_type=type(exc).__name__,
            message=str(exc),
            recoverable=recoverable,
            backend=self.manifest.backend_name,
            **payload,
        )

    def write_finish(self, event: str, *, status: str, **payload: Any) -> None:
        self.trace.write(
            event,
            status=status,
            trace_path=str(self.trace_path),
            **payload,
        )

    def write_backend_close(self) -> None:
        self.trace.write("backend_close", trace_path=str(self.trace_path))

    def start_backend(
        self,
        *,
        require_ready: bool = False,
        stage_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.session.start(
            require_ready=require_ready,
            stage_callback=stage_callback,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.session.close()
        finally:
            self.trace.close()

    def __enter__(self) -> Invocation:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
