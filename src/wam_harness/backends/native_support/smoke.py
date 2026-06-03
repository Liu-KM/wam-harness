from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from wam_harness.backends.native_support.runtime import (
    NATIVE_SMOKE_SPEC,
    NativeRuntimeError,
    resolve_native_runtime,
)
from wam_harness.core.action_contract import ActionContractError
from wam_harness.core.backend_session import BackendSession
from wam_harness.core.preflight import PreflightError
from wam_harness.core.registry import Registry, default_registry
from wam_harness.core.tracing import TraceWriter
from wam_harness.core.types import InferenceRequest, Manifest, Observation, RuntimeInfo


class NativeSmokeRunnerError(RuntimeError):
    """Raised when a native smoke run cannot be planned."""


@dataclass(frozen=True)
class NativeSmokeSummary:
    run_id: str
    model_id: str
    trace_path: Path
    status: str
    runtime_info: RuntimeInfo
    action_chunk_shape: list[int]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "trace_path": str(self.trace_path),
            "status": self.status,
            "runtime_info": self.runtime_info.to_dict(),
            "action_chunk_shape": self.action_chunk_shape,
            "warnings": self.warnings,
        }


class NativeSmokeRunner:
    """Run one synthetic observation through a native backend migration path."""

    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or default_registry()

    def run(
        self,
        model_id: str,
        enabled_opts: list[str] | None = None,
        trace_dir: str | Path | None = None,
        upstream_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        action_horizon: int | None = None,
        replan_steps: int | None = None,
        backend_overrides: dict[str, str] | None = None,
        require_ready: bool = False,
    ) -> NativeSmokeSummary:
        manifest = native_smoke_manifest(
            self.registry.load_manifest(model_id),
            upstream_dir=upstream_dir,
            cache_dir=cache_dir,
            backend_overrides=backend_overrides or {},
        )
        profiles = self.registry.build_optimization_profiles(manifest, enabled_opts or [])
        backend = self.registry.create_backend(manifest, profiles)
        processor = self.registry.create_processor(manifest)

        defaults = manifest.defaults
        effective_action_horizon = int(
            action_horizon or defaults.get("action_horizon") or _default_horizon(manifest)
        )
        effective_replan_steps = int(
            replan_steps or defaults.get("replan_steps") or effective_action_horizon
        )
        run_id = uuid.uuid4().hex[:12]
        output_dir = (Path(trace_dir) / run_id) if trace_dir is not None else Path("runs") / run_id
        trace_path = output_dir / "trace.jsonl"
        runtime_info = backend.runtime_info()
        warnings: list[str] = []
        action_shape: list[int] = []

        with TraceWriter(trace_path, run_id, runtime_info) as trace:
            with BackendSession(
                manifest=manifest,
                profiles=profiles,
                backend=backend,
                processor=processor,
                trace=trace,
            ) as session:
                trace.write(
                    "run_start",
                    mode="native_smoke",
                    output_dir=str(output_dir),
                    optimization_profiles=[profile.to_dict() for profile in profiles],
                    manifest_defaults=manifest.defaults,
                    known_gaps=manifest.known_gaps,
                    synthetic_observation=True,
                )
                stage = "preflight"
                try:
                    session.emit_runtime_contract()
                    session.emit_preflight(require_ready=require_ready)
                    stage = "processor_smoke_observation"
                    observation = processor.smoke_observation()
                    trace.write(
                        "processor_smoke_observation",
                        observation_summary=_observation_summary(observation),
                    )
                    stage = "backend_load"
                    session.load_backend(split_end_event=True)
                    runtime_info = session.runtime_info
                    stage = "backend_warmup"
                    session.warmup_backend()
                    stage = "backend_reset"
                    session.reset_backend()

                    stage = "inference"
                    request = InferenceRequest(
                        observation=observation,
                        action_horizon=effective_action_horizon,
                        replan_steps=effective_replan_steps,
                        optimization_profiles=profiles,
                        runtime_options={"native_smoke": True},
                    )
                    trace.write(
                        "inference_start",
                        action_horizon=effective_action_horizon,
                        replan_steps=effective_replan_steps,
                        observation_summary=_observation_summary(observation),
                    )
                    result = session.infer_and_trace(
                        request,
                        event="inference_end",
                        expected_horizon=effective_action_horizon,
                        validate_action_contract=True,
                    )
                    action_shape = [
                        result.action_chunk.horizon,
                        result.action_chunk.action_dim,
                    ]
                    warnings.extend(result.warnings)
                    trace.write(
                        "run_end",
                        status="ok",
                        model_calls=1,
                        steps=0,
                        warnings=warnings,
                        trace_path=str(trace_path),
                    )
                except Exception as exc:
                    setattr(exc, "trace_path", trace_path)
                    trace.write(
                        "error",
                        stage="preflight"
                        if isinstance(exc, PreflightError)
                        else "action_contract"
                        if isinstance(exc, ActionContractError)
                        else stage,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        recoverable=isinstance(exc, PreflightError),
                        backend=manifest.backend_name,
                        trace_path=str(trace_path),
                    )
                    trace.write(
                        "run_end",
                        status="error",
                        model_calls=0,
                        steps=0,
                        warnings=[str(exc)],
                        trace_path=str(trace_path),
                    )
                    raise

        return NativeSmokeSummary(
            run_id=run_id,
            model_id=model_id,
            trace_path=trace_path,
            status="ok",
            runtime_info=runtime_info,
            action_chunk_shape=action_shape,
            warnings=warnings,
        )


def native_smoke_manifest(
    manifest: Manifest,
    *,
    upstream_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    backend_overrides: dict[str, str] | None = None,
) -> Manifest:
    try:
        return resolve_native_runtime(
            manifest,
            NATIVE_SMOKE_SPEC,
            upstream_dir=upstream_dir,
            cache_dir=cache_dir,
            backend_overrides=backend_overrides,
        ).manifest
    except NativeRuntimeError as exc:
        raise NativeSmokeRunnerError(
            f"{manifest.id} does not declare backend.config.native_backend for native smoke"
        ) from exc


def _default_horizon(manifest: Manifest) -> int:
    action = manifest.processor.get("action", {})
    if isinstance(action, dict) and action.get("horizon") is not None:
        return int(action["horizon"])
    return 1


def _observation_summary(observation: Observation) -> dict[str, object]:
    return {
        "image_keys": sorted(observation.images.keys()),
        "state_keys": sorted(observation.state.keys()),
        "prompt": observation.prompt,
        "session_keys": sorted(observation.session.keys()),
    }
