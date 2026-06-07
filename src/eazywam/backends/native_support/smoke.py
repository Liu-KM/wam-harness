from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eazywam.core.action_contract import ActionContractError
from eazywam.core.invocation import Invocation
from eazywam.core.preflight import PreflightError
from eazywam.core.registry import Registry, default_registry
from eazywam.core.runtime import RuntimeResolutionError, RuntimeSpec
from eazywam.core.types import InferenceRequest, Manifest, Observation, RuntimeInfo


class NativeSmokeRunnerError(RuntimeError):
    """Raised when a native smoke run cannot be planned."""


NATIVE_SMOKE_RUNTIME_SPEC = RuntimeSpec(
    mode="native_smoke",
    workload_name="native_smoke",
    workload_config={"synthetic_observation": True},
    require_backend_mapping=True,
)


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
        try:
            reference_manifest = self.registry.load_manifest(model_id)
            runtime_plan = self.registry.resolve_runtime(
                reference_manifest,
                NATIVE_SMOKE_RUNTIME_SPEC,
                upstream_dir=upstream_dir,
                cache_dir=cache_dir,
                backend_overrides=backend_overrides or {},
            )
        except RuntimeResolutionError as exc:
            raise NativeSmokeRunnerError(
                f"{model_id} does not declare backend.config.native_backend for native smoke"
            ) from exc
        invocation = Invocation.from_runtime_plan(
            registry=self.registry,
            model_id=model_id,
            runtime_plan=runtime_plan,
            enabled_opts=enabled_opts,
            trace_dir=trace_dir,
        )
        manifest = invocation.manifest
        processor = invocation.processor

        defaults = manifest.defaults
        effective_action_horizon = int(
            action_horizon or defaults.get("action_horizon") or _default_horizon(manifest)
        )
        effective_replan_steps = int(
            replan_steps or defaults.get("replan_steps") or effective_action_horizon
        )
        runtime_info = invocation.backend.runtime_info()
        warnings: list[str] = []
        action_shape: list[int] = []

        with invocation:
            trace = invocation.trace
            session = invocation.session
            invocation.write_start(
                "run_start",
                mode="native_smoke",
                synthetic_observation=True,
            )
            stage = "preflight"
            try:
                def set_stage(value: str) -> None:
                    nonlocal stage
                    stage = value

                invocation.start_backend(
                    require_ready=require_ready,
                    stage_callback=set_stage,
                )
                runtime_info = session.runtime_info
                stage = "processor_smoke_observation"
                observation = processor.smoke_observation()
                trace.write(
                    "processor_smoke_observation",
                    observation_summary=_observation_summary(observation),
                )

                stage = "inference"
                request = InferenceRequest(
                    observation=observation,
                    action_horizon=effective_action_horizon,
                    replan_steps=effective_replan_steps,
                    optimization_profiles=invocation.profiles,
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
                invocation.write_finish(
                    "run_end",
                    status="ok",
                    model_calls=1,
                    steps=0,
                    warnings=warnings,
                )
            except Exception as exc:
                setattr(exc, "trace_path", invocation.trace_path)
                invocation.write_error(
                    exc=exc,
                    stage="preflight"
                    if isinstance(exc, PreflightError)
                    else "action_contract"
                    if isinstance(exc, ActionContractError)
                    else stage,
                    recoverable=isinstance(exc, PreflightError),
                    trace_path=str(invocation.trace_path),
                )
                invocation.write_finish(
                    "run_end",
                    status="error",
                    model_calls=0,
                    steps=0,
                    warnings=[str(exc)],
                )
                raise

        return NativeSmokeSummary(
            run_id=invocation.run_id,
            model_id=model_id,
            trace_path=invocation.trace_path,
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
    registry: Registry | None = None,
) -> Manifest:
    try:
        return (registry or default_registry()).resolve_runtime(
            manifest,
            NATIVE_SMOKE_RUNTIME_SPEC,
            upstream_dir=upstream_dir,
            cache_dir=cache_dir,
            backend_overrides=backend_overrides or {},
        ).manifest
    except RuntimeResolutionError as exc:
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
