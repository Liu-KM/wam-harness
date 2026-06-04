from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from wam_harness.core.action_contract import ActionContractError
from wam_harness.core.inference_trace import (
    observation_summary,
)
from wam_harness.core.invocation import Invocation
from wam_harness.core.preflight import PreflightError
from wam_harness.core.registry import Registry, default_registry
from wam_harness.core.runtime import INPUT_RUN_SPEC, RUN_SPEC
from wam_harness.core.types import (
    InferenceRequest,
    InferenceResult,
    Manifest,
    Observation,
    RuntimeInfo,
)
from wam_harness.workloads.processor_smoke import ProcessorSmokeWorkload
from wam_harness.workloads.single_observation import SingleObservationWorkload


class RunInputRequiredError(RuntimeError):
    """Raised when a real WAM run needs an explicit observation input."""


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    model_id: str
    trace_path: Path
    steps: int
    model_calls: int
    status: str
    runtime_info: RuntimeInfo
    result: InferenceResult | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "trace_path": str(self.trace_path),
            "steps": self.steps,
            "model_calls": self.model_calls,
            "status": self.status,
            "runtime_info": self.runtime_info.to_dict(),
        }
        if self.result is not None:
            data["result"] = self.result.to_dict()
        return data


class Runner:
    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or default_registry()

    def run(
        self,
        model_id: str,
        enabled_opts: list[str] | None = None,
        trace_dir: str | Path | None = None,
        episode_length: int | None = None,
        action_horizon: int | None = None,
        replan_steps: int | None = None,
        upstream_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        backend_overrides: dict[str, str] | None = None,
        observation: Observation | None = None,
        runtime_options: dict[str, object] | None = None,
    ) -> RunSummary:
        reference_manifest = self.registry.load_manifest(model_id)
        runtime_plan = self.registry.resolve_runtime(
            reference_manifest,
            INPUT_RUN_SPEC if observation is not None else RUN_SPEC,
            upstream_dir=upstream_dir,
            cache_dir=cache_dir,
            backend_overrides=backend_overrides or {},
        )
        if observation is None and runtime_plan.transformed:
            raise RunInputRequiredError(_run_input_required_message(model_id))
        invocation = Invocation.from_runtime_plan(
            registry=self.registry,
            model_id=model_id,
            runtime_plan=runtime_plan,
            enabled_opts=enabled_opts,
            trace_dir=trace_dir,
        )
        manifest = invocation.manifest
        workload = self._create_workload(manifest, observation, invocation.processor)

        defaults = manifest.defaults
        effective_action_horizon = int(
            action_horizon or defaults.get("action_horizon") or _default_horizon(manifest)
        )
        effective_replan_steps = int(
            replan_steps or defaults.get("replan_steps") or effective_action_horizon
        )
        if episode_length is not None:
            workload.episode_length = episode_length

        pending_actions: list[list[float]] = []
        model_calls = 0
        steps = 0
        last_result: InferenceResult | None = None
        runtime_info = invocation.backend.runtime_info()

        with invocation:
            trace = invocation.trace
            session = invocation.session
            invocation.write_start(
                "run_start",
                mode=str(manifest.backend.get("mode", "run")),
                telemetry_config={"trace_format": "jsonl"},
                synthetic_observation=manifest.workload_name == "processor_smoke",
            )
            try:
                invocation.start_backend()
                runtime_info = session.runtime_info
                workload.reset()
                trace.write("episode_start", episode_id=workload.episode_id)

                while not workload.done:
                    if not pending_actions or workload.steps_since_replan >= effective_replan_steps:
                        observation = workload.observation()
                        replan_id = model_calls
                        trace.write(
                            "replan_start",
                            episode_id=workload.episode_id,
                            step_id=workload.step_id,
                            replan_id=replan_id,
                            action_horizon=effective_action_horizon,
                            replan_steps=effective_replan_steps,
                            observation_summary=observation_summary(observation),
                            history_len=len(observation.history),
                        )
                        request = InferenceRequest(
                            observation=observation,
                            action_horizon=effective_action_horizon,
                            replan_steps=effective_replan_steps,
                            optimization_profiles=invocation.profiles,
                            runtime_options=runtime_options or {},
                        )
                        trace.write(
                            "inference_start",
                            episode_id=workload.episode_id,
                            step_id=workload.step_id,
                            replan_id=replan_id,
                        )
                        result = session.infer_and_trace(
                            request,
                            event="inference_end",
                            expected_horizon=effective_action_horizon,
                            payload={
                                "episode_id": workload.episode_id,
                                "step_id": workload.step_id,
                                "replan_id": replan_id,
                                "action_horizon": effective_action_horizon,
                                "replan_steps": effective_replan_steps,
                            },
                        )
                        last_result = result
                        pending_actions = list(result.action_chunk.actions)
                        workload.mark_replan()
                        model_calls += 1

                    action = pending_actions.pop(0) if pending_actions else []
                    from_stale_chunk = bool(pending_actions)
                    step_start = time.perf_counter()
                    workload.step(action)
                    steps += 1
                    trace.write(
                        "step",
                        episode_id=workload.episode_id,
                        step_id=workload.step_id,
                        action_horizon=effective_action_horizon,
                        replan_steps=effective_replan_steps,
                        action_chunk_len=len(pending_actions),
                        action_dim=len(action),
                        from_stale_chunk=from_stale_chunk,
                        env_step_ms=(time.perf_counter() - step_start) * 1000,
                        action=action,
                    )

                trace.write("episode_end", episode_id=workload.episode_id, steps=steps)
                trace.write(
                    "run_end",
                    status="ok",
                    model_calls=model_calls,
                    steps=steps,
                    warnings=[],
                    trace_path=str(invocation.trace_path),
                )
            except Exception as exc:
                trace.write(
                    "error",
                    stage="preflight"
                    if isinstance(exc, PreflightError)
                    else "action_contract"
                    if isinstance(exc, ActionContractError)
                    else "runner",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    recoverable=isinstance(exc, PreflightError),
                    backend=manifest.backend_name,
                )
                trace.write(
                    "run_end",
                    status="error",
                    model_calls=model_calls,
                    steps=steps,
                    warnings=[str(exc)],
                    trace_path=str(invocation.trace_path),
                )
                raise

        return RunSummary(
            run_id=invocation.run_id,
            model_id=model_id,
            trace_path=invocation.trace_path,
            steps=steps,
            model_calls=model_calls,
            status="ok",
            runtime_info=runtime_info,
            result=last_result,
        )

    def _create_workload(
        self,
        manifest: Manifest,
        observation: Observation | None,
        processor: object,
    ) -> object:
        if observation is not None:
            return SingleObservationWorkload.from_observation(manifest, observation)
        if manifest.workload_name == "processor_smoke":
            return ProcessorSmokeWorkload.from_processor(manifest, processor)
        return self.registry.create_workload(manifest)


def _default_horizon(manifest: Manifest) -> int:
    action = manifest.processor.get("action", {})
    if isinstance(action, dict) and action.get("horizon") is not None:
        return int(action["horizon"])
    return 1


def _run_input_required_message(model_id: str) -> str:
    return (
        f"{model_id} needs an observation input for `wam run`.\n"
        "Try:\n"
        f"  wam run {model_id} --input obs.json --output action.json\n"
        f"  wam eval {model_id} --workload libero-single-task --task-id 0 --num-trials 1\n"
        f"  wam serve {model_id}"
    )
