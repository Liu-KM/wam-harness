from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from wam_harness.core.action_contract import ActionContractError
from wam_harness.core.backend_capabilities import (
    action_contract_enabled,
    preflight_report,
    runtime_contract_payload,
)
from wam_harness.core.inference_trace import (
    inference_result_payload,
    observation_summary,
)
from wam_harness.core.memory import memory_snapshot
from wam_harness.core.preflight import PreflightError, assert_preflight
from wam_harness.core.registry import Registry, default_registry
from wam_harness.core.runtime import INPUT_RUN_SPEC, RUN_SPEC
from wam_harness.core.tracing import TraceWriter
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
        manifest = runtime_plan.manifest
        profiles = self.registry.build_optimization_profiles(manifest, enabled_opts or [])
        backend = self.registry.create_backend(manifest, profiles)
        workload, processor = self._create_workload(manifest, observation)

        defaults = manifest.defaults
        effective_action_horizon = int(
            action_horizon or defaults.get("action_horizon") or _default_horizon(manifest)
        )
        effective_replan_steps = int(
            replan_steps or defaults.get("replan_steps") or effective_action_horizon
        )
        if episode_length is not None:
            workload.episode_length = episode_length

        run_id = uuid.uuid4().hex[:12]
        output_dir = (Path(trace_dir) / run_id) if trace_dir is not None else Path("runs") / run_id
        trace_path = output_dir / "trace.jsonl"

        pending_actions: list[list[float]] = []
        model_calls = 0
        steps = 0
        last_result: InferenceResult | None = None
        runtime_info = backend.runtime_info()

        try:
            with TraceWriter(trace_path, run_id, runtime_info) as trace:
                trace.write(
                    "run_start",
                    mode=str(manifest.backend.get("mode", "run")),
                    output_dir=str(output_dir),
                    optimization_profiles=[profile.to_dict() for profile in profiles],
                    manifest_defaults=defaults,
                    known_gaps=manifest.known_gaps,
                    telemetry_config={"trace_format": "jsonl"},
                    synthetic_observation=manifest.workload_name == "processor_smoke",
                )
                try:
                    contract = runtime_contract_payload(
                        backend,
                        processor=processor,
                    )
                    if contract is not None:
                        trace.write("runtime_contract", **contract)
                    report = preflight_report(backend)
                    if report is not None:
                        trace.write("preflight", **report.to_trace_payload())
                    assert_preflight(report)
                    load_start = time.perf_counter()
                    trace.write("backend_load_start")
                    backend.load()
                    runtime_info = backend.runtime_info()
                    trace.set_runtime_info(runtime_info)
                    trace.write(
                        "backend_load",
                        timing={"total_ms": (time.perf_counter() - load_start) * 1000},
                        memory=memory_snapshot(),
                    )
                    start = time.perf_counter()
                    backend.warmup()
                    trace.write(
                        "backend_warmup",
                        timing={"total_ms": (time.perf_counter() - start) * 1000},
                        memory=memory_snapshot(),
                    )

                    workload.reset()
                    backend.reset()
                    trace.write("reset")
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
                                optimization_profiles=profiles,
                                runtime_options=runtime_options or {},
                            )
                            trace.write(
                                "inference_start",
                                episode_id=workload.episode_id,
                                step_id=workload.step_id,
                                replan_id=replan_id,
                            )
                            infer_start = time.perf_counter()
                            result = backend.infer(request)
                            last_result = result
                            infer_wall_ms = (time.perf_counter() - infer_start) * 1000
                            result_payload = inference_result_payload(
                                manifest,
                                result,
                                expected_horizon=effective_action_horizon,
                                wall_ms=infer_wall_ms,
                                validate_action_contract=action_contract_enabled(backend),
                            )
                            pending_actions = list(result.action_chunk.actions)
                            workload.mark_replan()
                            model_calls += 1
                            trace.write(
                                "inference_end",
                                episode_id=workload.episode_id,
                                step_id=workload.step_id,
                                replan_id=replan_id,
                                action_horizon=effective_action_horizon,
                                replan_steps=effective_replan_steps,
                                **result_payload,
                            )

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
                        trace_path=str(trace_path),
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
                        trace_path=str(trace_path),
                    )
                    raise
        finally:
            backend.close()

        return RunSummary(
            run_id=run_id,
            model_id=model_id,
            trace_path=trace_path,
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
    ) -> tuple[object, object | None]:
        if observation is not None:
            return (
                SingleObservationWorkload.from_observation(manifest, observation),
                self.registry.create_processor(manifest),
            )
        if manifest.workload_name == "processor_smoke":
            processor = self.registry.create_processor(manifest)
            return ProcessorSmokeWorkload.from_processor(manifest, processor), processor
        return self.registry.create_workload(manifest), None


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
