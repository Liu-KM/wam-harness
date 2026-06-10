from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eazywam.core._utils import default_cache_dir
from eazywam.core.eval_runner import EvalCommand, EvalRunnerError, EvalSummary
from eazywam.core.inference_trace import observation_summary
from eazywam.core.invocation import Invocation
from eazywam.core.registry import Registry
from eazywam.core.runtime import RuntimeSpec
from eazywam.core.types import InferenceRequest, Manifest, Observation

ROBOTWIN_SINGLE_TASK_SPEC = RuntimeSpec(
    mode="simulator_eval",
    workload_name="robotwin_single_task",
    workload_config={"simulator": "RoboTwin", "single_task": True},
    require_backend_mapping=True,
)


@dataclass(frozen=True)
class _EvalContext:
    task_name: str
    task_config: str
    instruction_type: str
    num_episodes: int
    action_horizon: int
    replan_steps: int
    seed: int | None
    output_dir: Path
    cache_dir: Path
    robotwin_root: Path
    values: dict[str, Any]


class RobotWinSingleTaskEvalRunner:
    """Harness-owned RoboTwin single-task eval through RoboTwin's policy plugin API."""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def run(
        self,
        *,
        model_id: str,
        reference_manifest: Manifest,
        eval_workload: str | None,
        eval_config: dict[str, Any],
        enabled_opts: list[str] | None = None,
        trace_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        upstream_dir: str | Path | None = None,
        dry_run: bool = False,
        overrides: dict[str, str] | None = None,
    ) -> EvalSummary:
        runtime_plan = self.registry.resolve_runtime(
            reference_manifest,
            ROBOTWIN_SINGLE_TASK_SPEC,
            upstream_dir=upstream_dir,
            cache_dir=cache_dir,
        )
        invocation = Invocation.from_runtime_plan(
            registry=self.registry,
            model_id=model_id,
            runtime_plan=runtime_plan,
            enabled_opts=enabled_opts,
            trace_dir=trace_dir,
        )
        try:
            context = _build_context(
                manifest=reference_manifest,
                runtime_manifest=invocation.manifest,
                eval_config=eval_config,
                run_id=invocation.run_id,
                output_dir=invocation.output_dir,
                cache_dir=cache_dir,
                upstream_dir=upstream_dir,
                overrides=overrides or {},
            )
            command = EvalCommand(
                argv=[
                    "wam-native-eval",
                    "robotwin-single-task",
                    f"model_id={model_id}",
                    f"task_name={context.task_name}",
                    f"num_episodes={context.num_episodes}",
                ],
                workdir=str(context.robotwin_root),
                env=_native_eval_env(context),
                display=(
                    "native RoboTwin single-task eval "
                    f"{model_id} task={context.task_name} "
                    f"episodes={context.num_episodes}"
                ),
            )
        except Exception:
            invocation.close()
            raise

        status = "planned" if dry_run else "ok"
        return_code: int | None = None if dry_run else 0
        metrics: dict[str, object] = {}
        runtime_info = invocation.runtime_info

        with invocation:
            invocation.write_start(
                "run_start",
                mode="simulator_eval",
                telemetry_config={"trace_format": "jsonl"},
                dry_run=dry_run,
                reference_eval=False,
                native_eval=True,
                eval_workload=eval_workload,
                eval_metadata=_eval_metadata(reference_manifest, eval_workload),
            )
            invocation.trace.write(
                "native_eval_plan",
                eval_runner="robotwin_single_task",
                eval_workload=eval_workload,
                task_name=context.task_name,
                task_config=context.task_config,
                instruction_type=context.instruction_type,
                num_episodes=context.num_episodes,
                seed=context.seed,
                action_horizon=context.action_horizon,
                replan_steps=context.replan_steps,
                robotwin_root=str(context.robotwin_root),
                command=command.to_dict(),
            )
            if dry_run:
                invocation.write_finish(
                    "run_end",
                    status="planned",
                    return_code=None,
                    warnings=[],
                )
                return EvalSummary(
                    run_id=invocation.run_id,
                    model_id=model_id,
                    workload=eval_workload,
                    trace_path=invocation.trace_path,
                    status="planned",
                    command=command,
                    return_code=None,
                    stdout_path=None,
                    stderr_path=None,
                    runtime_info=runtime_info,
                    metrics=metrics,
                )

            stage = "backend_start"
            try:
                _prepare_robotwin_environment(context)
                invocation.start_backend(require_ready=True, stage_callback=lambda value: None)
                runtime_info = invocation.runtime_info

                stage = "robotwin_import"
                eval_policy = _import_robotwin_eval_policy(context.robotwin_root)
                policy_name = f"eazywam_robotwin_policy_{invocation.run_id}"
                policy = _EazyWAMRobotWinPolicy(
                    invocation=invocation,
                    context=context,
                )
                policy_module = _install_policy_module(policy_name, policy)

                stage = "robotwin_eval"
                try:
                    result = _run_robotwin_eval_policy(
                        eval_policy,
                        context,
                        policy_name,
                    )
                finally:
                    sys.modules.pop(policy_name, None)
                    # Keep a local reference alive until after eval_policy.main returns.
                    del policy_module

                metrics = _collect_metrics(context, result, policy_name=policy_name)
                metrics["task_name"] = context.task_name
                metrics["task_config"] = context.task_config
                metrics["instruction_type"] = context.instruction_type
                metrics["requested_episodes"] = context.num_episodes
                metrics["valid_episodes"] = context.num_episodes
                metrics["invalid_setup_count"] = 0
                metrics["invalid_setups"] = []
                metrics["total_episodes"] = context.num_episodes
                metrics["success_rate"] = (
                    float(metrics["successes"]) / float(context.num_episodes)
                    if context.num_episodes
                    else 0.0
                )

                invocation.trace.write(
                    "native_eval_end",
                    eval_runner="robotwin_single_task",
                    task_name=context.task_name,
                    task_config=context.task_config,
                    successes=metrics["successes"],
                    total_episodes=context.num_episodes,
                    requested_episodes=context.num_episodes,
                    valid_episodes=context.num_episodes,
                    invalid_setup_count=0,
                    success_rate=metrics["success_rate"],
                    results_path=metrics.get("results_path"),
                )
                invocation.write_finish(
                    "run_end",
                    status="ok",
                    return_code=0,
                    warnings=[],
                )
            except Exception as exc:
                status = "error"
                return_code = 1
                invalid_setup = _invalid_setup_error_payload(exc, context=context, stage=stage)
                invocation.write_error(
                    exc=exc,
                    stage=stage,
                    recoverable=isinstance(exc, EvalRunnerError),
                    eval_workload=eval_workload,
                    eval_runner="robotwin_single_task",
                    simulator_setup=invalid_setup,
                )
                invocation.write_finish(
                    "run_end",
                    status="error",
                    return_code=1,
                    warnings=_error_warnings(exc, invalid_setup=invalid_setup),
                )
                raise

        return EvalSummary(
            run_id=invocation.run_id,
            model_id=model_id,
            workload=eval_workload,
            trace_path=invocation.trace_path,
            status=status,
            command=command,
            return_code=return_code,
            stdout_path=None,
            stderr_path=None,
            runtime_info=runtime_info,
            metrics=metrics,
        )


class _EazyWAMRobotWinPolicy:
    def __init__(self, *, invocation: Invocation, context: _EvalContext) -> None:
        self.invocation = invocation
        self.context = context
        self.pending_actions: deque[list[float]] = deque()
        self.episode_id = -1
        self.step_id = 0
        self.model_calls = 0

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_id += 1
        self.step_id = 0
        self.model_calls = 0
        self.invocation.trace.write(
            "episode_start",
            episode_id=self.episode_id,
            task_name=self.context.task_name,
            task_config=self.context.task_config,
        )

    def step(self, task_env: Any, observation: dict[str, Any] | None) -> None:
        if not self.pending_actions:
            if observation is None:
                raise EvalRunnerError(
                    "native RoboTwin eval needs an observation at replan time"
                )
            instruction = str(task_env.get_instruction())
            harness_observation = _observation_from_robotwin(
                observation,
                instruction=instruction,
                episode_id=self.episode_id,
                step_id=self.step_id,
                task_name=self.context.task_name,
            )
            replan_id = self.model_calls
            self.invocation.trace.write(
                "replan_start",
                episode_id=self.episode_id,
                step_id=self.step_id,
                replan_id=replan_id,
                action_horizon=self.context.action_horizon,
                replan_steps=self.context.replan_steps,
                observation_summary=observation_summary(harness_observation),
                history_len=len(harness_observation.history),
            )
            request = InferenceRequest(
                observation=harness_observation,
                action_horizon=self.context.action_horizon,
                replan_steps=self.context.replan_steps,
                optimization_profiles=self.invocation.profiles,
                reset=self.model_calls == 0,
                runtime_options=_runtime_options(self.context),
            )
            self.invocation.trace.write(
                "inference_start",
                episode_id=self.episode_id,
                step_id=self.step_id,
                replan_id=replan_id,
            )
            result = self.invocation.session.infer_and_trace(
                request,
                event="inference_end",
                expected_horizon=self.context.action_horizon,
                payload={
                    "episode_id": self.episode_id,
                    "step_id": self.step_id,
                    "replan_id": replan_id,
                    "action_horizon": self.context.action_horizon,
                    "replan_steps": self.context.replan_steps,
                },
            )
            for row in result.action_chunk.actions[: self.context.replan_steps]:
                self.pending_actions.append([float(value) for value in row])
            if not self.pending_actions:
                raise EvalRunnerError("native RoboTwin eval received an empty action chunk")
            self.model_calls += 1

        action = self.pending_actions.popleft()
        start = time.perf_counter()
        task_env.take_action(action, action_type="qpos")
        self.invocation.trace.write(
            "simulator_step",
            episode_id=self.episode_id,
            step_id=self.step_id,
            action=action,
            action_dim=len(action),
            pending_actions=len(self.pending_actions),
            done=bool(getattr(task_env, "eval_success", False)),
            env_step_ms=(time.perf_counter() - start) * 1000,
        )
        self.step_id += 1


def _install_policy_module(name: str, policy: _EazyWAMRobotWinPolicy) -> types.ModuleType:
    module = types.ModuleType(name)
    module.get_model = lambda usr_args: policy
    module.eval = lambda task_env, model, observation: model.step(task_env, observation)
    module.reset_model = lambda model: model.reset()
    sys.modules[name] = module
    return module


def _build_context(
    *,
    manifest: Manifest,
    runtime_manifest: Manifest,
    eval_config: dict[str, Any],
    run_id: str,
    output_dir: Path,
    cache_dir: str | Path | None,
    upstream_dir: str | Path | None,
    overrides: dict[str, str],
) -> _EvalContext:
    values = _resolved_eval_values(
        manifest,
        eval_config,
        run_id=run_id,
        output_dir=output_dir,
        cache_dir=cache_dir,
        upstream_dir=upstream_dir,
        overrides=overrides,
    )
    task_name = str(values.get("task_name", ""))
    if not task_name or task_name.lower() in {"none", "null"}:
        raise EvalRunnerError("native RoboTwin eval requires task_name")
    task_config = str(values.get("task_config", "demo_randomized"))
    instruction_type = str(values.get("instruction_type", "unseen"))
    num_episodes = _positive_int(
        values.get("num_episodes", values.get("eval_num_episodes", 1)),
        "num_episodes",
    )
    action_horizon_value = _optional_int(values.get("action_horizon"))
    action_horizon = (
        action_horizon_value
        if action_horizon_value is not None
        else _positive_int(runtime_manifest.defaults.get("action_horizon", 1), "action_horizon")
    )
    replan_steps_value = _optional_int(values.get("replan_steps"))
    replan_steps = (
        replan_steps_value
        if replan_steps_value is not None
        else _positive_int(
            runtime_manifest.defaults.get("replan_steps", action_horizon),
            "replan_steps",
        )
    )
    robotwin_root = Path(str(values["robotwin_root"])).expanduser().resolve()
    return _EvalContext(
        task_name=task_name,
        task_config=task_config,
        instruction_type=instruction_type,
        num_episodes=num_episodes,
        action_horizon=action_horizon,
        replan_steps=max(1, min(replan_steps, action_horizon)),
        seed=_optional_int(values.get("seed")),
        output_dir=output_dir,
        cache_dir=Path(str(values["cache_dir"])),
        robotwin_root=robotwin_root,
        values=values,
    )


def _resolved_eval_values(
    manifest: Manifest,
    eval_config: dict[str, Any],
    *,
    run_id: str,
    output_dir: Path,
    cache_dir: str | Path | None,
    upstream_dir: str | Path | None,
    overrides: dict[str, str],
) -> dict[str, Any]:
    defaults = manifest.eval.get("defaults", {})
    if defaults and not isinstance(defaults, dict):
        raise EvalRunnerError(f"{manifest.id} eval.defaults must be a mapping")
    workload_defaults = eval_config.get("defaults", {})
    if workload_defaults and not isinstance(workload_defaults, dict):
        raise EvalRunnerError(f"{manifest.id} native eval workload defaults must be a mapping")

    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    resolved_upstream_dir = Path(upstream_dir) if upstream_dir is not None else Path("/workspace/FastWAM")
    values: dict[str, Any] = {
        "model_id": manifest.id,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "trace_dir": str(output_dir.parent),
        "cache_dir": str(resolved_cache_dir),
        "upstream_dir": str(resolved_upstream_dir),
        "torch_force_no_weights_only_load": "1",
        "tokenizers_parallelism": "false",
        "wandb_mode": "offline",
    }
    values.update(defaults if isinstance(defaults, dict) else {})
    values.update(workload_defaults if isinstance(workload_defaults, dict) else {})
    values.update(overrides)
    if "num_episodes" not in values and "eval_num_episodes" in values:
        values["num_episodes"] = values["eval_num_episodes"]
    return _resolve_templates(manifest, values)


def _resolve_templates(manifest: Manifest, values: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(values)
    for _ in range(5):
        changed = False
        formatter = _StrictFormatDict({key: str(value) for key, value in resolved.items()})
        next_values: dict[str, Any] = {}
        for key, value in resolved.items():
            if isinstance(value, str):
                try:
                    rendered = value.format_map(formatter)
                except EvalRunnerError as exc:
                    raise EvalRunnerError(
                        f"{manifest.id} native eval value '{key}' could not be rendered: {exc}"
                    ) from exc
                next_values[key] = rendered
                changed = changed or rendered != value
            else:
                next_values[key] = value
        resolved = next_values
        if not changed:
            return resolved
    raise EvalRunnerError(f"{manifest.id} native eval context contains recursive templates")


class _StrictFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise EvalRunnerError(f"missing native eval template value: {key}") from None


def _native_eval_env(context: _EvalContext) -> dict[str, str]:
    env = {
        "WAM_ROBOTWIN_ROOT": str(context.robotwin_root),
        "TOKENIZERS_PARALLELISM": str(context.values.get("tokenizers_parallelism", "false")),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": str(
            context.values.get("torch_force_no_weights_only_load", "1")
        ),
        "WANDB_MODE": str(context.values.get("wandb_mode", "offline")),
    }
    for key in (
        "DIFFSYNTH_DOWNLOAD_SOURCE",
        "DIFFSYNTH_MODEL_BASE_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
    ):
        value_key = key.lower()
        if key == "MPLCONFIGDIR":
            value_key = "matplotlib_config_dir"
        elif key == "XDG_CACHE_HOME":
            value_key = "xdg_cache_home"
        value = context.values.get(value_key) or context.values.get(key)
        if value is not None:
            env[key] = str(value)
    return env


def _prepare_robotwin_environment(context: _EvalContext) -> None:
    if not context.robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {context.robotwin_root}")
    for key, value in _native_eval_env(context).items():
        os.environ.setdefault(key, value)


def _import_robotwin_eval_policy(robotwin_root: Path) -> Any:
    inserted = [str(robotwin_root), str(robotwin_root / "policy"), str(robotwin_root / "description" / "utils")]
    for value in reversed(inserted):
        if value not in sys.path:
            sys.path.insert(0, value)
    cwd = Path.cwd()
    try:
        os.chdir(robotwin_root)
        return importlib.import_module("script.eval_policy")
    finally:
        os.chdir(cwd)


def _run_robotwin_eval_policy(
    eval_policy: Any,
    context: _EvalContext,
    policy_name: str,
) -> Any:
    cwd = Path.cwd()
    original_eval_policy = getattr(eval_policy, "eval_policy", None)
    if callable(original_eval_policy):

        def limited_eval_policy(*args: Any, **kwargs: Any) -> Any:
            if len(args) >= 6:
                args = (*args[:5], context.num_episodes, *args[6:])
            kwargs["test_num"] = context.num_episodes
            return original_eval_policy(*args, **kwargs)

        setattr(eval_policy, "eval_policy", limited_eval_policy)
    try:
        os.chdir(context.robotwin_root)
        return eval_policy.main(_robotwin_user_args(context, policy_name))
    finally:
        os.chdir(cwd)
        if callable(original_eval_policy):
            setattr(eval_policy, "eval_policy", original_eval_policy)


def _robotwin_user_args(context: _EvalContext, policy_name: str) -> dict[str, Any]:
    values = context.values
    eval_output_dir = Path(str(values.get("eval_output_dir", context.output_dir / "fastwam_robotwin")))
    return {
        "policy_name": policy_name,
        "task_name": context.task_name,
        "task_config": context.task_config,
        "ckpt_setting": str(values.get("checkpoint_path", values.get("ckpt", "eazywam-native"))),
        "seed": int(context.seed or 0),
        "instruction_type": context.instruction_type,
        "eval_num_episodes": context.num_episodes,
        "eval_output_dir": str(eval_output_dir),
        "skip_get_obs_within_replan": _bool_value(
            values.get("skip_get_obs_within_replan", True)
        ),
    }


def _observation_from_robotwin(
    raw_observation: dict[str, Any],
    *,
    instruction: str,
    episode_id: int,
    step_id: int,
    task_name: str,
) -> Observation:
    images = raw_observation.get("observation", raw_observation)
    state = {}
    if isinstance(raw_observation.get("joint_action"), dict):
        state["joint_action"] = raw_observation["joint_action"]
    elif "joint_action" in raw_observation:
        state["joint_action"] = {"vector": raw_observation["joint_action"]}
    elif "proprio" in raw_observation:
        state["proprio"] = raw_observation["proprio"]
    return Observation(
        images=dict(images),
        state=state,
        prompt=instruction,
        session={
            "episode_id": episode_id,
            "step_id": step_id,
            "task_name": task_name,
            "simulator": "RoboTwin",
        },
        metadata={"task": task_name, "simulator": "RoboTwin"},
    )


def _runtime_options(context: _EvalContext) -> dict[str, object]:
    options: dict[str, object] = {}
    for key in (
        "num_inference_steps",
        "dit_cache_mode",
        "cuda_graph_mode",
        "torch_compile_mode",
        "sigma_shift",
        "text_cfg_scale",
        "negative_prompt",
        "rand_device",
        "tiled",
    ):
        value = context.values.get(key)
        if value is not None:
            options[key] = value
    return options


def _collect_metrics(
    context: _EvalContext,
    result: object,
    *,
    policy_name: str,
) -> dict[str, object]:
    if isinstance(result, dict) and "successes" in result:
        successes = int(result["successes"])
        return {"successes": successes}
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        try:
            successes = int(result[1])
        except (TypeError, ValueError):
            pass
        else:
            return {"successes": successes}

    result_path = _result_path(context, policy_name=policy_name)
    if result_path.exists():
        success_rate = _parse_success_rate(result_path)
        successes = round(success_rate * context.num_episodes)
        return {
            "successes": int(successes),
            "success_rate": success_rate,
            "results_path": str(result_path),
        }
    raise EvalRunnerError(
        "native RoboTwin eval completed but no result summary was returned or written: "
        f"{result_path}"
    )


def _invalid_setup_error_payload(
    exc: Exception,
    *,
    context: _EvalContext,
    stage: str,
) -> dict[str, object] | None:
    if stage != "robotwin_eval":
        return None
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).lower()
    if (
        isinstance(exc, IndexError)
        and "list index out of range" in text
        and context.task_name == "put_bottles_dustbin"
    ):
        return {
            "category": "simulator_setup_invalid",
            "policy_failure": False,
            "reason": "put_bottles_dustbin_expert_setup_index_error",
            "message": (
                "RoboTwin crashed during upstream expert/demo initialization before "
                "policy rollout."
            ),
            "task_name": context.task_name,
            "task_config": context.task_config,
            "requested_episodes": context.num_episodes,
            "valid_episodes": 0,
        }
    if "unstableerror" in text or "unstable error" in text:
        return {
            "category": "simulator_setup_invalid",
            "policy_failure": False,
            "reason": "robotwin_unstable_setup",
            "message": "RoboTwin marked the sampled simulator setup as unstable.",
            "task_name": context.task_name,
            "task_config": context.task_config,
            "requested_episodes": context.num_episodes,
            "valid_episodes": 0,
        }
    return None


def _error_warnings(
    exc: Exception,
    *,
    invalid_setup: dict[str, object] | None,
) -> list[str]:
    if invalid_setup is None:
        return [str(exc)]
    return [
        str(exc),
        (
            "simulator setup was classified as invalid before policy rollout; "
            "do not count it as a policy failure"
        ),
    ]


def _result_path(context: _EvalContext, *, policy_name: str) -> Path:
    explicit_dir = Path(
        str(context.values.get("eval_output_dir", context.output_dir / "fastwam_robotwin"))
    )
    explicit_path = explicit_dir / "_result.txt"
    if explicit_path.exists():
        return explicit_path

    result_root = (
        context.robotwin_root
        / "eval_result"
        / context.task_name
        / policy_name
        / context.task_config
    )
    candidates = sorted(
        result_root.rglob("_result.txt") if result_root.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    return explicit_path


def _parse_success_rate(path: Path) -> float:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            continue
    raise EvalRunnerError(f"Could not parse RoboTwin success rate from {path}")


def _eval_metadata(manifest: Manifest, eval_workload: str | None) -> dict[str, object]:
    return {
        "simulator": manifest.eval.get("simulator"),
        "suite": manifest.eval.get("suite"),
        "workload": eval_workload,
        "native_eval_runner": "robotwin_single_task",
    }


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    parsed = int(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise EvalRunnerError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return parsed


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return int(text)


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(value)
