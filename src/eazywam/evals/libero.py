from __future__ import annotations

import importlib
import json
import os
import sys
import time
import types
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

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

LIBERO_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
    "libero_90": 700,
}

LIBERO_SINGLE_TASK_SPEC = RuntimeSpec(
    mode="simulator_eval",
    workload_name="libero_single_task",
    workload_config={"simulator": "LIBERO", "single_task": True},
    require_backend_mapping=True,
)


@dataclass(frozen=True)
class _EvalContext:
    task_suite_name: str
    task_id: int
    num_trials: int
    action_horizon: int
    replan_steps: int
    num_steps_wait: int
    max_steps: int
    seed: int | None
    num_inference_steps: int | None
    output_dir: Path
    cache_dir: Path
    values: dict[str, Any]


@dataclass(frozen=True)
class _LiberoModules:
    benchmark: Any
    get_libero_path: Any
    offscreen_render_env: Any


class LiberoSingleTaskEvalRunner:
    """Harness-owned single-task LIBERO simulator loop.

    This runner depends on the LIBERO simulator package, but it does not execute
    a FastWAM official evaluator script. The model call flows through the normal
    invocation/session/backend/processor spine used by ``wam run`` and
    ``wam serve``.
    """

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
            LIBERO_SINGLE_TASK_SPEC,
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
                overrides=overrides or {},
            )
            command = EvalCommand(
                argv=[
                    "wam-native-eval",
                    "libero-single-task",
                    f"model_id={model_id}",
                    f"task_suite_name={context.task_suite_name}",
                    f"task_id={context.task_id}",
                    f"num_trials={context.num_trials}",
                ],
                workdir=str(invocation.output_dir),
                env=_native_eval_env(context),
                display=(
                    "native LIBERO single-task eval "
                    f"{model_id} task_suite={context.task_suite_name} "
                    f"task_id={context.task_id} num_trials={context.num_trials}"
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
                eval_runner="libero_single_task",
                eval_workload=eval_workload,
                task_suite_name=context.task_suite_name,
                task_id=context.task_id,
                num_trials=context.num_trials,
                seed=context.seed,
                max_steps=context.max_steps,
                num_steps_wait=context.num_steps_wait,
                action_horizon=context.action_horizon,
                replan_steps=context.replan_steps,
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

            stage = "libero_import"
            try:
                _set_global_seed(context.seed)
                _prepare_libero_environment(context)
                modules = _import_libero_modules()
                task_suite, task, initial_states = _load_task(
                    modules,
                    context.task_suite_name,
                    context.task_id,
                    context.num_trials,
                )
                task_description = str(getattr(task, "language", ""))

                stage = "backend_start"
                invocation.start_backend(require_ready=True)
                runtime_info = invocation.runtime_info

                stage = "libero_eval"
                metrics = self._run_task(
                    invocation=invocation,
                    modules=modules,
                    task=task,
                    initial_states=initial_states,
                    task_description=task_description,
                    context=context,
                )
                metrics["task_suite"] = context.task_suite_name
                metrics["task_id"] = context.task_id
                metrics["task_description"] = task_description
                metrics["total_episodes"] = context.num_trials
                metrics["success_rate"] = (
                    float(metrics["successes"]) / float(context.num_trials)
                    if context.num_trials
                    else 0.0
                )
                metrics["results_path"] = str(_write_results(context, metrics))
                metrics["libero_task_suite_class"] = type(task_suite).__name__

                invocation.trace.write(
                    "native_eval_end",
                    eval_runner="libero_single_task",
                    task_suite_name=context.task_suite_name,
                    task_id=context.task_id,
                    successes=metrics["successes"],
                    total_episodes=context.num_trials,
                    success_rate=metrics["success_rate"],
                    results_path=metrics["results_path"],
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
                invocation.write_error(
                    exc=exc,
                    stage=stage,
                    recoverable=isinstance(exc, EvalRunnerError),
                    eval_workload=eval_workload,
                    eval_runner="libero_single_task",
                )
                invocation.write_finish(
                    "run_end",
                    status="error",
                    return_code=1,
                    warnings=[str(exc)],
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

    def _run_task(
        self,
        *,
        invocation: Invocation,
        modules: _LiberoModules,
        task: Any,
        initial_states: list[Any],
        task_description: str,
        context: _EvalContext,
    ) -> dict[str, object]:
        env, resolved_description = _create_libero_env(
            modules,
            task,
            context.seed,
        )
        if resolved_description:
            task_description = resolved_description

        successes = 0
        steps_total = 0
        model_calls_total = 0
        success_episodes: list[int] = []
        failure_episodes: list[int] = []
        episodes: list[dict[str, object]] = []
        start = time.perf_counter()
        try:
            for episode_idx in range(context.num_trials):
                episode = self._run_episode(
                    invocation=invocation,
                    env=env,
                    initial_state=initial_states[episode_idx],
                    task_description=task_description,
                    context=context,
                    episode_idx=episode_idx,
                )
                steps_total += int(episode["steps"])
                model_calls_total += int(episode["model_calls"])
                if bool(episode["success"]):
                    successes += 1
                    success_episodes.append(episode_idx)
                else:
                    failure_episodes.append(episode_idx)
                episodes.append(episode)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        return {
            "successes": successes,
            "success_episodes": success_episodes,
            "failure_episodes": failure_episodes,
            "steps": steps_total,
            "model_calls": model_calls_total,
            "episodes": episodes,
            "duration_s": time.perf_counter() - start,
        }

    def _run_episode(
        self,
        *,
        invocation: Invocation,
        env: Any,
        initial_state: Any,
        task_description: str,
        context: _EvalContext,
        episode_idx: int,
    ) -> dict[str, object]:
        session = invocation.session
        trace = invocation.trace
        env.reset()
        obs = env.set_init_state(initial_state)
        done = False
        model_calls = 0
        steps = 0
        pending_actions: list[list[float]] = []
        steps_since_replan = context.replan_steps

        trace.write(
            "episode_start",
            episode_id=episode_idx,
            task_suite_name=context.task_suite_name,
            task_id=context.task_id,
            task_description=task_description,
        )

        for wait_step in range(context.num_steps_wait):
            obs, _, done, _ = env.step(list(LIBERO_DUMMY_ACTION))
            trace.write(
                "simulator_wait_step",
                episode_id=episode_idx,
                wait_step_id=wait_step,
                done=bool(done),
            )
            if done:
                break

        while not done and steps < context.max_steps:
            if not pending_actions or steps_since_replan >= context.replan_steps:
                observation = _observation_from_libero(
                    obs,
                    task_description=task_description,
                    episode_idx=episode_idx,
                    step_id=steps,
                    task_suite_name=context.task_suite_name,
                    task_id=context.task_id,
                )
                replan_id = model_calls
                trace.write(
                    "replan_start",
                    episode_id=episode_idx,
                    step_id=steps,
                    replan_id=replan_id,
                    action_horizon=context.action_horizon,
                    replan_steps=context.replan_steps,
                    observation_summary=observation_summary(observation),
                    history_len=len(observation.history),
                )
                request = InferenceRequest(
                    observation=observation,
                    action_horizon=context.action_horizon,
                    replan_steps=context.replan_steps,
                    optimization_profiles=invocation.profiles,
                    reset=model_calls == 0,
                    runtime_options=_runtime_options(context),
                )
                trace.write(
                    "inference_start",
                    episode_id=episode_idx,
                    step_id=steps,
                    replan_id=replan_id,
                )
                result = session.infer_and_trace(
                    request,
                    event="inference_end",
                    expected_horizon=context.action_horizon,
                    payload={
                        "episode_id": episode_idx,
                        "step_id": steps,
                        "replan_id": replan_id,
                        "action_horizon": context.action_horizon,
                        "replan_steps": context.replan_steps,
                    },
                )
                pending_actions = [
                    [float(value) for value in row]
                    for row in result.action_chunk.actions[: context.replan_steps]
                ]
                if not pending_actions:
                    raise EvalRunnerError("native LIBERO eval received an empty action chunk")
                model_calls += 1
                steps_since_replan = 0

            action = pending_actions.pop(0)
            step_start = time.perf_counter()
            obs, _, done, info = env.step(action)
            env_step_ms = (time.perf_counter() - step_start) * 1000
            trace.write(
                "simulator_step",
                episode_id=episode_idx,
                step_id=steps,
                action=action,
                action_dim=len(action),
                pending_actions=len(pending_actions),
                done=bool(done),
                info=_json_safe(info),
                env_step_ms=env_step_ms,
            )
            steps += 1
            steps_since_replan += 1

        episode = {
            "episode_id": episode_idx,
            "success": bool(done),
            "steps": steps,
            "model_calls": model_calls,
        }
        trace.write("episode_end", **episode)
        return episode


def _build_context(
    *,
    manifest: Manifest,
    runtime_manifest: Manifest,
    eval_config: dict[str, Any],
    run_id: str,
    output_dir: Path,
    cache_dir: str | Path | None,
    overrides: dict[str, str],
) -> _EvalContext:
    values = _resolved_eval_values(
        manifest,
        eval_config,
        run_id=run_id,
        output_dir=output_dir,
        cache_dir=cache_dir,
        overrides=overrides,
    )
    task_suite_name = str(values.get("task_suite_name", values.get("suite", "libero_10")))
    task_id = _positive_int(values.get("task_id", 0), "task_id", allow_zero=True)
    num_trials = _positive_int(values.get("num_trials", 1), "num_trials")
    action_horizon = _positive_int(
        values.get("action_horizon", runtime_manifest.defaults.get("action_horizon", 1)),
        "action_horizon",
    )
    replan_steps = _positive_int(
        values.get("replan_steps", runtime_manifest.defaults.get("replan_steps", action_horizon)),
        "replan_steps",
    )
    num_steps_wait = _positive_int(
        values.get("num_steps_wait", 30),
        "num_steps_wait",
        allow_zero=True,
    )
    max_steps_value = values.get("max_steps")
    max_steps = (
        _positive_int(max_steps_value, "max_steps")
        if max_steps_value is not None
        else LIBERO_MAX_STEPS.get(task_suite_name, 700)
    )
    return _EvalContext(
        task_suite_name=task_suite_name,
        task_id=task_id,
        num_trials=num_trials,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_steps_wait=num_steps_wait,
        max_steps=max_steps,
        seed=_optional_int(values.get("seed")),
        num_inference_steps=_optional_int(values.get("num_inference_steps")),
        output_dir=output_dir,
        cache_dir=Path(str(values["cache_dir"])),
        values=values,
    )


def _resolved_eval_values(
    manifest: Manifest,
    eval_config: dict[str, Any],
    *,
    run_id: str,
    output_dir: Path,
    cache_dir: str | Path | None,
    overrides: dict[str, str],
) -> dict[str, Any]:
    defaults = manifest.eval.get("defaults", {})
    if defaults and not isinstance(defaults, dict):
        raise EvalRunnerError(f"{manifest.id} eval.defaults must be a mapping")
    workload_defaults = eval_config.get("defaults", {})
    if workload_defaults and not isinstance(workload_defaults, dict):
        raise EvalRunnerError(f"{manifest.id} native eval workload defaults must be a mapping")

    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    values: dict[str, Any] = {
        "model_id": manifest.id,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "trace_dir": str(output_dir.parent),
        "cache_dir": str(resolved_cache_dir),
        "suite": manifest.eval.get("suite", "libero_10"),
        "torch_force_no_weights_only_load": "1",
        "tokenizers_parallelism": "false",
        "wandb_mode": "offline",
    }
    values.update(defaults if isinstance(defaults, dict) else {})
    values.update(workload_defaults if isinstance(workload_defaults, dict) else {})
    values.update(overrides)
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
    env = {}
    for key in (
        "LIBERO_CONFIG_PATH",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "TOKENIZERS_PARALLELISM",
        "WANDB_MODE",
    ):
        value = context.values.get(key.lower()) or context.values.get(key)
        if value is not None:
            env[key] = str(value)
    return env


def _prepare_libero_environment(context: _EvalContext) -> None:
    env_defaults = {
        "LIBERO_CONFIG_PATH": context.values.get("libero_config_path"),
        "MUJOCO_GL": context.values.get("mujoco_gl"),
        "PYOPENGL_PLATFORM": context.values.get("pyopengl_platform"),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": context.values.get(
            "torch_force_no_weights_only_load",
            "1",
        ),
        "TOKENIZERS_PARALLELISM": context.values.get("tokenizers_parallelism", "false"),
        "WANDB_MODE": context.values.get("wandb_mode", "offline"),
    }
    for key, value in env_defaults.items():
        if value is not None:
            os.environ.setdefault(key, str(value))

    pythonpath = context.values.get("libero_pythonpath")
    if pythonpath:
        path = Path(str(pythonpath)).expanduser()
        if path.exists():
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)


def _set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    try:
        pytorch_utils = importlib.import_module("fastwam.utils.pytorch_utils")
    except ModuleNotFoundError as exc:
        raise EvalRunnerError(
            "native LIBERO eval could not import fastwam.utils.pytorch_utils "
            "to match the official FastWAM seed setup."
        ) from exc
    set_global_seed = getattr(pytorch_utils, "set_global_seed", None)
    if not callable(set_global_seed):
        raise EvalRunnerError(
            "fastwam.utils.pytorch_utils does not expose set_global_seed()."
        )
    set_global_seed(int(seed), get_worker_init_fn=False)


def _import_libero_modules() -> _LiberoModules:
    try:
        return _load_libero_modules()
    except ModuleNotFoundError:
        try:
            _install_libero_namespace_alias()
            return _load_libero_modules()
        except (ModuleNotFoundError, ImportError) as alias_exc:
            raise EvalRunnerError(
                "native LIBERO eval requires the `libero` simulator package. "
                "Install LIBERO in the active environment or set eval.defaults.libero_pythonpath "
                "to a prepared LIBERO checkout."
            ) from alias_exc
    except ImportError as exc:
        raise EvalRunnerError(
            "native LIBERO eval requires the `libero` simulator package. "
            "Install LIBERO in the active environment or set eval.defaults.libero_pythonpath "
            "to a prepared LIBERO checkout."
        ) from exc


def _load_libero_modules() -> _LiberoModules:
    libero_root = importlib.import_module("libero.libero")
    benchmark = importlib.import_module("libero.libero.benchmark")
    envs = importlib.import_module("libero.libero.envs")

    get_libero_path = getattr(libero_root, "get_libero_path", None)
    offscreen_render_env = getattr(envs, "OffScreenRenderEnv", None)
    if not callable(get_libero_path) or offscreen_render_env is None:
        raise EvalRunnerError("LIBERO package does not expose the expected simulator API")
    return _LiberoModules(
        benchmark=benchmark,
        get_libero_path=get_libero_path,
        offscreen_render_env=offscreen_render_env,
    )


def _install_libero_namespace_alias() -> None:
    """Recover LIBERO installs that expose the inner package as top-level `libero`.

    Some editable/self-managed installs put ``LIBERO/libero`` on ``sys.path``.
    That makes ``import libero`` resolve to ``LIBERO/libero/libero`` while
    LIBERO source files still import ``libero.libero.*``. Rebuilding the outer
    namespace in ``sys.modules`` matches the layout obtained when the LIBERO
    repository root is on ``sys.path``.
    """

    plain = importlib.import_module("libero")
    package_file = getattr(plain, "__file__", None)
    if package_file is None or not hasattr(plain, "get_libero_path"):
        return

    inner_dir = Path(str(package_file)).resolve().parent
    if not (inner_dir / "benchmark").exists() or not (inner_dir / "envs").exists():
        return
    outer_dir = inner_dir.parent

    for name in list(sys.modules):
        if name == "libero" or name.startswith("libero."):
            del sys.modules[name]

    outer = types.ModuleType("libero")
    outer.__file__ = str(outer_dir / "__init__.py")
    outer.__package__ = "libero"
    outer.__path__ = [str(outer_dir)]  # type: ignore[attr-defined]
    sys.modules["libero"] = outer


def _load_task(
    modules: _LiberoModules,
    task_suite_name: str,
    task_id: int,
    num_trials: int,
) -> tuple[Any, Any, list[Any]]:
    benchmark_dict = modules.benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        known = ", ".join(sorted(str(name) for name in benchmark_dict))
        raise EvalRunnerError(
            f"unknown LIBERO task suite '{task_suite_name}'; known suites: {known}"
        )
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = list(task_suite.get_task_init_states(task_id))
    if not initial_states:
        raise EvalRunnerError(
            f"LIBERO task suite '{task_suite_name}' task {task_id} returned no init states"
        )
    while len(initial_states) < num_trials:
        remaining = num_trials - len(initial_states)
        initial_states.extend(initial_states[:remaining])
    return task_suite, task, initial_states[:num_trials]


def _create_libero_env(
    modules: _LiberoModules,
    task: Any,
    seed: int | None,
) -> tuple[Any, str]:
    task_description = str(getattr(task, "language", ""))
    task_bddl_file = (
        Path(str(modules.get_libero_path("bddl_files")))
        / str(getattr(task, "problem_folder"))
        / str(getattr(task, "bddl_file"))
    )
    env = modules.offscreen_render_env(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        seed_fn(seed)
    return env, task_description


def _observation_from_libero(
    obs: dict[str, Any],
    *,
    task_description: str,
    episode_idx: int,
    step_id: int,
    task_suite_name: str,
    task_id: int,
) -> Observation:
    missing = [
        key
        for key in (
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
        if key not in obs
    ]
    if missing:
        raise EvalRunnerError(
            "LIBERO observation is missing required key(s): " + ", ".join(missing)
        )
    return Observation(
        images={
            "agentview_image": obs["agentview_image"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
        },
        state={
            "robot0_eef_pos": obs["robot0_eef_pos"],
            "robot0_eef_quat": obs["robot0_eef_quat"],
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
        },
        prompt=task_description,
        session={
            "episode_id": episode_idx,
            "step_id": step_id,
            "task_suite_name": task_suite_name,
            "task_id": task_id,
        },
        metadata={
            "task": task_description,
            "task_suite": task_suite_name,
            "task_id": task_id,
        },
    )


def _runtime_options(context: _EvalContext) -> dict[str, object]:
    options: dict[str, object] = {}
    if context.num_inference_steps is not None:
        options["num_inference_steps"] = context.num_inference_steps
    for key in ("dit_cache_mode", "cuda_graph_mode", "torch_compile_mode"):
        if context.values.get(key) is not None:
            options[key] = context.values[key]
    return options


def _write_results(context: _EvalContext, metrics: dict[str, object]) -> Path:
    output_dir = context.output_dir / "native_eval" / context.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"task{context.task_id}_results.json"
    output_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    return output_path


def _eval_metadata(manifest: Manifest, workload: str | None) -> dict[str, object]:
    return {
        "workload": workload,
        "simulator": manifest.eval.get("simulator"),
        "suite": manifest.eval.get("suite"),
        "native_eval_runner": "libero_single_task",
    }


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise EvalRunnerError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0 or (parsed == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise EvalRunnerError(f"{name} must be {qualifier}, got {parsed}")
    return parsed


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return _positive_int(value, "optional integer", allow_zero=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
