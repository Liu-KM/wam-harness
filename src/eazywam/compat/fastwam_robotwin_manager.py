from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import yaml

PHASES = {
    "clean": "demo_clean",
    "random": "demo_randomized",
}


@dataclass(frozen=True)
class RunResult:
    return_code: int
    output_tail: str


@dataclass(frozen=True)
class InvalidSetupClassification:
    reason: str
    message: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    upstream_dir = args.upstream_dir.resolve()
    robotwin_root = args.robotwin_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _tasks(args.task_name, robotwin_root)
    num_episodes = _positive_int(args.num_episodes, "--num-episodes")
    max_worker_restarts_on_invalid_setup = _nonnegative_int(
        args.max_worker_restarts_on_invalid_setup or "0",
        "--max-worker-restarts-on-invalid-setup",
    )
    summary: dict[str, dict[str, float | None]] = {
        task: {phase: None for phase in PHASES} for task in tasks
    }
    failures: list[dict[str, object]] = []
    invalid_setups: list[dict[str, object]] = []
    candidate_summaries: list[dict[str, object]] = []
    if _bool_value(args.resume):
        _load_existing_summary(output_dir, summary)
        _load_existing_robotwin_results(robotwin_root, summary)
        invalid_setups = _load_existing_invalid_setups(output_dir)
        candidate_summaries = _load_existing_candidate_summaries(output_dir)

    for task_name in tasks:
        for phase, task_config in PHASES.items():
            if summary[task_name][phase] is not None:
                print(
                    "[eazywam-robotwin-manager] "
                    f"skip completed task={task_name} phase={phase} "
                    f"success_rate={summary[task_name][phase]:.4f}",
                    flush=True,
                )
                continue
            invalid_attempts = 0
            while True:
                seed = _seed_for_attempt(args.seed, invalid_attempts)
                started_at = time.time()
                cmd = _single_eval_command(
                    args=args,
                    upstream_dir=upstream_dir,
                    output_dir=output_dir,
                    task_name=task_name,
                    task_config=task_config,
                    seed=seed,
                )
                print(
                    "[eazywam-robotwin-manager] "
                    f"launch task={task_name} phase={phase} seed={seed}",
                    flush=True,
                )
                run_result = _coerce_run_result(_run_streaming(cmd, cwd=upstream_dir))
                if run_result.return_code != 0:
                    classification = _classify_invalid_setup_output(run_result.output_tail)
                    if classification is not None:
                        invalid_entry = {
                            "task_name": task_name,
                            "phase": phase,
                            "task_config": task_config,
                            "attempt": invalid_attempts + 1,
                            "seed": seed,
                            "return_code": run_result.return_code,
                            "category": "simulator_setup_invalid",
                            "policy_failure": False,
                            "reason": classification.reason,
                            "message": classification.message,
                            "output_tail": _output_excerpt(run_result.output_tail),
                        }
                        invalid_setups.append(invalid_entry)
                        _write_summary(
                            output_dir,
                            summary,
                            failures,
                            invalid_setups=invalid_setups,
                            candidate_summaries=candidate_summaries,
                            num_episodes=num_episodes,
                        )
                        if invalid_attempts < max_worker_restarts_on_invalid_setup:
                            print(
                                "[eazywam-robotwin-manager] "
                                f"invalid simulator setup task={task_name} phase={phase} "
                                f"seed={seed} reason={classification.reason}; "
                                "worker_restart "
                                f"{invalid_attempts + 1}/"
                                f"{max_worker_restarts_on_invalid_setup}",
                                flush=True,
                            )
                            invalid_attempts += 1
                            continue

                        failures.append(
                            {
                                "task_name": task_name,
                                "phase": phase,
                                "return_code": run_result.return_code,
                                "category": "simulator_setup_invalid",
                                "policy_failure": False,
                                "reason": "invalid_setup_exhausted",
                                "invalid_setup_reason": classification.reason,
                                "invalid_setup_attempts": invalid_attempts + 1,
                            }
                        )
                        _write_summary(
                            output_dir,
                            summary,
                            failures,
                            invalid_setups=invalid_setups,
                            candidate_summaries=candidate_summaries,
                            num_episodes=num_episodes,
                        )
                        return run_result.return_code

                    failures.append(
                        {
                            "task_name": task_name,
                            "phase": phase,
                            "return_code": run_result.return_code,
                            "category": "worker_failed",
                            "policy_failure": None,
                            "reason": "worker_failed",
                            "output_tail": _output_excerpt(run_result.output_tail),
                        }
                    )
                    _write_summary(
                        output_dir,
                        summary,
                        failures,
                        invalid_setups=invalid_setups,
                        candidate_summaries=candidate_summaries,
                        num_episodes=num_episodes,
                    )
                    return run_result.return_code

                output_invalid_setups = _invalid_setup_entries_from_output(
                    run_result.output_tail,
                    task_name=task_name,
                    phase=phase,
                    task_config=task_config,
                    seed=seed,
                    attempt=invalid_attempts + 1,
                    return_code=run_result.return_code,
                )
                if output_invalid_setups:
                    invalid_setups.extend(output_invalid_setups)
                    _write_summary(
                        output_dir,
                        summary,
                        failures,
                        invalid_setups=invalid_setups,
                        candidate_summaries=candidate_summaries,
                        num_episodes=num_episodes,
                    )

                result_path = _latest_result_path(
                    robotwin_root=robotwin_root,
                    task_name=task_name,
                    task_config=task_config,
                    started_at=started_at,
                )
                if result_path is None:
                    failures.append(
                        {
                            "task_name": task_name,
                            "phase": phase,
                            "return_code": 0,
                            "category": "missing_result",
                            "policy_failure": None,
                            "reason": "missing_result",
                        }
                    )
                    _write_summary(
                        output_dir,
                        summary,
                        failures,
                        invalid_setups=invalid_setups,
                        candidate_summaries=candidate_summaries,
                        num_episodes=num_episodes,
                    )
                    return 1

                rate = _parse_success_rate(result_path)
                summary[task_name][phase] = rate
                candidate_summaries.append(
                    _candidate_summary_from_output(
                        run_result.output_tail,
                        task_name=task_name,
                        phase=phase,
                        task_config=task_config,
                        seed=seed,
                        requested_valid_episodes=num_episodes,
                        valid_episodes=num_episodes,
                        observed_invalid_candidates=len(output_invalid_setups),
                    )
                )
                _write_summary(
                    output_dir,
                    summary,
                    failures,
                    invalid_setups=invalid_setups,
                    candidate_summaries=candidate_summaries,
                    num_episodes=num_episodes,
                )
                print(
                    "[eazywam-robotwin-manager] "
                    f"done task={task_name} phase={phase} success_rate={rate:.4f} "
                    f"result={result_path}",
                    flush=True,
                )
                break

    _write_summary(
        output_dir,
        summary,
        failures,
        invalid_setups=invalid_setups,
        candidate_summaries=candidate_summaries,
        num_episodes=num_episodes,
    )
    stats = _coverage_stats(
        summary,
        num_episodes,
        invalid_setups,
        failures,
        candidate_summaries,
    )
    print(
        "[eazywam-robotwin-manager] "
        f"requested_valid_episodes={stats['target_valid_episodes']} "
        f"valid_episodes={stats['valid_episodes']} "
        f"candidate_episodes_attempted={stats['candidate_episodes_attempted']} "
        f"invalid_candidate_episodes={stats['invalid_candidate_episodes']} "
        f"failures={stats['failure_count']}",
        flush=True,
    )
    print(f"[eazywam-robotwin-manager] summary={output_dir / 'summary.json'}", flush=True)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EazyWAM FastWAM RoboTwin manager wrapper")
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset-stats-path", type=Path, required=True)
    parser.add_argument("--task-name", default="null")
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--num-episodes", default="1")
    parser.add_argument("--replan-steps", default="24")
    parser.add_argument("--num-inference-steps", default="10")
    parser.add_argument("--skip-get-obs-within-replan", default="True")
    parser.add_argument("--redirect-common-files", default="False")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--resume", default="False")
    parser.add_argument(
        "--max-worker-restarts-on-invalid-setup",
        dest="max_worker_restarts_on_invalid_setup",
        default=None,
    )
    parser.add_argument(
        "--max-invalid-setup-retries",
        dest="max_worker_restarts_on_invalid_setup",
        default=None,
    )
    return parser.parse_args(argv)


def _tasks(task_name: str, robotwin_root: Path) -> list[str]:
    normalized = task_name.strip()
    if normalized and normalized.lower() not in {"none", "null"}:
        return [normalized]

    task_file = robotwin_root / "task_config" / "_eval_step_limit.yml"
    payload = yaml.safe_load(task_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Invalid RoboTwin task list: {task_file}")
    return list(dict.fromkeys(str(task) for task in payload))


def _single_eval_command(
    *,
    args: argparse.Namespace,
    upstream_dir: Path,
    output_dir: Path,
    task_name: str,
    task_config: str,
    seed: str | None = None,
) -> list[str]:
    resolved_seed = args.seed if seed is None else seed
    return [
        sys.executable,
        str(upstream_dir / "experiments" / "robotwin" / "eval_robotwin_single.py"),
        f"task={args.task}",
        f"seed={resolved_seed}",
        f"ckpt={args.ckpt}",
        f"EVALUATION.dataset_stats_path={args.dataset_stats_path}",
        f"EVALUATION.robotwin_root={args.robotwin_root}",
        f"EVALUATION.output_dir={output_dir / task_name / task_config}",
        f"EVALUATION.task_name={task_name}",
        f"EVALUATION.task_config={task_config}",
        f"EVALUATION.instruction_type={args.instruction_type}",
        f"EVALUATION.eval_num_episodes={_positive_int(args.num_episodes, '--num-episodes')}",
        f"EVALUATION.replan_steps={args.replan_steps}",
        f"EVALUATION.num_inference_steps={args.num_inference_steps}",
        f"EVALUATION.skip_get_obs_within_replan={args.skip_get_obs_within_replan}",
        f"model.redirect_common_files={args.redirect_common_files}",
        f"gpu_id={args.gpu_id}",
    ]


def _run_streaming(cmd: list[str], *, cwd: Path) -> RunResult:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output_tail: deque[str] = deque(maxlen=1000)
    for line in process.stdout:
        output_tail.append(line)
        print(line, end="", flush=True)
    return RunResult(return_code=process.wait(), output_tail="".join(output_tail))


def _coerce_run_result(result: RunResult | int) -> RunResult:
    if isinstance(result, RunResult):
        return result
    return RunResult(return_code=int(result), output_tail="")


def _latest_result_path(
    *,
    robotwin_root: Path,
    task_name: str,
    task_config: str,
    started_at: float,
) -> Path | None:
    result_root = robotwin_root / "eval_result" / task_name / "fastwam_policy" / task_config
    if not result_root.exists():
        return None
    candidates = [
        path
        for path in result_root.rglob("_result.txt")
        if path.stat().st_mtime >= started_at - 1.0
    ]
    if not candidates:
        candidates = list(result_root.rglob("_result.txt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_success_rate(path: Path) -> float:
    last_value: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            last_value = float(text)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Could not parse success rate from {path}")
    return last_value


def _write_summary(
    output_dir: Path,
    summary: dict[str, dict[str, float | None]],
    failures: list[dict[str, object]],
    *,
    invalid_setups: list[dict[str, object]] | None = None,
    candidate_summaries: list[dict[str, object]] | None = None,
    num_episodes: int = 1,
) -> None:
    clean_values = [value["clean"] for value in summary.values() if value["clean"] is not None]
    random_values = [value["random"] for value in summary.values() if value["random"] is not None]
    invalid_setups = list(invalid_setups or [])
    candidate_summaries = list(candidate_summaries or [])
    coverage = _coverage_stats(
        summary,
        num_episodes,
        invalid_setups,
        failures,
        candidate_summaries,
    )
    payload = {
        "requested": {
            "tasks": len(summary),
            "phases": len(summary) * len(PHASES),
            "episodes_per_phase": num_episodes,
            "target_valid_episodes": coverage["target_valid_episodes"],
        },
        "actual": coverage,
        "per_task": [
            {
                "task_name": task,
                "clean_success_rate": values["clean"],
                "random_success_rate": values["random"],
            }
            for task, values in summary.items()
        ],
        "overall": {
            "clean_mean_success_rate": _mean(clean_values),
            "random_mean_success_rate": _mean(random_values),
        },
        "candidate_episodes": candidate_summaries,
        "invalid_candidates": invalid_setups,
        "invalid_setups": invalid_setups,
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
        for task, values in summary.items():
            writer.writerow([task, values["clean"], values["random"]])
        writer.writerow([
            "__overall__",
            payload["overall"]["clean_mean_success_rate"],
            payload["overall"]["random_mean_success_rate"],
        ])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _coverage_stats(
    summary: dict[str, dict[str, float | None]],
    num_episodes: int,
    invalid_setups: list[dict[str, object]],
    failures: list[dict[str, object]],
    candidate_summaries: list[dict[str, object]],
) -> dict[str, object]:
    completed_values = [
        rate
        for values in summary.values()
        for rate in values.values()
        if rate is not None
    ]
    completed_phases = len(completed_values)
    policy_successes = sum(round(float(rate) * num_episodes) for rate in completed_values)
    target_phases = len(summary) * len(PHASES)
    target_valid_episodes = target_phases * num_episodes
    valid_episodes = completed_phases * num_episodes
    inferred_candidate_attempts = sum(
        _int_value(item.get("candidate_episodes_attempted"), 0)
        for item in candidate_summaries
    )
    inferred_invalid_candidates = sum(
        _int_value(item.get("invalid_candidate_episodes"), 0)
        for item in candidate_summaries
    )
    candidate_episodes_attempted = max(
        inferred_candidate_attempts,
        valid_episodes + len(invalid_setups),
    )
    invalid_candidate_episodes = max(
        inferred_invalid_candidates,
        len(invalid_setups),
        candidate_episodes_attempted - valid_episodes,
    )
    return {
        "completed_phases": completed_phases,
        "target_phases": target_phases,
        "valid_episodes": valid_episodes,
        "target_valid_episodes": target_valid_episodes,
        "candidate_episodes_attempted": candidate_episodes_attempted,
        "invalid_candidate_episodes": invalid_candidate_episodes,
        "policy_successes": int(policy_successes),
        "invalid_setup_count": invalid_candidate_episodes,
        "invalid_setup_events_recorded": len(invalid_setups),
        "failure_count": len(failures),
    }


def _load_existing_summary(
    output_dir: Path,
    summary: dict[str, dict[str, float | None]],
) -> None:
    path = output_dir / "summary.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_task = payload.get("per_task") if isinstance(payload, dict) else None
    if not isinstance(per_task, list):
        return
    for item in per_task:
        if not isinstance(item, dict):
            continue
        task_name = str(item.get("task_name", ""))
        if task_name not in summary:
            continue
        for phase, key in (
            ("clean", "clean_success_rate"),
            ("random", "random_success_rate"),
        ):
            value = item.get(key)
            if value is None:
                continue
            summary[task_name][phase] = float(value)


def _load_existing_invalid_setups(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("invalid_setups") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, dict)]


def _load_existing_candidate_summaries(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("candidate_episodes") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, dict)]


def _classify_invalid_setup_output(output: str) -> InvalidSetupClassification | None:
    classifications = _classify_invalid_setup_events(output)
    return classifications[0] if classifications else None


def _classify_invalid_setup_events(output: str) -> list[InvalidSetupClassification]:
    text = output.lower()
    classifications: list[InvalidSetupClassification] = []
    if (
        "indexerror: list index out of range" in text
        and "put_bottles_dustbin" in text
        and ("play_once" in text or "left_action" in text or "right_action" in text)
    ):
        count = max(1, text.count("indexerror: list index out of range"))
        classifications.extend(
            [
                InvalidSetupClassification(
                    reason="put_bottles_dustbin_expert_setup_index_error",
                    message=(
                        "RoboTwin generated a setup that crashed during upstream "
                        "expert/demo initialization before policy rollout."
                    ),
                )
                for _ in range(count)
            ]
        )
    unstable_count = text.count("unstableerror") + text.count("unstable error")
    classifications.extend(
        [
            InvalidSetupClassification(
                reason="robotwin_unstable_setup",
                message="RoboTwin marked the sampled simulator setup as unstable.",
            )
            for _ in range(unstable_count)
        ]
    )
    if (
        "motion planning" in text
        and ("play_once" in text or "setup_demo" in text or "expert" in text)
    ):
        count = max(1, text.count("motion planning"))
        classifications.extend(
            [
                InvalidSetupClassification(
                    reason="robotwin_motion_planning_setup_failed",
                    message=(
                        "RoboTwin failed during simulator setup or expert/demo motion "
                        "planning before policy rollout."
                    ),
                )
                for _ in range(count)
            ]
        )
    return classifications


def _invalid_setup_entries_from_output(
    output: str,
    *,
    task_name: str,
    phase: str,
    task_config: str,
    seed: str,
    attempt: int,
    return_code: int,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index, classification in enumerate(_classify_invalid_setup_events(output), start=1):
        entries.append(
            {
                "task_name": task_name,
                "phase": phase,
                "task_config": task_config,
                "attempt": attempt,
                "seed": seed,
                "event_index": index,
                "return_code": return_code,
                "category": "simulator_setup_invalid",
                "policy_failure": False,
                "reason": classification.reason,
                "message": classification.message,
                "source": "worker_output",
            }
        )
    return entries


def _candidate_summary_from_output(
    output: str,
    *,
    task_name: str,
    phase: str,
    task_config: str,
    seed: str,
    requested_valid_episodes: int,
    valid_episodes: int,
    observed_invalid_candidates: int,
) -> dict[str, object]:
    start_seed = _robotwin_internal_seed_start(seed)
    current_seeds = _robotwin_current_seed_values(output)
    last_valid_seed = current_seeds[-1] if current_seeds else None
    inferred_attempts = valid_episodes
    if start_seed is not None and last_valid_seed is not None and last_valid_seed >= start_seed:
        inferred_attempts = last_valid_seed - start_seed + 1
    candidate_attempts = max(
        inferred_attempts,
        valid_episodes + observed_invalid_candidates,
    )
    invalid_candidates = max(0, candidate_attempts - valid_episodes)
    return {
        "task_name": task_name,
        "phase": phase,
        "task_config": task_config,
        "seed": seed,
        "top_level_seed": seed,
        "internal_seed_start": start_seed,
        "last_valid_internal_seed": last_valid_seed,
        "requested_valid_episodes": requested_valid_episodes,
        "valid_episodes": valid_episodes,
        "candidate_episodes_attempted": candidate_attempts,
        "invalid_candidate_episodes": invalid_candidates,
    }


def _robotwin_internal_seed_start(seed: str) -> int | None:
    try:
        return 100000 * (1 + int(seed))
    except ValueError:
        return None


def _robotwin_current_seed_values(output: str) -> list[int]:
    values: list[int] = []
    for match in re.finditer(r"current seed:\s*\x1b\[[0-9;]*m?([0-9]+)", output):
        values.append(int(match.group(1)))
    if values:
        return values
    for match in re.finditer(r"current seed:\s*([0-9]+)", output):
        values.append(int(match.group(1)))
    return values


def _output_excerpt(output: str, *, max_chars: int = 4000) -> str:
    if len(output) <= max_chars:
        return output
    return output[-max_chars:]


def _seed_for_attempt(seed: str, attempt: int) -> str:
    if attempt <= 0:
        return str(seed)
    try:
        return str(int(seed) + attempt)
    except ValueError:
        return str(seed)


def _positive_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_existing_robotwin_results(
    robotwin_root: Path,
    summary: dict[str, dict[str, float | None]],
) -> None:
    for task_name, phase_values in summary.items():
        for phase, task_config in PHASES.items():
            if phase_values[phase] is not None:
                continue
            result_path = _latest_result_path(
                robotwin_root=robotwin_root,
                task_name=task_name,
                task_config=task_config,
                started_at=0.0,
            )
            if result_path is None:
                continue
            phase_values[phase] = _parse_success_rate(result_path)
            print(
                "[eazywam-robotwin-manager] "
                f"resume found task={task_name} phase={phase} "
                f"success_rate={phase_values[phase]:.4f} result={result_path}",
                flush=True,
            )


def _bool_value(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    return bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
