from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class AcceptanceError(RuntimeError):
    """Raised when an eval summary does not prove native eval acceptance."""


@dataclass(frozen=True)
class AcceptanceReport:
    summary_path: Path
    trace_path: Path
    results_path: Path
    expected_trials: int
    success_rate: float | int
    min_success_rate: float

    def message(self) -> str:
        return (
            "FastWAM LIBERO native eval acceptance passed: "
            f"summary={self.summary_path} trace={self.trace_path} "
            f"expected_trials={self.expected_trials} "
            f"success_rate={self.success_rate} "
            f"min_success_rate={self.min_success_rate}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "summary_path": str(self.summary_path),
            "trace_path": str(self.trace_path),
            "results_path": str(self.results_path),
            "expected_trials": self.expected_trials,
            "success_rate": self.success_rate,
            "min_success_rate": self.min_success_rate,
        }


def validate_native_eval_summary(
    summary_path: str | Path,
    *,
    expected_trials: int,
    min_success_rate: float = 0.0,
) -> AcceptanceReport:
    if expected_trials <= 0:
        raise AcceptanceError(f"expected_trials is {expected_trials!r}, expected a positive integer")
    if not 0.0 <= min_success_rate <= 1.0:
        raise AcceptanceError(
            f"min_success_rate is {min_success_rate!r}, expected a value in 0..1"
        )
    path = Path(summary_path).expanduser()
    summary_dir = path.resolve().parent
    path_bases = (summary_dir, summary_dir.parent)
    summary = _load_json(path)
    errors: list[str] = []

    if summary.get("status") != "ok":
        errors.append(f"summary.status is {summary.get('status')!r}, expected 'ok'")
    if summary.get("return_code") != 0:
        errors.append(f"summary.return_code is {summary.get('return_code')!r}, expected 0")

    runtime_info = _dict(summary.get("runtime_info"))
    runtime_backend = runtime_info.get("backend")
    runtime_mode = runtime_info.get("mode")
    if not runtime_backend:
        errors.append("summary.runtime_info.backend is missing")
    elif runtime_backend == "external_eval":
        errors.append("summary.runtime_info.backend is external_eval; expected native backend")
    if not runtime_mode:
        errors.append("summary.runtime_info.mode is missing")

    metrics = _dict(summary.get("metrics"))
    metric_total_episodes = metrics.get("total_episodes")
    metric_successes = metrics.get("successes")
    metric_success_rate = metrics.get("success_rate")
    if metric_total_episodes != expected_trials:
        errors.append(
            "metrics.total_episodes is "
            f"{metric_total_episodes!r}, expected {expected_trials}"
        )
    if metric_success_rate is None:
        errors.append("metrics.success_rate is missing")
    elif not _is_number(metric_success_rate):
        errors.append(f"metrics.success_rate is {metric_success_rate!r}, expected a number")
    else:
        metric_success_rate_value = float(metric_success_rate)
        if not 0.0 <= metric_success_rate_value <= 1.0:
            errors.append(f"metrics.success_rate is {metric_success_rate!r}, expected 0..1")
        elif metric_success_rate_value < min_success_rate:
            errors.append(
                "metrics.success_rate is "
                f"{metric_success_rate!r}, expected at least {min_success_rate}"
            )
    if metric_successes is None:
        errors.append("metrics.successes is missing")
    elif not isinstance(metric_successes, int) or isinstance(metric_successes, bool):
        errors.append(f"metrics.successes is {metric_successes!r}, expected an integer")
    elif not 0 <= metric_successes <= expected_trials:
        errors.append(
            f"metrics.successes is {metric_successes!r}, expected 0..{expected_trials}"
        )

    results_path = _path_value(metrics.get("results_path"), base_dirs=path_bases)
    results: dict[str, Any] = {}
    if results_path is None:
        errors.append("metrics.results_path is missing")
    elif not results_path.exists():
        errors.append(f"metrics.results_path does not exist: {results_path}")
    else:
        results = _load_json(results_path)
        result_total_episodes = results.get("total_episodes")
        result_successes = results.get("successes")
        result_success_rate = results.get("success_rate")
        result_episodes = results.get("episodes")
        if result_total_episodes != expected_trials:
            errors.append(
                "results.total_episodes is "
                f"{result_total_episodes!r}, expected {expected_trials}"
            )
        if result_successes != metric_successes:
            errors.append(
                "results.successes is "
                f"{result_successes!r}, expected metrics.successes {metric_successes!r}"
            )
        if not _numbers_equal(result_success_rate, metric_success_rate):
            errors.append(
                "results.success_rate is "
                f"{result_success_rate!r}, expected metrics.success_rate {metric_success_rate!r}"
            )
        if not isinstance(result_episodes, list):
            errors.append("results.episodes is missing or not a list")
        elif len(result_episodes) != expected_trials:
            errors.append(
                f"results.episodes has length {len(result_episodes)}, expected {expected_trials}"
            )

    trace_path = _path_value(summary.get("trace_path"), base_dirs=path_bases)
    trace_events: list[dict[str, Any]] = []
    if trace_path is None:
        errors.append("summary.trace_path is missing")
    elif not trace_path.exists():
        errors.append(f"summary.trace_path does not exist: {trace_path}")
    else:
        trace_events = _load_jsonl(trace_path)
        event_names = [event.get("event") for event in trace_events]
        if "native_eval_end" not in event_names:
            errors.append("trace is missing native_eval_end")
        else:
            native_eval_end = next(
                event for event in reversed(trace_events) if event.get("event") == "native_eval_end"
            )
            if native_eval_end.get("total_episodes") != expected_trials:
                errors.append(
                    "trace native_eval_end.total_episodes is "
                    f"{native_eval_end.get('total_episodes')!r}, expected {expected_trials}"
                )
            if native_eval_end.get("successes") != metric_successes:
                errors.append(
                    "trace native_eval_end.successes is "
                    f"{native_eval_end.get('successes')!r}, expected metrics.successes "
                    f"{metric_successes!r}"
                )
            if not _numbers_equal(native_eval_end.get("success_rate"), metric_success_rate):
                errors.append(
                    "trace native_eval_end.success_rate is "
                    f"{native_eval_end.get('success_rate')!r}, expected metrics.success_rate "
                    f"{metric_success_rate!r}"
                )
            trace_results_path = _path_value(
                native_eval_end.get("results_path"),
                base_dirs=path_bases,
            )
            if (
                results_path is not None
                and trace_results_path is not None
                and not _same_path(trace_results_path, results_path)
            ):
                errors.append(
                    "trace native_eval_end.results_path is "
                    f"{trace_results_path}, expected {results_path}"
                )
        if "external_eval_plan" in event_names:
            errors.append("trace contains external_eval_plan; expected native eval path")
        run_end = next(
            (event for event in reversed(trace_events) if event.get("event") == "run_end"),
            None,
        )
        if run_end is None:
            errors.append("trace is missing run_end")
        elif run_end.get("status") != "ok":
            errors.append(f"trace run_end.status is {run_end.get('status')!r}, expected 'ok'")

    if errors:
        raise AcceptanceError("\n".join(errors))

    assert trace_path is not None
    assert results_path is not None
    return AcceptanceReport(
        summary_path=path,
        trace_path=trace_path,
        results_path=results_path,
        expected_trials=expected_trials,
        success_rate=metrics["success_rate"],
        min_success_rate=min_success_rate,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    json_output = False
    if "--json" in args:
        json_output = True
        args = [arg for arg in args if arg != "--json"]
    if len(args) not in (2, 3):
        print(
            "usage: python -m eazywam.evals.acceptance [--json] "
            "SUMMARY_JSON EXPECTED_TRIALS [MIN_SUCCESS_RATE]",
            file=sys.stderr,
        )
        return 2
    try:
        expected_trials = int(args[1])
    except ValueError:
        print(f"error: EXPECTED_TRIALS must be an integer, got {args[1]!r}", file=sys.stderr)
        return 2
    if expected_trials <= 0:
        print(
            f"error: EXPECTED_TRIALS must be positive, got {expected_trials}",
            file=sys.stderr,
        )
        return 2
    try:
        min_success_rate = float(args[2]) if len(args) == 3 else 0.0
    except ValueError:
        print(f"error: MIN_SUCCESS_RATE must be a number, got {args[2]!r}", file=sys.stderr)
        return 2

    try:
        report = validate_native_eval_summary(
            args[0],
            expected_trials=expected_trials,
            min_success_rate=min_success_rate,
        )
    except (OSError, json.JSONDecodeError, AcceptanceError) as exc:
        print(f"acceptance error: {exc}", file=sys.stderr)
        return 1

    if json_output:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.message())
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AcceptanceError(f"summary JSON must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AcceptanceError(f"trace line {line_no} must be an object: {path}")
        events.append(value)
    return events


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _path_value(
    value: object,
    *,
    base_dirs: Sequence[Path] = (),
) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = [path]
    candidates.extend(base / path for base in base_dirs)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_equal(left: object, right: object) -> bool:
    if not _is_number(left) or not _is_number(right):
        return left == right
    return abs(float(left) - float(right)) <= 1e-9


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


if __name__ == "__main__":
    raise SystemExit(main())
