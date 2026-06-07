import json
from pathlib import Path

import pytest

from eazywam.evals.acceptance import (
    AcceptanceError,
    main,
    validate_native_eval_summary,
)


def test_validate_native_eval_summary_accepts_native_eval_trace(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"
    results_path.write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "successes": 1,
                "success_rate": 1.0,
                "episodes": [{"success": True}],
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        [
            {"event": "run_start"},
            {
                "event": "native_eval_end",
                "successes": 1,
                "total_episodes": 1,
                "success_rate": 1.0,
                "results_path": str(results_path),
            },
            {"event": "run_end", "status": "ok"},
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "return_code": 0,
                "trace_path": str(trace_path),
                "runtime_info": {"backend": "fastwam", "mode": "simulator_eval"},
                "metrics": {
                    "total_episodes": 1,
                    "successes": 1,
                    "success_rate": 1.0,
                    "results_path": str(results_path),
                },
            }
        ),
        encoding="utf-8",
    )

    report = validate_native_eval_summary(summary_path, expected_trials=1)

    assert report.trace_path == trace_path
    assert report.results_path == results_path
    assert report.expected_trials == 1
    assert report.success_rate == 1.0
    assert report.min_success_rate == 0.0
    assert report.to_dict()["status"] == "ok"
    assert "acceptance passed" in report.message()


def test_validate_native_eval_summary_resolves_paths_from_summary_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_root = tmp_path / "runs"
    run_dir = trace_root / "abc123"
    other_dir = tmp_path / "other"
    run_dir.mkdir(parents=True)
    other_dir.mkdir()
    results_path = run_dir / "native_eval" / "libero_10" / "task0_results.json"
    trace_path = run_dir / "trace.jsonl"
    summary_path = trace_root / "fastwam-libero-libero-single-task-eval-summary.json"
    results_path.parent.mkdir(parents=True)
    results_path.write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "successes": 1,
                "success_rate": 1.0,
                "episodes": [{"success": True}],
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        [
            {"event": "run_start"},
            {
                "event": "native_eval_end",
                "successes": 1,
                "total_episodes": 1,
                "success_rate": 1.0,
                "results_path": "runs/abc123/native_eval/libero_10/task0_results.json",
            },
            {"event": "run_end", "status": "ok"},
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "return_code": 0,
                "trace_path": "runs/abc123/trace.jsonl",
                "runtime_info": {"backend": "fastwam", "mode": "simulator_eval"},
                "metrics": {
                    "total_episodes": 1,
                    "successes": 1,
                    "success_rate": 1.0,
                    "results_path": "runs/abc123/native_eval/libero_10/task0_results.json",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(other_dir)

    report = validate_native_eval_summary(summary_path, expected_trials=1)

    assert report.trace_path == trace_path
    assert report.results_path == results_path


def test_validate_native_eval_summary_rejects_external_eval_trace(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"
    results_path.write_text("{}", encoding="utf-8")
    _write_trace(
        trace_path,
        [
            {"event": "run_start"},
            {"event": "external_eval_plan"},
            {"event": "run_end", "status": "ok"},
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "return_code": 0,
                "trace_path": str(trace_path),
                "runtime_info": {"backend": "external_eval", "mode": "reference_eval"},
                "metrics": {
                    "total_episodes": 1,
                    "successes": 1,
                    "success_rate": 1.0,
                    "results_path": str(results_path),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AcceptanceError) as exc_info:
        validate_native_eval_summary(summary_path, expected_trials=1)

    message = str(exc_info.value)
    assert "external_eval" in message
    assert "native_eval_end" in message
    assert "external_eval_plan" in message


def test_validate_native_eval_summary_rejects_missing_runtime_info(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"
    results_path.write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "successes": 1,
                "success_rate": 1.0,
                "episodes": [{"success": True}],
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        [
            {"event": "run_start"},
            {
                "event": "native_eval_end",
                "successes": 1,
                "total_episodes": 1,
                "success_rate": 1.0,
                "results_path": str(results_path),
            },
            {"event": "run_end", "status": "ok"},
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "return_code": 0,
                "trace_path": str(trace_path),
                "metrics": {
                    "total_episodes": 1,
                    "successes": 1,
                    "success_rate": 1.0,
                    "results_path": str(results_path),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AcceptanceError) as exc_info:
        validate_native_eval_summary(summary_path, expected_trials=1)

    message = str(exc_info.value)
    assert "summary.runtime_info.backend is missing" in message
    assert "summary.runtime_info.mode is missing" in message


def test_validate_native_eval_summary_rejects_inconsistent_results(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"
    results_path.write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "successes": 0,
                "success_rate": 0.0,
                "episodes": [{"success": False}],
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        [
            {"event": "run_start"},
            {
                "event": "native_eval_end",
                "successes": 1,
                "total_episodes": 1,
                "success_rate": 1.0,
                "results_path": str(results_path),
            },
            {"event": "run_end", "status": "ok"},
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "return_code": 0,
                "trace_path": str(trace_path),
                "runtime_info": {"backend": "fastwam", "mode": "simulator_eval"},
                "metrics": {
                    "total_episodes": 1,
                    "successes": 1,
                    "success_rate": 1.0,
                    "results_path": str(results_path),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AcceptanceError) as exc_info:
        validate_native_eval_summary(summary_path, expected_trials=1)

    message = str(exc_info.value)
    assert "results.successes" in message
    assert "results.success_rate" in message


def test_validate_native_eval_summary_rejects_below_success_rate_gate(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"
    results_path.write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "successes": 1,
                "success_rate": 0.5,
                "episodes": [{"success": True}, {"success": False}],
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        [
            {"event": "run_start"},
            {
                "event": "native_eval_end",
                "successes": 1,
                "total_episodes": 2,
                "success_rate": 0.5,
                "results_path": str(results_path),
            },
            {"event": "run_end", "status": "ok"},
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "return_code": 0,
                "trace_path": str(trace_path),
                "runtime_info": {"backend": "fastwam", "mode": "simulator_eval"},
                "metrics": {
                    "total_episodes": 2,
                    "successes": 1,
                    "success_rate": 0.5,
                    "results_path": str(results_path),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AcceptanceError) as exc_info:
        validate_native_eval_summary(
            summary_path,
            expected_trials=2,
            min_success_rate=1.0,
        )

    assert "expected at least 1.0" in str(exc_info.value)


def test_eval_acceptance_module_cli_reports_errors(tmp_path: Path, capsys) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"status": "error"}), encoding="utf-8")

    exit_code = main([str(summary_path), "1"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "acceptance error:" in captured.err
    assert "summary.status" in captured.err


def test_eval_acceptance_module_cli_accepts_min_success_rate_arg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_path = tmp_path / "results.json"
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"
    results_path.write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "successes": 1,
                "success_rate": 1.0,
                "episodes": [{"success": True}],
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        [
            {"event": "run_start"},
            {
                "event": "native_eval_end",
                "successes": 1,
                "total_episodes": 1,
                "success_rate": 1.0,
                "results_path": str(results_path),
            },
            {"event": "run_end", "status": "ok"},
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "return_code": 0,
                "trace_path": str(trace_path),
                "runtime_info": {"backend": "fastwam", "mode": "simulator_eval"},
                "metrics": {
                    "total_episodes": 1,
                    "successes": 1,
                    "success_rate": 1.0,
                    "results_path": str(results_path),
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main([str(summary_path), "1", "1.0"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "min_success_rate=1.0" in captured.out


def test_eval_acceptance_module_cli_outputs_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_path = tmp_path / "results.json"
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"
    results_path.write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "successes": 1,
                "success_rate": 1.0,
                "episodes": [{"success": True}],
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        [
            {"event": "run_start"},
            {
                "event": "native_eval_end",
                "successes": 1,
                "total_episodes": 1,
                "success_rate": 1.0,
                "results_path": str(results_path),
            },
            {"event": "run_end", "status": "ok"},
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "return_code": 0,
                "trace_path": str(trace_path),
                "runtime_info": {"backend": "fastwam", "mode": "simulator_eval"},
                "metrics": {
                    "total_episodes": 1,
                    "successes": 1,
                    "success_rate": 1.0,
                    "results_path": str(results_path),
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--json", str(summary_path), "1", "1.0"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["expected_trials"] == 1
    assert payload["min_success_rate"] == 1.0


def test_eval_acceptance_module_cli_rejects_non_positive_trials(capsys) -> None:
    exit_code = main(["summary.json", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "EXPECTED_TRIALS must be positive" in captured.err


def _write_trace(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
