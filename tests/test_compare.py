from __future__ import annotations

import json

import pytest

from wam_harness.cli import main
from wam_harness.core.compare import compare_traces


def write_trace(path, events) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )


def event(run_id: str, name: str, **payload):
    base = {
        "schema_version": 1,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": run_id,
        "event": name,
        "manifest_id": "fake-open-loop",
        "backend": "fake",
        "processor": "passthrough",
        "mode": "fake",
        "model_name": "Fake Open Loop WAM",
        "source_repo": "wam-harness/fake",
    }
    base.update(payload)
    return base


def native_contract(run_id: str, **overrides):
    payload = {
        "native": True,
        "backend": "fastwam",
        "processor": "fastwam_libero",
        "workload": "processor_smoke",
        "mode": "native_run",
        "runtime_mode": "in_process",
        "runtime_loader": "fastwam_runtime_loader",
        "model_adapter": "fastwam_model",
        "supported_optimizations": ["torch_compile"],
        "optimization_profile_status": [],
        "deployment": {"native_backend": "fastwam"},
        "backend_config_keys": ["cache_dir", "upstream_dir"],
        "processor_modality": {"processor": "fastwam_libero"},
    }
    payload.update(overrides)
    return event(run_id, "native_runtime_contract", **payload)


def test_compare_traces_reports_faster_variant(tmp_path) -> None:
    baseline = tmp_path / "baseline" / "trace.jsonl"
    variant = tmp_path / "variant" / "trace.jsonl"
    write_trace(
        baseline,
        [
            event("base", "run_start", optimization_profiles=[]),
            event("base", "inference_end", action_chunk_shape=[3, 4], timing={"wall_ms": 10.0}),
            event("base", "inference_end", action_chunk_shape=[3, 4], timing={"wall_ms": 12.0}),
            event("base", "run_end", status="ok"),
        ],
    )
    write_trace(
        variant,
        [
            event(
                "var",
                "run_start",
                optimization_profiles=[{"name": "fake_cache", "enabled": True, "params": {}}],
            ),
            event("var", "inference_end", action_chunk_shape=[3, 4], timing={"wall_ms": 5.0}),
            event("var", "inference_end", action_chunk_shape=[3, 4], timing={"wall_ms": 6.0}),
            event("var", "run_end", status="ok"),
        ],
    )

    summary = compare_traces(baseline.parent, variant.parent, min_effect=0.05)

    assert summary.decision == "faster"
    assert summary.output_gate_passed is True
    assert summary.baseline.latency_summary()["count"] == 2
    assert summary.variant.profiles == ["fake_cache"]
    assert summary.relative_change is not None
    assert summary.relative_change < 0


def test_compare_traces_allows_profile_change_under_same_native_contract(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event("base", "run_start", optimization_profiles=[]),
            native_contract("base"),
            event("base", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 20.0}),
            event("base", "run_end", status="ok"),
        ],
    )
    write_trace(
        variant,
        [
            event(
                "var",
                "run_start",
                optimization_profiles=[{"name": "torch_compile", "enabled": True, "params": {}}],
            ),
            native_contract(
                "var",
                optimization_profile_status=[
                    {
                        "name": "torch_compile",
                        "enabled": True,
                        "params": {},
                        "declared_supported": True,
                        "state": "requested",
                    }
                ],
            ),
            event("var", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 10.0}),
            event("var", "run_end", status="ok"),
        ],
    )

    summary = compare_traces(baseline, variant)

    assert summary.decision == "faster"
    assert summary.native_contract_gate == "native_contract_match"
    assert summary.native_contract_gate_passed is True
    assert summary.baseline.native_contract["backend"] == "fastwam"
    assert summary.variant.profiles == ["torch_compile"]


def test_compare_traces_invalidates_native_contract_mismatch(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event("base", "run_start", optimization_profiles=[]),
            native_contract("base"),
            event("base", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 20.0}),
            event("base", "run_end", status="ok"),
        ],
    )
    write_trace(
        variant,
        [
            event("var", "run_start", optimization_profiles=[]),
            native_contract("var", workload="serve"),
            event("var", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 10.0}),
            event("var", "run_end", status="ok"),
        ],
    )

    summary = compare_traces(baseline, variant)

    assert summary.decision == "invalid"
    assert summary.native_contract_gate_passed is False
    assert summary.native_contract_gate_details["differences"]["workload"] == {
        "baseline": "processor_smoke",
        "variant": "serve",
    }
    assert "native runtime contract mismatch" in summary.warnings


def test_compare_traces_invalidates_model_adapter_mismatch(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event("base", "run_start", optimization_profiles=[]),
            native_contract("base", model_adapter="fastwam_model"),
            event("base", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 20.0}),
            event("base", "run_end", status="ok"),
        ],
    )
    write_trace(
        variant,
        [
            event("var", "run_start", optimization_profiles=[]),
            native_contract("var", model_adapter="cosmos_policy_model"),
            event("var", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 10.0}),
            event("var", "run_end", status="ok"),
        ],
    )

    summary = compare_traces(baseline, variant)

    assert summary.decision == "invalid"
    assert summary.native_contract_gate == "native_contract_match"
    assert summary.native_contract_gate_passed is False
    assert summary.native_contract_gate_details["differences"]["model_adapter"] == {
        "baseline": "fastwam_model",
        "variant": "cosmos_policy_model",
    }
    assert "native runtime contract mismatch" in summary.warnings


def test_compare_traces_invalidates_runtime_loader_mismatch(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event("base", "run_start", optimization_profiles=[]),
            native_contract("base", runtime_loader="fastwam_runtime_loader"),
            event("base", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 20.0}),
            event("base", "run_end", status="ok"),
        ],
    )
    write_trace(
        variant,
        [
            event("var", "run_start", optimization_profiles=[]),
            native_contract("var", runtime_loader="cosmos_policy_runtime_loader"),
            event("var", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 10.0}),
            event("var", "run_end", status="ok"),
        ],
    )

    summary = compare_traces(baseline, variant)

    assert summary.decision == "invalid"
    assert summary.native_contract_gate == "native_contract_match"
    assert summary.native_contract_gate_passed is False
    assert summary.native_contract_gate_details["differences"]["runtime_loader"] == {
        "baseline": "fastwam_runtime_loader",
        "variant": "cosmos_policy_runtime_loader",
    }
    assert "native runtime contract mismatch" in summary.warnings


def test_compare_traces_invalidates_runtime_mode_mismatch(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event("base", "run_start", optimization_profiles=[]),
            native_contract("base", runtime_mode="in_process"),
            event("base", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 20.0}),
            event("base", "run_end", status="ok"),
        ],
    )
    write_trace(
        variant,
        [
            event("var", "run_start", optimization_profiles=[]),
            native_contract("var", runtime_mode="resident_server"),
            event("var", "inference_end", action_chunk_shape=[2, 7], timing={"wall_ms": 10.0}),
            event("var", "run_end", status="ok"),
        ],
    )

    summary = compare_traces(baseline, variant)

    assert summary.decision == "invalid"
    assert summary.native_contract_gate == "native_contract_match"
    assert summary.native_contract_gate_passed is False
    assert summary.native_contract_gate_details["differences"]["runtime_mode"] == {
        "baseline": "in_process",
        "variant": "resident_server",
    }
    assert "native runtime contract mismatch" in summary.warnings


def test_compare_traces_marks_shape_mismatch_invalid(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [event("base", "inference_end", action_chunk_shape=[3, 4], timing={"wall_ms": 10.0})],
    )
    write_trace(
        variant,
        [event("var", "inference_end", action_chunk_shape=[2, 4], timing={"wall_ms": 5.0})],
    )

    summary = compare_traces(baseline, variant)

    assert summary.decision == "invalid"
    assert summary.output_gate == "action_shape_match"
    assert summary.output_gate_passed is False


def test_compare_traces_marks_non_finite_actions_invalid(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event(
                "base",
                "inference_end",
                action_chunk_shape=[1, 2],
                action_summary={"shape": [1, 2], "finite": True, "mean": 1.0},
                timing={"wall_ms": 10.0},
            )
        ],
    )
    write_trace(
        variant,
        [
            event(
                "var",
                "inference_end",
                action_chunk_shape=[1, 2],
                action_summary={"shape": [1, 2], "finite": False, "mean": None},
                timing={"wall_ms": 5.0},
            )
        ],
    )

    summary = compare_traces(baseline, variant)

    assert summary.decision == "invalid"
    assert summary.output_gate == "action_shape_and_finite"
    assert summary.output_gate_passed is False
    assert "action summary contains non-finite values" in summary.warnings


def test_compare_traces_allows_action_summary_drift_within_tolerance(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event(
                "base",
                "inference_end",
                action_chunk_shape=[1, 2],
                action_summary={
                    "shape": [1, 2],
                    "finite": True,
                    "mean": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "max_abs": 2.0,
                },
                timing={"wall_ms": 10.0},
            )
        ],
    )
    write_trace(
        variant,
        [
            event(
                "var",
                "inference_end",
                action_chunk_shape=[1, 2],
                action_summary={
                    "shape": [1, 2],
                    "finite": True,
                    "mean": 1.0005,
                    "min": 0.0002,
                    "max": 2.0004,
                    "max_abs": 2.0004,
                },
                timing={"wall_ms": 5.0},
            )
        ],
    )

    summary = compare_traces(baseline, variant, max_action_drift=1e-3)

    assert summary.decision == "faster"
    assert summary.output_gate == "action_shape_finite_drift"
    assert summary.output_gate_passed is True
    assert summary.output_gate_details["observed"]["mean"] == pytest.approx(0.0005)


def test_compare_traces_marks_action_summary_drift_invalid(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event(
                "base",
                "inference_end",
                action_chunk_shape=[1, 2],
                action_summary={
                    "shape": [1, 2],
                    "finite": True,
                    "mean": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "max_abs": 2.0,
                },
                timing={"wall_ms": 10.0},
            )
        ],
    )
    write_trace(
        variant,
        [
            event(
                "var",
                "inference_end",
                action_chunk_shape=[1, 2],
                action_summary={
                    "shape": [1, 2],
                    "finite": True,
                    "mean": 1.2,
                    "min": 0.0,
                    "max": 2.0,
                    "max_abs": 2.0,
                },
                timing={"wall_ms": 5.0},
            )
        ],
    )

    summary = compare_traces(baseline, variant, max_action_drift=1e-3)

    assert summary.decision == "invalid"
    assert summary.output_gate == "action_shape_finite_drift"
    assert summary.output_gate_passed is False
    assert summary.output_gate_details["observed"]["mean"] == pytest.approx(0.2)
    assert "action summary drift exceeds tolerance" in summary.warnings


def test_compare_traces_allows_future_artifact_path_changes(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event(
                "base",
                "inference_end",
                action_chunk_shape=[1, 2],
                future_frames={
                    "present": True,
                    "count": 2,
                    "artifact_path": "baseline/future.json",
                },
                timing={"wall_ms": 10.0},
            )
        ],
    )
    write_trace(
        variant,
        [
            event(
                "var",
                "inference_end",
                action_chunk_shape=[1, 2],
                future_frames={
                    "present": True,
                    "count": 2,
                    "artifact_path": "variant/future.json",
                },
                timing={"wall_ms": 5.0},
            )
        ],
    )

    summary = compare_traces(baseline, variant)

    assert summary.decision == "faster"
    assert summary.output_gate == "action_shape_match"
    assert summary.output_gate_passed is True
    assert summary.output_gate_details["future_frames"] == "match"


def test_compare_traces_invalidates_future_summary_mismatch(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event(
                "base",
                "inference_end",
                action_chunk_shape=[1, 2],
                future_frames={"present": True, "count": 2},
                timing={"wall_ms": 10.0},
            )
        ],
    )
    write_trace(
        variant,
        [
            event(
                "var",
                "inference_end",
                action_chunk_shape=[1, 2],
                future_frames={"present": True, "count": 3},
                timing={"wall_ms": 5.0},
            )
        ],
    )

    summary = compare_traces(baseline, variant)

    assert summary.decision == "invalid"
    assert summary.output_gate == "action_shape_future_value_match"
    assert summary.output_gate_passed is False
    assert summary.output_gate_details["reason"] == "future frame summary mismatch"
    assert "future/value output mismatch" in summary.warnings


def test_compare_traces_invalidates_value_drift(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [
            event(
                "base",
                "inference_end",
                action_chunk_shape=[1, 2],
                value={"score": 0.5, "artifact_path": "baseline/value.json"},
                timing={"wall_ms": 10.0},
            )
        ],
    )
    write_trace(
        variant,
        [
            event(
                "var",
                "inference_end",
                action_chunk_shape=[1, 2],
                value={"score": 0.75, "artifact_path": "variant/value.json"},
                timing={"wall_ms": 5.0},
            )
        ],
    )

    summary = compare_traces(baseline, variant, max_action_drift=1e-3)

    assert summary.decision == "invalid"
    assert summary.output_gate == "action_shape_future_value_match"
    assert summary.output_gate_details["reason"] == "value mismatch"
    assert summary.output_gate_details["observed"]["value[0].score"] == pytest.approx(0.25)


def test_compare_traces_without_shapes_is_not_comparable(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(baseline, [event("base", "external_eval_end", timing={"wall_ms": 10.0})])
    write_trace(variant, [event("var", "external_eval_end", timing={"wall_ms": 5.0})])

    summary = compare_traces(baseline, variant)

    assert summary.decision == "not_comparable"
    assert summary.output_gate_passed is None


def test_cli_compare_outputs_json(tmp_path, capsys) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    write_trace(
        baseline,
        [event("base", "inference_end", action_chunk_shape=[1, 2], timing={"wall_ms": 10.0})],
    )
    write_trace(
        variant,
        [event("var", "inference_end", action_chunk_shape=[1, 2], timing={"wall_ms": 10.2})],
    )

    exit_code = main(
        [
            "compare",
            str(baseline),
            str(variant),
            "--min-effect",
            "0.05",
            "--max-action-drift",
            "0.01",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["decision"] == "same"
    assert payload["primary_metric"] == "latency_ms.mean"
    assert "output_gate_details" in payload
    assert "native_contract_gate" in payload
