from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


class CompareError(ValueError):
    """Raised when recorded traces cannot be compared."""


@dataclass(frozen=True)
class TraceStats:
    trace_path: Path
    run_id: str | None
    manifest_id: str | None
    backend: str | None
    processor: str | None
    mode: str | None
    profiles: list[str]
    runtime_contract: dict[str, Any] | None
    latency_samples_ms: list[float]
    action_shapes: list[list[int]]
    action_summaries: list[dict[str, Any]]
    future_frames: list[dict[str, Any]]
    values: list[Any]
    errors: list[str]

    def latency_summary(self) -> dict[str, float | int | None]:
        return _numeric_summary(self.latency_samples_ms)

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_path": str(self.trace_path),
            "run_id": self.run_id,
            "manifest_id": self.manifest_id,
            "backend": self.backend,
            "processor": self.processor,
            "mode": self.mode,
            "profiles": self.profiles,
            "runtime_contract": self.runtime_contract,
            "latency_ms": self.latency_summary(),
            "action_shapes": self.action_shapes,
            "action_summaries": self.action_summaries,
            "future_frames": self.future_frames,
            "values": self.values,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class CompareSummary:
    baseline: TraceStats
    variant: TraceStats
    primary_metric: str
    relative_change: float | None
    output_gate: str
    output_gate_passed: bool | None
    output_gate_details: dict[str, object]
    runtime_contract_gate: str
    runtime_contract_gate_passed: bool | None
    runtime_contract_gate_details: dict[str, object]
    decision: str
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "variant": self.variant.to_dict(),
            "primary_metric": self.primary_metric,
            "relative_change": self.relative_change,
            "output_gate": self.output_gate,
            "output_gate_passed": self.output_gate_passed,
            "output_gate_details": self.output_gate_details,
            "runtime_contract_gate": self.runtime_contract_gate,
            "runtime_contract_gate_passed": self.runtime_contract_gate_passed,
            "runtime_contract_gate_details": self.runtime_contract_gate_details,
            "decision": self.decision,
            "warnings": self.warnings,
        }


def compare_traces(
    baseline: str | Path,
    variant: str | Path,
    *,
    min_effect: float = 0.05,
    max_action_drift: float = 1e-3,
) -> CompareSummary:
    baseline_stats = load_trace_stats(baseline)
    variant_stats = load_trace_stats(variant)
    warnings: list[str] = []

    if baseline_stats.errors or variant_stats.errors:
        warnings.append("one or both traces contain errors")

    output_gate, output_gate_passed, output_gate_details = _output_gate(
        baseline_stats,
        variant_stats,
        max_action_drift=max_action_drift,
    )
    contract_gate, contract_gate_passed, contract_gate_details = _runtime_contract_gate(
        baseline_stats,
        variant_stats,
    )
    if output_gate_passed is False:
        if output_gate == "action_shape_and_finite":
            warnings.append("action summary contains non-finite values")
        elif output_gate == "action_shape_finite_drift":
            warnings.append("action summary drift exceeds tolerance")
        elif output_gate == "action_shape_future_value_match":
            warnings.append("future/value output mismatch")
        else:
            warnings.append("action shape mismatch")
    elif output_gate_passed is None:
        warnings.append("action shape gate unavailable")
    if contract_gate_passed is False:
        warnings.append("runtime contract mismatch")

    metric = "latency_ms.mean"
    baseline_mean = baseline_stats.latency_summary()["mean"]
    variant_mean = variant_stats.latency_summary()["mean"]
    relative_change = None
    decision = "not_comparable"
    if isinstance(baseline_mean, float) and isinstance(variant_mean, float) and baseline_mean > 0:
        relative_change = (variant_mean - baseline_mean) / baseline_mean
        if (
            baseline_stats.errors
            or variant_stats.errors
            or output_gate_passed is False
            or contract_gate_passed is False
        ):
            decision = "invalid"
        elif output_gate_passed is None:
            decision = "not_comparable"
        elif abs(relative_change) < min_effect:
            decision = "same"
        elif relative_change < 0:
            decision = "faster"
        else:
            decision = "slower"

    _profile_warnings(baseline_stats, variant_stats, warnings)
    return CompareSummary(
        baseline=baseline_stats,
        variant=variant_stats,
        primary_metric=metric,
        relative_change=relative_change,
        output_gate=output_gate,
        output_gate_passed=output_gate_passed,
        output_gate_details=output_gate_details,
        runtime_contract_gate=contract_gate,
        runtime_contract_gate_passed=contract_gate_passed,
        runtime_contract_gate_details=contract_gate_details,
        decision=decision,
        warnings=warnings,
    )


def load_trace_stats(path: str | Path) -> TraceStats:
    trace_path = _resolve_trace_path(path)
    events = _read_events(trace_path)
    if not events:
        raise CompareError(f"trace is empty: {trace_path}")

    metadata = _first_metadata(events)
    latencies = [_latency_ms(event) for event in events]
    latencies = [value for value in latencies if value is not None]
    shapes = [_shape(event) for event in events]
    shapes = [shape for shape in shapes if shape is not None]
    action_summaries = [_action_summary(event) for event in events]
    action_summaries = [summary for summary in action_summaries if summary is not None]
    future_frames = [_future_frames(event) for event in events]
    future_frames = [summary for summary in future_frames if summary is not None]
    values = [_value_output(event) for event in events]
    values = [value for value in values if value is not None]
    errors = [
        str(event.get("message") or event.get("error_type") or "error")
        for event in events
        if event.get("event") == "error"
    ]
    errors.extend(
        str(event.get("warnings") or event.get("status"))
        for event in events
        if event.get("event") in {"run_end", "serve_request_end"}
        and event.get("status") not in {None, "ok", "planned"}
    )
    return TraceStats(
        trace_path=trace_path,
        run_id=_first_str(events, "run_id"),
        manifest_id=metadata.get("manifest_id"),
        backend=metadata.get("backend"),
        processor=metadata.get("processor"),
        mode=metadata.get("mode"),
        profiles=_profile_names(metadata.get("optimization_profiles", [])),
        runtime_contract=_runtime_contract(events),
        latency_samples_ms=latencies,
        action_shapes=shapes,
        action_summaries=action_summaries,
        future_frames=future_frames,
        values=values,
        errors=errors,
    )


def _resolve_trace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "trace.jsonl"
    if not candidate.exists():
        raise CompareError(f"trace path does not exist: {candidate}")
    return candidate


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompareError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(event, dict):
            raise CompareError(f"trace event at {path}:{line_no} must be a JSON object")
        events.append(event)
    return events


def _first_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("manifest_id") or event.get("backend"):
            return event
    return {}


def _first_str(events: list[dict[str, Any]], key: str) -> str | None:
    for event in events:
        value = event.get(key)
        if value is not None:
            return str(value)
    return None


def _profile_names(raw_profiles: object) -> list[str]:
    if not isinstance(raw_profiles, list):
        return []
    names = []
    for item in raw_profiles:
        if isinstance(item, dict) and item.get("name") is not None:
            names.append(str(item["name"]))
    return names


def _runtime_contract(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    contract_keys = {
        "backend",
        "processor",
        "workload",
        "mode",
        "runtime_mode",
        "runtime_loader",
        "model_adapter",
        "supported_optimizations",
        "optimization_profile_status",
        "deployment",
        "backend_config_keys",
        "processor_modality",
    }
    for event in events:
        if event.get("event") != "runtime_contract":
            continue
        payload = {key: event[key] for key in contract_keys if key in event}
        return dict(payload)
    return None


def _latency_ms(event: dict[str, Any]) -> float | None:
    if event.get("event") not in {"inference_end", "serve_request_end", "external_eval_end"}:
        return None
    if event.get("event") == "serve_request_end" and event.get("status") not in {None, "ok"}:
        return None
    timing = event.get("timing")
    if isinstance(timing, dict):
        for key in ("wall_ms", "total_ms", "model_ms", "server_ms"):
            value = timing.get(key)
            if isinstance(value, int | float):
                return float(value)
    return None


def _shape(event: dict[str, Any]) -> list[int] | None:
    raw = event.get("action_chunk_shape")
    if isinstance(raw, list) and all(isinstance(item, int) for item in raw):
        return list(raw)
    return None


def _action_summary(event: dict[str, Any]) -> dict[str, Any] | None:
    raw = event.get("action_summary")
    if not isinstance(raw, dict):
        return None
    return dict(raw)


def _future_frames(event: dict[str, Any]) -> dict[str, Any] | None:
    raw = event.get("future_frames")
    if not isinstance(raw, dict):
        return None
    return dict(raw)


def _value_output(event: dict[str, Any]) -> Any | None:
    if "value" not in event or event["value"] is None:
        return None
    raw = event["value"]
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, str | int | float | bool):
        return raw
    return str(raw)


def _numeric_summary(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(samples),
        "mean": mean(samples),
        "p50": _percentile(samples, 50),
        "p95": _percentile(samples, 95),
    }


def _percentile(samples: list[float], percentile: int) -> float:
    ordered = sorted(samples)
    index = max(math.ceil((percentile / 100) * len(ordered)) - 1, 0)
    return ordered[index]


def _output_gate(
    baseline: TraceStats,
    variant: TraceStats,
    *,
    max_action_drift: float,
) -> tuple[str, bool | None, dict[str, object]]:
    if not baseline.action_shapes or not variant.action_shapes:
        return "action_shape_unavailable", None, {}
    shape_match = baseline.action_shapes == variant.action_shapes
    if not shape_match:
        return "action_shape_match", False, {
            "baseline_shapes": baseline.action_shapes,
            "variant_shapes": variant.action_shapes,
        }
    if baseline.action_summaries and variant.action_summaries:
        if len(baseline.action_summaries) != len(variant.action_summaries):
            return "action_shape_finite_drift", False, {
                "reason": "action summary count mismatch",
                "baseline_count": len(baseline.action_summaries),
                "variant_count": len(variant.action_summaries),
            }
        if not (
            _all_actions_finite(baseline.action_summaries)
            and _all_actions_finite(variant.action_summaries)
        ):
            return "action_shape_and_finite", False, {}
        drift = _action_summary_drift(baseline.action_summaries, variant.action_summaries)
        if drift:
            max_observed = max(drift.values())
            if max_observed > max_action_drift:
                return "action_shape_finite_drift", False, {
                    "max_action_drift": max_action_drift,
                    "observed": drift,
                }
            future_value_ok, future_value_details = _future_value_gate(
                baseline,
                variant,
                max_value_drift=max_action_drift,
            )
            if not future_value_ok:
                return "action_shape_future_value_match", False, future_value_details
            return "action_shape_finite_drift", True, {
                "max_action_drift": max_action_drift,
                "observed": drift,
            }
        future_value_ok, future_value_details = _future_value_gate(
            baseline,
            variant,
            max_value_drift=max_action_drift,
        )
        if not future_value_ok:
            return "action_shape_future_value_match", False, future_value_details
        return "action_shape_and_finite", True, future_value_details
    future_value_ok, future_value_details = _future_value_gate(
        baseline,
        variant,
        max_value_drift=max_action_drift,
    )
    if not future_value_ok:
        return "action_shape_future_value_match", False, future_value_details
    return "action_shape_match", True, future_value_details


def _future_value_gate(
    baseline: TraceStats,
    variant: TraceStats,
    *,
    max_value_drift: float,
) -> tuple[bool, dict[str, object]]:
    details: dict[str, object] = {}
    if bool(baseline.future_frames) != bool(variant.future_frames):
        return False, {
            "reason": "future frame presence mismatch",
            "baseline_count": len(baseline.future_frames),
            "variant_count": len(variant.future_frames),
        }
    if len(baseline.future_frames) != len(variant.future_frames):
        return False, {
            "reason": "future frame summary count mismatch",
            "baseline_count": len(baseline.future_frames),
            "variant_count": len(variant.future_frames),
        }
    if baseline.future_frames:
        baseline_future = [_strip_artifact_refs(item) for item in baseline.future_frames]
        variant_future = [_strip_artifact_refs(item) for item in variant.future_frames]
        if baseline_future != variant_future:
            return False, {
                "reason": "future frame summary mismatch",
                "baseline": baseline_future,
                "variant": variant_future,
            }
        details["future_frames"] = "match"

    if bool(baseline.values) != bool(variant.values):
        return False, {
            "reason": "value presence mismatch",
            "baseline_count": len(baseline.values),
            "variant_count": len(variant.values),
        }
    if len(baseline.values) != len(variant.values):
        return False, {
            "reason": "value count mismatch",
            "baseline_count": len(baseline.values),
            "variant_count": len(variant.values),
        }
    if baseline.values:
        value_drift: dict[str, float] = {}
        for index, (baseline_value, variant_value) in enumerate(
            zip(baseline.values, variant.values, strict=True)
        ):
            matched, drift = _json_match_with_numeric_tolerance(
                _strip_artifact_refs(baseline_value),
                _strip_artifact_refs(variant_value),
                max_value_drift=max_value_drift,
                path=f"value[{index}]",
            )
            value_drift.update(drift)
            if not matched:
                return False, {
                    "reason": "value mismatch",
                    "max_value_drift": max_value_drift,
                    "observed": value_drift,
                    "baseline": [_strip_artifact_refs(value) for value in baseline.values],
                    "variant": [_strip_artifact_refs(value) for value in variant.values],
                }
        details["value_drift"] = value_drift
    return True, details


def _runtime_contract_gate(
    baseline: TraceStats,
    variant: TraceStats,
) -> tuple[str, bool | None, dict[str, object]]:
    if baseline.runtime_contract is None and variant.runtime_contract is None:
        return "runtime_contract_unavailable", None, {}
    if baseline.runtime_contract is None or variant.runtime_contract is None:
        return "runtime_contract_presence", False, {
            "baseline_has_contract": baseline.runtime_contract is not None,
            "variant_has_contract": variant.runtime_contract is not None,
        }

    baseline_key = _comparable_runtime_contract(baseline.runtime_contract)
    variant_key = _comparable_runtime_contract(variant.runtime_contract)
    if baseline_key == variant_key:
        return "runtime_contract_match", True, {}

    differences = {
        key: {
            "baseline": baseline_key.get(key),
            "variant": variant_key.get(key),
        }
        for key in sorted(set(baseline_key) | set(variant_key))
        if baseline_key.get(key) != variant_key.get(key)
    }
    return "runtime_contract_match", False, {"differences": differences}


def _comparable_runtime_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key not in {"optimization_profile_status"}
    }


def _all_actions_finite(summaries: list[dict[str, Any]]) -> bool:
    return all(summary.get("finite") is not False for summary in summaries)


def _action_summary_drift(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
) -> dict[str, float]:
    fields = ("mean", "min", "max", "max_abs")
    drift: dict[str, float] = {}
    for field in fields:
        deltas = []
        for baseline_summary, variant_summary in zip(baseline, variant, strict=False):
            baseline_value = baseline_summary.get(field)
            variant_value = variant_summary.get(field)
            if not isinstance(baseline_value, int | float):
                continue
            if not isinstance(variant_value, int | float):
                continue
            deltas.append(abs(float(variant_value) - float(baseline_value)))
        if deltas:
            drift[field] = max(deltas)
    return drift


def _strip_artifact_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_artifact_refs(item)
            for key, item in value.items()
            if key not in {"artifact_path", "path"}
            and not str(key).endswith("_path")
        }
    if isinstance(value, list):
        return [_strip_artifact_refs(item) for item in value]
    return value


def _json_match_with_numeric_tolerance(
    baseline: Any,
    variant: Any,
    *,
    max_value_drift: float,
    path: str,
) -> tuple[bool, dict[str, float]]:
    if _is_number(baseline) and _is_number(variant):
        delta = abs(float(variant) - float(baseline))
        return delta <= max_value_drift, {path: delta}
    if isinstance(baseline, dict) and isinstance(variant, dict):
        if set(baseline) != set(variant):
            return False, {}
        drift: dict[str, float] = {}
        for key in sorted(baseline):
            matched, child_drift = _json_match_with_numeric_tolerance(
                baseline[key],
                variant[key],
                max_value_drift=max_value_drift,
                path=f"{path}.{key}",
            )
            drift.update(child_drift)
            if not matched:
                return False, drift
        return True, drift
    if isinstance(baseline, list) and isinstance(variant, list):
        if len(baseline) != len(variant):
            return False, {}
        drift: dict[str, float] = {}
        for index, (baseline_item, variant_item) in enumerate(
            zip(baseline, variant, strict=True)
        ):
            matched, child_drift = _json_match_with_numeric_tolerance(
                baseline_item,
                variant_item,
                max_value_drift=max_value_drift,
                path=f"{path}[{index}]",
            )
            drift.update(child_drift)
            if not matched:
                return False, drift
        return True, drift
    return baseline == variant, {}


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _profile_warnings(
    baseline: TraceStats,
    variant: TraceStats,
    warnings: list[str],
) -> None:
    if baseline.manifest_id != variant.manifest_id:
        warnings.append("manifest ids differ")
    if baseline.backend != variant.backend:
        warnings.append("backends differ")
    if baseline.processor != variant.processor:
        warnings.append("processors differ")
    if baseline.profiles == variant.profiles:
        warnings.append("optimization profiles are identical")
