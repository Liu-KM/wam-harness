# Trace Schema Notes

Trace files use JSON Lines. Each line is one event.

The trace schema exists to support systems experiments. It must make it
possible to compare a baseline and variant, identify where time and memory were
spent, and explain failures.

## Required Event Families

Phase 1 should support:

- `run_start`
- `run_end`
- `episode_start`
- `episode_end`
- `backend_load`
- `backend_warmup`
- `reset`
- `replan_start`
- `inference_start`
- `inference_end`
- `step`
- `memory_sample`
- `error`

Later phases may add:

- `server_request_start`
- `server_request_end`
- `cache_update_start`
- `cache_update_end`
- `graph_capture_start`
- `graph_capture_end`
- `compile_start`
- `compile_end`

## Comparison Events

The harness exists to compare runs, so the comparison itself is a trace event,
emitted by the analysis/`compare` step (not by a backend).

`comparison_result`

Required fields:

- `idea`
- `gate_type`: `exact`, `numeric_tolerance`, `success_rate`, `visual_metric`,
  or `not_comparable`.
- `baseline_run_id`
- `variant_run_id`
- `primary_metric`: name of the primary metric compared (e.g. `model_ms_p95`).
- `baseline_stats`: `{p50, p95, mean, std, n}` for the primary metric.
- `variant_stats`: `{p50, p95, mean, std, n}` for the primary metric.
- `relative_change`: signed fraction on the primary metric.
- `noise_floor`: run-to-run noise estimate for the primary metric.
- `decision`: `useful`, `neutral`, `regression`, `not_applicable`, or
  `not_comparable`.

Optional fields:

- `gate_passed`: whether the correctness gate held (true/false/null).
- `gate_detail`: tolerance used, mismatch count, or success-rate delta.
- `secondary`: map of secondary metric names to their stats.

The decision is computed per `docs/measurement.md` (effect size, noise floor,
direction consistency). A `comparison_result` must never claim `useful` when the
correctness gate failed.

## Common Fields

Every event should include:

- `schema_version`
- `event`
- `timestamp`
- `run_id`
- `episode_id` when available
- `step_id` when available
- `replan_id` when available
- `backend`
- `model_name`
- `source_repo`
- `mode`

## Experiment Fields

Run-level events should include:

- `idea`
- `baseline_or_variant`
- `pressure`
- `method`
- `optimization_flags`
- `correctness_gate`
- `decision_rule`

These fields may be compact strings in Phase 1. They should be enough to
connect a trace to the systems idea being tested.

## Workload Fields

Inference and step events should include:

- `action_horizon`
- `replan_steps`
- `action_chunk_len`
- `action_dim`
- `history_len`
- `image_shapes`
- `state_dims`

## Output Artifact Fields

Inference events that produce comparable outputs should reference artifacts by
path, never embed tensors:

- `action_chunk_path`: path to the persisted action chunk for this inference call.
- `action_chunk_shape`: shape-like metadata, e.g. `[T, D]`.
- `future_frames_path`: path to predicted frames, when produced. `null` otherwise.
- `value_path`: path to value output, when produced. `null` otherwise.
- `from_stale_chunk`: whether the executed action came from a previously generated
  chunk rather than a fresh inference call (supports chunk-scheduling analysis).

Artifacts are keyed by `(episode_id, step_id, replan_id)` so the comparator can
align the same decision point across baseline and variant runs.

## Characterization Fields

Run-level events for a `characterization` run should additionally include:

- `mode`: set to `characterization`.
- `stage_share`: fractional breakdown of `total_ms` across `preprocess`, `model`,
  `postprocess`, and `env_step`, so an existence check is data-backed.

Characterization runs emit no `comparison_result` event.

## Timing Fields

Inference events should include:

- `preprocess_ms`
- `model_ms`
- `postprocess_ms`
- `total_ms`

Step events should include:

- `env_step_ms`

Remote or later optimization events may include:

- `cuda_ms`
- `server_ms`
- `client_ms`
- `serialization_ms`
- `queue_ms`

## Memory Fields

Memory samples should include:

- `cuda_allocated_mb`
- `cuda_reserved_mb`
- `cuda_peak_allocated_mb`
- `cuda_peak_reserved_mb`
- `process_rss_mb`

Optional fields:

- `memory_delta_mb`
- `cache_bytes`
- `graph_pool_bytes`
- `snapshot_id`

CPU-only environments should still emit the same field names with `0` or
`null`, rather than changing event shape.

## Error Fields

Error events should include:

- `stage`
- `error_type`
- `message`
- `recoverable`
- `backend`

## Schema Policy

- Prefer append-only changes.
- Do not remove or rename fields without a schema version bump.
- Do not store backend-native tensors in traces.
- Store paths to large artifacts rather than embedding large data.
- Record missing measurements explicitly as `null`.
