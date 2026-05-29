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
