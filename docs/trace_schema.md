# Trace Schema

Trace files use JSON Lines. Each line is one event. The trace schema supports
deployment debugging, optional benchmark comparison, and optimization-profile
inspection.

The schema is operational: it explains what happened during a run and gives
`wam compare` enough metadata to compare two runs.

## Phase A Event Families

- `run_start`
- `run_end`
- `episode_start`
- `episode_end`
- `backend_load_start`
- `backend_load`
- `backend_load_end`
- `backend_warmup`
- `reset`
- `replan_start`
- `inference_start`
- `inference_end`
- `step`
- `memory_sample`
- `external_eval_plan`
- `external_eval_end`
- `preflight`
- `serve_start`
- `serve_ready`
- `serve_request_start`
- `serve_request_end`
- `backend_close`
- `error`

Later phases may add:

- `prepare_start`
- `prepare_end`
- `cache_update_start`
- `cache_update_end`
- `graph_capture_start`
- `graph_capture_end`
- `compile_start`
- `compile_end`
- `streaming_action_partial`
- `comparison_result`

## Common Fields

Every event should include:

- `schema_version`
- `event`
- `timestamp`
- `run_id`
- `manifest_id` when available
- `backend`
- `processor`
- `model_name`
- `source_repo`
- `mode`: `local`, `remote`, `fake`, `run`, `native_smoke`, or `serve`

Workload events may also include:

- `episode_id`
- `step_id`
- `replan_id`

## Run Metadata

`run_start` should include:

- `output_dir`
- `device`
- `dtype`
- `optimization_profiles`
- `manifest_defaults`
- `config_overrides`
- `telemetry_config`

`run_end` should include:

- `status`: `ok`, `error`, `planned`, or `interrupted`
- `model_calls`
- `steps`
- `warnings`
- `trace_path`

For `external_eval` workloads, `model_calls` and `steps` may be omitted because
the official upstream evaluator owns the simulator loop.

`wam run` opens the trace before backend load. If backend preflight is already
`blocked`, the trace should contain `run_start`, `preflight`, an `error`
with `stage="preflight"`, and `run_end.status="error"`; it should not
emit `backend_load_start` because model loading was intentionally skipped. If
preflight is not blocked and `backend.load()` still fails because imports,
checkpoints, or runtime assets are missing, the trace should contain
`backend_load_start`, `error.stage="backend_load"`, and
`run_end.status="error"`. Later native smoke lifecycle failures should keep the
same rule and report the narrow failing stage: `backend_warmup`,
`backend_reset`, `inference`, or `action_contract`.

Backend cleanup is a lifecycle invariant even when no `backend_close` event is
emitted yet. Future schema revisions may add that event for resident backends
that need observable server/process shutdown.

For `serve` mode, startup emits backend load/warmup/reset events before
`serve_ready`. Each `/infer` request emits `serve_request_start` and
`serve_request_end` with request timing, output shape, and action summary. Bad
requests also emit an `error` event and a failed `serve_request_end`. Startup
failures emit an `error` event and `backend_close` so resident serve failures
are traceable.

For `native_smoke` workloads, `model_calls` should be `1` on success and
`steps` should be `0` because the command validates the backend lifecycle with a
single synthetic observation rather than running a simulator episode.

Native smoke errors should be precise enough to guide upstream script
migration. Supported `error.stage` values are:

- `preflight` for blocked readiness before model loading;
- `processor_smoke_observation` for synthetic observation construction;
- `backend_load` for importing/loading native model objects and runtime assets;
- `backend_warmup` for warmup failures;
- `backend_reset` for reset/session initialization failures;
- `inference` for the native model call and processor result path;
- `action_contract` for invalid action chunks.

## External Eval Fields

`external_eval_plan` should include:

- `command.argv`
- `command.workdir`
- `command.env`
- `stdout_path`
- `stderr_path`

`external_eval_end` should include:

- `return_code`
- `timing.wall_ms`
- `stdout_path`
- `stderr_path`

No-execute planning mode is an internal/test path. It emits
`external_eval_plan` and finishes with `run_end.status="planned"`. Official
script planning/execution for native-migration entries requires explicit
`wam eval <model-id> --reference`. Public user docs should prefer `wam doctor <model-id>`,
`wam prepare <model-id>`, and `wam run <model-id>` over teaching users a
`--dry-run` flag.

Full simulator evaluations should keep upstream stdout/stderr as separate files
and keep the harness trace focused on orchestration metadata.

## Native Smoke Fields

`wam native-smoke <model-id>` should emit the normal backend lifecycle events:

- `run_start`
- `runtime_contract`
- `preflight`
- `processor_smoke_observation`
- `backend_load_start`
- `backend_load`
- `backend_load_end`
- `backend_warmup`
- `reset`
- `inference_start`
- `inference_end`
- `run_end`

`run_start` should include:

- `synthetic_observation: true`
- `optimization_profiles`
- `manifest_defaults`
- `known_gaps`

`runtime_contract` should include:

- `backend`, `processor`, `workload`, and `mode`;
- `runtime_mode`: `in_process` when the `wam` process directly loads and calls
  the model, or `resident_server` when the backend starts/connects to a
  job-local policy server;
- `runtime_loader`, the backend-native loader that constructs or connects the
  model/server runtime before inference;
- `model_adapter`, the declared native model/server adapter expected for this
  backend before heavy model loading starts;
- `supported_optimizations`;
- `optimization_profile_status`, with requested profile name, enabled flag,
  params, whether the profile is declared supported by the model entry, manifest
  scope, target layer (`native_backend`, `workload`, or `deployment`), and state;
- `deployment`;
- `backend_config_keys`;
- `processor_modality` when the registered processor exposes modality limits.

`processor_smoke_observation` should include an `observation_summary`. If
native readiness is `blocked` before load, the trace should contain
`runtime_contract`, `preflight`, an `error` with
`stage="preflight"` plus `trace_path`, and
`run_end.status="error"`. In that case `backend_load_start` is intentionally
absent. If preflight passes and `backend.load()` fails later, the trace should
still contain `backend_load_start`, `error`, and `run_end.status="error"`.

`preflight` should include:

- `status`: `ready`, `warning`, or `blocked`;
- `backend` and `label`;
- `runtime_mode`;
- `runtime_loader`;
- `model_adapter`;
- `required_assets`;
- `runtime_assets`;
- `required_python_modules`;
- `missing_required_assets`;
- `missing_runtime_assets`;
- `missing_python_modules`;
- `upstream.status`, `upstream.selected`, `upstream.required_paths`,
  `upstream.missing_paths`, and `upstream.candidates`;
- `upstream.expected_commit`, `upstream.selected_commit`, and
  `upstream.commit_status` when the model entry declares a known upstream
  checkout;
- `assets`, each with `name`, `status`, `path`, `required`, and `runtime`.
- `python_modules`, each with `name` and `status`.

`inference_start` should include an `observation_summary` with image keys,
state keys, prompt, and session keys. It should not persist large image payloads
inside the trace.

`inference_end` should include action shape, `action_summary`, `action_contract`
when an action contract was checked, timing, memory, backend metadata, and
warnings. This event is the first validation point for moving a model entry from
an official-script reference path to a native product path.

`action_contract` should include:

- `status`: `ok` or `error`;
- `expected_shape`: `[action_horizon, action_dim_or_null]`;
- `observed_shape`;
- `finite`;
- `rectangular`;
- `errors`.

If the backend returns an invalid action chunk, emit an `error` with
`stage="action_contract"` and do not emit a successful `inference_end` or
`serve_request_end`.

## Serve Fields

`wam serve <model-id>` with a native backend should emit:

- `serve_start`
- `runtime_contract`
- `preflight`
- `backend_load_start`
- `backend_load`
- `backend_warmup`
- `reset`
- `serve_ready`
- `serve_request_start`
- `serve_request_end`
- `error` for failed requests
- `backend_close`

For blocked backend preflight, `serve_start` should emit
`runtime_contract`, `preflight`, `error.stage="preflight"`,
and `backend_close` without
`backend_load_start`.

`serve_start` should include:

- `output_dir`
- `optimization_profiles`
- `manifest_defaults`
- `known_gaps`

`serve_request_start` should include:

- `request_id`
- `payload_keys`

`serve_request_end` should include:

- `request_id`
- `status`: `ok` or `error`
- `action_horizon`
- `replan_steps`
- `action_chunk_shape`
- `action_summary`
- `action_contract` when an action contract was checked
- `timing.wall_ms`
- `memory`
- `backend_metadata`
- `warnings`

`backend_close` records the trace path and marks backend cleanup for the resident
server process.

## Workload Fields

Inference and step events should include:

- `action_horizon`
- `replan_steps`
- `action_chunk_len`
- `action_dim`
- `history_len`
- `observation_summary`
- `from_stale_chunk`

## Timing Fields

Native backend inference events are produced through the common
`infer_with_processor` spine, so every native model should report the same stage
shape instead of inventing backend-specific timing fields for preprocessing or
postprocessing.

Inference events should include:

- `preprocess_ms`
- `model_ms` for in-process model calls, or `server_ms` for resident policy
  servers
- `postprocess_ms`
- `total_ms`

Step events should include:

- `env_step_ms`

Remote or later optimization events may include:

- `cuda_ms`
- `client_ms`
- `serialization_ms`
- `queue_ms`
- `ttfa_ms`
- `partial_action_ms`
- `compile_time_s`
- `capture_time_s`
- `graph_replay_count`
- `cache_hit_rate`
- `cache_update_ms`

## Memory Fields

Memory samples should include:

- `cuda_allocated_mb`
- `cuda_reserved_mb`
- `cuda_peak_allocated_mb`
- `cuda_peak_reserved_mb`
- `process_rss_mb`

CPU-only environments should emit the same field names with `0` or `null`.

## Artifact Fields

Inference events that persist outputs should reference artifacts by path:

- `action_chunk_path`
- `action_chunk_shape`
- `action_summary`
- `future_frames`: JSON-safe summary or artifact reference.
- `future_frames_path`
- `value`: JSON-safe value estimate or artifact reference.
- `value_path`

Artifacts should be keyed by `(episode_id, step_id, replan_id)` where possible.

## Comparison Events

`wam compare` returns a comparison summary and may later emit
`comparison_result` into an output trace. The current CLI summary includes:

- `primary_metric`
- `baseline`
- `variant`
- `relative_change`
- `output_gate`
- `output_gate_passed`
- `output_gate_details`
- `runtime_contract_gate`
- `runtime_contract_gate_passed`
- `runtime_contract_gate_details`
- `decision`: `faster`, `slower`, `same`, `invalid`, or `not_comparable`
- `warnings`

This is a product benchmark result. It should be honest about missing data and
should never claim a speedup when output checks fail.

The baseline and variant summaries include trace path, run id, manifest id,
backend, processor, mode, optimization profile names, latency statistics,
runtime contract payload, action shapes, action summaries, optional
future-frame summaries, optional value summaries, and errors.
When both traces provide action summaries, the output gate also checks that
action values are finite and that summary drift stays under the configured
tolerance before reporting a speedup.
`output_gate_details` records the observed scalar drift for fields such as
`mean`, `min`, `max`, and `max_abs` when that data is available.
When either trace reports `future_frames` or `value`, the output gate also
checks that the other trace reports the same kind of output. Future-frame
summaries are compared after stripping artifact-path fields, so run-specific
file paths do not invalidate a comparison. Value summaries are compared with
the same numeric drift tolerance used by the current action-summary gate.

When either trace contains `runtime_contract`, `wam compare` also applies
a runtime contract gate. It permits differences in requested optimization profile
status, but treats backend, processor, workload, mode, supported optimization
set, runtime mode, runtime loader, model adapter, processor modality,
deployment, or backend config-key differences as invalid for a speedup claim.

## Error Fields

Error events should include:

- `stage`
- `error_type`
- `message`
- `recoverable`
- `backend`
- `trace_path` when the command has already opened a trace file before failing

## Schema Policy

- Prefer append-only changes after Phase A.
- Do not store backend-native tensors in traces.
- Store paths to large artifacts rather than embedding arrays.
- Record missing measurements explicitly as `null`.
