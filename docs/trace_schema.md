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
- `backend_warmup`
- `reset`
- `optimization_profile_status`
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

When one or more optimization profiles are enabled, the backend lifecycle may
emit `optimization_profile_status` twice: once with `stage="plan"` before
`runtime_contract` or backend load, and once with `stage="post_load"` after the
backend has loaded. The plan event records each requested profile, whether it is
enabled and declared supported, its params, target/scope when available, and a
runtime state such as `requested`, `planned`, `disabled`, or
`unsupported_by_manifest`. Only the post-load event should report `applied`, and
only when the backend has actually applied the hook to the loaded runtime.
Backends may report `fallback` with a `reason` when a planned profile cannot be
activated.

FastWAM `dit_cache` status uses `hook="fastwam_video_kv_cache"`. In the plan
stage it should be `planned` when requested for `fastwam-libero` or
`fastwam-robotwin`. In the post-load stage it should be `applied` only when the
loaded model exposes the cache-mode-aware `infer_action()` and MoT video K/V
cache hook; otherwise it should be `fallback` with a reason such as
`backend_not_loaded` or `cache_hook_unavailable`.

FastWAM `cuda_graph` status uses `hook="fastwam_cuda_graph_action_body"`. The
first implementation only attempts to capture the action-body
`mot.forward_action_with_video_cache()` path, and only when `dit_cache` is in
`video_kv` mode. Capture failure, non-CUDA execution, unsupported model
signatures, or shape changes should fall back to eager execution and remain
trace-visible.

FastWAM `torch_compile` status uses `hook="fastwam_torch_compile_action_body"`.
The first implementation only attempts to compile the same action-body callable
used by CUDA Graph. It is experimental and disabled by default; failures should
fall back to eager execution and remain trace-visible. Request metadata should
treat `torch_compile_success=true` as evidence that the compiled action-body
path remained usable for that request; a runtime fallback should set it back to
false and populate `torch_compile_fallback_reason`.

FastWAM `inference_end.backend_metadata` may include request-local cache
metadata:

- `dit_cache_enabled`
- `dit_cache_mode`: `video_kv` or `recompute`
- `cuda_graph_enabled`
- `cuda_graph_mode`: `off` or `auto`
- `cuda_graph_capture_success`
- `cuda_graph_replay_count`
- `cuda_graph_fallback_reason`
- `cuda_graph_shape_key`
- `torch_compile_enabled`
- `torch_compile_mode`: `off`, `auto`, `default`, `reduce-overhead`, or `max-autotune`
- `torch_compile_success`
- `torch_compile_fallback_reason`
- `torch_compile_wall_ms`
- `dit_cache_hook`: `fastwam_video_kv_cache`
- `num_inference_steps`
- `video_seq_len`
- `action_seq_len`
- `cache_layers`
- `cache_prefill_wall_ms`
- `denoise_wall_ms`
- `cache_bytes`

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

## Native Simulator Eval Fields

Native simulator evals use the same backend lifecycle and `inference_end`
events as `wam run`, plus simulator-specific orchestration events:

- `native_eval_plan` with the eval runner name, workload, task id, trial count,
  action horizon, replan steps, and planned command summary.
- `episode_start` / `episode_end` for each simulator episode.
- `replan_start` and `inference_start` before every model call.
- `simulator_wait_step` for optional no-op stabilization steps.
- `simulator_step` for each action applied to the simulator.
- `native_eval_end` with successes, total episodes, success rate, and the
  results JSON path.

For native FastWAM LIBERO single-task eval, `wam eval fastwam-libero` defaults
to this path. The official FastWAM scripts remain available only through
explicit `--reference`.

The acceptance verifier for this path is:

```bash
python -m eazywam.evals.acceptance --json SUMMARY_JSON EXPECTED_TRIALS MIN_SUCCESS_RATE
```

It checks the saved summary and trace rather than rerunning the model. A passing
native eval summary must not contain `external_eval_plan`, must contain
`native_eval_end`, must finish with `run_end.status="ok"`, and must point to an
existing eval results JSON file. The summary metrics, `native_eval_end`, and
results JSON must agree on total episodes, successes, and success rate; the
success rate must also meet `MIN_SUCCESS_RATE`. `--json` prints a structured
acceptance report with the summary, trace, results path, expected trials, and
success-rate gate. The summary must also include `runtime_info.backend` and
`runtime_info.mode`, and `runtime_info.backend` must not be `external_eval`.
The end-to-end wrapper persists that report as
`<model-id>-<workload>-acceptance.json` beside the eval summary.

## Native Smoke Fields

`wam native-smoke <model-id>` should emit the normal backend lifecycle events:

- `run_start`
- `optimization_profile_status` when profiles are enabled
- `runtime_contract`
- `preflight`
- `backend_load_start`
- `backend_load`
- `backend_warmup`
- `reset`
- `processor_smoke_observation`
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
  scope, target layer (`native_backend`, `workload`, or `deployment`), state,
  and optional runtime `hook` or fallback `reason`; pre-load contracts should
  use `requested` or `planned`, not `applied`;
- `deployment`;
- `backend_config_keys`;
- `processor_modality` when the registered processor exposes modality limits.

`processor_smoke_observation` should include an `observation_summary` after the
backend has loaded, warmed up, and reset. This lets processors that require
runtime binding generate a realistic smoke observation. If native readiness is
`blocked` before load, the trace should contain
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
