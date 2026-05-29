# WAM Systems Experiment Contract

## Purpose

WAM Harness is an inference systems harness for testing whether ideas from LLM
inference systems transfer to world-action model workloads.

The project is not a generic WAM serving framework. Its main job is to make a
systems idea testable:

- What pressure did the idea solve in LLM inference?
- Does the same pressure exist in WAM inference?
- Does the method still apply under WAM workload structure?
- What measurements and correctness checks decide whether it worked?

This document defines the one public harness contract. Backend-specific tensor
layouts, preprocessing details, transport protocols, and cache mechanics must
stay outside the core contract.

## Non-Goals

- Training support.
- Production serving.
- A full model zoo.
- A replacement tensor runtime or CUDA allocator.
- A universal robotics environment interface.
- Optimization switches without baseline, variant, and measurement evidence.

## Workload Units

The harness should describe every run using these units.

`run`

A complete experiment invocation with one config, one model target, one
workload target, and one output directory.

`episode`

A task rollout or open-loop sample sequence. Simulator-backed runs may contain
many episodes. Open-loop smoke tests may contain one synthetic episode.

`step`

One environment or replay step. A step may consume an action from a previously
generated action chunk without calling the model.

`replan`

A point where the harness calls the model to produce a new action chunk.
Replanning is the WAM analogue of a repeated inference request in an LLM
serving workload.

`inference_call`

One backend invocation. It includes preprocess, model execution, postprocess,
and optional cache update.

`action_chunk`

A sequence of actions produced by one inference call. The runner may execute
all or part of the chunk before replanning.

## Inference Contract

The harness input and output contract is intentionally small.

### Observation

Required fields:

- `images`: named image views. Examples: `primary`, `wrist`, `exterior_0`.
- `prompt`: task instruction or natural language goal.

Optional fields:

- `state`: named numeric vectors such as `proprio`, `joint_position`,
  `cartesian_position`, or `gripper`.
- `history`: previous observations, actions, or compact backend state.
- `session`: run, episode, step, and session identifiers.
- `metadata`: extra data that does not affect core runner behavior.

### Inference Request

Required fields:

- `observation`
- `action_horizon`
- `replan_steps`

Optional fields:

- `num_inference_steps`
- `return_future`
- `return_value`
- `reset`
- `cache_control`
- `runtime_options`

### Inference Result

Required fields:

- `action_chunk`: actions with shape-like metadata, usually `[T, D]`.

Optional fields:

- `future_frames`: predicted visual futures.
- `value`: scalar or vector value-style prediction.
- `warnings`: non-fatal backend warnings.
- `backend_metadata`: backend-specific diagnostics that do not change core
  behavior.
- `timing`: stage timing from the backend if available.
- `memory`: backend memory data if available.

## Execution Stages

Every backend should be observable through these stages, even if some are
no-ops.

`load`

Load model weights, processors, normalizers, runtime resources, or remote
client state.

`warmup`

Run optional dry-run inference so first-call effects can be separated from
steady-state measurements.

`reset`

Clear episode, session, action buffer, cache, or backend history state.

`preprocess`

Convert harness observations into backend-native inputs.

`infer`

Run the model or remote policy call.

`postprocess`

Convert backend-native outputs into harness results. This includes action
denormalization when needed.

`env_step`

Consume an action in a simulator, open-loop replay source, or future robot
interface.

`trace`

Write structured measurement events. Trace writing must not silently drop
errors.

## Measurement Contract

The harness exists to answer systems questions. Each run should record enough
data to separate workload pressure, method effect, and side effects.

### Latency

Minimum fields:

- `preprocess_ms`
- `model_ms`
- `postprocess_ms`
- `env_step_ms`
- `total_ms`

Optional fields:

- `cuda_ms`
- `server_ms`
- `client_ms`
- `serialization_ms`
- `queue_ms`

### Memory

Minimum fields:

- `cuda_allocated_mb`
- `cuda_reserved_mb`
- `cuda_peak_allocated_mb`
- `cuda_peak_reserved_mb`
- `process_rss_mb`

Optional fields:

- `memory_delta_mb`
- `fragmentation_hint`
- `snapshot_id`
- `graph_pool_bytes`
- `cache_bytes`

### Workload Shape

Minimum fields:

- `image_shapes`
- `state_dims`
- `history_len`
- `action_horizon`
- `replan_steps`
- `action_dim`

### Runtime

Minimum fields:

- `model_name`
- `backend`
- `source_repo`
- `mode`: `local`, `remote`, or `fake`.
- `device`
- `dtype`
- `optimization_flags`

### Correctness

Optimization variants must not be judged by speed alone. Each experiment should
declare the output comparison it can support.

Examples:

- exact match for fake backend outputs.
- numeric tolerance for deterministic action chunks.
- task success rate for simulator rollouts.
- bounded visual metric or artifact inspection for future frames.
- explicit "not comparable" only for exploratory measurements.

### Failures

Failures should be trace events, not only terminal logs.

Minimum fields:

- `stage`
- `error_type`
- `message`
- `recoverable`
- `backend`
- `run_id`
- `episode_id` when available.

## Systems Idea Experiment Contract

Every systems idea should be expressed as a baseline-vs-variant experiment.

Required fields:

- `idea`: short name.
- `pressure`: the underlying systems pressure the idea addresses.
- `existence_check`: how to determine whether that pressure exists in WAM.
- `method`: the mechanism being tested.
- `assumptions`: conditions required for the method to apply.
- `baseline`: settings without the idea.
- `variant`: settings with the idea.
- `metrics`: primary and secondary measurements.
- `correctness_gate`: how outputs are compared.
- `decision_rule`: what result counts as useful, neutral, or negative.

Example:

```yaml
idea: cuda_graph
pressure: Repeated static-shape GPU inference can spend meaningful time in CPU
  launch overhead and allocator churn.
existence_check:
  - repeated replan calls share image shapes, state dims, action horizon, and dtype
  - model_ms has non-trivial CPU launch or dispatch overhead
method: Capture a stable inference path and replay it after warmup.
assumptions:
  - no allocation inside capture unless routed through a graph-safe pool
  - shapes stay stable across replans
  - randomness and cache state remain controlled
baseline:
  cuda_graph: false
variant:
  cuda_graph: true
metrics:
  primary: [model_ms_p50, model_ms_p95]
  secondary: [warmup_s, cuda_peak_allocated_mb, capture_failure_rate]
correctness_gate: action_chunk numeric tolerance against eager baseline
decision_rule: Useful only if p95 model latency improves without unacceptable
  memory growth or output drift.
```

## Minimal Runtime Info

Runtime info should be small at first. It is not a full model taxonomy.

```yaml
runtime_info:
  name: fastwam-libero-2cam
  backend: fastwam
  source_repo: FastWAM
  mode: local
  device: cuda:0
  dtype: bf16
  optimization_flags:
    torch_compile: false
    cuda_graph: false
```

## Invariants

- The core runner must not branch on upstream repository names.
- Backend-specific keys and tensor layouts must not leak into the core
  contract.
- Every optimization experiment needs a baseline, variant, measurement set, and
  correctness gate.
- Memory observation is a first-class requirement, not debug output.
- Trace schemas should be append-only where possible.
- A faster variant that breaks outputs, increases memory beyond the declared
  limit, or only moves cost to another unmeasured stage is not a success.

## Phase 1 Scope

Phase 1 should prove the contract with a fake backend and open-loop workload.
It should not depend on real WAM checkpoints.

Phase 1 must support:

- one fake backend.
- one open-loop runner.
- JSONL traces.
- basic latency and memory observation.
- baseline-vs-variant experiment records.
- tests for trace shape and runner behavior.

Phase 1 must defer:

- FastWAM checkpoint loading.
- LIBERO and RoboTwin.
- remote server implementation.
- CUDA Graph and torch.compile execution.
- multi-GPU scheduling.
- dashboards.
