# Systems Idea Transfer Map

## Purpose

This document tracks LLM inference systems ideas that may transfer to WAM
inference. Each idea is written in terms of its underlying pressure,
assumptions, required instrumentation, and decision rule.

The goal is not to copy surface features from LLM serving frameworks. The goal
is to test whether the reason an idea worked for LLMs still exists in WAM
workloads.

## Role In The Research Pipeline

The actual unit of work in this repo is *reproducing one LLM systems paper onto
WAM*, tracked per paper through stages S0-S5 in `.collab/PIPELINE.md`. This
document is the **idea reserve pool**: each card below is a condensed example of
the understanding (S1), existence check (S2), and transfer design (S3) for one
idea. When the user brings a concrete paper, it gets its own `papers/<slug>/`
pipeline rather than a new entry here.

## Idea Card Template

Each idea should use this structure:

```text
Name
Pressure
WAM Existence Check
Method
Assumptions
Measurements
Correctness Gate
Failure Modes
Initial Phase
```

## CUDA Graph

Pressure:

Repeated static-shape GPU inference can spend meaningful time in CPU launch
overhead, dispatcher overhead, and allocator activity.

WAM Existence Check:

- Replan calls repeat with stable image shapes, state dimensions, action
  horizon, dtype, and device.
- The workload has enough repeated inference calls after warmup.
- Model execution contains launch-heavy paths where graph replay can remove
  CPU overhead.

Method:

Capture a stable inference path after warmup and replay it for later replans.

Assumptions:

- Shapes and control flow remain stable.
- Capture does not allocate unsafely.
- Randomness, cache state, and output buffers are controlled.
- Preprocess and postprocess are either outside the graph or graph-safe.

Measurements:

- `model_ms_p50`
- `model_ms_p95`
- `cuda_ms`
- `warmup_s`
- `capture_success`
- `replay_count`
- `capture_failure_rate`
- `cuda_peak_allocated_mb`

Correctness Gate:

Compare eager and graphed action chunks under deterministic inputs when
possible. For stochastic models, compare task-level behavior or declared output
tolerance.

Failure Modes:

- Shape instability prevents capture.
- Memory rises due to graph-private pools.
- Output drift appears because RNG or cache state is not controlled.
- The bottleneck is actually preprocess, postprocess, or environment step.

Initial Phase:

Do not implement in Phase 1. Phase 1 should only make the trace fields and
baseline-vs-variant structure ready.

## torch.compile

Pressure:

Eager PyTorch execution can pay Python overhead, graph dispatch overhead, and
miss fusion opportunities.

WAM Existence Check:

- The model forward path is stable enough for graph capture and compilation.
- Dynamic control flow, model-specific processors, or data-dependent shapes do
  not dominate the path.
- Compile cost can be amortized across replans or episodes.

Method:

Compile selected model functions and compare first-call, warmup, and
steady-state behavior against eager execution.

Assumptions:

- Dynamic graph breaks are limited.
- Compile time is measured separately.
- The compiled path preserves output behavior.
- Processor and environment logic are not accidentally included in the wrong
  measurement bucket.

Measurements:

- `compile_time_s`
- `first_call_ms`
- `steady_model_ms_p50`
- `steady_model_ms_p95`
- `graph_break_count`
- `cuda_peak_allocated_mb`

Correctness Gate:

Compare action chunks or downstream task outcomes against eager baseline.

Failure Modes:

- Compile time overwhelms short experiments.
- Graph breaks remove most benefit.
- Memory increases beyond acceptable bounds.
- Dynamic WAM control flow prevents stable compiled regions.

Initial Phase:

Not implemented in Phase 1.

## Cache And History Reuse

Pressure:

Repeated requests may recompute context that changes slowly. In LLMs this often
appears as prefix or KV-cache reuse. In WAMs it may appear as visual history,
language instruction, latent world state, or causal frame buffers.

WAM Existence Check:

- Consecutive replans share prompt, camera setup, and part of the observation
  history.
- The backend has a reusable internal state or cache boundary.
- Cache memory growth can be bounded.

Method:

Keep reusable state across replans or sessions and compare against stateless
inference.

Assumptions:

- Cache contents remain valid across the tested replan interval.
- Reset semantics are explicit.
- Cache hits and misses can be measured.
- The cache does not silently change model behavior.

Measurements:

- `cache_hit_rate`
- `cache_update_ms`
- `model_ms_p50`
- `model_ms_p95`
- `cache_bytes`
- `cuda_peak_allocated_mb`
- `output_drift`

Correctness Gate:

Compare cached vs uncached outputs or simulator success under controlled seeds.

Failure Modes:

- Cache is invalidated too often to help.
- Cache state leaks across episodes.
- Memory growth offsets latency improvement.
- Stale history worsens control behavior.

Initial Phase:

Trace fields only in Phase 1. Real cache tests come after real backend support.

## Remote Inference

Pressure:

Robotics simulators and WAM models can have conflicting dependencies. Heavy
models may also need separate GPU hosts, longer-lived processes, or isolated
runtime environments.

WAM Existence Check:

- Model environment conflicts with simulator or evaluation environment.
- Model load time is high enough to justify a persistent server.
- Serialization and network costs do not dominate the action loop.

Method:

Run model inference in a separate process or server and call it through a
structured protocol.

Assumptions:

- Request and response schemas are stable.
- Reset and session semantics are explicit.
- Server timing and client timing are both recorded.
- Serialization size is measured.

Measurements:

- `client_total_ms`
- `server_model_ms`
- `serialization_ms`
- `network_ms`
- `payload_bytes`
- `server_queue_ms`
- `server_cuda_peak_allocated_mb`

Correctness Gate:

Compare remote and local outputs where local execution is available. Otherwise
compare remote fake backend against local fake backend.

Failure Modes:

- Serialization overhead dominates.
- Server state is not reset correctly.
- Network jitter hides model improvements.
- Dependency isolation helps engineering but not latency.

Initial Phase:

Not implemented in Phase 1. Contract should leave trace fields ready.

## Action Chunk Scheduling

Pressure:

WAMs often emit action chunks. The harness must decide how many actions to
execute before replanning. This creates a systems tradeoff between model call
frequency, environment feedback freshness, and control quality.

WAM Existence Check:

- The backend emits multi-step action chunks.
- Replanning more or less often changes latency and task behavior.
- Environment feedback matters within a chunk.

Method:

Sweep `action_horizon` and `replan_steps` while holding model, task, seed, and
observation processing fixed.

Assumptions:

- The runner can record when an action came from a stale chunk.
- The workload can distinguish model latency from environment step latency.
- The action schema is stable.

Measurements:

- `model_calls_per_episode`
- `total_model_ms_per_episode`
- `env_step_ms`
- `success_rate`
- `chunk_utilization`
- `stale_action_steps`

Correctness Gate:

Task success, action validity checks, or fake backend deterministic traces.

Failure Modes:

- Fewer replans improve speed but reduce task quality.
- More replans improve feedback but increase model load.
- Chunk scheduling interacts with cache or graph assumptions.

Initial Phase:

Implemented only with fake backend and open-loop runner in Phase 1.

## Memory Supervision

Pressure:

WAM inference can be memory-heavy due to image/video inputs, latent world
states, diffusion or transformer internals, action/future predictions, caches,
and simulator coexistence. A speed optimization that raises memory too much may
not be useful.

WAM Existence Check:

- Peak memory approaches device capacity.
- Memory grows across episodes or replans.
- Optimization variants allocate additional persistent state.

Method:

Record memory at run, episode, replan, inference, and error boundaries. Compare
memory deltas between baseline and variant.

Assumptions:

- CUDA memory stats and process RSS are sampled consistently.
- Warmup and steady-state memory are separated.
- Peak memory is reset or scoped per run where possible.

Measurements:

- `cuda_allocated_mb`
- `cuda_reserved_mb`
- `cuda_peak_allocated_mb`
- `cuda_peak_reserved_mb`
- `process_rss_mb`
- `memory_delta_mb`
- `oom_count`

Correctness Gate:

Memory supervision does not change model output. It gates whether another idea
is acceptable.

Failure Modes:

- Reserved memory hides true pressure.
- Peaks are not reset between experiments.
- Simulator memory and model memory are not separated.
- Memory logging itself becomes too expensive.

Initial Phase:

Implement basic memory observer in Phase 1.

## Speculative Or Planning-Style Inference

Pressure:

LLM speculative decoding reduces the cost of generating sequences by proposing
tokens cheaply and verifying them. WAM analogues may involve proposing action
chunks, future rollouts, or candidate plans and selecting or verifying them.

WAM Existence Check:

- There is a cheap proposal path and an expensive verification path.
- Candidate action chunks can be compared or scored.
- Verification cost is lower than generating all candidates with the expensive
  path.

Method:

Represent candidate action chunks or future rollouts as variants, then compare
cost, acceptance rate, and task outcome.

Assumptions:

- The WAM output can be scored or verified.
- Candidate diversity is useful.
- Added complexity does not hide failure in task-level metrics.

Measurements:

- `proposal_ms`
- `verification_ms`
- `acceptance_rate`
- `candidate_count`
- `total_model_ms`
- `success_rate`
- `memory_peak`

Correctness Gate:

Task outcome or explicit verifier agreement. Action numeric equality is usually
not the right gate for this idea.

Failure Modes:

- Verification is as expensive as direct inference.
- Candidate scoring does not correlate with task success.
- More candidates improve quality but exceed memory or latency budget.

Initial Phase:

Research note only. Not implemented in Phase 1.
