# Phase A Plan: Portable Deployment Spine

## Goal

Build the smallest runnable deployment spine: a fake model entry, fake
backend, open-loop workload, trace writer, optimization profile metadata, a
`wam run`-style CLI, and a portable smoke path.

Phase A is not about real WAM quality. It is about proving that the same public
path intended for real WAMs can resolve a model id, load a backend, run
inference, emit structured traces, observe timing and memory, carry optimization
profile metadata, and execute from a prepared runtime before heavy model
dependencies are added.

## Design Principles

- Implement the deployment spine before integrating real WAM checkpoints.
- Treat isolated backend runtimes as the deployment boundary from the start.
- Make the simple path easy: a model id should not require a large
  run config.
- Use a fake backend to keep failures local to the harness.
- Make memory and timing visible from the first runnable version.
- Treat optimization profiles as explicit deployment toggles, not hidden code
  edits.
- Still expose optimization profiles as explicit runtime toggles, following the
  inference-framework pattern where features are enabled through CLI/config and
  then inspected through telemetry and `wam compare`.
- Keep backend-specific logic out of the runner.

## Deliverables

### 1. Core Contract Types

Add lightweight Python types for:

- observation
- inference request
- inference result
- runtime info
- trace event
- model entry descriptor
- optimization profile descriptor

These types should match `docs/contract.md`. They should not include FastWAM,
DreamZero, Cosmos, LingBot, Motus, or Qi specific fields.

### 2. Model Spec Parser

Add a minimal model spec parser and one fake model entry.

Purpose:

- make `wam run fake-open-loop` resolve through the same path as future real
  models.
- define where backend, processor, default request shape, and optimization
  support live.
- keep user-facing defaults outside backend code.

### 3. Fake Backend

Add a backend that returns deterministic action chunks.

Purpose:

- test the runner without model dependencies.
- test action chunk scheduling.
- test trace shape with and without optimization profiles.
- provide exact output checks.

### 4. Open-Loop Workload

Add an open-loop workload that provides fixed observations without a simulator.

Purpose:

- avoid simulator dependency.
- make CI and local smoke tests cheap.
- provide a stable workload for trace tests.

### 5. Runner

Add a runner that:

- resets backend and workload.
- calls backend at replan points.
- stores pending actions.
- consumes action chunks over steps.
- emits trace events for run, episode, replan, inference, step, and errors.
- can run from a model entry plus optional CLI overrides.

### 6. Trace Writer

Write JSONL trace files.

Minimum events:

- `run_start`
- `episode_start`
- `backend_load`
- `backend_warmup`
- `replan_start`
- `inference_start`
- `inference_end`
- `step`
- `episode_end`
- `run_end`
- `error`

### 7. Observer

Record:

- wall-clock timing.
- optional CUDA memory stats when PyTorch CUDA is available.
- process RSS when supported.

The observer should degrade cleanly on CPU-only machines.

### 8. Minimal CLI

Add one command for open-loop runs through a model id.

Example target command:

```bash
uv run wam run fake-open-loop
```

The command may accept `--config` for advanced users, but the default path should
be model-id first.

### 9. Container Smoke Path

Add a Dockerfile-compatible core image recipe and a generic container smoke
path.

Purpose:

- prove the harness can run outside the developer's host environment.
- keep the future FastWAM/OpenVLA/VLA-Cache environments reproducible.
- keep scheduler-specific launch mechanics outside the core harness.

### 10. Tests

Add tests for:

- type serialization.
- fake backend determinism.
- action chunk scheduling.
- trace JSONL shape.
- CPU-safe memory observer behavior.
- optimization profile validation and trace serialization.
- model spec parsing and fake model resolution.

### 11. Optimization Profile Skeleton

Add config and trace support for runtime optimization profiles without executing
real model optimizations yet.

Purpose:

- prove that `wam run` can enable or disable named inference profiles.
- record exact profile parameters in `runtime_info` and trace events.
- keep future real integrations such as VLA-Cache, FASTER deployment controls,
  CUDA Graph, and torch.compile behind the same toggle contract.
- reject training-only methods as Phase A runtime toggles.

## Explicit Non-Goals

Do not implement these in Phase A:

- FastWAM checkpoint loading.
- LIBERO or RoboTwin integration.
- remote websocket server.
- CUDA Graph execution.
- torch.compile execution.
- real VLA-Cache/OpenVLA execution.
- FASTER/OpenPI policy-server execution.
- multi-GPU scheduling.
- dashboards.
- real robot support.

## Success Criteria

Phase A is complete when:

- `wam run fake-open-loop` produces a JSONL trace.
- the same fake run works from the core container path.
- tests pass with no GPU required.
- trace events include workload units, timing, memory fields, runtime info, and
  run descriptor fields.
- the runner can compare a baseline and variant fake run with exact output
  equality.
- no real WAM dependency is required.
- active optimization profiles appear in run metadata and trace output, even
  when Phase A uses fake no-op profiles.

## After Phase A

Phase B should make `wam serve fake-open-loop` run inside a container or
existing job allocation, with a job-local smoke check and trace output.

Phase C should add the first real backend, likely FastWAM, plus a curated model
entry such as `fastwam-libero`.

Phase D should run the first training-free inference optimization smoke test,
likely VLA-Cache on OpenVLA/OpenVLA-OFT if checkpoint and LIBERO setup are
available.

Optimization integrations such as CUDA Graph and torch.compile should only
start after the real backend can produce reliable baseline traces.
