# Phase 1 Plan

## Goal

Build the smallest runnable harness that proves the experiment contract with a
fake backend and open-loop workload.

Phase 1 is not about real WAM quality. It is about proving that the harness can
run a baseline-vs-variant experiment, emit structured traces, observe timing and
memory, and verify runner behavior before real model dependencies are added.

## Design Principles

- Implement the contract before integrating real WAM checkpoints.
- Use a fake backend to keep failures local to the harness.
- Make memory and timing visible from the first runnable version.
- Treat optimization ideas as experiments, not as boolean feature flags.
- Keep backend-specific logic out of the runner.

## Deliverables

### 1. Core Contract Types

Add lightweight Python types for:

- observation
- inference request
- inference result
- runtime info
- trace event
- experiment descriptor

These types should match `docs/contract.md`. They should not include FastWAM,
DreamZero, Cosmos, LingBot, Motus, or Qi specific fields.

### 2. Fake Backend

Add a backend that returns deterministic action chunks.

Purpose:

- test the runner without model dependencies.
- test action chunk scheduling.
- test baseline-vs-variant trace shape.
- provide exact correctness checks.

### 3. Open-Loop Workload

Add an open-loop workload that provides fixed observations without a simulator.

Purpose:

- avoid simulator dependency.
- make CI and local smoke tests cheap.
- provide a stable workload for trace tests.

### 4. Runner

Add a runner that:

- resets backend and workload.
- calls backend at replan points.
- stores pending actions.
- consumes action chunks over steps.
- emits trace events for run, episode, replan, inference, step, and errors.

### 5. Trace Writer

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

### 6. Observer

Record:

- wall-clock timing.
- optional CUDA memory stats when PyTorch CUDA is available.
- process RSS when supported.

The observer should degrade cleanly on CPU-only machines.

### 7. Minimal CLI

Add one command for open-loop runs.

Example target command:

```bash
uv run wam run --config configs/fake_open_loop.yaml
```

The exact command name can change during implementation, but the first CLI
should only run the fake open-loop path.

### 8. Tests

Add tests for:

- type serialization.
- fake backend determinism.
- action chunk scheduling.
- trace JSONL shape.
- CPU-safe memory observer behavior.
- baseline-vs-variant experiment descriptor validation.

## Explicit Non-Goals

Do not implement these in Phase 1:

- FastWAM checkpoint loading.
- LIBERO or RoboTwin integration.
- remote websocket server.
- CUDA Graph execution.
- torch.compile execution.
- multi-GPU scheduling.
- dashboards.
- real robot support.

## Success Criteria

Phase 1 is complete when:

- a fake open-loop run produces a JSONL trace.
- tests pass with no GPU required.
- trace events include workload units, timing, memory fields, runtime info, and
  experiment descriptor fields.
- the runner can compare a baseline and variant fake run with exact output
  equality.
- no real WAM dependency is required.

## After Phase 1

Phase 2 should add the first real local backend, likely FastWAM.

Phase 3 should add remote backend support based on server metadata and timing
fields.

Optimization experiments such as CUDA Graph and torch.compile should only start
after the real backend can produce reliable baseline traces.
