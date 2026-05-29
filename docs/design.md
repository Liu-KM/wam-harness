# Design Notes

## Core Principle

The harness core should stay stable while individual WAM systems vary behind
backends. The project is a systems experiment harness, not a generic serving
platform.

## Main Boundaries

- Core: contract types, runner loop, tracing, memory observation, registry.
- Model backends: load or connect to a specific WAM and expose the harness
  inference lifecycle.
- Workloads: provide observations and consume actions.
- Deploy: isolate heavy model dependencies through a server-client interface.
- Configs: define experiments without changing implementation code.

## Initial Execution Modes

1. Open-loop local inference.
2. Open-loop remote inference.
3. LIBERO simulator inference.
4. RoboTwin simulator inference.

## First Real Backend Target

FastWAM is the recommended first local real backend because its inference path
already exposes action chunks and simulator evaluation code.

Before FastWAM, Phase 1 should implement a fake backend and open-loop workload
so the experiment contract, trace schema, and runner behavior can be tested
without model dependencies.

## Stability Rules

- Add new WAM systems through backends.
- Add new simulators through workloads or environment integrations.
- Do not put model-specific logic into the runner.
- Do not treat CUDA Graph, torch.compile, cache reuse, or remote inference as
  successful without baseline-vs-variant measurements and correctness gates.
- Keep trace output append-only and machine-readable.
