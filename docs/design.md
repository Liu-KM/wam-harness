# Design Notes

## Core Principle

The deployment spine should feel simple even when individual WAM systems are
messy. Users should interact with model names, model entries, and optimization
profiles; backend-specific preprocessing, dependencies, checkpoints,
normalizers, tensor layouts, and server protocols stay behind backends and
processors.

The telemetry system sits on top of the same spine. It should make optimization
profiles inspectable without turning every simple run into a formal comparison.

## Main Boundaries

- Core: contract types, runner loop, tracing, memory observation, model spec
  resolution, and registry.
- Model registry: maps curated model ids to model entries.
- Model backends: load or connect to a specific WAM and expose the inference
  lifecycle.
- Processors: convert between harness observations/results and backend-native
  data.
- Workloads: provide observations and consume actions for open-loop, simulator,
  and future hardware paths.
- Deploy: isolate heavy model dependencies through container recipes or
  backend-specific self-managed environments. Scheduler submission and
  site-specific launch policy stay outside the core harness.
- Optimization profiles: enable measured inference-time behavior without hidden
  code edits.

## Initial Execution Modes

1. `wam run` with fake open-loop inference.
2. `wam run` inside the core container path.
3. `wam serve` inside a prepared runtime or existing job allocation.
4. `wam eval` with a curated real model entry and isolated backend runtime.
5. LIBERO or RoboTwin simulator inference when dependencies are isolated.

## First Vertical Slices

`A: portable deployment spine`

Restore or rebuild minimal contract types, fake backend, open-loop runner, trace
writer, `wam run`, and the core container smoke path. This proves the product
spine without heavy dependencies.

`B: portable serve smoke`

Run `wam serve fake-open-loop` inside a prepared runtime, then drive a job-local
inference smoke check and write health/trace output.

`C: first real model`

FastWAM is the recommended first local backend because its inference path
already exposes action chunks and simulator evaluation code. The milestone is a
curated model entry such as `fastwam-libero` that can resolve checkpoint assets,
dataset stats, default request shape, and backend configuration.

`C: first real trick`

VLA-Cache is the recommended first optimization profile candidate because the
upstream path already exposes a direct `--use_vla_cache True/False` switch. The
milestone is a user-visible `--opt vla_cache` profile and a trace-backed
`wam compare` report.

## Stability Rules

- Add new WAM systems through backends.
- Add curated model entries through model specs.
- Add new simulators through workloads or environment integrations.
- Do not put model-specific logic into the runner.
- Do not treat CUDA Graph, torch.compile, cache reuse, or remote inference as
  supported without trace-backed telemetry and a way to disable the profile.
- Keep trace output append-only and machine-readable.
