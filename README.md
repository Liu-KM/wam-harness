# EazyWAM

EazyWAM is a deployment and acceleration harness for world-action models.

It turns scattered WAM checkpoints, runtimes, assets, eval scripts, serving
paths, optimization toggles, and traces into one model-centric workflow through
a small `wam` CLI.

## Why EazyWAM?

World-action model code is usually split across separate repositories,
checkpoints, environment recipes, asset layouts, simulator scripts, and ad hoc
evaluation commands. That makes it hard to answer a simple question:

> How do I run this WAM, and what acceleration profiles can I safely turn on?

EazyWAM is built around five design principles:

- **Model-centric use:** users select curated model ids such as
  `fastwam-libero`; model entries provide the backend, processor, assets,
  workloads, and defaults.
- **One action contract:** backend-specific tensor layouts, normalization, and
  simulator details stay behind processors and backends.
- **Explicit runtime boundaries:** heavy WAM stacks live in containers or
  backend-managed environments instead of polluting the core package.
- **Optimization as profiles:** acceleration methods are requested by name,
  applied through lifecycle hooks, and recorded as runtime state.
- **Traceable execution:** runs, evals, serves, warnings, errors, timings, and
  output summaries are written to inspectable traces.

The goal is not to replace upstream WAM projects. The goal is to make them
usable through one deployment and acceleration layer.

## Quickstart

The fake model path is the fastest way to check the CLI, registry, runner,
server, and trace pipeline on any machine.

```bash
uv sync --dev

uv run wam list
uv run wam info fake-open-loop
uv run wam run fake-open-loop
uv run wam run fake-open-loop --opt fake_cache
uv run wam serve fake-open-loop --smoke
```

For real WAMs, `wam run` requires an explicit observation input because the
model needs images, robot state, prompt text, and optional session/history
metadata. See `examples/fastwam_libero/obs.json` for the first FastWAM input
example.

```bash
uv run wam run fastwam-libero \
  --input examples/fastwam_libero/obs.json \
  --output /tmp/fastwam-action.json \
  --cache-dir /path/to/wam-cache
```

## Model Library

| Model id | Backend | Workload | Runtime | Status | Known gaps |
| --- | --- | --- | --- | --- | --- |
| `fake-open-loop` | `fake` | Open-loop smoke workload | CPU | Stable development path | Deterministic fake actions only; no real WAM weights. |
| `fastwam-libero` | `fastwam` | LIBERO run, serve smoke, and single-task eval | GPU container or self-managed env | Native product path started; vendored runtime and native smoke available | Requires checkpoint assets, Wan/DiffSynth files, LIBERO, MuJoCo, and simulator parity hardening. |
| `cosmos-policy-libero` | `cosmos_policy` | LIBERO evaluation path | GPU container recommended | Native smoke and official-script parity integration started | Requires Cosmos-Policy environment and large assets; resident serving is not productized yet. |
| `dreamzero-droid-sim` | `dreamzero` | DROID sim and policy-server path | Multi-GPU container recommended | Native smoke and server path started | Requires DreamZero/Isaac simulation assets and split runtime environments. |

Use the CLI for the source of truth:

```bash
uv run wam list
uv run wam info fastwam-libero
uv run wam doctor fastwam-libero --cache-dir /path/to/wam-cache
uv run wam prepare fastwam-libero --cache-dir /path/to/wam-cache --download --asset eval
```

## Core Concepts

- **Model entry:** curated YAML metadata for a model id. It records the backend,
  processor, assets, defaults, workloads, and supported optimization profiles.
- **Backend:** model-specific runtime adapter that loads or connects to a WAM
  implementation and exposes inference lifecycle hooks.
- **Processor:** conversion layer between EazyWAM observations/results and the
  backend-native input/output format.
- **Workload:** a repeatable way to drive inference, such as open-loop smoke,
  single-observation inference, or simulator evaluation.
- **Optimization profile:** an explicit runtime toggle such as cache reuse,
  action scheduling, CUDA Graph, `torch.compile`, or quantization.
- **Trace:** JSONL telemetry recording run metadata, runtime contracts, profile
  status, timings, memory, warnings, errors, and output summaries.

## Architecture

```text
model id
  -> model entry
  -> processor + backend
  -> runner/workload or serve app
  -> action result + trace or HTTP policy response
```

The core runner works in contract terms: observations, requests, results,
profiles, traces, and runtime metadata. Model-specific tensor layouts,
normalization, checkpoints, simulator quirks, and transport protocols stay
inside registered backends and processors.

## Optimization Profiles

Optimization profiles are EazyWAM's runtime interface for inference
acceleration. They make acceleration methods selectable, inspectable, and
comparable without asking users to edit backend source code. A user should
eventually be able to write:

```bash
wam run fastwam-libero --input obs.json --opt vla_cache
wam serve fastwam-libero --opt cuda_graph --opt torch_compile
```

Current status:

- profile names, defaults, plan/apply status, and trace payloads exist;
- fake/profile smoke coverage exists for the development path;
- `action_chunk_scheduling` is wired as an early profile contract;
- real behavior-changing acceleration profiles are not shipped yet.

Planned profile targets include VLA-Cache, action chunk scheduling, CUDA Graph,
`torch.compile`, post-training quantization, observation compression, kernel
fusion, and prefill/decode style scheduling where they make sense for WAMs.

## Current Status

EazyWAM is an early alpha.

- The fake path is stable enough for CLI, registry, run, serve, and trace
  development.
- FastWAM is the first real WAM target. The native product path has started,
  and official-script parity eval remains available for correctness checks.
- Heavy runtime setup is intentionally outside the core Python environment:
  use backend containers when possible, or backend-specific self-managed setup
  scripts when containers are unavailable.
- Real optimization methods are the next product milestone after the FastWAM
  path is stable.

## Roadmap

Near-term public alpha goals:

1. Polish the CLI, model library, and first-user documentation.
2. Harden the FastWAM prepare, run, eval, and serve paths.
3. Make optimization profiles behavior-changing, not just metadata.
4. Add the first real acceleration profile with trace-backed comparison.
5. Improve examples, failure diagnostics, and setup documentation.

## Documentation

- `docs/product_direction.md` - positioning and product goals.
- `docs/cli_entrypoints.md` - public command behavior.
- `docs/wamfile.md` - model entry schema and examples.
- `docs/backends.md` - backend support notes.
- `docs/fastwam_libero_eval_setup.md` - FastWAM setup and eval workflow.
- `docs/dependency_isolation.md` - containers and self-managed environments.
- `docs/optimization_integration.md` - optimization profile design.
- `docs/trace_schema.md` - trace event schema.
- `docs/roadmap.md` - phase plan and current milestones.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

The package is installed as `eazywam`, but the user-facing CLI remains `wam`.

## License

EazyWAM is released under the MIT License. Vendored third-party code and
external model assets remain under their respective upstream licenses.
