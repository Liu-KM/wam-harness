# EazyWAM

EazyWAM is an open-source deployment and acceleration harness for world-action
models.

It makes WAMs easier to prepare, run, evaluate, serve, and optimize through a
small `wam` CLI.

The goal is to make WAM inference easy to run, easy to serve, and easy to
accelerate without turning the core project into a collection of upstream model
patches. A user should eventually be able to run or serve a curated model
through the native product path, then enable measured inference tricks through
explicit profiles such as `--opt vla_cache` or `--opt cuda_graph`.

The repository is an early alpha. Its first job is to make the model entry
workflow understandable before adding more heavy WAM checkpoints.

## Goals

- Provide a small local CLI for `wam list`, `wam info`, `wam doctor`,
  `wam prepare`, `wam eval`, `wam serve`, and `wam compare`.
- Maintain a curated WAM model registry with baked-in defaults through model
  entries.
- Run local or remote WAM backends behind one observation-to-action contract.
- Provide two environment paths for heavy WAM stacks: portable container
  recipes when a container runtime is available, and backend-specific
  self-managed install scripts when it is not.
- Expose inference-time optimizations through explicit profiles instead of
  hidden code edits.
- Record timing, memory, workload shape, runtime flags, and output checks so
  optimization toggles are inspectable.
- Keep model-specific preprocessing, normalization, tensor layouts, and server
  protocols behind backends and processors.
- Keep cluster scheduling outside the core project. Users and sites decide how
  to submit jobs; the harness owns the command, runtime boundary, model
  entries, traces, and model/runtime contracts.

## Product Direction

The project has two layers:

- **Deployment spine:** a simple user path for discovering a model entry,
  preparing its assets, running open-loop or simulator inference, and optionally
  serving a policy endpoint.
- **Telemetry layer:** trace, timing, memory, output artifacts, and comparison
  summaries for acceleration toggles.

See `docs/product_direction.md` for the full positioning.
See `docs/cli_entrypoints.md` for the public command design.
See `docs/next_goal.md` for the current execution target.
See `docs/runtime_abstraction.md` for the runtime boundary.

## Public Alpha CLI

The intended first-user path is:

```bash
wam list
wam info <model-id>
wam doctor [model-id]
wam prepare <model-id>
wam run <model-id> --input obs.json --output action.json
wam serve <model-id>
wam serve <model-id> --smoke --smoke-input obs.json
```

`prepare` is the public command for making a WAM model entry ready to use. It
creates cache directories, verifies declared assets, and reports manual
requirements. It does not install CUDA, Python environments, containers, or
cluster launchers. The project should not lead with infrastructure-style
`--dry-run` flows or require users to read internal YAML specs for the common
path.

Official simulator scripts are reference evaluators, not the default product
path. For maintainer parity checks, call them explicitly:

```bash
wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 1

wam eval fastwam-libero --reference --upstream-dir /path/to/FastWAM
wam eval fastwam-libero \
  --reference \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --upstream-dir /path/to/FastWAM
```

`fastwam-libero` is the model id. `libero-single-task` is an eval workload: it
selects the curated LIBERO single-task simulator loop without changing the
checkpoint or action contract. `--reference` switches the same workload to the
official upstream script for parity/debugging.

For real WAMs, `wam run` requires an explicit observation input. WAM inference
needs images, robot state, prompt, and optional session/history; the CLI does
not silently invent those inputs. Use `wam native-smoke <model-id>` for
maintainer synthetic-observation smoke tests.

FastWAM's first product-path input example lives at
`examples/fastwam_libero/obs.json`:

```bash
wam run fastwam-libero \
  --input examples/fastwam_libero/obs.json \
  --output /tmp/fastwam-action.json \
  --cache-dir /path/to/wam-cache

wam serve fastwam-libero \
  --smoke \
  --smoke-input examples/fastwam_libero/obs.json \
  --cache-dir /path/to/wam-cache
```

FastWAM product-path `doctor`, `run`, `native-smoke`, and `serve` use the
FastWAM runtime vendored in this package. `--upstream-dir` is only needed when
you intentionally run the official reference evaluator or debug against a
separate FastWAM checkout.

## Optimization Profiles

Optimization integrations are deployment-first. Training-free inference methods
such as cache reuse, action scheduling, CUDA Graph, torch.compile, and
post-training quantization are prioritized before training recipes or new model
architectures. See `docs/optimization_integration.md`.

## Non-Goals For The First Version

- Training.
- Real robot hardware control.
- Exhaustive model zoo coverage.
- Full benchmark dashboards.
- Multi-GPU scheduling.
- Model reimplementation.
- Replacing Hugging Face Hub or upstream checkpoint distribution.

## Current Layout

```text
src/eazywam/
  cli.py       Command-line entry point and command dispatch.
  serve.py     Local HTTP policy server.
  core/        Stable harness abstractions, runners, traces, and model entries.
  backends/    Model-specific backend adapters.
  processors/  Backend-specific input/output conversion.
  workloads/   Open-loop and single-observation workload drivers.
  manifests/   Curated model-entry YAML files.
  compat/      Small compatibility shims for upstream evaluator entrypoints.

configs/      Run, model, and optimization configuration examples.
containers/   Dockerfile-compatible backend runtime recipes.
examples/     Minimal inputs for smoke tests.
tests/        Unit and integration tests.
docs/         Design notes and schemas.
scripts/      Repository maintenance scripts.
```

Empty future packages are intentionally not kept in the source tree. CLI and
server code can move into `cli/` or `deploy/` packages when their surface area
is large enough to justify the split; until then, the README describes the
working code rather than a future architecture sketch.

The first runnable vertical slices are:

1. **A: portable deployment spine** - restore minimal contract types, fake
   backend, open-loop runner, trace writer, `wam run`, and a core container
   smoke path.
2. **B: portable serve smoke** - run `wam serve fake-open-loop` inside a
   container or existing job allocation and record health/runtime telemetry.
3. **C: first real model** - add a curated FastWAM model entry and backend
   container path that can prepare released checkpoint assets and emit actions.
4. **D: first real trick** - expose one real training-free inference
   optimization, likely VLA-Cache, as an on/off profile with trace-backed
   comparison.

Local development should keep the core package cheap to test. Heavy-model
validation can happen on any suitable internal GPU environment, but no specific
cluster should become a public dependency or the main abstraction. The public
target is: run the same `wam` command inside any environment that provides the
needed GPU runtime, backend dependencies, cache directory, and model assets.

## Reference Systems

The initial harness design is informed by:

- FastWAM: local action chunk inference and simulator evaluation patterns.
- DreamZero: WebSocket policy server and session-aware remote inference.
- Cosmos-Policy: unified action, future prediction, and value-style outputs.
- LingBot-VA: remote inference, cache control, and server timing metadata.
- Motus: standalone image-to-action smoke testing.
- Qi: optional inference optimization ideas such as CUDA Graph and torch.compile.
- Ollama: local model UX, simple model library commands, one-command run/serve
  flow, and default model metadata.

## Status

Phase A implementation has started. The current runnable target is the built-in
fake model:

```bash
uv run wam run fake-open-loop
uv run wam run fake-open-loop --opt fake_cache
uv run wam serve fake-open-loop --smoke
```

These commands exercise the same model-entry registry, backend, workload,
runner, and trace path that future real WAM backends should use.

## Environment

This project uses `uv` for environment management.

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

The project is packaged through `src/eazywam` and exposes the `wam` CLI.

There are two supported setup paths:

- **Core development environment:** use `uv sync --dev` and run fake/backend
  contract tests locally.
- **Heavy backend runtime:** either build a backend container image, or install
  the backend's self-managed environment script on a machine where containers
  are not available.

Build the core container image locally with:

```bash
docker build -f containers/core/Dockerfile -t eazywam-core:latest .
```

Build the FastWAM backend image when Docker or a compatible cluster container
runtime is available:

```bash
docker build -f containers/fastwam/Dockerfile -t eazywam-fastwam:latest .
```

Install the FastWAM runtime directly when containers are not available:

```bash
scripts/setup_fastwam_native_env.sh \
  --venv /path/to/.venv-fastwam \
  --cache-dir /path/to/wam-cache \
  --clone
```

Then activate the environment and use the normal `wam` commands:

```bash
source /path/to/.venv-fastwam/bin/activate
wam doctor fastwam-libero --cache-dir /path/to/wam-cache
wam prepare fastwam-libero --cache-dir /path/to/wam-cache --download --asset eval
```

Cluster users should launch prepared images or activated self-managed
environments with their site's normal tooling. Site-specific submission scripts
belong outside the public package.

## License

EazyWAM is released under the MIT License. Vendored third-party code and
external model assets remain under their respective upstream licenses.
