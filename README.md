# WAM Harness

WAM Harness is a lightweight inference systems harness for world-action models.

The project is currently a repository skeleton. It is intended to test whether
LLM inference systems ideas transfer to WAM inference workloads while keeping
model-specific code behind backends.

## Goals

- Run repeatable WAM inference experiments.
- Compare baseline and variant runs for systems ideas.
- Record timing, memory, workload shape, runtime flags, and correctness gates.
- Keep model-specific backends separate from the core runner.
- Start with fake and open-loop workloads before adding real WAM checkpoints.

## Research Workflow

The unit of work is reproducing one LLM inference-systems paper onto WAM
inference, driven per paper through a six-stage pipeline: understand the source
idea, check whether its pressure exists in WAM, design the transfer, reproduce
it, then write up new findings. See `.collab/PIPELINE.md` for the pipeline and
`.collab/WORKFLOW.md` for the Claude/Codex collaboration model.

## Non-Goals For The First Version

- Training.
- Full benchmark dashboards.
- Real robot deployment.
- Multi-GPU scheduling.
- Model reimplementation.

## Planned Layout

```text
src/wam_harness/
  core/       Stable harness abstractions.
  backends/   Model-specific backend layer.
  processors/ Backend-specific input/output conversion.
  envs/       Open-loop, simulator, and future robot environment integrations.
  deploy/     Server-client protocol and remote policy utilities.
  cli/        Command-line entry points.

configs/      Experiment configuration examples.
examples/     Minimal inputs for smoke tests.
tests/        Unit and integration tests.
docs/         Design notes and schemas.
scripts/      Repository maintenance scripts.
```

## Reference Systems

The initial harness design is informed by:

- FastWAM: local action chunk inference and simulator evaluation patterns.
- DreamZero: WebSocket policy server and session-aware remote inference.
- Cosmos-Policy: unified action, future prediction, and value-style outputs.
- LingBot-VA: remote inference, cache control, and server timing metadata.
- Motus: standalone image-to-action smoke testing.
- Qi: optional inference optimization ideas such as CUDA Graph and torch.compile.

## Status

No implementation code has been added yet. This repository only defines the
initial open-source project skeleton and systems experiment contract.

## Environment

This project uses `uv` for environment management.

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

The current repository is configured as a non-package project until
implementation files are added under `src/wam_harness`.

## License

The license has not been selected yet. Choose a license before making the
repository public as open source.
