# Agent Instructions

This repository is an open-source inference systems harness for world-action
models (WAMs). Treat it as a research artifact for testing systems ideas, not
as a collection of one-off scripts.

## Current Stage

The repository is still in the skeleton and design-contract stage.

- Do not add implementation code unless the user explicitly approves the next
  implementation phase.
- It is acceptable to add documentation, schemas, configuration examples, tests,
  and project metadata when they clarify the public contract.
- Keep the repository suitable for later publication as an open-source project.

## Design Direction

Use vLLM and SGLang as references for framework boundaries, and VibeTensor as a
reference for research-system discipline: correctness gates, observability,
baseline-vs-variant tests, and explicit failure modes.

Follow these principles:

- Provide a small public API and stable command-line entry points.
- Keep model-specific behavior behind registered backends.
- Treat model-specific preprocessing/postprocessing as processor logic.
- Keep runtime information minimal at first: name, backend, source repo, mode,
  device, dtype, and optimization flags.
- Prefer explicit experiment boundaries over per-model script compatibility.
- Require each optimization idea to define pressure, existence check, method
  assumptions, measurements, correctness gate, and decision rule.

Do not design the harness as a set of patches around each upstream WAM
repository. The harness owns one contract; backends conform to it.

## Backend And Processor Model

Use the terms `backend` and `processor` for future implementation design:

- `backend`: loads or connects to a WAM implementation and exposes the harness
  inference lifecycle.
- `processor`: converts between harness observations/results and
  backend-native model inputs/outputs.
- `registry`: maps model/backend names to backend implementations.
- `runtime_info`: minimal runtime metadata returned by a backend or remote
  server.

Avoid exposing backend-native names, tensor layouts, normalization details,
cache mechanics, or transport protocols through core harness interfaces.

## Runtime Information

Do not require users to hand-write a large `capabilities.yaml`.

Instead, follow the vLLM/SGLang style:

- The backend or server reports model support information.
- The registry knows which backend owns which target.
- The processor declares modality limits and input requirements.
- Documentation records known gaps.

User configuration may select a model/backend, but it should not be the primary
source of truth for what the model can do.

## Harness Boundary

The core harness should reason in contract terms:

- observations: images, state, prompt, history, session metadata
- requests: observation plus inference controls
- results: action chunks, optional future predictions, optional value estimates,
  warnings, backend metadata
- traces: timing, memory, run metadata, model info, errors

The core runner must not contain branches for specific WAM repositories such as
FastWAM, Cosmos-Policy, DreamZero, LingBot-VA, Motus, or Qi.

## Reference Lessons

Borrow these ideas from mature inference frameworks:

- vLLM: model registry, public API stability, supported model matrix,
  multimodal processor boundaries, and model-specific implementation isolation.
- SGLang: server/runtime information endpoints, model support registration,
  multimodal processing separation, and OpenAI-compatible public serving style.
- VibeTensor: measurement-first systems research, allocator and CUDA Graph
  observability, eager-vs-variant parity tests, and explicit warnings about
  local correctness not guaranteeing global performance.

Borrow these ideas from WAM repositories:

- FastWAM: action chunk inference and replan-loop behavior.
- DreamZero: remote policy server metadata, reset semantics, and session-aware
  inference.
- Cosmos-Policy: richer outputs such as actions, future predictions, and value.
- LingBot-VA: server timing, health checks, and cache-control concepts.
- Motus: standalone smoke-test style.
- Qi: optional optimization hooks such as CUDA Graph and torch.compile.

## Environment Management

Use `uv` for environment management.

Preferred commands:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

Do not introduce conda, pip-tools, Poetry, or ad hoc environment scripts unless
the user explicitly requests a change. Heavy WAM dependencies should be added as
optional extras or backend-specific dependency groups after the public contracts
are stable.

## Implementation Constraints

When implementation begins:

- Start with a fake backend and open-loop runner before integrating real WAM
  checkpoints.
- Add FastWAM as the first real local backend.
- Add remote backend support before deep integration with server-heavy systems.
- Keep CUDA Graph, torch.compile, multi-GPU scheduling, and benchmark dashboards
  optional and out of the initial core.
- Add tests proportional to the public contract being introduced.
- Do not treat an optimization as successful without a baseline, variant,
  correctness gate, and trace-backed measurement.

## Documentation Requirements

Before substantial implementation, add or update:

- `docs/contract.md`
- `docs/systems_ideas.md`
- `docs/phase1_plan.md`
- `docs/trace_schema.md`

These documents should define the systems experiment contract before code
depends on it.
