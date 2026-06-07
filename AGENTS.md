# Agent Instructions

This repository is an open-source, Ollama-like deployment platform for
world-action model (WAM) inference.

Treat this branch as a product branch, not as the older literature-workflow
branch. The priority is a simple deployment experience:

```bash
wam list
wam info <model-id>
wam doctor <model-id>
wam prepare <model-id>
wam eval <model-id>
wam eval <model-id> --opt <profile>
wam serve <model-id>
```

## Current Stage

The repository is still in the skeleton and design-contract stage, but the next
implementation phase is approved for the deployment spine.

Allowed now:

- core package scaffolding.
- model entry/spec parsing.
- fake backend and open-loop runner.
- trace/telemetry writer.
- CLI entry point for `wam run fake-open-loop`.
- tests and project metadata.

Defer until the deployment spine works:

- real WAM checkpoint execution.
- simulator integration.
- external endpoint serving implementation.
- CUDA Graph, torch.compile, quantization, and other real optimization methods.
- multi-GPU scheduling.

## Product Direction

Use Ollama as the product reference for model UX: curated model ids, simple
model-library commands, prepare/eval/serve flow, and simple toggles.

Use vLLM and SGLang as references for framework boundaries: registry,
processor/backend separation, runtime information, and server metadata.

Use upstream WAM repositories as backend targets, not as code shapes to copy
into the core runner.

## Core Principles

- Provide a small public API and stable command-line entry points.
- Users select model ids; model entries map those ids to backends, processors,
  assets, defaults, and supported optimization profiles.
- Keep model-specific behavior behind registered backends.
- Treat model-specific preprocessing/postprocessing as processor logic.
- Keep runtime information minimal at first: model entry id, name, backend,
  processor, source repo, mode, device, dtype, and optimization profiles.
- Make optimization methods explicit profiles that can be enabled, disabled,
  traced, and compared.
- Prefer dependency isolation over forcing all WAM stacks into one environment.

## Backend And Processor Model

- `backend`: loads or connects to a WAM implementation and exposes the inference
  lifecycle.
- `processor`: converts between harness observations/results and
  backend-native model inputs/outputs.
- `registry`: maps model ids, backend names, processor names, workload names,
  and optimization profile names to implementations.
- `runtime_info`: minimal runtime metadata returned by a backend or remote
  server.

Avoid exposing backend-native names, tensor layouts, normalization details,
cache mechanics, or transport protocols through core harness interfaces.

## Runtime Information

Do not require users to hand-write a large capabilities file.

- Model entries record curated defaults and known gaps.
- The backend or server reports runtime support information.
- The processor declares modality limits and input requirements.
- Documentation records known gaps.

## Harness Boundary

The core harness should reason in contract terms:

- observations: images, state, prompt, history, session metadata.
- requests: observation plus inference controls.
- results: action chunks, optional future predictions, optional value estimates,
  warnings, backend metadata.
- traces: timing, memory, run metadata, model info, optimization profiles,
  warnings, errors.

The core runner must not contain branches for specific WAM repositories such as
FastWAM, Cosmos-Policy, DreamZero, LingBot-VA, Motus, or Qi.

## Environment Management

Use `uv` for core Python environment management, and use containers as the
default deployment boundary for heavy WAM stacks.

Preferred commands:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

Do not introduce conda, pip-tools, Poetry, or ad hoc environment scripts for the
core package unless explicitly requested. Heavy WAM dependencies should be added
through backend container recipes.

Do not make a specific cluster scheduler part of the core public contract. The
harness owns the container-internal `wam` command, model specs, traces, and
backend contracts. Users and sites decide whether that command is launched by
a local container runtime, an existing cluster allocation, or another wrapper.

## Initial Implementation Order

1. Core package scaffold.
2. Model spec parser and fake model entry.
3. Registry for model entries, backends, processors, workloads, and optimization
   profiles.
4. Fake backend and open-loop workload.
5. Runner and JSONL telemetry.
6. `wam run fake-open-loop`.
7. Core container recipe and generic container smoke path.
8. `wam serve fake-open-loop` inside a container or existing job allocation.
9. Tests and CI-ready commands.
10. FastWAM model entry/backend as the first real model.
11. VLA-Cache profile as the first real optimization toggle.

## Documentation Requirements

Keep these documents aligned before or during implementation:

- `docs/product_direction.md`
- `docs/contract.md`
- `docs/cli_entrypoints.md`
- `docs/wamfile.md`
- `docs/runtime_abstraction.md`
- `docs/dependency_isolation.md`
- `docs/roadmap.md`
- `docs/trace_schema.md`

## Human-Facing Local Artifacts

When an HTML artifact is useful for a report, comparison, plan, or visual
summary, keep it as concise as possible. Prefer a small self-contained page
over a polished microsite, and include only the structure needed for a human to
scan the result.

Store generated HTML artifacts under `.local/human-html/`, not in the repository
root or tracked documentation paths. `.local/` is ignored by git and is for
local human-readable outputs, scratch reports, and review pages.

Automation agents such as Codex and Claude should not treat `.local/human-html/`
as source context. Do not read or summarize those files unless the user
explicitly asks for a specific artifact.
