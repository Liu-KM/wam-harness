# Dependency Isolation Strategy

Backend dependency isolation is a first-class product problem for WAM Harness.
WAM/VLA repositories often bind together model code, simulator code, CUDA
versions, JAX/PyTorch versions, dataset tools, and robot-control utilities. A
usable Ollama-like WAM tool must prevent those dependencies from contaminating
the core package while still supporting users who cannot run containers.

## Principles

- The core package stays small and installable with `uv`.
- Heavy model dependencies are backend-specific.
- Simulator dependencies are workload-specific.
- Dockerfile-compatible recipes are the recommended reproducible backend
  environments when a container runtime is available.
- Backend-specific self-managed install scripts are supported when a machine or
  cluster cannot run Docker, Apptainer, Singularity, or an equivalent container
  launcher.
- Container recipes and self-managed install scripts should share the same
  dependency contract whenever possible.
- Runtime launchers are external. The same backend image may be started by
  a local container runtime, an existing cluster allocation, or a site-specific
  wrapper.
- Conflicting model and simulator environments should run as separate
  containers, dedicated virtual environments, or job-local processes instead of
  being forced into the core Python environment.
- `uv` remains the Python package manager inside the core image.

## Isolation Levels

### Level 0: Core

Contains:

- contract types.
- model spec parser.
- Hugging Face Hub asset preparation for declared `hf://` assets.
- registry.
- runner.
- trace writer.
- fake backend.
- CLI.

Must not depend on torch, JAX, CUDA, simulator packages, or upstream WAM repos.
The Hugging Face Hub client is acceptable in core because it prepares declared
assets without importing model runtimes.

### Level 1: Core Container

Contains Level 0 plus the OS/Python environment needed to run the CLI and fake
backend:

- base image.
- `uv`.
- project install.
- test and smoke commands.

This image should run with Docker locally and inside any cluster container
runtime that can mount the repository, cache directory, and run-output
directory.

### Level 2: Backend Container

One image per heavy backend family, for example `fastwam`, `openvla-vla-cache`,
or `openpi-faster`.

Use when:

- CUDA/compiler versions are strict.
- upstream dependencies are fragile.
- model load time justifies a persistent `wam serve` process.
- the backend needs a different Python/CUDA/JAX stack.

Backend containers still expose the harness contract: `wam run`, `wam serve`,
runtime info, traces, and optimization profiles.

Backend containers must install the WAM Harness CLI into the backend runtime
environment. Mounting only the repository or only the upstream project is not
enough: `wam doctor`, `wam native-smoke`, `wam run`, and `wam serve` must inspect
the same Python modules and assets that `backend.load()` will use.

### Level 2b: Self-Managed Backend Environment

One dedicated virtual environment per heavy backend family, for example
FastWAM. This is the fallback path when containers are unavailable.

Use when:

- the user has direct access to a GPU machine or existing allocation;
- the site does not allow Docker or compatible container runtimes;
- the user can install Python packages and clone upstream repositories;
- the backend dependency stack is stable enough to install in one environment.

Self-managed scripts must stay site-agnostic. They may create a `uv` virtual
environment, clone upstream repositories, install backend packages, configure
backend-local runtime files, and print the next `wam doctor` / `wam prepare`
commands. They must not submit scheduler jobs, choose queues, hard-code lab
paths, or download large model assets by default.

### Level 3: Split Container Serving

Run the heavy model server and the workload/simulator in separate job-local
processes or containers.

Use when:

- model dependencies conflict with simulator dependencies.
- the upstream project already exposes a policy server.
- FASTER/OpenPI-style streaming control needs a resident server.
- a simulator stack cannot coexist with the model stack.

External laptop-to-cluster endpoint access is optional later work. The first
target is job-local serving inside whatever environment launched the container.

## Scheduler Boundary

WAM Harness should not teach users how to submit jobs to every cluster. The
core project owns the command that runs after compute has been allocated:

```bash
wam run <model-id> --input /mnt/obs/obs.json --trace-dir /mnt/runs --upstream-dir /mnt/upstream
```

Site-specific wrappers may set up GPUs, queues, accounts, modules, container
mounts, or scratch paths before running that command. Those wrappers are
examples, not public API.

Reference simulator scripts use an explicit reference flag:

```bash
wam eval <model-id> --reference --trace-dir /mnt/runs --upstream-dir /mnt/upstream
```

## Recommended Defaults By Target

| Target | Initial isolation level | Reason |
|---|---|---|
| Fake backend | Level 1 | CI, local Docker smoke, and generic container smoke path. |
| FastWAM | Level 2 or 2b | Heavy torch/CUDA dependencies and checkpoint assets; container preferred, self-managed script supported. |
| VLA-Cache/OpenVLA | Level 2 | Requires forked `transformers` and OpenVLA-specific environment. |
| FASTER/OpenPI | Level 3 | JAX/OpenPI policy server and client split is already natural. |
| ServoFlow | Level 2 or 3 | C++/CUDA runtime and compiler requirements. |
| LIBERO/RoboTwin | Level 3 if simulator conflicts appear | Simulator dependency conflicts are likely. |

## Serve Minimum Contract

A served backend should expose:

- runtime metadata on start.
- health check.
- reset/session call.
- infer call.
- optional streaming partial-action messages.
- server timing fields.
- error payloads that can be written to trace.

The transport can start job-local. WebSocket/msgpack remains useful because
several WAM repositories already use that shape; HTTP may be added later for
simpler stateless calls.

## Open Decisions

- Whether backend containers live in this repository or as generated recipes.
- Whether `wam prepare` should prepare container images, or only model assets.
- Which license to use before publishing curated model entries.
- How strict checksum validation should be in the first model spec schema.
