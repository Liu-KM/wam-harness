# Runtime Abstraction

EazyWAM owns the WAM command contract, not the user's scheduler.

The public goal is that a user can run the same model entry through the same
`wam` command in any environment that already provides the required compute,
storage, and backend runtime:

```bash
wam prepare <model-id>
wam run <model-id> --input obs.json
wam serve <model-id>
```

The environment may be a local workstation, a local container, an existing
cluster allocation, or another site-specific launcher. Those launchers are
outside the core harness.

## Harness Responsibilities

The repository should provide:

- stable CLI commands such as `wam prepare`, `wam run`, and `wam serve`;
- model entries with curated defaults, assets, known gaps, and supported
  optimization profiles;
- backend and processor boundaries for model-specific code;
- container recipes and self-managed install scripts for heavy backend
  environments;
- a standard cache/run/artifact layout inside the chosen runtime;
- trace files that record runtime metadata, success/failure, warnings, and
  output locations;
- small smoke paths that prove a model entry works.

For real WAMs, "backend and processor boundaries" means the harness owns the
inference lifecycle, not a long-running call to an upstream evaluation script.
Official scripts remain reference evaluators. Product inference should flow
through a native backend that loads or connects to the model, a processor that
packs observations and unpacks action chunks, and the common native inference
spine that records timing and metadata consistently.

## User Or Site Responsibilities

Users and cluster operators decide:

- how to request GPUs;
- how to submit jobs;
- which scheduler, queue, account, partition, or reservation to use;
- which container launcher is allowed on their site;
- whether they need a self-managed Python/CUDA environment instead of a
  container;
- where shared scratch/cache storage lives;
- how to expose or block network endpoints;
- whether long-running services are allowed.

These details vary too much across institutions to be a core EazyWAM
contribution.

## Runtime Profiles

The project may document or test several runtime profiles, but they should all
reduce to the same `wam` command contract:

| Profile | Purpose | Harness expectation |
|---|---|---|
| `local-uv` | Cheap core development and fake backend tests. | `uv run wam ...` works without heavy WAM dependencies. |
| `local-docker` | Workstation smoke tests and portable demos. | The project is mounted or installed, caches are mounted, and `wam ...` runs inside the image. |
| `cluster-container` | Any managed GPU allocation. | The site launches a prepared image and calls `wam ...` with mounted cache/run directories. |
| `self-managed-backend` | GPU machines where containers are unavailable. | A backend-specific virtual environment is activated and `wam ...` runs with the same cache/run paths. |
| `site-recipe` | Maintainer-owned examples for a specific cluster. | Kept as examples only; not the public API. |

## Standard Runtime Shape

Backend runtimes should expose a predictable shape:

```text
wam CLI                 installed in the active environment
cache directory         model/checkpoint/cache storage
run directory           traces, logs, videos, and artifacts
upstream directory      optional upstream repository checkout
```

Container examples use `/workspace/eazywam`, `/mnt/wam-cache`, `/mnt/runs`,
and `/opt/<backend>`. Self-managed environments may use any absolute paths as
long as the same paths are passed to `wam doctor`, `wam prepare`, `wam run`,
`wam native-smoke`, `wam eval`, and `wam serve`.

## Design Rule

Do not add scheduler-specific assumptions to core Python modules, model specs,
or public CLI behavior. Scheduler examples may live under scripts or examples, but
the core contract must be expressible as:

```bash
wam run <model-id> --input /mnt/obs/obs.json --trace-dir /mnt/runs --upstream-dir /mnt/upstream
```

Everything before that command is environment provisioning.

Official simulator evaluators remain available for parity with explicit
reference intent:

```bash
wam eval <model-id> --reference --trace-dir /mnt/runs --upstream-dir /mnt/upstream
```
