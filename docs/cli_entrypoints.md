# CLI Entry Points

WAM Harness should feel like a local model tool, not a cluster or systems
framework. The first user-facing commands should be few, plain, and ordered
around what a new user naturally needs to do.

## Primary User Flow

```bash
wam list
wam info fastwam-libero
wam doctor fastwam-libero
wam prepare fastwam-libero --cache-dir /mnt/wam-cache
wam run fastwam-libero --input obs.json --output action.json --cache-dir /mnt/wam-cache --trace-dir runs
wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 1 --cache-dir /mnt/wam-cache
wam serve fastwam-libero --cache-dir /mnt/wam-cache --trace-dir runs
```

These are the public words. Internal implementation may still use terms such as
manifest, backend, processor, or trace, but users should not need those concepts
to run a model.

Maintainers may also use migration commands while moving a real model from an
official-script reference path to a native backend. Those commands should stay
out of the first user-facing flow.

## Command Meanings

| Command | User meaning | Product intent |
|---|---|---|
| `wam list` | Show available model entries. | Make the model library visible. |
| `wam info <model>` | Explain what this model is and what it needs. | Translate internal YAML into readable model information. |
| `wam doctor [model]` | Check whether this machine or runtime can run WAM Harness, optionally for one model. | Diagnose missing GPU, cache path, required assets, and native backend repo mounts without modifying the environment. |
| `wam prepare <model>` | Prepare this model's assets for use. | Create cache directories, verify declared assets, and explain remaining manual requirements. |
| `wam run <model> --input obs.json` | Run one explicit observation through the product path. | Refuse to invent a WAM observation; one-shot inference uses the same observation contract as serve. |
| `wam eval <model>` | Run a curated simulator evaluation for this model. | Make real simulator smoke/full-suite runs easy while still tracing the command, environment, and outputs. |
| `wam serve <model>` | Start a local or job-local policy server. | Keep a model resident for repeated observation-to-action calls. |
| `wam compare <run-a> <run-b>` | Compare two recorded runs. | Report latency, memory, output drift, and optimization profile differences. |

For FastWAM, the first real simulator workload is the official LIBERO
single-task smoke path:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --upstream-dir /mnt/upstreams/FastWAM
```

Explicit reference mode stays available for parity runs and comparisons:

```bash
wam eval fastwam-libero --reference --upstream-dir /mnt/upstreams/FastWAM
wam eval fastwam-libero \
  --reference \
  --workload libero-single-task \
  --task-id 3 \
  --num-trials 1 \
  --upstream-dir /mnt/upstreams/FastWAM
```

The model id remains `fastwam-libero`. Short eval modes such as
`libero-single-task` are workloads, not separate model identities, because they
reuse the same checkpoint, processor, and action schema.

`wam eval` is a simulator workflow, not a resident inference backend. `wam run
--input` and `wam serve` keep using the native observation-to-action product
path; `wam eval` runs the curated simulator workload declared by the model
entry and records stdout/stderr plus trace metadata.

## Compare Output

`wam compare <baseline-trace-or-dir> <variant-trace-or-dir>` reads `trace.jsonl`
files and reports a JSON summary. A path may point directly to `trace.jsonl` or
to a run directory containing that file.

Current comparison is intentionally conservative:

- primary metric: mean latency from `inference_end`, `serve_request_end`, or
  `external_eval_end`;
- output gate: action chunk shape equality when both traces provide shape data;
  if both traces also provide `action_summary`, non-finite action values or
  summary drift beyond `--max-action-drift` make the comparison invalid; if
  traces provide `future_frames` or `value`, compare also checks those JSON-safe
  summaries while ignoring run-specific artifact paths;
- native contract gate: if either trace contains `native_runtime_contract`, the
  native backend/processor/workload contract must match except for requested
  optimization profile status;
- decision: `faster`, `slower`, `same`, `invalid`, or `not_comparable`;
- speedup is never reported if the action shape gate fails, the native contract
  gate fails, or the action shape gate is unavailable.

Example:

```bash
wam compare runs/baseline/trace.jsonl runs/variant/trace.jsonl --max-action-drift 0.001
```

## Serve Request Shape

`wam serve <model>` exposes a small JSON policy endpoint:

```bash
curl -X POST http://127.0.0.1:8000/infer \
  -H 'content-type: application/json' \
  -d '{
    "observation": {
      "images": {"primary": [[[0, 0, 0]]]},
      "state": {"proprio": [0.0]},
      "prompt": "open the drawer",
      "session": {"episode_id": 0, "step_id": 0}
    },
    "action_horizon": 8,
    "replan_steps": 4
  }'
```

Normal `/infer` requests must include an observation. Empty requests are only
accepted by `wam serve <model> --smoke`, where the server is started in an
internal health-check mode that uses the registered processor's
`smoke_observation()`. For reference entries that declare
`backend.config.native_backend`, `wam serve` maps the model entry to
`mode: native_serve`; it does not try to run an official simulator script as a
server.

When the backend container or job mounts upstream source outside the model
entry's default path, pass it explicitly:

```bash
wam serve fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --upstream-dir /mnt/upstreams/FastWAM
```

`/health` returns the resident run id, runtime info, and trace path. Each
`/infer` call emits `serve_request_start` and `serve_request_end`; bad requests
emit an `error` event and a failed `serve_request_end`.

## Maintainer Migration Commands

`wam native-smoke <model>` is for native backend bring-up. It temporarily maps a
curated reference model entry such as `fastwam-libero` to its native backend,
creates one synthetic contract observation, and runs `load/warmup/reset/infer`
under trace.

The mapping is declared in the model spec with `backend.config.native_backend`.
The synthetic observation is supplied by the registered processor, so adding a
new backend target should not require changes in the CLI or core smoke runner.

Example:

```bash
wam native-smoke fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --upstream-dir /path/to/FastWAM
```

`native-smoke` writes native readiness before model load. If readiness is
`blocked`, it fails immediately with a `native_preflight` trace error instead of
trying to import or load the upstream model. Add `--require-ready` for stricter
container smoke runs where runtime assets such as tokenizer or model-base caches
must also be present before load:

```bash
wam native-smoke fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --upstream-dir /path/to/FastWAM \
  --require-ready
```

This command is deliberately not named `run` or `eval`: it is not a real
benchmark, not a simulator evaluation, and not the polished user path. Its job
is to prove that a backend can execute a checkpoint under harness control before
the public model entry switches away from `external_eval`. A successful native
smoke must also pass the native action contract: non-empty rectangular actions,
finite values, the requested action horizon, and the declared action dimension
when the model entry provides one.

`wam run <model> --input obs.json` is the user-facing one-shot inference path.
The input file must contain an observation object with images, optional state,
prompt, history, session, and metadata. When a reference model entry declares
`backend.config.native_backend`, `run` maps it to `mode: native_run` and uses a
single external observation workload. It does not execute the official evaluator
and does not use synthetic smoke observations.

```bash
wam run fastwam-libero \
  --input obs.json \
  --output action.json \
  --cache-dir /mnt/wam-cache \
  --upstream-dir /mnt/upstreams/FastWAM
```

If `--input` is omitted for a real WAM, `wam run` prints the next choices:

```bash
wam run fastwam-libero --input obs.json --output action.json
wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 1
wam serve fastwam-libero
wam native-smoke fastwam-libero
```

This is intentional. Unlike text-only LLMs, a WAM cannot infer without a real
observation: images, robot state, prompt, and usually session context.

For a cheap backend migration smoke that uses the processor's synthetic
observation, use `wam native-smoke <model>` instead.

## Why `prepare` Instead Of `pull`

`pull` is familiar from Docker and Ollama, but WAMs are not just one model blob.
A runnable WAM entry may need checkpoints, dataset statistics, tokenizer or VAE
components, simulator assets, upstream code, and a backend container. `prepare`
sets the right expectation only if its boundary is strict: it prepares model
assets and cache state, or tells the user what is still missing.

`prepare` does not configure the machine. It must not install CUDA, system
packages, Python environments, cluster launchers, or containers. Runtime setup
belongs in documentation, container recipes, or backend-specific setup scripts;
runtime checking belongs in `wam doctor`.

By default, `wam prepare <model>` checks the cache and reports missing assets.
Use `--download` when you explicitly want to fetch pullable assets declared with
`hf://` URIs. Use repeated `--asset <name>` flags to prepare only a subset. For
FastWAM, `dataset_stats` is small and useful for checking that cache/download
plumbing works:

Prepare output marks native asset roles:

- `required`: the native backend cannot load without this asset.
- `runtime`: the asset is part of the native runtime/cache contract, even if a
  backend container might lazily fetch or locate it later.

```bash
wam prepare fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --download \
  --asset dataset_stats
```

The released FastWAM checkpoint is about 11.2 GiB, so download it only when the
backend runtime and storage mount are ready:

```bash
wam prepare fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --download \
  --asset checkpoint
```

Runtime assets use the same cache boundary. For FastWAM, `model_base` and
`tokenizer_components` resolve under `--cache-dir/diffsynth-models/...`; they
are large, so prepare them only when the backend runtime is ready to run:

```bash
wam prepare fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --download \
  --asset model_base \
  --asset tokenizer_components
```

This still stays inside the asset/cache boundary. It does not install the
backend environment, clone upstream source code, build containers, or submit
jobs.

Use the same cache path for `doctor`, `prepare`, `run`, `native-smoke`, and
`serve`. Passing `--cache-dir` is equivalent to configuring the native backend's
asset cache for that command; it avoids relying on a global `WAM_CACHE_DIR`
environment variable in containers.

An Ollama-style alias may be added later for users who expect that wording, but
documentation should prefer `wam prepare`.

## Why Not Lead With `--dry-run`

`--dry-run` is common in infrastructure tools, but it reads like an expert flag.
For this project, users usually want one of two things:

- "Is my environment okay?" -> `wam doctor <model>`
- "Make this model ready." -> `wam prepare <model>`

An internal no-execute mode can still exist for tests and maintainers, but the
public guide should not make `--dry-run` part of the first user path.

## Model Entry Terminology

Use these terms consistently:

- **model entry**: public term for a supported model/checkpoint/task bundle.
- **model spec**: developer-facing term for the YAML that defines a model
  entry.
- **manifest**: internal implementation term; acceptable in code and deep
  developer docs, but not the first user-facing word.

Example user-facing wording:

> `fastwam-libero` is a model entry for running the released FastWAM checkpoint
> on the LIBERO evaluation suite.

Avoid opening a README section with:

> `fastwam-libero` is a manifest.

## Expected Output Style

`wam info` should be readable:

```text
Model: fastwam-libero
Name: FastWAM LIBERO
Task: LIBERO simulator evaluation
Inputs: primary camera, wrist camera, robot state, task prompt
Outputs: action chunks; horizon=32; dim=7
Runtime: GPU container recommended
Deployment: product=native_backend_migration; reference=official_script; native=fastwam (native_smoke_verified); native_verified=true; parity_verified=false; next=full_libero_eval
Prepare: checkpoint and dataset stats required
Optimizations: action_chunk_scheduling
```

`wam doctor fastwam-libero` should be actionable:

```text
Core:
  Python package: ok
  Cache directory: ok

Runtime:
  GPU visible: missing
  Container runtime: docker found

Model:
  checkpoint: missing
  upstream repo: missing

Native next steps:
  Set WAM_FASTWAM_REPO=<repo> or pass --upstream-dir <repo>
  wam prepare fastwam-libero --cache-dir /mnt/wam-cache --download --asset checkpoint
  Run inside the backend container or install native dependencies
```

`wam prepare fastwam-libero` should make progress on assets or give the next
concrete asset step. It should not silently assume a specific scheduler,
cluster, host path, or environment installation.

For native backend migration, maintainers can point `doctor` at a mounted
upstream repository:

```bash
wam doctor fastwam-libero --upstream-dir /mnt/upstreams/FastWAM
```

For container bring-up and CI-style gates, use the same command in structured
strict mode before `native-smoke`:

```bash
wam doctor fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --upstream-dir /mnt/upstreams/FastWAM \
  --json \
  --strict
```

`--json` exposes cache status, asset paths, deployment status, and native
readiness, including the selected upstream repo path and commit when available.
`--strict` returns non-zero when the status is not `ok`, so scripts can stop
before attempting a heavy native load.

Doctor status is intentionally simple: `ok` means the checked path is runnable,
`warning` means something should be fixed or reviewed, and `blocked` means the
native backend is known to fail before `load()` because a hard preflight item is
missing.

This checks the model entry's declared native backend, required upstream files,
expected upstream commit, required Python modules in the current environment,
and required assets. It does not install the upstream repository, build an
image, submit a job, or start a server.
