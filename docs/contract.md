# WAM Deployment Contract

WAM Harness is an Ollama-like local deployment platform for world-action models.
The core contract exists to make `wam prepare`, `wam run`, `wam serve`, and
runtime optimization profiles work across messy upstream WAM/VLA repositories.

The project is product-first on this branch. Observability remains part of the
runtime because users should be able to inspect what an optimization toggle did.

## Public Surface

The intended user path is small:

```bash
wam list
wam info fastwam-libero
wam doctor fastwam-libero
wam prepare fastwam-libero
wam run fastwam-libero --input obs.json --output action.json
wam run fastwam-libero --input obs.json --opt vla_cache
wam serve fastwam-libero --port 8000
wam compare runs/baseline runs/variant
```

`wam list` and `wam info <model-id>` expose the curated model library without
requiring users to read YAML files.

`wam doctor [model-id]` checks whether the current machine or prepared runtime
can run the core package and, optionally, a specific model entry.

`wam prepare <model-id>` prepares model assets for use: it creates cache
directories, verifies declared assets, and reports remaining manual
requirements. It does not install environments, containers, CUDA, or cluster
launchers.

`wam run <model-id> --input obs.json` runs one explicit observation through the
native product path. Internally it resolves the model spec, locates prepared
assets, creates or connects to a backend and processor, emits trace metadata,
and writes outputs. For real WAM entries, `run` must not silently create a
synthetic observation.

`wam eval <model-id>` is reserved for curated evaluations once a native workload
is validated. Official simulator scripts are reference evaluators and require
explicit `wam eval <model-id> --reference`.

`wam serve <model-id>` keeps a backend resident and exposes an
observation-to-action policy endpoint for open-loop, simulator, or remote-client
usage.

The serve endpoint accepts a JSON object with an `observation` using the core
observation contract plus optional `action_horizon`, `replan_steps`, `reset`,
and `runtime_options`. Empty requests are reserved for smoke checks and use the
registered processor's synthetic smoke observation.

Serve is traced as a resident run. `/health` returns the run id and trace path;
each `/infer` request records request start/end timing and action shape.

`wam compare` is a product benchmark helper. It compares two recorded runs and
reports latency, memory, output drift, and warnings. It is optional for normal
one-command usage.

The first comparison gate is conservative: compare latency samples and action
chunk shapes from traces. A run is not reported as faster if the output shape
gate fails or is unavailable.

## Non-Goals

- Training support.
- Real robot safety or hardware orchestration.
- Exhaustive model zoo coverage.
- Replacing Hugging Face Hub or upstream checkpoint distribution.
- A universal simulator or robotics environment interface.
- A replacement tensor runtime or CUDA allocator.
- Hidden optimization switches that cannot be disabled or inspected.

## Model Entry Contract

Model entries are the public model-default layer. Internally, a model entry is
stored as a YAML model spec; the code may still call it a manifest. A model
entry binds:

- model id and display name.
- upstream source repository.
- checkpoint, normalizer, dataset-stat, and other assets.
- backend and processor registry keys.
- default request shape and dtype/device preferences.
- supported optimization profiles.
- known gaps and hardware requirements.

The model spec is not a user-authored capabilities file. Runtime information is
still reported by the backend or server.

See `docs/wamfile.md`.

## Core Data Contract

### Observation

Required:

- `images`: named image views such as `primary`, `wrist`, or `exterior_0`.
- `prompt`: task instruction or natural language goal.

Optional:

- `state`: named numeric vectors such as `proprio`, `joint_position`,
  `cartesian_position`, or `gripper`.
- `history`: previous observations, actions, or compact backend state.
- `session`: run, episode, step, and session identifiers.
- `metadata`: extra data that does not affect core runner behavior.

### Inference Request

Required:

- `observation`
- `action_horizon`
- `replan_steps`

Optional:

- `num_inference_steps`
- `return_future`
- `return_value`
- `reset`
- `cache_control`
- `runtime_options`
- `optimization_profiles`

### Inference Result

Required:

- `action_chunk`: actions with shape-like metadata, usually `[T, D]`.

Optional:

- `future_frames`: JSON-safe summary or artifact reference for predicted frames.
- `value`: JSON-safe value estimate or artifact reference.
- `warnings`
- `backend_metadata`
- `timing`
- `memory`

Large arrays should be written as artifacts referenced from trace metadata, not
embedded directly in JSONL traces or serve responses.

## Runtime Info

Backends or remote servers report minimal runtime information:

```yaml
runtime_info:
  manifest_id: fastwam-libero
  model_name: FastWAM LIBERO 2-camera
  backend: fastwam
  processor: fastwam_libero
  source_repo: yuantianyuan01/FastWAM
  mode: local
  device: cuda:0
  dtype: bf16
  optimization_profiles:
    action_chunk_scheduling:
      replan_steps: 4
```

The core runner should use this metadata for traces and user summaries, not for
backend-specific branching.

## Backend Boundary

Backends expose lifecycle operations:

- `load`
- `warmup`
- `reset`
- `infer`
- `runtime_info`
- `close`

The core runner must not branch on upstream repository names. A new WAM should
add a backend, processor, and model entry, not changes to the runner.

`close` releases resources created by the backend. It is especially important
for native backends that start resident policy servers, WebSocket clients, CUDA
graph pools, or temporary worker processes. Runners and smoke paths should call
it even when inference fails.

Official evaluation scripts are reference evaluators, not the long-term product
backend. See `docs/native_backend_design.md`.

## Processor Boundary

Processors translate between harness observations/results and backend-native
model inputs/outputs. They own:

- image view selection and preprocessing.
- prompt formatting.
- state vector mapping.
- action denormalization.
- future/value output conversion.
- synthetic smoke observations for native backend bring-up.
- modality limits and input requirements.

Backend-native tensor layouts, key names, normalization constants, and cache
mechanics must not leak into core interfaces.

## Optimization Profiles

Optimization profiles are explicit runtime toggles:

```yaml
optimizations:
  enabled: [vla_cache]
  profiles:
    vla_cache:
      enabled: true
      cache_scope: replan
```

Each profile declares:

- stable name.
- deployment class.
- scope: request, replan, episode, run, or server.
- parameters and defaults.
- backend requirements.
- conflicts.
- trace fields.

Profiles default to off unless a model entry explicitly states otherwise.

## Telemetry Contract

Every run should emit enough structured telemetry to debug deployment and
compare optional toggles:

- run metadata: model entry id, backend, processor, source repo, device, dtype.
- workload shape: image shapes, state dims, action horizon, replan steps.
- timing: preprocess, model, postprocess, total, and server/client timing when
  remote.
- memory: process RSS and CUDA memory when available.
- optimization metadata: enabled profiles and parameter values.
- warnings and errors.
- artifact paths for action chunks, future frames, and values when persisted.

Telemetry is part of the product. It makes optimization toggles inspectable, but
does not require every normal `wam run` to be a formal comparison.

## Phase A Scope

Phase A builds the no-heavy-dependency deployment spine:

- model spec parser and one fake model entry.
- registry for model entries, backends, processors, and optimization profiles.
- fake backend.
- open-loop workload.
- runner with action chunk scheduling.
- JSONL trace writer.
- minimal memory/timing observer.
- `wam run fake-open-loop`.
- core container recipe and generic container smoke path.
- tests for model spec parsing, fake inference, trace shape, and profile
  metadata.

Phase A defers real checkpoints, simulators, external endpoint serving, CUDA
Graph, torch.compile, and multi-GPU scheduling.

## Phase B/C/D Scope

`Phase B: portable serve smoke`

Run `wam serve fake-open-loop` inside a container or existing job allocation,
with job-local health and inference checks. External laptop-to-node endpoint
access is not required for this stage.

`Phase C: first real model`

Add the first curated WAM, likely `fastwam-libero`, with asset resolution and a
backend container path that emits action chunks.

`Phase D: first real trick`

Add the first real training-free inference optimization profile, likely
VLA-Cache if the OpenVLA/OpenVLA-OFT path is viable.
