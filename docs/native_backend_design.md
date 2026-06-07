# Native Backend Design

The current real-model path proves that FastWAM, Cosmos-Policy, and DreamZero
can run, but it still treats their official evaluation scripts as the execution
unit. That is useful for validation, not good enough for an Ollama-like WAM
deployment platform.

The next architecture target is:

```text
official script wrapper  ->  reference evaluator
native backend           ->  product inference path
processor                ->  model-specific input/output conversion
workload                 ->  open-loop or simulator driver
optimization profile     ->  explicit, trace-backed runtime toggle
```

## Design Philosophy

### 1. Official Scripts Are Reference Evaluators

Official scripts are valuable because they encode upstream assumptions about
checkpoints, assets, simulators, and evaluation defaults. Keep them, but do not
make them the long-term product path.

Use official scripts for:

- first bring-up;
- checkpoint and simulator validation;
- reference logs and videos;
- parity checks against a native backend;
- reproducing upstream results when needed.

Do not use official scripts for:

- resident serving;
- normal `wam eval` once a native backend is validated;
- optimization profiles such as `torch_compile` or `cuda_graph`;
- core trace timing beyond coarse subprocess timing.

### 2. The Harness Owns The Inference Lifecycle

A native backend must expose one lifecycle:

```text
load -> warmup -> reset -> infer -> runtime_info -> close
```

This is where the harness can insert:

- checkpoint loading and dtype/device placement;
- processor construction;
- warmup requests;
- `torch.compile`;
- CUDA Graph capture;
- cache reset;
- per-stage timing;
- memory sampling;
- failure and fallback reporting.
- cleanup of model-owned processes, sockets, graph pools, or temporary state.

If the lifecycle stays inside an upstream script, these controls remain
unreliable.

Lifecycle failures must be traceable. `wam native-smoke` and `wam serve` should
open the trace before backend load starts, then record `error` and cleanup
events if imports, checkpoint paths, warmup, or resident server startup fail.
This makes container bring-up debuggable without relying only on terminal
tracebacks.

### 3. Processors Are First-Class

The core runner must not learn model-specific camera names, state layouts,
normalizers, prompt formats, or action denormalization. Those belong in
processors.

Processor responsibilities:

- select and validate image views;
- normalize and pack robot state;
- format prompts or task ids;
- call model-specific normalizers;
- convert raw model outputs into harness `InferenceResult`;
- provide a small synthetic smoke observation for native backend bring-up;
- declare modality limits for `wam info` and `wam doctor`.

### 4. Workloads Drive Tasks, Not Models

Workloads own observation/action loops:

- `open_loop`: fixed observations and smoke tests;
- `libero`: simulator loop for LIBERO tasks;
- `droid_sim`: simulator loop for DreamZero-style DROID simulation;
- future robot hardware paths stay out of the first version.

Backends should not own simulator loops. Backends answer:

```text
observation -> action chunk
```

Workloads decide when to observe, infer, step, reset, and end an episode.

### 5. Optimizations Are Profiles, Not Patches

Every optimization must be:

- named;
- optional;
- backend-declared;
- trace-backed;
- disableable;
- compared against a baseline.

`torch_compile` and `cuda_graph` should become native backend profiles, not
flags hidden in upstream scripts. They may be default-on later only after the
backend proves stable fallback behavior.

Requested profiles first become a native optimization plan. The plan records the
profile name, params, manifest support, manifest scope, and target layer:

- `native_backend`: model/server lifecycle toggles such as `torch_compile`,
  `cuda_graph`, `dit_cache`, JPEG preprocessing, or backend-owned inference
  flags;
- `workload`: scheduling choices such as action chunk scheduling or simulator
  loop behavior;
- `deployment`: multi-process or multi-GPU choices that affect job shape rather
  than one model call.

Backends consume only the profiles that target their lifecycle. Workloads and
deployment recipes consume the others. This keeps `--opt` unified without
forcing every optimization into the backend class.

## Core Abstractions

### Native Backend

```python
class NativeBackend:
    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def reset(self) -> None: ...
    def infer(self, request: InferenceRequest) -> InferenceResult: ...
    def runtime_info(self) -> RuntimeInfo: ...
    def close(self) -> None: ...
```

Native backend responsibilities:

- import upstream model code;
- load weights and non-weight assets;
- own model/device/dtype state;
- own optional resident server process when needed;
- construct a native model adapter for the loaded runtime object;
- call the processor for input/output conversion;
- implement optimization profile hooks;
- report runtime metadata and warnings.
- close any resources the backend created, especially resident policy servers.

`NativeBackendBase` provides shared repo/asset/runtime plumbing plus the common
native inference spine. It should not know model tensor layouts, simulator APIs,
or cache mechanics. Concrete backends still own source/asset resolution and
model construction; the loaded model/server call should live behind a native
model adapter.

The native inference spine is:

```text
processor.to_model_inputs(observation)
  -> backend adapter.infer(request, model_inputs)
  -> processor.to_harness_result(raw_output)
  -> unified timing / native metadata / warnings
```

Backends should not hand-write their own `infer()` spine. `NativeBackendBase`
owns the public native `infer()` method and calls backend hooks for model
readiness, metadata, timing key, and the adapter-backed model/server call. A
native adapter call may return either the raw upstream output directly or a
`NativeModelCall` when it needs to provide a nonstandard timing key such as
`server_ms`, call-level metadata, or warnings. This keeps FastWAM,
Cosmos-Policy, DreamZero, and future WAMs on the same trace contract while
still letting each integration own its real model object or resident service.

### Native Model Adapter

Native backends should split model construction from model calls:

```text
NativeBackend.load()
  -> runtime loader builds/imports upstream model objects and processors
  -> backend binds harness processor runtime state
  -> backend creates NativeModelAdapter around the loaded model/server

NativeBackend.infer()
  -> NativeBackendBase shared spine
  -> processor.to_model_inputs(...)
  -> adapter.infer(...)
  -> processor.to_harness_result(...)
```

The runtime loader is backend-specific but explicit: every real native backend
should expose a stable `runtime_mode` and `runtime_loader` name in readiness,
runtime contract, and runtime metadata. `runtime_mode` is one of:

- `in_process`: the `wam` process directly loads and calls the model;
- `resident_server`: `wam` starts or connects to a job-local policy server and
  sends inference requests over a transport such as WebSocket or HTTP.

The loader may import heavy upstream packages, compose Hydra/YAML configs, load
checkpoints, load dataset statistics, construct upstream processor objects, or
start/connect a resident policy server. It must not call an official evaluator,
simulation manager, or CLI entry point. This keeps the product path independent
from upstream scripts while still reusing the real model code.

`NativeModelAdapter` is the narrow boundary around a loaded native runtime
object:

```python
class NativeModelAdapter:
    def require_ready(self) -> None: ...
    def warmup(self) -> None: ...
    def reset(self) -> None: ...
    def infer(self, request: InferenceRequest, model_inputs: object) -> object | NativeModelCall: ...
    def runtime_metadata(self) -> dict[str, object]: ...
    def inference_metadata(self) -> dict[str, object]: ...
    def close(self) -> None: ...
```

The adapter should contain the direct call to a model object or resident policy
server: for example, FastWAM's `infer_action` / `infer_joint` call. It should
not own upstream repo discovery, asset path resolution, CLI parsing, simulator
loops, or `wam` command behavior.

This split matters because the next optimization layer needs a consistent
place to wrap the actual runtime object:

- `torch_compile` can wrap model submodules during adapter construction;
- `cuda_graph` can capture stable-shape adapter calls after warmup;
- quantization can be applied before adapter readiness;
- resident-server backends can report `server_ms` while still sharing the
  same processor and trace contract.

New native backend integrations should therefore implement this shape:

```python
def load(self) -> None:
    ...
    self.model_adapter = MyModelAdapter(...)
    self.loaded = True
```

Once `self.model_adapter` is set, `NativeBackendBase` owns
`require_inference_ready()`, `native_model_timing_key()`,
`native_inference_metadata()`, `native_inference_warnings()`, and `_infer_model()`
by delegating to the adapter. Resident-server adapters such as DreamZero report
`server_ms` through `model_timing_key()` while still using the same
preprocess/model-call/postprocess spine. New real-model backends should avoid
overriding `infer()` or `_infer_model()` unless the base lifecycle itself is
insufficient and the exception is documented.

Do not copy an upstream evaluator's simulator loop into an adapter. The adapter
should perform only the product inference call:
`model_inputs -> raw model/server output`.

Native backends also declare their runtime requirements without importing heavy
dependencies:

- upstream repo environment variable, such as `WAM_FASTWAM_REPO`;
- required upstream files, such as the module that contains the model runtime;
- expected upstream commit when the model entry is tied to a known upstream
  checkout;
- required Python modules in the backend container, such as `torch`, `hydra`,
  or `websockets.sync.client`;
- required asset names, such as checkpoint and dataset statistics;
- runtime asset names, such as model-base or tokenizer caches that the backend
  may need during model construction or first inference.

These declarations are instance-level contracts. A backend may compute them
from the model entry and backend config, for example when a DreamZero entry uses
a custom resident server module or when a Cosmos-Policy entry points at a
different config file. `wam doctor` must inspect the same paths, assets, and
Python modules that `load()` will use. If those two drift apart, native bring-up
becomes unreliable and the harness silently falls back into official-script
thinking.

For config-driven upstreams, required upstream paths should include the
configuration entry points needed to construct the model, not only the Python
file that contains the official evaluator. FastWAM, for example, is loaded
through Hydra composition, so the native backend declares `configs/train.yaml`,
the selected task YAML, and the inferred data/model config group files in
addition to the runtime Python modules. This keeps incomplete upstream mounts
from reaching `backend.load()` and failing later with opaque Hydra errors.

`wam doctor <model> --upstream-dir /path/to/repo` uses this declaration to
check mounts and model assets before `load()` runs. The same declaration is used
by `load()`, so readiness checks and actual model loading do not drift apart.
`required_assets` means the backend fails early if the asset is absent;
`runtime_assets` means the asset is part of the native runtime contract and must
be visible in `doctor`, even when the backend can still use a container default
or upstream lazy download path.

Native readiness is reported with three states:

- `ready`: upstream source, hard assets, and runtime assets are visible;
- `warning`: the backend can pass hard readiness checks, but runtime assets such
  as model-base or tokenizer caches are still missing, or the selected upstream
  git commit does not match the model entry's expected commit;
- `blocked`: upstream source, required Python modules, or hard required assets
  are missing, so native `load()` is expected to fail before inference.

Use `wam doctor <model-id> --json --strict` as the preflight gate before real
native smoke runs. The JSON payload includes the same native readiness data that
the trace will later record, while `--strict` gives automation a simple exit
code: continue only when status is `ok`.

`wam run`, `wam native-smoke`, and `wam serve` also write a
`native_runtime_contract` event before `backend.load()`. This records the native
mode, backend/processor/workload boundary, processor modality, supported
optimization names, the native optimization plan, declared model adapter,
backend config keys, and deployment state. The backend runtime metadata includes
the same plan. This is the trace anchor for later comparing baseline and
optimized runs without guessing which native contract was active.

The same commands also enforce the readiness contract before `backend.load()`.
When readiness is `blocked`, they write `native_runtime_contract`,
`native_readiness`, emit an `error` with `stage="native_preflight"`, and stop
before model loading. A `warning` is allowed by default because some runtime
assets can still be supplied by a backend container or upstream lazy-load path.
For stricter container smoke checks, `wam native-smoke --require-ready` fails
unless readiness is exactly `ready`.

This keeps failed container bring-up traces actionable: the trace records
whether the failure was expected from missing upstream source, missing hard
assets, missing Python modules, an upstream checkout mismatch, or only a runtime
asset warning, and it avoids hiding obvious setup problems behind later
import/checkpoint exceptions.

After inference, native product modes also enforce an action contract before the
run is considered successful. The returned action chunk must be non-empty,
rectangular, finite, match the requested action horizon, and match the action
dimension declared by the model entry when one is known. Failures emit
`error.stage="action_contract"` and are not treated as a valid native smoke pass.

Model assets should be portable across machines and clusters. Use `local_path`
relative to `--cache-dir` for checkpoints, dataset stats, tokenizer components,
and model-base snapshots unless an upstream project truly requires an absolute
path. Cluster-specific mount points such as `/mnt/cache` belong in local
runtime scripts or container launch configuration, not in the public model entry
asset paths.

Each real model entry should also record migration state in `deployment`.
Do not mark `native_verified=true` until a real checkpoint/container
`wam native-smoke <model-id>` has passed. Do not mark `parity_verified=true`
until native output has been compared against the reference official path.

### Processor

```python
class Processor:
    def to_model_inputs(self, observation: Observation) -> object: ...
    def to_harness_result(self, raw_output: object) -> InferenceResult: ...
    def modality_limits(self) -> dict[str, object]: ...
    def smoke_observation(self) -> Observation: ...
```

This repository now has a processor registry and a `passthrough` processor for
the fake backend. Real backends should add processors such as:

- `fastwam_libero`;
- `cosmos_policy_libero`;
- `dreamzero_droid`.

### Reference Evaluator

A reference evaluator may still execute a command-backed official script. It is
not a backend. It should live behind an explicit reference mode, not the default
native backend path.

Reference evaluator responsibilities:

- run upstream official evaluation;
- write stdout/stderr;
- write coarse trace metadata;
- provide baseline artifacts for parity.

## Current Model Migration Plan

Current skeleton status:

| Model target | Native strategy | Product path | Reference path |
| --- | --- | --- | --- |
| FastWAM LIBERO | In-process PyTorch backend calling `model.infer_action(...)` | `fastwam` backend + `fastwam_libero` processor | Official LIBERO evaluator |
| Cosmos-Policy LIBERO | In-process PyTorch backend calling upstream `get_action(...)` after `get_model(...)` | `cosmos_policy` backend + `cosmos_policy_libero` processor | Official LIBERO evaluator |
| DreamZero DROID | Harness-owned resident WebSocket policy server | `dreamzero` backend + `dreamzero_droid` processor | Official server + DROID simulator eval |

### FastWAM: First Native Backend

FastWAM should be first because it is closest to the harness contract:

- action chunk output is central;
- LIBERO evaluation has already run successfully;
- released checkpoint assets are known;
- current manifests already describe primary/wrist images and proprio state.

Target:

```text
FastWAMBackend.load()
  load upstream model code
  load checkpoint + dataset stats
  construct FastWAMLiberoProcessor

FastWAMBackend.infer(request)
  processor.to_model_inputs(observation)
  model forward
  processor.to_harness_result(raw_output)
```

Acceptance:

- `wam native-smoke fastwam-libero` emits action chunks inside the FastWAM
  runtime container;
- action shape is `32x7`, finite, rectangular, and passes the native action
  contract before the smoke run is accepted;
- action value ranges match the official reference path during parity checks;
- trace includes preprocess/model/postprocess timing;
- trace includes action summary and records load failures;
- official evaluator remains available as reference mode.

Current migration notes:

- The official LIBERO script should not be wrapped as the product path. Its
  useful pieces are model construction, dataset-stat loading, observation
  preprocessing, action denormalization, and simulator reference evaluation.
- The native path should load FastWAM with Hydra composition, call
  `model.load_checkpoint(...)`, bind the upstream `FastWAMProcessor`, and call
  `model.infer_action(...)` for the normal action-only path. The implementation
  keeps the first part in `FastWAMRuntimeLoader` and the direct model call in
  `FastWAMModelAdapter`.
- The native backend registers upstream FastWAM config resolvers before Hydra
  composition and honors `EVALUATION.device`, because those details affect
  whether the same checkpoint/config can load in a container.
- If `EVALUATION.visualize_future_video=true`, the native backend mirrors the
  official evaluator and calls `model.infer_joint(...)`; otherwise it calls
  `model.infer_action(...)`.
- `FastWAMLiberoProcessor` owns LIBERO image selection, primary/wrist
  resize-concat, proprio normalization, prompt formatting, action
  denormalization, and gripper sign conversion.
- The curated LIBERO entry follows upstream defaults: `action_horizon=32`,
  `replan_steps=10`, and `action_dim=7`. The native backend also follows the
  official eval path by adding `num_video_frames` when the model's
  `infer_action` signature requires it.
- The native backend is registered separately from the current `external_eval`
  reference manifests so the public CLI does not imply that true checkpoint
  inference is ready before container validation.

### Cosmos-Policy: Second Native Backend

Cosmos-Policy has richer output potential and heavier runtime assumptions.

Target responsibilities:

- load public Cosmos-Policy LIBERO checkpoint;
- handle dataset stats and text embeddings;
- expose action chunks and optional future/value outputs when supported, using
  JSON-safe summaries or artifact references rather than embedding large arrays;
- preserve existing JPEG/parallel inference concepts as profiles only when they
  are trace-backed.

Current migration notes:

- The official LIBERO evaluator's model lifecycle is separable from the
  simulator loop. The native backend reuses upstream `get_model`,
  `load_dataset_stats`, `init_t5_text_embeddings_cache`, and `get_action`
  from `cosmos_utils`, but it does not import
  `cosmos_policy.experiments.robot.libero.run_libero_eval` as the product
  runtime entry.
- The minimal eval config and un-normalization key check live in the harness so
  the native path does not depend on LIBERO simulator imports just to construct
  the model.
- `CosmosPolicyLiberoProcessor` owns conversion from harness observations to
  `primary_image`, `wrist_image`, `proprio`, and task prompt.
- `parallel_inference` is not a first native default because it starts multiple
  model copies. It stays disabled unless the user explicitly requests
  `--opt parallel_inference`; the native backend maps that profile through the
  same optimization context used by the reference evaluator.

Acceptance:

- native backend can run a LIBERO smoke observation;
- processor owns image/state/prompt conversion;
- reference official eval remains available for full-suite comparison.

### DreamZero: Resident Server Native Backend

DreamZero may not start as a pure in-process PyTorch backend because its
upstream runtime uses a WebSocket policy server and multi-process launch.

That is acceptable if the harness owns the server lifecycle:

```text
DreamZeroBackend.load()
  runtime loader starts or connects to job-local policy server
  wait for health/runtime metadata
  bind DreamZeroPolicyServerAdapter

DreamZeroBackend.infer(request)
  processor packs observation
  adapter calls policy server
  processor converts response to InferenceResult

DreamZeroBackend.reset()
  send reset/session boundary

DreamZeroBackend.close()
  stop a harness-started policy server and drop the client connection
```

This is still native enough because the product path is
`observation -> action chunk`, not "run upstream evaluation script."
The default native required path is the policy-server module
`eval_utils/serve_dreamzero_wan22.py`; the simulator evaluator
`eval_utils/run_sim_eval.py` remains a reference workload and should not be
required for native smoke or resident serve.

Acceptance:

- backend owns server startup/health/reset/infer;
- backend cleans up server/client resources on runner, native-smoke, or serve
  shutdown;
- DROID sim loop becomes a workload/client, not the backend itself;
- `dit_cache` is a profile with trace-visible enable/failure status. The native
  backend maps the profile to resident server args through optimization context,
  not through a hidden hard-coded server command.

## Optimization Placement

### `torch_compile`

First native optimization target.

Backend requirements:

- model object is accessible in `load`;
- compile happens after model construction;
- warmup records compile overhead;
- failure falls back to eager with a warning unless user requested strict mode;
- trace records `compile_enabled`, `compile_success`, and `compile_wall_ms`.

### `cuda_graph`

Second native optimization target.

Backend requirements:

- stable input shapes;
- fixed device buffers or replay-compatible input staging;
- warmup before capture;
- explicit capture failure fallback;
- trace records `graph_capture_success`, `graph_replay_count`, and memory.

## Model Entry Shape After Migration

The native model entry should make the native path default and keep the official
script as reference:

```yaml
id: fastwam-libero
backend:
  name: fastwam
  mode: native
processor:
  name: fastwam_libero
workload:
  name: libero
reference_eval:
  name: fastwam_libero_official
  mode: official_script
```

During migration, existing `external_eval` entries can remain as compatibility
entries or be renamed to explicit reference ids, for example:

```text
fastwam-libero-reference
cosmos-policy-libero-reference
dreamzero-droid-sim-reference
```

Reference entries that have a native migration path should declare the native
backend in their model spec:

```yaml
backend:
  name: external_eval
  mode: official_script
  config:
    upstream: FastWAM
    simulator: LIBERO
    native_backend: fastwam
```

The product entry points read this field through the shared
`Registry.resolve_runtime(...)` path. `core.runtime.RuntimeSpec` declares the
product mode and workload for an entry point, and `core.runtime.RuntimePlan`
records whether the model stayed on its regular manifest or was mapped from a
reference entry to a native backend.

Current specs:

| Entry point | Mode | Workload | Native backend required |
|---|---|---|---|
| `wam prepare` | `native_prepare` | `native_prepare` | no |
| `wam doctor` | `native_doctor` | `native_doctor` | no |
| `wam run` | `native_run` | `processor_smoke` | no |
| `wam native-smoke` | `native_smoke` | `native_smoke` | yes |
| `wam serve` | `native_serve` | `serve` | no |

Therefore a new WAM migration should not add model-specific branches to
`core/runner.py`, `core/native_smoke.py`, `serve.py`, or `core/model_entry.py`.
It should add a backend, a processor, registry entries, and the model spec
declaration. The shared resolver swaps only backend/workload/mode/config for
native-capable reference entries and leaves ordinary entries such as
`fake-open-loop` untouched.

`wam serve` uses the same native backend declaration for resident policy
serving. Instead of running the reference evaluator, serve maps the entry to
`mode: native_serve` and expects `/infer` requests to provide a harness
observation JSON object. Empty requests are allowed only as a smoke path and use
`processor.smoke_observation()`. Serve runs keep a trace open across requests and
emit per-request start/end events so resident backends can be compared against
optimization variants.

Before the public model entry is switched, use the migration-only smoke command
to validate the native backend against the same model entry:

```bash
wam native-smoke fastwam-libero --require-ready
wam native-smoke cosmos-policy-libero --upstream-dir /path/to/cosmos-policy --require-ready
wam native-smoke dreamzero-droid-sim --upstream-dir /path/to/dreamzero --require-ready
```

`native-smoke` temporarily maps a curated reference model entry to its native
backend, builds one synthetic contract observation, runs `load/warmup/reset/infer`,
and writes trace events. It is a migration and maintainer command, not the main
user-facing flow.

## Non-Goals

- Do not rewrite model architectures.
- Do not vendor upstream repositories into core.
- Do not put backend-specific branches in the runner.
- Do not remove official reference paths before native parity exists.
- Do not default-enable compile/graph profiles without fallback and trace.
- Do not make simulator loops part of backend internals.

## Immediate Next Steps

1. Keep `external_eval` as reference evaluator wording in docs and model specs.
2. Use `wam native-smoke` as the container smoke entry for native validation.
3. Validate FastWAM native `load` and one fixed observation inside the FastWAM
   container.
4. Validate Cosmos-Policy native `load` and one fixed LIBERO-style observation
   inside the Cosmos container.
5. Validate DreamZero resident-server `load/reset/infer` inside the DreamZero
   container.
6. Add official-reference vs native-smoke parity checks.
7. Add `torch_compile` as an experimental profile after native smoke works.
8. Add `cuda_graph` after stable shape and warmup/capture requirements are met.
