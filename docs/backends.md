# Backend Notes

Backends hide model-specific loading, preprocessing, inference, postprocessing,
and deployment details.

See `docs/native_backend_design.md` for the migration plan from official script
wrappers to native backends.

Some upstream WAM projects use adapter-like wrappers. In this harness, the
preferred term is `backend` because the boundary is closer to vLLM/SGLang
backend registration than to a temporary compatibility patch.

In the Ollama-like product direction, users normally select a model id, not a
backend id. The model entry maps that model id to a backend, processor, assets,
defaults, and supported optimization profiles.

## Current Model Backends

- Fake: test backend for the harness contract.
- Reference eval: command-backed official simulator evaluation for upstream
  projects. This is a correctness baseline, not the long-term product backend.
- FastWAM: real LIBERO and RoboTwin targets through `fastwam-libero` and
  `fastwam-robotwin`; native backend is registered as `fastwam` and has passed
  real checkpoint native smoke, one-shot `wam run --input`, single-task LIBERO
  eval, `wam serve --smoke` with a real observation payload, reference
  full-suite LIBERO eval, and a native LIBERO sweep on SuperPod H800. The
  LIBERO native sweep currently reports 9/10 task successes, and aligned task6
  evidence reports native and reference single-task paths both at 4/5.
  RoboTwin has also passed SuperPod H800 single-task smoke, serve smoke,
  reference manager full-suite execution, and native full-suite execution over
  50 tasks x clean/randomized phases.
- Cosmos-Policy: real LIBERO target through `cosmos-policy-libero`; native
  backend is registered as `cosmos_policy` and has passed real checkpoint
  native smoke.
- DreamZero: DROID simulation target with an official WebSocket policy server;
  native resident-server backend is registered as `dreamzero` and has passed
  real checkpoint native smoke with a job-local policy server.
- Remote policy: future WebSocket/msgpack compatible resident serving target.

The native backends are becoming the public inference path model by model.
For FastWAM, `wam run --input`, `wam serve`, and `wam native-smoke` use the
native backend declaration, while `wam eval` runs the curated LIBERO simulator
workloads under trace. FastWAM has SuperPod H800 evidence for single-task
native eval, product-path serve smoke, reference full-suite manager eval, and a
native full-suite sweep. The LIBERO native sweep completed structurally but
scored 9/10, with task_id=6 failing after 700 steps in the one-trial sweep.
After aligning `seed=42` and `num_steps_wait=30`, a repeated task6 check with
`num_trials=5` produced native 4/5 and reference single-task 4/5, so task6 is
not a deterministic native failure. RoboTwin now has equivalent execution
coverage: reference manager full-suite execution reached 100/100 phases with
clean mean 1.0 and randomized mean 0.84; native single-task sweep reached
100/100 phases with 0 structural failures and aggregate success rate 0.88.
RoboTwin manager summaries now separate simulator setup invalidity from policy
failure: each summary records requested valid episodes, completed valid
episodes, attempted candidate episodes, invalid candidate episodes, and invalid
candidate reasons. This matches RoboTwin's seed semantics: keep the requested
top-level seed, skip invalid internal candidate episodes, and continue until
the requested number of valid episodes is reached. Process-level worker
restarts are opt-in.
Cosmos-Policy and DreamZero have now cleared native smoke; their next gates are
simulator eval/parity hardening before promoting more public eval and serve
paths.

For FastWAM, prepare either the backend container from
`containers/fastwam/Dockerfile` or a self-managed environment with:

```bash
scripts/setup_fastwam_native_env.sh \
  --venv /path/to/.venv-fastwam \
  --cache-dir /path/to/wam-cache \
  --clone
```

During that migration, use the maintainer smoke entrypoint to validate native
backends without changing public defaults:

```bash
wam native-smoke fastwam-libero --require-ready
wam native-smoke cosmos-policy-libero --upstream-dir /path/to/cosmos-policy --require-ready
wam native-smoke dreamzero-droid-sim --upstream-dir /path/to/dreamzero --require-ready
```

The smoke command runs one contract-shaped observation through
`load/warmup/reset/infer` and writes trace events. It is the bridge between
"official script works" and "`wam run` / `wam serve` can safely become native."
The native backend key comes from `backend.config.native_backend` in the model
spec, while the smoke observation comes from the registered processor.

## Planned Workloads

- Open-loop: load a fixed observation and run inference without a simulator.
- External eval: execute an upstream official simulator evaluator under trace.
- LIBERO: simulator evaluation.
- RoboTwin: simulator evaluation.
- DROID sim: simulator evaluation through DreamZero/openpi-style clients.

## Backend Compatibility Goal

A new WAM should require a new backend, processor, and model entry, not changes
to the core runner.

For migration smoke, that means:

- declare `backend.config.native_backend` in the reference model spec;
- implement `processor.smoke_observation()`;
- register the backend and processor;
- keep `core/native_smoke.py` free of model-specific camera/state branches.

Backends conform to the harness contract. They should not force backend-native
tensor layouts, key names, normalization details, or cache mechanics into the
core runner.

Backends also own cleanup through `close()`. For in-process PyTorch backends this
may be a no-op, but resident-server backends such as DreamZero must terminate
servers or clients they started. Runner, native-smoke, and serve shutdown paths
call `close()` so model-specific cleanup does not leak into core code.

The core migration rule is:

```text
official script wrapper = reference evaluator
native backend = product inference path
```

The official script path stays useful for parity and reproduction, but normal
`wam eval` / `wam serve` should move to native backends as each model is
validated.

`wam run --input` and `wam serve` already use the native backend declaration
when present. A reference entry with `backend.config.native_backend` is
temporarily mapped to `mode: run` for an explicit one-shot observation or
`mode: serve` for resident HTTP inference. Synthetic observations stay in
`wam native-smoke` and `wam serve --smoke`. This keeps official eval scripts out
of the product inference paths. Both paths accept the same backend-side
overrides. FastWAM uses vendored runtime code by default; `--upstream-dir` is
only a reference/debug override for that backend. Other backends may still need
mounted upstream source repositories until their runtime code is similarly
packaged.

Native backends resolve relative asset paths under their configured cache
directory. The public commands pass this with `--cache-dir`; if omitted, the
backend falls back to `WAM_CACHE_DIR` and then `~/.cache/wam`.
Backends distinguish hard `required_assets` from broader `runtime_assets`.
For example, FastWAM fails early on the checkpoint and dataset stats, but
`wam doctor` also surfaces Wan/model-base and tokenizer assets because those
are part of the native runtime environment.
`wam doctor` reports native readiness as `ready`, `warning`, or `blocked`:
`blocked` means `load()` is expected to fail, while `warning` means hard
requirements are present but runtime assets or defaults still need attention.

## Product Compatibility Goal

The minimum supported user path for a curated backend should be:

```bash
wam info <model-id>
wam doctor <model-id>
wam prepare <model-id>
wam run <model-id> --input obs.json
```

`wam serve <model-id>` is the persistent policy endpoint for backends that can
stay resident or that already expose a remote policy server.

For FastWAM real simulator measurements, the current verified product
entrypoint is:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --cache-dir /path/to/wam-cache
```

The maintainer evidence for that path is a SuperPod H800 run with
`success_rate=1.0` for `task_id=0`, `num_trials=1`.

FastWAM also exposes `dit_cache` as a native optimization profile. For FastWAM
this profile means request-local video K/V cache reuse inside one
`infer_action()` call:

- default mode: `video_kv`, which prefills video K/V once and reuses it during
  the action denoising loop;
- ablation mode: `recompute`, which skips prefill and recomputes the video path
  on every denoising step;
- hook name: `fastwam_video_kv_cache`.

This is not cross-replan cache, not step skipping, not token pruning, and not
VLA-Cache static token reuse. It is only the existing FastWAM video K/V cache
path made configurable and trace-visible.

FastWAM also exposes `cuda_graph` as a default `auto` native profile. The first
implementation is conservative: it captures only
`mot.forward_action_with_video_cache()` inside the action denoising body, after
video K/V prefill has already happened. `pre_dit`, `post_dit`, VAE/image
encoding, prompt/context handling, scheduler stepping, and CPU output copies
stay eager. The profile requires `dit_cache.mode=video_kv`; `recompute` remains
the ablation path. Unsupported CUDA Graph capture falls back to eager and emits
`cuda_graph_fallback_reason` in backend metadata. Use
`--set cuda_graph_mode=off` when you need the eager cached baseline.

SuperPod cached-vs-recompute ablation should keep task, seed, trial count,
`num_inference_steps`, `action_horizon`, `replan_steps`, dtype, checkpoint, and
dataset stats identical:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 5 \
  --opt dit_cache \
  --set seed=42 \
  --set num_inference_steps=20 \
  --set action_horizon=32 \
  --set replan_steps=10 \
  --set dit_cache_mode=video_kv \
  --trace-dir /path/to/traces/fastwam-libero-cache

wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 5 \
  --opt dit_cache \
  --set seed=42 \
  --set num_inference_steps=20 \
  --set action_horizon=32 \
  --set replan_steps=10 \
  --set dit_cache_mode=recompute \
  --trace-dir /path/to/traces/fastwam-libero-recompute

wam compare /path/to/traces/fastwam-libero-cache/<run>/trace.jsonl \
  /path/to/traces/fastwam-libero-recompute/<run>/trace.jsonl
```

The CUDA Graph latency ablation should compare the cached eager path against
the default cached plus graph-capture path:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 5 \
  --set seed=42 \
  --set num_inference_steps=20 \
  --set dit_cache_mode=video_kv \
  --set cuda_graph_mode=off \
  --trace-dir /path/to/traces/fastwam-libero-eager-cache

wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 5 \
  --set seed=42 \
  --set num_inference_steps=20 \
  --set dit_cache_mode=video_kv \
  --trace-dir /path/to/traces/fastwam-libero-cudagraph
```

Maintainer evidence on SuperPod H800, Slurm job `450394`, used
`task_id=0`, `num_trials=1`, `seed=42`, `num_inference_steps=10`,
`action_horizon=32`, and `replan_steps=10`. Both baseline and CUDA Graph runs
completed with `success_rate=1.0`. The CUDA Graph run reported
`cuda_graph_capture_success=True`, no fallback reason, and median replay count
of 10 per model call. The measured speedups were:

- eval duration: `17.86s -> 12.84s` (`1.39x`);
- mean `model_ms`: `1.93x`;
- mean `total_ms`: `1.91x`;
- mean `denoise_wall_ms`: `240.18ms -> 86.30ms` (`2.78x`).

SuperPod H800 job `450401` repeated the same comparison on FastWAM RoboTwin
`click_alarmclock`, `demo_randomized`, `num_episodes=1`. Both runs completed
with `success_rate=1.0`, no invalid setup, no CUDA Graph fallback, and median
replay count of 10. Mean `total_ms` improved from `651.08ms` to `342.85ms`
(`1.90x`), and median `denoise_wall_ms` improved from `237.50ms` to
`79.49ms`. The mean denoise speedup is lower (`1.30x`) because the single
episode only produced three model calls and includes first-shape capture cost.

FastWAM also has an experimental `torch_compile` profile for the same
action-body callable. It is disabled by default and should be measured as a
separate Phase-4 combination (`cuda_graph` plus `--opt torch_compile`) before
being promoted. SuperPod H800 job `450449` ran that Phase-4 combination on
FastWAM LIBERO `task_id=0`, `num_trials=1`, `num_inference_steps=10`, and
`max_steps=100`. CUDA Graph alone remained beneficial (`1.65x` mean
`total_ms`, `2.38x` mean `denoise_wall_ms` over eager cached execution), while
the `torch_compile` combination hit `call_failed:InductorError` fallback and
large first-request latency. That profile remains opt-in and experimental.

Record trace files, runtime metadata, success rate, model call count, duration,
warnings, and the cache metadata fields from `inference_end`. If RoboTwin
resources are available, repeat the same ablation with `fastwam-robotwin`,
the same seed and inference settings, and only the simulator/task fields
changed. Do not report parity or speedup until those SuperPod traces and result
files support the claim.

Reference-mode full-suite LIBERO manager eval has also passed on SuperPod H800
with `num_trials=1`: all 10 `libero_10` tasks succeeded. Native full-suite eval
has been run as a sequential sweep over the native `libero-single-task` runner:
9/10 tasks succeeded, while task_id=6 failed after 700 simulator steps.
After aligning `seed=42` and `num_steps_wait=30`, repeated task_id=6 evidence
with `num_trials=5` reports native 4/5 and reference single-task 4/5.
Native-vs-reference parity remains pending because this is not yet enough
statistical evidence to call the paths equivalent across the full suite.

Reference-mode simulator paths remain available for parity and reproduction:

```bash
wam eval fastwam-libero --reference --upstream-dir /path/to/FastWAM
wam eval cosmos-policy-libero --reference --upstream-dir /path/to/cosmos-policy
wam eval dreamzero-droid-sim --reference --upstream-dir /path/to/dreamzero --opt dit_cache
```

These are reference paths. They delegate to official simulator loops and record
the command, environment, assets, optimization flags, stdout/stderr, and trace.
They should remain available until a native backend has parity evidence.
