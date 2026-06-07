# Inference Optimization Integration

This document defines how WAM Harness should absorb existing single-point
WAM/VLA optimization repositories into an Ollama-like deployment platform
without becoming a fork of each upstream project.

The integration target is a deployment-first platform:

- Training-free inference optimizations are first-class runtime options.
- Post-training and training-time methods are tracked, but not early deployment
  targets.
- Heavy upstream dependencies stay behind backend or remote-server boundaries.
- Every optimization is enabled through configuration and recorded in trace
  output.
- User-facing toggles should eventually be available through `wam run <model>
  --opt <profile>`.

## Integration Principle

Do not merge upstream source trees into the harness core.

If a model runtime must be vendored to make the product path independent of a
manual upstream checkout, keep it isolated as a backend/runtime package with
provenance and license metadata. The core runner should still depend only on
the harness contract.

Instead, each optimization enters through one of these boundaries:

- `backend`: a local or remote implementation that can load, reset, warm up, and
  infer.
- `processor`: translation between harness observations/results and
  backend-native inputs/outputs.
- `optimization_profile`: configuration that turns an optimization on and sets
  its hyperparameters.
- `optimization_card`: product documentation for the profile, including scope,
  assumptions, telemetry fields, risks, and compatibility notes.

This mirrors the mature inference-framework pattern: vLLM exposes engine args
and JSON configs such as `--speculative-config`; SGLang exposes server flags and
YAML config; TensorRT-LLM exposes CLI flags plus extra YAML runtime options.
WAM Harness should use the same style, but with WAM-specific trace fields and
output-drift checks.

## Deployment Classes

Every method must be classified before implementation.

| Class | Meaning | Early deployment priority |
|---|---|---|
| `training_free_inference` | No weight update; changes runtime path, cache, scheduling, or token selection. | P0 |
| `runtime_system` | Runtime/kernel/server optimization; may require custom build or remote server. | P1 |
| `post_training` | Changes weights or representation after training, usually with calibration. | P2 |
| `training_recipe` | Requires training or fine-tuning data and scripts. | P3 |
| `model_architecture` | New model family or architecture whose efficiency comes from training-time design. | P3 |

The first runnable real integrations should come from P0 or a simple P1 remote
wrapper. P2/P3 methods can be backend targets, but they should not define the
initial harness core.

## Optimization Profile Shape

Harness configs should enable optimization through a small structured block:

```yaml
optimizations:
  enabled:
    - vla_cache
    - action_chunk_scheduling
  profiles:
    vla_cache:
      enabled: true
      cache_scope: replan
      invalidation: visual_change
      max_cache_bytes: null
    action_chunk_scheduling:
      replan_steps: 4
      action_horizon: 8
      mode: fixed
```

Rules:

- `enabled` gives the ordered set of active optimization profiles.
- Each profile must be serializable to runtime info and trace metadata.
- Profiles may set backend-private keys, but those keys must stay under that
  profile.
- `wam compare` compares profile changes, not hidden code edits.
- If two profiles conflict, config validation should fail before the backend
  loads.

## Toggle Contract

Each toggle must define:

- `name`: stable registry name, for example `vla_cache`.
- `class`: one of the deployment classes above.
- `default`: disabled unless a backend declares otherwise.
- `scope`: `request`, `replan`, `episode`, `run`, or `server`.
- `parameters`: typed hyperparameters and defaults.
- `requires`: backend capabilities or environment requirements.
- `conflicts`: incompatible toggles or backend modes.
- `trace_fields`: metrics or events the toggle must emit.
- `output_check`: how behavior is checked against a profile-disabled run.

Example:

```yaml
name: vla_cache
class: training_free_inference
default: false
scope: replan
parameters:
  cache_scope: replan
  invalidation: visual_change
  max_cache_bytes: null
requires:
  backend_capabilities: [kv_cache_reuse, visual_token_cache]
trace_fields:
  - cache_hit_rate
  - cache_update_ms
  - cache_bytes
output_check: action_drift_or_success_rate
```

## Lessons From Current Upstreams

The upstream survey was performed from shallow clones under
`/tmp/wam-harness-upstreams`.

| Upstream | Useful toggle pattern | Harness classification | Integration risk |
|---|---|---|---|
| VLA-Cache | CLI uses `--use_vla_cache True/False` for OpenVLA/OpenVLA-OFT LIBERO evaluation. | `training_free_inference` | Medium: relies on forked `transformers` and model-specific OpenVLA code. |
| FASTER | Policy server toggles `--infer-time-schedule=HAS`, `--alpha`, `--u0`, `--streaming`, `--early-stop-actions`; client toggles `--delay`, `--exec_horizon`. | Mixed: training recipe plus inference-time deployment scheduling. | High: OpenPI/JAX stack, checkpoint and robot/sim client assumptions. |
| FastWAM | Released-checkpoint evaluation uses Hydra-style overrides for task, checkpoint, dataset stats, and GPU count; RoboTwin config includes `skip_get_obs_within_replan`. | Model architecture/backend target with inference behavior to measure. | High: large Wan/FastWAM stack, simulator assets, checkpoint downloads. |
| ServoFlow | C++ runtime exposes condition cache, CUDA Graph, memory pool, fused ops, and benchmark binaries. | `runtime_system` | High: custom C++/CUDA build, RDT-specific runtime. |
| FastVLA | Combines 4-bit kernels, Triton action heads, action chunking, BC pretraining, and RL. | Mixed training recipe and runtime optimization. | Medium-high: useful as a model/backend target, not an early toggle. |
| TEAM-VLA | Training-free token compression direction. | `training_free_inference` | Unknown until a backend entrypoint is validated. |

VLA-Cache is the cleanest first candidate because it already exposes a direct
on/off flag. FASTER is valuable for action-chunk scheduling and streaming
contracts, but its method includes training-time assumptions, so the first
harness integration should reproduce its deployment toggles only after a policy
server adapter exists.

## Source Runbook

Source bring-up should happen outside this repository until a backend adapter is
ready.

### P0: VLA-Cache Smoke

Goal: verify baseline and cache-enabled OpenVLA evaluation commands reach model
load and emit enough timing/success data for a wrapper.

Expected upstream commands:

```bash
cd /tmp/wam-harness-upstreams/vla-cache/src/openvla
pip install -e .
python vla_cache_scripts/download_model_local.py \
  --model_id openvla/openvla-7b-finetuned-libero-spatial
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_cache False
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_cache True
```

Harness wrapper target:

```yaml
backend:
  name: openvla_libero
optimizations:
  enabled: [vla_cache]
  profiles:
    vla_cache:
      enabled: true
```

Blockers to record:

- checkpoint download size and location.
- LIBERO installation state.
- CUDA memory requirement on the target GPU.
- whether the cache path returns action chunks and timing in a machine-readable
  artifact.

### P1: FASTER Deployment Toggle Smoke

Goal: validate remote policy-server controls without reproducing training.

Expected upstream commands:

```bash
cd /tmp/wam-harness-upstreams/FASTER
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run scripts/serve_policy.py \
  --use-custom-sample-kwargs \
  --infer-time-schedule=HAS \
  --alpha=0.6 \
  --u0=0.9 \
  --streaming \
  --early-stop-actions=4 \
  policy:checkpoint \
  --policy.config=pi05_faster_agilex \
  --policy.dir=checkpoints/pi05_faster_agilex/my_experiment/49999
```

Harness wrapper target:

```yaml
backend:
  name: remote_policy
optimizations:
  enabled: [action_chunk_scheduling, streaming_actions]
  profiles:
    action_chunk_scheduling:
      mode: rtc
      delay_steps: 3
      exec_horizon: 4
    streaming_actions:
      enabled: true
      early_stop_actions: 4
```

Blockers to record:

- checkpoint availability.
- server metadata returned at connection.
- partial action message schema.
- TTFA and per-partial-action timing.

### P1: FastWAM Released Checkpoint Smoke

Goal: make FastWAM a backend target and measure its WAM-specific outputs before
adding runtime optimizations.

Expected upstream commands:

```bash
cd /tmp/wam-harness-upstreams/FastWAM
pip install -e .
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=1
```

Harness wrapper target:

```yaml
backend:
  name: fastwam_local
request:
  return_future: true
optimizations:
  enabled: [render_skip_within_replan]
  profiles:
    render_skip_within_replan:
      enabled: true
```

Blockers to record:

- Wan/FastWAM checkpoint size.
- LIBERO or RoboTwin environment setup.
- whether `MULTIRUN.num_gpus=1` is viable on the target GPU.
- future-frame/action artifact locations.

### P2: ServoFlow Runtime Smoke

Goal: treat ServoFlow as a runtime backend rather than a patch set.

Expected upstream commands:

```bash
cd /tmp/wam-harness-upstreams/servoflow
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSF_CUDA_ARCHS="86"
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
./build/benchmarks/bench_pipeline 10
```

Harness wrapper target:

```yaml
backend:
  name: servoflow_rdt
optimizations:
  enabled: [condition_cache, cuda_graph, static_memory_pool]
```

Blockers to record:

- whether current CUDA toolkit and compiler match the runtime.
- checkpoint converter status.
- Python binding or process boundary for harness calls.

## Metrics Required For Toggle Admission

An optimization is not admitted as `measured` until traces include:

- baseline and variant `run_id`.
- source repo and commit.
- enabled optimization profiles and parameter values.
- stage timing: preprocess, model, postprocess, total.
- memory: process RSS and CUDA peak/reserved when available.
- WAM workload shape: action horizon, replan steps, image shapes, state dims.
- output check result.
- comparison summary: faster/slower/same, output check result, and warnings.

## First Implementation Order

1. Add model entry/spec support for a fake model.
2. Add optimization profile parsing to the fake config schema.
3. Emit active profiles in `runtime_info` and `run_start`.
4. Add fake toggles for cache, compile, graph, and action scheduling; these do
   not optimize anything but prove trace shape and validation.
5. Add the core container recipe and generic container smoke path.
6. Add `wam serve fake-open-loop` for job-local serving inside a prepared
   runtime.
7. Add FastWAM as the first WAM backend target and curated model entry.
8. Bring up VLA-Cache as the first real `training_free_inference` case.
9. Add FASTER deployment controls after job-local policy-server traces work.
10. Treat ServoFlow as a runtime backend only after container/backend contracts
    are stable.
