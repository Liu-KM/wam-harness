# Model Entry Spec

A model entry is the curated unit users see in WAM Harness. It is the WAM
analogue of the default layer that makes local model tools easy to use: one
model id should resolve to the backend, processor, assets, request defaults, and
known optimization profiles needed to run inference.

The YAML file behind a model entry is the model spec. The implementation may
call this file a manifest, but user-facing docs should prefer "model entry" and
"model spec." The spec is maintained by the project or backend owner, while
runtime capability details are still reported by the backend or server.

## Goals

- Make `wam run <model-id> --input obs.json` possible without a large run config.
- Make `wam run <model-id> --input obs.json` possible for curated native model entries.
- Keep official simulator evaluators available as explicit
  `wam eval <model-id> --reference` paths.
- Make `wam info <model-id>` and `wam prepare <model-id>` possible without
  asking users to read YAML.
- Record the non-weight assets WAMs need, such as dataset stats and normalizers.
- Keep backend-specific keys out of the core runner.
- Publish known hardware requirements and gaps.
- Declare which optimization profiles are supported and measured.

## Non-Goals

- A full training recipe.
- A replacement for upstream documentation.
- A universal robotics environment spec.
- A place to store backend-native tensors or large artifacts.

## Example

```yaml
schema_version: 1
id: fastwam-libero
display_name: FastWAM LIBERO 2-camera

source:
  repo: yuantianyuan01/FastWAM
  reference: Fast-WAM: Do World Action Models Need Test-time Future Imagination?
  license: unknown

assets:
  checkpoint:
    uri: hf://yuanty/fastwam/libero_uncond_2cam224.pt
    local_path: checkpoints/fastwam_release/libero_uncond_2cam224.pt
    size_bytes: 12041735140
  dataset_stats:
    uri: hf://yuanty/fastwam/libero_uncond_2cam224_dataset_stats.json
    local_path: checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
    size_bytes: 40939
  wan22_vae:
    uri: hf://Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
    local_path: diffsynth-models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
  wan22_t5_encoder:
    uri: hf://Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
    local_path: diffsynth-models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
  wan21_tokenizer_spiece:
    uri: hf://Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model
    local_path: diffsynth-models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model
  wan21_tokenizer_json:
    uri: hf://Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json
    local_path: diffsynth-models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json
  wan21_tokenizer_config:
    uri: hf://Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json
    local_path: diffsynth-models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json
  wan21_special_tokens_map:
    uri: hf://Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json
    local_path: diffsynth-models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json

asset_groups:
  eval:
    assets:
      - checkpoint
      - dataset_stats
      - wan22_vae
      - wan22_t5_encoder
      - wan21_tokenizer_spiece
      - wan21_tokenizer_json
      - wan21_tokenizer_config
      - wan21_special_tokens_map

backend:
  name: fastwam
  mode: local
  extra: fastwam
  config:
    task: libero_uncond_2cam224_1e-4

processor:
  name: fastwam_libero
  observation:
    image_views: [primary, wrist]
    state: proprio
    prompt: required
  action:
    horizon: 32
    dim: 7
    control_hz: null

defaults:
  device: cuda:0
  dtype: bf16
  action_horizon: 32
  replan_steps: 10
  return_future: false
  return_value: false

optimizations:
  supported:
    - action_chunk_scheduling
    - render_skip_within_replan
  measured: []
  unsupported:
    vla_cache: Different OpenVLA backend family.

deployment:
  reference_path: official_script
  product_path: native_backend_migration
  native_backend: fastwam
  native_stage: skeleton_unverified
  native_verified: false
  parity_verified: false
  next_gate: native_smoke

known_gaps:
  - Requires external LIBERO installation for simulator evaluation.
  - Released evaluation managers default to multi-GPU settings upstream.
```

For current official simulator evaluations, a model spec may also declare an
`eval` section. This is a reference-evaluator path for upstream scripts. It is
useful for bring-up and parity, but native backends should become the product
path once validated:

```yaml
workload:
  name: external_eval
  config:
    simulator: LIBERO
    default_eval_workload: libero-manager
    eval_workloads:
      - libero-manager
      - libero-single-task

eval:
  simulator: LIBERO
  suite: libero_10
  default_workload: libero-manager
  upstream:
    repo: yuantianyuan01/FastWAM
    commit: 45d8e14
    local_env: WAM_FASTWAM_REPO
    default_dir: /workspace/FastWAM
  defaults:
    checkpoint_path: ./checkpoints/fastwam_release/libero_uncond_2cam224.pt
    dataset_stats_path: ./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
    num_trials: "1"
  workloads:
    libero-manager:
      command:
        workdir: "{upstream_dir}"
        argv:
          - python
          - experiments/libero/run_libero_manager.py
          - ckpt={checkpoint_path}
          - EVALUATION.dataset_stats_path={dataset_stats_path}
          - EVALUATION.output_dir={output_dir}/fastwam_libero
    libero-single-task:
      defaults:
        task_id: "0"
      command:
        workdir: "{upstream_dir}"
        argv:
          - python
          - experiments/libero/eval_libero_single.py
          - ckpt={checkpoint_path}
          - EVALUATION.dataset_stats_path={dataset_stats_path}
          - EVALUATION.task_id={task_id}
          - EVALUATION.num_trials={num_trials}
```

Use eval workloads for different official evaluation entrypoints that do not
change the model identity. For example, FastWAM's LIBERO manager and single-task
smoke evaluator share the same checkpoint, processor, and action schema, so
they should be selected with `wam eval fastwam-libero --workload
libero-single-task`, not with a separate `fastwam-libero-single-task` model id.

## Required Fields

| Field | Purpose |
|---|---|
| `schema_version` | Model spec schema version. |
| `id` | Stable model id used by `wam info`, `wam prepare`, `wam eval`, and `wam serve`. |
| `source.repo` | Upstream source repository. |
| `assets` | Pullable or user-provided artifacts. |
| `backend.name` | Registry key for backend creation. |
| `processor.name` | Registry key for observation/result conversion. |
| `defaults` | Run defaults used when the user does not pass a config. |
| `optimizations.supported` | Profiles that can be enabled for this model entry. |
| `known_gaps` | Explicit limitations. |

## Deployment Status

`deployment` records where the model entry is in the migration from official
scripts to a native product path. It is metadata for users, docs, and release
gates; it should not change runner behavior by itself.

Recommended fields:

| Field | Purpose |
|---|---|
| `reference_path` | Existing official-script or official-server path kept for parity. |
| `product_path` | Intended product path, for example `native_backend_migration`. |
| `native_backend` | Registry key for the native backend target. |
| `native_stage` | Current migration stage, such as `skeleton_unverified`, `native_smoke_verified`, or `parity_verified`. |
| `native_verified` | Whether a real checkpoint/container native smoke has passed. |
| `parity_verified` | Whether native output has been compared against the reference path. |
| `next_gate` | Next concrete gate, such as `native_smoke` or `reference_parity`. |

## Native Versus Reference Paths

A mature model entry should make the native backend explicit:

```yaml
backend:
  name: fastwam
  mode: native
processor:
  name: fastwam_libero
workload:
  name: libero
```

Official scripts should be retained as reference evaluators, either through an
explicit reference entry or a future `reference_eval` section. They should not
remain the only product execution path for a real WAM model.

During migration, a reference entry may still use `external_eval` while declaring
which native backend should be used for maintainer smoke validation:

```yaml
backend:
  name: external_eval
  mode: official_script
  config:
    upstream: FastWAM
    simulator: LIBERO
    native_backend: fastwam
processor:
  name: fastwam_libero
```

`wam native-smoke <model-id>` reads `backend.config.native_backend`, temporarily
creates the native backend, and asks the registered processor for a synthetic
contract observation. Adding a new model should not require a branch in the core
runner or native smoke runner.

The native backend implementation, not the core runner, declares the concrete
runtime requirements for this migration path: required Python modules, required
assets, runtime cache assets, and optional upstream files when a reference
checkout override is supplied. `wam doctor <model-id>` reads those requirements
through the backend registry and reports whether the current container or
machine has the right runtime and mounts. For FastWAM, the product native path
uses vendored runtime code; the `eval.upstream` section is only for official
reference-eval parity.

## Asset URI Policy

Use source-of-truth asset hosts instead of vendoring large files:

- `hf://org/repo/path` for Hugging Face Hub file assets.
- `hf://org/repo` for Hugging Face Hub repository snapshots.
- `https://...` for direct downloads when legally allowed.
- `local-or-user-provided` when redistribution is unclear or too large.

`wam prepare <model-id> --download` can materialize `hf://` assets into their
declared `local_path`. A repeated `--asset <name>` filter lets maintainers fetch
only the assets needed for native smoke bring-up without pulling every large
model component in the entry. `asset_groups` provide stable aliases such as
`eval` for a verified minimal set of file assets.

Relative `local_path` values are resolved under the command's cache directory.
Use the same `--cache-dir` across `prepare`, `doctor`, `run`, `native-smoke`,
and `serve` so native backends load the assets that were prepared.

`size_bytes` is optional metadata. It helps `wam prepare` and docs make large
downloads visible before a user starts moving model weights. It does not change
where the asset is stored or whether the asset is required for native readiness.

The model spec may specify expected local paths, checksums, and size hints in a
later schema version. The first version should prioritize clear source and
location over exhaustive metadata.

## Optimization Status

An optimization can appear in one of three states:

- `supported`: the backend can enable the profile, but it may not have a
  published measured result yet.
- `measured`: comparison traces exist and the profile has published telemetry
  notes.
- `unsupported`: known incompatible profile with a short reason.

Only `measured` profiles should be marketed as proven speedups.

## Relationship To Config

A user-facing run can be short:

```bash
wam run fastwam-libero --input obs.json --opt action_chunk_scheduling
```

Internally this expands to:

- model entry defaults.
- backend config.
- processor config.
- request config.
- optimization profile config.
- trace metadata.

Advanced users can still provide explicit YAML configs, but curated model
entries are the default path for adoption.
