# FastWAM LIBERO Eval Setup

This page records the current reproducible path for running the released
FastWAM LIBERO checkpoint through EazyWAM.

## What `wam prepare` Downloads

Use the verified eval asset group:

```bash
wam prepare fastwam-libero --cache-dir /path/to/wam-cache --download --asset eval
```

For FastWAM LIBERO, `eval` means:

- `checkpoint`: released FastWAM policy checkpoint, about 11.2 GiB.
- `dataset_stats`: released normalization statistics, about 40 KiB.
- `wan22_vae`: `Wan2.2_VAE.pth`, about 2.6 GiB.
- `wan22_t5_encoder`: `models_t5_umt5-xxl-enc-bf16.pth`, about 10.6 GiB.
- `wan21_tokenizer_*`: four Wan2.1 tokenizer files, about 21 MiB total.

It intentionally does not download:

- the full `Wan-AI/Wan2.2-TI2V-5B` repository snapshot;
- full Wan2.2 DiT/base weights that official FastWAM eval skips with
  `skip_dit_load_from_pretrain=True`;
- the full `Wan-AI/Wan2.1-T2V-1.3B` model snapshot.

The expected cache layout is:

```text
<cache-dir>/
  checkpoints/fastwam_release/
    libero_uncond_2cam224.pt
    libero_uncond_2cam224_dataset_stats.json
  diffsynth-models/Wan-AI/
    Wan2.2-TI2V-5B/
      Wan2.2_VAE.pth
      models_t5_umt5-xxl-enc-bf16.pth
    Wan2.1-T2V-1.3B/google/umt5-xxl/
      spiece.model
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
```

## Runtime Paths

FastWAM LIBERO now has two supported environment paths:

- **Self-managed uv environment**: use this when a machine or cluster cannot run
  Docker/Apptainer directly. This installs the same runtime dependencies into a
  dedicated Python environment.
- **Docker/prebuilt image**: use this when Docker or a site container launcher is
  available. The image contains EazyWAM, the vendored FastWAM runtime,
  LIBERO, robosuite, MuJoCo, and the helper commands.

Both paths use the same cache layout and the same product commands. Neither path
requires a user-provided FastWAM upstream checkout for native `run`,
`native-smoke`, `serve`, or `eval`.

## Self-Managed uv Runtime

When containers are not available, install the FastWAM runtime directly:

```bash
scripts/setup_fastwam_native_env.sh \
  --venv /path/to/.venv-fastwam \
  --cache-dir /path/to/wam-cache \
  --clone
```

What this does:

- creates a `uv` virtual environment;
- installs EazyWAM and the vendored FastWAM runtime;
- installs the FastWAM/LIBERO runtime dependencies, including robosuite and
  MuJoCo;
- clones LIBERO into `<cache-dir>/upstreams/LIBERO` when `--clone` is used;
- writes `<cache-dir>/libero/config/config.yaml`;
- runs a Python import smoke test.

What this does not do:

- it does not download the 25 GiB FastWAM eval assets;
- it does not submit scheduler jobs;
- it does not require the official FastWAM repo unless `--upstream-dir` is
  supplied for reference-eval parity checks.

Then run:

```bash
source /path/to/.venv-fastwam/bin/activate
export WAM_CACHE_DIR=/path/to/wam-cache
export LIBERO_CONFIG_PATH=$WAM_CACHE_DIR/libero/config

wam doctor fastwam-libero \
  --cache-dir "$WAM_CACHE_DIR"

wam prepare fastwam-libero \
  --cache-dir "$WAM_CACHE_DIR" \
  --download \
  --asset eval

wam native-smoke fastwam-libero \
  --cache-dir "$WAM_CACHE_DIR" \
  --trace-dir /path/to/runs \
  --require-ready
```

## Native Eval Acceptance

The portable end-to-end acceptance command is:

```bash
scripts/fastwam-libero-eval.sh \
  --cache-dir /path/to/wam-cache \
  --trace-dir /path/to/runs \
  --download-assets
```

The wrapper is scheduler-agnostic. Run it inside an activated self-managed
environment, inside a Docker/prebuilt image, or inside any already allocated GPU
shell. It auto-detects LIBERO at `<cache-dir>/upstreams/LIBERO` for local uv
installs and `/opt/LIBERO` for the Docker image. Override this with
`--libero-dir /path/to/LIBERO` if needed.

It runs the product path in order:

1. `wam prepare fastwam-libero --asset eval` for the minimal verified asset
   group.
2. `wam doctor fastwam-libero --strict` for asset and runtime readiness.
3. A LIBERO simulator preflight that creates the environment and converts one
   observation before loading the model.
4. `wam native-smoke fastwam-libero --require-ready`.
5. `wam eval fastwam-libero --workload libero-single-task --summary-path ...`.
6. `python -m eazywam.evals.acceptance ...` on the saved summary.

The wrapper defaults to EGL rendering:

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

Override those values only when the target runtime needs a different MuJoCo
rendering backend.

The summary is written under the selected trace directory:

```text
/path/to/runs/fastwam-libero-libero-single-task-eval-summary.json
/path/to/runs/fastwam-libero-libero-single-task-eval-output.txt
/path/to/runs/fastwam-libero-libero-single-task-acceptance.json
```

`*-eval-output.txt` keeps the raw `wam eval` console output. The clean
`*-eval-summary.json` is written directly by `wam eval --summary-path`, so the
wrapper does not need to scrape JSON out of stdout.

Re-check an existing run without rerunning the model:

```bash
python -m eazywam.evals.acceptance --json \
  /path/to/runs/fastwam-libero-libero-single-task-eval-summary.json \
  1 \
  1.0
```

Acceptance means the summary came from the harness-owned native eval path, the
trace contains `native_eval_end`, the trace finishes with
`run_end.status="ok"`, no `external_eval_plan` is present, and the eval results
JSON exists. The verifier also checks that the summary, trace `native_eval_end`,
and results JSON agree on trial count, successes, and success rate, and that
the success rate meets the requested minimum. Official FastWAM scripts are still
useful for parity checks, but they must be invoked explicitly with
`wam eval --reference` and are not accepted as product-path evidence.

For a completed run, keep both files. The summary describes the native eval
run, while `*-acceptance.json` is the machine-readable proof that the summary,
trace, results JSON, runtime metadata, trial count, and success-rate gate passed
the acceptance checks.

The native product path uses the FastWAM runtime vendored in EazyWAM and
does not need a FastWAM upstream checkout. Keep `--upstream-dir` only for
explicit reference-eval parity checks against the official scripts:

```bash
export WAM_FASTWAM_REPO=/path/to/FastWAM

wam eval fastwam-libero \
  --reference \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --cache-dir /path/to/wam-cache \
  --upstream-dir /path/to/FastWAM \
  --set mujoco_gl=egl \
  --set pyopengl_platform=egl
```

## Maintainer Verification Status

Current maintainer evidence was produced on SuperPod with the self-managed uv
runtime and one H800 GPU. The verified single-task run used
`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `task_id=0`, and `num_trials=1`.
It completed with:

```text
status=ok
success_rate=1.0
successes=1
total_episodes=1
steps=357
model_calls=36
duration_s=715.7
```

The acceptance report also passed with `min_success_rate=1.0`, and the saved
trace points to a native eval run with `runtime_info.backend=fastwam` and
`runtime_info.mode=simulator_eval`.

FastWAM `wam serve --smoke --smoke-input ...` has also been verified on the
same SuperPod runtime. The smoke request returned `status=ok` with action
shape `[32, 7]`, and the trace contained the expected resident-server lifecycle
events from `serve_start` through `backend_close`.

Reference-mode full-suite LIBERO manager eval has also been verified on
SuperPod through:

```bash
wam eval fastwam-libero \
  --reference \
  --workload libero-manager \
  --set num_trials=1
```

That run completed all 10 `libero_10` tasks with 10/10 successes and an
official-manager average success rate of 100%.

Native full-suite coverage has also been run as a sequential sweep over the
native `libero-single-task` runner for `task_id=0..9`, `num_trials=1`. That
native sweep completed structurally with 9/10 task successes; task_id=6 failed
after 700 simulator steps while the reference manager result for the same task
succeeded.

Task6 has since been repeated with `num_trials=5` on SuperPod H800. Before
control alignment, native used `num_steps_wait=5` and no explicit eval seed,
while the official reference script used `num_steps_wait=30` and `seed=42`.
After aligning `seed=42` and `num_steps_wait=30`, the native single-task path
scored 4/5 and the official reference single-task path also scored 4/5, with
the same failed episode index. This suggests task6 is not a deterministic
native failure, but native-vs-reference parity still needs more trials and
full-suite evidence. One caveat: the official FastWAM
manager overwrites `MULTIRUN.task_file` with a generated full-suite task list,
so task-only manager parity needs a wrapper around the lower-level manager
script rather than the public manager entrypoint.

This evidence proves the current single-task product path, serve smoke path,
reference full-suite eval path, and native full-suite sweep execution. It does
not yet prove native-vs-reference parity, long running serve stability, or a
minimum VRAM requirement on smaller GPUs. The SuperPod Slurm `MaxRSS` for the
single-task native eval job was about 37 GiB of process RSS; this is not a GPU
VRAM measurement. Keep using large-memory GPUs until a separate VRAM profiling
run establishes a tighter floor.

## Docker / Prebuilt Image Runtime

Build the FastWAM image from the repository root:

```bash
docker build \
  -f containers/fastwam/Dockerfile \
  -t eazywam-fastwam:cu128 \
  .
```

Run the same acceptance path inside the image:

```bash
mkdir -p /path/to/wam-cache /path/to/runs

docker run --rm --gpus all \
  -v /path/to/wam-cache:/mnt/wam-cache \
  -v /path/to/runs:/mnt/runs \
  eazywam-fastwam:cu128 \
  wam-fastwam-libero-eval \
    --cache-dir /mnt/wam-cache \
    --trace-dir /mnt/runs \
    --download-assets \
    --mujoco-gl egl \
    --pyopengl-platform egl
```

The image defines these defaults:

```text
WAM_CACHE_DIR=/mnt/wam-cache
WAM_TRACE_DIR=/mnt/runs
WAM_LIBERO_DIR=/opt/LIBERO
LIBERO_CONFIG_PATH=/mnt/wam-cache/libero/config
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

`wam-fastwam-libero-eval` writes the LIBERO config file into the mounted cache
when needed, pointing it at `/opt/LIBERO` inside the image. This avoids the
common failure where a host cache mount hides a config file created during image
build.

If the runtime has no internet access, prepare the cache elsewhere with:

```bash
wam prepare fastwam-libero --cache-dir /path/to/wam-cache --download --asset eval
```

Then mount that populated cache into the container and omit `--download-assets`.

The public project should keep Dockerfile-compatible recipes, but some rented
GPU services start users inside an existing container and do not allow nested
Docker. In that case, use the self-managed path above. Test Docker images on a
host or cluster runtime that provides Docker, Pyxis/Enroot, Apptainer, or
Singularity directly. If converting Docker images to Apptainer/SIF, make sure
the converted image is built from the current Dockerfile; older FastWAM SIFs may
miss LIBERO simulator dependencies such as `robosuite`.
