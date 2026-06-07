# Containers

Container recipes are the portable environment definitions for EazyWAM.

- `core/Dockerfile`: lightweight core image for the fake backend, `wam run`, and
  `wam serve --smoke`.
- `fastwam/Dockerfile`: FastWAM + LIBERO + MuJoCo evaluation environment.
- `cosmos-policy/Dockerfile`: Cosmos-Policy LIBERO evaluation environment based
  on the upstream uv dependency groups.
- `dreamzero/Dockerfile`: DreamZero DROID policy-server and sim-eval
  environment.

EazyWAM supports two heavy-backend setup paths:

- **Container path:** build or publish a backend image, then run the normal
  `wam` commands inside the site's container launcher.
- **Self-managed path:** when Docker, Apptainer, Singularity, or the local
  cluster container runtime is not available, install the same backend runtime
  into a dedicated Python environment and run the normal `wam` commands there.

Build locally:

```bash
docker build -f containers/core/Dockerfile -t eazywam-core:latest .
docker build -f containers/fastwam/Dockerfile -t eazywam-fastwam:cu128 .
docker build -f containers/cosmos-policy/Dockerfile -t eazywam-cosmos-policy:latest .
docker build -f containers/dreamzero/Dockerfile -t eazywam-dreamzero:latest .
```

On a cluster, run these images through the site's normal container launcher.
The harness does not prescribe scheduler commands or site-specific launch
mechanics.

Real backend simulator evaluations use the same harness entrypoint but require
a backend-specific image and model/cache storage mounted into the container:

```bash
wam doctor fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --json \
  --strict

wam native-smoke fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --trace-dir /mnt/runs \
  --require-ready

wam eval fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --trace-dir /mnt/runs \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1
```

Backend images install the `wam` CLI as part of the image. This is required for
the native migration path: the heavy dependency environment and the harness
entrypoint must be in the same container so `wam doctor`, `wam native-smoke`,
`wam run`, and `wam serve` see the same Python modules and mounted assets.
Reference eval may still mount upstream source explicitly with `--upstream-dir`.

For FastWAM, the image calls `scripts/setup_fastwam_native_env.sh` during build.
That script is also the public self-managed install path, so the container and
non-container environments share the same dependency list.

For FastWAM, the expected container-internal shape is:

```text
/workspace/eazywam  installed harness package and working directory
/opt/LIBERO             upstream LIBERO checkout
/mnt/wam-cache          mounted model/cache directory
/mnt/runs               mounted trace/output directory
```

The FastWAM image sets these defaults:

```text
WAM_CACHE_DIR=/mnt/wam-cache
WAM_TRACE_DIR=/mnt/runs
WAM_LIBERO_DIR=/opt/LIBERO
LIBERO_CONFIG_PATH=/mnt/wam-cache/libero/config
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

Then run the full acceptance path:

```bash
mkdir -p /path/to/wam-cache /path/to/runs

docker run --rm --gpus all \
  -v /path/to/wam-cache:/mnt/wam-cache \
  -v /path/to/runs:/mnt/runs \
  eazywam-fastwam:cu128 \
  wam-fastwam-libero-eval \
    --cache-dir /mnt/wam-cache \
    --trace-dir /mnt/runs \
    --download-assets
```

Or run the individual gates:

```bash
wam doctor fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --json \
  --strict

wam native-smoke fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --trace-dir /mnt/runs \
  --require-ready
```

Backend images also include a container-internal native smoke command that runs
the same gates in order:

```bash
wam-fastwam-native-smoke
```

FastWAM images also include the end-to-end native eval acceptance wrapper:

```bash
wam-fastwam-libero-eval
```

It runs `wam prepare --asset eval`, `wam doctor --strict`, `wam native-smoke
--require-ready`, and then the harness-owned `wam eval fastwam-libero
--workload libero-single-task` product path. Set `WAM_PREPARE_DOWNLOAD=1` or
pass `--download-assets` when the container should fetch missing model assets.
The wrapper saves the eval summary in the trace directory and immediately
validates it with `python -m eazywam.evals.acceptance`.

If the site cannot download from Hugging Face inside the GPU job, prepare the
cache elsewhere and mount it into `/mnt/wam-cache`. For FastWAM LIBERO, the
minimal eval asset group is:

```bash
wam prepare fastwam-libero --cache-dir /path/to/wam-cache --download --asset eval
```

This is about 25 GiB and does not download full Wan repository snapshots.

To re-check a saved run without rerunning the model:

```bash
python -m eazywam.evals.acceptance --json \
  /mnt/runs/fastwam-libero-libero-single-task-eval-summary.json \
  1 \
  1.0
```

The wrapper saves this JSON report as
`/mnt/runs/fastwam-libero-libero-single-task-acceptance.json`.

It expands to:

```bash
wam prepare fastwam-libero --cache-dir /mnt/wam-cache || prepare_status=$?
wam doctor fastwam-libero --cache-dir /mnt/wam-cache --json --strict
wam native-smoke fastwam-libero --cache-dir /mnt/wam-cache --trace-dir /mnt/runs --require-ready
```

`prepare` is allowed to report incomplete assets so the script can still print
the stricter native readiness report from `doctor --json --strict`. The doctor
gate is the authoritative preflight for Python modules, required assets,
runtime assets, optional reference source overrides, and the declared model
adapter.

The analogous backend commands are `wam-cosmos-policy-native-smoke` and
`wam-dreamzero-native-smoke`. Override `WAM_MODEL_ID`, `WAM_CACHE_DIR`,
`WAM_UPSTREAM_DIR`, `WAM_TRACE_DIR`, or set `WAM_PREPARE_DOWNLOAD=1` when a
site-specific launcher binds different paths or wants the script to download
missing pullable assets.

The checked-in Dockerfiles define the intended portable environments. Build or
publish them for the runtime your site supports before expecting long simulator
evaluations to be reproducible across machines.

For Apptainer/Singularity users, convert or build from the current Dockerfile.
Older local SIFs may only contain the FastWAM PyTorch runtime and can miss
simulator packages such as `robosuite`; those images will pass `wam doctor` but
fail at the LIBERO simulator import gate.

The checked-in Apptainer definitions are compatibility starting points for
sites that cannot run Docker images directly. The Dockerfiles are the current
source of truth for full backend environments and harness CLI installation.

## Self-Managed FastWAM Environment

Use this path when a machine has a usable CUDA/Python toolchain but cannot run a
container runtime:

```bash
scripts/setup_fastwam_native_env.sh \
  --venv /path/to/.venv-fastwam \
  --cache-dir /path/to/wam-cache \
  --clone
```

The script creates a `uv` virtual environment, clones LIBERO if requested,
installs the EazyWAM CLI and vendored FastWAM runtime into that
environment, installs the FastWAM/LIBERO runtime dependencies, writes a LIBERO
config file, and runs a Python import smoke test. It does not submit jobs,
download large model assets, or choose a cluster launcher. It clones FastWAM
only when `--upstream-dir` is supplied for optional reference-eval parity.

After installation:

```bash
source /path/to/.venv-fastwam/bin/activate
export WAM_CACHE_DIR=/path/to/wam-cache
export LIBERO_CONFIG_PATH=/path/to/wam-cache/libero/config

wam doctor fastwam-libero --cache-dir "$WAM_CACHE_DIR"
wam prepare fastwam-libero --cache-dir "$WAM_CACHE_DIR"
wam native-smoke fastwam-libero \
  --cache-dir "$WAM_CACHE_DIR" \
  --trace-dir /path/to/runs \
  --require-ready

scripts/fastwam-libero-eval.sh \
  --cache-dir "$WAM_CACHE_DIR" \
  --trace-dir /path/to/runs
```

Use `wam prepare ... --download --asset ...` only after the target storage mount
is ready for the released FastWAM checkpoint and runtime assets.
