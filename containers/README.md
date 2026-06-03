# Containers

Container recipes are the portable environment definitions for WAM Harness.

- `core/Dockerfile`: lightweight core image for the fake backend, `wam run`, and
  `wam serve --smoke`.
- `fastwam/Dockerfile`: FastWAM + LIBERO + MuJoCo evaluation environment.
- `cosmos-policy/Dockerfile`: Cosmos-Policy LIBERO evaluation environment based
  on the upstream uv dependency groups.
- `dreamzero/Dockerfile`: DreamZero DROID policy-server and sim-eval
  environment.

WAM Harness supports two heavy-backend setup paths:

- **Container path:** build or publish a backend image, then run the normal
  `wam` commands inside the site's container launcher.
- **Self-managed path:** when Docker, Apptainer, Singularity, or the local
  cluster container runtime is not available, install the same backend runtime
  into a dedicated Python environment and run the normal `wam` commands there.

Build locally:

```bash
docker build -f containers/core/Dockerfile -t wam-harness-core:latest .
docker build -f containers/fastwam/Dockerfile -t wam-harness-fastwam:latest .
docker build -f containers/cosmos-policy/Dockerfile -t wam-harness-cosmos-policy:latest .
docker build -f containers/dreamzero/Dockerfile -t wam-harness-dreamzero:latest .
```

On a cluster, run these images through the site's normal container launcher.
The harness does not prescribe scheduler commands or site-specific launch
mechanics.

Real backend simulator evaluations use the same harness entrypoint but require
a backend-specific image and upstream repo checkout mounted into the container:

```bash
wam doctor fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --upstream-dir /mnt/upstream \
  --json \
  --strict

wam native-smoke fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --upstream-dir /mnt/upstream \
  --trace-dir /mnt/runs \
  --require-ready

wam eval fastwam-libero --reference --trace-dir /mnt/runs --upstream-dir /mnt/upstream
```

Backend images install the `wam` CLI as part of the image. This is required for
the native migration path: the heavy dependency environment and the harness
entrypoint must be in the same container so `wam doctor`, `wam native-smoke`,
`wam run`, and `wam serve` see the same Python modules, upstream source, and
mounted assets.

For FastWAM, the image calls `scripts/setup_fastwam_native_env.sh` during build.
That script is also the public self-managed install path, so the container and
non-container environments share the same dependency list.

For FastWAM, the expected container-internal shape is:

```text
/workspace/wam-harness  installed harness package and working directory
/opt/FastWAM            upstream FastWAM checkout
/opt/LIBERO             upstream LIBERO checkout
/mnt/wam-cache          mounted model/cache directory
/mnt/runs               mounted trace/output directory
```

Then run:

```bash
wam doctor fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --upstream-dir /opt/FastWAM \
  --json \
  --strict

wam native-smoke fastwam-libero \
  --cache-dir /mnt/wam-cache \
  --upstream-dir /opt/FastWAM \
  --trace-dir /mnt/runs \
  --require-ready
```

Backend images also include a container-internal native smoke command that runs
the same gates in order:

```bash
wam-fastwam-native-smoke
```

It expands to:

```bash
wam prepare fastwam-libero --cache-dir /mnt/wam-cache || prepare_status=$?
wam doctor fastwam-libero --cache-dir /mnt/wam-cache --upstream-dir /opt/FastWAM --json --strict
wam native-smoke fastwam-libero --cache-dir /mnt/wam-cache --upstream-dir /opt/FastWAM --trace-dir /mnt/runs --require-ready
```

`prepare` is allowed to report incomplete assets so the script can still print
the stricter native readiness report from `doctor --json --strict`. The doctor
gate is the authoritative preflight for upstream source, Python modules,
required assets, runtime assets, and the declared model adapter.

The analogous backend commands are `wam-cosmos-policy-native-smoke` and
`wam-dreamzero-native-smoke`. Override `WAM_MODEL_ID`, `WAM_CACHE_DIR`,
`WAM_UPSTREAM_DIR`, `WAM_TRACE_DIR`, or set `WAM_PREPARE_DOWNLOAD=1` when a
site-specific launcher binds different paths or wants the script to download
missing pullable assets.

The checked-in Dockerfiles define the intended portable environments. Build or
publish them for the runtime your site supports before expecting long simulator
evaluations to be reproducible across machines.

The checked-in Apptainer definitions are compatibility starting points for
sites that cannot run Docker images directly. The Dockerfiles are the current
source of truth for full backend environments and harness CLI installation.

## Self-Managed FastWAM Environment

Use this path when a machine has a usable CUDA/Python toolchain but cannot run a
container runtime:

```bash
scripts/setup_fastwam_native_env.sh \
  --upstream-dir /path/to/FastWAM \
  --venv /path/to/.venv-fastwam \
  --cache-dir /path/to/wam-cache \
  --clone
```

The script creates a `uv` virtual environment, clones FastWAM and LIBERO if
requested, installs the WAM Harness CLI into that environment, installs the
FastWAM/LIBERO runtime dependencies, writes a LIBERO config file, and runs a
Python import smoke test. It does not submit jobs, download large model assets,
or choose a cluster launcher.

After installation:

```bash
source /path/to/.venv-fastwam/bin/activate
export WAM_FASTWAM_REPO=/path/to/FastWAM
export WAM_CACHE_DIR=/path/to/wam-cache
export LIBERO_CONFIG_PATH=/path/to/wam-cache/libero/config

wam doctor fastwam-libero --cache-dir "$WAM_CACHE_DIR" --upstream-dir "$WAM_FASTWAM_REPO"
wam prepare fastwam-libero --cache-dir "$WAM_CACHE_DIR"
wam native-smoke fastwam-libero \
  --cache-dir "$WAM_CACHE_DIR" \
  --upstream-dir "$WAM_FASTWAM_REPO" \
  --trace-dir /path/to/runs \
  --require-ready
```

Use `wam prepare ... --download --asset ...` only after the target storage mount
is ready for the released FastWAM checkpoint and runtime assets.
