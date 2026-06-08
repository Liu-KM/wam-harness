<p align="center">
  <img src="docs/assets/logo/eazywam-logo-readme.png" alt="EazyWAM logo" width="760">
</p>

<p align="center">
  <a href="README.zh-CN.md">中文</a> |
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
</p>

# EazyWAM

EazyWAM is a deployment and acceleration harness for world-action models. It
turns scattered WAM checkpoints, runtimes, assets, eval scripts, serving paths,
optimization toggles, and traces into one model-centric `wam` workflow.

## Quickstart

Install the core CLI from PyPI:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install eazywam
```

For development, clone the repository instead:

```bash
git clone https://github.com/Liu-KM/eazywam.git
cd eazywam
```

### Create A Source Environment

Use `uv` to create a clean Python 3.10+ environment and install the local
checkout. This installs the core package dependencies declared in
`pyproject.toml`; there is no separate `requirements.txt` step for the core
CLI.

Option A: manual setup

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Option B: setup script

```bash
scripts/setup_core_env.sh
source .venv/bin/activate
```

Heavy WAM runtimes, checkpoints, simulators, and GPU dependencies are handled by
model-specific `doctor` and `prepare` paths.

If you prefer Conda, create and activate a Python 3.10+ Conda environment, then
install the checkout with `python -m pip install -e .`.

### Discover Models

```bash
wam list
wam info fastwam-libero
```

### Verify The Local Harness

```bash
wam run fake-open-loop
wam run fake-open-loop --opt fake_cache
```

### Verify The Local Policy Server

```bash
wam serve fake-open-loop --smoke
```

### Prepare FastWAM Assets

```bash
wam doctor fastwam-libero --cache-dir /path/to/wam-cache
wam prepare fastwam-libero --cache-dir /path/to/wam-cache --download --asset eval
```

### Run FastWAM With An Observation

```bash
wam run fastwam-libero \
  --input examples/fastwam_libero/obs.json \
  --output /tmp/fastwam-action.json \
  --cache-dir /path/to/wam-cache
```

### Evaluate FastWAM In LIBERO

Run this inside a prepared FastWAM runtime.

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --cache-dir /path/to/wam-cache
```

### Inspect Traces

```bash
ls runs/*/trace.jsonl
```

## Model Library

The model library lists curated real WAM entries. The built-in smoke-test
backend is useful for checking the harness contract locally, but it is not a
model-library entry.

| Model id | Upstream | First command | Status |
| --- | --- | --- | --- |
| `fastwam-libero` | [![GitHub](https://img.shields.io/badge/GitHub-FastWAM-181717?logo=github)](https://github.com/yuantianyuan01/FastWAM) [![Hugging Face](https://img.shields.io/badge/HF-yuanty%2Ffastwam-FFD21E?logo=huggingface)](https://huggingface.co/yuanty/fastwam) | `wam prepare fastwam-libero --download --asset eval` | First real integration target. SuperPod H800 single-task native eval, serve smoke, reference full-suite eval, and native full-suite sweep are run; native sweep is 9/10, and aligned task6 evidence shows native and reference both at 4/5. |
| `cosmos-policy-libero` | [![GitHub](https://img.shields.io/badge/GitHub-Cosmos--Policy-181717?logo=github)](https://github.com/NVlabs/cosmos-policy) [![Hugging Face](https://img.shields.io/badge/HF-Cosmos--Policy--LIBERO-FFD21E?logo=huggingface)](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) | `wam info cosmos-policy-libero` | Native smoke and official-script parity integration started. |
| `dreamzero-droid-sim` | [![GitHub](https://img.shields.io/badge/GitHub-DreamZero-181717?logo=github)](https://github.com/dreamzero0/dreamzero) [![Hugging Face](https://img.shields.io/badge/HF-DreamZero--DROID-FFD21E?logo=huggingface)](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) [![Hugging Face](https://img.shields.io/badge/HF-DROID_sim_assets-FFD21E?logo=huggingface)](https://huggingface.co/owhan/DROID-sim-environments) | `wam info dreamzero-droid-sim` | Resident policy-server path started; DROID sim path requires a heavier multi-GPU runtime. |

## Commands

```bash
wam --help
wam <command> --help
wam list
wam info <model-id>
wam doctor <model-id>
wam prepare <model-id>
wam run <model-id> --input obs.json --output action.json
wam eval <model-id> --workload <workload>
wam serve <model-id>
wam compare <trace-a> <trace-b>
```

## Development

Core development uses `uv` for repeatable local checks:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

## Documentation

- `docs/cli_entrypoints.md` - command behavior.
- `docs/fastwam_libero_eval_setup.md` - FastWAM setup and eval workflow.
- `docs/dependency_isolation.md` - containers and self-managed environments.
- `docs/wamfile.md` - model entry schema.
- `docs/optimization_integration.md` - optimization profile design.
- `docs/trace_schema.md` - trace event schema.
- `docs/roadmap.md` - current milestones.

## License

EazyWAM is released under the [MIT License](LICENSE.md). Vendored third-party
code and external model assets remain under their respective upstream licenses.
