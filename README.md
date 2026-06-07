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

### Install

```bash
uv sync --dev
```

### Discover Models

```bash
uv run wam list
uv run wam info fake-open-loop
uv run wam info fastwam-libero
```

### Run A CPU Smoke Model

```bash
uv run wam run fake-open-loop
uv run wam run fake-open-loop --opt fake_cache
```

### Start A Local Policy Server Smoke Test

```bash
uv run wam serve fake-open-loop --smoke
```

### Prepare FastWAM Assets

```bash
uv run wam doctor fastwam-libero --cache-dir /path/to/wam-cache
uv run wam prepare fastwam-libero --cache-dir /path/to/wam-cache --download --asset eval
```

### Run FastWAM With An Observation

```bash
uv run wam run fastwam-libero \
  --input examples/fastwam_libero/obs.json \
  --output /tmp/fastwam-action.json \
  --cache-dir /path/to/wam-cache
```

### Evaluate FastWAM In LIBERO

Run this inside a prepared FastWAM runtime.

```bash
uv run wam eval fastwam-libero \
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

| Model id | Links | Minimal command | Status |
| --- | --- | --- | --- |
| `fake-open-loop` | built in | `uv run wam run fake-open-loop` | Stable CPU smoke path; no real WAM weights. |
| `fastwam-libero` | [![GitHub](https://img.shields.io/badge/GitHub-FastWAM-181717?logo=github)](https://github.com/yuantianyuan01/FastWAM) [![HF](https://img.shields.io/badge/HF-yuanty%2Ffastwam-FFD21E?logo=huggingface)](https://huggingface.co/yuanty/fastwam) | `uv run wam prepare fastwam-libero --download --asset eval` | First real WAM target; native run/serve path started; LIBERO eval path available. |
| `cosmos-policy-libero` | [![GitHub](https://img.shields.io/badge/GitHub-cosmos--policy-181717?logo=github)](https://github.com/NVlabs/cosmos-policy) [![HF](https://img.shields.io/badge/HF-Cosmos--Policy--LIBERO-FFD21E?logo=huggingface)](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) | `uv run wam info cosmos-policy-libero` | Native smoke and official-script parity integration started. |
| `dreamzero-droid-sim` | [![GitHub](https://img.shields.io/badge/GitHub-DreamZero-181717?logo=github)](https://github.com/dreamzero0/dreamzero) [![HF](https://img.shields.io/badge/HF-DreamZero--DROID-FFD21E?logo=huggingface)](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) [![HF gated](https://img.shields.io/badge/HF-gated_DROID_sim_assets-FFD21E?logo=huggingface)](https://huggingface.co/owhan/DROID-sim-environments) | `uv run wam info dreamzero-droid-sim` | Resident policy-server path started; DROID sim path requires a heavier multi-GPU runtime. |

## Commands

```bash
wam list
wam info <model-id>
wam doctor <model-id>
wam prepare <model-id>
wam run <model-id> --input obs.json --output action.json
wam eval <model-id> --workload <workload>
wam serve <model-id>
wam compare <trace-a> <trace-b>
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
