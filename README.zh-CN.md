<p align="center">
  <img src="docs/assets/logo/eazywam-logo-readme.png" alt="EazyWAM logo" width="760">
</p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
</p>

# EazyWAM

EazyWAM 是一个面向 world-action model 的部署与推理加速框架。它把分散在不同
仓库里的 checkpoint、运行环境、资产准备、评测脚本、服务入口、优化开关和
trace，整理成一个以 model id 为中心的 `wam` 工作流。

## 快速上手

### 安装

```bash
uv sync --dev
```

### 查看模型

```bash
uv run wam list
uv run wam info fake-open-loop
uv run wam info fastwam-libero
```

### 运行 CPU smoke 模型

```bash
uv run wam run fake-open-loop
uv run wam run fake-open-loop --opt fake_cache
```

### 启动本地 policy server smoke 测试

```bash
uv run wam serve fake-open-loop --smoke
```

### 准备 FastWAM 资产

```bash
uv run wam doctor fastwam-libero --cache-dir /path/to/wam-cache
uv run wam prepare fastwam-libero --cache-dir /path/to/wam-cache --download --asset eval
```

### 用一条 observation 跑 FastWAM

```bash
uv run wam run fastwam-libero \
  --input examples/fastwam_libero/obs.json \
  --output /tmp/fastwam-action.json \
  --cache-dir /path/to/wam-cache
```

### 在 LIBERO 里评测 FastWAM

需要在已经准备好的 FastWAM runtime 里运行。

```bash
uv run wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --cache-dir /path/to/wam-cache
```

### 查看 trace

```bash
ls runs/*/trace.jsonl
```

## 模型库

| Model id | 链接 | 最小命令 | 当前状态 |
| --- | --- | --- | --- |
| `fake-open-loop` | 内置 | `uv run wam run fake-open-loop` | 稳定的 CPU smoke 路径；不加载真实 WAM 权重。 |
| `fastwam-libero` | [![GitHub](https://img.shields.io/badge/GitHub-FastWAM-181717?logo=github)](https://github.com/yuantianyuan01/FastWAM) [![HF](https://img.shields.io/badge/HF-yuanty%2Ffastwam-FFD21E?logo=huggingface)](https://huggingface.co/yuanty/fastwam) | `uv run wam prepare fastwam-libero --download --asset eval` | 第一个真实 WAM 目标；native run/serve 路径已开始，LIBERO eval 路径可用。 |
| `cosmos-policy-libero` | [![GitHub](https://img.shields.io/badge/GitHub-cosmos--policy-181717?logo=github)](https://github.com/NVlabs/cosmos-policy) [![HF](https://img.shields.io/badge/HF-Cosmos--Policy--LIBERO-FFD21E?logo=huggingface)](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) | `uv run wam info cosmos-policy-libero` | native smoke 和官方脚本 parity 集成已开始。 |
| `dreamzero-droid-sim` | [![GitHub](https://img.shields.io/badge/GitHub-DreamZero-181717?logo=github)](https://github.com/dreamzero0/dreamzero) [![HF](https://img.shields.io/badge/HF-DreamZero--DROID-FFD21E?logo=huggingface)](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) [![HF gated](https://img.shields.io/badge/HF-gated_DROID_sim_assets-FFD21E?logo=huggingface)](https://huggingface.co/owhan/DROID-sim-environments) | `uv run wam info dreamzero-droid-sim` | resident policy-server 路径已开始；DROID sim 需要更重的多 GPU runtime。 |

## 常用命令

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

## 文档

- `docs/cli_entrypoints.md` - 命令行为。
- `docs/fastwam_libero_eval_setup.md` - FastWAM 环境和 eval 流程。
- `docs/dependency_isolation.md` - 容器和自管理环境。
- `docs/wamfile.md` - model entry schema。
- `docs/optimization_integration.md` - optimization profile 设计。
- `docs/trace_schema.md` - trace 事件 schema。
- `docs/roadmap.md` - 当前里程碑。

## License

EazyWAM 使用 [MIT License](LICENSE.md)。Vendored 第三方代码和外部模型资产仍遵循
各自上游许可证。
