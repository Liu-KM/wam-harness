# 配置 Schema（Config Schema）

实验通过 YAML 配置定义，不改实现代码（见 `docs/design.md`）。本文件定义 Phase 1
`fake_open_loop` 的配置结构。配置**选择** model/backend 与 workload，但不是模型能力的
真相来源（能力由 backend 上报，见 `AGENTS.md`）。

## 顶层结构

```yaml
# configs/fake_open_loop.yaml
run:
  name: fake-openloop-baseline      # run 标识，进 runtime_info.name
  output_dir: runs/fake_baseline    # trace 与输出 artifact 落盘根目录
  mode: experiment                  # experiment | characterization

backend:
  name: fake                        # registry 键；Phase 1 仅 fake
  device: cpu                       # cpu | cuda:0 ...
  dtype: fp32
  config: {}                        # backend 私有配置，核心不解释

workload:
  name: open_loop                   # registry 键
  episodes: 1                       # 开环冒烟默认 1
  steps_per_episode: 20
  config:
    observation_source: examples/open_loop/sample_obs.json

request:
  action_horizon: 8                 # 一次推理产出的动作步数
  replan_steps: 4                   # 每执行多少步重新规划一次
  num_inference_steps: null
  return_future: false
  return_value: false

measurement:                        # 见 docs/measurement.md
  seed: 0
  warmup_iters: 3
  repetitions: 30
  runs: 3
  min_effect: 0.05

experiment:                         # mode=experiment 时必填；characterization 时省略 variant
  idea: action_chunk_scheduling
  pressure: 替换 docs/systems_ideas.md 中对应 idea 的 pressure 摘要
  baseline:
    replan_steps: 4
  variant:
    replan_steps: 8
  metrics:
    primary: [model_calls_per_episode, total_model_ms_per_episode]
    secondary: [chunk_utilization, stale_action_steps]
  correctness_gate: exact          # exact | numeric_tolerance | success_rate | not_comparable
  decision_rule: 见 contract.md，由 measurement 协议判定 useful/neutral/regression
```

## 字段约束

| 字段 | 类型 | 约束 |
|---|---|---|
| `run.mode` | enum | `experiment` 或 `characterization` |
| `backend.name` | str | 必须在 registry.available() 中 |
| `request.action_horizon` | int | ≥ 1 |
| `request.replan_steps` | int | ≥ 1（与 action_horizon 的关系见 runner.md 边界表） |
| `measurement.seed` | int | baseline 与 variant 必须相同 |
| `experiment.correctness_gate` | enum | 见上 |

## 校验规则（实现时由 config loader 强制）

- `mode: experiment` 时 `experiment.baseline` 与 `experiment.variant` 必填；
  `mode: characterization` 时禁止出现 `variant`（见 contract.md 观测型实验）。
- `experiment` 的 baseline/variant 只能覆写白名单内的键（如 replan_steps、
  optimization_flags），不得改变 workload 形状或 seed，否则两 run 不可比。
- 未知顶层键报错（不静默忽略），避免配置漂移。

## 与 registry 的关系

`backend.name` / `workload.name` 是 registry 键。配置不描述模型能做什么，只声明
选哪个。能力信息由 backend 的 `load()`/`runtime_info()` 上报并写入 trace。
