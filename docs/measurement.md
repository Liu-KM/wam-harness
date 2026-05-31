# 计测协议（Measurement Protocol）

本文件定义 WAM Harness 如何采集时间与内存，以及如何判断 baseline 与 variant
之间的差异是否真实。它是所有延迟/内存数字的方法学真相来源。

契约文档（`docs/contract.md`）规定了**要记什么字段**；本文件规定**怎么记才有效**，
以及**怎么比才算数**。没有按本协议采集的数字，不得用于 decision rule。

## 为什么需要这份文档

measurement-first 的研究若没有计测协议，会犯两类经典错误：

1. **GPU 计时不同步** —— 用 wall-clock 包住一个异步 CUDA kernel，测到的是 launch
   返回时间而非 GPU 实际执行时间。数字看似有意义，实则无效。
2. **拿噪声当信号** —— 只跑一次、不丢 warmup、不看 run 间方差，就宣布 variant
   “快了 8%”。换一次运行结论可能反转。

本协议的目标是让这两类错误在契约层面无法发生。

## 计时协议

### 时钟选择

- 墙钟统一使用 `time.perf_counter()`（单调、高分辨率）。禁止 `time.time()`。
- 所有时间字段以毫秒（`*_ms`）记录，浮点。

### CPU 阶段计时

`preprocess_ms`、`postprocess_ms`、`env_step_ms` 等纯 CPU 阶段：直接用
`perf_counter()` 包裹该阶段即可。

### GPU 阶段计时（关键）

只要 `infer` 阶段在 CUDA 上执行，`model_ms` 必须满足以下之一，否则记为无效：

- **首选：CUDA events。** 在 kernel 前后插入 `torch.cuda.Event(enable_timing=True)`，
  以 `start.elapsed_time(end)` 取毫秒。这测的是 GPU 流上的真实执行时间。
- **次选：显式同步后用墙钟。** 调用 `torch.cuda.synchronize()` 后再读
  `perf_counter()`。同步本身的开销计入该阶段，需在文档中注明。

`cuda_ms` 字段专门存放 CUDA-event 测得的纯 GPU 时间；当采用 events 方案时，
`model_ms` 与 `cuda_ms` 可能一致或互为参照。CPU-only 运行 `cuda_ms` 记为 `null`。

### 阶段不重叠原则

`total_ms` 应约等于各阶段之和（容许调度抖动）。若某阶段把开销甩给另一个未计测的
阶段（例如把 GPU 同步成本藏进 env_step），按契约 invariant 视为**未成功**。

## 重复与采样

### 术语

- `warmup_iters`：丢弃的预热推理次数。用于排除首次调用效应（CUDA context、
  cache 填充、torch.compile/CUDA Graph 捕获、分配器升温）。
- `repetitions`（`n`）：warmup 之后用于统计的有效推理次数。
- `runs`：从进程启动到结束的整次独立执行。run 间方差衡量环境噪声。

### 默认取值（Phase 1 fake backend）

| 参数 | 默认 | 说明 |
|---|---|---|
| `warmup_iters` | 3 | 丢弃前 3 次 |
| `repetitions` | 30 | 每条路径至少 30 个有效样本 |
| `runs` | 3 | 同配置独立跑 3 次估 run 间方差 |

真实 backend 可按调用成本调整，但取值必须写进 run 的 runtime_info 并出现在 trace。

### warmup 必须可分离

warmup 阶段的样本要么不写入统计，要么打上 `warmup=true` 标记，使分析层能剔除。
绝不允许 warmup 样本混入 steady-state 统计。

## 统计判定（decision rule 的依据）

延迟分布是右偏的，所以汇总以分位数为主、均值为辅。

### 汇总量

每条被计测的路径（如 baseline 的 `model_ms`）至少报告：

- `p50`、`p95`（主）
- `mean`、`std`（辅）
- `min`、`max`、`n`（有效样本数，不含 warmup）

### baseline vs variant 的差异判定

只有同时满足以下条件，才允许 decision rule 宣布 variant “更快/更慢”：

1. **效应量**：主指标（如 `model_ms_p95`）的相对变化超过 `min_effect`（默认 5%）。
   小于该阈值视为“无差异（neutral）”，不论 p 值。
2. **超出噪声**：差值大于 run 间噪声底。噪声底由 `runs` 次独立运行的同指标标准差
   估计；变化量需大于 `2 × run_to_run_std` 才算可信。
3. **方向一致**：跨 `runs` 次运行，改善方向一致（不能一次快一次慢）。

任一条不满足，判定为 `neutral`，而非 `useful`。这条规则把“拿噪声当信号”挡在契约外。

### 噪声底（noise floor）

每个 run 配置应先跑一次 **A/A 对照**：相同设置当作 baseline 与 variant 各跑一遍。
真实差异应为零；A/A 测得的非零差异即该指标在当前机器上的噪声底。任何 A/B 改善
若小于其 A/A 噪声底，不得计为成功。

### 记入 trace

统计参数与判定输入必须出现在 trace，使结论可复算：

- run 级：`seed`、`warmup_iters`、`repetitions`、`runs`、`min_effect`。
- 比较级（`comparison_result` 事件，见 `docs/trace_schema.md`）：两侧的
  `p50/p95/mean/std/n`、相对变化、噪声底、最终 `decision`（`useful` /
  `neutral` / `regression` / `not_comparable`）。

## 内存计测

- CUDA 内存在每个 run 开始处用 `torch.cuda.reset_peak_memory_stats()` 归零，
  使 `*_peak_*` 字段的作用域限定在本 run。
- 采样点：run / episode / replan / inference / error 边界，写 `memory_sample` 事件。
- CPU-only 环境所有 CUDA 字段记 `0` 或 `null`，**字段名不变**（见 trace schema）。
- 内存判定与延迟对称：variant 若变快但 `cuda_peak_allocated_mb` 超过声明上限，
  按 invariant 视为未成功。

## 与契约的关系

- 本协议为 `docs/contract.md` 的 Measurement Contract 提供采集与判定方法。
- 新增字段 `seed`、`warmup_iters`、`repetitions`、`runs`、`min_effect` 见
  `docs/contract.md` 的实验描述子与 runtime_info。
- `comparison_result` 事件定义见 `docs/trace_schema.md`。
