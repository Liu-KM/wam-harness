# S3 迁移设计：<论文短标题>

> 阶段：S3 迁移设计 · 主导：Claude · 见 [../../PIPELINE.md](../../PIPELINE.md)
> 把 idea 正式转化成一个 WAM baseline-vs-variant 实验设计。
> 契约：`docs/contract.md` 的 Systems Idea Experiment Contract。

## 实验描述（experiment descriptor）

```yaml
idea: <短名>
pressure: <S2 已确认存在的 WAM 侧 pressure>
existence_check: <S2 的结论依据，一句话>
method: <要在 WAM 上测试的机制>
assumptions:
  - <方法成立所依赖的条件>
baseline:
  <不带 idea 的设置>
variant:
  <带 idea 的设置>
metrics:
  primary: [<主指标>]
  secondary: [<副指标 + 副作用指标，如内存>]
correctness_gate: <baseline 与 variant 输出怎么对齐、怎么比；见 contract 的输出对齐小节>
decision_rule: <什么结果算 useful / neutral / regression>
# 计测协议（见 docs/measurement.md）
seed: <同一 seed，baseline 与 variant 共用>
warmup_iters: <丢弃的前导调用数>
repetitions: <每条测量路径的样本数>
runs: <独立进程运行数，估 run-to-run 噪声>
min_effect: <主指标低于此相对变化即判 neutral>
```

## 转化可行性

<WAM 的工作负载结构是否让该方法能落地？哪些地方和 LLM 不同、需要改造。>

## failure modes

- <这个迁移可能怎么失败（pressure 被别的阶段主导 / 显存反增 / 输出漂移 …）。>

## 结论

- **判定**：`transferable`（进入 S4）/ `not_transferable`（归档，记入 PROGRESS 归档区）。
- **依据**：<为什么。>

## Gate

- [ ] User 认可设计可执行、baseline 与 variant 对照可比，进入 S4。
