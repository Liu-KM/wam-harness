# S4 复现结果：<论文短标题>

> 阶段：S4 复现 · 主导：Claude + Codex · 见 [../../PIPELINE.md](../../PIPELINE.md)
> 按 S3 的设计做复现。代码施工卡在 `../../tasks/<slug>-S4-NN-*.md`。

## 施工卡清单

| 卡号 | 目标 | 分支 | 验收 |
|---|---|---|---|
| `<slug>-S4-01-<slot>` | <一句话> | `codex/<slug>-s4-01` | ⬜ |

状态：⬜ 待办 / 🔄 进行中 / 🟡 验收中 / 🔁 返工 / ✅ 通过

## 实验配置

<最终跑实验用的 config（指向 configs/ 下文件）、backend、workload、seed 等。>

## 结果

| 指标 | baseline | variant | 差异 | 噪声底 |
|---|---|---|---|---|
| <primary> | | | | |
| <secondary> | | | | |

- **correctness_gate**：<通过 / 不通过；对齐方式与容差>
- **comparison_result.decision**：`useful` / `neutral` / `regression` / `not_comparable`
- **trace 路径**：<run 输出目录 / JSONL 路径>

## 结论

<复现是否成立；是否复现出源论文的收益；与源论文的差异。>

## Gate

- [ ] correctness_gate 通过且 decision 有定论，进入 S5。
