# papers/ — 每篇论文一条研究流水线

本目录下每个子目录 `<slug>/` 对应一个研究单元：把一篇 LLM systems 论文迁移 / 复现到 WAM。
流水线阶段定义见 [../PIPELINE.md](../PIPELINE.md)。

## 开一篇新论文

1. 选一个短 slug（kebab-case，能认出论文），如 `paged-attention`、`spec-decode`。
2. 建目录 `papers/<slug>/`，从 `papers/_TEMPLATE/` 复制需要的阶段文件。
3. 从 `S0-source.md` 开始，按 S0 → S5 逐阶段推进；每阶段经 User 认可再进下一阶段。
4. 在 [../PROGRESS.md](../PROGRESS.md) 看板加一行，标当前阶段与状态。

## 阶段产出物一览

| 文件 | 阶段 | 主导 | 内容 |
|---|---|---|---|
| `S0-source.md` | 录入 | User→Claude | 论文元信息、为什么挑它 |
| `S1-understanding.md` | 理解 | Claude | 动机 / 现象 / 问题 / 方法 / pressure |
| `S2-existence-check.md` | 存在性检查 | Claude | WAM 下 pressure 存不存在（结论 + 依据） |
| `S3-transfer.md` | 迁移设计 | Claude | baseline-vs-variant 实验设计 |
| `S4-results.md` | 复现 | Claude+Codex | 施工卡清单、实验结果、对照结论 |
| `S5-findings.md` | 新现象 | Claude+User | 新观察、可写 paper 的点 |

S4 的代码施工卡在 `../tasks/<slug>-S4-NN-*.md`，不放本目录。
