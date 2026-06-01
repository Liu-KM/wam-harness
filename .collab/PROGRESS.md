# 进度看板

> 由 Claude 维护。每篇论文走到哪个阶段，在此一目了然。
> 流水线定义见 [PIPELINE.md](PIPELINE.md)，协作流见 [WORKFLOW.md](WORKFLOW.md)。

## 当前状态

研究流水线框架已搭好，**等待 User 喂第一篇 LLM systems 论文**（触发 S0）。

## 论文看板

| slug | 论文 | 当前阶段 | 状态 | 备注 |
|---|---|---|---|---|
| — | （暂无） | — | — | 等待 User 喂第一篇论文 |

阶段：S0 录入 / S1 理解 / S2 存在性检查 / S3 迁移设计 / S4 复现 / S5 新现象
状态：⬜ 待办 / 🔄 进行中 / 🟡 待 User 认可 / ✅ 通过本阶段 / ⛔ 归档(not_applicable / not_transferable)

## harness 地基状态

地基是支撑复现（S4）的最小系统，按论文需要被拉起。当前为**契约就绪、实现待建**：

| 能力 | 状态 | 说明 |
|---|---|---|
| 实验契约文档 | ✅ 就绪 | `docs/contract.md` / `measurement.md` / `interfaces.md` / `trace_schema.md` / `config_schema.md` |
| 研究流水线脚手架 | ✅ 就绪 | `.collab/` PIPELINE / WORKFLOW / 看板 / 模板 / schema / 派发脚本 |
| 核心契约类型（src） | ⬜ 待建 | 回退清空，待第一篇论文走到 S4 时按需拉起（实现存档于 `archive/phase1-merged` tag 与 `codex/phase1-*` 分支） |
| JSONL trace writer | ⬜ 待建 | 同上，存档可复用 |
| fake backend / open-loop runner | ⬜ 待建 | S4 复现的最小支撑 |
| memory / timing observer | ⬜ 待建 | 计测协议支撑 |
| CLI `wam run` / `wam compare` | ⬜ 待建 | 实验入口 |

> 说明：早前的 001/002 实现（核心类型、trace writer）已从主线回退，但完整保留在
> `archive/phase1-merged` tag 与 `codex/phase1-types` / `codex/phase1-trace` 分支。
> 当某篇论文的 S4 复现需要这些地基时，作为施工卡的参考实现接回，而非预先铺开。

## 归档区（not_applicable / not_transferable）

> pressure 不存在或无法迁移的论文记录在此。这些是有价值的负结果。

（暂无）
