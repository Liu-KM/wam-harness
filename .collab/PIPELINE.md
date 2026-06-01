# 研究流水线：把一篇 LLM systems 工作复现到 WAM

> 本文件定义本仓库的**研究工作单元**与推进流程。它是「我们到底在做什么」的唯一真相来源。
> 协作分工与执行护栏见 [WORKFLOW.md](WORKFLOW.md)；当前进度见 [PROGRESS.md](PROGRESS.md)。

## 工作单元

本仓库的一个研究单元 = **把一篇 LLM 推理系统（systems）的工作，迁移 / 复现到 WAM（world-action model）推理上**。

不是「实现某个 harness 模块」。harness 只是支撑复现的地基；它按某篇论文的复现需要被拉起，而不是预先规划一堆脱离论文的施工卡。

每个研究单元对应 `papers/<slug>/` 下的一组阶段产出物，在 [PROGRESS.md](PROGRESS.md) 看板上占一行。

## 一篇论文的生命周期：S0 → S5

```
S0 录入        登记源论文（User 喂论文）
   │
S1 理解        读懂源工作：动机 / 现象 / 要解决的问题 / 方法 / 它针对的 pressure
   │
S2 存在性检查   这个 pressure 在 WAM 推理里到底存不存在？（用数据回答）
   │           ├─ 不存在 → 记为 not_applicable，归档（这是有价值的研究结果，流水线在此终止）
   │           └─ 存在 / 需复现才能判定 → 继续
S3 迁移设计     idea 怎么转化成 WAM？写成 baseline-vs-variant 实验设计
   │           └─ 转化不成立 → 记为 not_transferable，归档
S4 复现        按设计做复现：拆 Codex 施工卡 → 跑实验 → 出 trace 与对照结果
   │
S5 新现象      复现中抓到的新现象 → 尝试写成 paper
```

**主导方**：S0 由 User 触发；**S1–S3 是研究 / 设计工作，由 Claude 主导**，产出是 markdown 文档，给 User 认可；**S4 才是施工**，由 Claude 拆成 Codex 施工卡、Codex 执行、Claude 验收；S5 由 Claude + User 共同判断。

这条分界线是本次重构的核心：**「卡片」只在 S4 出现，且每张卡都明确属于某篇论文的某个阶段**。S1–S3 不产生 Codex 卡，只产生研究文档。

## 各阶段产出物与推进门禁

每阶段产出物落在 `papers/<slug>/` 下，模板见 `papers/_TEMPLATE/`。**上一阶段产出物经 User 认可，才进下一阶段**（gate）。

### S0 录入 → `papers/<slug>/S0-source.md`

- 论文标题、作者、链接、本地 PDF 路径（User 提供）。
- 一句话：它是干什么的、为什么挑它。
- **Gate**：User 确认 slug 与论文无误。

### S1 理解 → `papers/<slug>/S1-understanding.md`

读懂**源工作本身**，先不谈 WAM：

- **动机**：作者为什么做这件事。
- **现象**：它观察到的关键现象 / 测量。
- **问题**：它要解决的具体问题。
- **方法**：它怎么解决的（机制层面）。
- **pressure**：在 LLM 推理里，它针对的底层系统压力是什么（launch 开销 / 显存 / 重复计算 / 带宽 / 调度…）。
- **收益与前提**：它声称的收益，以及成立所依赖的假设。

**Gate**：User 认可「这篇论文被读懂了」。

### S2 存在性检查 → `papers/<slug>/S2-existence-check.md`

回答先决问题：**S1 里那个 pressure，在 WAM 推理工作负载里到底存不存在？**

- WAM 侧的候选对应物（哪个环节可能有同样的 pressure）。
- 可观测信号：用什么测量能判定它存在（接 `docs/contract.md` 的 **Characterization Experiment Contract**——无 variant 的画像实验）。
- 若需要画像实验来取证，这里写出该实验的设计；实验本身在 S4 用 harness 跑（早期可用 fake / open-loop 画像）。
- **结论三选一**：
  - `exists` — pressure 存在，继续 S3。
  - `not_applicable` — pressure 不存在 / 在 WAM 下不构成瓶颈。**归档终止**，这是有价值的研究结果。
  - `needs_reproduction` — 现有信息不足，需要先搭最小复现才能判定，谨慎进 S3/S4。

**Gate**：User 认可结论。`not_applicable` 走归档，不进 S3。

### S3 迁移设计 → `papers/<slug>/S3-transfer.md`

如果 pressure 存在，把 idea 正式转化成一个 WAM 实验设计（接 `docs/contract.md` 的 **Systems Idea Experiment Contract**）：

- `idea / pressure / existence_check / method / assumptions`
- `baseline`（不带 idea）/ `variant`（带 idea）
- `metrics`（primary / secondary）
- `correctness_gate`（baseline 与 variant 的输出怎么对齐、怎么比）
- `decision_rule`（什么结果算 useful / neutral / regression）
- `seed / warmup_iters / repetitions / runs / min_effect`（计测协议，见 `docs/measurement.md`）

若转化不成立（WAM 结构让方法无法落地），记为 `not_transferable` 并归档。

**Gate**：User 认可设计可执行、对照可比。

### S4 复现 → 代码 + `papers/<slug>/S4-results.md`

按 S3 的设计做复现。这是**唯一产生 Codex 施工卡**的阶段：

- Claude 把复现拆成 `tasks/<slug>-S4-NN-<slot>.md` 施工卡（绑定论文 + 阶段）。
- Codex 执行（最强模型 + 最强推理，见 [WORKFLOW.md](WORKFLOW.md)）、自写测试、回交结构化结果。
- Claude 验收（ruff / pytest / diff 不越界 / 测试覆盖）。
- 跑 baseline 与 variant，产出 trace 与 `comparison_result`，汇总到 `S4-results.md`。

**Gate**：correctness_gate 通过 + decision 有定论（useful / neutral / regression）。

### S5 新现象 → `papers/<slug>/S5-findings.md`

复现过程中几乎一定会冒出源论文没说的现象。这里把它们沉淀成可能的 paper：

- 新观察 / 反直觉结果 / WAM 特有的边界。
- 是否超出源论文的贡献。
- 能写成 paper 的点 + 还需要补的实验。

**Gate**：User 判断是否立项写作 / 加做实验。

## 与现有 docs 的关系

- `docs/contract.md`：实验契约。S2 用其 **Characterization** 契约，S3/S4 用其 **Systems Idea** 契约。
- `docs/systems_ideas.md`：通用 idea 储备池（CUDA Graph、torch.compile、cache 复用…）。可作为 S1–S3 的参考范例，但真正的研究单元由 User 喂的论文驱动，逐篇走 S0–S5。
- `docs/measurement.md`：S2/S3/S4 的计测协议（n / warmup / 方差 / 噪声底 / 显著性）。
- `docs/phase1_plan.md` / `docs/roadmap.md`：harness 地基的能力路线（fake → 真实 backend → 远程 → 模拟器）。地基能力按论文复现的需要被拉起。

## 状态记号

每篇论文在 [PROGRESS.md](PROGRESS.md) 标当前阶段与状态：

`S0` 录入 / `S1` 理解 / `S2` 存在性检查 / `S3` 迁移设计 / `S4` 复现 / `S5` 新现象
状态：⬜ 待办 / 🔄 进行中 / 🟡 待 User 认可 / ✅ 通过本阶段 / ⛔ 归档（not_applicable / not_transferable）
