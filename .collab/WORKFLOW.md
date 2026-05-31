# Claude ↔ Codex 合作流

本文件定义本仓库的人机协作分工与流程。它是协作约定的唯一真相来源。

## 角色

- **Claude（规划/验收方）**：负责文档、契约、上层接口（Protocol 签名、类型）、
  任务卡、验收门禁与进度管理。拥有契约的所有权。
- **Codex（实现方）**：负责具体实现，自行编写测试，让验收清单通过。
  通过 `codex exec` 非交互调用。
- **User（决策方）**：批准阶段推进、裁决方向分歧。

## 不可逾越的护栏

1. Codex **不得修改** Claude 定义的接口签名与契约文档（`docs/contract.md`、
   `docs/interfaces.md`、`docs/measurement.md`、`docs/trace_schema.md`、
   `docs/config_schema.md`）。需要改动时，必须在结构化回交的
   `interface_deviations` / `open_questions` 中申报，由 Claude 裁决。
2. Codex 只在当前任务分支提交，**不得推送 main**。
3. 每个最优化想法必须有 baseline、variant、correctness gate、trace 度量，
   否则不算成功（沿用 `docs/contract.md` 的 invariants）。

## 单个任务的生命周期

```
① Claude 写任务卡          → .collab/tasks/NNN-<slug>.md
② Claude 调 codex exec     → scripts/run_codex_task.sh NNN
③ Codex 回交结构化结果      → .collab/results/NNN.json (符合 result.schema.json)
                             + .collab/results/NNN.events.jsonl (全量事件流)
④ Claude 验收（门禁）：
     uv run ruff check .        必须通过
     uv run pytest              必须全绿
     git diff 审接口符合度       不得越界改契约
     审 Codex 自写测试的覆盖度    覆盖任务卡的验收清单
     (可选) codex review        独立第二眼
⑤ 判定：
     通过 → TaskUpdate 标完成；更新 PROGRESS.md；进入下一任务
     不过 → 写返工说明，codex exec resume --last 带上下文重做
            （最多 3 轮；超出则 Claude 接管或上报 User）
```

## 验收门禁（本仓库已定）

- 测试由 **Codex 编写**；Claude 审查测试是否覆盖任务卡的验收清单。
- Claude 提供接口签名 + 验收清单（acceptance checklist），不预先写测试。
- 客观门禁：`ruff` 干净 + `pytest` 全绿 + diff 不越界 + 测试覆盖到位。

## git 约定

- 每个阶段任务在独立分支：`codex/phaseN-<slug>`。
- Codex 完成任务卡后提交一个聚焦 commit，信息含任务号。
- Claude 审 diff 通过后合回主线；不直接推 main。

## 目录约定

```
.collab/
  WORKFLOW.md              本协议
  PROGRESS.md              进度看板（Claude 维护）
  tasks/                   任务卡 NNN-<slug>.md
  schema/result.schema.json  Codex 结构化回交 schema
  results/                 Codex 回交结果 + 事件流
  reviews/                 Claude 验收记录 + codex review 输出
```
