# Claude ↔ Codex 合作流

> 本文件定义**协作分工与执行护栏**。「我们在做什么」（研究流水线 S0–S5）见 [PIPELINE.md](PIPELINE.md)。
> 本文件是协作约定的唯一真相来源。

## 角色

- **User（决策方）**：喂源论文、认可每个阶段的产出、裁决方向分歧、决定是否立项写作。
- **Claude（研究 / 规划 / 验收方）**：
  - 主导研究阶段 **S1–S3**：读论文、写理解 / 存在性检查 / 迁移设计文档。
  - 主导施工阶段 **S4** 的拆解与验收：把复现拆成 Codex 施工卡、定义验收清单、验收回交。
  - 拥有契约文档与接口签名的所有权；维护 [PROGRESS.md](PROGRESS.md) 看板。
- **Codex（实现方）**：只在 **S4** 出场。执行施工卡、**自行编写测试**让验收清单通过，通过 `codex exec` 非交互调用。它需要明确、可机械执行的步骤。

## Codex 执行配置（默认最强）

Codex 默认使用**最强模型 + 最强推理等级**。由 `scripts/run_codex_task.sh` 钉死，不依赖全局 config 漂移：

- `model_reasoning_effort = "xhigh"`（最高推理等级，枚举值稳定，显式钉死）。
- `model`：默认 `gpt-5.5`（当前最强）；可用环境变量 `CODEX_MODEL` 覆盖，便于将来升级到更强模型而不改脚本。
- 非交互：`-s workspace-write`、`approval_policy="never"`、`--output-schema` 强约束回交格式。
- 沙箱行为（已实测）：`workspace-write` 下 `.git` 与默认 uv cache 只读。因此 **Codex 不碰 git**
  （提交由 Claude 验收后进行），跑 uv 命令时用 `UV_CACHE_DIR=/tmp/uv-cache`（脚本已预置）。

## 不可逾越的护栏

1. Codex **不得修改** Claude 定义的接口签名与契约文档（`docs/contract.md`、`docs/interfaces.md`、
   `docs/measurement.md`、`docs/trace_schema.md`、`docs/config_schema.md`）。需要改动时，必须在结构化
   回交的 `interface_deviations` / `open_questions` 中申报，由 Claude 裁决。
2. Codex 只在当前任务分支提交，**不得推送 main**。
3. **每个优化想法必须有 baseline、variant、correctness gate、trace 度量**，否则不算成功
   （沿用 `docs/contract.md` 的 invariants）。一个变快但破坏输出、或只是把开销挪到另一个未测量阶段的
   variant，不是成功。
4. 研究阶段（S1–S3）的结论必须可被 User 复核：理解要忠于论文，存在性检查的 `not_applicable` 要有依据。

## S4 单张施工卡的生命周期

施工卡只在复现阶段（S4）产生，每张卡绑定一篇论文。

```
① Claude 写施工卡 + 建卡分支 → .collab/tasks/<slug>-S4-NN-<slot>.md；git checkout -b codex/<slug>-s4-NN
② Claude 调 codex exec       → scripts/run_codex_task.sh <slug>-S4-NN
③ Codex 改工作树 + 回交       → 改源码/测试（不碰 git）；结构化回交
                              .collab/results/<id>.json (符合 schema) + <id>.events.jsonl
④ Claude 验收（门禁）：
     uv run ruff check .        必须通过
     uv run pytest              必须全绿
     git diff 审接口符合度       不得越界改契约
     审 Codex 自写测试的覆盖度    覆盖施工卡的验收清单
     (可选) codex exec review   独立第二眼
⑤ 判定：
     通过 → Claude 提交聚焦 commit（信息含 slug 与卡号）；TaskUpdate 标完成；
            更新 PROGRESS.md；进入下一张卡 / 下一阶段
     不过 → 写返工说明，codex exec resume --last 带上下文重做
            （最多 3 轮；超出则 Claude 接管或上报 User）
```

## 验收门禁（本仓库已定）

- 测试由 **Codex 编写**；Claude 审查测试是否覆盖施工卡的验收清单。
- Claude 提供接口签名 + 验收清单，不预先写测试。
- 客观门禁：`ruff` 干净 + `pytest` 全绿 + diff 不越界 + 测试覆盖到位。

## git 约定

- 每篇论文的复现在独立分支：`codex/<slug>-s4`（或按卡 `codex/<slug>-s4-NN`），由 Claude 创建。
- **Codex 不碰 git**（沙箱内 .git 只读）：它只改工作树并结构化回交。
- 验收通过后由 **Claude 提交**一个聚焦 commit（信息含 slug 与卡号），审 diff 后合回主线；不直接推 main。
- 研究文档（S0–S3、S5 的 markdown）由 Claude 直接提交到工作分支。

## 目录约定

```
.collab/
  PIPELINE.md              研究流水线 S0–S5（做什么）
  WORKFLOW.md              本协议（怎么协作执行）
  PROGRESS.md              进度看板（每篇论文走到哪个阶段，Claude 维护）
  papers/                  每篇论文一个目录（研究产出物）
    _TEMPLATE/             阶段产出物模板 S0–S5
    <slug>/                某篇论文的 S0-source.md / S1-understanding.md / ...
  tasks/                   S4 施工卡 <slug>-S4-NN-<slot>.md（绑定论文）
    _TEMPLATE.md           施工卡模板
  schema/result.schema.json  Codex 结构化回交 schema（strict）
  results/                 Codex 回交结果 + 事件流
  reviews/                 Claude 验收记录 + codex review 输出
```
