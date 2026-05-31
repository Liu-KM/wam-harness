# 任务卡 NNN：<标题>

> 给 Codex 的任务规格。Codex 只实现本卡内容，不得改动 Claude 定义的接口/契约。

## 上下文

- 阶段：Phase N
- 依赖任务：<NNN 或 无>
- 相关契约文档：<docs/xxx.md 的相关章节>

## 目标（一句话）

<这张卡要交付的东西>

## 接口契约（Claude 定义，不可改）

```python
# 粘贴 Claude 已确定的 Protocol 签名 / dataclass 字段。
# Codex 填实现体，不得改签名。若必须改，在 open_questions 申报。
```

## 实现要求

- <要点 1>
- <要点 2>

## 验收清单（acceptance checklist）

Codex 必须让以下每条成立，并**自行编写测试**覆盖它们：

- [ ] <可验证的行为 1>
- [ ] <可验证的行为 2>
- [ ] `uv run ruff check .` 通过
- [ ] `uv run pytest` 全绿
- [ ] 新增测试覆盖上述每条行为

## 禁止事项

- 不得修改 `docs/contract.md`、`docs/interfaces.md` 等契约文档。
- 不得引入未经批准的第三方依赖（重依赖走 optional extras，需先申报）。
- 不得直接提交到 main。

## 交付方式

- 在分支 `codex/phaseN-<slug>` 提交一个聚焦 commit。
- 回交结构化结果到 `.collab/results/NNN.json`（符合 `.collab/schema/result.schema.json`）。
