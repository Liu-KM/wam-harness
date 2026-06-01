# 施工卡 <slug>-S4-NN：<标题>

> 给 Codex 的 **S4 复现**任务规格。每张卡绑定一篇论文。
> Codex 只实现本卡内容，不得改动 Claude 定义的接口 / 契约。见 [../WORKFLOW.md](../WORKFLOW.md)。

## 归属

- **论文**：`<slug>`（见 `../papers/<slug>/`）
- **阶段**：S4 复现
- **本卡号**：`<slug>-S4-NN`
- **依赖卡**：`<其它卡号 或 无>`

## 上下文

- **迁移设计**：`../papers/<slug>/S3-transfer.md` 的相关小节。
- **相关契约文档**：`docs/<...>.md` 的相关章节。

## 目标（一句话）

<这张卡要交付的东西。>

## 接口契约（Claude 定义，不可改）

```python
# 粘贴 Claude 已确定的 Protocol 签名 / dataclass 字段。
# Codex 填实现体，不得改签名。若必须改，在回交的 interface_deviations / open_questions 申报。
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

- 不得修改 `docs/contract.md`、`docs/interfaces.md`、`docs/measurement.md`、
  `docs/trace_schema.md`、`docs/config_schema.md` 等契约文档。
- 不得引入未经批准的第三方依赖（重依赖走 optional extras，需先申报）。
- 不得直接提交到 main。

## 交付方式

- 只修改工作树文件，**不要执行 git 操作**（.git 在沙箱内只读；提交由 Claude 验收后进行）。
- 跑 uv 命令时加前缀 `UV_CACHE_DIR=/tmp/uv-cache`（沙箱默认 cache 只读）。
- 最终回复按 output schema 结构化回交（脚本保存到 `.collab/results/<slug>-S4-NN.json`）。
