# 进度看板

> 由 Claude 维护。每张任务卡的状态在此一目了然。

## 当前阶段

阶段 A：补契约文档 + 搭合作脚手架（纯 docs，不写 src）。

## 阶段 A：契约补强（Claude 自己做，无需 Codex）

| 文档 | 状态 | 说明 |
|---|---|---|
| `.collab/` 脚手架 | ✅ 完成 | WORKFLOW / schema / 模板 / 看板 / 脚本 |
| `docs/measurement.md` | ✅ 完成 | 计测协议、n/warmup/方差/噪声底/显著性 |
| `docs/interfaces.md` | ✅ 完成 | backend/processor/registry/observer Protocol 签名 |
| `docs/contract.md` 补全 | ✅ 完成 | seed/repetitions/warmup、观测型实验、artifact 对齐 |
| `docs/trace_schema.md` 补全 | ✅ 完成 | comparison_result 事件、artifact 路径、观测字段 |
| `docs/config_schema.md` | ✅ 完成 | fake_open_loop 配置 schema |
| `docs/runner.md` | ✅ 完成 | runner 循环伪代码 + 边界表 |

## 阶段 B：Phase 1 实现（Codex 做，待 User 批准后开）

| # | 任务卡 | 分支 | 状态 | 验收 |
|---|---|---|---|---|
| 001 | 核心契约类型 | codex/phase1-types | ⬜ | — |
| 002 | JSONL trace writer | codex/phase1-trace | ⬜ | — |
| 003 | memory observer | codex/phase1-observer | ⬜ | — |
| 004 | fake backend | codex/phase1-fake | ⬜ | — |
| 005 | open-loop workload | codex/phase1-openloop | ⬜ | — |
| 006 | runner + 调度 | codex/phase1-runner | ⬜ | — |
| 007 | 最小 CLI `wam run` | codex/phase1-cli | ⬜ | — |
| 008 | `wam compare` | codex/phase1-compare | ⬜ | — |

状态图例：⬜ 待办 / 🔄 进行中 / 🟡 验收中 / 🔁 返工 / ✅ 通过
