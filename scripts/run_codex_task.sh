#!/usr/bin/env bash
# 标准化调用 Codex 执行一张任务卡。
# 用法: scripts/run_codex_task.sh <NNN> [extra codex args...]
# 例:   scripts/run_codex_task.sh 001
#
# 约定:
#   - 任务卡:    .collab/tasks/<NNN>-*.md
#   - 结构化回交: .collab/results/<NNN>.json   (受 result.schema.json 约束)
#   - 全量事件流: .collab/results/<NNN>.events.jsonl
set -euo pipefail

NNN="${1:?需要任务号，如 001}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK_FILE="$(ls .collab/tasks/${NNN}-*.md 2>/dev/null | head -1 || true)"
if [[ -z "${TASK_FILE}" ]]; then
  echo "找不到任务卡 .collab/tasks/${NNN}-*.md" >&2
  exit 1
fi

SCHEMA=".collab/schema/result.schema.json"
RESULT=".collab/results/${NNN}.json"
EVENTS=".collab/results/${NNN}.events.jsonl"

PREAMBLE="你是本仓库的实现方。严格遵守 .collab/WORKFLOW.md 的护栏：
不得修改 Claude 定义的契约文档与接口签名；自行编写测试覆盖任务卡的验收清单；
完成后运行 'uv run ruff check .' 与 'uv run pytest' 并确保通过；
在分支 codex/<阶段>-<slug> 提交一个聚焦 commit。
你的最终回复必须符合给定的 output schema。下面是任务卡：

"

echo "▶ 派发任务 ${NNN}: ${TASK_FILE}"
printf '%s' "${PREAMBLE}$(cat "${TASK_FILE}")" | codex exec - \
  --output-schema "${SCHEMA}" \
  -o "${RESULT}" \
  --json "$@" | tee "${EVENTS}"

echo "✔ 结构化结果: ${RESULT}"
echo "✔ 事件流:     ${EVENTS}"
