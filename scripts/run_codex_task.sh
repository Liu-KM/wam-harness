#!/usr/bin/env bash
# 标准化调用 Codex 执行一张 S4 施工卡（默认最强模型 + 最强推理）。
# 用法: scripts/run_codex_task.sh <task-id> [extra codex args...]
# 例:   scripts/run_codex_task.sh paged-attention-S4-01
#
# 约定 (见 .collab/WORKFLOW.md):
#   - 施工卡:    .collab/tasks/<task-id>.md 或 <task-id>-*.md
#   - 结构化回交: .collab/results/<task-id>.json        (受 result.schema.json 约束)
#   - 全量事件流: .collab/results/<task-id>.events.jsonl
#
# Codex 执行配置（默认最强，见 .collab/WORKFLOW.md）:
#   - model_reasoning_effort=xhigh   最高推理等级（钉死，不依赖全局 config 漂移）
#   - model=${CODEX_MODEL:-gpt-5.5}  默认最强模型，可用环境变量覆盖以便将来升级
set -euo pipefail

TASK_ID="${1:?需要任务号，如 paged-attention-S4-01}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 精确匹配优先，其次前缀匹配（文件名形如 <task-id>-<slot>.md）
TASK_FILE=".collab/tasks/${TASK_ID}.md"
if [[ ! -f "${TASK_FILE}" ]]; then
  TASK_FILE="$(ls .collab/tasks/${TASK_ID}-*.md 2>/dev/null | head -1 || true)"
fi
if [[ -z "${TASK_FILE}" || ! -f "${TASK_FILE}" ]]; then
  echo "找不到施工卡 .collab/tasks/${TASK_ID}(.md|-*.md)" >&2
  exit 1
fi

CODEX_MODEL="${CODEX_MODEL:-gpt-5.5}"
SCHEMA=".collab/schema/result.schema.json"
RESULT=".collab/results/${TASK_ID}.json"
EVENTS=".collab/results/${TASK_ID}.events.jsonl"
mkdir -p .collab/results

PREAMBLE="你是本仓库的实现方（Codex）。严格遵守 .collab/WORKFLOW.md 的护栏：
不得修改 Claude 定义的契约文档与接口签名（docs/contract.md / interfaces.md / measurement.md /
trace_schema.md / config_schema.md）；自行编写测试覆盖施工卡的验收清单；
完成后运行 'uv run ruff check .' 与 'uv run pytest' 并确保通过；
在分支 codex/<slug>-s4-NN 提交一个聚焦 commit，信息含论文 slug 与卡号。
你的最终回复必须符合给定的 output schema。下面是施工卡：

"

echo "▶ 派发施工卡 ${TASK_ID}: ${TASK_FILE}"
echo "  模型=${CODEX_MODEL}  推理=xhigh"
printf '%s' "${PREAMBLE}$(cat "${TASK_FILE}")" | codex exec - \
  -m "${CODEX_MODEL}" \
  -c model_reasoning_effort="xhigh" \
  -s workspace-write \
  -c approval_policy="never" \
  --output-schema "${SCHEMA}" \
  -o "${RESULT}" \
  --json "$@" | tee "${EVENTS}"

echo "✔ 结构化结果: ${RESULT}"
echo "✔ 事件流:     ${EVENTS}"
