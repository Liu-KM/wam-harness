#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/fastwam-libero-eval.sh [options]

Purpose:
  Run the portable FastWAM LIBERO native-eval acceptance path. This script is
  intentionally scheduler-agnostic: run it inside any prepared GPU container,
  activated self-managed environment, or interactive GPU shell.

Options:
  --model-id ID          Model id. Default: fastwam-libero
  --cache-dir PATH       WAM cache path. Default: ${WAM_CACHE_DIR:-~/.cache/wam}
  --trace-dir PATH       Run trace directory. Default: ${WAM_TRACE_DIR:-runs}
  --workload NAME        Eval workload. Default: libero-single-task
  --task-id ID           LIBERO task id. Default: 0
  --num-trials N         Number of trials. Default: 1
  --min-success-rate R   Minimum success rate for acceptance. Default: 1.0
  --num-steps-wait N     No-op simulator warmup steps. Default: 5
  --max-steps N          Optional max simulator steps override.
  --libero-dir PATH      Prepared LIBERO checkout. Default:
                         ${WAM_LIBERO_DIR}, <cache>/upstreams/LIBERO, or /opt/LIBERO.
  --set KEY=VALUE        Extra eval override passed through to `wam eval`.
  --download-assets      Let `wam prepare` download the declared eval asset set.
  --skip-prepare         Skip `wam prepare`.
  --skip-smoke           Skip `wam native-smoke`.
  --skip-simulator-check Skip LIBERO simulator env/observation preflight.
  --mujoco-gl VALUE      MUJOCO_GL default. Default: osmesa
  --pyopengl-platform V  PYOPENGL_PLATFORM default. Default: osmesa
  --help                 Show this help.

Expected acceptance command:
  scripts/fastwam-libero-eval.sh \
    --cache-dir /path/to/wam-cache \
    --trace-dir /path/to/runs \
    --download-assets

Re-validate a saved eval summary:
  python -m eazywam.evals.acceptance --json \
    /path/to/runs/fastwam-libero-libero-single-task-eval-summary.json \
    1 \
    1.0
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

absolute_path() {
  python - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

model_id="${WAM_MODEL_ID:-fastwam-libero}"
cache_dir="${WAM_CACHE_DIR:-$HOME/.cache/wam}"
trace_dir="${WAM_TRACE_DIR:-runs}"
workload="libero-single-task"
task_id="0"
num_trials="1"
min_success_rate="${WAM_ACCEPT_MIN_SUCCESS_RATE:-1.0}"
num_steps_wait="5"
max_steps=""
libero_dir="${WAM_LIBERO_DIR:-}"
download_assets="${WAM_PREPARE_DOWNLOAD:-0}"
skip_prepare=0
skip_smoke=0
skip_simulator_check=0
mujoco_gl="${MUJOCO_GL:-osmesa}"
pyopengl_platform="${PYOPENGL_PLATFORM:-osmesa}"
extra_sets=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id)
      model_id="${2:-}"
      shift 2
      ;;
    --cache-dir)
      cache_dir="${2:-}"
      shift 2
      ;;
    --trace-dir)
      trace_dir="${2:-}"
      shift 2
      ;;
    --workload)
      workload="${2:-}"
      shift 2
      ;;
    --task-id)
      task_id="${2:-}"
      shift 2
      ;;
    --num-trials)
      num_trials="${2:-}"
      shift 2
      ;;
    --min-success-rate)
      min_success_rate="${2:-}"
      shift 2
      ;;
    --num-steps-wait)
      num_steps_wait="${2:-}"
      shift 2
      ;;
    --max-steps)
      max_steps="${2:-}"
      shift 2
      ;;
    --libero-dir)
      libero_dir="${2:-}"
      shift 2
      ;;
    --set)
      extra_sets+=("${2:-}")
      shift 2
      ;;
    --download-assets)
      download_assets=1
      shift
      ;;
    --skip-prepare)
      skip_prepare=1
      shift
      ;;
    --skip-smoke)
      skip_smoke=1
      shift
      ;;
    --skip-simulator-check)
      skip_simulator_check=1
      shift
      ;;
    --mujoco-gl)
      mujoco_gl="${2:-}"
      shift 2
      ;;
    --pyopengl-platform)
      pyopengl_platform="${2:-}"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "$model_id" ]] || die "--model-id must not be empty"
[[ -n "$cache_dir" ]] || die "--cache-dir must not be empty"
[[ -n "$trace_dir" ]] || die "--trace-dir must not be empty"
[[ -n "$workload" ]] || die "--workload must not be empty"
[[ -n "$task_id" ]] || die "--task-id must not be empty"
[[ -n "$num_trials" ]] || die "--num-trials must not be empty"
[[ -n "$min_success_rate" ]] || die "--min-success-rate must not be empty"
[[ -n "$num_steps_wait" ]] || die "--num-steps-wait must not be empty"

command -v wam >/dev/null 2>&1 || die "wam is not on PATH. Activate the backend environment first."

cache_dir="$(absolute_path "$cache_dir")"
trace_dir="$(absolute_path "$trace_dir")"
if [[ -n "$libero_dir" ]]; then
  libero_dir="$(absolute_path "$libero_dir")"
elif [[ -d "$cache_dir/upstreams/LIBERO/libero/libero" ]]; then
  libero_dir="$cache_dir/upstreams/LIBERO"
elif [[ -d /opt/LIBERO/libero/libero ]]; then
  libero_dir="/opt/LIBERO"
fi

export WAM_CACHE_DIR="$cache_dir"
default_libero_config_path="$cache_dir/libero/config"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$default_libero_config_path}"
if [[ -n "$libero_dir" && -d "$libero_dir" ]]; then
  export WAM_LIBERO_DIR="$libero_dir"
  export PYTHONPATH="$libero_dir:$libero_dir/libero:${PYTHONPATH:-}"
  if [[ "$LIBERO_CONFIG_PATH" == "$default_libero_config_path" && -d "$libero_dir/libero/libero" ]]; then
    mkdir -p "$LIBERO_CONFIG_PATH" "$cache_dir/libero/datasets"
    cat >"$LIBERO_CONFIG_PATH/config.yaml" <<EOF
assets: $libero_dir/libero/libero/assets
bddl_files: $libero_dir/libero/libero/bddl_files
benchmark_root: $libero_dir/libero/libero
datasets: $cache_dir/libero/datasets
init_states: $libero_dir/libero/libero/init_files
EOF
  fi
fi
export MUJOCO_GL="$mujoco_gl"
export PYOPENGL_PLATFORM="$pyopengl_platform"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$cache_dir" "$trace_dir"

echo "FastWAM LIBERO native eval acceptance"
echo "  model_id: $model_id"
echo "  workload: $workload"
echo "  task_id: $task_id"
echo "  num_trials: $num_trials"
echo "  min_success_rate: $min_success_rate"
echo "  cache_dir: $cache_dir"
echo "  trace_dir: $trace_dir"
echo "  WAM_LIBERO_DIR: ${WAM_LIBERO_DIR:-not set}"
echo "  LIBERO_CONFIG_PATH: $LIBERO_CONFIG_PATH"
echo "  MUJOCO_GL: $MUJOCO_GL"
echo "  PYOPENGL_PLATFORM: $PYOPENGL_PLATFORM"
echo "  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD: $TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"

if [[ "$skip_prepare" != "1" ]]; then
  prepare_args=(prepare "$model_id" --cache-dir "$cache_dir" --asset eval)
  if [[ "$download_assets" == "1" ]]; then
    prepare_args+=(--download)
  fi
  run_cmd wam "${prepare_args[@]}"
fi

run_cmd wam doctor "$model_id" --cache-dir "$cache_dir" --json --strict

if [[ "$skip_simulator_check" != "1" ]]; then
  run_cmd python - "$model_id" "$workload" "$task_id" "$num_trials" <<'PY'
import sys

from eazywam.core.manifest import load_builtin_manifest
from eazywam.evals.libero import (
    _create_libero_env,
    _import_libero_modules,
    _load_task,
    _observation_from_libero,
)

model_id, workload, task_id, num_trials = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
manifest = load_builtin_manifest(model_id)
eval_config = manifest.eval.get("workloads", {}).get(workload, {})
defaults = dict(manifest.eval.get("defaults", {}))
if isinstance(eval_config, dict):
    defaults.update(eval_config.get("defaults", {}))
task_suite_name = str(defaults.get("task_suite_name", manifest.eval.get("suite", "libero_10")))
modules = _import_libero_modules()
_, task, states = _load_task(modules, task_suite_name, task_id, num_trials)
env = None
try:
    env, task_description = _create_libero_env(modules, task, seed=0)
    env.reset()
    obs = env.set_init_state(states[0])
    observation = _observation_from_libero(
        obs,
        task_description=task_description,
        episode_idx=0,
        step_id=0,
        task_suite_name=task_suite_name,
        task_id=task_id,
    )
finally:
    if env is not None:
        close = getattr(env, "close", None)
        if callable(close):
            close()
print(
    "LIBERO simulator preflight: "
    f"suite={task_suite_name} task_id={task_id} "
    f"task={task.language!r} init_states={len(states)} "
    f"images={sorted(observation.images)} state={sorted(observation.state)}"
)
PY
fi

if [[ "$skip_smoke" != "1" ]]; then
  run_cmd wam native-smoke "$model_id" \
    --cache-dir "$cache_dir" \
    --trace-dir "$trace_dir" \
    --require-ready
fi

eval_args=(
  eval "$model_id"
  --workload "$workload"
  --task-id "$task_id"
  --num-trials "$num_trials"
  --cache-dir "$cache_dir"
  --trace-dir "$trace_dir"
  --set "num_steps_wait=$num_steps_wait"
  --set "mujoco_gl=$mujoco_gl"
  --set "pyopengl_platform=$pyopengl_platform"
)
if [[ -n "$max_steps" ]]; then
  eval_args+=(--set "max_steps=$max_steps")
fi
for item in "${extra_sets[@]}"; do
  [[ -n "$item" ]] || die "--set value must not be empty"
  eval_args+=(--set "$item")
done

eval_summary_path="$trace_dir/${model_id}-${workload}-eval-summary.json"
eval_raw_output_path="$trace_dir/${model_id}-${workload}-eval-output.txt"
acceptance_report_path="$trace_dir/${model_id}-${workload}-acceptance.json"
eval_args+=(--summary-path "$eval_summary_path")
print_cmd wam "${eval_args[@]}"
wam "${eval_args[@]}" | tee "$eval_raw_output_path"
cat "$eval_summary_path"
print_cmd python -m eazywam.evals.acceptance --json \
  "$eval_summary_path" \
  "$num_trials" \
  "$min_success_rate"
python -m eazywam.evals.acceptance --json \
  "$eval_summary_path" \
  "$num_trials" \
  "$min_success_rate" | tee "$acceptance_report_path"
