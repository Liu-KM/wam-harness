#!/usr/bin/env bash
set -euo pipefail

model_id="${1:-${WAM_MODEL_ID:-dreamzero-droid-sim}}"
cache_dir="${WAM_CACHE_DIR:-/mnt/wam-cache}"
upstream_dir="${WAM_UPSTREAM_DIR:-/opt/dreamzero}"
trace_dir="${WAM_TRACE_DIR:-/mnt/runs}"

prepare_args=()
if [[ "${WAM_PREPARE_DOWNLOAD:-0}" == "1" ]]; then
  prepare_args+=(--download)
fi

prepare_status=0
wam prepare "${model_id}" --cache-dir "${cache_dir}" "${prepare_args[@]}" || prepare_status=$?
if [[ "${prepare_status}" != "0" ]]; then
  echo "wam prepare reported incomplete assets; running wam doctor for native readiness." >&2
fi
wam doctor "${model_id}" --cache-dir "${cache_dir}" --upstream-dir "${upstream_dir}" --json --strict
wam native-smoke "${model_id}" \
  --cache-dir "${cache_dir}" \
  --upstream-dir "${upstream_dir}" \
  --trace-dir "${trace_dir}" \
  --require-ready
