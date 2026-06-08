#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_core_env.sh [options]

Create a lightweight EazyWAM core environment with uv and install the local
source checkout.

Options:
  --venv PATH       Virtual environment path. Default: .venv
  --python VERSION  Python version for uv venv. Default: 3.10
  --regular         Install the checkout as a regular local package.
  --editable        Install the checkout in editable mode. Default.
  -h, --help        Show this help message.

After setup:
  source <venv>/bin/activate
  wam list
EOF
}

venv_dir="${EAZYWAM_VENV:-.venv}"
python_version="${EAZYWAM_PYTHON:-3.10}"
editable=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      venv_dir="${2:-}"
      shift 2
      ;;
    --python)
      python_version="${2:-}"
      shift 2
      ;;
    --regular)
      editable=0
      shift
      ;;
    --editable)
      editable=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$venv_dir" ]] || {
  echo "error: --venv must not be empty" >&2
  exit 2
}

[[ -n "$python_version" ]] || {
  echo "error: --python must not be empty" >&2
  exit 2
}

command -v uv >/dev/null 2>&1 || {
  echo "error: uv is required. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv venv --python "$python_version" --allow-existing "$venv_dir"

python_bin="$venv_dir/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "error: expected Python executable not found: $python_bin" >&2
  exit 1
fi

if [[ "$editable" -eq 1 ]]; then
  uv pip install --python "$python_bin" -e "$repo_root"
else
  uv pip install --python "$python_bin" "$repo_root"
fi

cat <<EOF

EazyWAM core environment is ready.

Activate it with:
  source $venv_dir/bin/activate

Then run:
  wam list
EOF
