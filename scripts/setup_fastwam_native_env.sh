#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_fastwam_native_env.sh [options]

Purpose:
  Build a self-managed FastWAM native runtime environment without Docker,
  Apptainer, Slurm, or site-specific launchers. FastWAM runtime code is
  provided by WAM Harness; an upstream FastWAM checkout is only needed for
  explicit reference-eval parity checks.

Options:
  --upstream-dir PATH       Optional FastWAM checkout path for reference eval/debug.
  --venv PATH               Python venv path. Default: .venv-fastwam
  --harness-dir PATH        WAM Harness source path. Default: current directory.
  --cache-dir PATH          WAM cache path. Default: ${WAM_CACHE_DIR:-~/.cache/wam}
  --libero-dir PATH         LIBERO checkout path. Default: <cache-dir>/upstreams/LIBERO
  --python VERSION          Python version for uv venv. Default: 3.10
  --torch-backend BACKEND   uv PyTorch backend. Default: cu128
  --fastwam-ref REF         Optional FastWAM ref used with --clone and --upstream-dir. Default: 45d8e14
  --libero-ref REF          LIBERO git ref used with --clone. Default: master
  --clone                   Clone missing optional FastWAM reference repo and/or LIBERO.
  --no-harness              Do not install WAM Harness into the venv.
  --no-libero               Do not install LIBERO/simulator runtime packages.
  --no-configure-libero     Do not write LIBERO_CONFIG_PATH/config.yaml.
  --help                    Show this help.

After install:
  source <venv>/bin/activate
  export LIBERO_CONFIG_PATH=<cache-dir>/libero/config
  wam doctor fastwam-libero --cache-dir <cache-dir>
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_cmd="${PYTHON:-python3}"
venv_dir="${WAM_FASTWAM_VENV:-.venv-fastwam}"
harness_dir="$repo_root"
cache_dir="${WAM_CACHE_DIR:-$HOME/.cache/wam}"
upstream_dir="${WAM_FASTWAM_REPO:-}"
libero_dir=""
python_version="${WAM_FASTWAM_PYTHON:-3.10}"
torch_backend="${WAM_FASTWAM_TORCH_BACKEND:-cu128}"
fastwam_repo="${FASTWAM_REPO:-https://github.com/yuantianyuan01/FastWAM.git}"
fastwam_ref="${FASTWAM_REF:-45d8e14}"
libero_repo="${LIBERO_REPO:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
libero_ref="${LIBERO_REF:-master}"
clone_repos=0
install_harness=1
install_libero=1
configure_libero=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-dir)
      upstream_dir="${2:-}"
      shift 2
      ;;
    --venv)
      venv_dir="${2:-}"
      shift 2
      ;;
    --harness-dir)
      harness_dir="${2:-}"
      shift 2
      ;;
    --cache-dir)
      cache_dir="${2:-}"
      shift 2
      ;;
    --libero-dir)
      libero_dir="${2:-}"
      shift 2
      ;;
    --python)
      python_version="${2:-}"
      shift 2
      ;;
    --torch-backend)
      torch_backend="${2:-}"
      shift 2
      ;;
    --fastwam-ref)
      fastwam_ref="${2:-}"
      shift 2
      ;;
    --libero-ref)
      libero_ref="${2:-}"
      shift 2
      ;;
    --clone)
      clone_repos=1
      shift
      ;;
    --no-harness)
      install_harness=0
      shift
      ;;
    --no-libero)
      install_libero=0
      shift
      ;;
    --no-configure-libero)
      configure_libero=0
      shift
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

[[ -n "$venv_dir" ]] || die "--venv must not be empty"
[[ -n "$harness_dir" ]] || die "--harness-dir must not be empty"
[[ -n "$cache_dir" ]] || die "--cache-dir must not be empty"
[[ -n "$python_version" ]] || die "--python must not be empty"
[[ -n "$torch_backend" ]] || die "--torch-backend must not be empty"

if [[ -z "$libero_dir" ]]; then
  libero_dir="$cache_dir/upstreams/LIBERO"
fi

mkdir -p "$cache_dir"
command -v "$python_cmd" >/dev/null 2>&1 || die "$python_cmd is required before uv creates the target venv"
command -v uv >/dev/null 2>&1 || die "uv is required. Install uv first: https://docs.astral.sh/uv/"
command -v git >/dev/null 2>&1 || die "git is required"

mkdir -p "$cache_dir"
venv_dir="$("$python_cmd" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$venv_dir")"
harness_dir="$("$python_cmd" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$harness_dir")"
cache_dir="$("$python_cmd" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$cache_dir")"
libero_dir="$("$python_cmd" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$libero_dir")"

if [[ -n "$upstream_dir" ]]; then
  upstream_dir="$("$python_cmd" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$upstream_dir")"
  if [[ ! -d "$upstream_dir/.git" ]]; then
    if [[ "$clone_repos" != "1" ]]; then
      die "FastWAM reference repo not found at $upstream_dir. Omit --upstream-dir for the vendored native runtime, or pass --clone for reference eval."
    fi
    mkdir -p "$(dirname "$upstream_dir")"
    git clone "$fastwam_repo" "$upstream_dir"
  fi
  git -C "$upstream_dir" fetch --all --tags
  git -C "$upstream_dir" checkout "$fastwam_ref"
fi

if [[ "$install_libero" == "1" ]]; then
  if [[ ! -d "$libero_dir/.git" ]]; then
    if [[ "$clone_repos" != "1" ]]; then
      die "LIBERO repo not found at $libero_dir. Clone it first or pass --clone."
    fi
    mkdir -p "$(dirname "$libero_dir")"
    git clone "$libero_repo" "$libero_dir"
  fi
  git -C "$libero_dir" fetch --all --tags
  git -C "$libero_dir" checkout "$libero_ref"
fi

uv venv --python "$python_version" --allow-existing "$venv_dir"
python_bin="$venv_dir/bin/python"
uv pip install --python "$python_bin" pip
"$python_bin" -m pip install --no-cache-dir -U pip

uv pip install --python "$python_bin" --reinstall "cmake<4" huggingface_hub==0.29.2

mkdir -p "$cache_dir/bin"
cmake_real_bin="$("$python_bin" - <<'PY'
import os
import cmake

print(os.path.join(os.path.dirname(cmake.__file__), "data", "bin", "cmake"))
PY
)"
[[ -x "$cmake_real_bin" ]] || die "cmake wheel did not expose an executable at $cmake_real_bin"
ln -sf "$cmake_real_bin" "$cache_dir/bin/cmake"
export PATH="$cache_dir/bin:$PATH"

uv pip install --python "$python_bin" \
  --torch-backend "$torch_backend" \
  accelerate==1.12.0 \
  av==16.0.1 \
  boto3==1.35.99 \
  datasets==3.6.0 \
  einops==0.8.1 \
  gitpython==3.1.45 \
  huggingface-hub==0.29.2 \
  hydra-core==1.3.2 \
  imageio==2.37.0 \
  imageio-ffmpeg==0.6.0 \
  jsonlines==4.0.0 \
  numpy==1.26.4 \
  omegaconf==2.3.0 \
  packaging==25.0 \
  pandas==2.2.3 \
  pillow==12.0.0 \
  pyarrow==23.0.0 \
  regex==2025.11.3 \
  rich==14.2.0 \
  safetensors==0.5.3 \
  termcolor==2.5.0 \
  torch==2.7.1 \
  torchvision==0.22.1 \
  tqdm==4.66.5 \
  transformers==4.49.0 \
  typing-extensions==4.15.0

if [[ "$install_harness" == "1" ]]; then
  uv pip install --python "$python_bin" "$harness_dir"
fi

if [[ "$install_libero" == "1" ]]; then
  libero_config_path="${LIBERO_CONFIG_PATH:-$cache_dir/libero/config}"
  export LIBERO_CONFIG_PATH="$libero_config_path"

  uv pip install --python "$python_bin" \
    bddl==1.0.1 \
    cloudpickle==2.1.0 \
    easydict==1.9 \
    future==0.18.2 \
    gym==0.25.2 \
    matplotlib==3.8.4 \
    opencv-python-headless==4.6.0.66 \
    robomimic==0.2.0 \
    robosuite==1.4.0
  uv pip uninstall --python "$python_bin" opencv-python opencv-contrib-python || true
  uv pip install --python "$python_bin" --reinstall --no-deps \
    mujoco==3.3.2 \
    numpy==1.26.4 \
    opencv-python-headless==4.6.0.66
  "$python_bin" - <<'PY'
import importlib.util
from pathlib import Path
import shutil

spec = importlib.util.find_spec("robosuite")
if spec is not None and spec.submodule_search_locations:
    robosuite_root = Path(next(iter(spec.submodule_search_locations)))
    macros = robosuite_root / "macros.py"
    macros_private = robosuite_root / "macros_private.py"
    if macros.exists() and not macros_private.exists():
        shutil.copyfile(macros, macros_private)
PY
  uv pip install --python "$python_bin" -e "$libero_dir"

  [[ -d "$libero_dir/libero/libero" ]] || die "LIBERO package root not found at $libero_dir/libero/libero"
  "$python_bin" - "$libero_dir" <<'PY'
from pathlib import Path
import site
import sys

repo_root = Path(sys.argv[1]).resolve()
site_packages = site.getsitepackages()
if not site_packages:
    raise SystemExit("could not locate site-packages for LIBERO .pth install")
pth_path = Path(site_packages[0]) / "wam_fastwam_libero.pth"
pth_path.write_text(f"{repo_root}\n", encoding="utf-8")
PY

  if [[ "$configure_libero" == "1" ]]; then
    libero_datasets_path="${WAM_LIBERO_DATASETS_PATH:-$cache_dir/libero/datasets}"
    libero_benchmark_root="${WAM_LIBERO_BENCHMARK_ROOT:-$libero_dir/libero/libero}"
    mkdir -p "$libero_config_path" "$libero_datasets_path"
    cat >"$libero_config_path/config.yaml" <<EOF
assets: $libero_benchmark_root/assets
bddl_files: $libero_benchmark_root/bddl_files
benchmark_root: $libero_benchmark_root
datasets: $libero_datasets_path
init_states: $libero_benchmark_root/init_files
EOF
  fi
fi

"$python_bin" - <<'PY'
import importlib

mods = ["torch", "hydra", "omegaconf", "numpy", "PIL.Image", "einops", "fastwam"]
for name in mods:
    importlib.import_module(name)

try:
    import cv2
    import libero
    print(
        "FastWAM env smoke: "
        f"cv2={cv2.__version__} "
        f"libero={getattr(libero, '__file__', None)}"
    )
except ModuleNotFoundError:
    print("FastWAM env smoke: LIBERO runtime modules skipped")

try:
    import mujoco
    print(f"FastWAM env smoke: mujoco={mujoco.__version__}")
except ModuleNotFoundError:
    print("FastWAM env smoke: mujoco skipped")
except Exception as exc:
    print(f"FastWAM env smoke: mujoco warning: {type(exc).__name__}: {exc}")

import torch
print(f"FastWAM env smoke: torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

cat <<EOF

FastWAM native environment is installed.

Activate it:
  source "$venv_dir/bin/activate"

Use it:
  export WAM_CACHE_DIR="$cache_dir"
  export WAM_LIBERO_DIR="$libero_dir"
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$cache_dir/libero/config}"
  wam doctor fastwam-libero --cache-dir "$cache_dir"

Prepare model assets separately:
  wam prepare fastwam-libero --cache-dir "$cache_dir" --download --asset eval

Run the end-to-end acceptance path:
  "$harness_dir/scripts/fastwam-libero-eval.sh" --cache-dir "$cache_dir" --trace-dir <runs-dir>

EOF

if [[ -n "$upstream_dir" ]]; then
  cat <<EOF
Optional reference eval checkout:
  export WAM_FASTWAM_REPO="$upstream_dir"
  wam eval fastwam-libero --reference --upstream-dir "$upstream_dir"

EOF
fi
