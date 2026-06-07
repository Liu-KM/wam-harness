from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_backend_dockerfiles_install_wam_harness_cli() -> None:
    for relative in [
        "containers/fastwam/Dockerfile",
        "containers/cosmos-policy/Dockerfile",
        "containers/dreamzero/Dockerfile",
    ]:
        content = (ROOT / relative).read_text(encoding="utf-8")

        assert "COPY pyproject.toml README.md uv.lock ./" in content
        assert "COPY src ./src" in content
        assert "/workspace/wam-harness" in content
        assert "wam" in content


def test_core_dockerfile_uses_wam_cli_as_default_command() -> None:
    content = (ROOT / "containers/core/Dockerfile").read_text(encoding="utf-8")

    assert 'PATH="/workspace/wam-harness/.venv/bin:${PATH}"' in content
    assert 'CMD ["wam", "--help"]' in content


def test_fastwam_native_setup_script_defines_self_managed_environment() -> None:
    script_path = ROOT / "scripts/setup_fastwam_native_env.sh"
    content = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    assert "Optional FastWAM checkout path for reference eval/debug" in content
    assert "--upstream-dir is required" not in content
    assert "--venv" in content
    assert "--clone" in content
    assert "--torch-backend" in content
    assert "uv venv" in content
    assert "--allow-existing" in content
    assert "uv pip install --python \"$python_bin\" \\" in content
    assert "torch==2.7.1" in content
    assert "transformers==4.49.0" in content
    assert "-e \"$upstream_dir\"" not in content
    assert "WAM_FASTWAM_TORCH_BACKEND" in content
    assert "LIBERO_CONFIG_PATH" in content
    assert "WAM_LIBERO_DIR" in content
    assert "fastwam-libero-eval.sh" in content
    assert "wam_fastwam_libero.pth" in content
    assert 'pth_path.write_text(f"{repo_root}\\n", encoding="utf-8")' in content
    assert '[[ -d "$libero_dir/libero/libero" ]]' in content
    assert "compat_package = libero_package / \"libero\"" not in content
    assert "import libero" in content
    assert "wam doctor fastwam-libero" in content
    assert "wam prepare fastwam-libero" in content
    assert "bddl==1.0.1" in content
    assert "robosuite==1.4.0" in content
    assert "macros_private.py" in content
    assert "shutil.copyfile(macros, macros_private)" in content
    assert "mujoco==3.3.2" in content
    assert "FastWAM env smoke: mujoco warning" in content
    assert "except Exception as exc:" in content
    assert "numpy==1.26.4" in content
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" not in content
    assert "sbatch" not in content
    assert "srun" not in content


def test_fastwam_dockerfile_reuses_native_setup_script() -> None:
    content = (ROOT / "containers/fastwam/Dockerfile").read_text(encoding="utf-8")

    assert "ENV WAM_FASTWAM_VENV=/opt/wam-fastwam-venv" in content
    assert "ENV WAM_CACHE_DIR=/mnt/wam-cache" in content
    assert "ENV WAM_TRACE_DIR=/mnt/runs" in content
    assert "ENV WAM_LIBERO_DIR=/opt/LIBERO" in content
    assert "ENV LIBERO_CONFIG_PATH=/mnt/wam-cache/libero/config" in content
    assert "ENV MUJOCO_GL=egl" in content
    assert "ENV PYOPENGL_PLATFORM=egl" in content
    assert "ENV WAM_FASTWAM_REPO" not in content
    assert "COPY scripts/setup_fastwam_native_env.sh" in content
    assert "./scripts/setup_fastwam_native_env.sh" in content
    assert "--harness-dir /workspace/wam-harness" in content
    assert "--clone" in content


def test_fastwam_libero_eval_acceptance_script_checks_simulator_env() -> None:
    script_path = ROOT / "scripts/fastwam-libero-eval.sh"
    content = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    assert "--skip-simulator-check" in content
    assert "--min-success-rate" in content
    assert "--libero-dir" in content
    assert "WAM_LIBERO_DIR" in content
    assert 'min_success_rate="${WAM_ACCEPT_MIN_SUCCESS_RATE:-1.0}"' in content
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in content
    assert 'elif [[ -d /opt/LIBERO/libero/libero ]]; then' in content
    assert 'assets: $libero_dir/libero/libero/assets' in content
    assert "_import_libero_modules" in content
    assert "_load_task" in content
    assert "_create_libero_env" in content
    assert "_observation_from_libero" in content
    assert "env.set_init_state(states[0])" in content
    assert "absolute_path()" in content
    assert 'cache_dir="$(absolute_path "$cache_dir")"' in content
    assert 'trace_dir="$(absolute_path "$trace_dir")"' in content
    assert "wam native-smoke" in content
    assert 'print_cmd wam "${eval_args[@]}"' in content
    assert '--set "mujoco_gl=$mujoco_gl"' in content
    assert '--set "pyopengl_platform=$pyopengl_platform"' in content
    assert 'eval_raw_output_path="$trace_dir/${model_id}-${workload}-eval-output.txt"' in content
    assert 'eval_args+=(--summary-path "$eval_summary_path")' in content
    assert 'wam "${eval_args[@]}" | tee "$eval_raw_output_path"' in content
    assert "raw_decode" not in content
    assert "eval_summary_path=" in content
    assert "acceptance_report_path=" in content
    assert '${model_id}-${workload}-acceptance.json' in content
    assert "cat \"$eval_summary_path\"" in content
    assert "tee \"$acceptance_report_path\"" in content
    assert "python -m wam_harness.evals.acceptance" in content
    assert "python -m wam_harness.evals.acceptance --json" in content
    assert '"$min_success_rate"' in content


def test_backend_native_smoke_scripts_define_container_contract() -> None:
    scripts = {
        "fastwam": ("fastwam-libero", None, "wam-fastwam-native-smoke"),
        "cosmos-policy": (
            "cosmos-policy-libero",
            "/opt/cosmos-policy",
            "wam-cosmos-policy-native-smoke",
        ),
        "dreamzero": ("dreamzero-droid-sim", "/opt/dreamzero", "wam-dreamzero-native-smoke"),
    }

    for backend, (model_id, upstream_dir, command_name) in scripts.items():
        script_path = ROOT / "containers" / backend / "native-smoke.sh"
        content = script_path.read_text(encoding="utf-8")

        subprocess.run(["bash", "-n", str(script_path)], check=True)
        assert f"WAM_MODEL_ID:-{model_id}" in content
        if upstream_dir is None:
            assert "WAM_UPSTREAM_DIR" not in content
            assert "--upstream-dir" not in content
        else:
            assert f"WAM_UPSTREAM_DIR:-{upstream_dir}" in content
        assert "wam prepare" in content
        assert "prepare_status=0" in content
        assert "running wam doctor for native readiness" in content
        assert "wam doctor" in content
        assert "--json --strict" in content
        assert "wam native-smoke" in content
        assert "--require-ready" in content

        dockerfile = (ROOT / "containers" / backend / "Dockerfile").read_text(
            encoding="utf-8"
        )
        assert f"COPY containers/{backend}/native-smoke.sh /usr/local/bin/{command_name}" in dockerfile
        assert f"chmod +x /usr/local/bin/{command_name}" in dockerfile
