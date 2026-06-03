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
    assert "--upstream-dir" in content
    assert "--venv" in content
    assert "--clone" in content
    assert "--torch-backend" in content
    assert "uv venv" in content
    assert "--allow-existing" in content
    assert "WAM_FASTWAM_REPO" in content
    assert "WAM_FASTWAM_TORCH_BACKEND" in content
    assert "LIBERO_CONFIG_PATH" in content
    assert "wam_fastwam_libero.pth" in content
    assert "libero.py" in content
    assert "compat_package = libero_package / \"libero\"" in content
    assert "from .. import *" in content
    assert "__path__ = [str(Path(__file__).resolve().parents[1])]" in content
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
    assert "COPY scripts/setup_fastwam_native_env.sh" in content
    assert "./scripts/setup_fastwam_native_env.sh" in content
    assert "--upstream-dir \"${WAM_FASTWAM_REPO}\"" in content
    assert "--harness-dir /workspace/wam-harness" in content
    assert "--clone" in content


def test_backend_native_smoke_scripts_define_container_contract() -> None:
    scripts = {
        "fastwam": ("fastwam-libero", "/opt/FastWAM", "wam-fastwam-native-smoke"),
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
