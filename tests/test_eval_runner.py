import json

import pytest

from wam_harness.cli import main
from wam_harness.core.eval_runner import EvalRunner, EvalRunnerError
from wam_harness.core.manifest import load_builtin_manifest


def test_real_eval_manifests_load() -> None:
    for model_id in (
        "fastwam-libero",
        "cosmos-policy-libero",
        "dreamzero-droid-sim",
    ):
        manifest = load_builtin_manifest(model_id)

        assert manifest.workload_name == "external_eval"
        if "workloads" in manifest.eval:
            default_workload = manifest.eval["default_workload"]
            assert manifest.eval["workloads"][default_workload]["command"]["argv"]
        else:
            assert manifest.eval["command"]["argv"]
        assert manifest.assets


def test_eval_runner_dry_run_plans_fastwam(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="fastwam-libero",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        upstream_dir="/tmp/FastWAM",
        dry_run=True,
        overrides={"num_gpus": "1", "create_only": "True"},
    )

    assert summary.status == "planned"
    assert summary.workload == "libero-manager"
    assert summary.return_code is None
    assert "experiments/libero/run_libero_manager.py" in summary.command.argv
    assert "MULTIRUN.create_only=True" in summary.command.argv
    assert summary.command.env["HF_HOME"] == f"{tmp_path / 'cache'}/huggingface"
    assert summary.command.env["LIBERO_CONFIG_PATH"] == f"{tmp_path / 'cache'}/libero/config"
    assert summary.command.env["PYTHONPATH"] == f"{tmp_path / 'cache'}/upstreams/LIBERO"
    assert (
        f"ckpt={tmp_path / 'cache'}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
        in summary.command.argv
    )
    assert summary.trace_path.exists()
    events = [
        json.loads(line)
        for line in summary.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["mode"] == "simulator_eval"


def test_eval_runner_reference_mode_stays_available_for_fastwam(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="fastwam-libero",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        upstream_dir="/tmp/FastWAM",
        dry_run=True,
        reference=True,
    )

    events = [
        json.loads(line)
        for line in summary.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary.status == "planned"
    assert events[0]["mode"] == "reference_eval"


def test_eval_runner_dry_run_plans_fastwam_single_task(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="fastwam-libero",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        upstream_dir="/tmp/FastWAM",
        dry_run=True,
        reference=True,
        workload="libero-single-task",
        overrides={"task_id": "3", "num_trials": "1"},
    )

    assert summary.status == "planned"
    assert summary.model_id == "fastwam-libero"
    assert summary.workload == "libero-single-task"
    assert "experiments/libero/eval_libero_single.py" in summary.command.argv
    assert "EVALUATION.task_id=3" in summary.command.argv
    assert "model.redirect_common_files=False" in summary.command.argv
    assert summary.command.env["DIFFSYNTH_DOWNLOAD_SOURCE"] == "huggingface"
    assert (
        summary.command.env["DIFFSYNTH_MODEL_BASE_PATH"]
        == f"{tmp_path / 'cache'}/diffsynth-models"
    )
    assert summary.command.env["MUJOCO_GL"] == "osmesa"
    assert summary.command.env["PYOPENGL_PLATFORM"] == "osmesa"
    assert summary.command.env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_eval_runner_fastwam_single_task_allows_gl_backend_overrides(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="fastwam-libero",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        upstream_dir="/tmp/FastWAM",
        dry_run=True,
        workload="libero-single-task",
        overrides={
            "mujoco_gl": "egl",
            "pyopengl_platform": "egl",
        },
    )

    assert summary.command.env["MUJOCO_GL"] == "egl"
    assert summary.command.env["PYOPENGL_PLATFORM"] == "egl"


def test_eval_runner_rejects_unknown_fastwam_eval_workload(tmp_path) -> None:
    try:
        EvalRunner().run(
            model_id="fastwam-libero",
            trace_dir=tmp_path,
            upstream_dir="/tmp/FastWAM",
            dry_run=True,
            reference=True,
            workload="fastwam-libero-single-task",
        )
    except EvalRunnerError as exc:
        assert "unknown eval workload" in str(exc)
        assert "libero-single-task" in str(exc)
    else:
        raise AssertionError("expected EvalRunnerError")


def test_eval_runner_traces_execution_validation_failure(tmp_path) -> None:
    with pytest.raises(EvalRunnerError, match="external eval workdir does not exist"):
        EvalRunner().run(
            model_id="fastwam-libero",
            trace_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            upstream_dir=tmp_path / "missing-fastwam",
            dry_run=False,
            overrides={"create_only": "True"},
        )

    trace_paths = list(tmp_path.glob("*/trace.jsonl"))
    assert len(trace_paths) == 1
    events = [
        json.loads(line)
        for line in trace_paths[0].read_text(encoding="utf-8").splitlines()
    ]

    assert [event["event"] for event in events] == [
        "run_start",
        "external_eval_plan",
        "error",
        "run_end",
    ]
    assert events[-2]["stage"] == "external_eval_validation"
    assert events[-2]["recoverable"] is True
    assert events[-2]["backend"] == "external_eval"
    assert events[-1]["status"] == "error"
    assert events[-1]["return_code"] is None
    assert events[-1]["trace_path"] == str(trace_paths[0])


def test_eval_runner_profile_context_dreamzero_dit_cache(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="dreamzero-droid-sim",
        enabled_opts=["dit_cache"],
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        upstream_dir="/tmp/dreamzero",
        dry_run=True,
        reference=True,
    )

    assert "--enable-dit-cache" in summary.command.display
    assert summary.command.argv[:2] == ["bash", "-lc"]
    assert "PYTHONPATH" not in summary.command.env
    assert (
        "sim_eval_pythonpath="
        '"/mnt/wam-harness/src/wam_harness/compat/dreamzero_eval:'
        f"{tmp_path / 'cache'}/upstreams/sim-evals/src:/tmp/dreamzero/eval_utils"
        in summary.command.display
    )
    assert summary.command.env["ISAAC_SIM_CACHE_PATH"].endswith("/isaac-sim/cache")
    assert '"/tmp/dreamzero/.venv/bin/python" -m torch.distributed.run' in (
        summary.command.display
    )
    assert f'"{tmp_path / "cache"}/venvs/dreamzero-sim/bin/python" -m eval_utils.run_sim_eval' in (
        summary.command.display
    )
    assert f"--model-path {tmp_path / 'cache'}/GEAR-Dreams/DreamZero-DROID" in (
        summary.command.display
    )


def test_eval_runner_dry_run_plans_cosmos_libero_smoke(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="cosmos-policy-libero",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        upstream_dir="/tmp/cosmos-policy",
        dry_run=True,
        reference=True,
    )

    num_trials_idx = summary.command.argv.index("--num_trials_per_task")
    assert summary.command.argv[num_trials_idx + 1] == "1"
    assert summary.command.env["LIBERO_CONFIG_PATH"] == f"{tmp_path / 'cache'}/libero/config"
    assert summary.command.env["PYTHONPATH"] == f"{tmp_path / 'cache'}/upstreams/LIBERO"
    assert summary.command.env["MUJOCO_GL"] == "osmesa"
    assert summary.command.env["PYOPENGL_PLATFORM"] == "osmesa"
    assert summary.command.env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_cli_eval_dry_run(capsys, tmp_path) -> None:
    exit_code = main(
        [
            "eval",
            "fastwam-libero",
            "--trace-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--upstream-dir",
            "/tmp/FastWAM",
            "--dry-run",
            "--set",
            "create_only=True",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["model_id"] == "fastwam-libero"
    assert payload["workload"] == "libero-manager"
    assert payload["command"]["env"]["HF_HOME"] == f"{tmp_path / 'cache'}/huggingface"


def test_cli_eval_single_task_workload_shortcuts(capsys, tmp_path) -> None:
    exit_code = main(
        [
            "eval",
            "fastwam-libero",
            "--workload",
            "libero-single-task",
            "--trace-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--upstream-dir",
            "/tmp/FastWAM",
            "--dry-run",
            "--reference",
            "--task-id",
            "3",
            "--num-trials",
            "1",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["model_id"] == "fastwam-libero"
    assert payload["workload"] == "libero-single-task"
    assert "experiments/libero/eval_libero_single.py" in payload["command"]["argv"]
    assert "EVALUATION.task_id=3" in payload["command"]["argv"]
    assert "EVALUATION.num_trials=1" in payload["command"]["argv"]


def test_cli_eval_without_reference_runs_simulator_eval_plan(capsys, tmp_path) -> None:
    exit_code = main(
        [
            "eval",
            "fastwam-libero",
            "--trace-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--upstream-dir",
            "/tmp/FastWAM",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["workload"] == "libero-manager"
    assert "experiments/libero/run_libero_manager.py" in payload["command"]["argv"]
    assert payload["command"]["env"]["HF_HOME"] == f"{tmp_path / 'cache'}/huggingface"
    assert captured.err == ""
    assert "Traceback" not in captured.err
