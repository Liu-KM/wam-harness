import json
import sys
import types

import pytest

from eazywam.backends.fake import FakeBackend
from eazywam.backends.native_support.runtime import native_runtime_resolver
from eazywam.cli import main
from eazywam.core.eval_runner import EvalRunner, EvalRunnerError
from eazywam.core.manifest import load_builtin_manifest, manifest_from_dict
from eazywam.core.registry import Registry
from eazywam.evals.acceptance import validate_native_eval_summary
from eazywam.evals.libero import (
    LiberoSingleTaskEvalRunner,
    _EvalContext as LiberoEvalContext,
    _import_libero_modules,
    _runtime_options as libero_runtime_options,
)
from eazywam.evals.robotwin import (
    RobotWinSingleTaskEvalRunner,
    _EvalContext as RobotWinEvalContext,
    _runtime_options as robotwin_runtime_options,
)
from eazywam.processors.passthrough import PassthroughProcessor


def test_real_eval_manifests_load() -> None:
    for model_id in (
        "fastwam-libero",
        "fastwam-robotwin",
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
        reference=True,
        overrides={"num_gpus": "1", "create_only": "True"},
        workload="libero-manager",
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
    assert events[0]["mode"] == "reference_eval"


def test_libero_runtime_options_include_acceleration_modes(tmp_path) -> None:
    context = LiberoEvalContext(
        task_suite_name="libero_10",
        task_id=0,
        num_trials=1,
        action_horizon=32,
        replan_steps=10,
        num_steps_wait=30,
        max_steps=700,
        seed=42,
        num_inference_steps=10,
        output_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        values={
            "dit_cache_mode": "video_kv",
            "cuda_graph_mode": "off",
            "torch_compile_mode": "auto",
        },
    )

    assert libero_runtime_options(context) == {
        "num_inference_steps": 10,
        "dit_cache_mode": "video_kv",
        "cuda_graph_mode": "off",
        "torch_compile_mode": "auto",
    }


def test_robotwin_runtime_options_include_acceleration_modes(tmp_path) -> None:
    context = RobotWinEvalContext(
        task_name="click_alarmclock",
        task_config="demo_randomized",
        instruction_type="unseen",
        num_episodes=1,
        action_horizon=32,
        replan_steps=24,
        seed=42,
        output_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        robotwin_root=tmp_path / "RoboTwin",
        values={
            "num_inference_steps": 10,
            "dit_cache_mode": "video_kv",
            "cuda_graph_mode": "off",
            "torch_compile_mode": "auto",
        },
    )

    assert robotwin_runtime_options(context) == {
        "num_inference_steps": 10,
        "dit_cache_mode": "video_kv",
        "cuda_graph_mode": "off",
        "torch_compile_mode": "auto",
    }


def test_eval_runner_native_dry_run_is_default_for_fastwam_single_task(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="fastwam-libero",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        dry_run=True,
        overrides={"task_id": "3", "num_trials": "1"},
    )

    assert summary.status == "planned"
    assert summary.workload == "libero-single-task"
    assert summary.return_code is None
    assert summary.command.argv[0:2] == ["wam-native-eval", "libero-single-task"]
    assert not any("experiments/libero" in item for item in summary.command.argv)
    assert summary.runtime_info.backend == "fastwam"
    events = [
        json.loads(line)
        for line in summary.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_start",
        "native_eval_plan",
        "run_end",
    ]
    assert events[0]["mode"] == "simulator_eval"
    assert events[0]["native_eval"] is True
    assert events[1]["eval_runner"] == "libero_single_task"
    assert events[1]["task_id"] == 3
    assert events[1]["seed"] == 42
    assert events[1]["num_steps_wait"] == 30
    assert summary.command.env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert summary.command.env["TOKENIZERS_PARALLELISM"] == "false"
    assert summary.command.env["WANDB_MODE"] == "offline"


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


def test_eval_runner_dry_run_plans_fastwam_robotwin_single_task(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="fastwam-robotwin",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        upstream_dir="/tmp/FastWAM",
        dry_run=True,
        reference=True,
        workload="robotwin-single-task",
        overrides={"task_name": "click_alarmclock", "num_episodes": "1"},
    )

    assert summary.status == "planned"
    assert summary.model_id == "fastwam-robotwin"
    assert summary.workload == "robotwin-single-task"
    assert "experiments/robotwin/eval_robotwin_single.py" in summary.command.argv
    assert "task=robotwin_uncond_3cam_384_1e-4" in summary.command.argv
    assert "seed=42" in summary.command.argv
    assert (
        f"ckpt={tmp_path / 'cache'}/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
        in summary.command.argv
    )
    assert "EVALUATION.task_name=click_alarmclock" in summary.command.argv
    assert "EVALUATION.eval_num_episodes=1" in summary.command.argv


def test_eval_runner_native_robotwin_dry_run_is_default(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="fastwam-robotwin",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        upstream_dir="/tmp/FastWAM",
        dry_run=True,
        overrides={"task_name": "click_alarmclock", "num_episodes": "1"},
    )

    assert summary.status == "planned"
    assert summary.workload == "robotwin-single-task"
    assert summary.return_code is None
    assert summary.command.argv[0:2] == ["wam-native-eval", "robotwin-single-task"]
    assert not any("experiments/robotwin" in item for item in summary.command.argv)
    assert summary.command.env["WAM_ROBOTWIN_ROOT"] == "/tmp/FastWAM/third_party/RoboTwin"
    assert summary.command.env["DIFFSYNTH_DOWNLOAD_SOURCE"] == "huggingface"
    assert (
        summary.command.env["DIFFSYNTH_MODEL_BASE_PATH"]
        == f"{tmp_path / 'cache'}/diffsynth-models"
    )
    assert summary.command.env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    events = [
        json.loads(line)
        for line in summary.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    plan = next(event for event in events if event["event"] == "native_eval_plan")
    assert plan["replan_steps"] == 24
    assert plan["action_horizon"] == 32
    assert plan["task_name"] == "click_alarmclock"


def test_eval_runner_dry_run_plans_fastwam_robotwin_manager(tmp_path) -> None:
    summary = EvalRunner().run(
        model_id="fastwam-robotwin",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        upstream_dir="/tmp/FastWAM",
        dry_run=True,
        reference=True,
        workload="robotwin-manager",
        overrides={"num_gpus": "1", "max_tasks_per_gpu": "1", "num_episodes": "1"},
    )

    assert summary.status == "planned"
    assert summary.workload == "robotwin-manager"
    assert summary.command.argv[:3] == [
        "python",
        "-m",
        "eazywam.compat.fastwam_robotwin_manager",
    ]
    assert "--task-name" in summary.command.argv
    assert summary.command.argv[summary.command.argv.index("--task-name") + 1] == "null"
    assert "--num-episodes" in summary.command.argv
    assert summary.command.argv[summary.command.argv.index("--num-episodes") + 1] == "1"
    assert "--max-worker-restarts-on-invalid-setup" in summary.command.argv


def test_eval_runner_loads_external_summary_metrics_for_robotwin_manager(
    monkeypatch,
    tmp_path,
) -> None:
    upstream_dir = tmp_path / "FastWAM"
    upstream_dir.mkdir()

    def fake_subprocess_run(argv, **kwargs):  # noqa: ANN001, ANN202
        output_dir = tmp_path / "missing"
        if "--output-dir" in argv:
            output_dir = tmp_path / str(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "overall": {
                        "clean_mean_success_rate": 1.0,
                        "random_mean_success_rate": 0.5,
                    },
                    "per_task": [
                        {
                            "task_name": "click_alarmclock",
                            "clean_success_rate": 1.0,
                            "random_success_rate": 0.5,
                        }
                    ],
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )
        stdout = kwargs.get("stdout")
        if stdout is not None:
            stdout.write("ok\n")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    summary = EvalRunner().run(
        model_id="fastwam-robotwin",
        trace_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        upstream_dir=upstream_dir,
        dry_run=False,
        reference=True,
        workload="robotwin-manager",
        overrides={"task_name": "click_alarmclock", "num_episodes": "1"},
    )

    assert summary.status == "ok"
    assert summary.metrics["summary_metrics_found"] is True
    assert summary.metrics["overall"]["clean_mean_success_rate"] == 1.0
    assert summary.metrics["overall"]["random_mean_success_rate"] == 0.5
    assert summary.metrics["per_task"][0]["task_name"] == "click_alarmclock"

    events = [
        json.loads(line)
        for line in summary.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    end = next(event for event in events if event["event"] == "external_eval_end")
    assert end["metrics"]["overall"]["random_mean_success_rate"] == 0.5


def test_eval_runner_loads_robotwin_manager_summary_on_failure(
    monkeypatch,
    tmp_path,
) -> None:
    upstream_dir = tmp_path / "FastWAM"
    upstream_dir.mkdir()

    def fake_subprocess_run(argv, **kwargs):  # noqa: ANN001, ANN202
        output_dir = tmp_path / str(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "requested": {"target_valid_episodes": 2},
                    "actual": {
                        "valid_episodes": 0,
                        "invalid_setup_count": 1,
                    },
                    "invalid_setups": [
                        {
                            "category": "simulator_setup_invalid",
                            "policy_failure": False,
                            "reason": "put_bottles_dustbin_expert_setup_index_error",
                        }
                    ],
                    "failures": [
                        {
                            "category": "simulator_setup_invalid",
                            "policy_failure": False,
                            "reason": "invalid_setup_exhausted",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        stdout = kwargs.get("stdout")
        if stdout is not None:
            stdout.write("invalid setup\n")
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    summary = EvalRunner().run(
        model_id="fastwam-robotwin",
        trace_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        upstream_dir=upstream_dir,
        dry_run=False,
        reference=True,
        workload="robotwin-manager",
        overrides={"task_name": "put_bottles_dustbin", "num_episodes": "1"},
    )

    assert summary.status == "error"
    assert summary.return_code == 1
    assert summary.metrics["summary_metrics_found"] is True
    assert summary.metrics["actual"]["invalid_setup_count"] == 1
    assert summary.metrics["invalid_setups"][0]["policy_failure"] is False
    assert summary.metrics["failures"][0]["reason"] == "invalid_setup_exhausted"

    events = [
        json.loads(line)
        for line in summary.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    end = next(event for event in events if event["event"] == "external_eval_end")
    assert end["metrics"]["actual"]["invalid_setup_count"] == 1


def test_eval_runner_requires_reference_for_workload_without_native_runner(tmp_path) -> None:
    with pytest.raises(EvalRunnerError, match="does not declare a native product eval runner"):
        EvalRunner().run(
            model_id="fastwam-libero",
            trace_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            upstream_dir="/tmp/FastWAM",
            dry_run=True,
            workload="libero-manager",
        )


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
    assert "seed=42" in summary.command.argv
    assert "EVALUATION.task_id=3" in summary.command.argv
    assert "EVALUATION.num_steps_wait=30" in summary.command.argv
    assert "model.redirect_common_files=False" in summary.command.argv
    assert summary.command.env["DIFFSYNTH_DOWNLOAD_SOURCE"] == "huggingface"
    assert (
        summary.command.env["DIFFSYNTH_MODEL_BASE_PATH"]
        == f"{tmp_path / 'cache'}/diffsynth-models"
    )
    assert summary.command.env["MUJOCO_GL"] == "egl"
    assert summary.command.env["PYOPENGL_PLATFORM"] == "egl"
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
            "mujoco_gl": "osmesa",
            "pyopengl_platform": "osmesa",
        },
    )

    assert summary.command.env["MUJOCO_GL"] == "osmesa"
    assert summary.command.env["PYOPENGL_PLATFORM"] == "osmesa"


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
            reference=True,
            overrides={"create_only": "True"},
            workload="libero-manager",
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
        '"/mnt/eazywam/src/eazywam/compat/dreamzero_eval:'
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
            "--reference",
            "--workload",
            "libero-manager",
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


def test_cli_eval_summary_path_writes_clean_json(capsys, tmp_path) -> None:
    summary_path = tmp_path / "summary" / "eval-summary.json"

    exit_code = main(
        [
            "eval",
            "fastwam-libero",
            "--trace-dir",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--dry-run",
            "--summary-path",
            str(summary_path),
        ]
    )

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary_path.exists()
    assert file_payload == stdout_payload
    assert file_payload["status"] == "planned"
    assert file_payload["workload"] == "libero-single-task"


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
    assert "seed=42" in payload["command"]["argv"]
    assert "EVALUATION.task_id=3" in payload["command"]["argv"]
    assert "EVALUATION.num_trials=1" in payload["command"]["argv"]
    assert "EVALUATION.num_steps_wait=30" in payload["command"]["argv"]


def test_cli_eval_robotwin_shortcuts(capsys, tmp_path) -> None:
    exit_code = main(
        [
            "eval",
            "fastwam-robotwin",
            "--workload",
            "robotwin-single-task",
            "--trace-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--upstream-dir",
            "/tmp/FastWAM",
            "--dry-run",
            "--reference",
            "--task-name",
            "click_alarmclock",
            "--num-episodes",
            "1",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["model_id"] == "fastwam-robotwin"
    assert payload["workload"] == "robotwin-single-task"
    assert "EVALUATION.task_name=click_alarmclock" in payload["command"]["argv"]
    assert "EVALUATION.eval_num_episodes=1" in payload["command"]["argv"]


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
    assert payload["workload"] == "libero-single-task"
    assert payload["command"]["argv"][0:2] == ["wam-native-eval", "libero-single-task"]
    assert not any("experiments/libero" in item for item in payload["command"]["argv"])
    assert captured.err == ""
    assert "Traceback" not in captured.err


def test_eval_runner_native_libero_loop_runs_without_subprocess(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("subprocess.run", _fail_subprocess_run)
    _install_fake_libero(monkeypatch, tmp_path)
    registry = _fake_libero_registry()

    summary = EvalRunner(registry).run(
        model_id="fake-libero",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        dry_run=False,
        overrides={
            "num_trials": "2",
            "num_steps_wait": "0",
            "action_horizon": "2",
            "replan_steps": "1",
            "max_steps": "3",
        },
    )

    assert summary.status == "ok"
    assert summary.return_code == 0
    assert summary.metrics["successes"] == 2
    assert summary.metrics["total_episodes"] == 2
    assert summary.metrics["model_calls"] == 2
    assert summary.metrics["steps"] == 2
    assert summary.metrics["success_rate"] == 1.0
    assert summary.metrics["results_path"]

    events = [
        json.loads(line)
        for line in summary.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    event_names = [event["event"] for event in events]
    assert "external_eval_plan" not in event_names
    assert "native_eval_plan" in event_names
    assert event_names.count("episode_start") == 2
    assert event_names.count("inference_end") == 2
    assert event_names.count("simulator_step") == 2
    assert event_names[-2] == "native_eval_end"
    assert events[-2]["success_rate"] == 1.0
    assert events[-1]["event"] == "run_end"
    assert events[-1]["status"] == "ok"

    summary_path = tmp_path / "fake-libero-native-eval-summary.json"
    summary_path.write_text(json.dumps(summary.to_dict()), encoding="utf-8")
    report = validate_native_eval_summary(
        summary_path,
        expected_trials=2,
        min_success_rate=1.0,
    )
    assert report.success_rate == 1.0
    assert report.expected_trials == 2


def test_eval_runner_native_robotwin_policy_runs_without_subprocess(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("subprocess.run", _fail_subprocess_run)
    robotwin_root = tmp_path / "RoboTwin"
    _install_fake_robotwin(robotwin_root, monkeypatch)
    registry = _fake_robotwin_registry(robotwin_root)

    summary = EvalRunner(registry).run(
        model_id="fake-robotwin",
        trace_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        dry_run=False,
        overrides={
            "task_name": "click_alarmclock",
            "num_episodes": "2",
            "action_horizon": "2",
            "replan_steps": "1",
            "robotwin_root": str(robotwin_root),
        },
    )

    assert summary.status == "ok"
    assert summary.return_code == 0
    assert summary.metrics["successes"] == 2
    assert summary.metrics["requested_episodes"] == 2
    assert summary.metrics["valid_episodes"] == 2
    assert summary.metrics["invalid_setup_count"] == 0
    assert summary.metrics["invalid_setups"] == []
    assert summary.metrics["total_episodes"] == 2
    assert summary.metrics["success_rate"] == 1.0
    assert summary.metrics["task_name"] == "click_alarmclock"

    events = [
        json.loads(line)
        for line in summary.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    event_names = [event["event"] for event in events]
    assert "external_eval_plan" not in event_names
    assert "native_eval_plan" in event_names
    assert event_names.count("episode_start") == 2
    assert event_names.count("inference_end") == 2
    assert event_names.count("simulator_step") == 2
    assert events[-2]["event"] == "native_eval_end"
    assert events[-2]["success_rate"] == 1.0
    assert events[-1]["event"] == "run_end"
    assert events[-1]["status"] == "ok"


def test_libero_importer_handles_inner_package_exposed_as_top_level(
    monkeypatch,
    tmp_path,
) -> None:
    libero_root = tmp_path / "LIBERO"
    inner = libero_root / "libero" / "libero"
    (inner / "benchmark").mkdir(parents=True)
    (inner / "envs").mkdir()
    (inner / "__init__.py").write_text(
        "def get_libero_path(name):\n"
        "    return name\n",
        encoding="utf-8",
    )
    (inner / "benchmark" / "__init__.py").write_text(
        "from libero.libero import get_libero_path\n"
        "def get_benchmark_dict():\n"
        "    return {}\n",
        encoding="utf-8",
    )
    (inner / "envs" / "__init__.py").write_text(
        "from libero.libero import get_libero_path\n"
        "class OffScreenRenderEnv:\n"
        "    pass\n",
        encoding="utf-8",
    )
    for key in list(sys.modules):
        if key == "libero" or key.startswith("libero."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    monkeypatch.syspath_prepend(str(libero_root / "libero"))

    modules = _import_libero_modules()

    assert modules.get_libero_path("bddl_files") == "bddl_files"
    assert modules.benchmark.get_benchmark_dict() == {}
    assert modules.offscreen_render_env.__name__ == "OffScreenRenderEnv"


def _fail_subprocess_run(*args, **kwargs):  # noqa: ANN002, ANN003
    raise AssertionError("native eval must not call subprocess.run")


class _SingleManifestCatalog:
    def __init__(self, manifest):
        self.manifest = manifest

    def load_manifest(self, model_id: str):
        if model_id != self.manifest.id:
            raise KeyError(model_id)
        return self.manifest

    def list_model_ids(self) -> list[str]:
        return [self.manifest.id]


def _fake_libero_registry() -> Registry:
    manifest = manifest_from_dict(
        {
            "schema_version": 1,
            "id": "fake-libero",
            "display_name": "Fake LIBERO",
            "source": {"repo": "local/fake-libero"},
            "assets": {},
            "backend": {
                "name": "external_eval",
                "mode": "official_script",
                "config": {"native_backend": "fake", "action_dim": 7},
            },
            "processor": {
                "name": "passthrough",
                "action": {"horizon": 2, "dim": 7},
            },
            "workload": {"name": "external_eval", "config": {}},
            "defaults": {
                "device": "cpu",
                "dtype": "fp32",
                "action_horizon": 2,
                "replan_steps": 1,
            },
            "optimizations": {"supported": []},
            "eval": {
                "simulator": "LIBERO",
                "suite": "libero_10",
                "default_workload": "libero-single-task",
                "defaults": {
                    "task_suite_name": "libero_10",
                    "task_id": "0",
                    "num_trials": "1",
                    "num_steps_wait": "0",
                },
                "workloads": {
                    "libero-single-task": {
                        "native": {"runner": "libero_single_task"},
                        "defaults": {"task_suite_name": "libero_10"},
                    }
                },
            },
        }
    )
    registry = Registry(catalog=_SingleManifestCatalog(manifest))
    registry.register_backend("fake", FakeBackend)
    registry.register_processor("passthrough", PassthroughProcessor.from_manifest)
    registry.register_runtime_resolver(native_runtime_resolver)
    registry.register_eval_runner(
        "libero_single_task",
        lambda current_registry: LiberoSingleTaskEvalRunner(current_registry),
    )
    return registry


def _fake_robotwin_registry(robotwin_root) -> Registry:
    manifest = manifest_from_dict(
        {
            "schema_version": 1,
            "id": "fake-robotwin",
            "display_name": "Fake RoboTwin",
            "source": {"repo": "local/fake-robotwin"},
            "assets": {},
            "asset_groups": {},
            "backend": {
                "name": "external_eval",
                "mode": "official_script",
                "config": {"native_backend": "fake", "action_dim": 14},
            },
            "processor": {
                "name": "passthrough",
                "action": {"horizon": 2, "dim": 14},
            },
            "workload": {"name": "external_eval", "config": {}},
            "defaults": {
                "device": "cpu",
                "dtype": "fp32",
                "action_horizon": 2,
                "replan_steps": 1,
            },
            "optimizations": {"supported": []},
            "eval": {
                "simulator": "RoboTwin",
                "suite": "robotwin2.0",
                "default_workload": "robotwin-single-task",
                "defaults": {
                    "task_name": "click_alarmclock",
                    "task_config": "demo_randomized",
                    "instruction_type": "unseen",
                    "num_episodes": "1",
                    "robotwin_root": str(robotwin_root),
                    "seed": "42",
                    "replan_steps": "1",
                },
                "workloads": {
                    "robotwin-single-task": {
                        "native": {"runner": "robotwin_single_task"},
                        "defaults": {"task_name": "click_alarmclock"},
                    }
                },
            },
        }
    )
    registry = Registry(catalog=_SingleManifestCatalog(manifest))
    registry.register_backend("fake", FakeBackend)
    registry.register_processor("passthrough", PassthroughProcessor.from_manifest)
    registry.register_runtime_resolver(native_runtime_resolver)
    registry.register_eval_runner(
        "robotwin_single_task",
        lambda current_registry: RobotWinSingleTaskEvalRunner(current_registry),
    )
    return registry


def _install_fake_libero(monkeypatch, tmp_path) -> None:
    root_pkg = types.ModuleType("libero")
    libero_mod = types.ModuleType("libero.libero")
    benchmark_mod = types.ModuleType("libero.libero.benchmark")
    envs_mod = types.ModuleType("libero.libero.envs")

    class FakeTask:
        language = "open the fake drawer"
        problem_folder = "fake"
        bddl_file = "task.bddl"

    class FakeTaskSuite:
        def get_task(self, task_id: int):
            assert task_id == 0
            return FakeTask()

        def get_task_init_states(self, task_id: int):
            assert task_id == 0
            return [{"state": 0}]

    class FakeOffScreenRenderEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.step_count = 0
            self.closed = False

        def seed(self, seed):
            self.seed_value = seed

        def reset(self):
            self.step_count = 0

        def set_init_state(self, initial_state):
            self.initial_state = initial_state
            return _fake_libero_obs()

        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 1
            return _fake_libero_obs(), 0.0, done, {"action_dim": len(action)}

        def close(self):
            self.closed = True

    benchmark_mod.get_benchmark_dict = lambda: {"libero_10": FakeTaskSuite}
    libero_mod.get_libero_path = lambda name: str(tmp_path / name)
    envs_mod.OffScreenRenderEnv = FakeOffScreenRenderEnv
    root_pkg.libero = libero_mod
    libero_mod.benchmark = benchmark_mod
    libero_mod.envs = envs_mod

    monkeypatch.setitem(sys.modules, "libero", root_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_mod)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark_mod)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_mod)


def _fake_libero_obs() -> dict[str, object]:
    image = [
        [[0, 0, 0], [1, 1, 1]],
        [[2, 2, 2], [3, 3, 3]],
    ]
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image,
        "robot0_eef_pos": [0.0, 0.0, 0.0],
        "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
        "robot0_gripper_qpos": [0.0, 0.0],
    }


def _install_fake_robotwin(robotwin_root, monkeypatch) -> None:
    (robotwin_root / "script").mkdir(parents=True)
    (robotwin_root / "policy").mkdir()
    (robotwin_root / "task_config").mkdir()

    eval_policy_mod = types.ModuleType("script.eval_policy")
    observed_actions: list[list[float]] = []
    observed_test_nums: list[int] = []

    class FakeTaskEnv:
        def __init__(self) -> None:
            self.eval_success = False
            self.take_action_cnt = 0
            self.step_lim = 3
            self.closed = False

        def setup_demo(self, **kwargs):
            self.setup_kwargs = kwargs
            self.eval_success = False
            self.take_action_cnt = 0

        def play_once(self):
            return {"info": {"episode": 0}}

        def check_success(self):
            return True

        @property
        def plan_success(self):
            return True

        def close_env(self, clear_cache=False):  # noqa: FBT002
            self.closed = True

        def get_instruction(self):
            return "click the alarm clock"

        def set_instruction(self, instruction):
            self.instruction = instruction

        def get_obs(self):
            return _fake_robotwin_obs()

        def take_action(self, action, action_type):
            assert action_type == "qpos"
            observed_actions.append([float(value) for value in action])
            self.take_action_cnt += 1
            self.eval_success = True

    def fake_eval_policy(
        task_name,
        task_env,
        args,
        model,
        st_seed,
        test_num=100,
        video_size=None,
        instruction_type=None,
    ):
        assert task_name == "click_alarmclock"
        assert video_size is None
        assert instruction_type is None
        observed_test_nums.append(int(test_num))
        successes = 0
        policy_module = sys.modules[args["policy_name"]]
        for _ in range(int(test_num)):
            policy_module.reset_model(model)
            task_env.setup_demo()
            task_env.set_instruction("click the alarm clock")
            policy_module.eval(task_env, model, task_env.get_obs())
            successes += int(task_env.eval_success)
        assert observed_actions
        return st_seed + int(test_num), successes

    def fake_main(usr_args):
        from pathlib import Path

        assert Path.cwd() == robotwin_root
        policy_module = sys.modules[usr_args["policy_name"]]
        model = policy_module.get_model(usr_args)
        task_env = FakeTaskEnv()
        result = eval_policy_mod.eval_policy(
            "click_alarmclock",
            task_env,
            {"policy_name": usr_args["policy_name"]},
            model,
            0,
            test_num=100,
        )
        assert observed_test_nums == [int(usr_args["eval_num_episodes"])]
        return result

    eval_policy_mod.main = fake_main
    eval_policy_mod.eval_policy = fake_eval_policy

    monkeypatch.setitem(sys.modules, "script", types.ModuleType("script"))
    monkeypatch.setitem(sys.modules, "script.eval_policy", eval_policy_mod)


def _fake_robotwin_obs() -> dict[str, object]:
    image = [
        [[0, 0, 0], [1, 1, 1]],
        [[2, 2, 2], [3, 3, 3]],
    ]
    return {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image},
            "right_camera": {"rgb": image},
        },
        "joint_action": {"vector": [0.0] * 14},
    }
