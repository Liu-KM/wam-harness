import json
from pathlib import Path

import yaml

from eazywam.compat import fastwam_robotwin_manager


def test_robotwin_manager_writes_incremental_summary(monkeypatch, tmp_path) -> None:
    upstream_dir = tmp_path / "FastWAM"
    robotwin_root = tmp_path / "RoboTwin"
    output_dir = tmp_path / "out"
    (robotwin_root / "task_config").mkdir(parents=True)
    (robotwin_root / "task_config" / "_eval_step_limit.yml").write_text(
        yaml.safe_dump({"click_alarmclock": 400}),
        encoding="utf-8",
    )

    calls: list[str] = []
    saw_incremental_summary = False

    def fake_run_streaming(cmd: list[str], *, cwd: Path) -> int:
        nonlocal saw_incremental_summary
        assert cwd == upstream_dir
        task_name = _arg_value(cmd, "EVALUATION.task_name=")
        task_config = _arg_value(cmd, "EVALUATION.task_config=")
        calls.append(task_config)
        if task_config == "demo_randomized":
            payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            saw_incremental_summary = (
                payload["per_task"][0]["clean_success_rate"] == 1.0
                and payload["per_task"][0]["random_success_rate"] is None
            )

        result_dir = (
            robotwin_root
            / "eval_result"
            / task_name
            / "fastwam_policy"
            / task_config
            / "fake_ckpt"
            / f"run_{len(calls)}"
        )
        result_dir.mkdir(parents=True)
        (result_dir / "_result.txt").write_text("1.0\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(fastwam_robotwin_manager, "_run_streaming", fake_run_streaming)

    code = fastwam_robotwin_manager.main(
        [
            "--upstream-dir",
            str(upstream_dir),
            "--robotwin-root",
            str(robotwin_root),
            "--output-dir",
            str(output_dir),
            "--task",
            "robotwin_uncond_3cam_384_1e-4",
            "--ckpt",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-stats-path",
            str(tmp_path / "stats.json"),
            "--task-name",
            "click_alarmclock",
            "--num-episodes",
            "1",
        ]
    )

    assert code == 0
    assert calls == ["demo_clean", "demo_randomized"]
    assert saw_incremental_summary is True
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["overall"]["clean_mean_success_rate"] == 1.0
    assert payload["overall"]["random_mean_success_rate"] == 1.0
    assert payload["requested"]["target_valid_episodes"] == 2
    assert payload["actual"]["valid_episodes"] == 2
    assert payload["actual"]["candidate_episodes_attempted"] == 2
    assert payload["actual"]["invalid_candidate_episodes"] == 0
    assert payload["actual"]["invalid_setup_count"] == 0
    assert len(payload["candidate_episodes"]) == 2
    assert payload["invalid_setups"] == []


def test_robotwin_manager_resume_skips_completed_phase(monkeypatch, tmp_path) -> None:
    upstream_dir = tmp_path / "FastWAM"
    robotwin_root = tmp_path / "RoboTwin"
    output_dir = tmp_path / "out"
    (robotwin_root / "task_config").mkdir(parents=True)
    (robotwin_root / "task_config" / "_eval_step_limit.yml").write_text(
        yaml.safe_dump({"click_alarmclock": 400}),
        encoding="utf-8",
    )
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "per_task": [
                    {
                        "task_name": "click_alarmclock",
                        "clean_success_rate": 1.0,
                        "random_success_rate": None,
                    }
                ],
                "overall": {
                    "clean_mean_success_rate": 1.0,
                    "random_mean_success_rate": None,
                },
                "invalid_setups": [
                    {
                        "task_name": "click_alarmclock",
                        "phase": "clean",
                        "reason": "robotwin_unstable_setup",
                        "policy_failure": False,
                    }
                ],
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_run_streaming(cmd: list[str], *, cwd: Path) -> int:
        assert cwd == upstream_dir
        task_name = _arg_value(cmd, "EVALUATION.task_name=")
        task_config = _arg_value(cmd, "EVALUATION.task_config=")
        calls.append(task_config)
        result_dir = (
            robotwin_root
            / "eval_result"
            / task_name
            / "fastwam_policy"
            / task_config
            / "fake_ckpt"
            / "run_1"
        )
        result_dir.mkdir(parents=True)
        (result_dir / "_result.txt").write_text("0.0\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(fastwam_robotwin_manager, "_run_streaming", fake_run_streaming)

    code = fastwam_robotwin_manager.main(
        [
            "--upstream-dir",
            str(upstream_dir),
            "--robotwin-root",
            str(robotwin_root),
            "--output-dir",
            str(output_dir),
            "--task",
            "robotwin_uncond_3cam_384_1e-4",
            "--ckpt",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-stats-path",
            str(tmp_path / "stats.json"),
            "--task-name",
            "click_alarmclock",
            "--num-episodes",
            "1",
            "--resume",
            "True",
        ]
    )

    assert code == 0
    assert calls == ["demo_randomized"]
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["per_task"][0]["clean_success_rate"] == 1.0
    assert payload["per_task"][0]["random_success_rate"] == 0.0
    assert payload["actual"]["invalid_setup_count"] == 1
    assert payload["actual"]["invalid_candidate_episodes"] == 1
    assert payload["invalid_setups"][0]["reason"] == "robotwin_unstable_setup"


def test_robotwin_manager_resume_bootstraps_from_robotwin_results(
    monkeypatch,
    tmp_path,
) -> None:
    upstream_dir = tmp_path / "FastWAM"
    robotwin_root = tmp_path / "RoboTwin"
    output_dir = tmp_path / "out"
    (robotwin_root / "task_config").mkdir(parents=True)
    (robotwin_root / "task_config" / "_eval_step_limit.yml").write_text(
        yaml.safe_dump({"click_alarmclock": 400}),
        encoding="utf-8",
    )
    existing_result = (
        robotwin_root
        / "eval_result"
        / "click_alarmclock"
        / "fastwam_policy"
        / "demo_clean"
        / "fake_ckpt"
        / "old_run"
    )
    existing_result.mkdir(parents=True)
    (existing_result / "_result.txt").write_text("1.0\n", encoding="utf-8")
    calls: list[str] = []

    def fake_run_streaming(cmd: list[str], *, cwd: Path) -> int:
        assert cwd == upstream_dir
        task_name = _arg_value(cmd, "EVALUATION.task_name=")
        task_config = _arg_value(cmd, "EVALUATION.task_config=")
        calls.append(task_config)
        result_dir = (
            robotwin_root
            / "eval_result"
            / task_name
            / "fastwam_policy"
            / task_config
            / "fake_ckpt"
            / "new_run"
        )
        result_dir.mkdir(parents=True)
        (result_dir / "_result.txt").write_text("0.0\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(fastwam_robotwin_manager, "_run_streaming", fake_run_streaming)

    code = fastwam_robotwin_manager.main(
        [
            "--upstream-dir",
            str(upstream_dir),
            "--robotwin-root",
            str(robotwin_root),
            "--output-dir",
            str(output_dir),
            "--task",
            "robotwin_uncond_3cam_384_1e-4",
            "--ckpt",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-stats-path",
            str(tmp_path / "stats.json"),
            "--task-name",
            "click_alarmclock",
            "--num-episodes",
            "1",
            "--resume",
            "True",
        ]
    )

    assert code == 0
    assert calls == ["demo_randomized"]
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["per_task"][0]["clean_success_rate"] == 1.0
    assert payload["per_task"][0]["random_success_rate"] == 0.0


def test_robotwin_manager_retries_invalid_setup_without_policy_failure(
    monkeypatch,
    tmp_path,
) -> None:
    upstream_dir = tmp_path / "FastWAM"
    robotwin_root = tmp_path / "RoboTwin"
    output_dir = tmp_path / "out"
    (robotwin_root / "task_config").mkdir(parents=True)
    (robotwin_root / "task_config" / "_eval_step_limit.yml").write_text(
        yaml.safe_dump({"put_bottles_dustbin": 400}),
        encoding="utf-8",
    )

    calls: list[tuple[str, str]] = []

    def fake_run_streaming(cmd: list[str], *, cwd: Path):
        assert cwd == upstream_dir
        task_name = _arg_value(cmd, "EVALUATION.task_name=")
        task_config = _arg_value(cmd, "EVALUATION.task_config=")
        seed = _arg_value(cmd, "seed=")
        calls.append((task_config, seed))
        if len(calls) == 1:
            return fastwam_robotwin_manager.RunResult(
                return_code=1,
                output_tail=(
                    "Traceback\n"
                    "File \"envs/tasks/put_bottles_dustbin.py\", line 1, in play_once\n"
                    "left_action[1][0]\n"
                    "IndexError: list index out of range\n"
                ),
            )

        result_dir = (
            robotwin_root
            / "eval_result"
            / task_name
            / "fastwam_policy"
            / task_config
            / "fake_ckpt"
            / f"run_{len(calls)}"
        )
        result_dir.mkdir(parents=True)
        (result_dir / "_result.txt").write_text("1.0\n", encoding="utf-8")
        return fastwam_robotwin_manager.RunResult(return_code=0, output_tail="ok\n")

    monkeypatch.setattr(fastwam_robotwin_manager, "_run_streaming", fake_run_streaming)

    code = fastwam_robotwin_manager.main(
        [
            "--upstream-dir",
            str(upstream_dir),
            "--robotwin-root",
            str(robotwin_root),
            "--output-dir",
            str(output_dir),
            "--task",
            "robotwin_uncond_3cam_384_1e-4",
            "--ckpt",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-stats-path",
            str(tmp_path / "stats.json"),
            "--task-name",
            "put_bottles_dustbin",
            "--num-episodes",
            "1",
            "--max-invalid-setup-retries",
            "1",
        ]
    )

    assert code == 0
    assert calls == [("demo_clean", "42"), ("demo_clean", "43"), ("demo_randomized", "42")]
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["requested"]["target_valid_episodes"] == 2
    assert payload["actual"]["valid_episodes"] == 2
    assert payload["actual"]["candidate_episodes_attempted"] == 3
    assert payload["actual"]["invalid_candidate_episodes"] == 1
    assert payload["actual"]["policy_successes"] == 2
    assert payload["actual"]["invalid_setup_count"] == 1
    assert payload["failures"] == []
    assert payload["invalid_setups"][0]["category"] == "simulator_setup_invalid"
    assert payload["invalid_setups"][0]["policy_failure"] is False
    assert (
        payload["invalid_setups"][0]["reason"]
        == "put_bottles_dustbin_expert_setup_index_error"
    )


def test_robotwin_manager_records_worker_internal_invalid_setup(
    monkeypatch,
    tmp_path,
) -> None:
    upstream_dir = tmp_path / "FastWAM"
    robotwin_root = tmp_path / "RoboTwin"
    output_dir = tmp_path / "out"
    (robotwin_root / "task_config").mkdir(parents=True)
    (robotwin_root / "task_config" / "_eval_step_limit.yml").write_text(
        yaml.safe_dump({"put_bottles_dustbin": 400}),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_run_streaming(cmd: list[str], *, cwd: Path):
        assert cwd == upstream_dir
        task_name = _arg_value(cmd, "EVALUATION.task_name=")
        task_config = _arg_value(cmd, "EVALUATION.task_config=")
        seed = _arg_value(cmd, "seed=")
        calls.append(seed)
        result_dir = (
            robotwin_root
            / "eval_result"
            / task_name
            / "fastwam_policy"
            / task_config
            / "fake_ckpt"
            / f"run_{len(calls)}"
        )
        result_dir.mkdir(parents=True)
        (result_dir / "_result.txt").write_text("1.0\n", encoding="utf-8")
        output_tail = "ok\n"
        if task_config == "demo_clean":
            output_tail = (
                "Error: list index out of range\n"
                "Stack Trace: File \"envs/put_bottles_dustbin.py\", line 73, in play_once\n"
                "left_action[1][0]\n"
                "IndexError: list index out of range\n"
                "error occurs !\n"
                "Success rate: 1/1, current seed: 4300001\n"
            )
        return fastwam_robotwin_manager.RunResult(return_code=0, output_tail=output_tail)

    monkeypatch.setattr(fastwam_robotwin_manager, "_run_streaming", fake_run_streaming)

    code = fastwam_robotwin_manager.main(
        [
            "--upstream-dir",
            str(upstream_dir),
            "--robotwin-root",
            str(robotwin_root),
            "--output-dir",
            str(output_dir),
            "--task",
            "robotwin_uncond_3cam_384_1e-4",
            "--ckpt",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-stats-path",
            str(tmp_path / "stats.json"),
            "--task-name",
            "put_bottles_dustbin",
            "--num-episodes",
            "1",
        ]
    )

    assert code == 0
    assert calls == ["42", "42"]
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["actual"]["valid_episodes"] == 2
    assert payload["actual"]["candidate_episodes_attempted"] == 3
    assert payload["actual"]["invalid_candidate_episodes"] == 1
    assert payload["actual"]["invalid_setup_count"] == 1
    assert payload["candidate_episodes"][0]["candidate_episodes_attempted"] == 2
    assert payload["candidate_episodes"][0]["invalid_candidate_episodes"] == 1
    assert payload["candidate_episodes"][0]["internal_seed_start"] == 4300000
    assert payload["candidate_episodes"][0]["last_valid_internal_seed"] == 4300001
    assert payload["invalid_setups"][0]["source"] == "worker_output"
    assert payload["invalid_setups"][0]["policy_failure"] is False


def test_robotwin_manager_reports_invalid_setup_exhaustion_as_structural(
    monkeypatch,
    tmp_path,
) -> None:
    upstream_dir = tmp_path / "FastWAM"
    robotwin_root = tmp_path / "RoboTwin"
    output_dir = tmp_path / "out"
    (robotwin_root / "task_config").mkdir(parents=True)
    (robotwin_root / "task_config" / "_eval_step_limit.yml").write_text(
        yaml.safe_dump({"put_bottles_dustbin": 400}),
        encoding="utf-8",
    )

    def fake_run_streaming(cmd: list[str], *, cwd: Path):
        assert cwd == upstream_dir
        return fastwam_robotwin_manager.RunResult(
            return_code=1,
            output_tail=(
                "Traceback\n"
                "File \"envs/tasks/put_bottles_dustbin.py\", line 1, in play_once\n"
                "right_action[1][0]\n"
                "IndexError: list index out of range\n"
            ),
        )

    monkeypatch.setattr(fastwam_robotwin_manager, "_run_streaming", fake_run_streaming)

    code = fastwam_robotwin_manager.main(
        [
            "--upstream-dir",
            str(upstream_dir),
            "--robotwin-root",
            str(robotwin_root),
            "--output-dir",
            str(output_dir),
            "--task",
            "robotwin_uncond_3cam_384_1e-4",
            "--ckpt",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-stats-path",
            str(tmp_path / "stats.json"),
            "--task-name",
            "put_bottles_dustbin",
            "--num-episodes",
            "1",
            "--max-invalid-setup-retries",
            "0",
        ]
    )

    assert code == 1
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["requested"]["target_valid_episodes"] == 2
    assert payload["actual"]["valid_episodes"] == 0
    assert payload["actual"]["candidate_episodes_attempted"] == 1
    assert payload["actual"]["invalid_candidate_episodes"] == 1
    assert payload["actual"]["invalid_setup_count"] == 1
    assert payload["invalid_setups"][0]["policy_failure"] is False
    assert payload["failures"][0]["category"] == "simulator_setup_invalid"
    assert payload["failures"][0]["policy_failure"] is False
    assert payload["failures"][0]["reason"] == "invalid_setup_exhausted"


def _arg_value(argv: list[str], prefix: str) -> str:
    for arg in argv:
        if arg.startswith(prefix):
            return arg.removeprefix(prefix)
    raise AssertionError(f"missing argv prefix: {prefix}")
