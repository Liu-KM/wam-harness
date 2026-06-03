import json

from wam_harness.cli import build_parser, main


def write_fastwam_required_paths(repo) -> None:
    for relative in [
        "src/fastwam/runtime.py",
        "src/fastwam/utils/config_resolvers.py",
        "src/fastwam/datasets/lerobot/robot_video_dataset.py",
        "src/fastwam/datasets/lerobot/utils/normalizer.py",
        "configs/sim_libero.yaml",
        "configs/train.yaml",
        "configs/task/libero_uncond_2cam224_1e-4.yaml",
        "configs/data/libero_2cam.yaml",
        "configs/model/fastwam.yaml",
    ]:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# smoke\n", encoding="utf-8")


def test_cli_list_shows_model_entries(capsys) -> None:
    exit_code = main(["list"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "MODEL ID" in captured.out
    assert "fake-open-loop" in captured.out
    assert "fastwam-libero" in captured.out


def test_cli_info_translates_model_entry(capsys) -> None:
    exit_code = main(["info", "fastwam-libero"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Model: fastwam-libero" in captured.out
    assert "Inputs: images=primary,wrist; state=proprio; prompt=task_suite" in captured.out
    assert "Deployment: product=native_backend_migration" in captured.out
    assert "native=fastwam (native_smoke_verified)" in captured.out
    assert "native_verified=true" in captured.out
    assert "Supported opts: action_chunk_scheduling" in captured.out


def test_cli_doctor_checks_fake_model_without_fixing_environment(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("WAM_CACHE_DIR", str(tmp_path / "cache"))

    exit_code = main(["doctor", "fake-open-loop"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "WAM doctor" in captured.out
    assert "Model: fake-open-loop" in captured.out
    assert "Runtime setup: not modified" in captured.out
    assert "Status: ok" in captured.out


def test_cli_doctor_reports_native_backend_requirements(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("WAM_FASTWAM_REPO", raising=False)

    exit_code = main(
        [
            "doctor",
            "fastwam-libero",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--upstream-dir",
            str(tmp_path / "missing-fastwam"),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Deployment: product=native_backend_migration" in captured.out
    assert "Native backend: fastwam (FastWAM)" in captured.out
    assert "Native runtime mode: in_process" in captured.out
    assert "Native runtime loader: fastwam_runtime_loader" in captured.out
    assert "Native model adapter: fastwam_model" in captured.out
    assert "Native readiness: blocked" in captured.out
    assert "Native required assets: checkpoint,dataset_stats" in captured.out
    assert (
        "Native runtime assets: checkpoint,dataset_stats,model_base,tokenizer_components"
        in captured.out
    )
    assert "Native missing required assets: checkpoint,dataset_stats" in captured.out
    assert "Native missing runtime assets: model_base,tokenizer_components" in captured.out
    assert str(tmp_path / "cache" / "diffsynth-models" / "Wan-AI" / "Wan2.2-TI2V-5B") in captured.out
    assert "Upstream repo: missing" in captured.out
    assert "Upstream env: WAM_FASTWAM_REPO" in captured.out
    assert "src/fastwam/runtime.py" in captured.out
    assert "Native next steps:" in captured.out
    assert "Set WAM_FASTWAM_REPO=<repo> or pass --upstream-dir <repo>" in captured.out
    assert (
        f"wam prepare fastwam-libero --cache-dir {tmp_path / 'cache'} --download"
        in captured.out
    )
    assert "Run inside the backend container or install native dependencies" in captured.out
    assert "Status: blocked" in captured.out


def test_cli_doctor_json_reports_native_preflight_gate(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("WAM_FASTWAM_REPO", raising=False)

    exit_code = main(
        [
            "doctor",
            "fastwam-libero",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--upstream-dir",
            str(tmp_path / "missing-fastwam"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "blocked"
    assert payload["model_id"] == "fastwam-libero"
    assert payload["native"]["declared"] is True
    assert payload["native"]["runtime_mode"] == "in_process"
    assert payload["native"]["runtime_loader"] == "fastwam_runtime_loader"
    assert payload["native"]["model_adapter"] == "fastwam_model"
    assert payload["native"]["status"] == "blocked"
    assert payload["native"]["upstream"]["status"] == "missing"
    assert payload["native"]["missing_required_assets"] == ["checkpoint", "dataset_stats"]
    assert payload["native"]["next_steps"][0].startswith(
        "Set WAM_FASTWAM_REPO=<repo> or pass --upstream-dir <repo>"
    )
    assert "wam prepare fastwam-libero" in payload["native"]["next_steps"][1]
    assert "Run inside the backend container" in payload["native"]["next_steps"][2]
    assert payload["assets"][2]["expected_path"].endswith(
        "diffsynth-models/Wan-AI/Wan2.2-TI2V-5B"
    )


def test_cli_doctor_strict_returns_nonzero_for_warning(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("WAM_FASTWAM_REPO", raising=False)

    exit_code = main(
        [
            "doctor",
            "fastwam-libero",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--upstream-dir",
            str(tmp_path / "missing-fastwam"),
            "--strict",
        ]
    )

    capsys.readouterr()

    assert exit_code == 1


def test_cli_doctor_strict_keeps_fake_model_zero(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("WAM_CACHE_DIR", str(tmp_path / "cache"))

    exit_code = main(["doctor", "fake-open-loop", "--strict", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"


def test_cli_run_reports_native_preflight_without_traceback(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("WAM_FASTWAM_REPO", raising=False)
    input_path = tmp_path / "obs.json"
    input_path.write_text(
        json.dumps(
            {
                "observation": {
                    "images": {"primary": [], "wrist": []},
                    "state": {"proprio": [0.0]},
                    "prompt": "open the drawer",
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "fastwam-libero",
            "--input",
            str(input_path),
            "--trace-dir",
            str(tmp_path),
            "--upstream-dir",
            str(tmp_path / "missing-fastwam"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: fastwam native readiness is blocked" in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_real_wam_without_input_prints_next_steps(capsys) -> None:
    exit_code = main(["run", "fastwam-libero"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "needs an observation input" in captured.err
    assert "wam run fastwam-libero --input obs.json --output action.json" in captured.err
    assert "wam eval fastwam-libero --workload libero-single-task" in captured.err
    assert "wam serve fastwam-libero" in captured.err
    assert "Traceback" not in captured.err


def test_cli_doctor_uses_cache_dir_for_native_readiness(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("WAM_FASTWAM_REPO", raising=False)
    repo = tmp_path / "FastWAM"
    write_fastwam_required_paths(repo)
    cache_dir = tmp_path / "cache"
    checkpoint = cache_dir / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt"
    dataset_stats = (
        cache_dir
        / "checkpoints"
        / "fastwam_release"
        / "libero_uncond_2cam224_dataset_stats.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    dataset_stats.write_text("{}", encoding="utf-8")

    exit_code = main(
        [
            "doctor",
            "fastwam-libero",
            "--cache-dir",
            str(cache_dir),
            "--upstream-dir",
            str(repo),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Deployment: product=native_backend_migration" in captured.out
    assert "Native readiness: blocked" in captured.out
    assert "Native missing required assets" not in captured.out
    assert "Native missing runtime assets: model_base,tokenizer_components" in captured.out
    assert "Native missing Python modules:" in captured.out
    assert "torch" in captured.out
    assert f"Upstream repo: present ({repo.resolve()})" in captured.out


def test_cli_prepare_fake_model_creates_cache_only(tmp_path, capsys) -> None:
    cache_dir = tmp_path / "cache"

    exit_code = main(["prepare", "fake-open-loop", "--cache-dir", str(cache_dir)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert cache_dir.exists()
    assert "Model: fake-open-loop" in captured.out
    assert "Runtime setup: not modified" in captured.out
    assert "Assets: none declared" in captured.out
    assert "Status: ok" in captured.out


def test_cli_prepare_real_model_reports_missing_assets(tmp_path, capsys) -> None:
    exit_code = main(["prepare", "fastwam-libero", "--cache-dir", str(tmp_path / "cache")])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Model: fastwam-libero" in captured.out
    assert "checkpoint: missing" in captured.out
    assert "Runtime setup: not modified" in captured.out
    assert "Status: incomplete" in captured.out


def test_cli_run_writes_summary_and_trace(tmp_path, capsys) -> None:
    exit_code = main(["run", "fake-open-loop", "--opt", "fake_cache", "--trace-dir", str(tmp_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["model_id"] == "fake-open-loop"
    assert payload["status"] == "ok"
    assert payload["runtime_info"]["optimization_profiles"][0]["name"] == "fake_cache"


def test_cli_run_writes_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "action.json"

    exit_code = main(["run", "fake-open-loop", "--trace-dir", str(tmp_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert printed == saved
    assert saved["model_id"] == "fake-open-loop"
    assert saved["result"]["action_chunk"]["shape"] == [3, 4]


def test_cli_run_accepts_native_backend_overrides() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "fastwam-libero",
            "--cache-dir",
            "/cache/wam",
            "--input",
            "/tmp/obs.json",
            "--output",
            "/tmp/action.json",
            "--upstream-dir",
            "/repo/FastWAM",
            "--backend-set",
            "task=libero",
        ]
    )

    assert args.command == "run"
    assert args.cache_dir == "/cache/wam"
    assert args.input == "/tmp/obs.json"
    assert args.output == "/tmp/action.json"
    assert args.upstream_dir == "/repo/FastWAM"
    assert args.backend_set == ["task=libero"]


def test_cli_serve_accepts_native_backend_overrides() -> None:
    args = build_parser().parse_args(
        [
            "serve",
            "fastwam-libero",
            "--cache-dir",
            "/cache/wam",
            "--upstream-dir",
            "/repo/FastWAM",
            "--backend-set",
            "task=libero",
            "--smoke",
        ]
    )

    assert args.command == "serve"
    assert args.cache_dir == "/cache/wam"
    assert args.upstream_dir == "/repo/FastWAM"
    assert args.backend_set == ["task=libero"]
    assert args.smoke is True


def test_cli_rejects_malformed_overrides_without_traceback(capsys) -> None:
    cases = [
        ["run", "fake-open-loop", "--backend-set", "missing_equals"],
        ["eval", "fastwam-libero", "--set", "missing_equals"],
        ["native-smoke", "fastwam-libero", "--backend-set", "missing_equals"],
        ["serve", "fake-open-loop", "--smoke", "--backend-set", "missing_equals"],
    ]

    for argv in cases:
        exit_code = main(argv)
        captured = capsys.readouterr()

        assert exit_code == 2
        assert "error: override must be KEY=VALUE: missing_equals" in captured.err
        assert "Traceback" not in captured.err


def test_cli_serve_smoke(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "serve",
            "fake-open-loop",
            "--opt",
            "fake_cache",
            "--trace-dir",
            str(tmp_path),
            "--smoke",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["health"]["runtime_info"]["manifest_id"] == "fake-open-loop"
    assert payload["health"]["trace_path"].endswith("trace.jsonl")
    assert payload["inference"]["action_chunk"]["shape"] == [3, 4]
