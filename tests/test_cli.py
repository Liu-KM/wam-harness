import json
from contextlib import nullcontext

import pytest

from eazywam.backends.fastwam import FastWAMBackend, FastWAMModelAdapter
from eazywam.backends.native_support.runtime import native_runtime_resolver
from eazywam.cli import build_parser, main
from eazywam.core.registry import Registry
from eazywam.core.types import (
    ActionChunk,
    InferenceResult,
    Manifest,
    Observation,
    OptimizationProfile,
)


def test_cli_help_shows_product_examples(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "EazyWAM model workflow CLI." in captured.out
    assert "wam list" in captured.out
    assert "wam run fake-open-loop" in captured.out
    assert "Use `wam <command> --help`" in captured.out
    assert captured.err == ""


def test_cli_command_help_shows_command_options(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "usage: wam run" in captured.out
    assert "--input INPUT" in captured.out
    assert "--output OUTPUT" in captured.out
    assert "--opt OPT" in captured.out
    assert captured.err == ""


def write_fastwam_required_paths(repo) -> None:
    for relative in [
        "configs/sim_libero.yaml",
        "configs/sim_robotwin.yaml",
        "configs/train.yaml",
        "configs/task/libero_uncond_2cam224_1e-4.yaml",
        "configs/task/robotwin_uncond_3cam_384_1e-4.yaml",
        "configs/data/libero_2cam.yaml",
        "configs/data/robotwin.yaml",
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
    assert "fastwam-robotwin" in captured.out
    fastwam_line = next(line for line in captured.out.splitlines() if line.startswith("fastwam-libero"))
    assert "GPU container recommended (native: fastwam)" in fastwam_line
    assert "official_script" not in fastwam_line


def test_cli_info_translates_model_entry(capsys) -> None:
    exit_code = main(["info", "fastwam-libero"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Model: fastwam-libero" in captured.out
    assert "Inputs: images=primary,wrist; state=proprio; prompt=task_suite" in captured.out
    runtime_line = next(line for line in captured.out.splitlines() if line.startswith("Runtime:"))
    assert runtime_line == "Runtime: GPU container recommended (native: fastwam)"
    assert "official_script" not in runtime_line
    assert "Deployment: product=native_backend_migration" in captured.out
    assert "native=fastwam (single_task_eval_and_serve_verified)" in captured.out
    assert "next=statistical_native_reference_parity" in captured.out
    assert "native_verified=true" in captured.out
    assert "Supported opts: action_chunk_scheduling" in captured.out


def test_cli_info_translates_fastwam_robotwin_entry(capsys) -> None:
    exit_code = main(["info", "fastwam-robotwin"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Model: fastwam-robotwin" in captured.out
    assert "Task: RoboTwin robotwin2.0" in captured.out
    assert (
        "Inputs: images=head,left_wrist,right_wrist; state=joint_action; prompt=task_instruction"
        in captured.out
    )
    assert "Outputs: action chunks; horizon=32; dim=14" in captured.out
    assert "Runtime: GPU container recommended (native: fastwam)" in captured.out
    assert "official_script" not in captured.out


def test_cli_doctor_checks_fake_model_without_fixing_environment(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("WAM_CACHE_DIR", str(tmp_path / "cache"))

    exit_code = main(["doctor", "fake-open-loop"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "WAM doctor" in captured.out
    assert "Model: fake-open-loop" in captured.out
    assert "Runtime setup: not modified" in captured.out
    assert "Status: ok" in captured.out


def test_cli_doctor_reports_backend_requirements(tmp_path, monkeypatch, capsys) -> None:
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
    assert "Backend target: fastwam (FastWAM)" in captured.out
    assert "Backend runtime mode: in_process" in captured.out
    assert "Backend runtime loader: fastwam_runtime_loader" in captured.out
    assert "Backend model adapter: fastwam_model" in captured.out
    assert "Backend readiness: blocked" in captured.out
    assert "Backend required assets: checkpoint,dataset_stats" in captured.out
    assert "Backend runtime assets: checkpoint,dataset_stats,wan22_vae" in captured.out
    assert "Backend missing required assets: checkpoint,dataset_stats" in captured.out
    assert "Backend missing runtime assets: wan22_vae,wan22_t5_encoder" in captured.out
    assert (
        str(
            tmp_path
            / "cache"
            / "diffsynth-models"
            / "Wan-AI"
            / "Wan2.2-TI2V-5B"
            / "Wan2.2_VAE.pth"
        )
        in captured.out
    )
    assert "Upstream repo: missing" in captured.out
    assert "Upstream env: WAM_FASTWAM_REPO" in captured.out
    assert "src/fastwam/runtime.py" not in captured.out
    assert "Backend next steps:" in captured.out
    assert "Set WAM_FASTWAM_REPO=<repo> or pass --upstream-dir <repo>" in captured.out
    assert (
        f"wam prepare fastwam-libero --cache-dir {tmp_path / 'cache'} --download"
        in captured.out
    )
    assert "Run inside the backend container or install backend dependencies" in captured.out
    assert "Status: blocked" in captured.out


def test_cli_doctor_uses_vendored_fastwam_runtime_by_default(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("WAM_FASTWAM_REPO", raising=False)

    exit_code = main(
        [
            "doctor",
            "fastwam-libero",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Backend target: fastwam (FastWAM)" in captured.out
    assert "Upstream repo: present" in captured.out
    assert "src/fastwam/configs" in captured.out
    assert "Upstream selected commit: vendored:45d8e1458921d83f8ad6cf9ce993d371208dabd0" in captured.out
    assert "Upstream commit status: vendored" in captured.out
    assert "Set WAM_FASTWAM_REPO=<repo> or pass --upstream-dir <repo>" not in captured.out
    assert "Backend missing required assets: checkpoint,dataset_stats" in captured.out
    assert "Status: blocked" in captured.out


def test_cli_doctor_json_reports_preflight_gate(tmp_path, monkeypatch, capsys) -> None:
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
    assert payload["backend"]["declared"] is True
    assert payload["backend"]["runtime_mode"] == "in_process"
    assert payload["backend"]["runtime_loader"] == "fastwam_runtime_loader"
    assert payload["backend"]["model_adapter"] == "fastwam_model"
    assert payload["backend"]["status"] == "blocked"
    assert payload["backend"]["upstream"]["status"] == "missing"
    assert payload["backend"]["missing_required_assets"] == ["checkpoint", "dataset_stats"]
    assert payload["backend"]["next_steps"][0].startswith(
        "Set WAM_FASTWAM_REPO=<repo> or pass --upstream-dir <repo>"
    )
    assert "wam prepare fastwam-libero" in payload["backend"]["next_steps"][1]
    assert "Run inside the backend container" in payload["backend"]["next_steps"][2]
    assert payload["assets"][2]["expected_path"].endswith(
        "diffsynth-models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    )


def test_cli_prepare_unknown_asset_prints_clean_error(capsys) -> None:
    exit_code = main(["prepare", "fastwam-libero", "--asset", "typo"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: unknown asset(s) or asset group(s)")
    assert "Traceback" not in captured.err


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


def test_cli_run_reports_preflight_without_traceback(tmp_path, monkeypatch, capsys) -> None:
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
    assert "error: fastwam preflight is blocked" in captured.err
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


def test_cli_doctor_uses_cache_dir_for_backend_readiness(tmp_path, monkeypatch, capsys) -> None:
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
    assert "Backend readiness: blocked" in captured.out
    assert "Backend missing required assets" not in captured.out
    assert "Backend missing runtime assets: wan22_vae,wan22_t5_encoder" in captured.out
    assert "Backend missing Python modules:" in captured.out
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
            "--smoke-input",
            "/tmp/obs.json",
        ]
    )

    assert args.command == "serve"
    assert args.cache_dir == "/cache/wam"
    assert args.upstream_dir == "/repo/FastWAM"
    assert args.backend_set == ["task=libero"]
    assert args.smoke is True
    assert args.smoke_input == "/tmp/obs.json"


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


def test_cli_serve_smoke_posts_input_observation(tmp_path, capsys) -> None:
    input_path = tmp_path / "obs.json"
    input_path.write_text(
        json.dumps(
            {
                "observation": {
                    "images": {"primary": [[[1, 2, 3]]]},
                    "prompt": "serve this observation",
                    "session": {"episode_id": 2, "step_id": 3},
                },
                "action_horizon": 2,
                "replan_steps": 1,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "serve",
            "fake-open-loop",
            "--trace-dir",
            str(tmp_path / "runs"),
            "--smoke",
            "--smoke-input",
            str(input_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["inference"]["action_chunk"]["shape"] == [2, 4]
    assert payload["inference"]["action_chunk"]["actions"][0] == [2.3, 2.301, 2.302, 2.303]


def test_cli_serve_smoke_input_requires_observation(tmp_path, capsys) -> None:
    input_path = tmp_path / "bad.json"
    input_path.write_text(json.dumps({"action_horizon": 2}), encoding="utf-8")

    exit_code = main(
        [
            "serve",
            "fake-open-loop",
            "--smoke",
            "--smoke-input",
            str(input_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "error: smoke input JSON must contain observation.images" in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_fastwam_native_product_path_writes_action_output(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    registry, _created = _fastwam_product_registry(tmp_path)
    _patch_default_registry(monkeypatch, registry)
    input_path = _write_fastwam_input(tmp_path)
    output_path = tmp_path / "action.json"

    exit_code = main(
        [
            "run",
            "fastwam-libero",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--upstream-dir",
            str(tmp_path / "FastWAM"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--trace-dir",
            str(tmp_path / "runs"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    printed = json.loads(captured.out)

    assert exit_code == 0
    assert payload == printed
    assert payload["runtime_info"]["backend"] == "fastwam"
    assert payload["runtime_info"]["mode"] == "run"
    assert payload["result"]["action_chunk"]["shape"] == [32, 7]
    assert payload["result"]["backend_metadata"]["model_adapter"] == "fastwam_model"
    assert payload["result"]["backend_metadata"]["fastwam_call"] == "infer_action"

    trace_path = tmp_path / payload["trace_path"]
    events = _read_events(trace_path)
    names = [event["event"] for event in events]
    assert "runtime_contract" in names
    assert "preflight" in names
    assert "backend_load" in names
    inference_end = [event for event in events if event["event"] == "inference_end"][0]
    assert inference_end["action_chunk_shape"] == [32, 7]
    assert inference_end["action_contract"]["status"] == "ok"


def test_cli_run_fastwam_robotwin_native_product_path_writes_action_output(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    registry, _created = _fastwam_product_registry(tmp_path)
    _patch_default_registry(monkeypatch, registry)
    input_path = _write_fastwam_robotwin_input(tmp_path)
    output_path = tmp_path / "robotwin-action.json"

    exit_code = main(
        [
            "run",
            "fastwam-robotwin",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--upstream-dir",
            str(tmp_path / "FastWAM"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--trace-dir",
            str(tmp_path / "runs"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    printed = json.loads(captured.out)

    assert exit_code == 0
    assert payload == printed
    assert payload["runtime_info"]["backend"] == "fastwam"
    assert payload["runtime_info"]["mode"] == "run"
    assert payload["result"]["action_chunk"]["shape"] == [32, 14]
    assert payload["result"]["backend_metadata"]["model_adapter"] == "fastwam_model"
    assert payload["result"]["backend_metadata"]["fastwam_call"] == "infer_action"

    trace_path = tmp_path / payload["trace_path"]
    events = _read_events(trace_path)
    inference_end = [event for event in events if event["event"] == "inference_end"][0]
    assert inference_end["action_chunk_shape"] == [32, 14]
    assert inference_end["action_contract"]["status"] == "ok"


def test_cli_serve_fastwam_native_smoke_input_returns_action(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    registry, _created = _fastwam_product_registry(tmp_path)
    _patch_default_registry(monkeypatch, registry)
    input_path = _write_fastwam_input(tmp_path)

    exit_code = main(
        [
            "serve",
            "fastwam-libero",
            "--smoke",
            "--smoke-input",
            str(input_path),
            "--upstream-dir",
            str(tmp_path / "FastWAM"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--trace-dir",
            str(tmp_path / "serve-runs"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["health"]["runtime_info"]["backend"] == "fastwam"
    assert payload["health"]["runtime_info"]["mode"] == "serve"
    assert payload["inference"]["action_chunk"]["shape"] == [32, 7]
    assert payload["inference"]["backend_metadata"]["model_adapter"] == "fastwam_model"


def test_cli_serve_fastwam_robotwin_native_smoke_input_returns_action(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    registry, _created = _fastwam_product_registry(tmp_path)
    _patch_default_registry(monkeypatch, registry)
    input_path = _write_fastwam_robotwin_input(tmp_path)

    exit_code = main(
        [
            "serve",
            "fastwam-robotwin",
            "--smoke",
            "--smoke-input",
            str(input_path),
            "--upstream-dir",
            str(tmp_path / "FastWAM"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--trace-dir",
            str(tmp_path / "serve-runs"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["health"]["runtime_info"]["backend"] == "fastwam"
    assert payload["health"]["runtime_info"]["mode"] == "serve"
    assert payload["inference"]["action_chunk"]["shape"] == [32, 14]
    assert payload["inference"]["backend_metadata"]["model_adapter"] == "fastwam_model"


def _fastwam_product_registry(tmp_path):
    registry = Registry()
    registry.register_runtime_resolver(native_runtime_resolver)
    created: list[_LightFastWAMBackend] = []
    repo = tmp_path / "FastWAM"
    cache = tmp_path / "cache"
    write_fastwam_required_paths(repo)
    checkpoint = cache / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt"
    dataset_stats = (
        cache
        / "checkpoints"
        / "fastwam_release"
        / "libero_uncond_2cam224_dataset_stats.json"
    )
    robotwin_checkpoint = (
        cache / "checkpoints" / "fastwam_release" / "robotwin_uncond_3cam_384.pt"
    )
    robotwin_dataset_stats = (
        cache
        / "checkpoints"
        / "fastwam_release"
        / "robotwin_uncond_3cam_384_dataset_stats.json"
    )
    checkpoint.parent.mkdir(parents=True)
    for path in (checkpoint, robotwin_checkpoint):
        path.write_bytes(b"checkpoint")
    for path in (dataset_stats, robotwin_dataset_stats):
        path.write_text("{}", encoding="utf-8")

    def factory(manifest: Manifest, profiles: list[OptimizationProfile]) -> _LightFastWAMBackend:
        backend = _LightFastWAMBackend(manifest, profiles)
        created.append(backend)
        return backend

    registry.register_backend("fastwam", factory)
    registry.register_processor(
        "fastwam_libero",
        lambda manifest: _FastWAMContractProcessor(
            processor="fastwam_libero",
            images=["primary", "wrist"],
            state="proprio",
            prompt="task_suite",
            action_dim=7,
        ),
    )
    registry.register_processor(
        "fastwam_robotwin",
        lambda manifest: _FastWAMContractProcessor(
            processor="fastwam_robotwin",
            images=["head", "left_wrist", "right_wrist"],
            state="joint_action",
            prompt="task_instruction",
            action_dim=14,
        ),
    )
    return registry, created


def _patch_default_registry(monkeypatch, registry: Registry) -> None:
    import eazywam.defaults as defaults

    monkeypatch.setattr(defaults, "default_registry", lambda: registry)


def _write_fastwam_input(tmp_path):
    input_path = tmp_path / "obs.json"
    input_path.write_text(
        json.dumps(
            {
                "observation": {
                    "images": {
                        "primary": [[[1, 2, 3]]],
                        "wrist": [[[4, 5, 6]]],
                    },
                    "prompt": "open the drawer",
                    "state": {
                        "robot0_eef_pos": [0.0, 0.0, 0.0],
                        "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
                        "robot0_gripper_qpos": [0.0, 0.0],
                    },
                    "session": {"episode_id": 0, "step_id": 0},
                },
                "action_horizon": 32,
                "replan_steps": 10,
            }
        ),
        encoding="utf-8",
    )
    return input_path


def _write_fastwam_robotwin_input(tmp_path):
    input_path = tmp_path / "robotwin-obs.json"
    input_path.write_text(
        json.dumps(
            {
                "observation": {
                    "images": {
                        "head": [[[1, 2, 3]]],
                        "left_wrist": [[[4, 5, 6]]],
                        "right_wrist": [[[7, 8, 9]]],
                    },
                    "prompt": "click the alarm clock",
                    "state": {
                        "joint_action": {
                            "vector": [0.0] * 14,
                        },
                    },
                    "session": {"episode_id": 0, "step_id": 0},
                },
                "action_horizon": 32,
                "replan_steps": 24,
            }
        ),
        encoding="utf-8",
    )
    return input_path


def _read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _LightFastWAMBackend(FastWAMBackend):
    required_python_modules = ()
    runtime_asset_names = ("checkpoint", "dataset_stats")

    def load(self) -> None:
        repo = self.resolve_upstream_repo()
        self.checkpoint_path = self.resolve_required_asset("checkpoint")
        self.dataset_stats_path = self.resolve_required_asset("dataset_stats")
        self.processor = _FastWAMProductProcessor()
        self.model = _FastWAMActionModel(action_dim=_manifest_action_dim(self.manifest))
        self.cfg = {"EVALUATION": {"num_inference_steps": 1}}
        self.model_adapter = FastWAMModelAdapter(
            model=self.model,
            cfg=self.cfg,
            checkpoint_path=self.checkpoint_path,
            dataset_stats_path=self.dataset_stats_path,
            config=dict(self.config),
            dit_cache_params=self.profile_settings("dit_cache"),
            cuda_graph_params=self.profile_settings("cuda_graph"),
            cuda_graph_enabled=self.profile_enabled("cuda_graph"),
            torch_compile_params=self.profile_settings("torch_compile"),
            torch_compile_enabled=self.profile_enabled("torch_compile"),
            no_grad_factory=lambda: nullcontext(),
            error_cls=self.error_cls,
        )
        self.device = "cpu"
        self.upstream_repo = repo
        self.loaded = True


class _FastWAMActionModel:
    def __init__(self, action_dim: int = 7) -> None:
        self.action_dim = action_dim

    def infer_action(self, *, action_horizon, **kwargs):
        cuda_graph_mode = str(kwargs.get("cuda_graph_mode", "off"))
        torch_compile_mode = str(kwargs.get("torch_compile_mode", "off"))
        return {
            "action": [
                [float(col) for col in range(self.action_dim)]
                for _ in range(int(action_horizon))
            ],
            "metadata": {
                "cuda_graph_enabled": cuda_graph_mode != "off",
                "cuda_graph_mode": cuda_graph_mode,
                "cuda_graph_hook": "fastwam_cuda_graph_action_body",
                "torch_compile_enabled": torch_compile_mode != "off",
                "torch_compile_mode": torch_compile_mode,
                "torch_compile_hook": "fastwam_torch_compile_action_body",
            },
        }


class _FastWAMProductProcessor:
    def to_model_inputs(self, observation: Observation):
        return {
            "prompt": f"prompt: {observation.prompt}",
            "input_image": "image",
            "proprio": "proprio",
        }

    def to_harness_result(self, raw_output):
        return InferenceResult(
            action_chunk=ActionChunk(actions=raw_output["action"]),
            backend_metadata={"raw_keys": sorted(raw_output)},
        )


class _FastWAMContractProcessor(_FastWAMProductProcessor):
    def __init__(
        self,
        *,
        processor: str,
        images: list[str],
        state: str,
        prompt: str,
        action_dim: int,
    ) -> None:
        self.processor = processor
        self.images = images
        self.state = state
        self.prompt = prompt
        self.action_dim = action_dim

    def modality_limits(self):
        return {
            "processor": self.processor,
            "images": self.images,
            "state": self.state,
            "prompt": self.prompt,
            "action_dim": self.action_dim,
        }

    def smoke_observation(self):
        if self.processor == "fastwam_robotwin":
            return Observation(
                images={
                    "head": [[[1, 2, 3]]],
                    "left_wrist": [[[4, 5, 6]]],
                    "right_wrist": [[[7, 8, 9]]],
                },
                prompt="click the alarm clock",
                state={"joint_action": {"vector": [0.0] * 14}},
            )
        return Observation(
            images={"primary": [[[1, 2, 3]]], "wrist": [[[4, 5, 6]]]},
            prompt="open the drawer",
        )


def _manifest_action_dim(manifest: Manifest) -> int:
    action = manifest.processor.get("action", {})
    if isinstance(action, dict) and action.get("dim") is not None:
        return int(action["dim"])
    return 7
