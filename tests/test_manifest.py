from wam_harness.core.manifest import ManifestError, load_builtin_manifest, manifest_from_dict


def test_load_builtin_fake_manifest() -> None:
    manifest = load_builtin_manifest("fake-open-loop")

    assert manifest.id == "fake-open-loop"
    assert manifest.backend_name == "fake"
    assert manifest.workload_name == "open_loop"
    assert "fake_cache" in manifest.supported_optimizations


def test_fastwam_libero_manifest_matches_upstream_action_defaults() -> None:
    manifest = load_builtin_manifest("fastwam-libero")

    assert manifest.processor["action"]["horizon"] == 32
    assert manifest.processor["action"]["dim"] == 7
    assert manifest.defaults["action_horizon"] == 32
    assert manifest.defaults["replan_steps"] == 10


def test_dreamzero_manifest_matches_native_action_contract() -> None:
    manifest = load_builtin_manifest("dreamzero-droid-sim")

    assert manifest.processor["action"]["horizon"] == 24
    assert manifest.processor["action"]["dim"] == 8
    assert manifest.defaults["action_horizon"] == 24
    assert manifest.defaults["replan_steps"] == 24
    assert manifest.eval["defaults"]["server_startup_seconds"] == "1200"


def test_fastwam_libero_manifest_records_native_migration_status() -> None:
    manifest = load_builtin_manifest("fastwam-libero")

    assert manifest.deployment["reference_path"] == "official_script"
    assert manifest.deployment["product_path"] == "native_backend_migration"
    assert manifest.deployment["native_backend"] == "fastwam"
    assert manifest.deployment["native_stage"] == "native_smoke_verified"
    assert manifest.deployment["native_verified"] is True
    assert manifest.deployment["simulator_eval"] == "single_task_verified"
    assert manifest.deployment["parity_verified"] is False
    assert manifest.deployment["next_gate"] == "full_libero_eval"


def test_cosmos_policy_manifest_records_native_smoke_pass() -> None:
    manifest = load_builtin_manifest("cosmos-policy-libero")

    assert manifest.deployment["native_backend"] == "cosmos_policy"
    assert manifest.deployment["native_stage"] == "native_smoke_verified"
    assert manifest.deployment["native_verified"] is True
    assert manifest.deployment["parity_verified"] is False
    assert manifest.deployment["next_gate"] == "libero_eval"


def test_dreamzero_manifest_records_native_smoke_pass() -> None:
    manifest = load_builtin_manifest("dreamzero-droid-sim")

    assert manifest.deployment["native_backend"] == "dreamzero"
    assert manifest.deployment["native_stage"] == "native_smoke_verified"
    assert manifest.deployment["native_verified"] is True
    assert manifest.deployment["parity_verified"] is False
    assert manifest.deployment["next_gate"] == "droid_sim_eval"


def test_fastwam_libero_single_task_is_eval_workload_not_model_id() -> None:
    manifest = load_builtin_manifest("fastwam-libero")

    assert manifest.eval["default_workload"] == "libero-manager"
    assert "libero-manager" in manifest.eval["workloads"]
    assert "libero-single-task" in manifest.eval["workloads"]
    assert manifest.eval["workloads"]["libero-single-task"]["defaults"]["task_id"] == "0"


def test_manifest_requires_backend_name() -> None:
    data = load_builtin_manifest("fake-open-loop").to_dict()
    data["backend"] = {}

    try:
        manifest_from_dict(data)
    except ManifestError as exc:
        assert "backend" in str(exc)
    else:
        raise AssertionError("manifest missing backend name should fail")
