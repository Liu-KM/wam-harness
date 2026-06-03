import pytest

from wam_harness.core.manifest import load_builtin_manifest
from wam_harness.backends.native_support.runtime import (
    NATIVE_RUN_SPEC,
    NATIVE_SERVE_SPEC,
    NATIVE_SMOKE_SPEC,
    NativeRuntimeError,
    NativeRuntimeSpec,
    resolve_native_runtime,
)


def test_resolve_native_runtime_maps_reference_entry_to_native_backend(tmp_path) -> None:
    reference = load_builtin_manifest("fastwam-libero")

    plan = resolve_native_runtime(
        reference,
        NATIVE_SERVE_SPEC,
        upstream_dir=tmp_path / "FastWAM",
        cache_dir=tmp_path / "cache",
        backend_overrides={"task": "libero_10"},
    )

    assert plan.reference_manifest is reference
    assert plan.native_migration is True
    assert plan.native_backend == "fastwam"
    assert plan.mode == "native_serve"
    assert plan.workload_name == "serve"
    assert plan.manifest.backend_name == "fastwam"
    assert plan.manifest.backend["mode"] == "native_serve"
    assert plan.manifest.backend["config"]["upstream_dir"] == str(tmp_path / "FastWAM")
    assert plan.manifest.backend["config"]["cache_dir"] == str(tmp_path / "cache")
    assert plan.manifest.backend["config"]["task"] == "libero_10"
    assert "native_backend" not in plan.manifest.backend["config"]
    assert plan.manifest.workload_name == "serve"
    assert plan.manifest.workload["config"] == {"external_observation": True}


def test_resolve_native_runtime_falls_back_to_non_native_entry() -> None:
    manifest = load_builtin_manifest("fake-open-loop")

    plan = resolve_native_runtime(manifest, NATIVE_RUN_SPEC)

    assert plan.reference_manifest is manifest
    assert plan.manifest is manifest
    assert plan.native_migration is False
    assert plan.native_backend is None
    assert plan.mode == "fake"
    assert plan.workload_name == "open_loop"


def test_resolve_native_runtime_can_require_native_backend() -> None:
    manifest = load_builtin_manifest("fake-open-loop")

    with pytest.raises(NativeRuntimeError, match="does not declare backend.config.native_backend"):
        resolve_native_runtime(manifest, NATIVE_SMOKE_SPEC)


def test_native_runtime_spec_rejects_empty_mode_or_workload() -> None:
    with pytest.raises(NativeRuntimeError, match="mode"):
        resolve_native_runtime(
            load_builtin_manifest("fastwam-libero"),
            NativeRuntimeSpec(mode="", workload_name="serve"),
        )

    with pytest.raises(NativeRuntimeError, match="workload"):
        resolve_native_runtime(
            load_builtin_manifest("fastwam-libero"),
            NativeRuntimeSpec(mode="native_serve", workload_name=""),
        )
