import pytest

from wam_harness.core.manifest import load_builtin_manifest
from wam_harness.core.registry import default_registry
from wam_harness.core.runtime import RuntimeResolutionError, RuntimeSpec, SERVE_SPEC


def test_registry_runtime_resolver_maps_reference_entry_to_native_backend(tmp_path) -> None:
    reference = load_builtin_manifest("fastwam-libero")
    registry = default_registry()

    plan = registry.resolve_runtime(
        reference,
        SERVE_SPEC,
        upstream_dir=tmp_path / "FastWAM",
        cache_dir=tmp_path / "cache",
        backend_overrides={"task": "libero_10"},
    )

    assert plan.reference_manifest is reference
    assert plan.transformed is True
    assert plan.mapped_backend == "fastwam"
    assert plan.mode == "serve"
    assert plan.workload_name == "serve"
    assert plan.manifest.backend_name == "fastwam"
    assert plan.manifest.backend["mode"] == "serve"
    assert plan.manifest.backend["config"]["upstream_dir"] == str(tmp_path / "FastWAM")
    assert plan.manifest.backend["config"]["cache_dir"] == str(tmp_path / "cache")
    assert plan.manifest.backend["config"]["task"] == "libero_10"
    assert "native_backend" not in plan.manifest.backend["config"]
    assert plan.manifest.workload_name == "serve"
    assert plan.manifest.workload["config"] == {"external_observation": True}


def test_registry_runtime_resolver_falls_back_to_non_native_entry() -> None:
    manifest = load_builtin_manifest("fake-open-loop")
    registry = default_registry()
    spec = RuntimeSpec(
        mode="run",
        workload_name="processor_smoke",
        workload_config={"synthetic_observation": True, "episode_length": 1},
    )

    plan = registry.resolve_runtime(manifest, spec)

    assert plan.reference_manifest is manifest
    assert plan.manifest is manifest
    assert plan.transformed is False
    assert plan.mapped_backend is None
    assert plan.mode == "fake"
    assert plan.workload_name == "open_loop"


def test_registry_runtime_resolver_can_require_backend_mapping() -> None:
    manifest = load_builtin_manifest("fake-open-loop")
    registry = default_registry()
    spec = RuntimeSpec(
        mode="native_smoke",
        workload_name="native_smoke",
        require_backend_mapping=True,
    )

    with pytest.raises(RuntimeResolutionError, match="does not declare a backend mapping"):
        registry.resolve_runtime(manifest, spec)


def test_native_runtime_spec_rejects_empty_mode_or_workload() -> None:
    registry = default_registry()
    manifest = load_builtin_manifest("fastwam-libero")

    with pytest.raises(RuntimeResolutionError, match="mode"):
        registry.resolve_runtime(manifest, RuntimeSpec(mode="", workload_name="serve"))

    with pytest.raises(RuntimeResolutionError, match="workload"):
        registry.resolve_runtime(manifest, RuntimeSpec(mode="serve", workload_name=""))
