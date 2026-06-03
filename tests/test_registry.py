from wam_harness.core.registry import RegistryError, default_registry


def test_default_registry_resolves_fake_backend_and_workload() -> None:
    registry = default_registry()
    manifest = registry.load_manifest("fake-open-loop")
    profiles = registry.build_optimization_profiles(manifest, ["fake_cache"])

    backend = registry.create_backend(manifest, profiles)
    processor = registry.create_processor(manifest)
    workload = registry.create_workload(manifest)

    assert backend.runtime_info().manifest_id == "fake-open-loop"
    assert processor.modality_limits()["processor"] == "passthrough"
    assert processor.smoke_observation().images["primary"]
    assert workload.episode_length == 6
    assert profiles[0].name == "fake_cache"


def test_registry_rejects_unsupported_optimization() -> None:
    registry = default_registry()
    manifest = registry.load_manifest("fake-open-loop")

    try:
        registry.build_optimization_profiles(manifest, ["vla_cache"])
    except RegistryError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("unsupported optimization should fail")
