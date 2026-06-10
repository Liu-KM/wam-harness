from eazywam.core.manifest import load_builtin_manifest
from eazywam.core.registry import Registry, RegistryError, default_registry


class SingleManifestCatalog:
    def load_manifest(self, model_id: str):
        assert model_id == "fake-open-loop"
        return load_builtin_manifest(model_id)

    def list_model_ids(self) -> list[str]:
        return ["fake-open-loop"]


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


def test_registry_can_include_manifest_enabled_optimization_profiles() -> None:
    registry = default_registry()
    manifest = registry.load_manifest("fastwam-libero")

    profiles = registry.build_optimization_profiles(
        manifest,
        [],
        include_defaults=True,
    )
    by_name = {profile.name: profile for profile in profiles}

    assert by_name["dit_cache"].enabled is True
    assert by_name["dit_cache"].params == {"mode": "video_kv"}
    assert by_name["cuda_graph"].enabled is True
    assert by_name["cuda_graph"].params == {"mode": "auto", "capture": "action_body"}


def test_registry_deduplicates_explicit_and_default_optimization_profiles() -> None:
    registry = default_registry()
    manifest = registry.load_manifest("fastwam-libero")

    profiles = registry.build_optimization_profiles(
        manifest,
        ["dit_cache"],
        include_defaults=True,
    )

    assert [profile.name for profile in profiles].count("dit_cache") == 1
    assert profiles[0].name == "dit_cache"


def test_registry_does_not_include_disabled_manifest_profiles() -> None:
    registry = default_registry()
    manifest = registry.load_manifest("fake-open-loop")

    profiles = registry.build_optimization_profiles(
        manifest,
        [],
        include_defaults=True,
    )
    names = [profile.name for profile in profiles]

    assert "action_chunk_scheduling" in names
    assert "fake_cache" not in names


def test_registry_uses_injected_manifest_catalog() -> None:
    registry = Registry(catalog=SingleManifestCatalog())

    manifest = registry.load_manifest("fake-open-loop")

    assert manifest.id == "fake-open-loop"
    assert registry.list_model_ids() == ["fake-open-loop"]
