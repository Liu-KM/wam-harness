from eazywam.core.registry import default_registry
from eazywam.core.types import InferenceRequest


def test_fake_backend_is_deterministic() -> None:
    registry = default_registry()
    manifest = registry.load_manifest("fake-open-loop")
    profiles = registry.build_optimization_profiles(manifest, ["fake_cache"])
    backend = registry.create_backend(manifest, profiles)
    workload = registry.create_workload(manifest)

    backend.load()
    backend.warmup()
    backend.reset()
    workload.reset()
    observation = workload.observation()
    request = InferenceRequest(
        observation=observation,
        action_horizon=3,
        replan_steps=2,
        optimization_profiles=profiles,
    )

    first = backend.infer(request)
    second = backend.infer(request)

    assert first.action_chunk.actions == second.action_chunk.actions
    assert first.action_chunk.horizon == 3
    assert first.action_chunk.action_dim == 4
    assert first.backend_metadata["fake_cache_enabled"] is True
