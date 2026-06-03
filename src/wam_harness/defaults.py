from __future__ import annotations

from wam_harness.core.registry import Registry


def default_registry() -> Registry:
    from wam_harness.backends.cosmos_policy import CosmosPolicyBackend
    from wam_harness.backends.dreamzero import DreamZeroBackend
    from wam_harness.backends.fastwam import FastWAMBackend
    from wam_harness.backends.fake import FakeBackend
    from wam_harness.backends.native_support.runtime import native_runtime_resolver
    from wam_harness.processors.cosmos_policy_libero import CosmosPolicyLiberoProcessor
    from wam_harness.processors.dreamzero_droid import DreamZeroDroidProcessor
    from wam_harness.processors.fastwam_libero import FastWAMLiberoProcessor
    from wam_harness.processors.passthrough import PassthroughProcessor
    from wam_harness.workloads.open_loop import OpenLoopWorkload

    registry = Registry()
    registry.register_backend("fake", FakeBackend)
    registry.register_backend("cosmos_policy", CosmosPolicyBackend)
    registry.register_backend("dreamzero", DreamZeroBackend)
    registry.register_backend("fastwam", FastWAMBackend)
    registry.register_processor("passthrough", PassthroughProcessor.from_manifest)
    registry.register_processor("fastwam_libero", FastWAMLiberoProcessor.from_manifest)
    registry.register_processor(
        "cosmos_policy_libero",
        CosmosPolicyLiberoProcessor.from_manifest,
    )
    registry.register_processor("dreamzero_droid", DreamZeroDroidProcessor.from_manifest)
    registry.register_workload("open_loop", OpenLoopWorkload.from_manifest)
    registry.register_runtime_resolver(native_runtime_resolver)
    registry.register_optimization("fake_cache", {"cache_scope": "replan"})
    registry.register_optimization("action_chunk_scheduling", {})
    registry.register_optimization("dit_cache", {})
    registry.register_optimization("jpeg_observation_compression", {})
    registry.register_optimization("parallel_inference", {})
    return registry
