from __future__ import annotations

from eazywam.core.registry import Registry


def default_registry() -> Registry:
    from eazywam.backends.cosmos_policy import CosmosPolicyBackend
    from eazywam.backends.dreamzero import DreamZeroBackend
    from eazywam.backends.fastwam import FastWAMBackend
    from eazywam.backends.fake import FakeBackend
    from eazywam.backends.native_support.runtime import native_runtime_resolver
    from eazywam.processors.cosmos_policy_libero import CosmosPolicyLiberoProcessor
    from eazywam.processors.dreamzero_droid import DreamZeroDroidProcessor
    from eazywam.processors.fastwam_libero import FastWAMLiberoProcessor
    from eazywam.processors.fastwam_robotwin import FastWAMRobotWinProcessor
    from eazywam.processors.passthrough import PassthroughProcessor
    from eazywam.evals.libero import LiberoSingleTaskEvalRunner
    from eazywam.evals.robotwin import RobotWinSingleTaskEvalRunner
    from eazywam.workloads.open_loop import OpenLoopWorkload

    registry = Registry()
    registry.register_backend("fake", FakeBackend)
    registry.register_backend("cosmos_policy", CosmosPolicyBackend)
    registry.register_backend("dreamzero", DreamZeroBackend)
    registry.register_backend("fastwam", FastWAMBackend)
    registry.register_processor("passthrough", PassthroughProcessor.from_manifest)
    registry.register_processor("fastwam_libero", FastWAMLiberoProcessor.from_manifest)
    registry.register_processor("fastwam_robotwin", FastWAMRobotWinProcessor.from_manifest)
    registry.register_processor(
        "cosmos_policy_libero",
        CosmosPolicyLiberoProcessor.from_manifest,
    )
    registry.register_processor("dreamzero_droid", DreamZeroDroidProcessor.from_manifest)
    registry.register_workload("open_loop", OpenLoopWorkload.from_manifest)
    registry.register_eval_runner(
        "libero_single_task",
        lambda current_registry: LiberoSingleTaskEvalRunner(current_registry),
    )
    registry.register_eval_runner(
        "robotwin_single_task",
        lambda current_registry: RobotWinSingleTaskEvalRunner(current_registry),
    )
    registry.register_runtime_resolver(native_runtime_resolver)
    registry.register_optimization("fake_cache", {"cache_scope": "replan"})
    registry.register_optimization("action_chunk_scheduling", {})
    registry.register_optimization("dit_cache", {})
    registry.register_optimization("cuda_graph", {"mode": "auto", "capture": "action_body"})
    registry.register_optimization("torch_compile", {"mode": "auto", "target": "action_body"})
    registry.register_optimization("jpeg_observation_compression", {})
    registry.register_optimization("parallel_inference", {})
    return registry
