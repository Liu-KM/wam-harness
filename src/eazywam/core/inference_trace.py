from __future__ import annotations

from typing import Any

from eazywam.core.action_contract import maybe_validate_action_contract
from eazywam.core.action_summary import action_chunk_summary
from eazywam.core.memory import memory_snapshot
from eazywam.core.types import InferenceResult, Manifest


def observation_summary(observation: object) -> dict[str, object]:
    images = getattr(observation, "images", {})
    state = getattr(observation, "state", {})
    history = getattr(observation, "history", [])
    session = getattr(observation, "session", {})
    return {
        "image_keys": sorted(images.keys()) if isinstance(images, dict) else [],
        "state_keys": sorted(state.keys()) if isinstance(state, dict) else [],
        "prompt": str(getattr(observation, "prompt", "")),
        "history_len": len(history) if isinstance(history, list) else 0,
        "session_keys": sorted(session.keys()) if isinstance(session, dict) else [],
    }


def inference_result_payload(
    manifest: Manifest,
    result: InferenceResult,
    *,
    expected_horizon: int,
    wall_ms: float,
    validate_action_contract: bool = False,
    memory: dict[str, Any] | None = None,
) -> dict[str, object]:
    action_contract = maybe_validate_action_contract(
        manifest,
        result.action_chunk,
        expected_horizon=expected_horizon,
        enabled=validate_action_contract,
    )
    return {
        "action_chunk_len": result.action_chunk.horizon,
        "action_dim": result.action_chunk.action_dim,
        "action_chunk_shape": [
            result.action_chunk.horizon,
            result.action_chunk.action_dim,
        ],
        "action_summary": action_chunk_summary(result.action_chunk),
        "future_frames": result.future_frames,
        "value": result.value,
        "timing": {**result.timing, "wall_ms": wall_ms},
        "memory": {**(memory if memory is not None else memory_snapshot()), **result.memory},
        "backend_metadata": result.backend_metadata,
        "action_contract": (
            action_contract.to_dict() if action_contract is not None else None
        ),
        "warnings": result.warnings,
    }
