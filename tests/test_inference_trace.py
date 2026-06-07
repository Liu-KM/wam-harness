from __future__ import annotations

import pytest

from eazywam.core.action_contract import ActionContractError
from eazywam.core.inference_trace import (
    inference_result_payload,
    observation_summary,
)
from eazywam.core.manifest import load_builtin_manifest, manifest_from_dict
from eazywam.core.types import ActionChunk, InferenceResult, Observation


def test_observation_summary_reports_contract_keys() -> None:
    observation = Observation(
        images={"wrist": [], "primary": []},
        prompt="pick",
        state={"proprio": [0.0]},
        history=[{"action": [0.0]}],
        session={"episode_id": 1},
    )

    assert observation_summary(observation) == {
        "image_keys": ["primary", "wrist"],
        "state_keys": ["proprio"],
        "prompt": "pick",
        "history_len": 1,
        "session_keys": ["episode_id"],
    }


def test_inference_result_payload_preserves_trace_fields() -> None:
    manifest = load_builtin_manifest("fake-open-loop")
    result = InferenceResult(
        action_chunk=ActionChunk(actions=[[1.0, 2.0], [3.0, 4.0]]),
        warnings=["smoke warning"],
        backend_metadata={"backend": "fake"},
        timing={"model_ms": 1.5},
        memory={"backend_mb": 12.0},
        future_frames={"present": True},
        value={"score": 0.25},
    )

    payload = inference_result_payload(
        manifest,
        result,
        expected_horizon=2,
        wall_ms=3.5,
        memory={"rss_mb": 10.0},
    )

    assert payload["action_chunk_len"] == 2
    assert payload["action_dim"] == 2
    assert payload["action_chunk_shape"] == [2, 2]
    assert payload["action_summary"]["mean"] == 2.5
    assert payload["future_frames"] == {"present": True}
    assert payload["value"] == {"score": 0.25}
    assert payload["timing"] == {"model_ms": 1.5, "wall_ms": 3.5}
    assert payload["memory"] == {"rss_mb": 10.0, "backend_mb": 12.0}
    assert payload["backend_metadata"] == {"backend": "fake"}
    assert payload["action_contract"] is None
    assert payload["warnings"] == ["smoke warning"]


def test_inference_result_payload_validates_action_contract_when_enabled() -> None:
    data = load_builtin_manifest("fastwam-libero").to_dict()
    data["backend"] = {"name": "fastwam", "mode": "run", "config": {}}
    manifest = manifest_from_dict(data)
    result = InferenceResult(action_chunk=ActionChunk(actions=[[0.0] * 7]))

    with pytest.raises(ActionContractError, match="action horizon mismatch"):
        inference_result_payload(
            manifest,
            result,
            expected_horizon=2,
            wall_ms=1.0,
            validate_action_contract=True,
        )
