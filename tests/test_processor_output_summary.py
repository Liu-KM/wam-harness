import pytest

from wam_harness.core.manifest import load_builtin_manifest
from wam_harness.core.registry import default_registry
from wam_harness.processors.cosmos_policy_libero import CosmosPolicyProcessorError
from wam_harness.processors.dreamzero_droid import DreamZeroProcessorError
from wam_harness.processors.output_summary import (
    action_chunk_from_raw,
    summarize_future_output,
    summarize_value_output,
)


def test_output_summary_records_shape_and_stats_without_raw_payload() -> None:
    summary = summarize_future_output(
        {"future_frames": None, "future_state": [[1, 2], [3, 4]]},
        ("future_frames", "future_state"),
    )

    assert summary == {
        "present": True,
        "key": "future_state",
        "kind": "sequence",
        "shape": [2, 2],
        "count": 2,
        "item_types": ["list"],
        "numeric_count": 4,
        "finite": True,
        "min": 1.0,
        "max": 4.0,
        "mean": 2.5,
    }


def test_output_summary_records_scalar_value() -> None:
    assert summarize_value_output({"score": 0.75}, ("value", "score")) == {
        "key": "score",
        "kind": "number",
        "scalar": 0.75,
        "finite": True,
    }


def test_action_chunk_from_raw_accepts_array_like_payload() -> None:
    chunk = action_chunk_from_raw(
        {"action": _ArrayLike([[1, 2], [3, 4]])},
        label="Unit",
    )

    assert chunk.actions == [[1.0, 2.0], [3.0, 4.0]]


def test_action_chunk_from_raw_rejects_ragged_rows() -> None:
    with pytest.raises(ValueError, match="ragged rows"):
        action_chunk_from_raw({"actions": [[1, 2], [3]]}, label="Unit")


def test_cosmos_policy_processor_preserves_future_and_value_summaries() -> None:
    processor = default_registry().create_processor(load_builtin_manifest("cosmos-policy-libero"))

    result = processor.to_harness_result(
        {
            "actions": [[0, 1], [2, 3]],
            "future_state": [[1, 2], [3, 4]],
            "value": [0.25, 0.75],
        }
    )

    assert result.action_chunk.actions == [[0.0, 1.0], [2.0, 3.0]]
    assert result.future_frames == {
        "present": True,
        "key": "future_state",
        "kind": "sequence",
        "shape": [2, 2],
        "count": 2,
        "item_types": ["list"],
        "numeric_count": 4,
        "finite": True,
        "min": 1.0,
        "max": 4.0,
        "mean": 2.5,
    }
    assert result.value == {
        "key": "value",
        "kind": "sequence",
        "shape": [2],
        "count": 2,
        "item_types": ["float"],
        "numeric_count": 2,
        "finite": True,
        "min": 0.25,
        "max": 0.75,
        "mean": 0.5,
    }


def test_dreamzero_processor_preserves_future_and_value_summaries() -> None:
    processor = default_registry().create_processor(load_builtin_manifest("dreamzero-droid-sim"))

    result = processor.to_harness_result(
        {
            "action": _ArrayLike([[0.1, 0.2, 0.3]]),
            "future_frames": ["frame-a", "frame-b"],
            "score": 0.75,
        }
    )

    assert result.action_chunk.actions == [[0.1, 0.2, 0.3]]
    assert result.future_frames == {
        "present": True,
        "key": "future_frames",
        "kind": "sequence",
        "shape": [2],
        "count": 2,
        "item_types": ["str"],
    }
    assert result.value == {
        "key": "score",
        "kind": "number",
        "scalar": 0.75,
        "finite": True,
    }


def test_native_processors_raise_backend_specific_errors_for_missing_actions() -> None:
    cosmos = default_registry().create_processor(load_builtin_manifest("cosmos-policy-libero"))
    dreamzero = default_registry().create_processor(load_builtin_manifest("dreamzero-droid-sim"))

    with pytest.raises(CosmosPolicyProcessorError, match="missing actions"):
        cosmos.to_harness_result({"future_state": [[1, 2]]})
    with pytest.raises(DreamZeroProcessorError, match="missing actions"):
        dreamzero.to_harness_result({"future_frames": ["frame-a"]})


class _ArrayLike:
    def __init__(self, payload: list[list[float]]) -> None:
        self.payload = payload

    def tolist(self) -> list[list[float]]:
        return self.payload
