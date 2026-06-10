from importlib import import_module

import pytest

torch = pytest.importorskip("torch")

ActionBodyCudaGraphManager = import_module(
    "fastwam.models.wan22.cuda_graph"
).ActionBodyCudaGraphManager


def test_action_body_cuda_graph_manager_falls_back_without_cuda_inputs() -> None:
    manager = ActionBodyCudaGraphManager()
    calls = 0

    def action_body(**kwargs):
        nonlocal calls
        calls += 1
        return kwargs["action_tokens"]

    result = manager.run(
        action_body,
        shape_metadata={
            "cache_mode": "video_kv",
            "image_hw": [224, 224],
            "action_horizon": 4,
            "video_seq_len": 2,
            "action_seq_len": 4,
        },
        action_tokens=torch.zeros(1, 4, 8),
        action_freqs=torch.zeros(1, 4, 8),
        action_t_mod=torch.zeros(1, 1, 8),
        action_context_payload={
            "context": torch.zeros(1, 3, 8),
            "mask": torch.ones(1, 3, dtype=torch.bool),
        },
        video_kv_cache=[
            {
                "k": torch.zeros(1, 2, 8),
                "v": torch.zeros(1, 2, 8),
            }
        ],
        attention_mask=torch.ones(6, 6, dtype=torch.bool),
        video_seq_len=2,
    )

    assert calls == 0
    assert result.output is None
    assert result.capture_success is False
    assert result.replayed is False
    assert result.fallback_reason in {"cuda_unavailable", "non_cuda_tensor"}
    assert result.shape_key is not None
    assert result.shape_key["metadata"]["image_hw"] == [224, 224]
    assert result.shape_key["metadata"]["cache_mode"] == "video_kv"
