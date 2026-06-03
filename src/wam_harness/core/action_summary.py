from __future__ import annotations

import math

from wam_harness.core.types import ActionChunk


def action_chunk_summary(chunk: ActionChunk) -> dict[str, object]:
    values = [float(value) for row in chunk.actions for value in row]
    finite = all(math.isfinite(value) for value in values)
    summary: dict[str, object] = {
        "shape": [chunk.horizon, chunk.action_dim],
        "count": len(values),
        "finite": finite,
    }
    if values:
        summary.update(
            {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "max_abs": max(abs(value) for value in values),
            }
        )
    else:
        summary.update({"min": None, "max": None, "mean": None, "max_abs": None})
    return summary
