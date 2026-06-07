from __future__ import annotations

import resource


def memory_snapshot() -> dict[str, float | None]:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "process_rss_mb": rss_kb / 1024,
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
        "cuda_peak_allocated_mb": None,
        "cuda_peak_reserved_mb": None,
    }
