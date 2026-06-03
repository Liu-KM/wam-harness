from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wam_harness.core.types import RuntimeInfo

TRACE_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceWriter:
    def __init__(
        self,
        path: Path,
        run_id: str,
        runtime_info: RuntimeInfo | None = None,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.runtime_info = runtime_info
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def set_runtime_info(self, runtime_info: RuntimeInfo) -> None:
        self.runtime_info = runtime_info

    def write(self, event: str, **payload: Any) -> None:
        row: dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "event": event,
            "timestamp": utc_now(),
            "run_id": self.run_id,
        }
        if self.runtime_info is not None:
            row.update(self.runtime_info.to_dict())
        row.update(payload)
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
