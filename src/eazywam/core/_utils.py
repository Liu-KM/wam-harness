from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


def default_cache_dir() -> Path:
    return Path(os.environ.get("WAM_CACHE_DIR", str(Path.home() / ".cache" / "wam")))


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def csv_text(value: object, default: str) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if value is None:
        return default
    return str(value)


def format_size(size_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(size_bytes)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.1f} {unit}"
