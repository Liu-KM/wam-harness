from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wam_harness.core.types import Observation


def load_json_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("input file must contain a JSON object")
    return payload


def write_json_payload(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def observation_from_payload(payload: dict[str, Any]) -> Observation | None:
    raw = payload.get("observation", payload)
    if not isinstance(raw, dict) or "images" not in raw:
        return None
    images = raw.get("images")
    if not isinstance(images, dict):
        raise ValueError("observation.images must be a JSON object")
    return Observation(
        images=images,
        prompt=str(raw.get("prompt", "")),
        state=dict_or_empty(raw.get("state"), "observation.state"),
        history=list_or_empty(raw.get("history"), "observation.history"),
        session=dict_or_empty(raw.get("session"), "observation.session"),
        metadata=dict_or_empty(raw.get("metadata"), "observation.metadata"),
    )


def dict_or_empty(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def list_or_empty(value: object, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} entries must be JSON objects")
    return value
