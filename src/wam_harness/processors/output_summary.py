from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from wam_harness.core.types import ActionChunk, JsonDict

_TEXT_TYPES = (str, bytes, bytearray)


def action_chunk_from_raw(
    raw_output: object,
    *,
    action_keys: tuple[str, ...] = ("actions", "action"),
    label: str,
) -> ActionChunk:
    actions = raw_output
    if isinstance(raw_output, Mapping):
        action_entry = first_present_mapping_value(raw_output, action_keys)
        actions = action_entry[1] if action_entry is not None else None
    if actions is None:
        raise ValueError(f"{label} raw output is missing actions.")

    rows = _coerce_action_rows(actions, label=label)
    return ActionChunk(actions=rows)


def summarize_future_output(
    raw_output: object,
    candidate_keys: tuple[str, ...],
) -> JsonDict | None:
    entry = first_present_mapping_value(raw_output, candidate_keys)
    if entry is None:
        return None
    key, value = entry
    return {"present": True, "key": key, **_summarize_payload(value)}


def summarize_value_output(
    raw_output: object,
    candidate_keys: tuple[str, ...],
) -> JsonDict | None:
    entry = first_present_mapping_value(raw_output, candidate_keys)
    if entry is None:
        return None
    key, value = entry
    return {"key": key, **_summarize_payload(value)}


def first_present_mapping_value(
    raw_output: object,
    candidate_keys: tuple[str, ...],
) -> tuple[str, object] | None:
    if not isinstance(raw_output, Mapping):
        return None
    for key in candidate_keys:
        if key in raw_output and raw_output[key] is not None:
            return key, raw_output[key]
    return None


def _coerce_action_rows(actions: object, *, label: str) -> list[list[float]]:
    plain_actions = _to_plain(actions)
    if not _is_sequence(plain_actions):
        raise ValueError(f"Expected {label} actions [T,D], got {type(actions).__name__}.")

    rows: list[list[float]] = []
    expected_width: int | None = None
    for row in plain_actions:
        plain_row = _to_plain(row)
        if not _is_sequence(plain_row):
            raise ValueError(f"Expected {label} actions [T,D], got a non-sequence row.")
        values = [float(value) for value in plain_row]
        if expected_width is None:
            expected_width = len(values)
        elif len(values) != expected_width:
            raise ValueError(f"Expected {label} actions [T,D], got ragged rows.")
        rows.append(values)

    if not rows:
        raise ValueError(f"Expected {label} actions [T,D], got an empty sequence.")
    return rows


def _summarize_payload(value: object) -> JsonDict:
    plain_value = _to_plain(value)
    summary: JsonDict = {"kind": _kind(plain_value)}

    shape = _shape(plain_value)
    if shape is not None:
        summary["shape"] = shape

    if isinstance(plain_value, Mapping):
        summary["keys"] = sorted(str(key) for key in plain_value.keys())
        return summary

    if _is_sequence(plain_value):
        summary["count"] = len(plain_value)
        item_types = sorted({type(_to_plain(item)).__name__ for item in plain_value})
        if item_types:
            summary["item_types"] = item_types

    if _is_number(plain_value):
        scalar = float(plain_value)
        summary["scalar"] = scalar
        summary["finite"] = math.isfinite(scalar)
        return summary

    numbers = _flatten_numbers(plain_value)
    if numbers is None:
        return summary

    summary["numeric_count"] = len(numbers)
    finite_values = [number for number in numbers if math.isfinite(number)]
    summary["finite"] = len(finite_values) == len(numbers)
    if finite_values:
        summary["min"] = min(finite_values)
        summary["max"] = max(finite_values)
        summary["mean"] = sum(finite_values) / len(finite_values)
    return summary


def _to_plain(value: object) -> object:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            try:
                current = method()
            except TypeError:
                pass
    tolist = getattr(current, "tolist", None)
    if callable(tolist) and not isinstance(current, Mapping) and not isinstance(current, _TEXT_TYPES):
        try:
            return tolist()
        except TypeError:
            return current
    return current


def _kind(value: object) -> str:
    if isinstance(value, Mapping):
        return "mapping"
    if _is_sequence(value):
        return "sequence"
    if _is_number(value):
        return "number"
    return type(value).__name__


def _shape(value: object) -> list[int] | None:
    if not _is_sequence(value):
        return None

    length = len(value)
    if length == 0:
        return [0]

    child_shapes = [_shape(_to_plain(item)) for item in value]
    if all(shape == child_shapes[0] for shape in child_shapes):
        child = child_shapes[0]
        return [length, *child] if child is not None else [length]
    return [length]


def _flatten_numbers(value: object) -> list[float] | None:
    if _is_number(value):
        return [float(value)]
    if not _is_sequence(value):
        return None

    numbers: list[float] = []
    for item in value:
        child = _flatten_numbers(_to_plain(item))
        if child is None:
            return None
        numbers.extend(child)
    return numbers


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, _TEXT_TYPES)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
