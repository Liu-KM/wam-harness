from __future__ import annotations

import math
from dataclasses import dataclass

from wam_harness.core.types import ActionChunk, Manifest


class ActionContractError(ValueError):
    """Raised when a native backend returns an invalid action chunk."""


@dataclass(frozen=True)
class ActionContractCheck:
    status: str
    expected_shape: list[int | None]
    observed_shape: list[int]
    finite: bool
    rectangular: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "expected_shape": self.expected_shape,
            "observed_shape": self.observed_shape,
            "finite": self.finite,
            "rectangular": self.rectangular,
            "errors": self.errors,
        }


def validate_native_action_contract(
    manifest: Manifest,
    chunk: ActionChunk,
    *,
    expected_horizon: int,
) -> ActionContractCheck:
    """Validate the minimum action contract for native product inference."""

    expected_dim = _expected_action_dim(manifest)
    observed_shape = [chunk.horizon, chunk.action_dim]
    expected_shape = [int(expected_horizon), expected_dim]
    rectangular = _is_rectangular(chunk)
    finite = _is_finite(chunk)
    errors: list[str] = []

    if chunk.horizon <= 0:
        errors.append("action chunk is empty")
    if not rectangular:
        errors.append("action rows are not rectangular")
    if not finite:
        errors.append("action values must be finite")
    if chunk.horizon != int(expected_horizon):
        errors.append(
            f"action horizon mismatch: expected {expected_horizon}, got {chunk.horizon}"
        )
    if expected_dim is not None and chunk.action_dim != expected_dim:
        errors.append(
            f"action dim mismatch: expected {expected_dim}, got {chunk.action_dim}"
        )

    check = ActionContractCheck(
        status="ok" if not errors else "error",
        expected_shape=expected_shape,
        observed_shape=observed_shape,
        finite=finite,
        rectangular=rectangular,
        errors=errors,
    )
    if errors:
        raise ActionContractError(
            f"{manifest.id} native action contract failed: {'; '.join(errors)}"
        )
    return check


def maybe_validate_native_action_contract(
    manifest: Manifest,
    chunk: ActionChunk,
    *,
    expected_horizon: int,
) -> ActionContractCheck | None:
    if not _is_native_mode(manifest):
        return None
    return validate_native_action_contract(
        manifest,
        chunk,
        expected_horizon=expected_horizon,
    )


def _expected_action_dim(manifest: Manifest) -> int | None:
    action = manifest.processor.get("action", {})
    if not isinstance(action, dict):
        return None
    value = action.get("dim")
    if value is None:
        return None
    return int(value)


def _is_native_mode(manifest: Manifest) -> bool:
    return str(manifest.backend.get("mode", "")).startswith("native_")


def _is_rectangular(chunk: ActionChunk) -> bool:
    if not chunk.actions:
        return True
    width = len(chunk.actions[0])
    return all(len(row) == width for row in chunk.actions)


def _is_finite(chunk: ActionChunk) -> bool:
    return all(math.isfinite(float(value)) for row in chunk.actions for value in row)
