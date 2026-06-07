import math

import pytest

from eazywam.core.action_contract import (
    ActionContractError,
    maybe_validate_action_contract,
    validate_action_contract,
)
from eazywam.core.manifest import load_builtin_manifest, manifest_from_dict
from eazywam.core.types import ActionChunk


def test_validate_action_contract_accepts_declared_shape() -> None:
    manifest = _native_manifest("fastwam-libero", "fastwam")
    chunk = ActionChunk(actions=[[0.0] * 7 for _ in range(32)])

    check = validate_action_contract(manifest, chunk, expected_horizon=32)

    assert check.status == "ok"
    assert check.expected_shape == [32, 7]
    assert check.observed_shape == [32, 7]
    assert check.finite is True
    assert check.rectangular is True


def test_validate_action_contract_rejects_bad_shape() -> None:
    manifest = _native_manifest("fastwam-libero", "fastwam")
    chunk = ActionChunk(actions=[[0.0] * 6 for _ in range(31)])

    with pytest.raises(ActionContractError, match="action horizon mismatch"):
        validate_action_contract(manifest, chunk, expected_horizon=32)


def test_validate_action_contract_rejects_non_finite_values() -> None:
    manifest = _native_manifest("fastwam-libero", "fastwam")
    chunk = ActionChunk(actions=[[math.inf] * 7 for _ in range(32)])

    with pytest.raises(ActionContractError, match="action values must be finite"):
        validate_action_contract(manifest, chunk, expected_horizon=32)


def test_maybe_validate_action_contract_skips_when_disabled() -> None:
    manifest = load_builtin_manifest("fake-open-loop")
    chunk = ActionChunk(actions=[[math.inf]])

    assert (
        maybe_validate_action_contract(
            manifest,
            chunk,
            expected_horizon=3,
            enabled=False,
        )
        is None
    )


def _native_manifest(model_id: str, backend_name: str):
    data = load_builtin_manifest(model_id).to_dict()
    data["backend"] = {"name": backend_name, "mode": "native_smoke", "config": {}}
    return manifest_from_dict(data)
