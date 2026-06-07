from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from wam_harness.core.types import Manifest

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "id",
    "display_name",
    "source",
    "backend",
    "processor",
    "workload",
    "defaults",
    "optimizations",
}


class ManifestError(ValueError):
    """Raised when a Wamfile manifest is missing required contract fields."""


def manifest_from_dict(data: dict[str, Any]) -> Manifest:
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - data.keys())
    if missing:
        raise ManifestError(f"manifest is missing required fields: {', '.join(missing)}")

    for section in ("backend", "processor", "workload"):
        value = data.get(section)
        if not isinstance(value, dict) or "name" not in value:
            raise ManifestError(f"manifest section '{section}' must contain a name")

    defaults = data.get("defaults")
    if not isinstance(defaults, dict):
        raise ManifestError("manifest defaults must be a mapping")

    optimizations = data.get("optimizations")
    if not isinstance(optimizations, dict) or "supported" not in optimizations:
        raise ManifestError("manifest optimizations must contain supported profiles")

    assets = dict(data.get("assets", {}))
    asset_groups = _validate_asset_groups(data.get("asset_groups", {}), assets)

    return Manifest(
        schema_version=int(data["schema_version"]),
        id=str(data["id"]),
        display_name=str(data["display_name"]),
        source=dict(data["source"]),
        assets=assets,
        asset_groups=asset_groups,
        backend=dict(data["backend"]),
        processor=dict(data["processor"]),
        workload=dict(data["workload"]),
        defaults=dict(defaults),
        optimizations=dict(optimizations),
        deployment=dict(data.get("deployment", {})),
        eval=dict(data.get("eval", {})),
        known_gaps=[str(item) for item in data.get("known_gaps", [])],
    )


def _validate_asset_groups(raw_groups: object, assets: dict[str, Any]) -> dict[str, Any]:
    if raw_groups is None:
        return {}
    if not isinstance(raw_groups, dict):
        raise ManifestError("manifest asset_groups must be a mapping")

    known_assets = set(assets)
    groups: dict[str, Any] = {}
    for name, raw in raw_groups.items():
        group_name = str(name)
        if isinstance(raw, list):
            group_assets = raw
            groups[group_name] = [str(item) for item in group_assets]
        elif isinstance(raw, dict):
            group_assets = raw.get("assets")
            if not isinstance(group_assets, list):
                raise ManifestError(f"asset group '{group_name}' assets must be a list")
            group_data = dict(raw)
            group_data["assets"] = [str(item) for item in group_assets]
            groups[group_name] = group_data
        else:
            raise ManifestError(f"asset group '{group_name}' must be a list or mapping")

        missing = sorted(str(item) for item in group_assets if str(item) not in known_assets)
        if missing:
            raise ManifestError(
                f"asset group '{group_name}' references unknown asset(s): {', '.join(missing)}"
            )
    return groups


def load_manifest(path: str | Path) -> Manifest:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError(f"manifest at {path} must be a mapping")
    return manifest_from_dict(data)


def load_builtin_manifest(model_id: str) -> Manifest:
    filename = f"{model_id}.yaml"
    try:
        text = resources.files("wam_harness.manifests").joinpath(filename).read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise ManifestError(f"unknown built-in model manifest: {model_id}") from exc

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ManifestError(f"built-in manifest {filename} must be a mapping")
    return manifest_from_dict(data)


def list_builtin_manifest_ids() -> list[str]:
    manifests = resources.files("wam_harness.manifests")
    return sorted(
        item.name.removesuffix(".yaml")
        for item in manifests.iterdir()
        if item.name.endswith(".yaml")
    )
