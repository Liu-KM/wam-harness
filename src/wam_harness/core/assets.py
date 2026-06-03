from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class AssetError(RuntimeError):
    """Raised when an asset URI cannot be prepared."""


@dataclass(frozen=True)
class HfAssetRef:
    repo_id: str
    filename: str | None


class AssetDownloader(Protocol):
    def download(self, uri: str, expected_path: Path) -> None: ...


class HuggingFaceAssetDownloader:
    def download(self, uri: str, expected_path: Path) -> None:
        ref = parse_hf_uri(uri)
        if ref.filename is None:
            _snapshot_download(ref.repo_id, expected_path)
            return

        downloaded_path = _hf_hub_download(ref.repo_id, ref.filename, expected_path.parent)
        if downloaded_path != expected_path:
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_path, expected_path)


def parse_hf_uri(uri: str) -> HfAssetRef:
    if not uri.startswith("hf://"):
        raise AssetError(f"unsupported asset URI: {uri}")
    path = uri.removeprefix("hf://").strip("/")
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        raise AssetError(f"Hugging Face asset URI must include org/repo: {uri}")

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:]) if len(parts) > 2 else None
    return HfAssetRef(repo_id=repo_id, filename=filename)


def _hf_hub_download(repo_id: str, filename: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise AssetError(
            "huggingface_hub is required for `wam prepare --download` with hf:// assets. "
            "Install it in the core/container environment before downloading assets."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )
    )


def _snapshot_download(repo_id: str, local_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise AssetError(
            "huggingface_hub is required for `wam prepare --download` with hf:// assets. "
            "Install it in the core/container environment before downloading assets."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
