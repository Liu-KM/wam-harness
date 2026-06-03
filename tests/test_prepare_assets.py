from __future__ import annotations

from pathlib import Path

import pytest

from wam_harness.core.assets import parse_hf_uri
from wam_harness.cli_render import render_prepare
from wam_harness.core.model_entry import prepare_model_entry


class RecordingDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def download(self, uri: str, expected_path: Path) -> None:
        self.calls.append((uri, expected_path))
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text("asset\n", encoding="utf-8")


def test_parse_hf_uri_for_file_asset() -> None:
    ref = parse_hf_uri("hf://yuanty/fastwam/libero_uncond_2cam224.pt")

    assert ref.repo_id == "yuanty/fastwam"
    assert ref.filename == "libero_uncond_2cam224.pt"


def test_parse_hf_uri_for_snapshot_asset() -> None:
    ref = parse_hf_uri("hf://Wan-AI/Wan2.2-TI2V-5B")

    assert ref.repo_id == "Wan-AI/Wan2.2-TI2V-5B"
    assert ref.filename is None


def test_prepare_downloads_selected_assets_only(tmp_path) -> None:
    downloader = RecordingDownloader()

    summary = prepare_model_entry(
        "fastwam-libero",
        cache_dir=tmp_path / "cache",
        download=True,
        selected_assets=["checkpoint", "dataset_stats"],
        downloader=downloader,
    )

    assert summary.status == "ok"
    assert summary.selected_assets == ["checkpoint", "dataset_stats"]
    checkpoint = next(asset for asset in summary.assets if asset.name == "checkpoint")
    dataset_stats = next(asset for asset in summary.assets if asset.name == "dataset_stats")
    assert checkpoint.size_bytes == 12041735140
    assert dataset_stats.size_bytes == 40939
    assert checkpoint.required is True
    assert checkpoint.runtime is True
    assert dataset_stats.required is True
    assert dataset_stats.runtime is True
    assert checkpoint.to_dict()["size_bytes"] == 12041735140
    assert checkpoint.to_dict()["required"] is True
    assert checkpoint.to_dict()["runtime"] is True
    downloaded = {asset.name for asset in summary.assets if asset.downloaded}
    assert downloaded == {"checkpoint", "dataset_stats"}
    assert {call[0] for call in downloader.calls} == {
        "hf://yuanty/fastwam/libero_uncond_2cam224.pt",
        "hf://yuanty/fastwam/libero_uncond_2cam224_dataset_stats.json",
    }

    output = render_prepare(summary)
    assert "Selected assets: checkpoint,dataset_stats" in output
    assert "Download: enabled" in output
    assert "checkpoint: present" in output
    assert "[required,runtime]" in output
    assert "[11.2 GiB]" in output
    assert "[40.0 KiB]" in output
    assert "[downloaded]" in output


def test_prepare_resolves_fastwam_runtime_assets_under_cache_dir(tmp_path) -> None:
    downloader = RecordingDownloader()
    cache_dir = tmp_path / "cache"

    summary = prepare_model_entry(
        "fastwam-libero",
        cache_dir=cache_dir,
        download=True,
        selected_assets=["model_base"],
        downloader=downloader,
    )

    expected_path = cache_dir / "diffsynth-models" / "Wan-AI" / "Wan2.2-TI2V-5B"
    assert summary.status == "ok"
    assert downloader.calls == [("hf://Wan-AI/Wan2.2-TI2V-5B", expected_path)]
    model_base = next(asset for asset in summary.assets if asset.name == "model_base")
    assert model_base.expected_path == expected_path
    assert model_base.downloaded is True
    assert model_base.required is False
    assert model_base.runtime is True


def test_prepare_marks_fastwam_native_asset_roles(tmp_path) -> None:
    summary = prepare_model_entry(
        "fastwam-libero",
        cache_dir=tmp_path / "cache",
    )

    roles = {asset.name: (asset.required, asset.runtime) for asset in summary.assets}

    assert roles == {
        "checkpoint": (True, True),
        "dataset_stats": (True, True),
        "model_base": (False, True),
        "tokenizer_components": (False, True),
    }


def test_prepare_rejects_unknown_selected_asset(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown asset"):
        prepare_model_entry(
            "fastwam-libero",
            cache_dir=tmp_path / "cache",
            selected_assets=["missing_asset"],
        )
