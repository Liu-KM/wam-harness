from __future__ import annotations

from wam_harness.core._utils import csv_text, format_size
from wam_harness.core.manifest import load_builtin_manifest
from wam_harness.core.model_entry import (
    AssetStatus,
    DoctorSummary,
    PrepareSummary,
    load_model_entries,
)
from wam_harness.model_entry_labels import (
    model_assets_label,
    model_deployment_label,
    model_input_label,
    model_output_label,
    model_runtime_label,
    model_supported_opts_label,
    model_task_label,
)


def render_model_list() -> str:
    entries = load_model_entries()
    id_width = max([len("MODEL ID"), *(len(entry.id) for entry in entries)])
    task_width = max([len("TASK"), *(len(model_task_label(entry)) for entry in entries)])
    lines = [f"{'MODEL ID':<{id_width}}  {'TASK':<{task_width}}  RUNTIME"]
    for entry in entries:
        lines.append(
            f"{entry.id:<{id_width}}  "
            f"{model_task_label(entry):<{task_width}}  "
            f"{model_runtime_label(entry)}"
        )
    return "\n".join(lines)


def render_model_info(model_id: str) -> str:
    entry = load_builtin_manifest(model_id)
    source = entry.source_repo or "unknown"
    lines = [
        f"Model: {entry.id}",
        f"Name: {entry.display_name}",
        f"Task: {model_task_label(entry)}",
        f"Source: {source}",
        f"Inputs: {model_input_label(entry)}",
        f"Outputs: {model_output_label(entry)}",
        f"Runtime: {model_runtime_label(entry)}",
        f"Deployment: {model_deployment_label(entry)}",
        f"Assets: {model_assets_label(entry)}",
        f"Supported opts: {model_supported_opts_label(entry)}",
    ]
    if entry.known_gaps:
        lines.append("Known gaps:")
        lines.extend(f"- {gap}" for gap in entry.known_gaps)
    return "\n".join(lines)


def render_doctor(summary: DoctorSummary) -> str:
    lines = [
        "WAM doctor",
        f"Cache directory: {summary.cache_dir} ({summary.cache_status})",
        f"Runtime setup: {summary.runtime_setup}",
    ]

    if summary.model_id is not None:
        lines.append(f"Model: {summary.model_id}")
        if summary.runtime is not None:
            lines.append(f"Runtime: {summary.runtime}")
        if summary.deployment is not None:
            lines.append(f"Deployment: {summary.deployment}")
        if summary.gpu is not None:
            lines.append(f"GPU: {summary.gpu}")

        if summary.assets:
            lines.append("Assets:")
            for asset in summary.assets:
                lines.append(f"- {asset.name}: {asset.status} ({asset.expected_path})")
        else:
            lines.append("Assets: none declared")

        lines.extend(summary.backend_lines)

    lines.append(f"Status: {summary.status}")
    return "\n".join(lines)


def render_prepare(summary: PrepareSummary) -> str:
    lines = [
        f"Model: {summary.model_id}",
        f"Cache directory: {summary.cache_dir}",
        "Runtime setup: not modified",
    ]
    if summary.selected_assets is not None:
        lines.append(f"Selected assets: {csv_text(summary.selected_assets, default='none')}")
    if summary.download:
        lines.append("Download: enabled")
    if summary.assets:
        lines.append("Assets:")
        for asset in summary.assets:
            line = f"- {asset.name}: {asset.status} ({asset.expected_path})"
            roles = _asset_role_labels(asset)
            if roles:
                line += f" [{','.join(roles)}]"
            if asset.size_bytes is not None:
                line += f" [{format_size(asset.size_bytes)}]"
            if asset.downloaded:
                line += " [downloaded]"
            if asset.message:
                line += f" - {asset.message}"
            lines.append(line)
    else:
        lines.append("Assets: none declared")
    lines.append(f"Status: {summary.status}")
    return "\n".join(lines)


def _asset_role_labels(asset: AssetStatus) -> list[str]:
    labels = []
    if asset.required:
        labels.append("required")
    if asset.runtime:
        labels.append("runtime")
    return labels
