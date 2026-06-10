from __future__ import annotations

from eazywam.core._utils import csv_text
from eazywam.core.types import Manifest


def model_task_label(entry: Manifest) -> str:
    eval_config = entry.eval
    if eval_config:
        simulator = eval_config.get("simulator") or entry.workload.get("config", {}).get("simulator")
        suite = eval_config.get("suite") or entry.workload.get("config", {}).get("suite")
        if simulator and suite:
            return f"{simulator} {suite}"
        if simulator:
            return str(simulator)
    return str(entry.workload_name)


def model_runtime_label(entry: Manifest) -> str:
    device = str(entry.defaults.get("device", "unknown"))
    mode = _product_runtime_mode(entry)
    if device.startswith("cuda"):
        return f"GPU container recommended ({mode})"
    return f"CPU ok ({mode})"


def _product_runtime_mode(entry: Manifest) -> str:
    deployment = entry.deployment
    native_backend = deployment.get("native_backend") if deployment else None
    if native_backend:
        return f"native: {native_backend}"
    if deployment and deployment.get("product_path"):
        return str(deployment["product_path"])
    return str(entry.backend.get("mode", entry.backend_name))


def model_deployment_label(entry: Manifest) -> str:
    deployment = entry.deployment
    if not deployment:
        return "native"
    product = str(deployment.get("product_path", "unknown"))
    native = deployment.get("native_backend")
    stage = str(deployment.get("native_stage", "unknown"))
    verified = bool(deployment.get("native_verified", False))
    parity = bool(deployment.get("parity_verified", False))
    parts = [f"product={product}"]
    if native is not None:
        parts.append(f"native={native} ({stage})")
    parts.append(f"native_verified={str(verified).lower()}")
    parts.append(f"parity_verified={str(parity).lower()}")
    next_gate = deployment.get("next_gate")
    if next_gate is not None:
        parts.append(f"next={next_gate}")
    return "; ".join(parts)


def model_input_label(entry: Manifest) -> str:
    observation = entry.processor.get("observation", {})
    if not isinstance(observation, dict):
        return "unknown"
    images = csv_text(observation.get("image_views"), default="none")
    state = str(observation.get("state", "none"))
    prompt = str(observation.get("prompt", "none"))
    return f"images={images}; state={state}; prompt={prompt}"


def model_output_label(entry: Manifest) -> str:
    action = entry.processor.get("action", {})
    if not isinstance(action, dict):
        return "action chunks"
    horizon = action.get("horizon")
    dim = action.get("dim")
    parts = ["action chunks"]
    if horizon is not None:
        parts.append(f"horizon={horizon}")
    if dim is not None:
        parts.append(f"dim={dim}")
    return "; ".join(parts)


def model_assets_label(entry: Manifest) -> str:
    if not entry.assets:
        return "none declared"
    return ", ".join(str(name) for name in entry.assets)


def model_supported_opts_label(entry: Manifest) -> str:
    if not entry.supported_optimizations:
        return "none"
    return ", ".join(entry.supported_optimizations)
