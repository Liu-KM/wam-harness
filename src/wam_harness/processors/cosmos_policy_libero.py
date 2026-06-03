from __future__ import annotations

import importlib
from typing import Any

from wam_harness.core.types import InferenceResult, Manifest, Observation
from wam_harness.processors.output_summary import (
    action_chunk_from_raw,
    summarize_future_output,
    summarize_value_output,
)
from wam_harness.processors.smoke import rgb_image

_FUTURE_OUTPUT_KEYS = (
    "future_frames",
    "future_images",
    "future_video",
    "video",
    "future_state",
    "future_states",
    "future_observations",
    "future_predictions",
    "predicted_future",
)
_VALUE_OUTPUT_KEYS = (
    "value",
    "values",
    "value_prediction",
    "predicted_value",
    "q_value",
    "score",
    "scores",
    "reward",
    "rewards",
)


class CosmosPolicyProcessorError(ValueError):
    """Raised when a LIBERO observation cannot be converted for Cosmos-Policy."""


class CosmosPolicyLiberoProcessor:
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest
        observation_config = manifest.processor.get("observation", {})
        action_config = manifest.processor.get("action", {})
        self.image_views = _as_str_list(observation_config.get("image_views"), ["primary", "wrist"])
        self.state_key = str(observation_config.get("state", "proprio"))
        self.prompt_source = str(observation_config.get("prompt", "task"))
        self.declared_action_horizon = action_config.get("horizon")
        self.declared_action_dim = action_config.get("dim")

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> CosmosPolicyLiberoProcessor:
        return cls(manifest)

    def to_model_inputs(self, observation: Observation) -> dict[str, Any]:
        return {
            "observation": {
                "primary_image": self._select_image(observation.images, "primary"),
                "wrist_image": self._select_image(observation.images, "wrist"),
                "proprio": self._extract_proprio(observation.state),
            },
            "prompt": observation.prompt or str(observation.metadata.get("task", "")),
        }

    def to_harness_result(self, raw_output: object) -> InferenceResult:
        try:
            action_chunk = action_chunk_from_raw(raw_output, label="Cosmos-Policy")
        except ValueError as exc:
            raise CosmosPolicyProcessorError(str(exc)) from exc
        return InferenceResult(
            action_chunk=action_chunk,
            future_frames=summarize_future_output(raw_output, _FUTURE_OUTPUT_KEYS),
            value=summarize_value_output(raw_output, _VALUE_OUTPUT_KEYS),
            backend_metadata={"processor": "cosmos_policy_libero"},
        )

    def modality_limits(self) -> dict[str, object]:
        return {
            "processor": "cosmos_policy_libero",
            "images": self.image_views,
            "state": self.state_key,
            "prompt": self.prompt_source,
            "action_horizon": self.declared_action_horizon,
            "action_dim": self.declared_action_dim,
        }

    def smoke_observation(self) -> Observation:
        return Observation(
            images={
                "primary": rgb_image([16, 80, 128]),
                "wrist": rgb_image([128, 80, 16]),
            },
            state={
                "robot0_gripper_qpos": [0.0, 0.0],
                "robot0_eef_pos": [0.0, 0.0, 0.0],
                "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
            },
            prompt="open the top drawer and put the bowl inside",
            session={"episode_id": 0, "session_id": "native-smoke"},
        )

    def _select_image(self, images: dict[str, Any], view: str) -> Any:
        np = importlib.import_module("numpy")
        if view == "primary":
            aliases = ("primary", "image", "agentview_image")
        elif view == "wrist":
            aliases = ("wrist", "wrist_image", "robot0_eye_in_hand_image")
        else:
            aliases = (view,)

        for key in aliases:
            if key in images:
                array = _to_rgb_array(images[key])
                if key in {"agentview_image", "robot0_eye_in_hand_image"}:
                    array = np.ascontiguousarray(array[::-1, ::-1])
                return array
        raise CosmosPolicyProcessorError(
            f"Cosmos-Policy LIBERO observation is missing image view '{view}'. "
            f"Accepted keys: {', '.join(aliases)}."
        )

    def _extract_proprio(self, state: dict[str, Any]) -> Any:
        np = importlib.import_module("numpy")
        if "proprio" in state:
            return np.asarray(state["proprio"], dtype=np.float32)
        required = ("robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat")
        if all(key in state for key in required):
            return np.concatenate(
                (
                    np.asarray(state["robot0_gripper_qpos"], dtype=np.float32),
                    np.asarray(state["robot0_eef_pos"], dtype=np.float32),
                    np.asarray(state["robot0_eef_quat"], dtype=np.float32),
                )
            ).astype(np.float32)
        raise CosmosPolicyProcessorError(
            "Cosmos-Policy LIBERO state must contain `proprio` or LIBERO simulator keys: "
            "robot0_gripper_qpos, robot0_eef_pos, robot0_eef_quat."
        )


def _to_rgb_array(value: Any) -> Any:
    np = importlib.import_module("numpy")
    array = np.asarray(value)
    if array.ndim != 3:
        raise CosmosPolicyProcessorError(f"Expected RGB image with 3 dims, got shape {array.shape}.")
    if array.shape[0] in {3, 4} and array.shape[-1] not in {3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise CosmosPolicyProcessorError(f"Expected RGB image last dim 3, got shape {array.shape}.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_str_list(value: object, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
