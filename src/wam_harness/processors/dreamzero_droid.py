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
    "predicted_video",
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


class DreamZeroProcessorError(ValueError):
    """Raised when a DROID observation cannot be converted for DreamZero."""


class DreamZeroDroidProcessor:
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest
        observation_config = manifest.processor.get("observation", {})
        action_config = manifest.processor.get("action", {})
        self.image_views = _as_str_list(
            observation_config.get("image_views"), ["right", "left", "wrist"]
        )
        self.state_key = str(observation_config.get("state", "joint_position_and_gripper"))
        self.prompt_source = str(observation_config.get("prompt", "task_instruction"))
        self.declared_action_horizon = action_config.get("horizon")
        self.declared_action_dim = action_config.get("dim")

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> DreamZeroDroidProcessor:
        return cls(manifest)

    def to_model_inputs(self, observation: Observation) -> dict[str, Any]:
        payload = {
            "observation/exterior_image_0_left": self._select_image(observation.images, "right"),
            "observation/exterior_image_1_left": self._select_image(observation.images, "left"),
            "observation/wrist_image_left": self._select_image(observation.images, "wrist"),
            "prompt": observation.prompt or str(observation.metadata.get("task", "")),
        }
        joint_position, gripper_position = self._extract_state(observation.state)
        payload["observation/joint_position"] = joint_position
        payload["observation/gripper_position"] = gripper_position
        session_id = observation.session.get("session_id") or observation.session.get("episode_id")
        if session_id is not None:
            payload["session_id"] = str(session_id)
        return payload

    def to_harness_result(self, raw_output: object) -> InferenceResult:
        try:
            action_chunk = action_chunk_from_raw(raw_output, label="DreamZero")
        except ValueError as exc:
            raise DreamZeroProcessorError(str(exc)) from exc
        return InferenceResult(
            action_chunk=action_chunk,
            future_frames=summarize_future_output(raw_output, _FUTURE_OUTPUT_KEYS),
            value=summarize_value_output(raw_output, _VALUE_OUTPUT_KEYS),
            backend_metadata={"processor": "dreamzero_droid"},
        )

    def modality_limits(self) -> dict[str, object]:
        return {
            "processor": "dreamzero_droid",
            "images": self.image_views,
            "state": self.state_key,
            "prompt": self.prompt_source,
            "action_horizon": self.declared_action_horizon,
            "action_dim": self.declared_action_dim,
            "transport": "websocket_policy_server",
        }

    def smoke_observation(self) -> Observation:
        return Observation(
            images={
                "right": rgb_image([24, 48, 96]),
                "left": rgb_image([96, 48, 24]),
                "wrist": rgb_image([48, 96, 24]),
            },
            state={
                "joint_position": [0.0] * 7,
                "gripper_position": [0.0],
            },
            prompt="open the drawer",
            session={"episode_id": 0, "session_id": "native-smoke"},
        )

    def _select_image(self, images: dict[str, Any], view: str) -> Any:
        if view == "right":
            aliases = ("right", "exterior_image_0_left", "external_cam", "external_cam_0")
        elif view == "left":
            aliases = ("left", "exterior_image_1_left", "external_cam_2", "external_cam_1")
        elif view == "wrist":
            aliases = ("wrist", "wrist_image_left", "wrist_cam")
        else:
            aliases = (view,)
        for key in aliases:
            if key in images:
                return _to_rgb_array(images[key])
        raise DreamZeroProcessorError(
            f"DreamZero DROID observation is missing image view '{view}'. "
            f"Accepted keys: {', '.join(aliases)}."
        )

    def _extract_state(self, state: dict[str, Any]) -> tuple[Any, Any]:
        np = importlib.import_module("numpy")
        if "joint_position" in state:
            joint_position = np.asarray(state["joint_position"], dtype=np.float32)
        elif "observation/joint_position" in state:
            joint_position = np.asarray(state["observation/joint_position"], dtype=np.float32)
        else:
            raise DreamZeroProcessorError(
                "DreamZero DROID state must contain `joint_position` or "
                "`observation/joint_position`."
            )

        if "gripper_position" in state:
            gripper_position = np.asarray(state["gripper_position"], dtype=np.float32)
        elif "observation/gripper_position" in state:
            gripper_position = np.asarray(state["observation/gripper_position"], dtype=np.float32)
        else:
            gripper_position = np.zeros((1,), dtype=np.float32)

        return joint_position, gripper_position


def _to_rgb_array(value: Any) -> Any:
    np = importlib.import_module("numpy")
    array = np.asarray(value)
    if array.ndim != 3 and array.ndim != 4:
        raise DreamZeroProcessorError(
            f"Expected RGB image or frame stack with 3/4 dims, got shape {array.shape}."
        )
    if array.ndim == 3 and array.shape[0] in {3, 4} and array.shape[-1] not in {3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise DreamZeroProcessorError(f"Expected RGB image last dim 3, got shape {array.shape}.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_str_list(value: object, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
