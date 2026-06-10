from __future__ import annotations

import importlib
from typing import Any

from eazywam.core.types import ActionChunk, InferenceResult, Manifest, Observation
from eazywam.processors.fastwam_libero import DEFAULT_FASTWAM_PROMPT
from eazywam.processors.smoke import rgb_image


class FastWAMRobotWinProcessorError(ValueError):
    """Raised when a RoboTwin observation cannot be converted for FastWAM."""


class FastWAMRobotWinProcessor:
    """Processor for FastWAM RoboTwin observations and action chunks."""

    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest
        observation_config = manifest.processor.get("observation", {})
        action_config = manifest.processor.get("action", {})
        self.image_views = _as_str_list(
            observation_config.get("image_views"),
            ["head", "left_wrist", "right_wrist"],
        )
        self.state_key = str(observation_config.get("state", "joint_action"))
        self.prompt_source = str(observation_config.get("prompt", "task"))
        self.declared_action_dim = action_config.get("dim")
        self.upstream_processor: Any | None = None
        self.cfg: Any | None = None
        self.device = str(manifest.defaults.get("device", "cuda"))
        self.dtype: Any | None = None
        self.prompt_template = DEFAULT_FASTWAM_PROMPT

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> FastWAMRobotWinProcessor:
        return cls(manifest)

    def bind_runtime(
        self,
        *,
        upstream_processor: Any,
        cfg: Any,
        device: str,
        dtype: Any,
        prompt_template: str,
    ) -> None:
        self.upstream_processor = upstream_processor
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.prompt_template = prompt_template

    def to_model_inputs(self, observation: Observation) -> dict[str, Any]:
        self._require_runtime()
        assert self.upstream_processor is not None

        torch = importlib.import_module("torch")

        rgb = self._build_robotwin_rgb(observation.images)
        input_image = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        input_image = input_image.to(device=self.device, dtype=self.dtype)
        input_image = input_image * (2.0 / 255.0) - 1.0

        proprio = self._normalize_state(self._extract_state(observation.state))
        task = observation.prompt or str(observation.metadata.get("task", ""))
        prompt = self.prompt_template.format(task=task)
        return {
            "prompt": prompt,
            "input_image": input_image,
            "proprio": proprio,
        }

    def to_harness_result(self, raw_output: object) -> InferenceResult:
        self._require_runtime()
        action = self._extract_raw_action(raw_output)
        action = self._denormalize_action(action)[0]
        actions = [[float(value) for value in row] for row in action.tolist()]
        return InferenceResult(
            action_chunk=ActionChunk(actions=actions),
            future_frames=_future_frames_summary(raw_output),
            backend_metadata={
                "processor": "fastwam_robotwin",
                "action_denormalized": True,
                "action_type": "qpos",
            },
        )

    def modality_limits(self) -> dict[str, object]:
        return {
            "processor": "fastwam_robotwin",
            "images": self.image_views,
            "state": self.state_key,
            "prompt": self.prompt_source,
            "action_dim": self.declared_action_dim,
            "requires_runtime_binding": self.upstream_processor is None,
        }

    def smoke_observation(self) -> Observation:
        return Observation(
            images={
                "head": rgb_image([32, 64, 96]),
                "left_wrist": rgb_image([96, 64, 32]),
                "right_wrist": rgb_image([64, 96, 128]),
            },
            state={
                "joint_action": {
                    "vector": [0.0] * 14,
                },
            },
            prompt="click the alarm clock",
            session={"episode_id": 0, "session_id": "native-smoke"},
        )

    def _build_robotwin_rgb(self, images: dict[str, Any]) -> Any:
        np = importlib.import_module("numpy")
        head = self._resize_rgb(self._select_image(images, "head"), width=320, height=256)
        left = self._resize_rgb(self._select_image(images, "left_wrist"), width=160, height=128)
        right = self._resize_rgb(self._select_image(images, "right_wrist"), width=160, height=128)
        bottom = np.concatenate([left, right], axis=1)
        rgb = np.concatenate([head, bottom], axis=0)
        expected_h, expected_w = self._input_height_width()
        actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
        if (actual_h, actual_w) != (expected_h, expected_w):
            raise FastWAMRobotWinProcessorError(
                "FastWAM RoboTwin image size mismatch after resize/concat: "
                f"got HxW=({actual_h},{actual_w}), expected ({expected_h},{expected_w})."
            )
        return rgb

    def _select_image(self, images: dict[str, Any], view: str) -> Any:
        aliases = {
            "head": ("head", "high", "cam_high", "head_camera", "primary"),
            "left_wrist": ("left_wrist", "cam_left_wrist", "left_camera", "left"),
            "right_wrist": ("right_wrist", "cam_right_wrist", "right_camera", "right"),
        }[view]
        for key in aliases:
            if key not in images:
                continue
            value = images[key]
            if isinstance(value, dict) and "rgb" in value:
                value = value["rgb"]
            return self._to_rgb_array(value)
        raise FastWAMRobotWinProcessorError(
            f"FastWAM RoboTwin observation is missing image view '{view}'. "
            f"Accepted keys: {', '.join(aliases)}."
        )

    def _to_rgb_array(self, value: Any) -> Any:
        np = importlib.import_module("numpy")
        array = np.asarray(value)
        if array.ndim != 3:
            raise FastWAMRobotWinProcessorError(
                f"Expected RGB image with 3 dims, got shape {array.shape}."
            )
        if array.shape[0] in {3, 4} and array.shape[-1] not in {3, 4}:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            raise FastWAMRobotWinProcessorError(
                f"Expected RGB image last dim 3, got shape {array.shape}."
            )
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def _resize_rgb(self, image: Any, *, width: int, height: int) -> Any:
        np = importlib.import_module("numpy")
        pil_module = importlib.import_module("PIL.Image")
        pil_image = pil_module.fromarray(image)
        resampling = getattr(pil_module, "Resampling", pil_module)
        resized = pil_image.resize((width, height), resample=resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)

    def _extract_state(self, state: dict[str, Any]) -> Any:
        np = importlib.import_module("numpy")
        for key in ("joint_action", "proprio", "default", "vector"):
            if key not in state:
                continue
            value = state[key]
            if isinstance(value, dict) and "vector" in value:
                value = value["vector"]
            array = np.asarray(value, dtype=np.float32)
            if array.shape[-1] != 14:
                raise FastWAMRobotWinProcessorError(
                    f"FastWAM RoboTwin state '{key}' must have last dim 14, got {array.shape}."
                )
            return array
        raise FastWAMRobotWinProcessorError(
            "FastWAM RoboTwin state must contain `joint_action.vector`, "
            "`proprio`, `default`, or `vector` with 14 values."
        )

    def _normalize_state(self, state: Any) -> Any:
        assert self.upstream_processor is not None
        torch = importlib.import_module("torch")
        state_meta = self.upstream_processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise FastWAMRobotWinProcessorError(
                "FastWAM RoboTwin expects a single merged state key in shape_meta['state']."
            )
        state_key = state_meta[0]["key"]
        state_batch = {
            "state": {
                state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            }
        }
        state_batch = self.upstream_processor.action_state_transform(state_batch)
        state_batch = self.upstream_processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: Any) -> Any:
        assert self.upstream_processor is not None
        torch = importlib.import_module("torch")
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise FastWAMRobotWinProcessorError(
                f"Expected action tensor [B,T,D], got {tuple(action.shape)}."
            )
        action_meta = self.upstream_processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise FastWAMRobotWinProcessorError(
                "FastWAM RoboTwin expects a single merged action key in shape_meta['action']."
            )
        action_key = action_meta[0]["key"]
        normalizer = self.upstream_processor.normalizer.normalizers["action"][action_key]
        action = action.to(dtype=torch.float32, device="cpu")
        return normalizer.backward(action).numpy()

    def _extract_raw_action(self, raw_output: object) -> Any:
        if not isinstance(raw_output, dict) or "action" not in raw_output:
            raise FastWAMRobotWinProcessorError(
                "FastWAM raw output must be a mapping with key `action`."
            )
        return raw_output["action"]

    def _input_height_width(self) -> tuple[int, int]:
        if self.cfg is None:
            return 384, 320
        video_size = self.cfg.data.train.get("video_size", [384, 320])
        if len(video_size) != 2:
            raise FastWAMRobotWinProcessorError(
                f"FastWAM data.train.video_size must be [H,W], got {video_size}."
            )
        return int(video_size[0]), int(video_size[1])

    def _require_runtime(self) -> None:
        if self.upstream_processor is None or self.cfg is None:
            raise FastWAMRobotWinProcessorError(
                "FastWAMRobotWinProcessor is not bound to an upstream FastWAM runtime. "
                "Call FastWAMBackend.load() before inference."
            )


def _as_str_list(value: object, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _future_frames_summary(raw_output: object) -> dict[str, object] | None:
    if not isinstance(raw_output, dict) or "video" not in raw_output:
        return None
    video = raw_output.get("video")
    if not isinstance(video, list):
        return {
            "present": video is not None,
            "count": None,
            "format": type(video).__name__,
        }
    frame_types = sorted({type(frame).__name__ for frame in video})
    return {
        "present": len(video) > 0,
        "count": len(video),
        "format": "frame_list",
        "frame_types": frame_types,
    }
