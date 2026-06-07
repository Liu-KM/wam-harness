from __future__ import annotations

import importlib
import math
from typing import Any

from eazywam.core.types import ActionChunk, InferenceResult, Manifest, Observation
from eazywam.processors.smoke import rgb_image

DEFAULT_FASTWAM_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


class FastWAMProcessorError(ValueError):
    """Raised when a LIBERO observation cannot be converted for FastWAM."""


class FastWAMLiberoProcessor:
    """Processor for FastWAM LIBERO observations and action chunks.

    Heavy dependencies such as torch, numpy, PIL, and upstream FastWAM are loaded
    only when runtime conversion is requested.
    """

    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest
        observation_config = manifest.processor.get("observation", {})
        action_config = manifest.processor.get("action", {})
        self.image_views = _as_str_list(observation_config.get("image_views"), ["primary", "wrist"])
        self.state_key = str(observation_config.get("state", "proprio"))
        self.prompt_source = str(observation_config.get("prompt", "task"))
        self.declared_action_dim = action_config.get("dim")
        self.upstream_processor: Any | None = None
        self.cfg: Any | None = None
        self.device = str(manifest.defaults.get("device", "cuda"))
        self.dtype: Any | None = None
        self.prompt_template = DEFAULT_FASTWAM_PROMPT

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> FastWAMLiberoProcessor:
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
        assert self.cfg is not None

        torch = importlib.import_module("torch")

        image_meta = self.upstream_processor.shape_meta["images"]
        if len(image_meta) < int(self.upstream_processor.num_output_cameras):
            raise FastWAMProcessorError(
                f"shape_meta.images has {len(image_meta)} entries, but "
                f"num_output_cameras={self.upstream_processor.num_output_cameras}."
            )

        input_h, input_w = self._input_height_width()
        rgb = self._build_concatenated_rgb(observation.images, image_meta)
        actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
        if (actual_h, actual_w) != (input_h, input_w):
            raise FastWAMProcessorError(
                "FastWAM LIBERO image size mismatch after resize/concat: "
                f"got HxW=({actual_h},{actual_w}), expected ({input_h},{input_w})."
            )

        input_image = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        input_image = input_image.to(device=self.device, dtype=self.dtype)
        input_image = input_image * (2.0 / 255.0) - 1.0

        proprio = self._normalize_proprio(self._extract_proprio(observation.state))
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
        action = self._denormalize_action(action)
        action = action[0]

        np = importlib.import_module("numpy")
        action[..., -1] = action[..., -1] * 2 - 1
        action[..., -1] = action[..., -1] * -1.0
        if bool(self._evaluation_value("binarize_gripper", False)):
            action[..., -1] = np.sign(action[..., -1])

        actions = [[float(value) for value in row] for row in action.tolist()]
        return InferenceResult(
            action_chunk=ActionChunk(actions=actions),
            future_frames=_future_frames_summary(raw_output),
            backend_metadata={
                "processor": "fastwam_libero",
                "action_denormalized": True,
            },
        )

    def modality_limits(self) -> dict[str, object]:
        return {
            "processor": "fastwam_libero",
            "images": self.image_views,
            "state": self.state_key,
            "prompt": self.prompt_source,
            "action_dim": self.declared_action_dim,
            "requires_runtime_binding": self.upstream_processor is None,
        }

    def smoke_observation(self) -> Observation:
        return Observation(
            images={
                "primary": rgb_image([32, 64, 96]),
                "wrist": rgb_image([96, 64, 32]),
            },
            state={
                "robot0_eef_pos": [0.0, 0.0, 0.0],
                "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
                "robot0_gripper_qpos": [0.0, 0.0],
            },
            prompt="open the drawer",
            session={"episode_id": 0, "session_id": "native-smoke"},
        )

    def _build_concatenated_rgb(self, images: dict[str, Any], image_meta: list[dict[str, Any]]) -> Any:
        num_cameras = int(self.upstream_processor.num_output_cameras)
        if num_cameras == 1:
            primary_h, primary_w = self._meta_height_width(image_meta[0], camera_idx=0)
            return self._center_crop_resize(
                self._select_image(images, "primary"),
                width=primary_w,
                height=primary_h,
            )
        if num_cameras == 2:
            np = importlib.import_module("numpy")
            primary_h, primary_w = self._meta_height_width(image_meta[0], camera_idx=0)
            wrist_h, wrist_w = self._meta_height_width(image_meta[1], camera_idx=1)
            primary = self._center_crop_resize(
                self._select_image(images, "primary"),
                width=primary_w,
                height=primary_h,
            )
            wrist = self._center_crop_resize(
                self._select_image(images, "wrist"),
                width=wrist_w,
                height=wrist_h,
            )
            concatenation = str(self.cfg.data.train.get("concat_multi_camera", "horizontal"))
            if concatenation == "horizontal":
                return np.concatenate([primary, wrist], axis=1)
            if concatenation == "vertical":
                return np.concatenate([primary, wrist], axis=0)
            raise FastWAMProcessorError(f"Unsupported FastWAM camera concat mode: {concatenation}")
        raise FastWAMProcessorError(
            f"FastWAM LIBERO supports one or two output cameras, got {num_cameras}."
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
                array = self._to_rgb_array(images[key])
                if key in {"agentview_image", "robot0_eye_in_hand_image"}:
                    array = np.ascontiguousarray(array[::-1, ::-1])
                return array
        raise FastWAMProcessorError(
            f"FastWAM LIBERO observation is missing image view '{view}'. "
            f"Accepted keys: {', '.join(aliases)}."
        )

    def _to_rgb_array(self, value: Any) -> Any:
        np = importlib.import_module("numpy")
        array = np.asarray(value)
        if array.ndim != 3:
            raise FastWAMProcessorError(f"Expected RGB image with 3 dims, got shape {array.shape}.")
        if array.shape[0] in {3, 4} and array.shape[-1] not in {3, 4}:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            raise FastWAMProcessorError(f"Expected RGB image last dim 3, got shape {array.shape}.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def _center_crop_resize(self, image: Any, *, width: int, height: int) -> Any:
        np = importlib.import_module("numpy")
        pil_module = importlib.import_module("PIL.Image")
        pil_image = pil_module.fromarray(image)
        src_w, src_h = pil_image.size
        scale = max(width / src_w, height / src_h)
        resampling = getattr(pil_module, "Resampling", pil_module)
        resized = pil_image.resize(
            (round(src_w * scale), round(src_h * scale)),
            resample=resampling.BILINEAR,
        )
        resized_w, resized_h = resized.size
        left = max((resized_w - width) // 2, 0)
        top = max((resized_h - height) // 2, 0)
        cropped = resized.crop((left, top, left + width, top + height))
        return np.asarray(cropped, dtype=np.uint8)

    def _extract_proprio(self, state: dict[str, Any]) -> Any:
        np = importlib.import_module("numpy")
        if "proprio" in state:
            return np.asarray(state["proprio"], dtype=np.float32)
        if "default" in state:
            return np.asarray(state["default"], dtype=np.float32)
        required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
        if all(key in state for key in required):
            return np.concatenate(
                (
                    np.asarray(state["robot0_eef_pos"], dtype=np.float32),
                    self._quat_to_axis_angle(np.asarray(state["robot0_eef_quat"], dtype=np.float32)),
                    np.asarray(state["robot0_gripper_qpos"], dtype=np.float32),
                )
            ).astype(np.float32)
        raise FastWAMProcessorError(
            "FastWAM LIBERO state must contain `proprio`, `default`, or LIBERO simulator keys: "
            "robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos."
        )

    def _normalize_proprio(self, proprio: Any) -> Any:
        assert self.upstream_processor is not None
        torch = importlib.import_module("torch")
        state_meta = self.upstream_processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise FastWAMProcessorError(
                "FastWAM LIBERO expects a single merged state key in shape_meta['state']."
            )
        state_key = state_meta[0]["key"]
        state_batch = {
            "state": {
                state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)
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
            raise FastWAMProcessorError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}.")
        action_meta = self.upstream_processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise FastWAMProcessorError(
                "FastWAM LIBERO expects a single merged action key in shape_meta['action']."
            )
        action_key = action_meta[0]["key"]
        normalizer = self.upstream_processor.normalizer.normalizers["action"][action_key]
        action = action.to(dtype=torch.float32, device="cpu")
        return normalizer.backward(action).numpy()

    def _extract_raw_action(self, raw_output: object) -> Any:
        if not isinstance(raw_output, dict) or "action" not in raw_output:
            raise FastWAMProcessorError("FastWAM raw output must be a mapping with key `action`.")
        return raw_output["action"]

    def _input_height_width(self) -> tuple[int, int]:
        video_size = self.cfg.data.train.get("video_size", [224, 224])
        if len(video_size) != 2:
            raise FastWAMProcessorError(f"FastWAM data.train.video_size must be [H,W], got {video_size}.")
        return int(video_size[0]), int(video_size[1])

    def _meta_height_width(self, meta: dict[str, Any], *, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise FastWAMProcessorError(
                f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}."
            )
        return int(shape[1]), int(shape[2])

    def _evaluation_value(self, key: str, default: Any) -> Any:
        if self.cfg is None:
            return default
        return self.cfg.get("EVALUATION", {}).get(key, default)

    def _require_runtime(self) -> None:
        if self.upstream_processor is None or self.cfg is None:
            raise FastWAMProcessorError(
                "FastWAMLiberoProcessor is not bound to an upstream FastWAM runtime. "
                "Call FastWAMBackend.load() before inference."
            )

    def _quat_to_axis_angle(self, quat: Any) -> Any:
        np = importlib.import_module("numpy")
        quat = quat.copy()
        quat[3] = min(max(float(quat[3]), -1.0), 1.0)
        den = math.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3, dtype=np.float32)
        return (quat[:3] * 2.0 * math.acos(quat[3]) / den).astype(np.float32)


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
