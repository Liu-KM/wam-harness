from __future__ import annotations

from eazywam.core.types import InferenceResult, Manifest, Observation
from eazywam.processors.smoke import rgb_image, zero_vector


class PassthroughProcessor:
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> "PassthroughProcessor":
        return cls(manifest)

    def to_model_inputs(self, observation: Observation) -> Observation:
        return observation

    def to_harness_result(self, raw_output: object) -> InferenceResult:
        if not isinstance(raw_output, InferenceResult):
            raise TypeError("passthrough processor expects an InferenceResult")
        return raw_output

    def modality_limits(self) -> dict[str, object]:
        return {
            "processor": self.manifest.processor_name,
            "observation": self.manifest.processor.get("observation", {}),
            "action": self.manifest.processor.get("action", {}),
        }

    def smoke_observation(self) -> Observation:
        config = self.manifest.workload.get("config", {})
        image_shapes = config.get("image_shapes", {})
        state_dims = config.get("state_dims", {})
        images = {}
        if isinstance(image_shapes, dict):
            for key, shape in image_shapes.items():
                height, width = _image_height_width(shape)
                images[str(key)] = rgb_image([0, 0, 0], width=width, height=height)
        state = {}
        if isinstance(state_dims, dict):
            for key, dim in state_dims.items():
                state[str(key)] = zero_vector(dim)
        return Observation(
            images=images or {"primary": rgb_image([0, 0, 0])},
            state=state,
            prompt=str(config.get("prompt", "smoke observation")),
            session={"episode_id": 0, "session_id": "native-smoke"},
        )


def _image_height_width(shape: object) -> tuple[int, int]:
    if isinstance(shape, list) and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return 8, 8
