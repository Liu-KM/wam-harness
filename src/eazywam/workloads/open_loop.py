from __future__ import annotations

from dataclasses import dataclass, field

from eazywam.core.types import Manifest, Observation


@dataclass
class OpenLoopWorkload:
    episode_length: int
    replan_interval: int
    prompt: str
    image_shapes: dict[str, list[int]]
    state_dims: dict[str, int]
    episode_id: int = 0
    step_id: int = 0
    steps_since_replan: int = 0
    consumed_actions: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> "OpenLoopWorkload":
        config = manifest.workload.get("config", {})
        defaults = manifest.defaults
        return cls(
            episode_length=int(config.get("episode_length", 6)),
            replan_interval=int(defaults.get("replan_steps", 1)),
            prompt=str(config.get("prompt", "move the fake end effector")),
            image_shapes={
                str(name): [int(value) for value in shape]
                for name, shape in config.get("image_shapes", {"primary": [64, 64, 3]}).items()
            },
            state_dims={
                str(name): int(dim)
                for name, dim in config.get("state_dims", {"proprio": 7}).items()
            },
        )

    @property
    def done(self) -> bool:
        return self.step_id >= self.episode_length

    def reset(self) -> None:
        self.episode_id += 1
        self.step_id = 0
        self.steps_since_replan = self.replan_interval
        self.consumed_actions = []

    def observation(self) -> Observation:
        state = {
            name: {
                "dim": dim,
                "values": [float(self.step_id)] * dim,
            }
            for name, dim in self.state_dims.items()
        }
        images = {
            name: {
                "shape": shape,
                "encoding": "shape-only",
            }
            for name, shape in self.image_shapes.items()
        }
        return Observation(
            images=images,
            prompt=self.prompt,
            state=state,
            session={
                "episode_id": self.episode_id,
                "step_id": self.step_id,
            },
            metadata={"workload": "open_loop"},
        )

    def mark_replan(self) -> None:
        self.steps_since_replan = 0

    def step(self, action: list[float]) -> None:
        self.consumed_actions.append(action)
        self.step_id += 1
        self.steps_since_replan += 1
        if self.steps_since_replan > self.replan_interval:
            self.steps_since_replan = self.replan_interval
