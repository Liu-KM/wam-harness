from __future__ import annotations

from dataclasses import dataclass, field, replace

from eazywam.core.types import Manifest, Observation


@dataclass
class SingleObservationWorkload:
    observation_template: Observation
    episode_length: int = 1
    episode_id: int = 0
    step_id: int = 0
    steps_since_replan: int = 1
    consumed_actions: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_observation(
        cls,
        manifest: Manifest,
        observation: Observation,
    ) -> "SingleObservationWorkload":
        config = manifest.workload.get("config", {})
        return cls(
            observation_template=observation,
            episode_length=int(config.get("episode_length", 1)),
        )

    @property
    def done(self) -> bool:
        return self.step_id >= self.episode_length

    def reset(self) -> None:
        self.episode_id += 1
        self.step_id = 0
        self.steps_since_replan = 1
        self.consumed_actions = []

    def observation(self) -> Observation:
        session = {
            **self.observation_template.session,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
        }
        metadata = {
            **self.observation_template.metadata,
            "workload": "single_observation",
            "external_observation": True,
        }
        return replace(self.observation_template, session=session, metadata=metadata)

    def mark_replan(self) -> None:
        self.steps_since_replan = 0

    def step(self, action: list[float]) -> None:
        self.consumed_actions.append(action)
        self.step_id += 1
        self.steps_since_replan += 1
