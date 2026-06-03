from __future__ import annotations

from dataclasses import dataclass, field, replace

from wam_harness.core.registry import Processor
from wam_harness.core.types import Manifest, Observation


@dataclass
class ProcessorSmokeWorkload:
    observation_template: Observation
    episode_length: int
    replan_interval: int
    episode_id: int = 0
    step_id: int = 0
    steps_since_replan: int = 0
    consumed_actions: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_processor(
        cls,
        manifest: Manifest,
        processor: Processor,
    ) -> "ProcessorSmokeWorkload":
        config = manifest.workload.get("config", {})
        defaults = manifest.defaults
        action_horizon = int(defaults.get("action_horizon") or 1)
        return cls(
            observation_template=processor.smoke_observation(),
            episode_length=int(config.get("episode_length", 1)),
            replan_interval=int(defaults.get("replan_steps") or action_horizon),
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
        session = {
            **self.observation_template.session,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
        }
        metadata = {
            **self.observation_template.metadata,
            "workload": "processor_smoke",
            "synthetic_observation": True,
        }
        return replace(self.observation_template, session=session, metadata=metadata)

    def mark_replan(self) -> None:
        self.steps_since_replan = 0

    def step(self, action: list[float]) -> None:
        self.consumed_actions.append(action)
        self.step_id += 1
        self.steps_since_replan += 1
        if self.steps_since_replan > self.replan_interval:
            self.steps_since_replan = self.replan_interval
