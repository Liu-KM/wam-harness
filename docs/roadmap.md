# Roadmap

## Phase 0: Skeleton

- Repository structure.
- Design notes.
- Empty package directories.
- Configuration and example placeholders.
- Agent instructions and uv environment metadata.

## Phase 0.5: Experiment Contract

- WAM systems experiment contract.
- Systems idea transfer map.
- Trace schema tied to baseline-vs-variant experiments.
- Phase 1 implementation plan.

## Phase 1: Minimal Open-Loop Harness

- Core contract types.
- JSONL trace writer.
- Memory observer.
- Fake backend.
- Open-loop workload.
- Runner with action chunk scheduling.
- Minimal CLI and tests.

## Phase 2: First Real Backend

- FastWAM local backend.
- Open-loop real-model smoke test.
- Baseline traces for action chunk inference.

## Phase 3: Remote Inference

- WebSocket client backend.
- Optional policy server wrapper.
- Server timing propagation.
- Session reset support.

## Phase 4: Simulator Integration

- LIBERO workload integration.
- RoboTwin workload integration.
- Episode-level summaries.

## Phase 5: Systems Idea Experiments

- torch.compile experiment.
- CUDA Graph experiment.
- cache/history reuse experiment.
- remote inference overhead experiment.
- action chunk scheduling experiment.
