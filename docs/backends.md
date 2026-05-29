# Backend Notes

Backends hide model-specific loading, preprocessing, inference, postprocessing,
and deployment details.

Some upstream WAM projects use adapter-like wrappers. In this harness, the
preferred term is `backend` because the boundary is closer to vLLM/SGLang
backend registration than to a temporary compatibility patch.

## Planned Model Backends

- Fake: first target for testing the harness contract.
- FastWAM: first real local inference target.
- Remote policy: WebSocket/msgpack compatible policy server target.
- Cosmos-Policy: future prediction and value output target.
- Motus: standalone smoke-test target.

## Planned Workloads

- Open-loop: load a fixed observation and run inference without a simulator.
- LIBERO: simulator evaluation.
- RoboTwin: simulator evaluation.

## Backend Compatibility Goal

A new WAM should require a new backend and config file, not changes to the core
runner.

Backends conform to the harness contract. They should not force backend-native
tensor layouts, key names, normalization details, or cache mechanics into the
core runner.
