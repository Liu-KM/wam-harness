# Reference Repositories

This harness is not a copy of any one WAM repository. It borrows design lessons
from local model tools, inference systems, and WAM/VLA repositories.

## Ollama

Useful as the product reference for a local model UX: model ids, simple model
library commands, prepare/run/serve flow, and curated defaults that hide common
setup details.

WAM Harness cannot copy Ollama directly because WAM assets include checkpoints,
normalizers, action schemas, camera conventions, processors, and often simulator
dependencies. Curated model entries should provide the analogous default layer.

## LeRobot

Useful as the ecosystem reference for robot-learning datasets, policies, and
model distribution on Hugging Face. WAM Harness should complement this ecosystem
rather than replace it: use upstream assets and provide a deployment plus
inference-optimization wrapper.

## FastWAM

Useful for local model loading, action chunk inference, dataset statistics, and
LIBERO/RoboTwin evaluation structure.

## DreamZero

Useful for remote policy serving, session-aware inference, reset semantics, and
msgpack observation/action transport.

## Cosmos-Policy

Useful for configuration structure and richer WAM outputs such as future image
predictions and value predictions.

## LingBot-VA

Useful for server-client deployment, cache control, health checks, and server
timing metadata.

## Motus

Useful for a standalone smoke test that maps image and instruction inputs to
action chunks and predicted frames.

## Qi

Useful as a later optimization reference for CUDA Graph, torch.compile, cache
management, and no-execute planning benchmarks.
