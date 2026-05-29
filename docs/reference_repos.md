# Reference Repositories

This harness is not a copy of any one WAM repository. It borrows design lessons
from several inference systems.

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
management, and dry-run benchmarks.

## VibeTensor

Useful as a systems research artifact reference. The harness should borrow its
measurement-first discipline, baseline-vs-variant checks, allocator and CUDA
Graph observability mindset, and warning that local component correctness does
not guarantee global system performance.

The harness should not borrow VibeTensor's self-contained tensor runtime,
autograd engine, C++/CUDA extension stack, Node.js frontend, or production
allocator implementation.
