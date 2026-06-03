# Native Backend Goal Status

Migrate official script wrappers into native backends.

Status: native smoke has now passed for FastWAM, Cosmos-Policy, and DreamZero.
This document remains as the record of the migration goal and the next hardening
items.

The public alpha entry surface is useful, but the core product will remain weak
if real models are executed primarily by shelling out to each upstream official
script. Official scripts should become reference evaluators. Native backends
should become the product path.

## Objective

Build a clear, extensible native backend path:

```text
model entry -> native backend -> processor -> workload -> trace
```

FastWAM, Cosmos-Policy, and DreamZero now follow the same core contract.
DreamZero uses a harness-owned resident policy server rather than a pure
in-process PyTorch module.

## Design Requirements

- Keep official scripts as reference evaluators for parity and reproduction.
- Do not put model-specific branches in the core runner.
- Make processors first-class registered objects.
- Keep camera/state/prompt/normalizer/action conversion inside processors.
- Keep simulator loops in workloads, not inside backends.
- Make optimization profiles attach to native backend lifecycle points.
  The native runtime now records a shared optimization plan with profile scope
  and target layer. DreamZero `dit_cache` and Cosmos `parallel_inference` are
  mapped from optimization context on the native path.
- Require trace-visible fallback for `torch_compile` and `cuda_graph`.

## Immediate Tasks

### 1. Backend Architecture

- Document the native backend design and migration philosophy. Done.
- Make `external_eval` explicitly a reference mode, not the product backend.
  Done in docs; public manifests still keep reference-mode entries until native
  parity exists.
- Add processor registry support so model entries can resolve processor names.
  Done.
- Add shared native backend support for repo discovery, asset resolution, and
  runtime metadata. Done via `NativeBackendBase`.

### 2. Native Backend Skeletons

- Add `FastWAMBackend` with `load`, `warmup`, `reset`, `infer`, and
  `runtime_info`. Skeleton is in place and dynamically imports FastWAM only
  inside `load`.
- Add `FastWAMLiberoProcessor`. Skeleton is in place and owns LIBERO image,
  proprio, prompt, action denormalization, and gripper conversion logic.
- Add `CosmosPolicyBackend` and `CosmosPolicyLiberoProcessor`. Skeleton is in
  place and uses upstream `get_model`, dataset stats, T5 embeddings, and
  `get_action` as the native inference boundary.
- Add `DreamZeroBackend` and `DreamZeroDroidProcessor`. Skeleton is in place as
  a harness-owned resident WebSocket policy server backend.
- Fail early with clear asset/runtime errors if upstream code or assets are not
  available. Covered by tests for missing upstream repo.
- Keep official eval/server scripts as reference mode until native parity exists.

### 3. Native Smoke And Parity

- Use `wam native-smoke <model-id>` as the migration-time entry for validating
  native backends, while official simulator scripts require explicit
  `wam eval <model-id> --reference`.
- Keep `native-smoke` extensible: the native backend key must come from
  `backend.config.native_backend`, and the synthetic observation must come from
  the registered processor.
- Validate the native backend inside the FastWAM container with upstream
  dependencies, checkpoint, and dataset stats mounted.
- Validate Cosmos-Policy native backend inside the Cosmos container with
  checkpoint, dataset stats, and text embeddings mounted.
- Validate DreamZero resident backend inside the DreamZero container with a
  job-local policy server and one fixed DROID-style observation.
- Run fixed smoke observations through each native backend.
- Check action chunk shape and value range.
- Compare against official reference where feasible.
- Emit trace fields for preprocess/model/postprocess timing.

Container smoke commands:

```bash
wam native-smoke fastwam-libero --upstream-dir /path/to/FastWAM --require-ready
wam native-smoke cosmos-policy-libero --upstream-dir /path/to/cosmos-policy --require-ready
wam native-smoke dreamzero-droid-sim --upstream-dir /path/to/dreamzero --require-ready
```

These commands are maintainer validation commands. They should not replace the
simple public path of `wam info`, `wam doctor`, `wam prepare`, and `wam run`.

### 4. Optimization Profiles

- Add `torch_compile` only after the native smoke path is stable.
- Add `cuda_graph` only after static shape and warmup/capture requirements are
  explicit.
- Both profiles must be optional, disableable, and trace-backed.

## Non-Goals

- Do not rewrite upstream model architectures.
- Do not vendor upstream repositories into core.
- Do not delete official reference scripts before native parity exists.
- Do not default-enable experimental optimizations.
- Do not make `prepare` install environments.

## Done Criteria

This goal is done for native smoke once FastWAM, Cosmos-Policy, and DreamZero
emit action chunks through the harness contract, with official script execution
retained as a reference evaluator and enough trace metadata to support later
optimization profiles. The next goals are simulator eval parity, resident
serving polish, and measured optimization profiles.
