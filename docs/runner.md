# Runner Contract

The runner drives a backend at replan points, caches action chunks, consumes
actions step by step, and emits trace events. It only uses harness contract
terms and never branches on upstream repository names.

## State

Within one episode the runner maintains:

- `pending_actions`: actions from the current action chunk that have not been
  consumed.
- `steps_since_replan`: steps executed since the last inference call.
- `replan_id`: monotonically increasing inference counter.

## Pseudocode

```text
run(model_id, overrides):
  model_entry = resolve_model_entry(model_id)
  config = merge(model_entry.defaults, overrides)
  profiles = registry.build_optimization_profiles(model_entry, config.enabled_opts)
  backend = registry.create_backend(model_entry, profiles)
  processor = registry.create_processor(model_entry)
  workload = registry.create_workload(model_entry)

  runtime_info = backend.load()
  emit run_start(runtime_info, model_entry, config)

  for episode in workload.episodes():
    backend.reset(seed=config.seed)
    workload.reset(episode)
    pending_actions = []
    steps_since_replan = 0

    backend.warmup(sample_request)

    for step in range(episode.steps):
      if need_replan(pending_actions, steps_since_replan, config.replan_steps):
        obs = workload.observe()
        request = build_request(obs, config, optimization_profiles)
        emit replan_start(replan_id)
        emit inference_start(replan_id, workload_fields)
        result = backend.infer(request)
        persist_artifacts_if_enabled(result)
        emit inference_end(replan_id, timing, artifact_paths, shape)
        emit memory_sample(sample_memory())
        pending_actions = list(result.action_chunk)
        steps_since_replan = 0
        replan_id += 1

      action = pending_actions.pop_front()
      from_stale = steps_since_replan > 0
      workload.step(action)
      emit step(step_id, env_step_ms, from_stale_chunk=from_stale)
      steps_since_replan += 1

    emit episode_end(episode_summary)

  emit run_end(run_summary)
```

## Replan Rule

```text
need_replan(pending, steps_since_replan, replan_steps):
  if len(pending) == 0:
    return True
  if steps_since_replan >= replan_steps:
    return True
  return False
```

## Boundary Behavior

| Case | Expected behavior |
|---|---|
| `action_horizon > replan_steps` | Replan after `replan_steps`; unused tail actions are dropped and counted. |
| `action_horizon < replan_steps` | Replan when the chunk is exhausted. |
| `action_horizon == replan_steps` | Consume the whole chunk, then replan. |
| `action_horizon == 1` | Replan every step. |
| backend returns empty chunk | Emit unrecoverable `error` and stop the episode. |
| episode changes | Call `backend.reset(seed=...)`; pending actions and cache must not leak. |
| stale action | `step` event sets `from_stale_chunk=true` when consuming a non-fresh action. |
| CPU-only runtime | CUDA fields remain present with `0` or `null`. |

## Invariants

- The runner depends on interfaces, not concrete backend modules.
- A model id resolves through the model-entry registry, not runner branches.
- Trace writing is append-only during a run.
- Large outputs are artifacts, not embedded JSON.
