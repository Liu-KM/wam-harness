# Config Schema

WAM Harness has two configuration layers:

- Model entries define curated defaults for `wam run <model-id>` and
  `wam serve <model-id>`.
- Optional run configs override those defaults for advanced users and scripted
  benchmarks.

Common users should start with a model id, not a large YAML file.

## Run Config Shape

```yaml
run:
  name: fake-openloop
  output_dir: runs/fake-openloop

model:
  id: fake-open-loop
  model_spec: models/fake-open-loop.yaml

backend:
  name: fake
  device: cpu
  dtype: fp32
  config: {}

workload:
  name: open_loop
  episodes: 1
  steps_per_episode: 20
  config:
    observation_source: examples/open_loop/sample_obs.json

request:
  action_horizon: 8
  replan_steps: 4
  num_inference_steps: null
  return_future: false
  return_value: false

optimizations:
  enabled: []
  profiles: {}

telemetry:
  trace_jsonl: true
  persist_actions: true
  sample_memory: true
```

## Validation Rules

- `model.id` must resolve to a model entry unless an explicit backend-only
  config is used for development.
- Model entry defaults are applied first; explicit run config values override them.
- Overrides should be recorded in trace metadata.
- `backend.name`, `processor.name`, `workload.name`, and optimization names must
  be registered.
- Unknown top-level keys are errors.
- Optimization profiles may only contain fields declared by their profile spec.
- Conflicting profiles should fail before backend load.
- Training-only methods cannot be enabled as runtime profiles.

## Relationship To Model Entries

The config does not describe model capabilities. The model entry records
curated defaults and known gaps; the backend or remote server reports runtime
support.

See `docs/wamfile.md`.
