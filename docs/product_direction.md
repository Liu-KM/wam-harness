# Product Direction: Ollama For WAM

WAM Harness should be the easiest way to try, serve, and optimize WAM/VLA
inference in a portable runtime environment.

The target user is a new embodied-AI user or systems builder who wants to run a
model before becoming an expert in every upstream repository. The project should
hide setup complexity where possible, while still exposing enough telemetry to
make optimization toggles inspectable.

## Positioning

WAM Harness is not just a benchmark wrapper. It should become a local
deployment layer for WAMs:

```bash
wam list
wam info fastwam-libero
wam doctor fastwam-libero
wam prepare fastwam-libero
wam run fastwam-libero --input obs.json --output action.json
wam run fastwam-libero --input obs.json --opt vla_cache
wam serve fastwam-libero
```

The closest analogy is Ollama, but WAMs are harder than LLMs because a usable WAM
entry includes more than weights:

- checkpoint files.
- dataset statistics or normalizers.
- camera and state conventions.
- action schema and control frequency.
- processor logic.
- optional simulator dependencies.
- backend-specific runtime assumptions.

The project wins if it makes this bundle feel like one curated model entry.

The primary heavy-deployment abstraction is the `wam` command inside a prepared
backend runtime, not a specific cluster. A local container, self-managed virtual
environment, existing GPU allocation, or lab script should all reduce to the
same public command surface. Cluster submission is site policy and stays outside
the core project.

## Two Layers

### Deployment Spine

This is the default user path:

1. Resolve a model id to a curated model entry.
2. Prepare or locate model assets.
3. Load the registered backend and processor.
4. Run explicit-observation, simulator, or job-local server inference.
5. Apply explicit optimization profiles when requested.
6. Emit runtime info and minimal trace metadata.

The deployment spine optimizes for low friction.

### Telemetry Layer

This is the product observability path:

1. Record where time and memory go.
2. Record active optimization profiles and their parameters.
3. Persist action/future/value artifacts when requested.
4. Compare two recorded runs with `wam compare`.
5. Publish known speed, memory, compatibility, and output-drift notes in the
   support matrix.

`wam compare` is a trust tool, not a marketing badge. It only reports
`faster`/`slower` when both traces have comparable action shapes and usable
latency samples; missing output gates are `not_comparable`.

The telemetry layer optimizes for operational trust. It should not make the
simple run path hard to use.

## What Changes From The Earlier Branch

- The product spine is now `wam list` / `wam info` / `wam doctor` /
  `wam prepare` / `wam run` / `wam serve`.
- The old literature-reproduction workflow has been removed from this branch.
- A curated model library is a goal, but exhaustive model coverage is not.
- Local serving is a goal, but real robot safety and hardware orchestration are
  not first-version goals.
- Model entries become the main default layer, similar in spirit to Ollama's
  model metadata and Modelfile defaults. The internal YAML may still be called a
  manifest in code, but user-facing docs should prefer "model entry" or
  "model spec."

## First Vertical Slices

`A: portable deployment spine`

Rebuild the minimum package: fake model entry, fake backend, open-loop runner,
trace writer, optimization profile metadata, `wam run`, and a container smoke
path.

`B: portable serve smoke`

Start `wam serve fake-open-loop` inside a container or existing job allocation
and run a job-local inference smoke check.

`C: first real model`

Make one curated WAM model run end to end. The recommended first target is
FastWAM because it already exposes action chunks and released checkpoints.

`D: first real trick`

Make one real inference-time optimization toggleable and measurable. The
recommended first candidate is VLA-Cache because the upstream code already has a
clear cache on/off switch.

## Differentiation

LeRobot is the likely default place users look for robot learning datasets,
policies, and training workflows. WAM Harness should not compete by being a
larger robot-learning framework.

The wedge is:

- one-command WAM inference deployment.
- curated defaults for known model/checkpoint/task bundles.
- explicit, composable inference optimization profiles.
- trace-backed telemetry for latency, memory, output drift, and compatibility.

In short: use Hugging Face Hub and upstream repos as sources of assets; provide
the deployment and optimization wrapper users wish those repos had.
