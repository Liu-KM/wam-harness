# Roadmap

The project direction is WAM deployment first: build the smallest Ollama-like
spine that can run and serve a model in a portable runtime, then add measured
acceleration profiles.

## Phase 0: Direction And Contract

- Repository skeleton.
- Deployment and runtime contract.
- Optimization profile contract.
- Model entry/spec design.
- Dependency isolation strategy.
- Agent and collaboration instructions.

## Phase A: Portable Deployment Spine

Goal: make `wam run` real with no heavy model dependency and put it in a
runtime path that can run locally or inside an externally managed allocation.

- Core contract types.
- Model spec parser and one fake model entry.
- Registry for model entries, backends, processors, and optimization profiles.
- Fake backend.
- Open-loop workload.
- Runner with action chunk scheduling.
- JSONL trace writer.
- Minimal memory/timing observer.
- CLI entry point for `wam run`.
- Core Dockerfile-compatible image recipe.
- Generic container smoke path.
- Tests for config, model spec parsing, trace shape, fake inference, and profile
  metadata.

This phase is complete when a user can run a fake model through the same public
path intended for real models, including inside a container or activated local
environment launched by the user's local or cluster runtime.

## Phase B: Public Alpha Entry Surface

Goal: make the repository understandable before asking users to run heavy WAMs.

- `wam list` shows curated model entries.
- `wam info <model-id>` explains a model entry in readable language.
- `wam doctor [model-id]` checks the core package, cache directory, optional
  GPU visibility, backend runtime availability, and model-specific
  requirements.
- `wam prepare <model-id>` creates cache directories, checks required assets,
  downloads what the model spec can legally download, and reports any remaining
  manual steps.
- Existing no-execute planning modes remain available for tests, but public docs
  do not lead with `--dry-run`.

This phase is complete when a new user can clone the repository, run the four
entry commands above, and understand what each supported model needs without
opening a YAML file.

## Phase C: Native Backend Migration

Goal: stop treating official scripts as the product execution path.

- Keep current command-backed official evaluation as reference evaluators.
- Add native `fastwam`, `cosmos_policy`, and `dreamzero` backends and their
  processors.
- Run real-checkpoint native smoke workloads through `load/warmup/reset/infer`.
- Record native smoke output shape/range and timing in traces.
- Keep `wam eval fastwam-libero --reference` as the official-script parity path,
  while product `wam run` / `wam serve` use the native backend declaration.

This phase is complete when FastWAM, Cosmos-Policy, and DreamZero have native
backend paths that emit action chunks through the harness contract, with
official scripts retained as reference evaluators.

Current status: FastWAM, Cosmos-Policy, and DreamZero have passed real
checkpoint/container native smoke. FastWAM has also passed SuperPod H800
single-task LIBERO native eval acceptance, product-path serve smoke, and
reference full-suite LIBERO manager eval, and a native full-suite sweep. The
native sweep completed with 9/10 successes; the next hardening work is FastWAM
statistical native/reference parity, a proper native manager runner, and
broader model serving polish.
FastWAM RoboTwin is now the second real simulator closure: SuperPod H800
reference manager full-suite execution completed 100/100 clean/randomized
phases over 50 tasks, and native `robotwin-single-task` execution completed the
same 100/100 phase coverage with 0 structural failures. The next RoboTwin
hardening work is collecting stricter native/reference success-rate parity
evidence and turning task-side setup robustness into a public, upstreamable
patch plan. Manager summaries already separate invalid simulator setup from
policy failure by recording requested valid episodes, completed valid episodes,
attempted candidate episodes, invalid candidate episodes, and invalid candidate
reasons while preserving the requested top-level seed by default.

## Phase D: Portable Serve Smoke

Goal: make `wam serve` work as a persistent policy process inside a prepared
backend runtime or existing GPU allocation.

- `wam serve fake-open-loop`.
- Health/runtime metadata.
- Job-local inference client or workload driver.
- Runtime paths for repo, run output, and model/cache directories.
- Logs and traces written outside the runtime image or environment.

This phase is complete when a prepared runtime starts `wam serve`, runs a
job-local inference smoke check, and writes trace/health output. FastWAM has
passed this gate on SuperPod H800 through `wam serve fastwam-libero --smoke`
with a real observation payload.

External laptop-to-node endpoint access is not a Phase B requirement. The first
serving target is operational serving inside the environment that launched the
backend runtime.

## Phase E: First Native Real Model Evaluation

Goal: make one real curated WAM run through the public path.

- `fastwam-libero` model entry.
- Asset prepare plan for checkpoint, dataset stats, and required model files.
- Harness-owned simulator eval loop for curated LIBERO workloads; official
  scripts retained only behind explicit reference mode.
- FastWAM processor metadata for observations, action chunks, and optional future output.
- FastWAM container recipe and self-managed setup script.
- First LIBERO single-task simulator evaluation through
  `wam eval fastwam-libero --workload libero-single-task`.
- First full LIBERO manager evaluation on top of the native eval loop.

This phase is complete when `wam prepare fastwam-libero` can prepare or locate
released assets, `wam run fastwam-libero --input obs.json` can emit actions
through the native backend, and `wam eval fastwam-libero --workload
libero-single-task --task-id 0 --num-trials 1` can run a real LIBERO simulator
episode. FastWAM has passed this single-task gate on SuperPod H800 with
`success_rate=1.0`. Its reference full-suite manager path has also passed with
10/10 `libero_10` task successes at `num_trials=1`; native full-suite manager
coverage has been run as a sequential native single-task sweep with 9/10
successes. After aligning `seed=42` and `num_steps_wait=30`, repeating
task_id=6 with `num_trials=5` produced native 4/5 and reference single-task
4/5. The next hardening steps are statistical native-vs-reference parity, a
proper native manager runner, and a reference-manager task filter that does not
rely on the official manager's overwritten `MULTIRUN.task_file`.
As the second FastWAM simulator target, `fastwam-robotwin` has also passed
SuperPod H800 full-suite execution gates: reference manager eval reached
100/100 phases with clean mean 1.0 and randomized mean 0.84, and native
single-task sweep reached 100/100 phases with 0 structural failures and
aggregate success rate 0.88. RoboTwin invalid candidate episodes are now a
first-class manager statistic rather than a policy failure bucket;
process-level worker restarts are explicit, not default.

Expected deployment target: any supported GPU environment with enough memory
and either container support or a compatible self-managed backend environment.
A lab cluster can be used for validation, but cluster submission mechanics are
not part of the harness contract.

## Phase F: First Real Optimization Trick

Goal: show that a real acceleration method can be enabled as a profile.

- First target: DreamZero `dit_cache` or Cosmos parallel/JPEG profiles if those
  are already exposed by the official evaluator.
- FastWAM `dit_cache` L1 is the request-local video K/V cache profile:
  `video_kv` is the default product path and `recompute` is the ablation mode.
  It intentionally excludes cross-replan cache, token pruning, step skipping,
  and `torch.compile`.
- FastWAM `cuda_graph` is now the default `auto` FastWAM profile: the first
  version captures only the action-body `mot.forward_action_with_video_cache()`
  path and falls back to eager execution when CUDA Graph capture is unavailable.
  SuperPod H800 evidence shows `1.91x` mean total inference latency speedup on
  LIBERO `task_id=0`, `num_trials=1`, and `1.90x` on RoboTwin
  `click_alarmclock`, `num_episodes=1`.
- FastWAM `torch_compile` has been tested as a Phase-4 companion profile and
  remains experimental. SuperPod H800 job `450449` showed that CUDA Graph alone
  still improves the short LIBERO latency run, while CUDA Graph plus
  `torch_compile` can trigger Inductor fallback and heavy first-request latency.
- Later target: VLA-Cache with an OpenVLA/OpenVLA-OFT backend or wrapper path.
- Baseline run with cache disabled.
- Variant run with cache enabled.
- Trace fields for cache hit/update timing, latency, memory, and output checks.
- `wam compare` output for the two recorded runs.

This phase is complete when a user can enable the trick with a small profile or
CLI flag and the telemetry layer can report latency, memory, and output drift.

Expected deployment target: any supported GPU environment with the backend's
required runtime environment and mounted cache/run directories.

## Phase G: Curated Model Library

Goal: become useful to new WAM/VLA users without requiring them to read every
upstream repository first.

- 3-5 curated model entries with known-good defaults.
- Model support matrix.
- Known gaps and hardware requirements per model entry.
- Quickstarts for open-loop and simulator paths.
- MIT license selected before public release.

## Later Optimization Profiles

After the product spine works, add more deployment profiles:

- torch.compile.
- CUDA Graph.
- cache/history reuse.
- remote inference overhead.
- action chunk scheduling.
- quantization and post-training compression.
