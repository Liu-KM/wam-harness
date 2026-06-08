# Scripts

This directory contains repository maintenance scripts and portable backend
environment installers.

The public harness contract is the `wam` command, not any scheduler submission
command. Site-specific scripts for a lab cluster, private queue, or local
operations workflow should stay outside this directory, for example under an
ignored `.local/` directory.

Allowed here:

- backend-agnostic repository maintenance scripts;
- backend-specific environment installers that are not tied to a scheduler or
  lab path.

Not allowed here:

- Slurm/PBS/LSF submission scripts;
- private cluster modules, account names, partitions, or scratch paths;
- one-off maintainer launch wrappers.

Core EazyWAM setup:

```bash
scripts/setup_core_env.sh
source .venv/bin/activate
wam list
```

This creates a lightweight `uv` virtual environment for the core package and
installs the local checkout in editable mode. It does not install heavy backend
runtime dependencies, download checkpoints, or configure simulators.

FastWAM self-managed install:

```bash
scripts/setup_fastwam_native_env.sh \
  --venv /path/to/.venv-fastwam \
  --cache-dir /path/to/wam-cache \
  --clone
```

This creates a dedicated `uv` virtual environment for FastWAM when a user cannot
run a backend container. FastWAM runtime code comes from EazyWAM; an
upstream FastWAM checkout is optional and only needed for explicit
`wam eval --reference` parity checks. The script does not download large model
assets or submit jobs. With `--clone`, it prepares LIBERO under
`<cache-dir>/upstreams/LIBERO` and writes `<cache-dir>/libero/config/config.yaml`.

Inside any prepared container, activated self-managed environment, or existing
allocation, the portable command shape is:

```bash
wam native-smoke fastwam-libero --trace-dir /mnt/runs --require-ready
wam serve <model-id>
```

FastWAM LIBERO native eval acceptance:

```bash
scripts/fastwam-libero-eval.sh \
  --cache-dir /path/to/wam-cache \
  --trace-dir /path/to/runs \
  --download-assets
```

This runs the product path, not the official FastWAM evaluator: `wam prepare`
for the verified eval asset group, `wam doctor --strict`, `wam native-smoke
--require-ready`, and then `wam eval fastwam-libero --workload
libero-single-task`. Use `wam eval fastwam-libero --reference` separately for
official-script parity.

The wrapper auto-detects LIBERO from the self-managed layout
`<cache-dir>/upstreams/LIBERO` and from the Docker image layout `/opt/LIBERO`.
Use `--libero-dir /path/to/LIBERO` only when the checkout lives elsewhere. The
same wrapper is symlinked inside the FastWAM Docker image as:

```bash
wam-fastwam-libero-eval
```

The wrapper passes `--summary-path` to `wam eval`, which writes the JSON summary
under the selected trace directory, for example:

```text
/path/to/runs/fastwam-libero-libero-single-task-eval-output.txt
/path/to/runs/fastwam-libero-libero-single-task-eval-summary.json
/path/to/runs/fastwam-libero-libero-single-task-acceptance.json
```

`*-eval-output.txt` is the raw console output. `*-eval-summary.json` is the
clean machine-readable summary written by the CLI, so the acceptance verifier
does not depend on stdout parsing.

You can re-check an existing summary and its trace without rerunning the model:

```bash
python -m eazywam.evals.acceptance --json \
  /path/to/runs/fastwam-libero-libero-single-task-eval-summary.json \
  1 \
  1.0
```

That verifier rejects summaries that came from `--reference`/external eval,
checks that the trace contains `native_eval_end` and `run_end.status="ok"`, and
checks that the summary, trace `native_eval_end`, and eval results JSON agree on
trial count, successes, and success rate. The final argument is the minimum
accepted success rate; the wrapper defaults to `1.0` for the single-task
acceptance smoke. `--json` prints a machine-readable acceptance report for
remote logs or CI artifacts. The wrapper also saves that report as
`*-acceptance.json` so remote runs leave a durable pass/fail artifact.
