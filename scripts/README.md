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

FastWAM self-managed install:

```bash
scripts/setup_fastwam_native_env.sh \
  --upstream-dir /path/to/FastWAM \
  --venv /path/to/.venv-fastwam \
  --cache-dir /path/to/wam-cache \
  --clone
```

This creates a dedicated `uv` virtual environment for FastWAM when a user cannot
run a backend container. It does not download large model assets or submit jobs.

Inside any prepared container, activated self-managed environment, or existing
allocation, the portable command shape is:

```bash
wam eval <model-id> --trace-dir /mnt/runs --upstream-dir /mnt/upstream
wam serve <model-id>
```
