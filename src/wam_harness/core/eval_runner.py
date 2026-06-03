from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wam_harness.core.memory import memory_snapshot
from wam_harness.core.registry import Registry, RegistryError, default_registry
from wam_harness.core.tracing import TraceWriter
from wam_harness.core.types import JsonDict, Manifest, OptimizationProfile, RuntimeInfo


class EvalRunnerError(RuntimeError):
    """Raised when an external simulator evaluation cannot be planned or run."""


@dataclass(frozen=True)
class EvalCommand:
    argv: list[str]
    workdir: str
    env: dict[str, str]
    display: str

    def to_dict(self) -> JsonDict:
        return {
            "argv": self.argv,
            "workdir": self.workdir,
            "env": self.env,
            "display": self.display,
        }


@dataclass(frozen=True)
class EvalSummary:
    run_id: str
    model_id: str
    workload: str | None
    trace_path: Path
    status: str
    command: EvalCommand
    return_code: int | None
    stdout_path: Path | None
    stderr_path: Path | None
    runtime_info: RuntimeInfo

    def to_dict(self) -> JsonDict:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "workload": self.workload,
            "trace_path": str(self.trace_path),
            "status": self.status,
            "command": self.command.to_dict(),
            "return_code": self.return_code,
            "stdout_path": str(self.stdout_path) if self.stdout_path is not None else None,
            "stderr_path": str(self.stderr_path) if self.stderr_path is not None else None,
            "runtime_info": self.runtime_info.to_dict(),
        }


class _StrictFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise EvalRunnerError(f"missing external eval template value: {key}") from None


@dataclass(frozen=True)
class _EvalWorkload:
    name: str | None
    config: dict[str, Any]


class EvalRunner:
    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or default_registry()

    def run(
        self,
        model_id: str,
        enabled_opts: list[str] | None = None,
        trace_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        upstream_dir: str | Path | None = None,
        dry_run: bool = False,
        reference: bool = False,
        workload: str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> EvalSummary:
        manifest = self.registry.load_manifest(model_id)
        if manifest.workload_name != "external_eval":
            raise RegistryError(
                f"{manifest.id} uses workload '{manifest.workload_name}', not external_eval"
            )
        if not manifest.eval:
            raise EvalRunnerError(f"{manifest.id} does not declare an eval command")

        eval_workload = self._select_eval_workload(manifest, workload)
        profiles = self.registry.build_optimization_profiles(manifest, enabled_opts or [])
        run_id = uuid.uuid4().hex[:12]
        output_dir = (Path(trace_dir) / run_id) if trace_dir is not None else Path("runs") / run_id
        trace_path = output_dir / "trace.jsonl"
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"

        runtime_info = self._runtime_info(manifest, profiles, eval_workload.name)
        command = self.plan_command(
            manifest=manifest,
            profiles=profiles,
            run_id=run_id,
            output_dir=output_dir,
            cache_dir=cache_dir,
            upstream_dir=upstream_dir,
            dry_run=dry_run,
            workload=eval_workload.name,
            overrides=overrides or {},
        )

        with TraceWriter(trace_path, run_id, runtime_info) as trace:
            trace.write(
                "run_start",
                mode="reference_eval" if reference else "simulator_eval",
                output_dir=str(output_dir),
                dry_run=dry_run,
                reference_eval=reference,
                optimization_profiles=[profile.to_dict() for profile in profiles],
                manifest_defaults=manifest.defaults,
                assets=manifest.assets,
                known_gaps=manifest.known_gaps,
                eval_workload=eval_workload.name,
                eval_metadata=self._eval_metadata(manifest, eval_workload.name),
            )
            trace.write(
                "external_eval_plan",
                eval_workload=eval_workload.name,
                command=command.to_dict(),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
            if dry_run:
                trace.write(
                    "run_end",
                    status="planned",
                    return_code=None,
                    trace_path=str(trace_path),
                    warnings=[],
                )
                return EvalSummary(
                    run_id=run_id,
                    model_id=model_id,
                    workload=eval_workload.name,
                    trace_path=trace_path,
                    status="planned",
                    command=command,
                    return_code=None,
                    stdout_path=None,
                    stderr_path=None,
                    runtime_info=runtime_info,
                )

            self._validate_execution(manifest, command)
            trace.write("backend_load", memory=memory_snapshot())
            output_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.update(command.env)
            start = time.perf_counter()
            with stdout_path.open("w", encoding="utf-8") as stdout:
                with stderr_path.open("w", encoding="utf-8") as stderr:
                    process = subprocess.run(
                        command.argv,
                        cwd=command.workdir,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        check=False,
                    )
            elapsed_ms = (time.perf_counter() - start) * 1000
            status = "ok" if process.returncode == 0 else "error"
            trace.write(
                "external_eval_end",
                return_code=process.returncode,
                timing={"wall_ms": elapsed_ms},
                memory=memory_snapshot(),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
            trace.write(
                "run_end",
                status=status,
                return_code=process.returncode,
                trace_path=str(trace_path),
                warnings=[] if process.returncode == 0 else ["external eval command failed"],
            )

        return EvalSummary(
            run_id=run_id,
            model_id=model_id,
            workload=eval_workload.name,
            trace_path=trace_path,
            status=status,
            command=command,
            return_code=process.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            runtime_info=runtime_info,
        )

    def plan_command(
        self,
        manifest: Manifest,
        profiles: list[OptimizationProfile] | None = None,
        run_id: str | None = None,
        output_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        upstream_dir: str | Path | None = None,
        dry_run: bool = False,
        workload: str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> EvalCommand:
        eval_workload = self._select_eval_workload(manifest, workload)
        command_config = self._eval_command_config(manifest, eval_workload)
        if not isinstance(command_config, dict):
            suffix = (
                f" eval.workloads.{eval_workload.name}.command"
                if eval_workload.name is not None
                else " eval.command"
            )
            raise EvalRunnerError(f"{manifest.id}{suffix} must be a mapping")

        context = self._build_context(
            manifest=manifest,
            eval_workload=eval_workload,
            profiles=profiles or [],
            run_id=run_id or uuid.uuid4().hex[:12],
            output_dir=Path(output_dir or "runs/planned"),
            cache_dir=cache_dir,
            upstream_dir=upstream_dir,
            dry_run=dry_run,
            overrides=overrides or {},
        )
        formatter = _StrictFormatDict({key: str(value) for key, value in context.items()})

        argv = command_config.get("argv")
        if not isinstance(argv, list) or not argv:
            raise EvalRunnerError(f"{manifest.id} eval.command.argv must be a non-empty list")
        rendered_argv = [str(item).format_map(formatter) for item in argv]
        rendered_argv.extend(
            self._profile_args(manifest, eval_workload, profiles or [], formatter)
        )
        workdir = str(command_config.get("workdir", "{upstream_dir}")).format_map(formatter)

        env_config = command_config.get("env", {})
        if not isinstance(env_config, dict):
            raise EvalRunnerError(f"{manifest.id} eval.command.env must be a mapping")
        env = {
            str(key): str(value).format_map(formatter)
            for key, value in env_config.items()
            if value is not None
        }
        display = " ".join(rendered_argv)
        return EvalCommand(argv=rendered_argv, workdir=workdir, env=env, display=display)

    def _profile_args(
        self,
        manifest: Manifest,
        eval_workload: _EvalWorkload,
        profiles: list[OptimizationProfile],
        formatter: _StrictFormatDict,
    ) -> list[str]:
        optimization_args = self._merged_eval_mapping(
            manifest,
            eval_workload,
            "optimization_args",
        )
        if not isinstance(optimization_args, dict):
            raise EvalRunnerError(f"{manifest.id} eval.optimization_args must be a mapping")

        args: list[str] = []
        for profile in profiles:
            profile_args = optimization_args.get(profile.name, [])
            if not isinstance(profile_args, list):
                raise EvalRunnerError(
                    f"{manifest.id} eval.optimization_args.{profile.name} must be a list"
                )
            args.extend(str(item).format_map(formatter) for item in profile_args)
        return args

    def _build_context(
        self,
        manifest: Manifest,
        eval_workload: _EvalWorkload,
        profiles: list[OptimizationProfile],
        run_id: str,
        output_dir: Path,
        cache_dir: str | Path | None,
        upstream_dir: str | Path | None,
        dry_run: bool,
        overrides: dict[str, str],
    ) -> dict[str, Any]:
        eval_config = manifest.eval
        defaults = eval_config.get("defaults", {})
        if defaults and not isinstance(defaults, dict):
            raise EvalRunnerError(f"{manifest.id} eval.defaults must be a mapping")
        workload_defaults = eval_workload.config.get("defaults", {})
        if workload_defaults and not isinstance(workload_defaults, dict):
            raise EvalRunnerError(
                f"{manifest.id} eval.workloads.{eval_workload.name}.defaults must be a mapping"
            )

        resolved_cache_dir = Path(
            cache_dir
            if cache_dir is not None
            else os.environ.get("WAM_CACHE_DIR", str(Path.home() / ".cache" / "wam"))
        )
        context: dict[str, Any] = {
            "model_id": manifest.id,
            "eval_workload": eval_workload.name or "",
            "run_id": run_id,
            "output_dir": str(output_dir),
            "trace_dir": str(output_dir.parent),
            "cache_dir": str(resolved_cache_dir),
            "opt_names": ",".join(profile.name for profile in profiles),
        }
        context.update(defaults)
        context.update(workload_defaults)
        context.update(self._profile_context(manifest, eval_workload, profiles))

        upstream = eval_config.get("upstream", {})
        if not isinstance(upstream, dict):
            raise EvalRunnerError(f"{manifest.id} eval.upstream must be a mapping")
        context["upstream_dir"] = self._resolve_upstream_dir(upstream, upstream_dir, dry_run)

        env_defaults = eval_config.get("env_defaults", {})
        if not isinstance(env_defaults, dict):
            raise EvalRunnerError(f"{manifest.id} eval.env_defaults must be a mapping")
        for key, env_name in env_defaults.items():
            if key in context or key in overrides:
                continue
            env_value = os.environ.get(str(env_name))
            if env_value:
                context[str(key)] = env_value
            elif dry_run:
                context[str(key)] = f"${env_name}"
            else:
                raise EvalRunnerError(f"missing environment variable for {key}: {env_name}")

        context.update(overrides)
        context = self._resolve_context_templates(manifest, context)
        return context

    def _resolve_context_templates(
        self,
        manifest: Manifest,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        resolved = dict(context)
        for _ in range(5):
            changed = False
            formatter = _StrictFormatDict({key: str(value) for key, value in resolved.items()})
            next_context: dict[str, Any] = {}
            for key, value in resolved.items():
                if isinstance(value, str):
                    try:
                        rendered = value.format_map(formatter)
                    except EvalRunnerError as exc:
                        raise EvalRunnerError(
                            f"{manifest.id} eval context value '{key}' could not be rendered: {exc}"
                        ) from exc
                    next_context[key] = rendered
                    changed = changed or rendered != value
                else:
                    next_context[key] = value
            resolved = next_context
            if not changed:
                return resolved
        raise EvalRunnerError(f"{manifest.id} eval context contains recursive template values")

    def _profile_context(
        self,
        manifest: Manifest,
        eval_workload: _EvalWorkload,
        profiles: list[OptimizationProfile],
    ) -> dict[str, Any]:
        optimization_context = self._merged_eval_mapping(
            manifest,
            eval_workload,
            "optimization_context",
        )
        if not isinstance(optimization_context, dict):
            raise EvalRunnerError(f"{manifest.id} eval.optimization_context must be a mapping")

        context: dict[str, Any] = {}
        for profile in profiles:
            profile_context = optimization_context.get(profile.name, {})
            if not isinstance(profile_context, dict):
                raise EvalRunnerError(
                    f"{manifest.id} eval.optimization_context.{profile.name} must be a mapping"
                )
            context.update(profile_context)
        return context

    def _select_eval_workload(
        self,
        manifest: Manifest,
        workload: str | None,
    ) -> _EvalWorkload:
        eval_config = manifest.eval
        workloads = eval_config.get("workloads", {})
        if workload is None:
            default_workload = eval_config.get("default_workload")
            if default_workload is not None:
                workload = str(default_workload)
        if workload is None:
            return _EvalWorkload(name=None, config={})

        if not isinstance(workloads, dict):
            raise EvalRunnerError(f"{manifest.id} eval.workloads must be a mapping")
        workload_config = workloads.get(workload)
        if workload_config is None:
            known = ", ".join(sorted(str(name) for name in workloads)) or "<none>"
            raise EvalRunnerError(
                f"unknown eval workload '{workload}' for {manifest.id}; known workloads: {known}"
            )
        if not isinstance(workload_config, dict):
            raise EvalRunnerError(
                f"{manifest.id} eval.workloads.{workload} must be a mapping"
            )
        return _EvalWorkload(name=workload, config=dict(workload_config))

    def _eval_command_config(
        self,
        manifest: Manifest,
        eval_workload: _EvalWorkload,
    ) -> object:
        if "command" in eval_workload.config:
            return eval_workload.config.get("command")
        return manifest.eval.get("command")

    def _merged_eval_mapping(
        self,
        manifest: Manifest,
        eval_workload: _EvalWorkload,
        key: str,
    ) -> dict[str, Any]:
        base = manifest.eval.get(key, {})
        if base and not isinstance(base, dict):
            raise EvalRunnerError(f"{manifest.id} eval.{key} must be a mapping")
        workload_value = eval_workload.config.get(key, {})
        if workload_value and not isinstance(workload_value, dict):
            raise EvalRunnerError(
                f"{manifest.id} eval.workloads.{eval_workload.name}.{key} must be a mapping"
            )
        merged: dict[str, Any] = dict(base) if isinstance(base, dict) else {}
        if isinstance(workload_value, dict):
            merged.update(workload_value)
        return merged

    def _resolve_upstream_dir(
        self,
        upstream: dict[str, Any],
        upstream_dir: str | Path | None,
        dry_run: bool,
    ) -> str:
        if upstream_dir is not None:
            return str(upstream_dir)

        env_name = upstream.get("local_env")
        if env_name:
            env_value = os.environ.get(str(env_name))
            if env_value:
                return env_value

        default_dir = upstream.get("default_dir")
        if default_dir:
            return str(default_dir)

        if dry_run:
            return "<upstream_dir>"

        raise EvalRunnerError("external eval requires --upstream-dir or an upstream default_dir")

    def _validate_execution(self, manifest: Manifest, command: EvalCommand) -> None:
        required_env = manifest.eval.get("required_env", [])
        if not isinstance(required_env, list):
            raise EvalRunnerError(f"{manifest.id} eval.required_env must be a list")
        missing_env = [str(name) for name in required_env if str(name) not in os.environ]
        if missing_env:
            raise EvalRunnerError(
                f"missing required environment variables: {', '.join(missing_env)}"
            )

        if not Path(command.workdir).exists():
            raise EvalRunnerError(f"external eval workdir does not exist: {command.workdir}")

    def _runtime_info(
        self,
        manifest: Manifest,
        profiles: list[OptimizationProfile],
        workload: str | None,
    ) -> RuntimeInfo:
        return RuntimeInfo(
            manifest_id=manifest.id,
            model_name=manifest.display_name,
            backend=manifest.backend_name,
            processor=manifest.processor_name,
            source_repo=manifest.source_repo,
            mode=str(manifest.backend.get("mode", "external_eval")),
            device=str(manifest.defaults.get("device", "cuda")),
            dtype=str(manifest.defaults.get("dtype", "unknown")),
            optimization_profiles=profiles,
            metadata=self._eval_metadata(manifest, workload),
        )

    def _eval_metadata(self, manifest: Manifest, workload: str | None) -> JsonDict:
        eval_config = manifest.eval
        upstream = eval_config.get("upstream", {})
        return {
            "workload": workload,
            "simulator": eval_config.get("simulator"),
            "suite": eval_config.get("suite"),
            "upstream_repo": upstream.get("repo") if isinstance(upstream, dict) else None,
            "upstream_commit": upstream.get("commit") if isinstance(upstream, dict) else None,
        }
