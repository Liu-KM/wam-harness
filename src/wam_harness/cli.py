from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from wam_harness.core.action_contract import ActionContractError
from wam_harness.core.compare import compare_traces
from wam_harness.core.eval_runner import EvalRunner, EvalRunnerError
from wam_harness.core.model_entry import (
    doctor_model_entry,
    prepare_model_entry,
)
from wam_harness.cli_render import (
    render_doctor,
    render_model_info,
    render_model_list,
    render_prepare,
)
from wam_harness.backends.native_support.smoke import (
    NativeSmokeRunner,
    NativeSmokeRunnerError,
)
from wam_harness.backends.native_support.readiness import NativePreflightError
from wam_harness.core.observation_io import (
    dict_or_empty,
    load_json_payload,
    observation_from_payload,
    write_json_payload,
)
from wam_harness.core.preflight import PreflightError
from wam_harness.core.runner import RunInputRequiredError, Runner
from wam_harness.serve import serve, smoke_serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wam")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List curated WAM model entries")

    info_parser = subparsers.add_parser("info", help="Show a model entry")
    info_parser.add_argument("model_id")

    doctor_parser = subparsers.add_parser("doctor", help="Check the runtime without modifying it")
    doctor_parser.add_argument("model_id", nargs="?")
    doctor_parser.add_argument("--cache-dir", default=None)
    doctor_parser.add_argument("--upstream-dir", default=None)
    doctor_parser.add_argument("--json", action="store_true", help="Print a machine-readable doctor summary")
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when the doctor status is not ok",
    )

    prepare_parser = subparsers.add_parser("prepare", help="Prepare model assets and cache")
    prepare_parser.add_argument("model_id")
    prepare_parser.add_argument("--cache-dir", default=None)
    prepare_parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing pullable assets declared by the model entry",
    )
    prepare_parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Limit prepare/download to one asset name; may be repeated",
    )

    run_parser = subparsers.add_parser("run", help="Run a WAM model")
    run_parser.add_argument("model_id")
    run_parser.add_argument("--opt", action="append", default=[], help="Enable optimization profile")
    run_parser.add_argument("--trace-dir", default=None)
    run_parser.add_argument("--cache-dir", default=None)
    run_parser.add_argument(
        "--input",
        default=None,
        help="Observation JSON file for one-shot inference",
    )
    run_parser.add_argument(
        "--output",
        default=None,
        help="Write the run summary and action result JSON to this file",
    )
    run_parser.add_argument("--episode-length", type=int, default=None)
    run_parser.add_argument("--action-horizon", type=int, default=None)
    run_parser.add_argument("--replan-steps", type=int, default=None)
    run_parser.add_argument("--upstream-dir", default=None)
    run_parser.add_argument(
        "--backend-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override native backend config for this run",
    )

    eval_parser = subparsers.add_parser("eval", help="Run a curated simulator evaluation")
    eval_parser.add_argument("model_id")
    eval_parser.add_argument("--opt", action="append", default=[], help="Enable optimization profile")
    eval_parser.add_argument(
        "--workload",
        default=None,
        help="Select an eval workload declared by the model entry",
    )
    eval_parser.add_argument("--trace-dir", default=None)
    eval_parser.add_argument("--cache-dir", default=None)
    eval_parser.add_argument("--upstream-dir", default=None)
    eval_parser.add_argument(
        "--reference",
        action="store_true",
        help="Run the official reference evaluator instead of the native product path",
    )
    eval_parser.add_argument("--dry-run", action="store_true", help="Print and trace the command only")
    eval_parser.add_argument(
        "--task-id",
        default=None,
        help="Shortcut for --set task_id=VALUE when the selected workload supports it",
    )
    eval_parser.add_argument(
        "--num-trials",
        default=None,
        help="Shortcut for --set num_trials=VALUE when the selected workload supports it",
    )
    eval_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an eval manifest template value",
    )

    native_smoke_parser = subparsers.add_parser(
        "native-smoke",
        help="Run one synthetic observation through a native backend migration path",
        description="Run one synthetic observation through a native backend migration path",
    )
    native_smoke_parser.add_argument("model_id")
    native_smoke_parser.add_argument("--opt", action="append", default=[], help="Enable optimization profile")
    native_smoke_parser.add_argument("--trace-dir", default=None)
    native_smoke_parser.add_argument("--cache-dir", default=None)
    native_smoke_parser.add_argument("--upstream-dir", default=None)
    native_smoke_parser.add_argument("--action-horizon", type=int, default=None)
    native_smoke_parser.add_argument("--replan-steps", type=int, default=None)
    native_smoke_parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Fail before load unless native readiness is ready",
    )
    native_smoke_parser.add_argument(
        "--backend-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override native backend config for this smoke run",
    )

    serve_parser = subparsers.add_parser("serve", help="Serve a WAM model")
    serve_parser.add_argument("model_id")
    serve_parser.add_argument("--opt", action="append", default=[], help="Enable optimization profile")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--trace-dir", default=None)
    serve_parser.add_argument("--cache-dir", default=None)
    serve_parser.add_argument("--upstream-dir", default=None)
    serve_parser.add_argument(
        "--backend-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override native backend config for this serve run",
    )
    serve_parser.add_argument("--smoke", action="store_true", help="Run a job-local serve smoke check")
    serve_parser.add_argument(
        "--smoke-input",
        default=None,
        help="Observation JSON file to POST during --smoke",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare two recorded traces")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("variant")
    compare_parser.add_argument(
        "--min-effect",
        type=float,
        default=0.05,
        help="Minimum relative latency change before faster/slower is reported",
    )
    compare_parser.add_argument(
        "--max-action-drift",
        type=float,
        default=1e-3,
        help="Maximum allowed drift across action summary scalar fields",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        print(render_model_list())
        return 0

    if args.command == "info":
        print(render_model_info(args.model_id))
        return 0

    if args.command == "doctor":
        summary = doctor_model_entry(
            args.model_id,
            cache_dir=args.cache_dir,
            upstream_dir=args.upstream_dir,
        )
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        else:
            print(render_doctor(summary))
        return 0 if not args.strict or summary.status == "ok" else 1

    if args.command == "prepare":
        summary = prepare_model_entry(
            args.model_id,
            cache_dir=args.cache_dir,
            download=args.download,
            selected_assets=args.asset or None,
        )
        print(render_prepare(summary))
        return 0 if summary.status == "ok" else 1

    if args.command == "run":
        try:
            backend_overrides = _parse_overrides(args.backend_set)
        except argparse.ArgumentTypeError as exc:
            _print_cli_error(exc)
            return 2
        observation = None
        runtime_options = None
        if args.input is not None:
            try:
                payload = load_json_payload(args.input)
                observation = observation_from_payload(payload)
                if observation is None:
                    raise ValueError("input JSON must contain observation.images")
                runtime_options = dict_or_empty(
                    payload.get("runtime_options"),
                    "runtime_options",
                )
                if args.action_horizon is None and payload.get("action_horizon") is not None:
                    args.action_horizon = int(payload["action_horizon"])
                if args.replan_steps is None and payload.get("replan_steps") is not None:
                    args.replan_steps = int(payload["replan_steps"])
            except (OSError, ValueError, TypeError) as exc:
                _print_cli_error(exc)
                return 1
        try:
            summary = Runner().run(
                model_id=args.model_id,
                enabled_opts=args.opt,
                trace_dir=args.trace_dir,
                episode_length=args.episode_length,
                action_horizon=args.action_horizon,
                replan_steps=args.replan_steps,
                upstream_dir=args.upstream_dir,
                cache_dir=args.cache_dir,
                backend_overrides=backend_overrides,
                observation=observation,
                runtime_options=runtime_options,
            )
        except _CLI_KNOWN_ERRORS as exc:
            _print_cli_error(exc)
            return 2 if isinstance(exc, RunInputRequiredError) else 1
        output = summary.to_dict()
        if args.output is not None:
            write_json_payload(args.output, output)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    if args.command == "eval":
        try:
            overrides = _parse_overrides(args.set)
        except argparse.ArgumentTypeError as exc:
            _print_cli_error(exc)
            return 2
        _set_override_if_present(overrides, "task_id", args.task_id)
        _set_override_if_present(overrides, "num_trials", args.num_trials)
        try:
            summary = EvalRunner().run(
                model_id=args.model_id,
                enabled_opts=args.opt,
                trace_dir=args.trace_dir,
                cache_dir=args.cache_dir,
                upstream_dir=args.upstream_dir,
                dry_run=args.dry_run,
                reference=args.reference,
                workload=args.workload,
                overrides=overrides,
            )
        except _CLI_KNOWN_ERRORS as exc:
            _print_cli_error(exc)
            return 1
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        if summary.return_code is None:
            return 0
        return int(summary.return_code)

    if args.command == "native-smoke":
        try:
            backend_overrides = _parse_overrides(args.backend_set)
        except argparse.ArgumentTypeError as exc:
            _print_cli_error(exc)
            return 2
        try:
            summary = NativeSmokeRunner().run(
                model_id=args.model_id,
                enabled_opts=args.opt,
                trace_dir=args.trace_dir,
                upstream_dir=args.upstream_dir,
                cache_dir=args.cache_dir,
                action_horizon=args.action_horizon,
                replan_steps=args.replan_steps,
                backend_overrides=backend_overrides,
                require_ready=args.require_ready,
            )
        except _CLI_KNOWN_ERRORS as exc:
            _print_cli_error(exc)
            return 1
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "serve":
        try:
            backend_overrides = _parse_overrides(args.backend_set)
        except argparse.ArgumentTypeError as exc:
            _print_cli_error(exc)
            return 2
        if args.smoke:
            smoke_payload = None
            if args.smoke_input is not None:
                try:
                    smoke_payload = load_json_payload(args.smoke_input)
                    if observation_from_payload(smoke_payload) is None:
                        raise ValueError("smoke input JSON must contain observation.images")
                except (OSError, ValueError, TypeError) as exc:
                    _print_cli_error(exc)
                    return 1
            try:
                result = smoke_serve(
                    model_id=args.model_id,
                    enabled_opts=args.opt,
                    trace_dir=args.trace_dir,
                    upstream_dir=args.upstream_dir,
                    cache_dir=args.cache_dir,
                    backend_overrides=backend_overrides,
                    payload=smoke_payload,
                )
            except _CLI_KNOWN_ERRORS as exc:
                _print_cli_error(exc)
                return 1
            print(json.dumps({"status": "ok", **result}, indent=2, sort_keys=True))
            return 0

        try:
            server = serve(
                model_id=args.model_id,
                enabled_opts=args.opt,
                host=args.host,
                port=args.port,
                trace_dir=args.trace_dir,
                upstream_dir=args.upstream_dir,
                cache_dir=args.cache_dir,
                backend_overrides=backend_overrides,
            )
        except _CLI_KNOWN_ERRORS as exc:
            _print_cli_error(exc)
            return 1
        host, port = server.server_address
        print(f"wam serve listening on http://{host}:{port}", flush=True)
        print("POST /infer expects JSON with observation.images", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 130
        finally:
            server.server_close()
        return 0

    if args.command == "compare":
        summary = compare_traces(
            args.baseline,
            args.variant,
            min_effect=args.min_effect,
            max_action_drift=args.max_action_drift,
        )
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _parse_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"override must be KEY=VALUE: {value}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"override key must not be empty: {value}")
        overrides[key] = raw
    return overrides


def _set_override_if_present(overrides: dict[str, str], key: str, value: object) -> None:
    if value is not None:
        overrides[key] = str(value)


_CLI_KNOWN_ERRORS = (
    PreflightError,
    ActionContractError,
    EvalRunnerError,
    NativeSmokeRunnerError,
    NativePreflightError,
    RunInputRequiredError,
)


def _print_cli_error(exc: Exception) -> None:
    print(f"error: {exc}", file=sys.stderr)
    trace_path = getattr(exc, "trace_path", None)
    if trace_path:
        print(f"trace: {trace_path}", file=sys.stderr)
