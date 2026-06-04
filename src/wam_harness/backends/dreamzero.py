from __future__ import annotations

import importlib
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TextIO

from wam_harness.backends.native import (
    NativeBackendBase,
    NativeBackendError,
    NativeModelAdapter,
    NativeModelCall,
    NativeRuntimeLoader,
)
from wam_harness.core.types import (
    InferenceRequest,
    Manifest,
    OptimizationProfile,
    RuntimeInfo,
)


class DreamZeroNativeBackendError(NativeBackendError):
    """Raised when the native DreamZero resident backend cannot load."""


@dataclass(frozen=True)
class DreamZeroPolicyServerRuntimeBundle:
    """Connected DreamZero policy-server runtime before adapter binding."""

    client: Any
    server_process: subprocess.Popen[str] | None
    server_metadata: dict[str, Any]


class DreamZeroPolicyServerRuntimeLoader(NativeRuntimeLoader):
    """Start/connect the resident DreamZero policy server owned by the harness."""

    name = "dreamzero_policy_server_runtime_loader"
    runtime_mode = "resident_server"

    def __init__(self, backend: DreamZeroBackend) -> None:
        self.backend = backend

    def load(
        self,
        *,
        repo: Any,
        checkpoint_path: object,
    ) -> DreamZeroPolicyServerRuntimeBundle:
        server_process: subprocess.Popen[str] | None = None
        try:
            if not bool(self.backend.config.get("connect_only", False)):
                server_process = self.backend._start_server(repo, checkpoint_path)
            client, metadata = self.backend._connect_client(server_process)
            return DreamZeroPolicyServerRuntimeBundle(
                client=client,
                server_process=server_process,
                server_metadata=metadata,
            )
        except Exception:
            if server_process is not None:
                self.backend._stop_server_process(server_process)
            raise


class DreamZeroPolicyServerAdapter(NativeModelAdapter):
    """Native adapter around a connected DreamZero policy server client."""

    name = "dreamzero_policy_server"

    def __init__(
        self,
        *,
        client: Any,
        server_metadata: dict[str, Any],
        checkpoint_path: object | None,
        error_cls: type[DreamZeroNativeBackendError],
    ) -> None:
        self.client = client
        self.server_metadata = dict(server_metadata)
        self.checkpoint_path = checkpoint_path
        self.error_cls = error_cls

    def require_ready(self) -> None:
        if self.client is None:
            raise self.error_cls("DreamZero policy server adapter is not connected")

    def model_timing_key(self) -> str:
        return "server_ms"

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "model_adapter": self.name,
            "transport": "websocket",
            "server_metadata": self.server_metadata,
        }

    def inference_metadata(self) -> dict[str, object]:
        return {
            "model_adapter": self.name,
            "transport": "websocket",
            "server_metadata": self.server_metadata,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }

    def infer(self, _request: InferenceRequest, payload: object) -> NativeModelCall:
        self.require_ready()
        return NativeModelCall(
            raw_output=self.client.infer(payload),
            timing_key="server_ms",
            metadata={"server_metadata": self.server_metadata},
        )

    def reset(self) -> None:
        self.require_ready()
        self.client.reset({})

    def close(self) -> None:
        if self.client is not None:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
        self.client = None


class DreamZeroBackend(NativeBackendBase):
    error_cls = DreamZeroNativeBackendError
    default_upstream_env = "WAM_DREAMZERO_REPO"
    required_upstream_paths = ("eval_utils/serve_dreamzero_wan22.py",)
    required_asset_names = ("checkpoint",)
    runtime_asset_names = ("checkpoint",)
    required_python_modules = (
        "numpy",
        "openpi_client",
        "websockets.sync.client",
        "typing_extensions",
    )
    model_adapter_name = DreamZeroPolicyServerAdapter.name
    optimization_hooks: ClassVar[dict[str, str]] = {
        **NativeBackendBase.optimization_hooks,
        "dit_cache": "dreamzero_server_dit_cache_arg",
    }

    def __init__(self, manifest: Manifest, profiles: list[OptimizationProfile]) -> None:
        super().__init__(manifest, profiles, backend_label="DreamZero")
        self.client: Any | None = None
        self.server_process: subprocess.Popen[str] | None = None
        self.server_log_handle: TextIO | None = None
        self.server_log_path: Path | None = None
        self.server_metadata: dict[str, Any] = {}
        self.checkpoint_path = None
        self.master_port: int | None = None
        self.host = str(self.config.get("host", "127.0.0.1"))
        defaults = self.eval_defaults()
        self.port = int(self.config.get("port", defaults.get("policy_port", 6000)))
        self.runtime_loader = DreamZeroPolicyServerRuntimeLoader(self)

    def native_required_upstream_paths(self) -> tuple[str, ...]:
        return (self._server_module_repo_path(),)

    def load(self) -> None:
        repo = self.resolve_upstream_repo()
        checkpoint_path = self.resolve_required_asset("checkpoint")
        self.checkpoint_path = checkpoint_path
        self._apply_runtime_env_defaults()
        self.add_upstream_paths(repo, repo / "eval_utils")

        runtime = self.runtime_loader.load(repo=repo, checkpoint_path=checkpoint_path)
        self.client = runtime.client
        self.server_process = runtime.server_process
        self.server_metadata = runtime.server_metadata
        self.model_adapter = self._create_model_adapter()
        self.upstream_repo = repo
        self.loaded = True

    def warmup(self) -> None:
        self.require_loaded()
        self.native_model_adapter(required=True).warmup()
        self.warmed = True

    def reset(self) -> None:
        self.require_loaded()
        self.native_model_adapter(required=True).reset()

    def runtime_info(self) -> RuntimeInfo:
        return super().runtime_info(
            {
                "transport": "websocket",
                "server_started_by_harness": self.server_process is not None,
            }
        )

    def close(self) -> None:
        try:
            if self.server_process is not None:
                self._stop_server_process(self.server_process)
                self.server_process = None
            self._close_server_log()
            self.client = None
        finally:
            super().close()

    def _create_model_adapter(self) -> DreamZeroPolicyServerAdapter:
        return DreamZeroPolicyServerAdapter(
            client=self.client,
            server_metadata=self.server_metadata,
            checkpoint_path=self.checkpoint_path,
            error_cls=self.error_cls,
        )

    def _start_server(
        self,
        repo: Any,
        checkpoint_path: object | None = None,
    ) -> subprocess.Popen[str]:
        defaults = self.eval_defaults()
        server_python = self._render_runtime_template(
            self.config.get("server_python", defaults.get("server_python", sys.executable)),
            repo=repo,
        )
        module = self._server_module()
        argv = [
            server_python,
            "-m",
            module,
            "--model_path",
            str(checkpoint_path or self.checkpoint_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        argv.extend(self._profile_server_args())
        env = self._server_env()
        existing_pythonpath = env.get("PYTHONPATH")
        repo_paths = [str(repo), str(repo / "eval_utils")]
        env["PYTHONPATH"] = (
            os.pathsep.join(repo_paths + [existing_pythonpath])
            if existing_pythonpath
            else os.pathsep.join(repo_paths)
        )
        log_path = self._server_log_target(repo)
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"# wam DreamZero policy server command: {shlex.join(argv)}\n")
        log_handle.write(
            "# wam DreamZero policy server env: "
            f"HF_HOME={env.get('HF_HOME', '')} "
            f"HF_HUB_CACHE={env.get('HF_HUB_CACHE', '')} "
            f"TORCH_COMPILE_DISABLE={env.get('TORCH_COMPILE_DISABLE', '')} "
            f"MASTER_ADDR={env.get('MASTER_ADDR', '')} "
            f"MASTER_PORT={env.get('MASTER_PORT', '')} "
            f"TORCHINDUCTOR_CACHE_DIR={env.get('TORCHINDUCTOR_CACHE_DIR', '')} "
            f"TRITON_CACHE_DIR={env.get('TRITON_CACHE_DIR', '')} "
            f"WAM_DREAMZERO_SKIP_WEIGHT_INIT={env.get('WAM_DREAMZERO_SKIP_WEIGHT_INIT', '')}\n"
        )
        log_handle.flush()
        self.server_log_path = log_path
        self.server_log_handle = log_handle
        try:
            return subprocess.Popen(
                argv,
                cwd=str(repo),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            self._close_server_log()
            raise

    def _render_runtime_template(self, value: object, *, repo: Any) -> str:
        return (
            str(value)
            .replace("{upstream_dir}", str(repo))
            .replace("{cache_dir}", str(self.cache_dir()))
        )

    def _server_log_target(self, repo: Any) -> Path:
        defaults = self.eval_defaults()
        configured = self.config.get("server_log_path", defaults.get("server_log_path"))
        if configured is None:
            path = Path(self.cache_dir()) / "logs" / (
                f"dreamzero-policy-server-{os.getpid()}-{self.port}.log"
            )
        else:
            path = Path(self._render_runtime_template(configured, repo=repo))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _server_log_hint(self) -> str:
        return f" server_log={self.server_log_path}" if self.server_log_path else ""

    def _apply_runtime_env_defaults(self) -> None:
        for key, value in self._runtime_env_values().items():
            if value is not None:
                os.environ.setdefault(str(key), str(value))

    def _server_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key, value in self._runtime_env_values().items():
            if value is None:
                continue
            if key in {
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_XET_CACHE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TORCH_COMPILE_DISABLE",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "WAM_CACHE_DIR",
                "WAM_DREAMZERO_SKIP_WEIGHT_INIT",
            }:
                env[str(key)] = str(value)
            else:
                env.setdefault(str(key), str(value))
        return env

    def _runtime_env_values(self) -> dict[str, object | None]:
        hf_home = self.cache_dir() / "huggingface"
        runtime_defaults: dict[str, object | None] = {
            "HF_HOME": hf_home,
            "HF_HUB_CACHE": hf_home / "hub",
            "HF_XET_CACHE": hf_home / "xet",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": self._distributed_master_port(),
            "TORCHINDUCTOR_CACHE_DIR": self.cache_dir() / "torchinductor",
            "TRITON_CACHE_DIR": self.cache_dir() / "triton",
            "WAM_CACHE_DIR": self.cache_dir(),
            "WAM_DREAMZERO_SKIP_WEIGHT_INIT": "1",
            "WANDB_MODE": "offline",
            "TOKENIZERS_PARALLELISM": "false",
        }
        if self._config_bool("disable_torch_compile", default=True):
            runtime_defaults["TORCH_COMPILE_DISABLE"] = "1"
        return runtime_defaults

    def _config_bool(self, key: str, *, default: bool) -> bool:
        value = self.config.get(key, self.eval_defaults().get(key, default))
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() not in {"", "0", "false", "no", "off"}

    def _distributed_master_port(self) -> int:
        configured = self.config.get("master_port", self.eval_defaults().get("master_port"))
        if configured is not None:
            return int(configured)
        if self.master_port is None:
            self.master_port = _find_free_local_port()
        return self.master_port

    def _connect_client(
        self,
        server_process: subprocess.Popen[str] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        try:
            client_module = importlib.import_module(
                "wam_harness.compat.dreamzero_eval.policy_client"
            )
        except ModuleNotFoundError as exc:
            raise self.error_cls(
                "DreamZero policy client dependencies are not importable. "
                "Run inside a DreamZero-compatible container."
            ) from exc

        timeout_s = self._server_startup_seconds()
        deadline = time.time() + timeout_s
        last_error: Exception | None = None
        while time.time() < deadline:
            if server_process is not None and server_process.poll() is not None:
                raise self.error_cls(
                    "DreamZero policy server exited before readiness; "
                    f"return_code={server_process.returncode}."
                    f"{self._server_log_hint()}"
                )
            try:
                client = client_module.WebsocketClientPolicy(host=self.host, port=self.port)
                metadata = client.get_server_metadata()
                server_metadata = dict(metadata) if isinstance(metadata, dict) else {}
                return client, server_metadata
            except Exception as exc:  # pragma: no cover - depends on external server timing.
                last_error = exc
                time.sleep(1)
        raise self.error_cls(
            f"DreamZero policy server did not become ready at {self.host}:{self.port} "
            f"within {timeout_s}s."
            f"{self._server_log_hint()}"
        ) from last_error

    def _stop_server_process(self, process: subprocess.Popen[str]) -> None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        finally:
            self._close_server_log()

    def _close_server_log(self) -> None:
        if self.server_log_handle is not None:
            self.server_log_handle.close()
            self.server_log_handle = None

    def _profile_server_args(self) -> list[str]:
        args: list[str] = []
        if not self.profile_enabled("dit_cache"):
            return args
        configured = self.config.get("dit_cache_args")
        if configured is None:
            configured = self.profile_settings("dit_cache").get("dit_cache_arg")
        if configured is None:
            configured = self.eval_defaults().get("dit_cache_arg")
        if configured:
            args.extend(shlex.split(str(configured)))
        return args

    def _server_module(self) -> str:
        if self.config.get("server_module") is not None:
            return str(self.config["server_module"])
        if self._config_bool("disable_torch_compile", default=True):
            return "wam_harness.compat.dreamzero_eval.serve_dreamzero_no_compile"
        defaults = self.eval_defaults()
        return str(defaults.get("server_module", "eval_utils.serve_dreamzero_wan22"))

    def _server_startup_seconds(self) -> int:
        return int(
            self.config.get(
                "server_startup_seconds",
                self.eval_defaults().get("server_startup_seconds", 180),
            )
        )

    def _server_module_repo_path(self) -> str:
        module = self._server_module().split(":", maxsplit=1)[0]
        if module == "wam_harness.compat.dreamzero_eval.serve_dreamzero_no_compile":
            return "eval_utils/serve_dreamzero_wan22.py"
        if module.endswith(".py") or "/" in module:
            return module
        return f"{module.replace('.', '/')}.py"


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
