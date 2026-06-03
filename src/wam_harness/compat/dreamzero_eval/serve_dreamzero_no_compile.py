from __future__ import annotations

import os
import faulthandler
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import torch

_F = TypeVar("_F", bound=Callable[..., Any])
_disabled_compile_calls = 0
_skipped_init_calls = 0


def _compile_identity(fn: _F | None = None, *args: Any, **kwargs: Any) -> _F | Callable[[_F], _F]:
    """Disable upstream torch.compile decorators before importing DreamZero modules."""

    global _disabled_compile_calls
    _disabled_compile_calls += 1
    target = getattr(fn, "__qualname__", None) if fn is not None else "<decorator>"
    print(
        "# wam DreamZero no-compile launcher bypassed torch.compile "
        f"target={target} call={_disabled_compile_calls}",
        flush=True,
    )

    def decorator(inner: _F) -> _F:
        print(
            "# wam DreamZero no-compile launcher applied decorator "
            f"target={getattr(inner, '__qualname__', repr(inner))}",
            flush=True,
        )
        return inner

    if fn is None:
        return decorator
    return fn


torch.compile = _compile_identity  # type: ignore[assignment]
if hasattr(torch, "compiler") and hasattr(torch.compiler, "compile"):
    torch.compiler.compile = _compile_identity  # type: ignore[assignment]
if hasattr(torch, "_dynamo"):
    torch._dynamo.optimize = _compile_identity  # type: ignore[attr-defined]
    torch._dynamo.optimize_assert = _compile_identity  # type: ignore[attr-defined]
    if hasattr(torch._dynamo, "eval_frame"):
        torch._dynamo.eval_frame.optimize = _compile_identity
        torch._dynamo.eval_frame.optimize_assert = _compile_identity

if os.environ.get("WAM_DREAMZERO_SKIP_WEIGHT_INIT", "1").lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}:

    def _skip_init(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        global _skipped_init_calls
        _skipped_init_calls += 1
        if _skipped_init_calls <= 20:
            print(
                "# wam DreamZero launcher skipped weight init "
                f"shape={tuple(tensor.shape)} call={_skipped_init_calls}",
                flush=True,
            )
        return tensor

    for _init_name in (
        "kaiming_uniform_",
        "kaiming_normal_",
        "xavier_uniform_",
        "xavier_normal_",
        "normal_",
        "uniform_",
        "trunc_normal_",
    ):
        if hasattr(torch.nn.init, _init_name):
            setattr(torch.nn.init, _init_name, _skip_init)

_cache_root = Path(os.environ.get("WAM_CACHE_DIR", os.environ.get("HF_HOME", "/tmp")))
os.environ.setdefault("TRITON_CACHE_DIR", str(_cache_root / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache_root / "torchinductor"))
_stack_dump_seconds = int(os.environ.get("WAM_DREAMZERO_STACK_DUMP_SECONDS", "0"))
if _stack_dump_seconds > 0:
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(
        _stack_dump_seconds,
        repeat=True,
        file=sys.stderr,
    )

from eval_utils import serve_dreamzero_wan22  # noqa: E402
import tyro  # noqa: E402


def main() -> None:
    tyro.cli(serve_dreamzero_wan22.main)


if __name__ == "__main__":
    main()
