"""Device selection and CUDA-error helpers for CLI scripts.

Many scripts support a user-facing ``--device`` flag and optional
"auto-select GPU if available" behavior. This module centralizes those small
policies so script entry points stay focused on experiment logic.
"""

from __future__ import annotations

import torch


def resolve_device(device_arg: str) -> str:
    """Resolve a CLI ``--device`` value into a concrete device string.

    Policy:
        - If user passes anything except ``"auto"``, return it verbatim.
        - If user passes ``"auto"``, prefer CUDA when available, else CPU.

    Args:
        device_arg: Raw CLI device argument, e.g. ``"auto"``, ``"cpu"``,
            ``"cuda"``, or ``"cuda:0"``.

    Returns:
        Concrete torch-compatible device string.
    """
    # Keep explicit user requests untouched.
    if device_arg != "auto":
        return device_arg
    # "auto" uses CUDA opportunistically, with CPU fallback.
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_cuda_oom(exc: BaseException) -> bool:
    """Heuristically detect CUDA out-of-memory errors from exception text.

    This is intentionally string-based because callers may receive different
    exception classes depending on backend/device/runtime path.

    Args:
        exc: Exception raised by a torch operation.

    Returns:
        ``True`` when the exception message looks like a CUDA OOM condition.
    """
    msg = str(exc).lower()
    # Cover common PyTorch and CUDA runtime wording variants.
    return "out of memory" in msg or "cudaerrormemoryallocation" in msg
