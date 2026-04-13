"""Per-run metadata and directory contract for thesis experiments.

This module is the single source of truth for the contract specified in
``EXPERIMENT_PLAN_FINAL.MD`` Sections 2 and 4.1. Every thesis experiment script
must:

1. Instantiate a :class:`ThesisRunDir` for its ``phase`` (one of
   ``controls``, ``theoremA``, ``theoremB``, ``theoremC``, ``architectures``,
   ``scaling_laws``, ``robustness``). The run directory is created eagerly.
2. Enter a :class:`RunContext` with the exact config and seed list. The context
   manager writes ``config.json`` and an initial ``metadata.json`` *before* any
   computation begins, so that a crashed run still leaves evidence of its
   inputs on disk.
3. Record the analytic compute proxy and per-step wall-clock measurements while
   the run progresses.
4. Provide a summary dict or string before exit so that ``summary.txt`` is
   written in a human-readable form.

The run-directory layout is::

    outputs/thesis/<phase>/<script_stem>/<run_id>/
        figures/      PNG figures
        pdfs/         vector PDF figures
        npz/          numpy archives
        pt/           torch tensors / checkpoints
        config.json   exact config used for the run
        metadata.json run_id, seeds, git, env, wall-clock, compute proxy, status
        summary.txt   human-readable summary of fitted quantities
        run.log       optional stdout/stderr log (written by the launcher)

The module is intentionally tolerant of missing tooling: if ``git`` is absent
or the repository is outside a git worktree, commit-hash fields are recorded as
``null``. Likewise, ``torch``/CUDA fields degrade gracefully.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a hard dep but guard anyway
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Run identifiers and environment capture
# ---------------------------------------------------------------------------


def make_run_id(script_stem: str) -> str:
    """Return a unique run identifier of the form
    ``<script_stem>-<UTC-timestamp>-<8-char-hex>``.

    The timestamp is in compact ISO-8601 (``YYYYMMDDTHHMMSSZ``). The trailing
    UUID fragment makes collisions between rapidly launched runs vanishingly
    improbable.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{script_stem}-{ts}-{short}"


def git_commit_hash(cwd: Path | str | None = None) -> str | None:
    """Return the short git commit hash for *cwd*, or ``None`` if unavailable."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return None


def git_is_dirty(cwd: Path | str | None = None) -> bool | None:
    """Return ``True`` if the working tree has uncommitted changes.

    Returns ``None`` if git is unavailable or the call fails for any reason.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return bool(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return None


def env_fingerprint() -> dict[str, Any]:
    """Capture Python / PyTorch / CUDA / OS description for reproducibility."""
    fp: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_full_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    if _HAS_TORCH:
        fp["torch_version"] = torch.__version__
        fp["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            fp["cuda_version"] = torch.version.cuda
            fp["cudnn_version"] = (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            )
            fp["gpu_count"] = torch.cuda.device_count()
            fp["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    return fp


# ---------------------------------------------------------------------------
# Config serialization helper
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of arbitrary config-like objects to JSON-safe form.

    - Dataclass instances are converted via :func:`dataclasses.asdict`.
    - ``dict`` / ``list`` / ``tuple`` are recursed into.
    - ``pathlib.Path`` becomes ``str``.
    - ``torch.dtype`` becomes its ``repr`` (e.g., ``"torch.float64"``).
    - Scalars pass through; everything else falls back to ``repr``.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if _HAS_TORCH and isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return obj
    return repr(obj)


# ---------------------------------------------------------------------------
# Run directory layout
# ---------------------------------------------------------------------------


_KNOWN_PHASES = {
    "controls",
    "theoremA",
    "theoremB",
    "theoremC",
    "architectures",
    "scaling_laws",
    "robustness",
}


def _default_project_root(script_file: Path) -> Path:
    """Locate the project root by searching upward for ``pyproject.toml``.

    Falls back to ``script_file.resolve().parents[3]`` (the canonical
    ``scripts/thesis/<phase>/<script>.py`` layout) if no marker is found.
    """
    p = Path(script_file).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Canonical fallback: scripts/thesis/<phase>/<script>.py -> parents[3]
    return p.parents[3] if len(p.parents) >= 4 else p.parents[-1]


class ThesisRunDir:
    """Directory layout and path helpers for a single thesis run.

    Parameters
    ----------
    script_file
        The ``__file__`` of the calling script. Used to derive the script
        stem and (by default) the project root.
    phase
        One of the known phases (``controls``, ``theoremA``, ``theoremB``,
        ``theoremC``, ``architectures``, ``scaling_laws``, ``robustness``).
        Unknown phases are accepted with a warning so that ad-hoc experiments
        are still supported.
    base
        Optional override for the ``outputs/thesis`` root. If omitted, the
        project root is detected via ``pyproject.toml`` and ``outputs/thesis``
        is appended.
    run_id
        Optional fixed run identifier. If omitted, :func:`make_run_id` is used.
    """

    def __init__(
        self,
        script_file: str | Path,
        phase: str,
        *,
        base: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        script_path = Path(script_file)
        stem = script_path.stem
        if phase not in _KNOWN_PHASES:
            # Non-fatal: allow ad-hoc phases (e.g., scratch experiments).
            import warnings

            warnings.warn(
                f"unknown thesis phase {phase!r}; expected one of {sorted(_KNOWN_PHASES)}",
                stacklevel=2,
            )
        if base is None:
            base = _default_project_root(script_path) / "outputs" / "thesis"
        base = Path(base)

        self.phase = phase
        self.script_stem = stem
        self.run_id = run_id or make_run_id(stem)
        self.root = base / phase / stem / self.run_id
        self.figures = self.root / "figures"
        self.pdfs = self.root / "pdfs"
        self.npz = self.root / "npz"
        self.pt = self.root / "pt"
        for d in (self.figures, self.pdfs, self.npz, self.pt):
            d.mkdir(parents=True, exist_ok=True)

    # ---- path accessors ----------------------------------------------------

    def png(self, name: str) -> Path:
        """Return the path ``figures/<name>.png``."""
        return self.figures / f"{name}.png"

    def pdf(self, name: str) -> Path:
        """Return the path ``pdfs/<name>.pdf``."""
        return self.pdfs / f"{name}.pdf"

    def npz_path(self, name: str) -> Path:
        """Return the path ``npz/<name>.npz``."""
        return self.npz / f"{name}.npz"

    def pt_path(self, name: str) -> Path:
        """Return the path ``pt/<name>.pt``."""
        return self.pt / f"{name}.pt"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.txt"

    @property
    def log_path(self) -> Path:
        return self.root / "run.log"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ThesisRunDir(phase={self.phase!r}, script={self.script_stem!r}, run_id={self.run_id!r})"


# ---------------------------------------------------------------------------
# RunContext: metadata lifecycle
# ---------------------------------------------------------------------------


class RunContext(AbstractContextManager):
    """Context manager enforcing the metadata contract.

    On ``__enter__`` it writes ``config.json`` and an initial ``metadata.json``
    (``status="started"``) so that the run is reconstructible even if it is
    killed mid-computation. On ``__exit__`` it updates ``metadata.json``
    (``status="completed"`` or ``"failed"``), records the wall-clock duration,
    the analytic compute proxy, per-step timing statistics, and writes a
    human-readable ``summary.txt``.

    Exceptions raised inside the ``with`` block are **not suppressed**; they
    propagate after the metadata is flushed.

    Parameters
    ----------
    run_dir
        The :class:`ThesisRunDir` for this run.
    config
        Arbitrary config object. Dataclasses are serialized via ``asdict``;
        everything else falls back through :func:`_jsonable`.
    seeds
        List of integer seeds used during the run (may be empty).
    notes
        Optional free-form note attached to the metadata.
    """

    def __init__(
        self,
        run_dir: ThesisRunDir,
        *,
        config: Any = None,
        seeds: list[int] | tuple[int, ...] | None = None,
        notes: str | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.seeds: list[int] = list(seeds) if seeds is not None else []
        self.notes = notes
        self._t0: float | None = None
        self._step_times: list[float] = []
        self._compute_proxy: float | None = None
        self._measured_compute: float | None = None
        self._summary: dict[str, Any] | str | None = None
        self._extras: dict[str, Any] = {}
        self._finalized = False

    # ---- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "RunContext":
        self._t0 = time.time()
        self.run_dir.config_path.write_text(
            json.dumps(_jsonable(self.config) if self.config is not None else {}, indent=2),
            encoding="utf-8",
        )
        self.run_dir.metadata_path.write_text(
            json.dumps(self._initial_metadata(), indent=2),
            encoding="utf-8",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._finalize(
            status="failed" if exc_type is not None else "completed",
            error=(repr(exc_val) if exc_val is not None else None),
        )
        return False  # never suppress exceptions

    # ---- user-facing recorders --------------------------------------------

    def record_compute_proxy(self, proxy: float) -> None:
        """Record the analytic compute proxy used by this run (e.g. ``t·P²·N²·L``)."""
        self._compute_proxy = float(proxy)

    def record_measured_compute(self, value: float) -> None:
        """Record a measured compute quantity (e.g. total FLOPs or TFLOP·s)."""
        self._measured_compute = float(value)

    def record_step_time(self, dt: float) -> None:
        """Append a per-step wall-clock measurement (seconds)."""
        self._step_times.append(float(dt))

    def record_extra(self, key: str, value: Any) -> None:
        """Attach an extra key/value pair to the metadata (JSON-serialized)."""
        self._extras[str(key)] = _jsonable(value)

    def write_summary(self, summary: dict[str, Any] | str) -> None:
        """Set the final summary (written to ``summary.txt`` on exit)."""
        self._summary = summary

    # ---- internals ---------------------------------------------------------

    def _initial_metadata(self) -> dict[str, Any]:
        repo_root = _default_project_root(Path(self.run_dir.root))
        return {
            "run_id": self.run_dir.run_id,
            "script_stem": self.run_dir.script_stem,
            "phase": self.run_dir.phase,
            "root": str(self.run_dir.root),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": list(self.seeds),
            "git_commit": git_commit_hash(repo_root),
            "git_dirty": git_is_dirty(repo_root),
            "env": env_fingerprint(),
            "notes": self.notes,
            "status": "started",
        }

    def _per_step_stats(self) -> dict[str, Any]:
        n = len(self._step_times)
        if n == 0:
            return {"n_steps": 0, "mean": None, "median": None, "total": None}
        sorted_t = sorted(self._step_times)
        return {
            "n_steps": n,
            "mean": float(sum(self._step_times) / n),
            "median": float(sorted_t[n // 2]),
            "total": float(sum(self._step_times)),
        }

    def _finalize(self, status: str, error: str | None = None) -> None:
        if self._finalized:
            return
        t0 = self._t0 if self._t0 is not None else time.time()
        elapsed = time.time() - t0
        try:
            meta = json.loads(self.run_dir.metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            meta = self._initial_metadata()
        meta.update(
            {
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "wall_clock_seconds": float(elapsed),
                "compute_proxy": self._compute_proxy,
                "measured_compute": self._measured_compute,
                "per_step_wall_clock": self._per_step_stats(),
                "status": status,
                "error": error,
            }
        )
        if self._extras:
            meta.setdefault("extras", {}).update(self._extras)
        self.run_dir.metadata_path.write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        self._write_summary_file(status=status, elapsed=elapsed, error=error)
        self._finalized = True

    def _write_summary_file(
        self, *, status: str, elapsed: float, error: str | None
    ) -> None:
        header = [
            f"run_id: {self.run_dir.run_id}",
            f"phase: {self.run_dir.phase}",
            f"script: {self.run_dir.script_stem}",
            f"status: {status}",
            f"wall_clock_seconds: {elapsed:.3f}",
        ]
        if self._compute_proxy is not None:
            header.append(f"compute_proxy: {self._compute_proxy:.6g}")
        if self._measured_compute is not None:
            header.append(f"measured_compute: {self._measured_compute:.6g}")
        if error:
            header.append(f"error: {error}")

        if self._summary is None:
            body = []
        elif isinstance(self._summary, str):
            body = [self._summary.rstrip()]
        else:
            body = [f"{k}: {v}" for k, v in self._summary.items()]

        text = "\n".join(header + ([""] + body if body else [])) + "\n"
        self.run_dir.summary_path.write_text(text, encoding="utf-8")


__all__ = [
    "env_fingerprint",
    "git_commit_hash",
    "git_is_dirty",
    "make_run_id",
    "RunContext",
    "ThesisRunDir",
]
