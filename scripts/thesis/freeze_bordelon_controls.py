"""Freeze the Bordelon control-suite outputs under ``outputs/thesis/controls/``.

Per ``EXPERIMENT_PLAN_FINAL.MD`` §5, the minimal reproduced control suite is:

- ``run_isotropic_depth_vs_alpha``  (ISO depth-vs-alpha phase diagram)
- ``run_fixed_covariance``           (FS depth sweep + OOD brittleness)
- ``run_reduced_gamma_dynamics``     (reduced-Γ landscape + min-loss vs depth)
- ``run_compute_scaling_joint``      (joint N-L compute scaling)
- ``run_linear_attention_dynamics``  (full-vs-reduced dynamics overlay)
- ``run_softmax_depth_sweep``        (softmax-attention depth sweep, optional sixth)

The plan's freeze contract: outputs must be placed under
``outputs/thesis/controls/`` with fixed seeds / configs and made
non-overwritable. This driver copies the existing Bordelon artifacts (produced
from the repo's frozen-dataclass default configs at the recorded git commit)
to ``outputs/thesis/controls/<script_stem>/artifacts/``, records a per-control
``metadata.json`` with SHA-256 hashes of every copied file, pins the git
commit, and sets every frozen file to read-only (``0444``). A top-level
``FROZEN.json`` indexes all six.

Running::

    python -u scripts/thesis/freeze_bordelon_controls.py

Idempotency: if ``outputs/thesis/controls/<stem>/artifacts/`` already exists
for any control, the driver aborts with a clear error instead of overwriting
a frozen record. To re-freeze, remove the specific subdirectory manually
(intentional friction).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Six controls from EXPERIMENT_PLAN_FINAL.MD §5.
CONTROLS: list[str] = [
    "run_isotropic_depth_vs_alpha",
    "run_fixed_covariance",
    "run_reduced_gamma_dynamics",
    "run_compute_scaling_joint",
    "run_linear_attention_dynamics",
    "run_softmax_depth_sweep",
]

# Per-control reproduction command. The Bordelon scripts build their final
# state from frozen-dataclass defaults under configs/, so the canonical
# reproduction call is with no overrides (default alpha/beta/seed etc.).
_DEFAULT_CLI = (
    "python -u scripts/{stem}.py --device cuda --dtype float64 --no-show"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_ROOT = _REPO_ROOT / "outputs"
_FROZEN_ROOT = _OUT_ROOT / "thesis" / "controls"
_READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH  # 0444


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _git_cmd(args: list[str]) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return None


def git_commit_hash() -> str | None:
    return _git_cmd(["rev-parse", "--short", "HEAD"])


def git_is_dirty() -> bool | None:
    porcelain = _git_cmd(["status", "--porcelain"])
    if porcelain is None:
        return None
    return bool(porcelain)


def env_fingerprint() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_full_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch_version"] = None
    return info


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-control freeze
# ---------------------------------------------------------------------------


def copy_artifacts(src_dir: Path, dst_dir: Path) -> list[dict[str, Any]]:
    """Recursively copy all files under ``src_dir`` to ``dst_dir`` preserving
    structure. Returns an artifact manifest, one entry per file with its
    path (relative to ``dst_dir``), SHA-256, and size.
    """
    manifest: list[dict[str, Any]] = []
    if not src_dir.exists():
        raise FileNotFoundError(f"source does not exist: {src_dir}")
    for src_file in sorted(src_dir.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        sha = compute_file_sha256(dst_file)
        size = dst_file.stat().st_size
        manifest.append(
            {
                "path": str(rel).replace(os.sep, "/"),
                "sha256": sha,
                "size_bytes": size,
            }
        )
    return manifest


def make_readonly(root: Path) -> None:
    """Recursively set every regular file under ``root`` to read-only (0444)."""
    for p in root.rglob("*"):
        if p.is_file():
            try:
                os.chmod(p, _READ_ONLY)
            except PermissionError:
                # If the file is already read-only from a prior freeze, skip.
                pass


def freeze_control(
    script_stem: str, index: int, total: int, commit: str | None, dirty: bool | None
) -> dict[str, Any]:
    src = _OUT_ROOT / script_stem
    dst = _FROZEN_ROOT / script_stem
    artifacts_dst = dst / "artifacts"

    print(f"[{index}/{total}] {script_stem}")
    if not src.exists():
        print(f"  SKIP: source {src} does not exist")
        return {
            "script_stem": script_stem,
            "status": "skipped",
            "reason": "source_missing",
            "source_path": str(src),
        }
    if artifacts_dst.exists():
        raise RuntimeError(
            f"destination already exists: {artifacts_dst}\n"
            "  freeze is non-overwritable by construction; remove this directory "
            "manually (chmod + rm) if you need to re-freeze."
        )

    dst.mkdir(parents=True, exist_ok=True)
    print(f"  src: {src}")
    print(f"  dst: {artifacts_dst}")
    manifest = copy_artifacts(src, artifacts_dst)
    n_files = len(manifest)
    total_bytes = sum(m["size_bytes"] for m in manifest)
    print(f"  copied {n_files} file(s), {total_bytes / 1024:.1f} KB")

    meta = {
        "script_stem": script_stem,
        "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §5",
        "source_path": str(src),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty_at_freeze": dirty,
        "env": env_fingerprint(),
        "config_source": (
            "frozen-dataclass defaults in configs/ at the recorded git commit; "
            "reproduced by re-running the listed reproduction_command from the "
            "same commit"
        ),
        "reproduction_command": _DEFAULT_CLI.format(stem=script_stem),
        "artifacts": manifest,
        "n_files": n_files,
        "total_bytes": total_bytes,
    }
    meta_path = dst / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    # Read-only pass over every copied file AND the metadata.json.
    make_readonly(dst)
    return {
        "script_stem": script_stem,
        "status": "frozen",
        "metadata_path": str(meta_path),
        "n_files": n_files,
        "total_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    _FROZEN_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(" Bordelon control-suite freeze")
    print(" Plan reference: EXPERIMENT_PLAN_FINAL.MD §5")
    print(f" Frozen root:    {_FROZEN_ROOT}")
    print("=" * 72)

    commit = git_commit_hash()
    dirty = git_is_dirty()
    if dirty:
        print(
            "  WARNING: repository has uncommitted changes at freeze time; "
            "record this in the top-level FROZEN.json but do not block."
        )

    results: list[dict[str, Any]] = []
    for i, ctrl in enumerate(CONTROLS, 1):
        results.append(freeze_control(ctrl, i, len(CONTROLS), commit, dirty))

    index = {
        "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §5",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty_at_freeze": dirty,
        "env": env_fingerprint(),
        "controls": results,
        "summary": {
            "frozen": sum(1 for r in results if r["status"] == "frozen"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "total": len(results),
        },
    }
    index_path = _FROZEN_ROOT / "FROZEN.json"
    # If a prior FROZEN.json exists and is read-only, we must clear it to
    # re-write (this is the single mutable record that enumerates all frozen
    # controls; its read-only bit is a convention, not a hard seal).
    if index_path.exists():
        try:
            os.chmod(index_path, stat.S_IWUSR | _READ_ONLY)
        except OSError:
            pass
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    os.chmod(index_path, _READ_ONLY)

    n_frozen = index["summary"]["frozen"]
    n_skipped = index["summary"]["skipped"]
    print()
    print(
        f"Summary: {n_frozen} frozen, {n_skipped} skipped "
        f"({n_frozen + n_skipped}/{len(CONTROLS)})"
    )
    print(f"Top-level index: {index_path}")
    return 0 if n_skipped == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
