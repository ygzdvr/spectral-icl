"""Experiment C5 strict-drops patch: record the Corollary 3.16 strict
direction of the refinement advantage, applied to the full dyadic ladder.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §7.5 (C5).
Chapter reference: ``thesis/theorem_c.txt`` — Proposition 3.15
(weak-direction monotonicity along a refinement ladder) and
Corollary 3.16 (strict hybrid gain: if the coarse block contains modes
with different preferred scalars then the refined-class optimum is
strictly smaller).

Purpose
-------
The canonical C5 script (``run_theoremC_refinement_monotonicity``)
already validates the weak direction (``L★(j+1) ≤ L★(j) + 1e-8``). It
reports ``strict_drop_counts`` as raw data (6/6 at every κ > 1 under
the L = 1 slice in the canonical run) but does NOT lift that count to a
formal acceptance gate. This minimal patch fills that gap, mirroring
the structure of ``run_theoremC_c4_strict_gain_patch``.

It is a pure acceptance-record fix: no new sweep, no new figure.

Checks
------
1. ``strict_drops_all_kappa_gt1_ok`` — for every ``κ > 1`` and every
   ``L`` in the C5 sweep, the ladder has ``strict_drops == total_steps``
   (= 6 for the D = 64 dyadic ladder). A step ``j → j+1`` counts as a
   strict drop when ``L★(j) − L★(j+1) > strict_tol``.
2. ``kappa1_no_strict_drops_ok`` — at ``κ = 1`` the block is
   homogeneous so ``L★ ≡ 0`` at every ladder level; no step should
   produce a numerically-meaningful drop.
3. Resolved diagnostic at ``L = L_deeper`` (L = 4 in the canonical C5
   run), following the pattern of
   ``run_theoremC_c4_strict_gain_patch``:
   - ``strict_drops_resolved_ok`` — restrict to steps where **both**
     the coarser and finer level's optimum exceed ``optimizer_floor``
     (default ``1e-9``); every such step must produce a strict drop.
   - ``failing_above_precision`` — number of steps whose start-point is
     above precision but whose drop is not strict. Must be zero.

Data source
-----------
Default: load the most recent canonical C5 ``refinement_ladder.npz``
from
``outputs/thesis/theoremC/run_theoremC_refinement_monotonicity/<run_id>/npz/``.
Fallback: re-run the sweep locally when ``--recompute`` is passed or
when no canonical C5 run is available.

Output
------
Canonical run directory
``outputs/thesis/theoremC/c5_strict_drops_patch/<run_id>/``
(matches the user-specified output contract; ``ThesisRunDir`` stem is
pinned to ``c5_strict_drops_patch`` via a synthetic script path).

Contents: ``config.json``, ``metadata.json``, ``summary.txt``,
``c5_strict_drops_patch_summary.txt`` (the human-readable acceptance
record).

Run
---
::

    python -u scripts/thesis/theoremC/run_theoremC_c5_strict_drops_patch.py \\
           --no-show
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np

from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C5StrictDropsConfig:
    """Config for the patch. Defaults match the canonical C5 run exactly."""

    D: int = 64
    kappa_list: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0)
    L_list: tuple[int, ...] = (1, 4)
    # Dyadic ladder on D = 64: level_sizes = (64, 32, 16, 8, 4, 2, 1),
    # per_level_nblocks = (1, 2, 4, 8, 16, 32, 64). Seven ladder levels,
    # six refinement steps.
    expected_level_sizes: tuple[int, ...] = (64, 32, 16, 8, 4, 2, 1)

    # Strict-drop threshold (mirrors the user-specified threshold in the
    # C4 strict-gain patch).
    strict_tol: float = 1e-15

    # L-BFGS precision floor below which a "strict drop" is indistinguishable
    # from optimizer noise (mirrors the c4 strict-gain patch).
    optimizer_floor: float = 1e-9

    # Which L is treated as ``L_deeper`` for the resolved diagnostic
    # (plan §7.5: C5 sweeps L ∈ {1, 4}; L = 4 is the deeper slice whose
    # losses saturate to the eps floor after a few refinement steps).
    L_deeper: int = 4


# ---------------------------------------------------------------------------
# Canonical C5 data loader
# ---------------------------------------------------------------------------


_CANONICAL_C5_ROOT = (
    "outputs/thesis/theoremC/run_theoremC_refinement_monotonicity"
)


def _find_latest_c5_run(project_root: Path) -> Path | None:
    """Return the path of the most recent canonical C5 run directory."""
    root = project_root / _CANONICAL_C5_ROOT
    if not root.is_dir():
        return None
    runs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    return runs[-1] if runs else None


def _load_c5_npz(run_dir: Path) -> dict[str, np.ndarray]:
    npz_path = run_dir / "npz" / "refinement_ladder.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"canonical C5 npz not found at {npz_path}"
        )
    data = np.load(npz_path)
    return {k: data[k] for k in data.keys()}


def _validate_axes(
    cfg: C5StrictDropsConfig, data: dict[str, np.ndarray]
) -> None:
    expected_k = np.asarray(cfg.kappa_list, dtype=np.float64)
    expected_L = np.asarray(cfg.L_list, dtype=np.int64)
    expected_sizes = np.asarray(
        cfg.expected_level_sizes, dtype=np.int64
    )
    got_k = np.asarray(data["kappa_list"]).astype(np.float64)
    got_L = np.asarray(data["L_list"]).astype(np.int64)
    got_sizes = np.asarray(data["level_sizes"]).astype(np.int64)
    if got_k.shape != expected_k.shape or not np.allclose(expected_k, got_k):
        raise ValueError(
            f"kappa_list mismatch: canonical run has {list(got_k)}, "
            f"expected {list(expected_k)}"
        )
    if not np.array_equal(expected_L, got_L):
        raise ValueError(
            f"L_list mismatch: canonical run has {list(got_L)}, expected "
            f"{list(expected_L)}"
        )
    if not np.array_equal(expected_sizes, got_sizes):
        raise ValueError(
            f"level_sizes mismatch: canonical run has "
            f"{list(got_sizes)}, expected {list(expected_sizes)}"
        )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _run_checks(
    cfg: C5StrictDropsConfig, data: dict[str, np.ndarray]
) -> dict[str, Any]:
    kappa_arr = np.asarray(data["kappa_list"]).astype(np.float64)
    L_arr = np.asarray(data["L_list"]).astype(np.int64)
    loss = np.asarray(data["loss_grid"]).astype(np.float64)
    # loss shape: (n_levels, n_kappa, n_L)
    n_levels = loss.shape[0]
    total_steps = n_levels - 1  # = 6 for the D=64 dyadic ladder

    # Per-(kappa, L) strict-drop count.
    drops = loss[:-1, :, :] - loss[1:, :, :]  # (n_steps, n_kappa, n_L)
    strict_mask = drops > cfg.strict_tol

    # ---- Check 1: every κ > 1 produces total_steps strict drops ----
    kappa_gt1_mask = kappa_arr > 1.0
    per_cell_strict_counts = strict_mask.sum(axis=0)  # (n_kappa, n_L)
    gt1_cells: list[dict[str, Any]] = []
    for ki, kappa in enumerate(kappa_arr):
        if not kappa_gt1_mask[ki]:
            continue
        for li, Lval in enumerate(L_arr):
            count = int(per_cell_strict_counts[ki, li])
            gt1_cells.append(
                {
                    "kappa": float(kappa),
                    "L": int(Lval),
                    "strict_drops": count,
                    "total_steps": int(total_steps),
                    "ok": bool(count == total_steps),
                }
            )
    strict_drops_all_kappa_gt1_ok = all(c["ok"] for c in gt1_cells)
    gt1_worst = (
        min(gt1_cells, key=lambda c: c["strict_drops"])
        if gt1_cells else None
    )

    # ---- Check 2: κ = 1 ⇒ zero strict drops ----
    kappa_eq1_cells: list[dict[str, Any]] = []
    kappa_eq1_indices = np.where(~kappa_gt1_mask)[0]
    kappa_eq1_max_loss_per_L: list[dict[str, Any]] = []
    for ki in kappa_eq1_indices:
        for li, Lval in enumerate(L_arr):
            count = int(per_cell_strict_counts[ki, li])
            max_loss_this_L = float(loss[:, ki, li].max())
            kappa_eq1_cells.append(
                {
                    "kappa": float(kappa_arr[ki]),
                    "L": int(Lval),
                    "strict_drops": count,
                    "max_loss_all_levels": max_loss_this_L,
                    "ok": bool(count == 0),
                }
            )
            kappa_eq1_max_loss_per_L.append(
                {
                    "kappa": float(kappa_arr[ki]),
                    "L": int(Lval),
                    "max_loss_all_levels": max_loss_this_L,
                    "resolved": bool(
                        max_loss_this_L > cfg.optimizer_floor
                    ),
                }
            )
    kappa1_no_strict_drops_ok = all(c["ok"] for c in kappa_eq1_cells)
    # Resolved variant: at κ = 1, if every level's loss is below
    # optimizer_floor the ladder is effectively flat at the precision
    # floor; any "strict drops" there are optimizer noise, not theorem
    # violations. The theorem-level gate asks that the max loss across
    # all levels at κ = 1 stays below optimizer_floor — i.e. that the
    # ladder is actually flat at the precision scale.
    kappa_eq1_max_loss_overall = 0.0
    for ki in kappa_eq1_indices:
        kappa_eq1_max_loss_overall = max(
            kappa_eq1_max_loss_overall,
            float(loss[:, ki, :].max()),
        )
    kappa1_no_strict_drops_resolved_ok = (
        kappa_eq1_max_loss_overall < cfg.optimizer_floor
    )

    # ---- Check 3: resolved diagnostic at L_deeper ----
    # For each step j → j+1 at κ > 1, a step is "resolvable" iff both the
    # coarser ``loss[j]`` and finer ``loss[j+1]`` exceed optimizer_floor.
    # At resolvable steps the drop must be strictly positive.
    if int(cfg.L_deeper) in list(L_arr):
        L_deeper_idx = int(np.where(L_arr == int(cfg.L_deeper))[0][0])
    else:
        L_deeper_idx = int(len(L_arr) - 1)
        print(
            f"[C5-STRICT] warning: L_deeper = {cfg.L_deeper} not in "
            f"L_list {list(L_arr)}; falling back to last L = "
            f"{int(L_arr[L_deeper_idx])}"
        )

    L_deeper_value = int(L_arr[L_deeper_idx])
    coarser = loss[:-1, :, L_deeper_idx]  # (n_steps, n_kappa)
    finer = loss[1:, :, L_deeper_idx]
    resolved_mask = (coarser > cfg.optimizer_floor) & (
        finer > cfg.optimizer_floor
    )
    # Restrict to κ > 1 cells only (κ = 1 has no theorem-predicted strict drop).
    kappa_row_mask = np.broadcast_to(
        kappa_gt1_mask.reshape(1, -1), resolved_mask.shape
    )
    resolved_relevant = resolved_mask & kappa_row_mask

    resolved_drops = coarser - finer
    resolved_count_total = int(resolved_relevant.sum())
    resolved_count_strict = int(
        (resolved_relevant & (resolved_drops > cfg.strict_tol)).sum()
    )
    # Failures above precision: resolved-relevant cells whose drop is
    # NOT strict. Must be zero to pass the theorem-level check.
    failing_above_mask = resolved_relevant & (
        resolved_drops <= cfg.strict_tol
    )
    failing_above_precision = int(failing_above_mask.sum())

    resolved_failing_cells: list[dict[str, Any]] = []
    for (si, ki) in np.argwhere(failing_above_mask):
        resolved_failing_cells.append(
            {
                "step": int(si),
                "level_from": int(si),
                "level_to": int(si + 1),
                "kappa": float(kappa_arr[ki]),
                "L": int(L_deeper_value),
                "loss_coarser": float(coarser[si, ki]),
                "loss_finer": float(finer[si, ki]),
                "drop": float(resolved_drops[si, ki]),
            }
        )

    if resolved_count_total > 0:
        flat_resolved_idx = np.argwhere(resolved_relevant)
        drops_at_resolved = resolved_drops[resolved_relevant]
        min_drop_idx = int(np.argmin(drops_at_resolved))
        loc = flat_resolved_idx[min_drop_idx]
        resolved_min_drop = float(drops_at_resolved[min_drop_idx])
        resolved_min_loc = {
            "step": int(loc[0]),
            "level_from": int(loc[0]),
            "level_to": int(loc[0] + 1),
            "kappa": float(kappa_arr[loc[1]]),
            "L": int(L_deeper_value),
            "loss_coarser": float(coarser[loc[0], loc[1]]),
            "loss_finer": float(finer[loc[0], loc[1]]),
        }
    else:
        resolved_min_drop = float("inf")
        resolved_min_loc = {
            "step": None, "level_from": None, "level_to": None,
            "kappa": None, "L": None,
            "loss_coarser": None, "loss_finer": None,
        }

    # Also compute the below-precision count for context.
    below_precision_mask = kappa_row_mask & ~resolved_mask
    failing_below_mask = below_precision_mask & (
        resolved_drops <= cfg.strict_tol
    )
    failing_below_precision = int(failing_below_mask.sum())
    failing_below_cells: list[dict[str, Any]] = []
    for (si, ki) in np.argwhere(failing_below_mask):
        failing_below_cells.append(
            {
                "step": int(si),
                "level_from": int(si),
                "level_to": int(si + 1),
                "kappa": float(kappa_arr[ki]),
                "L": int(L_deeper_value),
                "loss_coarser": float(coarser[si, ki]),
                "loss_finer": float(finer[si, ki]),
                "drop": float(resolved_drops[si, ki]),
            }
        )

    strict_drops_resolved_ok = (
        resolved_count_total > 0
        and resolved_count_strict == resolved_count_total
        and failing_above_precision == 0
    )

    return {
        # Check 1
        "strict_drops_all_kappa_gt1_ok": bool(
            strict_drops_all_kappa_gt1_ok
        ),
        "gt1_cells": gt1_cells,
        "gt1_worst": gt1_worst,
        "total_steps": int(total_steps),
        # Check 2
        "kappa1_no_strict_drops_ok": bool(kappa1_no_strict_drops_ok),
        "kappa1_no_strict_drops_resolved_ok": bool(
            kappa1_no_strict_drops_resolved_ok
        ),
        "kappa_eq1_max_loss_overall": float(kappa_eq1_max_loss_overall),
        "kappa_eq1_cells": kappa_eq1_cells,
        # Check 3 (resolved diagnostic at L_deeper)
        "L_deeper": int(L_deeper_value),
        "strict_drops_resolved_ok": bool(strict_drops_resolved_ok),
        "resolved_count_total": int(resolved_count_total),
        "resolved_count_strict": int(resolved_count_strict),
        "resolved_min_drop": float(resolved_min_drop),
        "resolved_min_loc": resolved_min_loc,
        "failing_above_precision": int(failing_above_precision),
        "failing_above_cells": resolved_failing_cells,
        "failing_below_precision": int(failing_below_precision),
        "failing_below_cells": failing_below_cells,
    }


# ---------------------------------------------------------------------------
# Recompute fallback
# ---------------------------------------------------------------------------


def _try_recompute(cfg: C5StrictDropsConfig) -> dict[str, np.ndarray]:
    """Re-run the C5 sweep locally (~ 11 s)."""
    from scripts.thesis.theoremC import (  # noqa: F401
        run_theoremC_refinement_monotonicity as _c5,
    )

    c5_cfg = _c5.C5Config(
        D=cfg.D,
        kappa_list=cfg.kappa_list,
        L_list=cfg.L_list,
    )
    data = _c5.run_refinement_ladder(c5_cfg)
    return {
        "kappa_list": np.asarray(cfg.kappa_list, dtype=np.float64),
        "L_list": np.asarray(cfg.L_list, dtype=np.int64),
        "level_sizes": np.asarray(
            cfg.expected_level_sizes, dtype=np.int64
        ),
        "per_level_nblocks": np.asarray(
            [cfg.D // s for s in cfg.expected_level_sizes],
            dtype=np.int64,
        ),
        "loss_grid": data["loss_grid"],
        "converged_grid": data["converged_grid"],
    }


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_summary(
    run_dir: ThesisRunDir,
    cfg: C5StrictDropsConfig,
    results: dict[str, Any],
    source: str,
) -> Path:
    path = run_dir.root / "c5_strict_drops_patch_summary.txt"
    lines: list[str] = []
    lines.append(
        "Experiment C5 strict-drops patch — per-item acceptance summary"
    )
    lines.append("=" * 72)
    lines.append("Plan ref: EXPERIMENT_PLAN_FINAL.MD §7.5 (C5)")
    lines.append(
        "Theorem ref: thesis/theorem_c.txt — Proposition 3.15 "
        "(monotonicity, weak direction), Corollary 3.16 (strict "
        "hybrid gain along the refinement ladder)."
    )
    lines.append(
        f"Config: D = {cfg.D}, kappa_list = {list(cfg.kappa_list)}, "
        f"L_list = {list(cfg.L_list)}, level_sizes = "
        f"{list(cfg.expected_level_sizes)}"
    )
    lines.append(f"Data source: {source}")
    lines.append("")

    def _mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    total_steps = results["total_steps"]

    # Check 1
    lines.append(
        "Check 1 — Corollary 3.16 strict drops at every κ > 1"
    )
    lines.append(
        f"  Assertion: strict_drops == total_steps (= {total_steps}) "
        f"for every (κ > 1, L)"
    )
    for c in results["gt1_cells"]:
        lines.append(
            f"    κ = {c['kappa']:>5.2f}  L = {c['L']:>2d}  "
            f"strict_drops = {c['strict_drops']:>2d} / "
            f"{c['total_steps']:>2d}   {_mark(c['ok'])}"
        )
    lines.append(
        f"  → strict_drops_all_kappa_gt1_ok: "
        f"{_mark(results['strict_drops_all_kappa_gt1_ok'])}"
    )
    lines.append("")

    # Check 2
    lines.append("Check 2 — κ = 1 ⇒ no strict drops")
    lines.append(
        f"  Assertion (raw): strict_drops == 0 at κ = 1 for every L"
    )
    for c in results["kappa_eq1_cells"]:
        lines.append(
            f"    κ = {c['kappa']:>5.2f}  L = {c['L']:>2d}  "
            f"strict_drops = {c['strict_drops']:>2d}  "
            f"max_loss = {c['max_loss_all_levels']:.3e}   "
            f"{_mark(c['ok'])}"
        )
    lines.append(
        f"  → kappa1_no_strict_drops_ok: "
        f"{_mark(results['kappa1_no_strict_drops_ok'])}"
    )
    lines.append(
        f"  Assertion (resolved): max_loss at κ = 1 < "
        f"{cfg.optimizer_floor:.1e} (ladder is flat at precision floor)"
    )
    lines.append(
        f"    max loss across all levels at κ = 1 = "
        f"{results['kappa_eq1_max_loss_overall']:.3e}"
    )
    lines.append(
        f"  → kappa1_no_strict_drops_resolved_ok: "
        f"{_mark(results['kappa1_no_strict_drops_resolved_ok'])}"
    )
    lines.append("")

    # Check 3
    lines.append(
        f"Check 3 — resolved diagnostic at L = {results['L_deeper']}"
    )
    lines.append(
        f"  Restriction: κ > 1 steps where both loss_coarser and "
        f"loss_finer > {cfg.optimizer_floor:.1e}"
    )
    lines.append(
        f"    resolved_count_total  = {results['resolved_count_total']}"
    )
    lines.append(
        f"    resolved_count_strict = "
        f"{results['resolved_count_strict']} / "
        f"{results['resolved_count_total']}"
    )
    lines.append(
        f"    resolved_min_drop     = {results['resolved_min_drop']:.3e}"
    )
    rloc = results["resolved_min_loc"]
    if rloc["kappa"] is not None:
        lines.append(
            f"    resolved min step     = (j = {rloc['step']} → "
            f"{rloc['level_to']}, κ = {rloc['kappa']:.3g}, "
            f"L = {rloc['L']})"
        )
        lines.append(
            f"      loss_coarser = {rloc['loss_coarser']:.4e}  "
            f"loss_finer = {rloc['loss_finer']:.4e}"
        )
    lines.append(
        f"    failing_above_precision = "
        f"{results['failing_above_precision']}  "
        f"(must be 0 for theorem-level PASS)"
    )
    lines.append(
        f"    failing_below_precision = "
        f"{results['failing_below_precision']}  "
        f"(below {cfg.optimizer_floor:.1e}; optimizer noise, "
        f"not theorem violation)"
    )
    if results["failing_above_cells"]:
        lines.append(
            "    ABOVE-precision failing cells "
            "(theorem-level failures — investigate):"
        )
        for c in results["failing_above_cells"]:
            lines.append(
                f"      step {c['level_from']}→{c['level_to']}  "
                f"κ = {c['kappa']:.3g}  L = {c['L']}  "
                f"drop = {c['drop']:+.3e}  "
                f"Lc = {c['loss_coarser']:.3e}  "
                f"Lf = {c['loss_finer']:.3e}"
            )
    if results["failing_below_cells"]:
        lines.append(
            "    below-precision failing cells (diagnostic):"
        )
        for c in results["failing_below_cells"]:
            lines.append(
                f"      step {c['level_from']}→{c['level_to']}  "
                f"κ = {c['kappa']:.3g}  L = {c['L']}  "
                f"drop = {c['drop']:+.3e}  "
                f"Lc = {c['loss_coarser']:.3e}  "
                f"Lf = {c['loss_finer']:.3e}"
            )
    lines.append(
        f"  → strict_drops_resolved_ok: "
        f"{_mark(results['strict_drops_resolved_ok'])}"
    )
    lines.append("")

    raw_top_line_ok = (
        results["strict_drops_all_kappa_gt1_ok"]
        and results["kappa1_no_strict_drops_ok"]
    )
    theorem_level_ok = (
        results["strict_drops_resolved_ok"]
        and results["kappa1_no_strict_drops_resolved_ok"]
        and results["failing_above_precision"] == 0
    )
    lines.append("=" * 72)
    lines.append(
        f"Top-line status (raw {cfg.strict_tol:.0e} threshold): "
        f"{_mark(raw_top_line_ok)}"
    )
    lines.append(
        f"Theorem-level status (resolved cells only): "
        f"{_mark(theorem_level_ok)}"
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment C5 strict-drops patch: load C5 loss_grid and "
            "assert Cor 3.16 strict positivity along the ladder for "
            "every κ > 1, plus the complementary κ = 1 zero-drop "
            "region and a resolved diagnostic at L_deeper."
        )
    )
    p.add_argument(
        "--c5-run", type=str, default=None,
        help=(
            "Path to a canonical C5 run directory (default: most recent "
            "outputs/thesis/theoremC/run_theoremC_refinement_monotonicity/"
            "<run_id>)"
        ),
    )
    p.add_argument(
        "--recompute", action="store_true",
        help=(
            "Re-run the C5 sweep locally (~ 11 s) instead of loading "
            "the canonical npz."
        ),
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _cli()
    cfg = C5StrictDropsConfig()

    synthetic_script = Path(__file__).parent / "c5_strict_drops_patch.py"
    run = ThesisRunDir(synthetic_script, phase="theoremC")
    print(f"[C5-STRICT] output: {run.root}")

    with RunContext(
        run,
        config=cfg,
        seeds=[0],
        notes=(
            "C5 strict-drops patch: acceptance-record fix recording the "
            "Corollary 3.16 strict positivity of ladder refinement drops "
            "at every κ > 1, the κ = 1 zero-drop assertion, and the "
            "resolved diagnostic at L_deeper."
        ),
    ) as ctx:
        project_root = _PROJ
        source_desc = "unknown"
        if args.recompute:
            print("[C5-STRICT] --recompute: re-running the C5 sweep locally")
            import time
            t0 = time.perf_counter()
            data = _try_recompute(cfg)
            dt = time.perf_counter() - t0
            source_desc = f"recomputed locally ({dt:.1f} s)"
            ctx.record_compute_proxy(float(dt))
            ctx.record_step_time(dt)
        else:
            if args.c5_run:
                c5_run = Path(args.c5_run)
            else:
                latest = _find_latest_c5_run(project_root)
                if latest is None:
                    raise FileNotFoundError(
                        "No canonical C5 run found under "
                        f"{project_root / _CANONICAL_C5_ROOT}. "
                        "Pass --c5-run or use --recompute."
                    )
                c5_run = latest
            print(f"[C5-STRICT] loading canonical C5 run: {c5_run}")
            data = _load_c5_npz(c5_run)
            source_desc = (
                f"canonical C5 npz: "
                f"{c5_run.relative_to(project_root)}"
                "/npz/refinement_ladder.npz"
            )
            ctx.record_extra("canonical_c5_run", str(c5_run))

        _validate_axes(cfg, data)
        results = _run_checks(cfg, data)

        ctx.record_extra("checks", results)
        ctx.record_extra("data_source", source_desc)

        summary_path = _write_summary(run, cfg, results, source_desc)

        raw_ok = (
            results["strict_drops_all_kappa_gt1_ok"]
            and results["kappa1_no_strict_drops_ok"]
        )
        theorem_level_ok = (
            results["strict_drops_resolved_ok"]
            and results["kappa1_no_strict_drops_resolved_ok"]
            and results["failing_above_precision"] == 0
        )

        ctx.write_summary(
            {
                "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §7.5 (C5)",
                "theorem_reference": (
                    "thesis/theorem_c.txt — Proposition 3.15 "
                    "(monotonicity, weak direction) + Corollary 3.16 "
                    "(strict hybrid gain applied along the full dyadic "
                    "refinement ladder)."
                ),
                "category": (
                    "acceptance-record-only patch on existing C5 data. "
                    "No new sweep, no new figure. Lifts C5's raw "
                    "strict_drop_counts to formal acceptance gates, "
                    "mirroring the C4 strict-gain patch."
                ),
                "interpretation": (
                    "Proposition 3.15 (weak direction) was already "
                    "confirmed by C5's monotonicity_ok gate. This "
                    "patch records the strict direction: every κ > 1 "
                    "cell should produce a strict loss decrease at "
                    "every step of the dyadic ladder (total_steps = "
                    "n_levels − 1 = 6 for the D = 64 ladder), and κ = "
                    "1 should produce none. At the deeper slice L = "
                    "L_deeper the loss can saturate to the L-BFGS "
                    "precision floor before the ladder ends; below "
                    "that floor the sign of a 'strict drop' is "
                    "optimizer noise, not a theorem violation. Two "
                    "resolved diagnostics capture this: "
                    "strict_drops_resolved_ok restricts the κ > 1 "
                    "check to steps where both the coarser and finer "
                    "levels exceed optimizer_floor = 1e-9 "
                    "(failing_above_precision must be 0); "
                    "kappa1_no_strict_drops_resolved_ok replaces the "
                    "raw κ = 1 zero-drop assertion with the "
                    "more-faithful theorem-level gate 'max loss across "
                    "all ladder levels at κ = 1 stays below "
                    "optimizer_floor' — i.e. the ladder is flat at "
                    "the precision floor, which is what the theorem "
                    "actually predicts."
                ),
                "data_source": source_desc,
                "D": cfg.D,
                "kappa_list": list(cfg.kappa_list),
                "L_list": list(cfg.L_list),
                "expected_level_sizes": list(cfg.expected_level_sizes),
                "strict_tol": cfg.strict_tol,
                "optimizer_floor": cfg.optimizer_floor,
                "L_deeper": int(cfg.L_deeper),
                "strict_drops_all_kappa_gt1_ok": results[
                    "strict_drops_all_kappa_gt1_ok"
                ],
                "kappa1_no_strict_drops_ok": results[
                    "kappa1_no_strict_drops_ok"
                ],
                "kappa1_no_strict_drops_resolved_ok": results[
                    "kappa1_no_strict_drops_resolved_ok"
                ],
                "kappa_eq1_max_loss_overall": results[
                    "kappa_eq1_max_loss_overall"
                ],
                "strict_drops_resolved_ok": results[
                    "strict_drops_resolved_ok"
                ],
                "resolved_count_total": results["resolved_count_total"],
                "resolved_count_strict": results["resolved_count_strict"],
                "resolved_min_drop": results["resolved_min_drop"],
                "resolved_min_loc": results["resolved_min_loc"],
                "failing_above_precision": results[
                    "failing_above_precision"
                ],
                "failing_below_precision": results[
                    "failing_below_precision"
                ],
                "total_steps": results["total_steps"],
                "raw_top_line_ok": bool(raw_ok),
                "theorem_level_ok": bool(theorem_level_ok),
                "status": (
                    ("strict_kappa_gt1_ok"
                     if results["strict_drops_all_kappa_gt1_ok"]
                     else "strict_kappa_gt1_fail")
                    + "+"
                    + ("kappa1_zero_raw_ok"
                       if results["kappa1_no_strict_drops_ok"]
                       else "kappa1_zero_raw_fail")
                    + "+"
                    + ("kappa1_zero_resolved_ok"
                       if results["kappa1_no_strict_drops_resolved_ok"]
                       else "kappa1_zero_resolved_fail")
                    + "+"
                    + ("resolved_ok"
                       if results["strict_drops_resolved_ok"]
                       else "resolved_fail")
                ),
                "patch_summary_path": str(summary_path),
            }
        )

        print()
        print("=" * 72)
        print(" C5 strict-drops patch")
        total_steps = results["total_steps"]
        worst = results.get("gt1_worst")
        if worst is not None:
            print(
                f"   Check 1 (strict drops at κ > 1):  "
                f"worst = (κ={worst['kappa']:.2f}, "
                f"L={worst['L']}, "
                f"{worst['strict_drops']}/{total_steps})  "
                f"{'OK' if results['strict_drops_all_kappa_gt1_ok'] else 'FAIL'}"
            )
        print(
            f"   Check 2 raw (κ = 1 ⇒ 0 drops):      "
            f"{'OK' if results['kappa1_no_strict_drops_ok'] else 'FAIL'}"
        )
        print(
            f"   Check 2 resolved (max loss @ κ=1 < "
            f"{cfg.optimizer_floor:.1e}):  "
            f"max = {results['kappa_eq1_max_loss_overall']:.3e}  "
            f"{'OK' if results['kappa1_no_strict_drops_resolved_ok'] else 'FAIL'}"
        )
        print(
            f"   Check 3 resolved (L = "
            f"{results['L_deeper']}, scale > "
            f"{cfg.optimizer_floor:.1e}):  "
            f"{results['resolved_count_strict']}/"
            f"{results['resolved_count_total']}  "
            f"min drop = {results['resolved_min_drop']:.3e}  "
            f"{'OK' if results['strict_drops_resolved_ok'] else 'FAIL'}"
        )
        print(
            f"     failing above precision: "
            f"{results['failing_above_precision']}   "
            f"failing below precision: "
            f"{results['failing_below_precision']}"
        )
        print(
            f"   Theorem-level status: "
            f"{'OK' if theorem_level_ok else 'FAIL'}"
        )
        print(f"   summary: {summary_path}")
        print("=" * 72)

        return 0 if theorem_level_ok else 1


if __name__ == "__main__":
    sys.exit(main())
