"""Experiment C5: theorem-C refinement monotonicity ladder.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §7.5.

Purpose
-------
Validate the theorem-C **refinement monotonicity** statement exactly at the
operator level, extending C4's single dyadic step to a full ladder from the
coarsest partition down to singletons:

    B^{(0)}  ⊒  B^{(1)}  ⊒  ...  ⊒  B^{(J)},

with ``J + 1`` ladder levels and ``J`` refinement steps. On the default
D = 64 dyadic ladder this is **7 ladder levels and 6 refinement steps**
(block sizes 64 → 32 → 16 → 8 → 4 → 2 → 1).

For a fixed ambient spectrum ``(λ, ω)`` and a fixed depth ``L``, let

    L★(j, κ, L) =
        min_{q ∈ C(B^{(j)})}  Σ_k  ω_k · λ_k · (1 − L⁻¹ · λ_k · q_{b(k)})^(2L),

i.e. the theorem-C optimum over the block-commutant class at level ``j``.
Refinement monotonicity asserts

    L★(0) ≥ L★(1) ≥ ... ≥ L★(J) = 0,

with **strict** drops at every step that separates heterogeneous modes. C5
makes this monotonicity visually explicit across a sweep of within-block
heterogeneity ``κ``, revealing the transition from

- κ = 1 (no heterogeneity): every level trivially attains L★ = 0 and the
  ladder is flat;

to

- large κ: each dyadic split that separates heterogeneous modes gives a
  strict drop, and the full ladder traces a monotone staircase from a
  large L★(0) down to 0 at the singleton.

C5 extends C4's single-step picture (coarse vs one-step-finer) to the full
multi-level hybrid-refinement story; it is the exact-theorem counterpart
to the architecture-aligned "hybrid advantage" figure in the later tiers.

Spectrum construction (mass-preserving, matched to C3 / C4)
-----------------------------------------------------------
Uses :func:`data_generators.g3_generate_from_config` with
``ladder_kind="dyadic"`` and ``reference_partition_index = 0`` (the
coarsest, single-block level). In that configuration, the heterogeneity
κ spreads across the *entire* spectrum under the mass-preserving
linear-ξ construction, and the resulting ``(λ, ω)`` is shared bitwise
across every ladder level — only the partition-being-optimized-over
changes.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``G3Config``, ``g3_generate_from_config``.
- :mod:`scripts.thesis.utils.metrics`:
    ``oracle_commutant_loss`` (L-BFGS over block scalars at any L).
- :mod:`scripts.thesis.utils.partitions`:
    ``BlockPartition`` (consumed via G3 return).
- :mod:`scripts.thesis.utils.plotting`, :mod:`run_metadata`: standard.

Primary outputs
---------------
**Main ladder figure** (``c5_refinement_ladder``): loss-vs-refinement-level
line plot, one line per κ, overlaid for two depths. Shows the κ = 1 flat
curve and the staircase for large κ side-by-side, making the transition
visually explicit.

**Per-level drops** (``c5_level_drops``): bar chart of per-step drop
``L★(j) − L★(j+1)`` vs level index, one bar group per κ (primary depth).
Quantifies where the strict improvements actually happen.

**Ladder heatmap** (``c5_ladder_heatmap``): ``L★(level, κ)`` heatmap,
block-size (equivalent to level index) on y, κ on x, log color. The 2-D
view of the monotone-staircase structure.

**Depth comparison** (``c5_depth_comparison``): two-panel side-by-side
``c5_refinement_ladder`` at ``L = 1`` and ``L = L_deeper`` to confirm the
ladder structure persists across depth (secondary axis).

Acceptance
----------
1. **Monotonicity**. For every (κ, L) and every consecutive pair j → j+1:
   ``L★(j+1) ≤ L★(j) + monotonicity_tol``. (Strictly a ``≤`` with a small
   tolerance for float noise; C5's positive claim is non-increase.)
2. **κ = 1 degeneracy**. At κ = 1 every level's optimum is 0 to float eps.
3. **Finest level ≡ 0**. At the singleton level (last entry in the ladder),
   ``L★`` is 0 within ``finest_tol`` for every κ and every L.
4. **Diagnostic (strict drops)**. For κ above a threshold, the number of
   strictly-decreasing steps along the ladder is reported; at large κ we
   expect every step with heterogeneous-mode separation to drop. Not a
   hard gate (the linear-ξ construction can produce a marginal step at
   extreme κ where one mode dominates).

Run
---
::

    python -u scripts/thesis/theoremC/run_theoremC_refinement_monotonicity.py \\
           --device cuda --dtype float64 --no-show
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import matplotlib
import numpy as np
import torch

from scripts.thesis.utils.data_generators import (
    G3Config,
    g3_generate_from_config,
)
from scripts.thesis.utils.metrics import oracle_commutant_loss
from scripts.thesis.utils.partitions import BlockPartition
from scripts.thesis.utils.plotting import (
    apply_thesis_style,
    overlay_reference,
    phase_heatmap,
    save_both,
    sequential_colors,
)
from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C5Config:
    """Frozen configuration for the refinement-monotonicity ladder.

    Default ladder is the full dyadic refinement on D = 64:
    **7 ladder levels and 6 refinement steps** (block sizes
    64 → 32 → 16 → 8 → 4 → 2 → 1). Default κ sweep has 7 values that span
    the flat-ladder regime to the full-staircase regime. Two depths
    confirm the ladder structure across L.
    """

    D: int = 64
    # Dyadic ladder depth. ``None`` ⇒ full dyadic (log2(D) levels).
    J: int | None = None
    # Reference partition for mass-preserving spectrum construction. Index
    # 0 = coarsest = single block; κ then spreads across ALL D modes.
    reference_partition_index: int = 0
    # Uniform base block mean at the reference level (length 1 because the
    # reference level is the coarsest single-block partition).
    base_block_mean_lam: float = 1.0
    base_block_mean_omega: float = 1.0

    # Heterogeneity sweep.
    kappa_list: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0)

    # Depth axis (secondary per §7.5 / user constraint: at least two L).
    L_list: tuple[int, ...] = (1, 4)
    L_primary: int = 1
    L_deeper: int = 4

    # Mass-preserving conventions (matched to C3 / C4).
    xi_shape: str = "linear"
    spectral_basis_kind: str = "dct2"

    # Acceptance thresholds.
    monotonicity_tol: float = 1e-8
    kappa_1_tol: float = 1e-9
    finest_tol: float = 1e-8

    # "Strict drop" diagnostic threshold as a fraction of L★(0).
    strict_drop_fraction: float = 1e-4

    # L-BFGS.
    optimizer: str = "lbfgs"
    max_iter: int = 500

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_g3_config(cfg: C5Config, kappa: float) -> G3Config:
    # The reference partition is index 0 (coarsest), which has n_blocks = 1
    # under the dyadic ladder. So base_block_means / base_block_kappas each
    # have length 1.
    return G3Config(
        D=cfg.D,
        ladder_kind="dyadic",
        ladder_params={"J": cfg.J} if cfg.J is not None else {},
        reference_partition_index=int(cfg.reference_partition_index),
        base_block_means_lam=(float(cfg.base_block_mean_lam),),
        base_block_kappas_lam=(float(kappa),),
        base_block_means_omega=(float(cfg.base_block_mean_omega),),
        base_block_kappas_omega=(float(kappa),),
        xi_shape=cfg.xi_shape,
        spectral_basis_kind=cfg.spectral_basis_kind,
        dtype=cfg.dtype,
    )


def _optimize_over_level(
    cfg: C5Config, lam: torch.Tensor, omega: torch.Tensor,
    partition: BlockPartition, L: int,
) -> dict[str, Any]:
    return oracle_commutant_loss(
        lam, omega, partition,
        L=int(L),
        q_init=None,
        optimizer=cfg.optimizer,
        max_iter=cfg.max_iter,
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _run_sweep(cfg: C5Config) -> dict[str, Any]:
    kappa_list = list(cfg.kappa_list)
    L_list = list(cfg.L_list)

    # Build one representative ladder to get J / level block sizes (same
    # ladder used at every κ).
    rep_levels = g3_generate_from_config(_build_g3_config(cfg, 1.0))
    n_levels = len(rep_levels)
    level_sizes = [int(entry["partition"].sizes[0]) for entry in rep_levels]
    # Dyadic ladder: partition size halves at each step. All blocks within
    # a level share one size.

    shape = (n_levels, len(kappa_list), len(L_list))
    loss_grid = np.zeros(shape)
    converged_grid = np.ones(shape, dtype=bool)
    per_level_nblocks = np.zeros(n_levels, dtype=np.int64)

    n_total = len(kappa_list) * n_levels * len(L_list)
    idx = 0
    t_start = time.perf_counter()
    for i_k, kappa in enumerate(kappa_list):
        levels = g3_generate_from_config(_build_g3_config(cfg, float(kappa)))
        if len(levels) != n_levels:
            raise AssertionError(
                f"ladder length mismatch at κ = {kappa}: "
                f"{len(levels)} vs {n_levels}"
            )
        lam = levels[0]["Lambda"]
        omega = levels[0]["Omega"]
        for i_j, entry in enumerate(levels):
            partition = entry["partition"]
            per_level_nblocks[i_j] = partition.n_blocks
            for i_L, L_val in enumerate(L_list):
                idx += 1
                t0 = time.perf_counter()
                res = _optimize_over_level(
                    cfg, lam, omega, partition, int(L_val)
                )
                dt = time.perf_counter() - t0
                loss_grid[i_j, i_k, i_L] = float(res["loss_star"])
                converged_grid[i_j, i_k, i_L] = bool(res["converged"])
                print(
                    f"[{idx:>4d}/{n_total}] "
                    f"κ = {float(kappa):>5.2f}  "
                    f"lvl = {i_j:>2d} (m = {level_sizes[i_j]:>3d}, "
                    f"n_blocks = {partition.n_blocks:>3d})  "
                    f"L = {int(L_val):>2d}  "
                    f"L* = {loss_grid[i_j, i_k, i_L]:.6e}  "
                    f"conv = {bool(res['converged'])}  "
                    f"({dt*1000:.1f} ms)"
                )
    total_wall = time.perf_counter() - t_start

    return {
        "n_levels": n_levels,
        "level_sizes": level_sizes,
        "per_level_nblocks": per_level_nblocks,
        "kappa_list": kappa_list,
        "L_list": L_list,
        "loss_grid": loss_grid,
        "converged_grid": converged_grid,
        "total_wallclock": total_wall,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _level_axis(result: dict[str, Any]) -> np.ndarray:
    """Integer level index 0..J (coarsest → finest)."""
    return np.arange(result["n_levels"], dtype=float)


def _plot_refinement_ladder(
    cfg: C5Config, result: dict[str, Any], run_dir: ThesisRunDir,
    *, L_value: int, fname: str, title: str,
) -> None:
    """Plot L★ vs refinement level, one line per κ, at a single depth."""
    import matplotlib.pyplot as plt

    L_list = list(result["L_list"])
    if int(L_value) not in L_list:
        return
    i_L = L_list.index(int(L_value))
    loss_slice = result["loss_grid"][:, :, i_L]  # (n_levels, n_kappa)

    kappa_list = list(result["kappa_list"])
    n_kappa = len(kappa_list)
    level_axis = _level_axis(result)
    level_sizes = result["level_sizes"]
    k_colors = sequential_colors(n_kappa, palette="rocket")
    floor = 1e-18

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for color, i_k in zip(k_colors, range(n_kappa)):
        y = loss_slice[:, i_k]
        y_plot = np.where(y > floor, y, np.nan)
        ax.plot(
            level_axis, y_plot, color=color, lw=1.5, marker="o", ms=4.5,
            label=rf"$\kappa = {kappa_list[i_k]:.2g}$",
        )

    ax.set_yscale("log")
    ax.set_xlabel(
        r"refinement level $j$ (coarsest $\to$ finest)"
    )
    ax.set_ylabel(r"block-commutant optimum $L^\star(j, \kappa)$")
    # Dual x-axis label showing block size per level.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(list(level_axis))
    ax2.set_xticklabels([f"m = {s}" for s in level_sizes], fontsize=7)
    ax2.tick_params(axis="x", which="both", labelsize=7)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    save_both(fig, run_dir, fname)
    plt.close(fig)


def _plot_depth_comparison(
    cfg: C5Config, result: dict[str, Any], run_dir: ThesisRunDir
) -> None:
    """Side-by-side refinement ladders at L_primary and L_deeper."""
    import matplotlib.pyplot as plt

    L_list = list(result["L_list"])
    if (
        int(cfg.L_primary) not in L_list
        or int(cfg.L_deeper) not in L_list
    ):
        return
    ilp = L_list.index(int(cfg.L_primary))
    ild = L_list.index(int(cfg.L_deeper))

    kappa_list = list(result["kappa_list"])
    level_axis = _level_axis(result)
    level_sizes = result["level_sizes"]
    k_colors = sequential_colors(len(kappa_list), palette="rocket")
    floor = 1e-18

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    for ax, i_L, L_val in (
        (axes[0], ilp, cfg.L_primary),
        (axes[1], ild, cfg.L_deeper),
    ):
        slice_ = result["loss_grid"][:, :, i_L]
        for color, i_k in zip(k_colors, range(len(kappa_list))):
            y = slice_[:, i_k]
            y_plot = np.where(y > floor, y, np.nan)
            ax.plot(
                level_axis, y_plot, color=color, lw=1.5, marker="o", ms=4.5,
                label=rf"$\kappa = {kappa_list[i_k]:.2g}$",
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"refinement level $j$ (coarsest $\to$ finest)")
        ax.set_title(rf"L = {int(L_val)}", fontsize=11)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    axes[0].set_ylabel(r"$L^\star(j, \kappa)$")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "C5 refinement ladder across depth: same staircase structure "
        "persists",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_both(fig, run_dir, "c5_depth_comparison")
    plt.close(fig)


def _plot_level_drops(
    cfg: C5Config, result: dict[str, Any], run_dir: ThesisRunDir
) -> None:
    """Per-level drop L★(j) − L★(j+1) at the primary depth, one grouped
    bar per κ."""
    import matplotlib.pyplot as plt

    L_list = list(result["L_list"])
    if int(cfg.L_primary) not in L_list:
        return
    i_L = L_list.index(int(cfg.L_primary))
    loss_slice = result["loss_grid"][:, :, i_L]
    drops = loss_slice[:-1, :] - loss_slice[1:, :]  # (n_levels - 1, n_kappa)
    n_levels = result["n_levels"]
    kappa_list = list(result["kappa_list"])
    k_colors = sequential_colors(len(kappa_list), palette="rocket")

    # Bar plot: x is j → j+1 step index; one group per step with n_kappa bars.
    n_steps = n_levels - 1
    step_index = np.arange(n_steps)
    width = 0.8 / len(kappa_list)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    floor = 1e-30
    for i_k in range(len(kappa_list)):
        offset = (i_k - (len(kappa_list) - 1) / 2) * width
        heights = np.where(drops[:, i_k] > floor, drops[:, i_k], floor)
        ax.bar(
            step_index + offset, heights, width=width * 0.92,
            color=k_colors[i_k], edgecolor="black", lw=0.3,
            label=rf"$\kappa = {kappa_list[i_k]:.2g}$",
        )
    ax.set_yscale("log")
    ax.set_xlabel("dyadic refinement step (j → j + 1)")
    ax.set_ylabel(r"per-step drop $L^\star(j) - L^\star(j+1)$")
    ax.set_xticks(step_index)
    ax.set_xticklabels(
        [
            f"{result['level_sizes'][i]}→{result['level_sizes'][i + 1]}"
            for i in range(n_steps)
        ],
        fontsize=8,
    )
    ax.set_title(
        rf"C5 per-step refinement drops (L = {cfg.L_primary}); "
        r"every step is nonnegative (monotonicity); strict > 0 where "
        r"heterogeneous modes are separated",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    save_both(fig, run_dir, "c5_level_drops")
    plt.close(fig)


def _plot_ladder_heatmap(
    cfg: C5Config, result: dict[str, Any], run_dir: ThesisRunDir
) -> None:
    """2D heatmap of L★(level, κ) at the primary depth."""
    import matplotlib.pyplot as plt

    L_list = list(result["L_list"])
    if int(cfg.L_primary) not in L_list:
        return
    i_L = L_list.index(int(cfg.L_primary))
    loss_slice = result["loss_grid"][:, :, i_L]
    level_sizes = result["level_sizes"]
    kappa_list = list(result["kappa_list"])
    k_arr = np.asarray(kappa_list, dtype=float)
    m_arr = np.asarray(level_sizes, dtype=float)
    floor = 1e-18
    loss_plot = np.where(loss_slice > floor, loss_slice, floor)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    _pc, _cb = phase_heatmap(
        ax, loss_plot,
        x_coords=k_arr, y_coords=m_arr,
        xlabel=r"within-block heterogeneity $\kappa$",
        ylabel=r"block size $m$ (level: coarsest=$D$, finest=$1$)",
        cbar_label=r"$L^\star(j, \kappa)$",
        log_z=True, log_x=True, log_y=True,
    )
    ax.set_title(
        rf"C5 ladder heatmap at L = {cfg.L_primary}: monotone staircase "
        "along m, flat at κ = 1",
        fontsize=10,
    )
    fig.tight_layout()
    save_both(fig, run_dir, "c5_ladder_heatmap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_list_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment C5: theorem-C refinement monotonicity ladder "
            "(plan §7.5)."
        )
    )
    p.add_argument(
        "--device", type=str, default="cuda", choices=("cpu", "cuda", "auto")
    )
    p.add_argument(
        "--dtype", type=str, default="float64", choices=("float32", "float64")
    )
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--D", type=int, default=None)
    p.add_argument("--J", type=int, default=None,
                   help="ladder depth; None = full dyadic log2(D)")
    p.add_argument("--kappa-list", type=str, default=None)
    p.add_argument("--L-list", type=str, default=None)
    p.add_argument("--L-primary", type=int, default=None)
    p.add_argument("--L-deeper", type=int, default=None)
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> C5Config:
    base = C5Config()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.D is not None:
        overrides["D"] = int(args.D)
    if args.J is not None:
        overrides["J"] = int(args.J)
    if args.kappa_list is not None:
        overrides["kappa_list"] = _parse_list_floats(args.kappa_list)
    if args.L_list is not None:
        overrides["L_list"] = _parse_list_ints(args.L_list)
    if args.L_primary is not None:
        overrides["L_primary"] = int(args.L_primary)
    if args.L_deeper is not None:
        overrides["L_deeper"] = int(args.L_deeper)
    return replace(base, **overrides) if overrides else base


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Source starter.sh in an environment with CUDA."
            )
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> int:
    args = _cli()
    if args.no_show:
        matplotlib.use("Agg")

    cfg = _config_from_cli(args)
    device = _resolve_device(cfg.device)
    print(f"[C5] device = {device}")

    run = ThesisRunDir(__file__, phase="theoremC")
    with RunContext(
        run,
        config=cfg,
        seeds=[0, 1, 2, 3],
        notes=(
            "C5 theorem-C refinement monotonicity ladder. Operator-level "
            "exact optimization only; no learned architectures. Extends "
            "C4's one-step dyadic gain to a full multi-level ladder."
        ),
    ) as ctx:
        apply_thesis_style()

        result = _run_sweep(cfg)

        # --- Figures ---
        _plot_refinement_ladder(
            cfg, result, run,
            L_value=cfg.L_primary,
            fname="c5_refinement_ladder",
            title=(
                rf"C5 refinement monotonicity ladder (L = {cfg.L_primary}); "
                r"κ=1 flat; large κ staircase"
            ),
        )
        _plot_level_drops(cfg, result, run)
        _plot_ladder_heatmap(cfg, result, run)
        _plot_depth_comparison(cfg, result, run)

        # --- Save npz ---
        npz_payload: dict[str, np.ndarray] = {
            "kappa_list": np.asarray(result["kappa_list"], dtype=np.float64),
            "L_list": np.asarray(result["L_list"], dtype=np.int64),
            "level_sizes": np.asarray(result["level_sizes"], dtype=np.int64),
            "per_level_nblocks": result["per_level_nblocks"],
            "loss_grid": result["loss_grid"],
            "converged_grid": result["converged_grid"],
        }
        np.savez_compressed(run.npz_path("refinement_ladder"), **npz_payload)

        # --- Per-cell JSON ---
        rows: list[dict[str, Any]] = []
        for i_j in range(result["n_levels"]):
            for i_k, kappa in enumerate(result["kappa_list"]):
                for i_L, L_val in enumerate(result["L_list"]):
                    rows.append(
                        {
                            "level": int(i_j),
                            "block_size": int(result["level_sizes"][i_j]),
                            "n_blocks": int(
                                result["per_level_nblocks"][i_j]
                            ),
                            "kappa": float(kappa),
                            "L": int(L_val),
                            "L_star": float(
                                result["loss_grid"][i_j, i_k, i_L]
                            ),
                            "converged": bool(
                                result["converged_grid"][i_j, i_k, i_L]
                            ),
                        }
                    )
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

        # --- Acceptance ---
        loss_grid = result["loss_grid"]

        # 1. Monotonicity along the ladder for every (κ, L).
        worst_mono_increase = 0.0
        worst_mono_cell: dict[str, Any] | None = None
        for i_k, kappa in enumerate(result["kappa_list"]):
            for i_L, L_val in enumerate(result["L_list"]):
                for i_j in range(result["n_levels"] - 1):
                    delta = (
                        loss_grid[i_j + 1, i_k, i_L]
                        - loss_grid[i_j, i_k, i_L]
                    )
                    # delta > 0 means L★ INCREASED with refinement — a
                    # violation of refinement monotonicity.
                    if delta > worst_mono_increase:
                        worst_mono_increase = float(delta)
                        worst_mono_cell = {
                            "level": int(i_j),
                            "kappa": float(kappa),
                            "L": int(L_val),
                            "L_star_coarser": float(loss_grid[i_j, i_k, i_L]),
                            "L_star_finer": float(
                                loss_grid[i_j + 1, i_k, i_L]
                            ),
                            "delta": float(delta),
                        }
        mono_ok = worst_mono_increase <= cfg.monotonicity_tol

        # 2. κ = 1 degenerate case.
        kappa_one_worst = 0.0
        if 1.0 in result["kappa_list"]:
            i_k1 = result["kappa_list"].index(1.0)
            kappa_one_worst = float(
                np.abs(loss_grid[:, i_k1, :]).max()
            )
        kappa_one_ok = kappa_one_worst <= cfg.kappa_1_tol

        # 3. Finest level ≡ 0.
        finest_worst = float(np.abs(loss_grid[-1, :, :]).max())
        finest_ok = finest_worst <= cfg.finest_tol

        # 4. Strict-drop diagnostic: count strict drops per κ at L_primary.
        strict_drop_counts: list[dict[str, Any]] = []
        if int(cfg.L_primary) in list(result["L_list"]):
            i_L_p = list(result["L_list"]).index(int(cfg.L_primary))
            for i_k, kappa in enumerate(result["kappa_list"]):
                series = loss_grid[:, i_k, i_L_p]
                scale = max(float(series[0]), 1e-12)
                threshold = cfg.strict_drop_fraction * scale
                strict_count = int(
                    sum(
                        1 for i_j in range(result["n_levels"] - 1)
                        if (series[i_j] - series[i_j + 1]) > threshold
                    )
                )
                strict_drop_counts.append(
                    {
                        "kappa": float(kappa),
                        "L_star_0": float(series[0]),
                        "strict_drops": strict_count,
                        "total_steps": int(result["n_levels"] - 1),
                    }
                )

        status_parts: list[str] = []
        status_parts.append(
            "monotonicity_ok" if mono_ok else
            f"monotonicity_violated(worst={worst_mono_increase:.2e})"
        )
        status_parts.append(
            "kappa1_zero_ok" if kappa_one_ok else
            f"kappa1_zero_violated(worst={kappa_one_worst:.2e})"
        )
        status_parts.append(
            "finest_zero_ok" if finest_ok else
            f"finest_zero_violated(worst={finest_worst:.2e})"
        )
        status = "+".join(status_parts)

        ctx.record_compute_proxy(float(result["total_wallclock"]))
        ctx.record_extra("worst_mono_increase", worst_mono_increase)
        ctx.record_extra("worst_mono_cell", worst_mono_cell)
        ctx.record_extra("kappa_one_worst", kappa_one_worst)
        ctx.record_extra("finest_worst", finest_worst)
        ctx.record_extra("strict_drop_counts", strict_drop_counts)

        ctx.write_summary(
            {
                "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §7.5 (C5)",
                "category": (
                    "operator-level exact theorem-C validation — "
                    "multi-level refinement ladder from coarsest "
                    "single-block to singleton partition. Extends C4's "
                    "one-step dyadic gain to the full monotone staircase. "
                    "No learned architecture."
                ),
                "interpretation": (
                    "For every κ and every depth L, the block-commutant "
                    "optimum L★(j) is monotonically non-increasing along "
                    "the dyadic refinement ladder of "
                    f"{result['n_levels']} ladder levels and "
                    f"{result['n_levels'] - 1} refinement steps "
                    "(j = 0 → J), reaching 0 at the singleton partition. "
                    "At κ = 1 every level attains 0 (trivial flat "
                    "ladder, no heterogeneity to resolve). At large κ, "
                    "every dyadic step that separates heterogeneous "
                    "modes produces a strict drop, and the full ladder "
                    "traces a monotone staircase from a large L★(0) "
                    "down to 0. The κ sweep makes this transition "
                    "visually explicit. Depth enters only as a "
                    "secondary axis; the staircase structure persists "
                    "across L, confirming that the theorem-C hybrid-"
                    "refinement story is depth-robust. The singleton "
                    "level is an oracle sanity endpoint only, not the "
                    "theorem object under test."
                ),
                "device": str(device),
                "D": cfg.D,
                "n_ladder_levels": result["n_levels"],
                "n_refinement_steps": result["n_levels"] - 1,
                "level_sizes": result["level_sizes"],
                "per_level_nblocks": result["per_level_nblocks"].tolist(),
                "kappa_list": list(cfg.kappa_list),
                "L_list": list(cfg.L_list),
                "L_primary": cfg.L_primary,
                "L_deeper": cfg.L_deeper,
                "status": status,
                "monotonicity_tol": cfg.monotonicity_tol,
                "kappa_1_tol": cfg.kappa_1_tol,
                "finest_tol": cfg.finest_tol,
                "worst_mono_increase": worst_mono_increase,
                "worst_mono_cell": worst_mono_cell,
                "kappa_one_worst": float(kappa_one_worst),
                "finest_worst": float(finest_worst),
                "strict_drop_counts": strict_drop_counts,
                "sweep_wallclock_seconds": round(
                    float(result["total_wallclock"]), 3
                ),
            }
        )

        print()
        print("=" * 72)
        print(f" C5 refinement monotonicity ladder on {device}")
        print(
            f"   monotonicity (L★(j+1) ≤ L★(j) + tol): worst Δ = "
            f"{worst_mono_increase:.3e}  "
            f"{'OK' if mono_ok else 'FAIL'}  (tol = {cfg.monotonicity_tol:.1e})"
        )
        if worst_mono_cell is not None and worst_mono_cell["delta"] > 0:
            print(
                f"     worst at lvl = {worst_mono_cell['level']}→"
                f"{worst_mono_cell['level'] + 1}, "
                f"κ = {worst_mono_cell['kappa']}, "
                f"L = {worst_mono_cell['L']}"
            )
        print(
            f"   κ = 1 → L★ ≡ 0:  worst = {kappa_one_worst:.3e}  "
            f"{'OK' if kappa_one_ok else 'FAIL'}  (tol = {cfg.kappa_1_tol:.1e})"
        )
        print(
            f"   finest (m = 1) ≡ 0: worst = {finest_worst:.3e}  "
            f"{'OK' if finest_ok else 'FAIL'}  (tol = {cfg.finest_tol:.1e})"
        )
        print("   Strict-drop counts per κ (diagnostic) at L = "
              f"{cfg.L_primary}:")
        for row in strict_drop_counts:
            print(
                f"     κ = {row['kappa']:>5.2f}  "
                f"L★(0) = {row['L_star_0']:.3e}  "
                f"strict drops = {row['strict_drops']:>2d} / "
                f"{row['total_steps']}"
            )
        print("=" * 72)

        if not mono_ok or not kappa_one_ok or not finest_ok:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
