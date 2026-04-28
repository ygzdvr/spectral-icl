"""Experiment C4: theorem-C heterogeneity phase diagram (headline figure).

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §7.4.

This is the **main headline figure of the thesis experiments**. It is a
theorem-level map of where hybrid refinement matters, drawn purely at the
operator-level exact-optimization layer — no trained networks, no learned
basis adaptation. Architecture comparisons belong to the architecture-
aligned tier later.

Purpose
-------
Sweep the band partition's block size ``m`` and the within-block
heterogeneity parameter ``κ`` on a 2D grid, holding block-average scale
fixed through the mass-preserving band-RRS generator
(:func:`data_generators.g2_generate_operator`). At each ``(m, κ)``
(and secondary depth slice ``L``) produce three quantities:

(a) ``L_coarse(m, κ, L)`` — the **spectral-only optimum** in the
    coarse block-commutant class ``C(B_m)``. Obtained by
    :func:`metrics.oracle_commutant_loss` on the equal-``m`` partition.
(b) ``L_fine(m, κ, L)`` — the **oracle refined optimum** in the
    dyadically-finer block-commutant class ``C(B_{m/2})``. Same
    ``(λ, ω)``; different partition class. (At ``m = 1`` no finer
    class exists, so we set ``L_fine = 0`` trivially; the fully
    unconstrained per-mode optimum is identically zero in the matched
    regime at any depth, since per-mode ``q_k = L/λ_k`` gives
    ``(1 − L⁻¹λ_k q_k)^(2L) = 0``.)
(c) ``gap(m, κ, L) = L_coarse − L_fine`` — the **theorem-C refinement
    gain** at one dyadic step.

Plus a dedicated line-plot figure that makes the ``κ`` dependence
visually explicit (§7.4 binding: "It is appropriate to add contour lines
or slices at fixed m or fixed κ").

Secondary axis: depth ``L``. C3 covered only ``L = 1`` with a
closed-form comparison; C4 extends the sweep to several ``L`` values
so the phase diagram reveals how depth interacts with heterogeneity.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``G2Config``, ``g2_generate_operator`` (mass-preserving band-RRS).
- :mod:`scripts.thesis.utils.partitions`:
    ``equal_blocks`` (to build the dyadically-finer partition).
- :mod:`scripts.thesis.utils.metrics`:
    ``oracle_commutant_loss`` (L-BFGS over block scalars at any L).
- :mod:`scripts.thesis.utils.plotting`, :mod:`run_metadata`: standard.

Primary outputs
---------------
**Headline 3-panel heatmap** (``c4_phase_diagram_main``) at depth
``L_primary``:

- Panel (a): ``L_coarse(m, κ)`` with overlaid contour lines.
- Panel (b): ``L_fine(m, κ)`` (one-step dyadic refinement).
- Panel (c): refinement gain ``gap(m, κ)`` with overlaid contours —
  the hybrid-refinement map.

**κ-dependence figure** (``c4_kappa_slices``): two line panels, one for
``L_coarse`` and one for the refinement gain, plotting vs ``κ`` with
one line per ``m`` at ``L = L_primary``. Makes the heterogeneity axis
directly legible.

**Depth-interaction figure** (``c4_depth_interaction``): three
``(m, L)`` heatmaps of the refinement gain at three representative
``κ`` values (shallow, moderate, strong heterogeneity), showing how
depth modulates the gain as the block structure tightens.

**Diagnostic** (``c4_full_oracle_sanity``): full-oracle gap
``L_coarse − L_full_oracle`` where ``L_full_oracle = 0`` by the matched
per-mode argument; equivalent to ``L_coarse`` itself, included as a
sanity check that the numerical optimization at singleton partition
also returns zero.

Acceptance
----------
1. **Refinement nonnegativity.** For every ``(m, κ, L)`` with ``m ≥ 2``:
   ``gap ≥ −refinement_nonnegativity_tol``. This is the theorem-C
   refinement-monotonicity corollary at a single dyadic step (the
   full monotonicity ladder is C5).
2. **κ = 1 degeneracy.** At ``κ = 1`` (homogeneous blocks),
   ``L_coarse = 0`` for every ``m`` and ``L`` — within ``kappa_1_tol``.
3. **Full oracle = 0.** The singleton-partition numerical optimum
   (computed as a sanity check) must be ≤ ``full_oracle_tol``
   everywhere.

Run
---
::

    python -u scripts/thesis/theoremC/run_theoremC_phase_diagram.py \\
           --device cuda --dtype float64 --no-show
"""

from __future__ import annotations

import argparse
import json
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

from scripts.thesis.utils.data_generators import G2Config, g2_generate_operator
from scripts.thesis.utils.metrics import oracle_commutant_loss
from scripts.thesis.utils.partitions import BlockPartition, equal_blocks
from scripts.thesis.utils.plotting import (
    PALETTE_PHASE,
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
class C4Config:
    """Frozen configuration for the theorem-C heterogeneity phase diagram.

    Default grid: 6 block sizes × 7 κ values × 5 depths = 210 cells, with
    two L-BFGS optimizations per cell (coarse + dyadic-fine) for a total of
    ~420 optimizations. Each L-BFGS run is cheap (a few dozen parameters),
    so the full sweep completes in roughly 1–3 minutes on 1 CPU.
    """

    D: int = 64
    # Primary axes (§7.4 binding: m and κ).
    m_list: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    kappa_list: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0)

    # Secondary axis: depth L (plan §7.4: "if natural"; C3 covers only L=1).
    L_list: tuple[int, ...] = (1, 2, 4, 8, 16)
    L_primary: int = 4  # depth used for the headline m×κ heatmaps

    # Uniform block-level parameters so κ is the only varying quantity.
    block_mean_lam: float = 1.0
    block_mean_omega: float = 1.0
    kappa_omega_matches_lam: bool = True
    xi_shape: str = "linear"
    spectral_basis_kind: str = "dct2"

    # Depth-interaction figure: κ slices to show.
    depth_interaction_kappas: tuple[float, ...] = (1.2, 2.0, 5.0)

    # Acceptance thresholds.
    refinement_nonnegativity_tol: float = 1e-7
    kappa_1_tol: float = 1e-9
    full_oracle_tol: float = 1e-7

    # L-BFGS.
    optimizer: str = "lbfgs"
    max_iter: int = 500

    # Also compute the full-oracle (singleton) numerical optimum as a
    # sanity check. Adds 1 L-BFGS call per (m, κ, L). It's fast and it's
    # the cleanest check that singleton optimization really hits 0.
    compute_full_oracle: bool = True

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_g2_config(cfg: C4Config, m: int, kappa: float) -> G2Config:
    n_blocks = cfg.D // int(m)
    if cfg.D % int(m) != 0 or n_blocks < 1:
        raise ValueError(
            f"D = {cfg.D} not divisible by m = {m}; got n_blocks = {n_blocks}"
        )
    block_means_lam = tuple([float(cfg.block_mean_lam)] * n_blocks)
    block_kappas_lam = tuple([float(kappa)] * n_blocks)
    block_means_omega = tuple([float(cfg.block_mean_omega)] * n_blocks)
    if cfg.kappa_omega_matches_lam:
        block_kappas_omega = block_kappas_lam
    else:
        block_kappas_omega = tuple([1.0] * n_blocks)
    return G2Config(
        D=cfg.D,
        partition_kind="equal",
        partition_params={"m": int(m)},
        block_means_lam=block_means_lam,
        block_kappas_lam=block_kappas_lam,
        block_means_omega=block_means_omega,
        block_kappas_omega=block_kappas_omega,
        xi_shape=cfg.xi_shape,
        spectral_basis_kind=cfg.spectral_basis_kind,
        label_norm="sqrt_D",
        sigma=0.0,
        dtype=cfg.dtype,
    )


def _optimize_over_partition(
    cfg: C4Config,
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
    L: int,
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


def _run_sweep(cfg: C4Config) -> dict[str, Any]:
    m_list = list(cfg.m_list)
    kappa_list = list(cfg.kappa_list)
    L_list = list(cfg.L_list)

    shape = (len(m_list), len(kappa_list), len(L_list))
    L_coarse = np.zeros(shape)
    L_fine = np.zeros(shape)
    L_full_oracle = np.zeros(shape)
    converged_all = np.ones(shape, dtype=bool)

    # Build a cache of spectra per (m, κ) and the partitions used.
    singleton = equal_blocks(cfg.D, 1)

    n_total = len(m_list) * len(kappa_list) * len(L_list)
    idx = 0
    t_start = time.perf_counter()
    for i_m, m in enumerate(m_list):
        # Dyadically-finer partition (for the oracle-refined optimum).
        m_fine = max(int(m) // 2, 1)
        partition_fine = equal_blocks(cfg.D, m_fine)
        for i_k, kappa in enumerate(kappa_list):
            g2_cfg = _build_g2_config(cfg, int(m), float(kappa))
            op = g2_generate_operator(g2_cfg)
            lam = op["Lambda"]
            omega = op["Omega"]
            partition_coarse = op["partition"]
            for i_L, L_val in enumerate(L_list):
                idx += 1
                t0 = time.perf_counter()
                res_c = _optimize_over_partition(
                    cfg, lam, omega, partition_coarse, int(L_val)
                )
                L_coarse[i_m, i_k, i_L] = float(res_c["loss_star"])
                if int(m) == 1:
                    # Coarse is already the singleton; finer partition
                    # doesn't exist. Set L_fine = L_coarse (gap = 0).
                    L_fine[i_m, i_k, i_L] = float(res_c["loss_star"])
                else:
                    res_f = _optimize_over_partition(
                        cfg, lam, omega, partition_fine, int(L_val)
                    )
                    L_fine[i_m, i_k, i_L] = float(res_f["loss_star"])
                    converged_all[i_m, i_k, i_L] = (
                        converged_all[i_m, i_k, i_L]
                        and bool(res_f["converged"])
                    )
                converged_all[i_m, i_k, i_L] = (
                    converged_all[i_m, i_k, i_L] and bool(res_c["converged"])
                )
                if cfg.compute_full_oracle:
                    res_o = _optimize_over_partition(
                        cfg, lam, omega, singleton, int(L_val)
                    )
                    L_full_oracle[i_m, i_k, i_L] = float(res_o["loss_star"])
                    converged_all[i_m, i_k, i_L] = (
                        converged_all[i_m, i_k, i_L]
                        and bool(res_o["converged"])
                    )
                dt = time.perf_counter() - t0
                print(
                    f"[{idx:>4d}/{n_total}] "
                    f"m = {int(m):>2d}  κ = {float(kappa):>5.2f}  "
                    f"L = {int(L_val):>2d}  "
                    f"L_c = {L_coarse[i_m, i_k, i_L]:.4e}  "
                    f"L_f = {L_fine[i_m, i_k, i_L]:.4e}  "
                    f"gap = {L_coarse[i_m, i_k, i_L] - L_fine[i_m, i_k, i_L]:+.3e}  "
                    f"L_oracle = {L_full_oracle[i_m, i_k, i_L]:.2e}  "
                    f"({dt*1000:.1f} ms)"
                )

    total_wall = time.perf_counter() - t_start
    gap = L_coarse - L_fine
    full_gap = L_coarse - L_full_oracle

    return {
        "m_list": m_list,
        "kappa_list": kappa_list,
        "L_list": L_list,
        "L_coarse": L_coarse,
        "L_fine": L_fine,
        "L_full_oracle": L_full_oracle,
        "gap": gap,
        "full_gap": full_gap,
        "converged_all": converged_all,
        "total_wallclock": total_wall,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _overlay_log_contours(
    ax: Any, x_coords: np.ndarray, y_coords: np.ndarray, values: np.ndarray,
    *, levels: tuple[float, ...] | None = None,
) -> None:
    """Overlay log-spaced contour lines on a heatmap for readability."""
    positive = values[values > 0]
    if positive.size == 0:
        return
    vmin = max(float(positive.min()), 1e-12)
    vmax = float(values.max())
    if vmax <= vmin:
        return
    if levels is None:
        exps = np.arange(
            int(np.floor(np.log10(vmin))),
            int(np.ceil(np.log10(vmax))) + 1,
        )
        levels = tuple(float(10.0 ** e) for e in exps)
    X, Y = np.meshgrid(x_coords, y_coords)
    cs = ax.contour(
        X, Y, values, levels=levels, colors="white", linewidths=0.8, alpha=0.75,
    )
    ax.clabel(cs, cs.levels, fontsize=6, inline=True, fmt="%.0e")


def _plot_phase_diagram_main(
    cfg: C4Config, result: dict[str, Any], run_dir: ThesisRunDir
) -> None:
    """Headline 3-panel heatmap at L = L_primary."""
    import matplotlib.pyplot as plt

    m_arr = np.asarray(result["m_list"], dtype=float)
    k_arr = np.asarray(result["kappa_list"], dtype=float)
    L_list = list(result["L_list"])
    if int(cfg.L_primary) not in L_list:
        return
    i_L = L_list.index(int(cfg.L_primary))
    L_coarse = result["L_coarse"][:, :, i_L]
    L_fine = result["L_fine"][:, :, i_L]
    gap = result["gap"][:, :, i_L]

    floor = 1e-12  # for log-scale plotting of zeros
    lc_plot = np.where(L_coarse > floor, L_coarse, floor)
    lf_plot = np.where(L_fine > floor, L_fine, floor)
    gap_plot = np.where(gap > floor, gap, floor)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, data, title, cbar_label in (
        (
            axes[0], lc_plot,
            r"(a) spectral-only optimum $L_{\mathrm{coarse}}(m, \kappa)$",
            r"$L^\star$ (coarse class $\mathcal{C}(\mathfrak{B}_m)$)",
        ),
        (
            axes[1], lf_plot,
            r"(b) oracle refined optimum $L_{\mathrm{fine}}(m, \kappa)$",
            r"$L^\star$ (dyadic finer class $\mathcal{C}(\mathfrak{B}_{m/2})$)",
        ),
        (
            axes[2], gap_plot,
            r"(c) theorem-C refinement gain "
            r"$\mathrm{gap} = L_{\mathrm{coarse}} - L_{\mathrm{fine}}$",
            r"refinement gain",
        ),
    ):
        _pc, _cb = phase_heatmap(
            ax, data,
            x_coords=k_arr, y_coords=m_arr,
            xlabel=r"within-block heterogeneity $\kappa$",
            ylabel=r"block size $m$",
            cbar_label=cbar_label,
            log_z=True, log_x=True, log_y=True,
        )
        ax.set_title(title)
        _overlay_log_contours(ax, k_arr, m_arr, data)

    fig.tight_layout()
    save_both(fig, run_dir, "c4_phase_diagram_main")
    plt.close(fig)


def _plot_kappa_slices(
    cfg: C4Config, result: dict[str, Any], run_dir: ThesisRunDir
) -> None:
    """Two-panel line figure: L_coarse vs κ and gap vs κ, one line per m,
    at the primary depth."""
    import matplotlib.pyplot as plt

    m_list = list(result["m_list"])
    k_arr = np.asarray(result["kappa_list"], dtype=float)
    L_list = list(result["L_list"])
    if int(cfg.L_primary) not in L_list:
        return
    i_L = L_list.index(int(cfg.L_primary))

    L_coarse = result["L_coarse"][:, :, i_L]
    gap = result["gap"][:, :, i_L]

    m_colors = sequential_colors(len(m_list), palette="mako")
    floor = 1e-18

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for ax, mat, title, ylabel in (
        (
            axes[0], L_coarse,
            r"$L_{\mathrm{coarse}}$ vs $\kappa$",
            r"spectral-only optimum $L^\star_{\mathrm{coarse}}$",
        ),
        (
            axes[1], gap,
            r"refinement gain $\mathrm{gap}$ vs $\kappa$",
            r"$L_{\mathrm{coarse}} - L_{\mathrm{fine}}$",
        ),
    ):
        for color, i in zip(m_colors, range(len(m_list))):
            y = mat[i, :]
            y_plot = np.where(y > floor, y, np.nan)
            ax.plot(
                k_arr, y_plot, color=color, lw=1.5, marker="o", ms=4.0,
                label=f"m = {m_list[i]}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"within-block heterogeneity $\kappa$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    save_both(fig, run_dir, "c4_kappa_slices")
    plt.close(fig)


def _plot_kappa_slices_update(
    cfg: C4Config, result: dict[str, Any], run_dir: ThesisRunDir
) -> None:
    """Variant of the κ-slice figure: the left panel (L_coarse vs κ) omits
    the m = 1 row so the y-scale is not dominated by its floor-level values;
    the right panel (refinement gain) is unchanged."""
    import matplotlib.pyplot as plt

    m_list = list(result["m_list"])
    k_arr = np.asarray(result["kappa_list"], dtype=float)
    L_list = list(result["L_list"])
    if int(cfg.L_primary) not in L_list:
        return
    i_L = L_list.index(int(cfg.L_primary))

    L_coarse = result["L_coarse"][:, :, i_L]
    gap = result["gap"][:, :, i_L]

    m_colors = sequential_colors(len(m_list), palette="mako")
    floor = 1e-18

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax_l = axes[0]
    for color, i, m in zip(m_colors, range(len(m_list)), m_list):
        if int(m) == 1:
            continue
        y = L_coarse[i, :]
        y_plot = np.where(y > floor, y, np.nan)
        ax_l.plot(
            k_arr, y_plot, color=color, lw=1.5, marker="o", ms=4.0,
            label=f"m = {m}",
        )
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel(r"within-block heterogeneity $\kappa$")
    ax_l.set_ylabel(r"spectral-only optimum $L^\star_{\mathrm{coarse}}$")
    ax_l.set_title(
        r"$L_{\mathrm{coarse}}$ vs $\kappa$ (excluding $m=1$)", fontsize=10
    )
    ax_l.legend(fontsize=8, loc="best")

    ax_r = axes[1]
    for color, i, m in zip(m_colors, range(len(m_list)), m_list):
        y = gap[i, :]
        y_plot = np.where(y > floor, y, np.nan)
        ax_r.plot(
            k_arr, y_plot, color=color, lw=1.5, marker="o", ms=4.0,
            label=f"m = {m}",
        )
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel(r"within-block heterogeneity $\kappa$")
    ax_r.set_ylabel(r"$L_{\mathrm{coarse}} - L_{\mathrm{fine}}$")
    ax_r.set_title(r"refinement gain $\mathrm{gap}$ vs $\kappa$", fontsize=10)
    ax_r.legend(fontsize=8, loc="best")

    fig.tight_layout()
    save_both(fig, run_dir, "c4_kappa_slices_update")
    plt.close(fig)


def _plot_depth_interaction(
    cfg: C4Config, result: dict[str, Any], run_dir: ThesisRunDir
) -> None:
    """Depth-interaction figure: refinement gap as (m, L) heatmap at three
    κ values (shallow/moderate/strong heterogeneity)."""
    import matplotlib.pyplot as plt

    m_list = list(result["m_list"])
    kappa_list = list(result["kappa_list"])
    L_list = list(result["L_list"])
    m_arr = np.asarray(m_list, dtype=float)
    L_arr = np.asarray(L_list, dtype=float)

    kappas_to_show: list[float] = []
    for k in cfg.depth_interaction_kappas:
        if float(k) in kappa_list:
            kappas_to_show.append(float(k))
        else:
            # Snap to the closest available κ.
            i_k = int(
                np.argmin([abs(float(kk) - float(k)) for kk in kappa_list])
            )
            kappas_to_show.append(float(kappa_list[i_k]))
    # Deduplicate while preserving order.
    seen: set[float] = set()
    kappas_unique: list[float] = []
    for k in kappas_to_show:
        if k not in seen:
            kappas_unique.append(k)
            seen.add(k)
    if not kappas_unique:
        return

    n_k = len(kappas_unique)
    fig, axes = plt.subplots(1, n_k, figsize=(4.8 * n_k, 4.2))
    if n_k == 1:
        axes = [axes]
    floor = 1e-12
    for ax, k_target in zip(axes, kappas_unique):
        i_k = kappa_list.index(k_target)
        # Shape (m, L).
        mat = result["gap"][:, i_k, :]
        mat_plot = np.where(mat > floor, mat, floor)
        _pc, _cb = phase_heatmap(
            ax, mat_plot,
            x_coords=L_arr, y_coords=m_arr,
            xlabel=r"depth $L$",
            ylabel=r"block size $m$",
            cbar_label=r"refinement gain",
            log_z=True, log_x=True, log_y=True,
        )
        _overlay_log_contours(ax, L_arr, m_arr, mat_plot)
        ax.set_title(rf"$\kappa = {k_target:.2g}$", fontsize=10)

    fig.tight_layout()
    save_both(fig, run_dir, "c4_depth_interaction")
    plt.close(fig)


def _plot_full_oracle_sanity(
    cfg: C4Config, result: dict[str, Any], run_dir: ThesisRunDir
) -> None:
    """Sanity: singleton-partition numerical optimum must be zero. Plot
    max over L of L_full_oracle as an (m, κ) heatmap; expected floor-level.
    """
    import matplotlib.pyplot as plt

    m_arr = np.asarray(result["m_list"], dtype=float)
    k_arr = np.asarray(result["kappa_list"], dtype=float)
    full_oracle = result["L_full_oracle"].max(axis=2)  # worst over L per cell
    floor = 1e-18
    full_plot = np.where(full_oracle > floor, full_oracle, floor)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _pc, _cb = phase_heatmap(
        ax, full_plot,
        x_coords=k_arr, y_coords=m_arr,
        xlabel=r"within-block heterogeneity $\kappa$",
        ylabel=r"coarse block size $m$",
        cbar_label=r"max over $L$ of $L^\star_{\mathrm{full\ oracle}}$",
        log_z=True, log_x=True, log_y=True,
    )
    ax.axhline(float(cfg.full_oracle_tol), color="red", lw=0.8, ls="--")
    fig.tight_layout()
    save_both(fig, run_dir, "c4_full_oracle_sanity")
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
            "Experiment C4: theorem-C heterogeneity phase diagram "
            "(plan §7.4, headline figure)."
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
    p.add_argument("--m-list", type=str, default=None)
    p.add_argument("--kappa-list", type=str, default=None)
    p.add_argument("--L-list", type=str, default=None)
    p.add_argument("--L-primary", type=int, default=None)
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> C4Config:
    base = C4Config()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.D is not None:
        overrides["D"] = int(args.D)
    if args.m_list is not None:
        overrides["m_list"] = _parse_list_ints(args.m_list)
    if args.kappa_list is not None:
        overrides["kappa_list"] = _parse_list_floats(args.kappa_list)
    if args.L_list is not None:
        overrides["L_list"] = _parse_list_ints(args.L_list)
    if args.L_primary is not None:
        overrides["L_primary"] = int(args.L_primary)
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
    print(f"[C4] device = {device}")

    run = ThesisRunDir(__file__, phase="theoremC")
    with RunContext(
        run,
        config=cfg,
        seeds=[0, 1, 2, 3],
        notes=(
            "C4 theorem-C heterogeneity phase diagram. Operator-level "
            "exact optimization only (L-BFGS over block scalars); no "
            "learned architectures. Depth swept as a secondary axis."
        ),
    ) as ctx:
        apply_thesis_style()

        result = _run_sweep(cfg)

        # --- Figures ---
        _plot_phase_diagram_main(cfg, result, run)
        _plot_kappa_slices(cfg, result, run)
        _plot_kappa_slices_update(cfg, result, run)
        _plot_depth_interaction(cfg, result, run)
        if cfg.compute_full_oracle:
            _plot_full_oracle_sanity(cfg, result, run)

        # --- Save npz ---
        npz_payload: dict[str, np.ndarray] = {
            "m_list": np.asarray(result["m_list"], dtype=np.int64),
            "kappa_list": np.asarray(result["kappa_list"], dtype=np.float64),
            "L_list": np.asarray(result["L_list"], dtype=np.int64),
            "L_coarse": result["L_coarse"],
            "L_fine": result["L_fine"],
            "L_full_oracle": result["L_full_oracle"],
            "gap": result["gap"],
            "full_gap": result["full_gap"],
            "converged_all": result["converged_all"],
        }
        np.savez_compressed(run.npz_path("phase_diagram"), **npz_payload)

        # --- Per-cell JSON ---
        rows: list[dict[str, Any]] = []
        for i_m, m in enumerate(result["m_list"]):
            for i_k, kappa in enumerate(result["kappa_list"]):
                for i_L, L in enumerate(result["L_list"]):
                    rows.append(
                        {
                            "m": int(m),
                            "kappa": float(kappa),
                            "L": int(L),
                            "L_coarse": float(result["L_coarse"][i_m, i_k, i_L]),
                            "L_fine": float(result["L_fine"][i_m, i_k, i_L]),
                            "gap": float(result["gap"][i_m, i_k, i_L]),
                            "L_full_oracle": float(
                                result["L_full_oracle"][i_m, i_k, i_L]
                            ),
                            "full_gap": float(
                                result["full_gap"][i_m, i_k, i_L]
                            ),
                            "converged": bool(
                                result["converged_all"][i_m, i_k, i_L]
                            ),
                        }
                    )
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

        # --- Acceptance ---
        # 1. Refinement nonnegativity at m >= 2.
        worst_neg_gap = 0.0
        worst_neg_cell: dict[str, Any] | None = None
        for i_m, m in enumerate(result["m_list"]):
            if int(m) < 2:
                continue
            gap_slab = result["gap"][i_m, :, :]
            min_val = float(gap_slab.min())
            if min_val < worst_neg_gap:
                worst_neg_gap = min_val
                i_k, i_L = np.unravel_index(
                    np.argmin(gap_slab), gap_slab.shape
                )
                worst_neg_cell = {
                    "m": int(m),
                    "kappa": float(result["kappa_list"][i_k]),
                    "L": int(result["L_list"][i_L]),
                    "gap": min_val,
                }
        refinement_ok = (
            worst_neg_gap >= -cfg.refinement_nonnegativity_tol
        )

        # 2. κ = 1 → L_coarse ≡ 0.
        kappa_one_worst = 0.0
        if 1.0 in result["kappa_list"]:
            i_k1 = result["kappa_list"].index(1.0)
            kappa_one_worst = float(
                np.abs(result["L_coarse"][:, i_k1, :]).max()
            )
        kappa_one_ok = kappa_one_worst <= cfg.kappa_1_tol

        # 3. Full oracle ≡ 0 everywhere (singleton optimization).
        full_oracle_worst = (
            float(np.abs(result["L_full_oracle"]).max())
            if cfg.compute_full_oracle else 0.0
        )
        full_oracle_ok = full_oracle_worst <= cfg.full_oracle_tol

        status_parts: list[str] = []
        status_parts.append(
            "refinement_nonneg_ok" if refinement_ok else
            f"refinement_nonneg_violated(worst={worst_neg_gap:.2e})"
        )
        status_parts.append(
            "kappa1_zero_ok" if kappa_one_ok else
            f"kappa1_zero_violated(worst={kappa_one_worst:.2e})"
        )
        status_parts.append(
            "full_oracle_ok" if full_oracle_ok else
            f"full_oracle_violated(worst={full_oracle_worst:.2e})"
        )
        status = "+".join(status_parts)

        ctx.record_compute_proxy(float(result["total_wallclock"]))
        ctx.record_extra("worst_neg_gap", worst_neg_gap)
        ctx.record_extra("worst_neg_cell", worst_neg_cell)
        ctx.record_extra("kappa_one_worst", kappa_one_worst)
        ctx.record_extra("full_oracle_worst", full_oracle_worst)

        # Record a few "phase diagram at a glance" aggregates.
        L_list = result["L_list"]
        if int(cfg.L_primary) in L_list:
            i_Lp = L_list.index(int(cfg.L_primary))
            aggregates = {
                "max_L_coarse_at_L_primary": float(
                    result["L_coarse"][:, :, i_Lp].max()
                ),
                "max_gap_at_L_primary": float(
                    result["gap"][:, :, i_Lp].max()
                ),
                "argmax_cell": {},
            }
            i_m, i_k = np.unravel_index(
                np.argmax(result["gap"][:, :, i_Lp]),
                result["gap"][:, :, i_Lp].shape,
            )
            aggregates["argmax_cell"] = {
                "m": int(result["m_list"][i_m]),
                "kappa": float(result["kappa_list"][i_k]),
                "L": int(cfg.L_primary),
                "gap": float(result["gap"][i_m, i_k, i_Lp]),
            }
        else:
            aggregates = {}
        ctx.record_extra("aggregates_at_L_primary", aggregates)

        ctx.write_summary(
            {
                "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §7.4 (C4)",
                "headline_status": (
                    "C4 is the main headline figure of the thesis "
                    "experiments. The phase diagram is a theorem-level "
                    "map of where hybrid refinement matters. Drawn purely "
                    "at the operator-level exact-optimization layer; no "
                    "learned architectures, no trained networks."
                ),
                "category": (
                    "operator-level exact theorem-C validation — "
                    "mass-preserving band-RRS with (m, κ, L) as axes; "
                    "L-BFGS over block scalars in coarse and "
                    "dyadically-finer commutant classes."
                ),
                "interpretation": (
                    "Panel (a) is the spectral-only obstruction "
                    "L_coarse(m, κ) at fixed depth. Panel (b) is the "
                    "oracle refined optimum L_fine(m, κ) at the "
                    "dyadically-finer partition m/2 (theorem-C oracle "
                    "hybrid = direct optimization over the refined "
                    "commutant class; no learned projector). Panel (c) "
                    "is their difference — the theorem-C refinement "
                    "gain of one dyadic step. The homogeneous slice κ=1 "
                    "lies at zero everywhere (no heterogeneity, no "
                    "obstruction) and the m=1 row is identically zero "
                    "(singleton = fully unconstrained). Depth enters "
                    "only as a secondary axis and shows that the "
                    "refinement gain persists across L; the headline "
                    "2D picture is therefore robust to depth. Any "
                    "architectural comparison of these quantities "
                    "belongs to the later architecture-aligned tier."
                ),
                "device": str(device),
                "D": cfg.D,
                "m_list": list(cfg.m_list),
                "kappa_list": list(cfg.kappa_list),
                "L_list": list(cfg.L_list),
                "L_primary": cfg.L_primary,
                "status": status,
                "refinement_nonnegativity_tol": (
                    cfg.refinement_nonnegativity_tol
                ),
                "kappa_1_tol": cfg.kappa_1_tol,
                "full_oracle_tol": cfg.full_oracle_tol,
                "worst_neg_gap": float(worst_neg_gap),
                "worst_neg_cell": worst_neg_cell,
                "kappa_one_worst": float(kappa_one_worst),
                "full_oracle_worst": float(full_oracle_worst),
                "aggregates_at_L_primary": aggregates,
                "sweep_wallclock_seconds": round(
                    float(result["total_wallclock"]), 3
                ),
            }
        )

        print()
        print("=" * 72)
        print(f" C4 theorem-C heterogeneity phase diagram on {device}")
        print(
            f"   refinement nonnegativity: worst neg gap = "
            f"{worst_neg_gap:.3e}  "
            f"{'OK' if refinement_ok else 'FAIL'}  "
            f"(tol = {-cfg.refinement_nonnegativity_tol:.1e})"
        )
        if worst_neg_cell is not None:
            print(
                f"      (at m = {worst_neg_cell['m']}, "
                f"κ = {worst_neg_cell['kappa']}, "
                f"L = {worst_neg_cell['L']})"
            )
        print(
            f"   κ = 1 → L_coarse ≡ 0: worst = "
            f"{kappa_one_worst:.3e}  "
            f"{'OK' if kappa_one_ok else 'FAIL'}  "
            f"(tol = {cfg.kappa_1_tol:.1e})"
        )
        if cfg.compute_full_oracle:
            print(
                f"   full-oracle ≡ 0: worst = "
                f"{full_oracle_worst:.3e}  "
                f"{'OK' if full_oracle_ok else 'FAIL'}  "
                f"(tol = {cfg.full_oracle_tol:.1e})"
            )
        if aggregates:
            amax = aggregates["argmax_cell"]
            print(
                f"   max refinement gain at L = {cfg.L_primary}: "
                f"{aggregates['max_gap_at_L_primary']:.3e}  "
                f"at (m = {amax['m']}, κ = {amax['kappa']})"
            )
        print("=" * 72)

        if (
            not refinement_ok or not kappa_one_ok
            or (cfg.compute_full_oracle and not full_oracle_ok)
        ):
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
