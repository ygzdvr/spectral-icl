"""Theorem-C cleanup script: three non-blocking housekeeping items.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §7.3–§7.5 (C3, C4, C5).
Chapter reference: ``thesis/theorem_c.txt`` — Corollary 3.11 (inactive
block convention: ``L_b ≡ 0`` and every ``q_b`` optimal) and
Corollary 3.12 (heterogeneity criterion: ``L★_b > 0`` iff block has
more than one active λ).

Purpose
-------
This single script bundles three small cleanup items to avoid
introducing three separate one-shot files:

Item 1 — heatmap zero-region visual encoding
    Replace the log-scale colorbar-floor rendering of exact zeros
    (``m = 1`` rows / ``κ = 1`` columns) across the C3, C4, C5 heatmaps
    with an explicit mask: light-gray fill with diagonal hatching plus
    a corner annotation ``"≡ 0 (Cor. 3.12)"``. Regenerated from saved
    npz data (no re-computation).

Item 2 — inactive-block edge case (Corollary 3.11)
    Construct a D = 64, m = 8 configuration with block 0 deliberately
    inactive (``ω_i = 0`` for ``i ∈ [0, 8)``). Verify that
    (a) ``L_b(q_b) = 0`` for every ``q_b`` tested, (b) the L = 1
    closed-form optimum treats the inactive block as contributing zero.

Item 3 — κ-monotonicity artifact note
    Load C3's npz, confirm numerically that the κ-monotonicity
    violations observed at ``m = 2`` do NOT occur at ``m ≥ 4``, and
    emit a LaTeX-ready footnote explaining the mass-preserving
    linear-ξ parameterization artifact.

Output
------
``outputs/thesis/theoremC/cleanup/<run_id>/``:

- ``figures/cleaned_c3_obstruction_heatmap.png`` (from C3 npz)
- ``figures/cleaned_c3_obstruction_heatmap_from_patch.png``
  (from c3_patch npz, if present)
- ``figures/cleaned_c4_phase_diagram_main.png`` (3 panels)
- ``figures/cleaned_c5_ladder_heatmap.png``
- ``cleanup_summary.txt``
- ``cleanup_nonmonotonicity_note.txt`` (LaTeX-ready)

Canonical run directory is
``outputs/thesis/theoremC/cleanup/<run_id>/`` — the ``ThesisRunDir``
stem is pinned to ``cleanup`` via a synthetic script path.

Run
---
::

    python -u scripts/thesis/theoremC/run_theoremC_cleanup.py --no-show
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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
from scripts.thesis.utils.partitions import BlockPartition
from scripts.thesis.utils.plotting import (
    PALETTE_PHASE,
    apply_thesis_style,
    save_both,
    sequential_colors,
)
from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanupConfig:
    D: int = 64
    dtype: str = "float64"

    # Item-2 inactive-block test.
    item2_m: int = 8
    item2_kappa: float = 3.0
    item2_inactive_block_idx: int = 0
    item2_q_probes: tuple[float, ...] = (-10.0, 0.0, 1.0, 100.0)
    item2_zero_tol: float = 1e-12


# ---------------------------------------------------------------------------
# Canonical-run locators
# ---------------------------------------------------------------------------


def _latest_run(project_root: Path, script_stem: str) -> Path | None:
    root = (
        project_root / "outputs" / "thesis" / "theoremC" / script_stem
    )
    if not root.is_dir():
        return None
    runs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    return runs[-1] if runs else None


# ---------------------------------------------------------------------------
# ITEM 1 — heatmap zero-region masking helper
# ---------------------------------------------------------------------------


def _log_cell_edges(coords: np.ndarray) -> np.ndarray:
    """Return length-``n+1`` cell-edge array for log-scaled centers.

    Edges sit halfway between neighbouring centers in log-space, and the
    outer edges are extrapolated symmetrically. Matches the convention
    used for log-scaled pcolormesh so hatched overlays align exactly.
    """
    log_c = np.log10(np.asarray(coords, dtype=float))
    mids = 0.5 * (log_c[:-1] + log_c[1:])
    left = log_c[0] - 0.5 * (log_c[1] - log_c[0])
    right = log_c[-1] + 0.5 * (log_c[-1] - log_c[-2])
    return 10.0 ** np.concatenate([[left], mids, [right]])


def _hatch_masked_region(
    ax,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    mask: np.ndarray,
    *,
    hatch_color: str = "lightgray",
    hatch_pattern: str = "///",
    edge_color: str = "gray",
) -> None:
    """Overlay hatched rectangles at every ``True`` cell of ``mask``.

    ``x_edges`` (length ``n_x+1``) and ``y_edges`` (length ``n_y+1``) are
    the shared pcolormesh cell-edge arrays; the heatmap is shape
    ``(n_y, n_x)``. Using the same edges for the mesh and the overlay
    guarantees the hatched cells sit flush against their unmasked
    neighbours.

    The default style matches the thesis convention: light-gray fill
    with diagonal hatch, thin gray border.
    """
    from matplotlib.patches import Rectangle

    n_y, n_x = mask.shape
    for j in range(n_y):
        for i in range(n_x):
            if not bool(mask[j, i]):
                continue
            x0, x1 = x_edges[i], x_edges[i + 1]
            y0, y1 = y_edges[j], y_edges[j + 1]
            rect = Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor=hatch_color,
                edgecolor=edge_color,
                hatch=hatch_pattern,
                linewidth=0.4,
                zorder=5,
            )
            ax.add_patch(rect)


def _hatch_legend_proxy(ax) -> None:
    """Add an invisible line to the legend describing the hatching."""
    from matplotlib.patches import Patch

    proxy = Patch(
        facecolor="lightgray",
        edgecolor="gray",
        hatch="///",
        label="≡ 0 (Cor.)",
    )
    handles = [proxy]
    ax.legend(
        handles=handles, loc="lower left", fontsize=7.5,
        frameon=True, framealpha=0.9, handleheight=1.2,
    )


def _plot_masked_heatmap(
    ax,
    values: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    mask: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    title: str = "",
    floor: float = 1e-18,
) -> None:
    """Core helper: draw a log-scale pcolormesh and overlay hatched cells
    where ``mask`` is ``True``. The masked cells' values are replaced
    with ``NaN`` on the colorbar-side to avoid miscolouring.
    """
    from matplotlib.colors import LogNorm

    vals = np.where(mask, np.nan, values)
    vals_display = np.where(
        np.isnan(vals), np.nan,
        np.where(vals > floor, vals, floor),
    )
    # Compute log color range from unmasked non-floor cells.
    unmasked = vals_display[~np.isnan(vals_display)]
    if unmasked.size == 0:
        vmin, vmax = floor, 10 * floor
    else:
        vmin = max(float(np.nanmin(unmasked)), floor)
        vmax = float(np.nanmax(unmasked))
        if vmax <= vmin:
            vmax = vmin * 10
    cmap = PALETTE_PHASE  # 'mako' per plotting.py
    x_edges = _log_cell_edges(x_coords)
    y_edges = _log_cell_edges(y_coords)
    mesh = ax.pcolormesh(
        x_edges, y_edges, vals_display,
        cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="flat",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _hatch_masked_region(ax, x_edges, y_edges, mask)
    ax.set_xlim(float(min(x_edges[0], x_edges[-1])),
                float(max(x_edges[0], x_edges[-1])))
    ax.set_ylim(float(min(y_edges[0], y_edges[-1])),
                float(max(y_edges[0], y_edges[-1])))
    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(cbar_label)


def _mask_m_or_kappa_zero(
    y_coords: np.ndarray, x_coords: np.ndarray
) -> np.ndarray:
    """Return an ``(n_y, n_x)`` boolean mask of cells where the theorem
    predicts exact zero: ``y == 1`` (singleton blocks) or ``x == 1``
    (homogeneous κ).
    """
    mask = np.zeros((y_coords.size, x_coords.size), dtype=bool)
    mask |= (np.isclose(y_coords, 1.0).reshape(-1, 1))
    mask |= (np.isclose(x_coords, 1.0).reshape(1, -1))
    return mask


def _mask_c6_hybrid_zero(
    m_coords: np.ndarray, k_coords: np.ndarray
) -> np.ndarray:
    """C6 L_hybrid mask: κ=1 column (Cor 3.12) AND m=2 row (refined
    partition at m = 2 IS the singleton, so L_hybrid ≡ L_unconstrained = 0
    in the matched regime). This is the panel where the zero-region
    hatching most clearly applies in the three-way figure."""
    mask = _mask_m_or_kappa_zero(m_coords, k_coords)
    mask |= np.isclose(m_coords, 2.0).reshape(-1, 1)
    return mask


def _plot_masked_heatmap_linear(
    ax,
    values: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    mask: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Linear-colorscale variant of ``_plot_masked_heatmap`` for bounded
    ratios (e.g. C6's captured fraction ``F ∈ [0, 1]``). Uses the same
    log-scaled cell-edge grid and hatched-mask overlay as the log variant
    so that the two styles align visually in a shared three-panel figure.
    """
    vals = np.where(mask, np.nan, values)
    cmap = PALETTE_PHASE
    x_edges = _log_cell_edges(x_coords)
    y_edges = _log_cell_edges(y_coords)
    mesh = ax.pcolormesh(
        x_edges, y_edges, vals,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="flat",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _hatch_masked_region(ax, x_edges, y_edges, mask)
    ax.set_xlim(float(min(x_edges[0], x_edges[-1])),
                float(max(x_edges[0], x_edges[-1])))
    ax.set_ylim(float(min(y_edges[0], y_edges[-1])),
                float(max(y_edges[0], y_edges[-1])))
    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(cbar_label)


# ---------------------------------------------------------------------------
# ITEM 1 — per-source regenerators
# ---------------------------------------------------------------------------


def _regen_c3_heatmap(
    run_dir: ThesisRunDir,
    npz_path: Path,
    output_stem: str,
    title_suffix: str = "",
) -> Path | None:
    """Regenerate the C3 obstruction heatmap with zero-region masking."""
    import matplotlib.pyplot as plt

    if not npz_path.is_file():
        return None
    d = np.load(npz_path)
    m_arr = np.asarray(d["partition_m_list"]).astype(float)
    k_arr = np.asarray(d["kappa_list"]).astype(float)
    L = np.asarray(d["L_cf_grid"]).astype(float)  # (|m|, |κ|)

    mask = _mask_m_or_kappa_zero(m_arr, k_arr)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    _plot_masked_heatmap(
        ax, L, x_coords=k_arr, y_coords=m_arr, mask=mask,
        xlabel=r"within-block heterogeneity $\kappa$",
        ylabel=r"block size $m$",
        cbar_label=r"$L^\star_{L=1}(m, \kappa)$",
        title=(
            "Cleaned C3 obstruction heatmap" + title_suffix +
            " — exact-zero region hatched"
        ),
    )
    _hatch_legend_proxy(ax)
    fig.tight_layout()
    save_both(fig, run_dir, output_stem)
    plt.close(fig)
    return run_dir.png(output_stem)


def _overlay_log_contours(
    ax: Any, x_coords: np.ndarray, y_coords: np.ndarray, values: np.ndarray,
    *, mask: np.ndarray | None = None,
) -> None:
    """White log-spaced contour lines (matching the canonical C4 plotting
    style). Cells in ``mask`` are set to NaN so contours skip them.
    """
    vals = values.astype(float).copy()
    if mask is not None:
        vals[mask] = np.nan
    positive = vals[np.isfinite(vals) & (vals > 0)]
    if positive.size == 0:
        return
    vmin = max(float(positive.min()), 1e-12)
    vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        return
    exps = np.arange(
        int(np.floor(np.log10(vmin))),
        int(np.ceil(np.log10(vmax))) + 1,
    )
    levels = tuple(float(10.0 ** e) for e in exps)
    X, Y = np.meshgrid(x_coords, y_coords)
    cs = ax.contour(
        X, Y, vals, levels=levels, colors="white", linewidths=0.8, alpha=0.75,
    )
    ax.clabel(cs, cs.levels, fontsize=6, inline=True, fmt="%.0e")


def _regen_c4_main(
    run_dir: ThesisRunDir, npz_path: Path
) -> Path | None:
    """Regenerate the 3-panel C4 phase diagram with zero-region masking.
    Uses the L = 4 slice (C4's default L_primary). Two variants are saved:
    the plain masked heatmap (``cleaned_c4_phase_diagram_main``) and the
    same heatmap with white log-spaced contour overlays
    (``cleaned_c4_phase_diagram_main_with_lines``), matching the canonical
    C4 plotting style.
    """
    import matplotlib.pyplot as plt

    if not npz_path.is_file():
        return None
    d = np.load(npz_path)
    m_arr = np.asarray(d["m_list"]).astype(float)
    k_arr = np.asarray(d["kappa_list"]).astype(float)
    L_list = list(np.asarray(d["L_list"]).astype(int).tolist())
    # C4's canonical primary is L=4.
    target = 4 if 4 in L_list else L_list[0]
    i_L = L_list.index(target)
    L_coarse = np.asarray(d["L_coarse"])[:, :, i_L].astype(float)
    L_fine = np.asarray(d["L_fine"])[:, :, i_L].astype(float)
    gap = np.asarray(d["gap"])[:, :, i_L].astype(float)

    mask = _mask_m_or_kappa_zero(m_arr, k_arr)
    panels = [
        (
            L_coarse,
            r"$L^\star$ (coarse class)",
            r"(a) spectral-only optimum $L_{\mathrm{coarse}}(m, \kappa)$",
        ),
        (
            L_fine,
            r"$L^\star$ (dyadic finer class)",
            r"(b) oracle refined optimum $L_{\mathrm{fine}}(m, \kappa)$",
        ),
        (
            gap,
            r"refinement gain",
            r"(c) theorem-C refinement gain "
            r"$\mathrm{gap} = L_{\mathrm{coarse}} - L_{\mathrm{fine}}$",
        ),
    ]

    def _build_fig():
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
        for ax, (data, cbar_label, title) in zip(axes, panels):
            _plot_masked_heatmap(
                ax, data, x_coords=k_arr, y_coords=m_arr, mask=mask,
                xlabel=r"within-block heterogeneity $\kappa$",
                ylabel=r"block size $m$",
                cbar_label=cbar_label,
            )
            ax.set_title(title)
            _hatch_legend_proxy(ax)
        return fig, axes

    # Plain version.
    fig, _ = _build_fig()
    fig.tight_layout()
    save_both(fig, run_dir, "cleaned_c4_phase_diagram_main")
    plt.close(fig)

    # Version with white contour overlays.
    fig, axes = _build_fig()
    for ax, (data, _, _) in zip(axes, panels):
        _overlay_log_contours(ax, k_arr, m_arr, data, mask=mask)
    fig.tight_layout()
    save_both(fig, run_dir, "cleaned_c4_phase_diagram_main_with_lines")
    plt.close(fig)

    return run_dir.png("cleaned_c4_phase_diagram_main")


def _regen_c5_ladder(
    run_dir: ThesisRunDir, npz_path: Path
) -> Path | None:
    """Regenerate the C5 ladder heatmap with zero-region masking. C5 uses
    L_primary = 1 by default."""
    import matplotlib.pyplot as plt

    if not npz_path.is_file():
        return None
    d = np.load(npz_path)
    level_sizes = np.asarray(d["level_sizes"]).astype(float)  # (n_levels,)
    k_arr = np.asarray(d["kappa_list"]).astype(float)
    L_list = list(np.asarray(d["L_list"]).astype(int).tolist())
    target = 1 if 1 in L_list else L_list[0]
    i_L = L_list.index(target)
    loss_slice = np.asarray(d["loss_grid"])[:, :, i_L].astype(float)

    # Mask: κ = 1 column is exact zero; also the finest level (m = 1)
    # row is the singleton oracle, L★ ≡ 0 by Cor 3.12.
    mask = _mask_m_or_kappa_zero(level_sizes, k_arr)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _plot_masked_heatmap(
        ax, loss_slice,
        x_coords=k_arr, y_coords=level_sizes, mask=mask,
        xlabel=r"within-block heterogeneity $\kappa$",
        ylabel=r"ladder level block size $m$ "
               r"(coarsest $D$ to finest $1$)",
        cbar_label=r"$L^\star(j, \kappa)$",
        title=(
            f"Cleaned C5 ladder heatmap (L = {target}) — "
            "exact-zero region hatched"
        ),
    )
    _hatch_legend_proxy(ax)
    fig.tight_layout()
    save_both(fig, run_dir, "cleaned_c5_ladder_heatmap")
    plt.close(fig)
    return run_dir.png("cleaned_c5_ladder_heatmap")


def _regen_c6_three_way_heatmap(
    run_dir: ThesisRunDir, npz_path: Path
) -> Path | None:
    """Regenerate C6's three-way heatmap with theorem-level zero regions
    hatched. Panel (a) L_coarse: κ=1 column (Cor 3.12 — homogeneous block).
    Panel (b) L_hybrid (the middle panel, where the zero-region convention
    most clearly applies): κ=1 column AND m=2 row — at m=2 the refined
    partition IS the singleton, so L_hybrid ≡ L_unconstrained = 0 in the
    matched regime. Panel (c) F: κ=1 column, where F is a 0/0 NaN.

    Two variants are saved, matching the C4 convention: a plain version
    with no contour overlays (``cleaned_c6_three_way_heatmap``) and a
    ``_with_lines`` version with white log-spaced contours on panels (a)
    and (b) and linear F-contours on panel (c).
    """
    import matplotlib.pyplot as plt

    if not npz_path.is_file():
        return None
    d = np.load(npz_path)
    m_arr = np.asarray(d["m_list"]).astype(float)
    k_arr = np.asarray(d["kappa_list"]).astype(float)
    L_list = list(np.asarray(d["L_list"]).astype(int).tolist())
    target = 4 if 4 in L_list else L_list[0]
    i_L = L_list.index(target)
    L_coarse = np.asarray(d["L_coarse"])[:, :, i_L].astype(float)
    L_hybrid = np.asarray(d["L_hybrid"])[:, :, i_L].astype(float)
    F = np.asarray(d["captured_fraction"])[:, :, i_L].astype(float)

    mask_coarse = _mask_m_or_kappa_zero(m_arr, k_arr)
    mask_hybrid = _mask_c6_hybrid_zero(m_arr, k_arr)
    mask_F = np.isnan(F) | mask_coarse

    def _build_fig():
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
        _plot_masked_heatmap(
            axes[0], L_coarse,
            x_coords=k_arr, y_coords=m_arr, mask=mask_coarse,
            xlabel=r"within-block heterogeneity $\kappa$",
            ylabel=r"coarse block size $m$",
            cbar_label=r"$L_{\mathrm{coarse}}$",
        )
        axes[0].set_title(
            r"(a) spectral-only coarse-class optimum $L_{\mathrm{coarse}}$",
            fontsize=10,
        )
        _hatch_legend_proxy(axes[0])

        _plot_masked_heatmap(
            axes[1], L_hybrid,
            x_coords=k_arr, y_coords=m_arr, mask=mask_hybrid,
            xlabel=r"within-block heterogeneity $\kappa$",
            ylabel=r"coarse block size $m$",
            cbar_label=(
                r"$L_{\mathrm{hybrid}}$ "
                r"(refined class $C(\mathfrak{B}_{m/2})$)"
            ),
        )
        axes[1].set_title(
            r"(b) oracle hybrid $L_{\mathrm{hybrid}}$, refined commutant",
            fontsize=10,
        )
        _hatch_legend_proxy(axes[1])

        _plot_masked_heatmap_linear(
            axes[2], F,
            x_coords=k_arr, y_coords=m_arr, mask=mask_F,
            xlabel=r"within-block heterogeneity $\kappa$",
            ylabel=r"coarse block size $m$",
            cbar_label=(
                r"$F = (L_{\mathrm{coarse}} - L_{\mathrm{hybrid}}) / "
                r"L_{\mathrm{coarse}}$"
            ),
            vmin=0.0, vmax=1.0,
        )
        axes[2].set_title("(c) captured fraction $F$", fontsize=10)
        _hatch_legend_proxy(axes[2])
        return fig, axes

    fig, _ = _build_fig()
    fig.tight_layout()
    save_both(fig, run_dir, "cleaned_c6_three_way_heatmap")
    plt.close(fig)

    fig, axes = _build_fig()
    _overlay_log_contours(axes[0], k_arr, m_arr, L_coarse, mask=mask_coarse)
    _overlay_log_contours(axes[1], k_arr, m_arr, L_hybrid, mask=mask_hybrid)
    F_for_contour = np.where(mask_F, np.nan, F)
    F_finite = F_for_contour[np.isfinite(F_for_contour)]
    if F_finite.size > 0:
        # Data-adaptive quantile levels: F saturates near 1 in the matched
        # regime, so fixed levels like 0.25 / 0.5 / 0.75 fall below the
        # observed minimum and draw nothing.
        f_levels = tuple(np.unique(np.quantile(F_finite, (0.2, 0.4, 0.6, 0.8))))
        if len(f_levels) >= 2:
            X, Y = np.meshgrid(k_arr, m_arr)
            cs = axes[2].contour(
                X, Y, F_for_contour, levels=f_levels,
                colors="white", linewidths=0.8, alpha=0.85,
            )
            axes[2].clabel(cs, cs.levels, fontsize=6, inline=True, fmt="%.3f")
    fig.tight_layout()
    save_both(fig, run_dir, "cleaned_c6_three_way_heatmap_with_lines")
    plt.close(fig)

    return run_dir.png("cleaned_c6_three_way_heatmap")


# ---------------------------------------------------------------------------
# ITEM 2 — inactive-block test (Corollary 3.11)
# ---------------------------------------------------------------------------


def _run_item2_inactive_block(
    cfg: CleanupConfig,
) -> dict[str, Any]:
    """Construct D=64, m=8 with block 0 made inactive by zeroing ω[0:8],
    and verify Cor 3.11 conventions numerically.

    Returns a dict with per-probe block losses, total closed-form, and
    the final ``inactive_block_zero_ok`` verdict.
    """
    D = cfg.D
    m = cfg.item2_m
    n_blocks = D // m
    kappa = cfg.item2_kappa
    inactive_idx = int(cfg.item2_inactive_block_idx)

    g2 = G2Config(
        D=D,
        partition_kind="equal",
        partition_params={"m": int(m)},
        block_means_lam=tuple([1.0] * n_blocks),
        block_kappas_lam=tuple([float(kappa)] * n_blocks),
        block_means_omega=tuple([1.0] * n_blocks),
        block_kappas_omega=tuple([float(kappa)] * n_blocks),
        xi_shape="linear",
        spectral_basis_kind="dct2",
        label_norm="sqrt_D",
        sigma=0.0,
        dtype=cfg.dtype,
    )
    op = g2_generate_operator(g2)
    lam = op["Lambda"].detach().cpu().to(torch.float64).clone()
    omega_full = op["Omega"].detach().cpu().to(torch.float64).clone()
    partition: BlockPartition = op["partition"]

    omega_inactive = omega_full.clone()
    inactive_indices = list(partition.blocks[inactive_idx])
    omega_inactive[inactive_indices] = 0.0

    lam_b = lam[inactive_indices]
    om_b = omega_inactive[inactive_indices]

    # Check 1: block-0 loss at multiple q_b probes (L = 1).
    probes = cfg.item2_q_probes
    per_probe = []
    max_probe_loss = 0.0
    for q_b in probes:
        residual = 1.0 - float(q_b) * lam_b
        block_loss = float(
            (om_b * lam_b * residual.pow(2)).sum().item()
        )
        per_probe.append(
            {
                "q_b": float(q_b),
                "block_loss_L1": block_loss,
                "ok": abs(block_loss) <= cfg.item2_zero_tol,
            }
        )
        max_probe_loss = max(max_probe_loss, abs(block_loss))

    probes_ok = all(p["ok"] for p in per_probe)

    # Check 2: closed-form L = 1 optimum with ω[block 0] = 0.
    # per-block a,b,c; block 0 should give a_0 = b_0 = c_0 = 0 and
    # contribute loss 0 under the Cor 3.11 convention.
    n = partition.n_blocks
    per_block_loss_cf = torch.zeros(n, dtype=torch.float64)
    for b_idx, block in enumerate(partition.blocks):
        idx = list(block)
        lb = lam[idx]
        ob = omega_inactive[idx]
        a_b = float((ob * lb).sum().item())
        c_b = float((ob * lb.pow(3)).sum().item())
        if c_b <= 0:
            # Inactive branch: Cor 3.11 convention → loss = 0.
            per_block_loss_cf[b_idx] = 0.0
        else:
            b_b = float((ob * lb.pow(2)).sum().item())
            per_block_loss_cf[b_idx] = a_b - (b_b * b_b) / c_b
    total_cf = float(per_block_loss_cf.sum().item())
    inactive_contribution_cf = float(
        per_block_loss_cf[inactive_idx].item()
    )
    cf_inactive_zero_ok = (
        abs(inactive_contribution_cf) <= cfg.item2_zero_tol
    )

    # Check 3: L-BFGS on the same problem returns the same total loss.
    numeric = oracle_commutant_loss(
        lam, omega_inactive, partition,
        L=1, q_init=None, optimizer="lbfgs", max_iter=500,
    )
    total_num = float(numeric["loss_star"])
    cf_vs_num_abs_err = abs(total_cf - total_num)
    cf_vs_num_rel_err = (
        cf_vs_num_abs_err / max(abs(total_cf), 1e-30)
        if abs(total_cf) > 1e-30 else cf_vs_num_abs_err
    )
    cf_vs_num_ok = cf_vs_num_abs_err <= 1e-8

    inactive_block_zero_ok = (
        probes_ok and cf_inactive_zero_ok and cf_vs_num_ok
    )

    return {
        "D": int(D),
        "m": int(m),
        "n_blocks": int(n),
        "kappa": float(kappa),
        "inactive_block_idx": inactive_idx,
        "inactive_indices": [int(k) for k in inactive_indices],
        "probes": per_probe,
        "max_probe_loss": float(max_probe_loss),
        "probes_ok": bool(probes_ok),
        "cf_total_loss": float(total_cf),
        "cf_per_block_loss": [
            float(x) for x in per_block_loss_cf.tolist()
        ],
        "cf_inactive_contribution": float(inactive_contribution_cf),
        "cf_inactive_zero_ok": bool(cf_inactive_zero_ok),
        "numerical_total_loss": float(total_num),
        "cf_vs_num_abs_err": float(cf_vs_num_abs_err),
        "cf_vs_num_rel_err": float(cf_vs_num_rel_err),
        "cf_vs_num_ok": bool(cf_vs_num_ok),
        "inactive_block_zero_ok": bool(inactive_block_zero_ok),
    }


# ---------------------------------------------------------------------------
# ITEM 3 — κ-monotonicity artifact documentation
# ---------------------------------------------------------------------------


def _run_item3_monotonicity_note(
    project_root: Path,
) -> dict[str, Any]:
    """Verify numerically that C3's κ-monotonicity violations occur only
    at ``m = 2`` and document the artifact for the thesis footnote."""
    c3_run = _latest_run(project_root, "run_theoremC_L1_closed_form")
    if c3_run is None:
        return {
            "c3_npz_found": False,
            "violations": [],
            "m_all_gte_4_monotonic": None,
            "summary": (
                "C3 npz not found; cannot verify κ-monotonicity "
                "claim numerically."
            ),
        }
    npz_path = c3_run / "npz" / "L1_closed_form.npz"
    if not npz_path.is_file():
        return {
            "c3_npz_found": False,
            "violations": [],
            "m_all_gte_4_monotonic": None,
            "summary": f"C3 npz missing at {npz_path}",
        }
    d = np.load(npz_path)
    m_list = list(np.asarray(d["partition_m_list"]).astype(int).tolist())
    k_list = list(np.asarray(d["kappa_list"]).astype(float).tolist())
    L = np.asarray(d["L_cf_grid"]).astype(float)  # (|m|, |κ|)

    # Find all κ-monotonicity violations (L★(κ) decreasing).
    violations: list[dict[str, Any]] = []
    for mi, m in enumerate(m_list):
        for ki in range(1, len(k_list)):
            cur = float(L[mi, ki])
            prev = float(L[mi, ki - 1])
            if cur + 1e-12 < prev:
                violations.append(
                    {
                        "m": int(m),
                        "kappa_prev": float(k_list[ki - 1]),
                        "kappa_curr": float(k_list[ki]),
                        "L_prev": prev,
                        "L_curr": cur,
                        "drop": prev - cur,
                    }
                )
    v_m_values = sorted({v["m"] for v in violations})
    v_only_at_m2 = bool(len(v_m_values) > 0 and v_m_values == [2])

    # Explicitly verify monotonicity at m ≥ 4 across the tested κ range.
    m_ge_4_monotonic = True
    m_ge_4_checked: list[dict[str, Any]] = []
    for mi, m in enumerate(m_list):
        if int(m) < 4:
            continue
        violations_m = 0
        for ki in range(1, len(k_list)):
            if float(L[mi, ki]) + 1e-12 < float(L[mi, ki - 1]):
                violations_m += 1
        if violations_m > 0:
            m_ge_4_monotonic = False
        m_ge_4_checked.append(
            {"m": int(m), "violations": violations_m}
        )

    # m = 2 κ-trajectory for the note.
    if 2 in m_list:
        m2_idx = m_list.index(2)
        m2_traj = [
            {"kappa": float(k), "L_star": float(L[m2_idx, ki])}
            for ki, k in enumerate(k_list)
        ]
    else:
        m2_traj = []

    # All-m trajectories for the table in the note.
    traj_by_m = {
        int(m): [
            {"kappa": float(k), "L_star": float(L[mi, ki])}
            for ki, k in enumerate(k_list)
        ]
        for mi, m in enumerate(m_list)
    }

    return {
        "c3_npz_found": True,
        "c3_npz_path": str(npz_path),
        "violations": violations,
        "violation_m_values": v_m_values,
        "violations_only_at_m2": v_only_at_m2,
        "m_ge_4_checked": m_ge_4_checked,
        "m_all_gte_4_monotonic": m_ge_4_monotonic,
        "m2_trajectory": m2_traj,
        "trajectory_by_m": traj_by_m,
        "m_list": m_list,
        "kappa_list": k_list,
    }


def _write_nonmonotonicity_note(
    run_dir: ThesisRunDir, item3: dict[str, Any]
) -> Path:
    """Emit a LaTeX-ready footnote string at
    ``cleanup_nonmonotonicity_note.txt``."""
    path = run_dir.root / "cleanup_nonmonotonicity_note.txt"
    lines: list[str] = []
    lines.append(
        "% Non-monotonicity-in-κ documentation for the C3 L=1 block "
        "optimum."
    )
    lines.append(
        "% Theorem reference: thesis/theorem_c.txt — Corollary 3.12."
    )
    lines.append(
        "% This note should be inserted in the thesis as a footnote "
        "or remark in the C3 section."
    )
    lines.append(
        "% Auto-generated by scripts/thesis/theoremC/run_theoremC_cleanup.py"
    )
    lines.append("")
    if not item3.get("c3_npz_found"):
        lines.append(
            "% C3 npz not available at runtime; note not generated."
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    only_m2 = item3["violations_only_at_m2"]
    m_ge_4_mono = item3["m_all_gte_4_monotonic"]
    violations = item3["violations"]

    lines.append(
        r"% (a) violations occur only at m = 2: "
        + ("CONFIRMED" if only_m2 else "NOT confirmed")
    )
    lines.append(
        r"% (b) monotonicity holds at every m >= 4: "
        + ("CONFIRMED" if m_ge_4_mono else "NOT confirmed")
    )
    lines.append(
        r"% (c) violating cells (L*_prev -> L*_curr, κ_prev -> κ_curr):"
    )
    for v in violations:
        lines.append(
            f"%   m = {v['m']}  κ: {v['kappa_prev']:.2g} → "
            f"{v['kappa_curr']:.2g}  L*: {v['L_prev']:.4e} → "
            f"{v['L_curr']:.4e}  drop = {v['drop']:.3e}"
        )
    lines.append("")
    lines.append("\\begin{footnote}")
    # Build a concise numeric table fragment for the note.
    m2 = item3["m2_trajectory"]
    if m2:
        kappa_values = ", ".join(
            f"\\kappa = {p['kappa']:.1f}" for p in m2
        )
        lstar_values = "; ".join(
            f"L^\\star({p['kappa']:.1f}) = {p['L_star']:.3e}"
            for p in m2
        )
    else:
        kappa_values = ""
        lstar_values = ""
    note = (
        "The apparent non-monotonicity of $L^\\star(m, \\kappa)$ at "
        "$m = 2$ for $\\kappa \\in \\{5, 10\\}$ is an artifact of the "
        "mass-preserving linear-$\\xi$ parameterization used to "
        "construct within-block spectra: at extreme $\\kappa$ and "
        "very small $m$ the mass-preserving normalization "
        "concentrates spectral weight onto a single mode, so the "
        "per-block $L^\\star_b$ depends on the full weighted spectrum "
        "$\\{\\omega_i, \\lambda_i\\}_{i \\in B_b}$ rather than on "
        "$\\kappa_b$ alone. Corollary 3.12 of Theorem~C only asserts "
        "that $L^\\star_b > 0$ iff the block is heterogeneous; the "
        "corollary makes no monotonicity claim in $\\kappa$. At "
        "$m \\ge 4$ the weight distribution is sufficiently "
        "spread that the measured $L^\\star$ is monotone in $\\kappa$ "
        "across the tested range. For reference the $m = 2$ "
        "trajectory is "
        f"{lstar_values}."
    )
    lines.append(note)
    lines.append("\\end{footnote}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_cleanup_summary(
    run_dir: ThesisRunDir,
    cfg: CleanupConfig,
    item1: dict[str, Any],
    item2: dict[str, Any],
    item3: dict[str, Any],
) -> Path:
    path = run_dir.root / "cleanup_summary.txt"
    lines: list[str] = []
    lines.append(
        "Theorem-C cleanup — three non-blocking items"
    )
    lines.append("=" * 72)
    lines.append(
        "Plan ref: EXPERIMENT_PLAN_FINAL.MD §7.3 (C3), §7.4 (C4), "
        "§7.5 (C5)"
    )
    lines.append(
        "Theorem ref: thesis/theorem_c.txt — Cor 3.11 (inactive "
        "block convention), Cor 3.12 (heterogeneity criterion)."
    )
    lines.append("")

    def _mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    # --- Item 1 ---
    lines.append("Item 1 — heatmap zero-region visual encoding")
    lines.append(
        "  Regenerated heatmaps with light-gray diagonal-hatched "
        "overlay on (m = 1) rows and (κ = 1) columns, plus a "
        "legend proxy '≡ 0 (Cor. 3.12)'."
    )
    for name, fig_path in item1["regenerated"].items():
        status = "written" if fig_path is not None else "SKIPPED (npz missing)"
        lines.append(f"    {name}: {status}  {fig_path or ''}")
    lines.append("")

    # --- Item 2 ---
    lines.append(
        "Item 2 — inactive-block Corollary 3.11 edge case"
    )
    lines.append(
        f"  Setup: D = {item2['D']}, m = {item2['m']} "
        f"(n_blocks = {item2['n_blocks']}), κ = {item2['kappa']:.1f}; "
        f"block {item2['inactive_block_idx']} made inactive by "
        f"ω[{item2['inactive_indices'][0]}:{item2['inactive_indices'][-1]+1}] = 0."
    )
    lines.append(
        f"  Probe test (L = 1, inactive block's L_b(q_b) for several q_b):"
    )
    for p in item2["probes"]:
        lines.append(
            f"    q_b = {p['q_b']:>+7.2f}  L_b = {p['block_loss_L1']:.3e}  "
            f"{_mark(p['ok'])}"
        )
    lines.append(
        f"    max |L_b| = {item2['max_probe_loss']:.3e}  "
        f"tol = {cfg.item2_zero_tol:.0e}  {_mark(item2['probes_ok'])}"
    )
    lines.append(
        f"  Closed-form L = 1 optimum contribution of inactive block = "
        f"{item2['cf_inactive_contribution']:.3e}  "
        f"{_mark(item2['cf_inactive_zero_ok'])}"
    )
    lines.append(
        f"  Numerical (L-BFGS) total = {item2['numerical_total_loss']:.4e}; "
        f"closed-form total = {item2['cf_total_loss']:.4e}; "
        f"abs err = {item2['cf_vs_num_abs_err']:.3e}  "
        f"{_mark(item2['cf_vs_num_ok'])}"
    )
    lines.append(
        f"  → inactive_block_zero_ok: "
        f"{_mark(item2['inactive_block_zero_ok'])}"
    )
    lines.append("")

    # --- Item 3 ---
    lines.append(
        "Item 3 — κ-monotonicity parameterization artifact"
    )
    if not item3.get("c3_npz_found"):
        lines.append(
            "  C3 npz not available; artifact-verification skipped."
        )
        lines.append(f"  {item3.get('summary', '')}")
    else:
        lines.append(
            f"  C3 npz: {item3['c3_npz_path']}"
        )
        lines.append(
            f"  Total violations detected = {len(item3['violations'])}"
        )
        lines.append(
            f"  Violating m values = {item3['violation_m_values']}"
        )
        lines.append(
            f"  (a) violations only at m = 2: "
            f"{_mark(item3['violations_only_at_m2'])}"
        )
        lines.append(
            f"  (b) m >= 4 monotonic in κ: "
            f"{_mark(item3['m_all_gte_4_monotonic'])}"
        )
        for row in item3["m_ge_4_checked"]:
            lines.append(
                f"      m = {row['m']:>2d}  violations = {row['violations']}"
            )
        lines.append(
            "  (c) m = 2 violating cells (κ_prev → κ_curr, L* turnover):"
        )
        for v in item3["violations"]:
            lines.append(
                f"      m = {v['m']}  "
                f"κ: {v['kappa_prev']:.2f} → {v['kappa_curr']:.2f}  "
                f"L*: {v['L_prev']:.4e} → {v['L_curr']:.4e}  "
                f"drop = {v['drop']:.3e}"
            )
        lines.append(
            "  LaTeX-ready note written to "
            "cleanup_nonmonotonicity_note.txt"
        )
    lines.append("")

    top_ok = (
        all(p is not None for p in item1["regenerated"].values())
        and item2["inactive_block_zero_ok"]
        and (
            not item3.get("c3_npz_found")
            or (
                item3["violations_only_at_m2"]
                and item3["m_all_gte_4_monotonic"]
            )
        )
    )
    lines.append("=" * 72)
    lines.append(f"Top-line status: {_mark(top_ok)}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Theorem-C cleanup: three non-blocking items "
            "(heatmap zero-region hatching, Cor 3.11 inactive-block "
            "test, κ-monotonicity artifact note)."
        )
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _cli()
    if args.no_show:
        matplotlib.use("Agg")

    cfg = CleanupConfig()
    project_root = _PROJ

    synthetic_script = Path(__file__).parent / "cleanup.py"
    run = ThesisRunDir(synthetic_script, phase="theoremC")
    print(f"[C-CLEANUP] output: {run.root}")

    with RunContext(
        run,
        config=cfg,
        seeds=[0],
        notes=(
            "Theorem-C cleanup: hatched zero-region heatmaps for "
            "C3, C4, C5; inactive-block Cor 3.11 edge case; "
            "κ-monotonicity artifact note."
        ),
    ) as ctx:
        apply_thesis_style()

        # --------------- Item 1: regenerate heatmaps ---------------
        t0 = time.perf_counter()
        regen: dict[str, Path | None] = {}

        c3_run = _latest_run(project_root, "run_theoremC_L1_closed_form")
        if c3_run is not None:
            regen["cleaned_c3_obstruction_heatmap"] = _regen_c3_heatmap(
                run,
                c3_run / "npz" / "L1_closed_form.npz",
                output_stem="cleaned_c3_obstruction_heatmap",
                title_suffix=" (from L1_closed_form)",
            )
        else:
            regen["cleaned_c3_obstruction_heatmap"] = None

        # Also regenerate from c3_patch npz if present — the patch run
        # saves its own L_cf_grid under a different npz name.
        c3_patch_run = _latest_run(project_root, "c3_patch")
        if c3_patch_run is not None:
            npz_patch = c3_patch_run / "npz" / "c3_patch.npz"
            regen["cleaned_c3_obstruction_heatmap_from_patch"] = (
                _regen_c3_heatmap(
                    run, npz_patch,
                    output_stem="cleaned_c3_obstruction_heatmap_from_patch",
                    title_suffix=" (from c3_patch)",
                )
            )
        else:
            regen["cleaned_c3_obstruction_heatmap_from_patch"] = None

        c4_run = _latest_run(project_root, "run_theoremC_phase_diagram")
        if c4_run is not None:
            regen["cleaned_c4_phase_diagram_main"] = _regen_c4_main(
                run, c4_run / "npz" / "phase_diagram.npz"
            )
        else:
            regen["cleaned_c4_phase_diagram_main"] = None

        c5_run = _latest_run(project_root, "run_theoremC_refinement_monotonicity")
        if c5_run is not None:
            regen["cleaned_c5_ladder_heatmap"] = _regen_c5_ladder(
                run, c5_run / "npz" / "refinement_ladder.npz"
            )
        else:
            regen["cleaned_c5_ladder_heatmap"] = None

        c6_run = _latest_run(project_root, "run_theoremC_oracle_hybrid")
        if c6_run is not None:
            regen["cleaned_c6_three_way_heatmap"] = (
                _regen_c6_three_way_heatmap(
                    run, c6_run / "npz" / "oracle_hybrid.npz"
                )
            )
        else:
            regen["cleaned_c6_three_way_heatmap"] = None

        item1 = {"regenerated": regen}
        dt_1 = time.perf_counter() - t0
        ctx.record_step_time(dt_1)
        print(f"[C-CLEANUP] item 1 done in {dt_1:.2f} s")
        for k, v in regen.items():
            tag = "OK" if v is not None else "SKIPPED"
            print(f"    {k}: {tag}  {v or '(source npz missing)'}")

        # --------------- Item 2: inactive-block test ---------------
        t0 = time.perf_counter()
        item2 = _run_item2_inactive_block(cfg)
        dt_2 = time.perf_counter() - t0
        ctx.record_step_time(dt_2)
        print(
            f"[C-CLEANUP] item 2 done in {dt_2:.2f} s — "
            f"inactive_block_zero_ok = {item2['inactive_block_zero_ok']}"
        )
        for p in item2["probes"]:
            print(
                f"     q_b = {p['q_b']:>+7.2f}  "
                f"L_b = {p['block_loss_L1']:.3e}  ok = {p['ok']}"
            )
        print(
            f"     closed-form inactive contribution = "
            f"{item2['cf_inactive_contribution']:.3e}  "
            f"ok = {item2['cf_inactive_zero_ok']}"
        )
        print(
            f"     numeric total = {item2['numerical_total_loss']:.4e}, "
            f"cf total = {item2['cf_total_loss']:.4e}  "
            f"ok = {item2['cf_vs_num_ok']}"
        )

        # --------------- Item 3: κ-monotonicity note ---------------
        t0 = time.perf_counter()
        item3 = _run_item3_monotonicity_note(project_root)
        note_path = _write_nonmonotonicity_note(run, item3)
        dt_3 = time.perf_counter() - t0
        ctx.record_step_time(dt_3)
        if item3.get("c3_npz_found"):
            print(
                f"[C-CLEANUP] item 3 done in {dt_3:.2f} s — "
                f"violations = {len(item3['violations'])}  "
                f"only at m=2: {item3['violations_only_at_m2']}  "
                f"m>=4 monotone: {item3['m_all_gte_4_monotonic']}"
            )
        else:
            print(
                f"[C-CLEANUP] item 3 skipped (C3 npz missing): "
                f"{item3.get('summary')}"
            )
        print(f"    note written: {note_path}")

        # --------------- Summary ---------------
        summary_path = _write_cleanup_summary(
            run, cfg, item1, item2, item3
        )

        ctx.record_extra("item1_regenerated", {
            k: (str(v) if v is not None else None)
            for k, v in regen.items()
        })
        ctx.record_extra("item2", item2)
        ctx.record_extra(
            "item3",
            {
                "c3_npz_found": item3.get("c3_npz_found"),
                "violations_only_at_m2": item3.get(
                    "violations_only_at_m2"
                ),
                "m_all_gte_4_monotonic": item3.get(
                    "m_all_gte_4_monotonic"
                ),
                "violation_m_values": item3.get("violation_m_values"),
                "violations": item3.get("violations"),
            },
        )

        top_ok = (
            all(v is not None for v in regen.values())
            and item2["inactive_block_zero_ok"]
            and (
                not item3.get("c3_npz_found")
                or (
                    item3["violations_only_at_m2"]
                    and item3["m_all_gte_4_monotonic"]
                )
            )
        )

        ctx.write_summary(
            {
                "plan_reference": (
                    "EXPERIMENT_PLAN_FINAL.MD §7.3 (C3), §7.4 (C4), "
                    "§7.5 (C5)"
                ),
                "theorem_reference": (
                    "thesis/theorem_c.txt — Corollary 3.11 "
                    "(inactive-block convention), Corollary 3.12 "
                    "(heterogeneity criterion)."
                ),
                "category": (
                    "theorem-C cleanup: three non-blocking items "
                    "bundled into one artifact run. No re-training; "
                    "item 1 regenerates figures from saved npz, item "
                    "2 is a small G2 + closed-form calculation, item "
                    "3 is a verification + LaTeX-ready footnote."
                ),
                "interpretation": (
                    "Item 1 fixes the visual ambiguity introduced by "
                    "rendering Cor 3.12-exact zeros at the log "
                    "colorbar floor by masking them explicitly and "
                    "hatching. Item 2 validates the Cor 3.11 "
                    "inactive-block convention numerically. Item 3 "
                    "verifies and documents the κ-monotonicity "
                    "artifact already flagged in C3 (state-dump "
                    "entry) — the thesis note at "
                    "cleanup_nonmonotonicity_note.txt is ready to be "
                    "inserted as a footnote."
                ),
                "item1_regenerated": {
                    k: (str(v) if v is not None else None)
                    for k, v in regen.items()
                },
                "item2": item2,
                "item3": {
                    k: item3.get(k)
                    for k in (
                        "c3_npz_found",
                        "violations_only_at_m2",
                        "m_all_gte_4_monotonic",
                        "violation_m_values",
                    )
                },
                "top_line_ok": bool(top_ok),
                "status": (
                    ("item1_ok" if all(v is not None for v in regen.values())
                     else "item1_partial")
                    + "+"
                    + ("item2_ok" if item2["inactive_block_zero_ok"]
                       else "item2_fail")
                    + "+"
                    + ("item3_ok"
                       if (not item3.get("c3_npz_found")
                           or (item3["violations_only_at_m2"]
                               and item3["m_all_gte_4_monotonic"]))
                       else "item3_fail")
                ),
                "cleanup_summary_path": str(summary_path),
                "cleanup_nonmonotonicity_note_path": str(note_path),
            }
        )

        print()
        print("=" * 72)
        print(" Theorem-C cleanup")
        print(
            f"   item 1 (hatched heatmaps): "
            f"{sum(1 for v in regen.values() if v is not None)} / "
            f"{len(regen)} regenerated"
        )
        print(
            f"   item 2 (inactive-block test): "
            f"{'OK' if item2['inactive_block_zero_ok'] else 'FAIL'}"
        )
        print(
            f"   item 3 (κ-monotonicity note): "
            f"{'OK' if item3.get('c3_npz_found') and item3['violations_only_at_m2'] and item3['m_all_gte_4_monotonic'] else 'n/a or FAIL'}"
        )
        print(f"   summary: {summary_path}")
        print(f"   note:    {note_path}")
        print("=" * 72)

        return 0 if top_ok else 1


if __name__ == "__main__":
    sys.exit(main())
