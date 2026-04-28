"""Experiment C7 patch: overlay the **actual** Corollary 3.13 Chebyshev
bound and gate it as a formal acceptance check.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §7.7 (C7).
Chapter reference: ``thesis/theorem_c.txt`` — Corollary 3.13
(condition-number diagnostic: ``ρ_b^♯ = (κ_b − 1)/(κ_b + 1)`` and
block-loss bound ``(Σ_{i ∈ B_b} ω_i λ_i)·(ρ_b^♯)^{2L}``).

Purpose
-------
The canonical C7 script's ``c7_contraction_overlay`` figure uses a
*heuristic* anchored reference

    L★(1) · (ρ★)^{2(L − 1)}

for visual contraction context. This is NOT the Corollary 3.13 bound:

1. The theorem bound is the per-block ``(Σ ω·λ)·ρ_b^{2L}`` summed over
   blocks, *not* an anchor through the observed ``L★(1)`` multiplied by
   a power of ``(L − 1)``.
2. At large ``L`` the heuristic reference can dip *below* the empirical
   optima (single-root polynomial is slower than Chebyshev-optimal),
   producing the misleading observed/reference ratios > 1 documented in
   the state dump.

This patch adds the theorem-correct Cor 3.13 bound — both as a figure
overlay and as a formal acceptance gate — following the pattern of the
C4 strict-gain patch.

Data source
-----------
Loads the canonical C7 ``depth_scaling.npz`` to recover ``loss_grid``
at every ``(m, κ, L)``. The Cor 3.13 bound requires the per-block
eigenvalues ``λ_b`` and teacher weights ``ω_b``, which are NOT saved by
C7; they are re-derived on the fly by calling ``g2_generate_operator``
with the exact same generator config. This is cheap (no optimization —
only constructs the mass-preserving spectrum). A ``--recompute`` flag
is available to re-run the full L-BFGS sweep (~ 24 s) instead.

Convention note (binding)
-------------------------
C7's canonical ``loss_grid`` is the **un-normalized**
``Σ_i ω_i λ_i (1 − λ_i q_{b(i)} / L)^{2L}`` produced by
:func:`metrics.oracle_commutant_loss`, without any ``1/P`` factor
(same convention used by C3's ``L_cf`` and the C3 patch). To be
comparable to ``loss_grid`` the Cor 3.13 bound must be computed in the
**same** convention:

    chebyshev_bound(m, κ, L) = Σ_b (Σ_{i ∈ B_b} ω_i λ_i) · ρ_b^{2L}
    with   ρ_b = (κ_b − 1)/(κ_b + 1),
           κ_b = max_{i ∈ B_b ∩ I_act} λ_i / min_{i ∈ B_b ∩ I_act} λ_i.

The user-specified formula with ``1/P`` is the Theorem-3.8
population-loss convention; we honor C7's un-normalized convention to
make the bound directly stackable on the existing figure. Both the
1/P-normalized and un-normalized bounds are saved in the npz so a
downstream reader can recover either.

Checks
------
1. ``chebyshev_bound_general_L_ok`` (raw, strict_tol=1e-15) — for every
   ``(m > 1, κ > 1, L)`` cell, ``L★_observed ≤ bound + strict_tol``.
2. ``chebyshev_bound_resolved_ok`` — restrict to cells where
   ``L★_observed > optimizer_floor = 1e-9``; on those cells
   ``L★ ≤ bound`` must hold. This is the theorem-level gate (mirrors
   the C4 strict-gain patch precision handling).
3. ``failing_above_precision`` (must be 0) — count of resolved cells
   where the bound is violated.

Figures
-------
- ``c7_contraction_overlay.png`` (corrected):
    - solid colored lines: empirical ``L★(m = m_headline, κ, L)``,
    - dashed colored lines: the **actual Cor 3.13 bound** at the same κ,
    - dotted light-color lines: the original anchored heuristic
      ``L★(1)·(ρ★)^{2(L−1)}`` for comparison.
- ``c7_bound_tightness.png`` (new):
    ratio ``L★_observed / bound`` vs ``L`` per κ at ``m = m_headline``.
    Must be ≤ 1 everywhere (on resolved cells) — shows how the bound
    loosens with depth.

Output
------
Canonical run directory
``outputs/thesis/theoremC/c7_chebyshev_bound_patch/<run_id>/``
(``ThesisRunDir`` stem pinned to ``c7_chebyshev_bound_patch`` via a
synthetic script path). Contents: ``figures/``, ``pdfs/``,
``config.json``, ``metadata.json``, ``summary.txt``,
``c7_chebyshev_bound_patch_summary.txt``, ``npz/c7_chebyshev_bound.npz``.

Run
---
::

    python -u scripts/thesis/theoremC/run_theoremC_c7_chebyshev_bound_patch.py \\
           --no-show
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
from scripts.thesis.utils.partitions import BlockPartition
from scripts.thesis.utils.plotting import (
    apply_thesis_style,
    save_both,
    sequential_colors,
)
from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C7ChebyshevPatchConfig:
    """Config for the patch. Defaults match the canonical C7 run exactly."""

    D: int = 64
    m_list: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    kappa_list: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    L_list: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

    # Generator knobs (must match C7 C7Config).
    block_mean_lam: float = 1.0
    block_mean_omega: float = 1.0
    kappa_omega_matches_lam: bool = True
    xi_shape: str = "linear"
    spectral_basis_kind: str = "dct2"

    # Thresholds.
    strict_tol: float = 1e-15
    optimizer_floor: float = 1e-9

    # Figure headline block size.
    m_headline: int = 8

    dtype: str = "float64"


# ---------------------------------------------------------------------------
# Canonical C7 loader
# ---------------------------------------------------------------------------


_CANONICAL_C7_ROOT = "outputs/thesis/theoremC/run_theoremC_depth_scaling"


def _find_latest_c7_run(project_root: Path) -> Path | None:
    root = project_root / _CANONICAL_C7_ROOT
    if not root.is_dir():
        return None
    runs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    return runs[-1] if runs else None


def _load_c7_npz(run_dir: Path) -> dict[str, np.ndarray]:
    npz_path = run_dir / "npz" / "depth_scaling.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"canonical C7 npz not found at {npz_path}"
        )
    data = np.load(npz_path)
    return {k: data[k] for k in data.keys()}


def _validate_axes(
    cfg: C7ChebyshevPatchConfig, data: dict[str, np.ndarray]
) -> None:
    expected_m = np.asarray(cfg.m_list, dtype=np.int64)
    expected_k = np.asarray(cfg.kappa_list, dtype=np.float64)
    expected_L = np.asarray(cfg.L_list, dtype=np.int64)
    got_m = np.asarray(data["m_list"]).astype(np.int64)
    got_k = np.asarray(data["kappa_list"]).astype(np.float64)
    got_L = np.asarray(data["L_list"]).astype(np.int64)
    if not np.array_equal(expected_m, got_m):
        raise ValueError(
            f"m_list mismatch: got {list(got_m)}, expected "
            f"{list(expected_m)}"
        )
    if got_k.shape != expected_k.shape or not np.allclose(expected_k, got_k):
        raise ValueError(
            f"kappa_list mismatch: got {list(got_k)}, expected "
            f"{list(expected_k)}"
        )
    if not np.array_equal(expected_L, got_L):
        raise ValueError(
            f"L_list mismatch: got {list(got_L)}, expected "
            f"{list(expected_L)}"
        )


# ---------------------------------------------------------------------------
# Chebyshev bound (Corollary 3.13)
# ---------------------------------------------------------------------------


def _build_g2_config(
    cfg: C7ChebyshevPatchConfig, m: int, kappa: float
) -> G2Config:
    n_blocks = cfg.D // int(m)
    block_means_lam = tuple([float(cfg.block_mean_lam)] * n_blocks)
    block_kappas_lam = tuple([float(kappa)] * n_blocks)
    block_means_omega = tuple([float(cfg.block_mean_omega)] * n_blocks)
    block_kappas_omega = (
        block_kappas_lam if cfg.kappa_omega_matches_lam
        else tuple([1.0] * n_blocks)
    )
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


def _per_block_rho_and_mass(
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
    active_tol: float = 1e-30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block ``ρ_b = (κ_b − 1)/(κ_b + 1)`` and block mass
    ``M_b = Σ_{i ∈ B_b} ω_i λ_i``, computed over the block's
    *active support* (``ω_i · λ_i > active_tol``).

    For blocks with ``|active support| ≤ 1`` we set ``ρ_b = 0`` (a
    single active mode can be fit exactly by a single block scalar, so
    the Cor 3.13 bound is zero there).
    """
    lam64 = lam.to(torch.float64)
    om64 = omega.to(torch.float64)
    n = partition.n_blocks
    rho = torch.zeros(n, dtype=torch.float64)
    mass = torch.zeros(n, dtype=torch.float64)
    for b_idx, block in enumerate(partition.blocks):
        idx = list(block)
        lam_b = lam64[idx]
        om_b = om64[idx]
        active = (om_b * lam_b) > active_tol
        n_active = int(active.sum().item())
        mass[b_idx] = float((om_b * lam_b).sum().item())
        if n_active <= 1:
            rho[b_idx] = 0.0
            continue
        lam_active = lam_b[active]
        lam_min = float(lam_active.min().item())
        lam_max = float(lam_active.max().item())
        if lam_min <= 0:
            rho[b_idx] = 0.0
            continue
        kappa_b = lam_max / lam_min
        rho[b_idx] = (kappa_b - 1.0) / (kappa_b + 1.0)
    return rho, mass


def _chebyshev_bound_all_L(
    rho: torch.Tensor, mass: torch.Tensor, L_list: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Cor 3.13 total bound at every ``L``.

    Returns
    -------
    bound_total : float64 array of length ``|L_list|``; un-normalized
        (no ``1/P``), matching the C7 ``loss_grid`` convention. Equals
        ``Σ_b M_b · ρ_b^{2L}``.
    bound_total_over_P : float64 array of length ``|L_list|``; same
        quantity divided by ``P = lam.shape[0]`` for reference (matches
        the Theorem 3.8 population-loss convention).
    """
    rho_np = rho.detach().cpu().numpy()
    mass_np = mass.detach().cpu().numpy()
    P_dim = int(mass_np.shape[0])  # placeholder; real P = D (ambient).
    out = np.zeros(len(L_list), dtype=np.float64)
    for i, L in enumerate(L_list):
        per_block = mass_np * np.power(rho_np, 2.0 * float(L))
        out[i] = float(np.sum(per_block))
    # Note: the true ambient dimension is cfg.D; we pass it in the
    # caller. This helper only returns the un-normalized sum.
    return out, out.copy()  # second slot filled by caller


# ---------------------------------------------------------------------------
# Bound computation across the (m, κ, L) grid
# ---------------------------------------------------------------------------


def _compute_bounds(
    cfg: C7ChebyshevPatchConfig,
) -> dict[str, np.ndarray]:
    """For every ``(m, κ)`` pair in the C7 grid, re-derive the G2
    spectrum and compute the Cor 3.13 bound across all ``L`` values.

    Returns a dict with
    ``bound_grid`` (``|m|, |k|, |L|``) — un-normalized, matches C7,
    ``bound_grid_over_P`` — same but divided by ``cfg.D``,
    ``rho_grid`` (``|m|, |k|, max_n_blocks``) — per-block ρ_b
    (right-padded with NaN where n_blocks < max_n_blocks),
    ``mass_grid`` — per-block M_b (same shape / padding).
    """
    n_m = len(cfg.m_list)
    n_k = len(cfg.kappa_list)
    n_L = len(cfg.L_list)
    max_n_blocks = max(cfg.D // m for m in cfg.m_list)

    bound = np.zeros((n_m, n_k, n_L), dtype=np.float64)
    bound_p = np.zeros_like(bound)
    rho_grid = np.full(
        (n_m, n_k, max_n_blocks), np.nan, dtype=np.float64
    )
    mass_grid = np.full(
        (n_m, n_k, max_n_blocks), np.nan, dtype=np.float64
    )

    for i, m in enumerate(cfg.m_list):
        for j, kappa in enumerate(cfg.kappa_list):
            g2 = _build_g2_config(cfg, int(m), float(kappa))
            op = g2_generate_operator(g2)
            lam = op["Lambda"].detach().cpu()
            omega = op["Omega"].detach().cpu()
            partition: BlockPartition = op["partition"]
            rho, mass = _per_block_rho_and_mass(lam, omega, partition)
            rho_np = rho.detach().cpu().numpy()
            mass_np = mass.detach().cpu().numpy()
            n_b = int(rho_np.shape[0])
            rho_grid[i, j, :n_b] = rho_np
            mass_grid[i, j, :n_b] = mass_np
            for k, L in enumerate(cfg.L_list):
                per_block = mass_np * np.power(rho_np, 2.0 * float(L))
                total = float(np.sum(per_block))
                bound[i, j, k] = total
                bound_p[i, j, k] = total / float(cfg.D)
    return {
        "bound_grid": bound,
        "bound_grid_over_P": bound_p,
        "rho_grid": rho_grid,
        "mass_grid": mass_grid,
    }


# ---------------------------------------------------------------------------
# Anchored heuristic overlay (the one the original C7 used)
# ---------------------------------------------------------------------------


def _anchored_reference(
    L_list: np.ndarray, L_star_at_1: float, kappa: float
) -> np.ndarray:
    """Original C7 anchored reference ``L★(1) · (ρ★)^{2(L−1)}`` (the
    misleading heuristic). Kept here for side-by-side comparison in
    the corrected overlay figure."""
    k = float(kappa)
    if k <= 1.0:
        return np.full_like(np.asarray(L_list, dtype=float), np.nan)
    rho = (k - 1.0) / (k + 1.0)
    L_arr = np.asarray(L_list, dtype=float)
    return float(L_star_at_1) * np.power(rho, 2.0 * (L_arr - 1.0))


# ---------------------------------------------------------------------------
# Acceptance checks
# ---------------------------------------------------------------------------


def _run_checks(
    cfg: C7ChebyshevPatchConfig,
    loss_grid: np.ndarray,
    bound_grid: np.ndarray,
) -> dict[str, Any]:
    """Chebyshev bound checks across (m > 1, κ > 1, L) cells."""
    m_arr = np.asarray(cfg.m_list, dtype=np.int64)
    k_arr = np.asarray(cfg.kappa_list, dtype=np.float64)
    L_arr = np.asarray(cfg.L_list, dtype=np.int64)

    # Build the (m > 1, κ > 1) relevance mask.
    m_rel = m_arr > 1
    k_rel = k_arr > 1.0
    rel_mask = (
        m_rel[:, None, None]
        & k_rel[None, :, None]
        & np.ones_like(L_arr, dtype=bool)[None, None, :]
    )

    slack = bound_grid - loss_grid  # (|m|, |k|, |L|)

    # Raw check: loss ≤ bound + strict_tol on every relevant cell.
    rel_violations = rel_mask & (slack < -cfg.strict_tol)
    raw_count_total = int(rel_mask.sum())
    raw_count_violations = int(rel_violations.sum())
    raw_min_slack = float(slack[rel_mask].min()) if raw_count_total > 0 \
        else float("inf")
    chebyshev_bound_general_L_ok = bool(raw_count_violations == 0)

    # Resolved check: restrict to cells where loss_star > optimizer_floor.
    resolved_mask = rel_mask & (loss_grid > cfg.optimizer_floor)
    resolved_count_total = int(resolved_mask.sum())
    resolved_violations_mask = resolved_mask & (slack < -cfg.strict_tol)
    failing_above_precision = int(resolved_violations_mask.sum())
    if resolved_count_total > 0:
        resolved_min_slack = float(slack[resolved_mask].min())
    else:
        resolved_min_slack = float("inf")
    chebyshev_bound_resolved_ok = bool(failing_above_precision == 0)

    # Cell-location of the worst raw violation (for the summary).
    if raw_count_total > 0:
        idx_flat = int(np.argmin(np.where(rel_mask, slack, np.inf)))
        loc = np.unravel_index(idx_flat, slack.shape)
        worst_raw_loc = {
            "m": int(m_arr[loc[0]]),
            "kappa": float(k_arr[loc[1]]),
            "L": int(L_arr[loc[2]]),
            "loss": float(loss_grid[loc]),
            "bound": float(bound_grid[loc]),
            "slack": float(slack[loc]),
        }
    else:
        worst_raw_loc = {
            "m": None, "kappa": None, "L": None,
            "loss": None, "bound": None, "slack": None,
        }

    # Above-precision failing cells for the breakdown table.
    failing_above_cells = []
    for (i, j, k) in np.argwhere(resolved_violations_mask):
        failing_above_cells.append(
            {
                "m": int(m_arr[i]),
                "kappa": float(k_arr[j]),
                "L": int(L_arr[k]),
                "loss": float(loss_grid[i, j, k]),
                "bound": float(bound_grid[i, j, k]),
                "slack": float(slack[i, j, k]),
                "ratio": float(
                    loss_grid[i, j, k]
                    / max(bound_grid[i, j, k], 1e-300)
                ),
            }
        )

    # Below-precision failing cells (diagnostic only).
    below_precision_mask = rel_mask & ~(loss_grid > cfg.optimizer_floor)
    failing_below_mask = below_precision_mask & (slack < -cfg.strict_tol)
    failing_below_precision = int(failing_below_mask.sum())

    # Max ratio loss / bound on resolved cells (tightness summary).
    if resolved_count_total > 0:
        ratios = np.where(
            bound_grid > 1e-300,
            loss_grid / np.maximum(bound_grid, 1e-300),
            0.0,
        )
        max_resolved_ratio = float(ratios[resolved_mask].max())
    else:
        max_resolved_ratio = 0.0

    return {
        "chebyshev_bound_general_L_ok": bool(chebyshev_bound_general_L_ok),
        "chebyshev_bound_resolved_ok": bool(chebyshev_bound_resolved_ok),
        "raw_count_total": raw_count_total,
        "raw_count_violations": raw_count_violations,
        "raw_min_slack": float(raw_min_slack),
        "worst_raw_loc": worst_raw_loc,
        "resolved_count_total": resolved_count_total,
        "resolved_min_slack": float(resolved_min_slack),
        "max_resolved_ratio": float(max_resolved_ratio),
        "failing_above_precision": int(failing_above_precision),
        "failing_above_cells": failing_above_cells,
        "failing_below_precision": int(failing_below_precision),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_corrected_overlay(
    cfg: C7ChebyshevPatchConfig,
    loss_grid: np.ndarray,
    bound_grid: np.ndarray,
    run_dir: ThesisRunDir,
) -> None:
    """Corrected overlay: empirical L★ (solid), Cor 3.13 bound (dashed),
    original anchored heuristic (dotted light) — all at m_headline.
    """
    import matplotlib.pyplot as plt

    m_list = list(cfg.m_list)
    k_list = list(cfg.kappa_list)
    L_arr = np.asarray(cfg.L_list, dtype=float)

    if int(cfg.m_headline) not in m_list:
        m_idx = m_list.index(min(m_list, key=lambda m: abs(m - cfg.m_headline)))
    else:
        m_idx = m_list.index(int(cfg.m_headline))
    m_used = m_list[m_idx]

    # Drop κ = 1 from the overlay (ρ = 0, bound is 0 and L★ is ~eps).
    plotted_k = [(j, k) for j, k in enumerate(k_list) if float(k) > 1.0]
    colors = sequential_colors(len(plotted_k), palette="mako")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    floor = 1e-18
    for color, (j, kappa) in zip(colors, plotted_k):
        y_emp = loss_grid[m_idx, j, :]
        y_emp_safe = np.where(y_emp > floor, y_emp, np.nan)
        y_bnd = bound_grid[m_idx, j, :]
        y_bnd_safe = np.where(y_bnd > floor, y_bnd, np.nan)
        ax.plot(
            L_arr, y_emp_safe, color=color, lw=1.6, marker="o", ms=4.0,
            label=rf"$\kappa = {kappa:.3g}$ (exact)",
        )
        ax.plot(
            L_arr, y_bnd_safe, color=color, lw=1.1, ls="--",
        )
        # Anchored heuristic for comparison.
        if y_emp[0] > floor:
            y_anchor = _anchored_reference(
                L_arr, float(y_emp[0]), float(kappa)
            )
            y_anchor = np.where(y_anchor > floor, y_anchor, np.nan)
            ax.plot(
                L_arr, y_anchor, color=color, lw=0.9, ls=":",
                alpha=0.55,
            )

    # Legend proxies for the three line styles.
    ax.plot(
        [], [], color="gray", lw=1.1, ls="--",
        label="Cor. bound $(\\Sigma\\ \\omega\\lambda)\\,"
        "\\rho_b^{2L}$",
    )
    ax.plot(
        [], [], color="gray", lw=0.9, ls=":", alpha=0.55,
        label=r"anchored heuristic $L^\star(1)\cdot(\rho^\star)^{2(L-1)}$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"depth $L$")
    ax.set_ylabel(r"block-commutant optimum $L^\star(m, \kappa, L)$")
    ax.legend(fontsize=7.2, loc="best", ncol=2)
    fig.tight_layout()
    save_both(fig, run_dir, "c7_contraction_overlay")
    plt.close(fig)


def _plot_bound_tightness(
    cfg: C7ChebyshevPatchConfig,
    loss_grid: np.ndarray,
    bound_grid: np.ndarray,
    run_dir: ThesisRunDir,
) -> None:
    """Ratio ``L★_obs / bound`` vs ``L`` per κ at m_headline."""
    import matplotlib.pyplot as plt

    m_list = list(cfg.m_list)
    k_list = list(cfg.kappa_list)
    L_arr = np.asarray(cfg.L_list, dtype=float)

    if int(cfg.m_headline) not in m_list:
        m_idx = m_list.index(
            min(m_list, key=lambda m: abs(m - cfg.m_headline))
        )
    else:
        m_idx = m_list.index(int(cfg.m_headline))
    m_used = m_list[m_idx]

    plotted_k = [(j, k) for j, k in enumerate(k_list) if float(k) > 1.0]
    colors = sequential_colors(len(plotted_k), palette="mako")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for color, (j, kappa) in zip(colors, plotted_k):
        y_emp = loss_grid[m_idx, j, :]
        y_bnd = bound_grid[m_idx, j, :]
        # Mask unresolved cells (either loss below optimizer floor).
        resolved = y_emp > cfg.optimizer_floor
        ratio = np.where(
            (y_bnd > 1e-300) & resolved,
            y_emp / np.maximum(y_bnd, 1e-300),
            np.nan,
        )
        ax.plot(
            L_arr, ratio, color=color, lw=1.4, marker="o", ms=4.0,
            label=rf"$\kappa = {kappa:.3g}$",
        )
    ax.axhline(
        1.0, color="red", lw=0.9, ls="--",
        label="bound = exact (tight)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"depth $L$")
    ax.set_ylabel(r"$L^\star_{\mathrm{obs}}\ /\ L^{\mathrm{Cheb}}$")
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "c7_bound_tightness")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_summary(
    run_dir: ThesisRunDir,
    cfg: C7ChebyshevPatchConfig,
    checks: dict[str, Any],
    source: str,
) -> Path:
    path = run_dir.root / "c7_chebyshev_bound_patch_summary.txt"
    lines: list[str] = []
    lines.append(
        "Experiment C7 Chebyshev-bound patch — per-item acceptance "
        "summary"
    )
    lines.append("=" * 72)
    lines.append("Plan ref: EXPERIMENT_PLAN_FINAL.MD §7.7 (C7)")
    lines.append(
        "Theorem ref: thesis/theorem_c.txt — Corollary 3.13 "
        "(condition-number diagnostic ρ_b = (κ_b − 1)/(κ_b + 1) "
        "and block loss bound (Σ ω·λ)·ρ_b^{2L})."
    )
    lines.append(
        f"Config: D = {cfg.D}, m_list = {list(cfg.m_list)}, "
        f"kappa_list = {list(cfg.kappa_list)}, L_list = "
        f"{list(cfg.L_list)}"
    )
    lines.append(f"Data source: {source}")
    lines.append(
        "Convention: un-normalized bound Σ_b (Σ_{i∈B_b} ω·λ)·ρ_b^{2L} "
        "to match C7's un-normalized loss_grid (Σ_i ω·λ·(1-λq/L)^{2L})."
    )
    lines.append("")

    def _mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    # Check 1: raw
    lines.append(
        "Check — Corollary 3.13 bound validity at every (m > 1, κ > 1, L)"
    )
    lines.append(
        f"  Assertion: L★_obs ≤ bound + {cfg.strict_tol:.0e}"
    )
    lines.append(
        f"    n_cells evaluated = {checks['raw_count_total']}"
    )
    lines.append(
        f"    violations        = {checks['raw_count_violations']}"
    )
    lines.append(
        f"    min slack (bound − loss) = "
        f"{checks['raw_min_slack']:.3e}"
    )
    w = checks["worst_raw_loc"]
    if w["m"] is not None:
        lines.append(
            f"    worst cell        = (m = {w['m']}, "
            f"kappa = {w['kappa']:.3g}, L = {w['L']})"
        )
        lines.append(
            f"      loss = {w['loss']:.4e}  "
            f"bound = {w['bound']:.4e}  "
            f"slack = {w['slack']:+.3e}"
        )
    lines.append(
        f"  → chebyshev_bound_general_L_ok: "
        f"{_mark(checks['chebyshev_bound_general_L_ok'])}"
    )
    lines.append("")

    # Resolved diagnostic
    lines.append(
        f"Check — resolved diagnostic (loss_obs > "
        f"{cfg.optimizer_floor:.0e})"
    )
    lines.append(
        f"  n resolved cells = {checks['resolved_count_total']}"
    )
    lines.append(
        f"  failing_above_precision = "
        f"{checks['failing_above_precision']}  (must be 0)"
    )
    lines.append(
        f"  failing_below_precision = "
        f"{checks['failing_below_precision']}  (diagnostic only)"
    )
    lines.append(
        f"  max ratio (loss / bound) on resolved cells = "
        f"{checks['max_resolved_ratio']:.4f}"
    )
    if checks["failing_above_cells"]:
        lines.append(
            "  ABOVE-precision failing cells (theorem-level failures):"
        )
        for c in checks["failing_above_cells"]:
            lines.append(
                f"    m={c['m']:>2d}  κ={c['kappa']:.3g}  "
                f"L={c['L']:>2d}  loss={c['loss']:.3e}  "
                f"bound={c['bound']:.3e}  "
                f"ratio={c['ratio']:.3e}"
            )
    lines.append(
        f"  → chebyshev_bound_resolved_ok: "
        f"{_mark(checks['chebyshev_bound_resolved_ok'])}"
    )
    lines.append("")

    top_raw_ok = checks["chebyshev_bound_general_L_ok"]
    theorem_level_ok = checks["chebyshev_bound_resolved_ok"]
    lines.append("=" * 72)
    lines.append(
        f"Top-line status (raw {cfg.strict_tol:.0e} threshold): "
        f"{_mark(top_raw_ok)}"
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
            "C7 patch: overlay the actual Corollary 3.13 bound and "
            "record it as an acceptance check."
        )
    )
    p.add_argument(
        "--c7-run", type=str, default=None,
        help=(
            "Path to a canonical C7 run directory "
            "(default: most recent "
            "outputs/thesis/theoremC/run_theoremC_depth_scaling/<run_id>)"
        ),
    )
    p.add_argument(
        "--recompute", action="store_true",
        help=(
            "Re-run the full C7 sweep locally (~ 24 s) instead of "
            "loading the canonical npz."
        ),
    )
    p.add_argument(
        "--kappa-list", type=str, default=None,
        help=(
            "Override default kappa_list (comma-separated floats). "
            "Auto-enables --recompute."
        ),
    )
    p.add_argument(
        "--L-list", type=str, default=None,
        help=(
            "Override default L_list (comma-separated ints). "
            "Auto-enables --recompute."
        ),
    )
    p.add_argument(
        "--m-list", type=str, default=None,
        help=(
            "Override default m_list (comma-separated ints). "
            "Auto-enables --recompute."
        ),
    )
    p.add_argument(
        "--m-headline", type=int, default=None,
        help="Override the headline m used for the overlay figure.",
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def _recompute_c7(
    cfg: C7ChebyshevPatchConfig,
) -> dict[str, np.ndarray]:
    """Re-run the full C7 sweep locally. Delegates to C7's runtime."""
    from scripts.thesis.theoremC import (  # noqa: F401
        run_theoremC_depth_scaling as _c7,
    )

    c7_cfg = _c7.C7Config(
        D=cfg.D,
        m_list=cfg.m_list,
        kappa_list=cfg.kappa_list,
        L_list=cfg.L_list,
        block_mean_lam=cfg.block_mean_lam,
        block_mean_omega=cfg.block_mean_omega,
        kappa_omega_matches_lam=cfg.kappa_omega_matches_lam,
        xi_shape=cfg.xi_shape,
        spectral_basis_kind=cfg.spectral_basis_kind,
        dtype=cfg.dtype,
    )
    sweep = _c7._run_sweep(c7_cfg)
    return {
        "m_list": np.asarray(cfg.m_list, dtype=np.int64),
        "kappa_list": np.asarray(cfg.kappa_list, dtype=np.float64),
        "L_list": np.asarray(cfg.L_list, dtype=np.int64),
        "loss_grid": sweep["loss_grid"],
        "converged_grid": sweep["converged_grid"],
    }


def main() -> int:
    args = _cli()
    if args.no_show:
        matplotlib.use("Agg")

    cfg = C7ChebyshevPatchConfig()
    overrides: dict[str, Any] = {}
    if args.kappa_list:
        overrides["kappa_list"] = tuple(
            float(x) for x in args.kappa_list.split(",") if x.strip()
        )
    if args.L_list:
        overrides["L_list"] = tuple(
            int(x) for x in args.L_list.split(",") if x.strip()
        )
    if args.m_list:
        overrides["m_list"] = tuple(
            int(x) for x in args.m_list.split(",") if x.strip()
        )
    if args.m_headline is not None:
        overrides["m_headline"] = int(args.m_headline)
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
        if not args.recompute:
            print(
                "[C7-CHEB] custom axes provided; auto-enabling --recompute"
            )
            args.recompute = True

    synthetic_script = Path(__file__).parent / "c7_chebyshev_bound_patch.py"
    run = ThesisRunDir(synthetic_script, phase="theoremC")
    print(f"[C7-CHEB] output: {run.root}")

    with RunContext(
        run,
        config=cfg,
        seeds=[0],
        notes=(
            "C7 Chebyshev-bound patch: recompute per-block ρ_b, "
            "Mass, and total Cor 3.13 bound; add acceptance gate; "
            "emit corrected overlay + tightness figures."
        ),
    ) as ctx:
        project_root = _PROJ
        source_desc = "unknown"
        if args.recompute:
            print("[C7-CHEB] --recompute: re-running the C7 sweep locally")
            t0 = time.perf_counter()
            data = _recompute_c7(cfg)
            dt = time.perf_counter() - t0
            source_desc = f"recomputed locally ({dt:.1f} s)"
            ctx.record_compute_proxy(float(dt))
            ctx.record_step_time(dt)
        else:
            if args.c7_run:
                c7_run = Path(args.c7_run)
            else:
                latest = _find_latest_c7_run(project_root)
                if latest is None:
                    raise FileNotFoundError(
                        "No canonical C7 run found under "
                        f"{project_root / _CANONICAL_C7_ROOT}. "
                        "Pass --c7-run or use --recompute."
                    )
                c7_run = latest
            print(f"[C7-CHEB] loading canonical C7 run: {c7_run}")
            data = _load_c7_npz(c7_run)
            source_desc = (
                f"canonical C7 npz: "
                f"{c7_run.relative_to(project_root)}"
                "/npz/depth_scaling.npz"
            )
            ctx.record_extra("canonical_c7_run", str(c7_run))

        _validate_axes(cfg, data)
        apply_thesis_style()

        # Compute per-block (ρ_b, M_b) and the full bound grid.
        print("[C7-CHEB] computing per-(m, κ) per-block ρ_b and Cor 3.13 bound ...")
        t0 = time.perf_counter()
        bound_data = _compute_bounds(cfg)
        bound_dt = time.perf_counter() - t0
        print(f"[C7-CHEB] bound computation done ({bound_dt:.2f} s)")

        loss_grid = np.asarray(data["loss_grid"]).astype(np.float64)
        bound_grid = bound_data["bound_grid"]

        checks = _run_checks(cfg, loss_grid, bound_grid)

        # Figures.
        _plot_corrected_overlay(cfg, loss_grid, bound_grid, run)
        _plot_bound_tightness(cfg, loss_grid, bound_grid, run)

        # Npz.
        npz_payload = {
            "m_list": np.asarray(cfg.m_list, dtype=np.int64),
            "kappa_list": np.asarray(cfg.kappa_list, dtype=np.float64),
            "L_list": np.asarray(cfg.L_list, dtype=np.int64),
            "loss_grid": loss_grid,
            "bound_grid": bound_grid,
            "bound_grid_over_P": bound_data["bound_grid_over_P"],
            "rho_grid_per_block": bound_data["rho_grid"],
            "mass_grid_per_block": bound_data["mass_grid"],
        }
        np.savez_compressed(
            run.npz_path("c7_chebyshev_bound"), **npz_payload
        )

        summary_path = _write_summary(run, cfg, checks, source_desc)

        # Store structured checks for the canonical summary.json.
        ctx.record_extra("checks", checks)
        ctx.record_extra("data_source", source_desc)

        theorem_level_ok = checks["chebyshev_bound_resolved_ok"]

        ctx.write_summary(
            {
                "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §7.7 (C7)",
                "theorem_reference": (
                    "thesis/theorem_c.txt — Corollary 3.13 "
                    "(block loss bound (Σ ω·λ)·ρ_b^{2L})."
                ),
                "category": (
                    "acceptance-record + figure-fix patch on existing "
                    "C7 data. Replaces C7's heuristic anchored overlay "
                    "with the actual Cor 3.13 bound both as a figure "
                    "overlay and as a formal acceptance gate."
                ),
                "interpretation": (
                    "C7's original overlay L★(1)·(ρ★)^{2(L−1)} is an "
                    "anchored heuristic — NOT the Cor 3.13 bound — and "
                    "can dip below the empirical L★ at large L. The "
                    "actual Cor 3.13 bound is (Σ_b (Σ_{i∈B_b} ω_i λ_i) "
                    "· ρ_b^{2L}); it is a true upper bound by "
                    "construction (each block uses the condition-"
                    "number contraction factor ρ_b from its own λ "
                    "range). This patch adds it as an overlay (dashed) "
                    "alongside the exact L★ (solid) and the original "
                    "heuristic (dotted). The acceptance gate "
                    "chebyshev_bound_resolved_ok restricts to cells "
                    "where L★ > optimizer_floor = 1e-9, where L-BFGS "
                    "resolves the observed loss; below-precision cells "
                    "are reported separately as diagnostic. "
                    "failing_above_precision must be 0."
                ),
                "convention": (
                    "Un-normalized bound, matching C7's un-normalized "
                    "loss_grid. Formula is un-normalized Σ_b M_b·ρ_b^{2L} "
                    "(no 1/P). The 1/P-normalized variant is also "
                    "saved to the npz as bound_grid_over_P for users "
                    "who need the Theorem-3.8 population-loss scale."
                ),
                "data_source": source_desc,
                "D": cfg.D,
                "m_list": list(cfg.m_list),
                "kappa_list": list(cfg.kappa_list),
                "L_list": list(cfg.L_list),
                "m_headline": cfg.m_headline,
                "strict_tol": cfg.strict_tol,
                "optimizer_floor": cfg.optimizer_floor,
                "chebyshev_bound_general_L_ok": checks[
                    "chebyshev_bound_general_L_ok"
                ],
                "chebyshev_bound_resolved_ok": checks[
                    "chebyshev_bound_resolved_ok"
                ],
                "raw_count_total": checks["raw_count_total"],
                "raw_count_violations": checks["raw_count_violations"],
                "raw_min_slack": checks["raw_min_slack"],
                "worst_raw_loc": checks["worst_raw_loc"],
                "resolved_count_total": checks["resolved_count_total"],
                "resolved_min_slack": checks["resolved_min_slack"],
                "max_resolved_ratio": checks["max_resolved_ratio"],
                "failing_above_precision": checks[
                    "failing_above_precision"
                ],
                "failing_below_precision": checks[
                    "failing_below_precision"
                ],
                "all_ok": bool(theorem_level_ok),
                "status": (
                    "raw_" + ("ok" if checks[
                        "chebyshev_bound_general_L_ok"
                    ] else "fail")
                    + "+resolved_" + ("ok" if checks[
                        "chebyshev_bound_resolved_ok"
                    ] else "fail")
                ),
                "patch_summary_path": str(summary_path),
            }
        )

        print()
        print("=" * 72)
        print(" C7 Chebyshev-bound patch")
        print(
            f"   raw   (every m>1, κ>1, L):                  "
            f"violations = {checks['raw_count_violations']} / "
            f"{checks['raw_count_total']}  "
            f"min slack = {checks['raw_min_slack']:.3e}  "
            f"{'OK' if checks['chebyshev_bound_general_L_ok'] else 'FAIL'}"
        )
        print(
            f"   resolved (loss > {cfg.optimizer_floor:.0e}):"
            f"   failing_above_precision = "
            f"{checks['failing_above_precision']}  "
            f"max ratio loss/bound = "
            f"{checks['max_resolved_ratio']:.3f}  "
            f"{'OK' if checks['chebyshev_bound_resolved_ok'] else 'FAIL'}"
        )
        print(
            f"   failing below precision: "
            f"{checks['failing_below_precision']}  (diagnostic only)"
        )
        print(
            f"   theorem-level status: "
            f"{'OK' if theorem_level_ok else 'FAIL'}"
        )
        print(f"   summary: {summary_path}")
        print("=" * 72)

        return 0 if theorem_level_ok else 1


if __name__ == "__main__":
    sys.exit(main())
