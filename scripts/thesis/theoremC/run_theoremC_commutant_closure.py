"""Experiments C1 + C2: band-RRS commutant closure and grouped dynamics.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §7.1 (C1) and §7.2 (C2).

Purpose
-------
**C1 — operator-level band-RRS commutant closure.** In the band-RRS regime
(G2 operator-only), the population reduced-Γ recursion

    Γ(t + 1) = Γ(t) + η · E_R[Ω_c · Σ_c² · (I − L⁻¹ · Σ_c · Γ(t))^(2L − 1)]

with ``Σ_c = R_c @ diag(λ) @ R_c^T`` and ``Ω_c = R_c @ diag(ω) @ R_c^T``
(per-context block-Haar rotation ``R_c``) **projects the update into the
block commutant C(B)** associated with the chosen band partition. Starting
from the symmetric initialization ``Γ(0) = 0 ∈ C(B)``, the recursion
therefore preserves block-scalar structure exactly: ``Γ(t) ∈ C(B)`` for all
``t``, i.e., there exists a block-scalar vector ``q ∈ R^{n_blocks}`` with

    Γ(t) = Σ_b  q_b(t) · Π_b.

C1 validates this structurally at machine precision by running the
matrix-level recursion and tracking commutant violation
``||Γ − Π_C(Γ)||_F² / ||Γ||_F²`` at every step. A naive per-F-mode recursion
(one that drops the R-averaging step) is tracked alongside as a diagnostic:
under any within-block heterogeneity ``κ_b > 1``, the per-F-mode path
violates the commutant and the violation grows over time. Contrasting the
two is the cleanest way to show that R-averaging is what enforces the
theorem-C block commutant.

**C2 — grouped scalar dynamics.** The R-averaged per-block update is

    δq_b = η · (1/m_b) · Σ_{j ∈ block b} ω_j · λ_j² · (1 − L⁻¹ · λ_j · q_b)^(2L−1).

C2 runs this grouped scalar ODE directly on a length-``n_blocks`` state
vector ``q(t)``, reconstructs the implied matrix ``Σ_b q_b(t) · Π_b``, and
compares against the matrix-level trajectory from C1. The two must agree
to float eps — this is the analog of B1's per-mode ↔ matrix closure test,
but at the block-scalar level.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``G2Config``, ``g2_generate_operator``.
- :mod:`scripts.thesis.utils.commutants`:
    ``commutant_projection``, ``commutant_violation``,
    ``extract_block_scalars``, ``reconstruct_from_block_scalars``.
- :mod:`scripts.thesis.utils.partitions`:
    ``BlockPartition`` (consumed only via ``g2_generate_operator``'s return).
- :mod:`scripts.thesis.utils.plotting`, :mod:`run_metadata`: standard.

Primary outputs
---------------
**C1 (commutant closure):**

- ``c1_commutant_violation_trajectory`` — commutant violation vs t (log-y)
  for (a) R-averaged matrix recursion (expected: flat at float eps), and
  (b) diagnostic naive per-F-mode recursion (expected: grows from 0 as γ
  evolves, then saturates).
- ``c1_grouped_operator_heatmap`` — terminal ``Γ(T)`` as a D×D heatmap
  for both paths. The R-averaged path shows constant diagonals within
  each block; the naive path shows within-block variation.

**C2 (grouped dynamics):**

- ``c2_grouped_trajectories`` — per-block ``q_b(t)`` trajectories. Two
  overlaid curves per block: the matrix-level block scalars (extracted
  from the R-averaged matrix path) and the grouped-ODE integration. The
  two must overlap to float eps.
- ``c2_matrix_vs_ode_closure`` — max |q_mat − q_ode| across all (t, b)
  plotted as a diagnostic vs t, log-y; expected flat at float eps.

**C1 Monte Carlo Haar consistency diagnostic:**

- ``c1_mc_haar_consistency`` — at a small set of trajectory checkpoints
  t ∈ {0, T/2, T}, compare the algebraic commutant-projected update
  ``Π_C(diag(naive_grad))`` used by the matrix-level path above against a
  Monte Carlo estimate of the raw block-Haar average
  ``E_R[R · diag(naive_grad) · R^T]``. Plot ``||MC − alg||_F / ||alg||_F``
  vs ``N`` (number of Haar samples) on log-log. This closes the gap
  between "the projection enforces the commutant by construction" and
  "the population Haar average is what the projection computes": the
  rotated-covariance recursion's R-average (computed by MC over Haar on
  each block) must converge to the algebraic block-mean as ``N → ∞``.

Acceptance
----------
1. **C1 commutant closure at machine precision.** The R-averaged
   trajectory's commutant violation must stay below ``commutant_tol``
   (default ``1e-12`` — ``float64`` eps-scale, not a percentage).
2. **C1 diagnostic contrast.** The naive per-F-mode trajectory's terminal
   commutant violation must exceed ``commutant_tol`` by at least
   ``naive_contrast_ratio`` (default ``1e8``×). This is not a theorem-C
   test — it verifies that the metric and the contrast between "correct"
   and "naive" dynamics are themselves well-posed, and serves as a hard
   negative control.
3. **C2 matrix-ODE agreement at machine precision.** For every ``(t, b)``
   the extracted block-scalar from the matrix-level trajectory must
   match the grouped-ODE output to within ``c2_tol`` (default
   ``1e-12``). C2 is only interpreted AFTER C1 acceptance passes.
4. **C1 MC-Haar consistency.** At the largest Monte Carlo sample count
   ``N = mc_n_samples_list[-1]``, the relative Frobenius error
   ``||MC − alg||_F / ||alg||_F`` must fall below ``mc_haar_tol`` at
   every checkpoint ``t``. This confirms that the algebraic
   commutant projection used by C1 equals the intended band-Haar
   population average.

Run
---
::

    python -u scripts/thesis/theoremC/run_theoremC_commutant_closure.py \\
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

from scripts.thesis.utils.commutants import (
    commutant_projection,
    commutant_violation,
    extract_block_scalars,
    reconstruct_from_block_scalars,
)
from scripts.thesis.utils.data_generators import G2Config, g2_generate_operator
from scripts.thesis.utils.partitions import BlockPartition
from scripts.thesis.utils.plotting import (
    apply_thesis_style,
    overlay_reference,
    save_both,
    sequential_colors,
)
from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C1C2Config:
    """Frozen configuration for C1 (commutant closure) and C2 (grouped
    dynamics). A single G2 band-RRS operator is evolved at several depths
    L to produce both figures.
    """

    # Ambient operator size + band partition.
    D: int = 64
    partition_kind: str = "equal"
    partition_m: int = 8  # block size; n_blocks = D / m

    # Within-block heterogeneity (plan §7.1 calls for moderate/default).
    # κ_b > 1 means genuine within-block variation — the regime in which
    # per-F-mode (non-R-averaged) dynamics leave the commutant.
    block_means_lam: tuple[float, ...] = (
        1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
    )
    block_kappas_lam: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
    block_means_omega: tuple[float, ...] = (
        1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
    )
    block_kappas_omega: tuple[float, ...] = (
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    )
    xi_shape: str = "linear"
    spectral_basis_kind: str = "dct2"

    # Depth sweep. A single L is enough to produce the C1/C2 figures, but
    # multiple L confirm the closure is depth-uniform.
    L_list: tuple[int, ...] = (1, 2, 4, 8)

    # Recursion horizon and step size. Chosen conservatively relative to
    # the theorem-B-style stability boundary η · ω · λ³ · (2L−1)/L < 2
    # (per B1/B2 analysis). Default defaults give
    # max(ω · λ²) ≲ 2 after mass-preserving block normalization, so
    # η = 5e-3 is well within stability.
    T: int = 5000
    eta: float = 5e-3

    # Acceptance thresholds.
    commutant_tol: float = 1e-12
    c2_tol: float = 1e-12
    naive_contrast_ratio: float = 1e8

    # Figure slice for the grouped-operator heatmap.
    heatmap_L: int = 4

    # Monte Carlo Haar consistency diagnostic.
    # Checkpoints in [0, T] (as fractions) at which to compare the algebraic
    # projection against the MC-averaged block-Haar update.
    mc_checkpoint_fractions: tuple[float, ...] = (0.0, 0.5, 1.0)
    # Sample counts used to trace 1/sqrt(N) convergence. The final entry
    # supplies the acceptance point.
    mc_n_samples_list: tuple[int, ...] = (100, 1000, 10000, 50000)
    # Acceptance tolerance at the largest N. Relative Frobenius error of the
    # MC average against the algebraic projection. At N = 5e4 the expected
    # MC noise floor is roughly 1/sqrt(N) × κ ≲ 5e-3, so 2e-2 leaves
    # ~4× margin over the statistical floor.
    mc_haar_tol: float = 2e-2
    mc_seed: int = 42

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_g2_config(cfg: C1C2Config) -> G2Config:
    n_blocks = cfg.D // cfg.partition_m
    if len(cfg.block_means_lam) != n_blocks:
        raise ValueError(
            f"block_means_lam length {len(cfg.block_means_lam)} does not match "
            f"n_blocks = D // m = {n_blocks}"
        )
    return G2Config(
        D=cfg.D,
        partition_kind=cfg.partition_kind,
        partition_params={"m": cfg.partition_m},
        block_means_lam=cfg.block_means_lam,
        block_kappas_lam=cfg.block_kappas_lam,
        block_means_omega=cfg.block_means_omega,
        block_kappas_omega=cfg.block_kappas_omega,
        xi_shape=cfg.xi_shape,
        spectral_basis_kind=cfg.spectral_basis_kind,
        label_norm="sqrt_D",
        sigma=0.0,
        dtype=cfg.dtype,
    )


def _block_mean_vector(
    v: torch.Tensor, partition: BlockPartition
) -> torch.Tensor:
    """Return the length-D vector whose entries are the block-mean of ``v``
    within each block of ``partition``. Equivalent to
    ``diag(Π_C(diag(v)))`` but avoids the matrix round-trip.
    """
    u = torch.zeros_like(v)
    for block in partition.blocks:
        idx = list(block)
        u[idx] = v[idx].mean()
    return u


# ---------------------------------------------------------------------------
# Recursions
# ---------------------------------------------------------------------------


def _evolve_matrix_r_averaged(
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
    *,
    L: int,
    eta: float,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """**C1 primary path.** Evolve the diagonal γ(t) ∈ R^D with the
    R-averaged theorem-C population-loss update. At each step the naive
    per-F-mode gradient is projected into the block commutant by block-
    averaging, so γ(t) remains block-scalar by construction (C1 claim).

    Returns
    -------
    gamma_traj : (T+1, D) tensor, the diagonal of Γ(t).
    cv_traj    : (T+1,) tensor, commutant violation of diag(γ(t)).
    """
    D = int(lam.shape[0])
    gamma = torch.zeros(D, dtype=torch.float64)
    gamma_traj = torch.zeros(T + 1, D, dtype=torch.float64)
    cv_traj = torch.zeros(T + 1, dtype=torch.float64)
    lam64 = lam.to(torch.float64)
    omega64 = omega.to(torch.float64)
    lam_sq = lam64.pow(2)
    exponent = 2 * int(L) - 1
    for t in range(T):
        residual = 1.0 - lam64 * gamma / int(L)
        naive = omega64 * lam_sq * residual.pow(exponent)  # (D,)
        r_avg = _block_mean_vector(naive, partition)       # (D,)
        gamma = gamma + eta * r_avg
        gamma_traj[t + 1] = gamma
        cv_traj[t + 1] = float(
            commutant_violation(torch.diag(gamma), partition)
        )
    return gamma_traj, cv_traj


def _evolve_matrix_naive_per_mode(
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
    *,
    L: int,
    eta: float,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """**C1 diagnostic negative control.** Same recursion as above but
    *without* R-averaging — each F-mode index updates independently using
    its own ``(ω_k, λ_k)``. Under within-block heterogeneity (κ_b > 1) the
    update is not block-scalar, so γ(t) leaves the commutant.
    """
    D = int(lam.shape[0])
    gamma = torch.zeros(D, dtype=torch.float64)
    gamma_traj = torch.zeros(T + 1, D, dtype=torch.float64)
    cv_traj = torch.zeros(T + 1, dtype=torch.float64)
    lam64 = lam.to(torch.float64)
    omega64 = omega.to(torch.float64)
    lam_sq = lam64.pow(2)
    exponent = 2 * int(L) - 1
    for t in range(T):
        residual = 1.0 - lam64 * gamma / int(L)
        naive = omega64 * lam_sq * residual.pow(exponent)
        gamma = gamma + eta * naive
        gamma_traj[t + 1] = gamma
        cv_traj[t + 1] = float(
            commutant_violation(torch.diag(gamma), partition)
        )
    return gamma_traj, cv_traj


def _evolve_grouped_ode(
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
    *,
    L: int,
    eta: float,
    T: int,
) -> torch.Tensor:
    """**C2 grouped ODE integrator.** Evolve the block-scalar state
    ``q ∈ R^{n_blocks}`` directly via

        δq_b = η · (1/m_b) · Σ_{j ∈ block b} ω_j · λ_j² · (1 − L⁻¹ · λ_j · q_b)^(2L−1).

    Returns a ``(T+1, n_blocks)`` tensor.
    """
    n_blocks = partition.n_blocks
    q = torch.zeros(n_blocks, dtype=torch.float64)
    traj = torch.zeros(T + 1, n_blocks, dtype=torch.float64)
    lam64 = lam.to(torch.float64)
    omega64 = omega.to(torch.float64)
    exponent = 2 * int(L) - 1
    block_idx_lists: list[list[int]] = [
        list(block) for block in partition.blocks
    ]
    for t in range(T):
        delta_q = torch.zeros(n_blocks, dtype=torch.float64)
        for b_idx, idx in enumerate(block_idx_lists):
            lam_b = lam64[idx]
            om_b = omega64[idx]
            res_b = 1.0 - lam_b * q[b_idx] / int(L)
            delta_q[b_idx] = (om_b * lam_b.pow(2) * res_b.pow(exponent)).mean()
        q = q + eta * delta_q
        traj[t + 1] = q
    return traj


# ---------------------------------------------------------------------------
# C1 MC Haar consistency diagnostic
# ---------------------------------------------------------------------------


def _sample_block_haar_many(
    partition: BlockPartition,
    n_samples: int,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample ``n_samples`` block-diagonal random rotations, each block Haar-
    distributed on O(m_b). Uses the QR + Mezzadri-sign construction for true
    Haar measure. Returns a ``(n_samples, D, D)`` tensor.

    Inlined here rather than pulled from ``_sample_block_haar`` in
    ``data_generators.py`` so the C1 diagnostic remains readable and
    self-contained.
    """
    D = partition.D
    R = torch.zeros(n_samples, D, D, dtype=dtype)
    for block in partition.blocks:
        m = len(block)
        idx = torch.tensor(list(block), dtype=torch.long)
        A = torch.randn(n_samples, m, m, generator=generator, dtype=dtype)
        Q_raw, R_upper = torch.linalg.qr(A)
        # Mezzadri sign correction ⇒ true Haar.
        d = torch.sign(torch.diagonal(R_upper, dim1=-2, dim2=-1))
        # torch.sign returns 0 for 0 input. Map 0 -> 1 (probability-zero event
        # for continuous Gaussian A; defensive).
        d = torch.where(d == 0, torch.ones_like(d), d)
        Q = Q_raw * d.unsqueeze(-2)
        # Scatter each Q_i into the (block × block) submatrix of R_i.
        R[:, idx[:, None], idx[None, :]] = Q
    return R


def _mc_haar_average_update(
    u_diag: torch.Tensor,
    partition: BlockPartition,
    n_samples: int,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Monte Carlo estimate of the population block-Haar average

        E_R [R · diag(u_diag) · R^T] .

    Returns a ``(D, D)`` tensor. As ``n_samples → ∞`` this converges (almost
    surely) to the block-scalar matrix whose ``b``-th block is
    ``mean(u_diag|_b) · I_{m_b}`` — exactly the algebraic
    :func:`commutants.commutant_projection` of ``diag(u_diag)``.
    """
    R = _sample_block_haar_many(partition, int(n_samples), generator, dtype)
    R_scaled = R * u_diag.view(1, 1, -1).to(dtype)
    result = torch.matmul(R_scaled, R.transpose(-1, -2))
    return result.mean(dim=0)


def _run_mc_consistency(
    cfg: C1C2Config,
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """At the configured checkpoints (applied to the ``heatmap_L`` trial),
    sweep ``mc_n_samples_list`` and compare the MC-averaged block-Haar
    update against the algebraic commutant projection.
    """
    slice_trials = [t for t in trials if t["L"] == int(cfg.heatmap_L)]
    if not slice_trials:
        return []
    trial = slice_trials[0]
    L = int(trial["L"])

    # Translate fractions → integer step indices in [0, T].
    checkpoint_steps = sorted(
        {max(0, min(cfg.T, int(round(f * cfg.T))))
         for f in cfg.mc_checkpoint_fractions}
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.mc_seed))

    rows: list[dict[str, Any]] = []
    for step in checkpoint_steps:
        gamma_t = trial["gamma_r"][step].to(torch.float64)
        residual = 1.0 - lam.to(torch.float64) * gamma_t / L
        naive_u = (
            omega.to(torch.float64)
            * lam.to(torch.float64).pow(2)
            * residual.pow(2 * L - 1)
        )
        # Algebraic target: block-scalar mean of u per block.
        alg_u = _block_mean_vector(naive_u, partition)
        alg_matrix = torch.diag(alg_u)
        alg_norm = float(alg_matrix.norm().item())

        for N in cfg.mc_n_samples_list:
            t0 = time.perf_counter()
            mc_matrix = _mc_haar_average_update(
                naive_u, partition, int(N), generator
            )
            dt = time.perf_counter() - t0
            err_abs = float((mc_matrix - alg_matrix).norm().item())
            err_rel = err_abs / (alg_norm + 1e-30)
            rows.append(
                {
                    "checkpoint_step": int(step),
                    "checkpoint_fraction": step / max(cfg.T, 1),
                    "N": int(N),
                    "L": L,
                    "abs_err_frobenius": err_abs,
                    "rel_err_frobenius": err_rel,
                    "alg_norm": alg_norm,
                    "seconds": float(dt),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def _run_trial(
    cfg: C1C2Config, lam: torch.Tensor, omega: torch.Tensor,
    partition: BlockPartition, L: int
) -> dict[str, Any]:
    gamma_r, cv_r = _evolve_matrix_r_averaged(
        lam, omega, partition, L=int(L), eta=cfg.eta, T=cfg.T
    )
    gamma_n, cv_n = _evolve_matrix_naive_per_mode(
        lam, omega, partition, L=int(L), eta=cfg.eta, T=cfg.T
    )
    q_ode = _evolve_grouped_ode(
        lam, omega, partition, L=int(L), eta=cfg.eta, T=cfg.T
    )
    # Extract block scalars from the matrix trajectory for the C2 closure
    # test. The matrix path is already block-scalar (C1), so
    # ``extract_block_scalars`` gives the same value as any within-block
    # entry but normalizes for float noise.
    q_mat = torch.zeros_like(q_ode)
    for t in range(cfg.T + 1):
        q_mat[t] = extract_block_scalars(
            torch.diag(gamma_r[t]), partition
        )
    q_abs_err = (q_mat - q_ode).abs()
    q_err_max = float(q_abs_err.max().item())
    q_err_max_per_t = q_abs_err.max(dim=1).values  # (T+1,)

    return {
        "L": int(L),
        "gamma_r": gamma_r,
        "gamma_n": gamma_n,
        "cv_r": cv_r,
        "cv_n": cv_n,
        "q_ode": q_ode,
        "q_mat": q_mat,
        "q_err_max": q_err_max,
        "q_err_max_per_t": q_err_max_per_t,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_c1_commutant_trajectory(
    cfg: C1C2Config,
    trials: list[dict[str, Any]],
    run_dir: ThesisRunDir,
) -> None:
    """C1 primary: commutant violation vs t for both R-averaged and naive
    paths, across depths."""
    import matplotlib.pyplot as plt

    L_colors = sequential_colors(len(cfg.L_list), palette="rocket")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)

    t_axis = np.arange(0, cfg.T + 1, dtype=float)
    # Avoid log(0) by masking zeros.
    floor = cfg.commutant_tol * 1e-3

    for ax, key, title in (
        (axes[0], "cv_r", "C1 R-averaged: commutant violation (expected ≈ float eps)"),
        (axes[1], "cv_n", "C1 diagnostic: naive per-F-mode (expected grows; not theorem-C)"),
    ):
        for color, trial in zip(L_colors, trials):
            cv = trial[key].numpy()
            cv_plot = np.where(cv > floor, cv, floor)
            ax.plot(t_axis[1:], cv_plot[1:], color=color, lw=1.3,
                    label=f"L = {trial['L']}")
        ax.axhline(cfg.commutant_tol, color="red", lw=0.8, ls="--",
                   label=f"tol = {cfg.commutant_tol:.0e}")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("step t")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(
        r"commutant violation $\|\Gamma - \Pi_C(\Gamma)\|_F^2/\|\Gamma\|_F^2$"
    )
    axes[0].legend(fontsize=8, loc="best")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "c1_commutant_violation_trajectory")
    plt.close(fig)


def _plot_c1_grouped_operator_heatmap(
    cfg: C1C2Config,
    trials: list[dict[str, Any]],
    partition: BlockPartition,
    run_dir: ThesisRunDir,
) -> None:
    """C1 primary secondary: terminal Γ(T) heatmap for the ``heatmap_L``
    slice, R-averaged vs naive. Block scalar structure shows as constant
    diagonals within each block."""
    import matplotlib.pyplot as plt

    slice_trials = [t for t in trials if t["L"] == int(cfg.heatmap_L)]
    if not slice_trials:
        return
    trial = slice_trials[0]
    gamma_r_final = trial["gamma_r"][-1].numpy()  # (D,)
    gamma_n_final = trial["gamma_n"][-1].numpy()

    # Build full matrices for display.
    Gamma_r = np.diag(gamma_r_final)
    Gamma_n = np.diag(gamma_n_final)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, M, title in (
        (axes[0], Gamma_r,
         f"C1 R-averaged Γ(T) — block-scalar by construction (L = {cfg.heatmap_L})"),
        (axes[1], Gamma_n,
         f"C1 naive per-F-mode Γ(T) — shows within-block variation"),
    ):
        vmax = float(np.abs(M).max())
        if vmax == 0:
            vmax = 1.0
        im = ax.imshow(
            M, aspect="equal", cmap="rocket_r", vmin=0.0, vmax=vmax,
            interpolation="nearest",
        )
        # Overlay block boundaries.
        for block in partition.blocks:
            if not block:
                continue
            hi = max(block) + 0.5
            lo = min(block) - 0.5
            ax.axvline(hi, color="white", lw=0.6)
            ax.axhline(hi, color="white", lw=0.6)
            ax.axvline(lo, color="white", lw=0.6)
            ax.axhline(lo, color="white", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("index k")
        ax.set_ylabel("index k")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    save_both(fig, run_dir, "c1_grouped_operator_heatmap")
    plt.close(fig)


def _plot_c2_grouped_trajectories(
    cfg: C1C2Config,
    trials: list[dict[str, Any]],
    partition: BlockPartition,
    run_dir: ThesisRunDir,
) -> None:
    """C2 primary: per-block q_b(t) for the ``heatmap_L`` slice with matrix
    extraction and grouped ODE overlaid."""
    import matplotlib.pyplot as plt

    slice_trials = [t for t in trials if t["L"] == int(cfg.heatmap_L)]
    if not slice_trials:
        return
    trial = slice_trials[0]
    q_mat = trial["q_mat"].numpy()  # (T+1, n_blocks)
    q_ode = trial["q_ode"].numpy()

    n_blocks = partition.n_blocks
    colors = sequential_colors(n_blocks, palette="rocket")
    t_axis = np.arange(1, cfg.T + 1, dtype=float)  # skip t=0 for log
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for b in range(n_blocks):
        y_mat = q_mat[1:, b]
        y_ode = q_ode[1:, b]
        y_mat = np.where(y_mat > 0, y_mat, np.nan)
        y_ode = np.where(y_ode > 0, y_ode, np.nan)
        ax.plot(
            t_axis, y_mat, color=colors[b], lw=1.4,
            label=f"b = {b} (matrix)",
        )
        overlay_reference(
            ax, t_axis, y_ode, label=None, style="--",
            color="black", lw=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel(r"block scalar $q_b(t)$")
    ax.set_title(
        f"C2 grouped dynamics (L = {cfg.heatmap_L}): matrix-level "
        r"$q_b$ (solid) vs grouped ODE (dashed black)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    save_both(fig, run_dir, "c2_grouped_trajectories")
    plt.close(fig)


def _plot_c2_matrix_vs_ode_closure(
    cfg: C1C2Config,
    trials: list[dict[str, Any]],
    run_dir: ThesisRunDir,
) -> None:
    """C2 diagnostic: max over blocks of |q_mat − q_ode| vs t. Expected
    flat at float eps for every L."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    L_colors = sequential_colors(len(cfg.L_list), palette="rocket")
    floor = cfg.c2_tol * 1e-3
    t_axis = np.arange(1, cfg.T + 1, dtype=float)
    for color, trial in zip(L_colors, trials):
        errs = trial["q_err_max_per_t"].numpy()[1:]
        errs = np.where(errs > floor, errs, floor)
        ax.plot(t_axis, errs, color=color, lw=1.3,
                label=f"L = {trial['L']}")
    ax.axhline(cfg.c2_tol, color="red", lw=0.8, ls="--",
               label=f"tol = {cfg.c2_tol:.0e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel(r"max over blocks $|q_b^{\mathrm{mat}}(t) - q_b^{\mathrm{ode}}(t)|$")
    ax.set_title(
        "C2 matrix↔ODE closure: max block-scalar error across (t, b)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "c2_matrix_vs_ode_closure")
    plt.close(fig)


def _plot_c1_mc_consistency(
    cfg: C1C2Config,
    mc_rows: list[dict[str, Any]],
    run_dir: ThesisRunDir,
) -> None:
    """Plot relative Frobenius error of the MC-averaged block-Haar update
    against the algebraic commutant projection, vs N, per checkpoint. A
    1/sqrt(N) reference is overlaid.
    """
    import matplotlib.pyplot as plt

    if not mc_rows:
        return
    checkpoints = sorted({row["checkpoint_step"] for row in mc_rows})
    N_values = sorted({row["N"] for row in mc_rows})
    colors = sequential_colors(len(checkpoints), palette="rocket")

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for color, step in zip(colors, checkpoints):
        Ns = []
        errs = []
        for row in mc_rows:
            if row["checkpoint_step"] != step:
                continue
            Ns.append(row["N"])
            errs.append(row["rel_err_frobenius"])
        order = np.argsort(Ns)
        Ns_arr = np.asarray(Ns)[order]
        errs_arr = np.asarray(errs)[order]
        frac = step / max(cfg.T, 1)
        ax.plot(
            Ns_arr, errs_arr, color=color, lw=1.4, marker="o", ms=4.0,
            label=f"t = {step} ({frac:.2f}·T)",
        )
    # 1/sqrt(N) reference through the rightmost datapoint of the first
    # checkpoint for visual scale.
    if N_values:
        N_ref = np.asarray(N_values, dtype=float)
        # Pick a reasonable coefficient (~1/sqrt(N_min)) to anchor visually.
        anchor = next(
            (row for row in mc_rows
             if row["N"] == N_values[0] and row["checkpoint_step"] == checkpoints[0]),
            None,
        )
        if anchor is not None:
            coef = anchor["rel_err_frobenius"] * (N_values[0] ** 0.5)
            ref_y = coef * N_ref ** (-0.5)
            overlay_reference(
                ax, N_ref, ref_y,
                label=r"$N^{-1/2}$ reference",
                style="--", color="black", lw=1.0,
            )
    ax.axhline(cfg.mc_haar_tol, color="red", lw=0.8, ls="--",
               label=f"acceptance tol = {cfg.mc_haar_tol:.1e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Monte Carlo sample count $N$")
    ax.set_ylabel(
        r"relative Frobenius error $\|\mathrm{MC}-\mathrm{alg}\|_F/\|\mathrm{alg}\|_F$"
    )
    ax.set_title(
        f"C1 MC-Haar consistency (L = {cfg.heatmap_L}): algebraic commutant "
        "projection = population block-Haar average",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "c1_mc_haar_consistency")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiments C1 + C2: band-RRS commutant closure and grouped "
            "dynamics (plan §§7.1–7.2)."
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
    p.add_argument("--partition-m", type=int, default=None)
    p.add_argument("--L-list", type=str, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--eta", type=float, default=None)
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> C1C2Config:
    base = C1C2Config()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.D is not None:
        overrides["D"] = int(args.D)
    if args.partition_m is not None:
        overrides["partition_m"] = int(args.partition_m)
    if args.L_list is not None:
        overrides["L_list"] = _parse_list_ints(args.L_list)
    if args.T is not None:
        overrides["T"] = int(args.T)
    if args.eta is not None:
        overrides["eta"] = float(args.eta)
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
    print(f"[C1C2] device = {device}")

    g2_cfg = _build_g2_config(cfg)
    op = g2_generate_operator(g2_cfg)
    lam = op["Lambda"]
    omega = op["Omega"]
    partition: BlockPartition = op["partition"]
    print(
        f"[C1C2] D={cfg.D}  partition=equal×{cfg.partition_m}  "
        f"n_blocks={partition.n_blocks}"
    )

    run = ThesisRunDir(__file__, phase="theoremC")
    with RunContext(
        run,
        config=cfg,
        seeds=[0, 1, 2, 3],
        notes=(
            "C1: operator-level band-RRS commutant closure. "
            "C2: grouped scalar dynamics validated against the matrix path."
        ),
    ) as ctx:
        apply_thesis_style()

        trials: list[dict[str, Any]] = []
        t_sweep_start = time.perf_counter()
        for L in cfg.L_list:
            t0 = time.perf_counter()
            trial = _run_trial(cfg, lam, omega, partition, int(L))
            dt = time.perf_counter() - t0
            ctx.record_step_time(dt)
            cv_r_max = float(trial["cv_r"].max().item())
            cv_n_max = float(trial["cv_n"].max().item())
            print(
                f"[L = {int(L):>2d}]  "
                f"cv_r_max = {cv_r_max:.3e}  "
                f"cv_n_max = {cv_n_max:.3e}  "
                f"q_err_max = {trial['q_err_max']:.3e}  "
                f"({dt:.2f} s)"
            )
            trials.append(trial)
        sweep_wall = time.perf_counter() - t_sweep_start

        # --- MC Haar consistency diagnostic ---
        print("[C1 MC-Haar] running consistency diagnostic ...")
        t_mc_start = time.perf_counter()
        mc_rows = _run_mc_consistency(cfg, lam, omega, partition, trials)
        t_mc = time.perf_counter() - t_mc_start
        for row in mc_rows:
            print(
                f"   t = {row['checkpoint_step']:>5d}  N = {row['N']:>6d}  "
                f"rel_err = {row['rel_err_frobenius']:.3e}  "
                f"abs_err = {row['abs_err_frobenius']:.3e}  "
                f"({row['seconds']:.2f} s)"
            )
        print(f"[C1 MC-Haar] diagnostic total = {t_mc:.2f} s")

        # --- Figures ---
        _plot_c1_commutant_trajectory(cfg, trials, run)
        _plot_c1_grouped_operator_heatmap(cfg, trials, partition, run)
        _plot_c2_grouped_trajectories(cfg, trials, partition, run)
        _plot_c2_matrix_vs_ode_closure(cfg, trials, run)
        _plot_c1_mc_consistency(cfg, mc_rows, run)

        # --- Save npz ---
        npz_payload: dict[str, np.ndarray] = {
            "Lambda": lam.detach().cpu().numpy(),
            "Omega": omega.detach().cpu().numpy(),
            "block_sizes": np.asarray(partition.sizes, dtype=np.int64),
            "L_list": np.asarray(cfg.L_list, dtype=np.int64),
        }
        for trial in trials:
            L = trial["L"]
            npz_payload[f"L{L}__gamma_r_final"] = trial["gamma_r"][-1].numpy()
            npz_payload[f"L{L}__gamma_n_final"] = trial["gamma_n"][-1].numpy()
            npz_payload[f"L{L}__cv_r"] = trial["cv_r"].numpy()
            npz_payload[f"L{L}__cv_n"] = trial["cv_n"].numpy()
            npz_payload[f"L{L}__q_mat"] = trial["q_mat"].numpy()
            npz_payload[f"L{L}__q_ode"] = trial["q_ode"].numpy()
        np.savez_compressed(run.npz_path("commutant_closure"), **npz_payload)

        # --- Per-trial summary JSON ---
        per_trial_rows = []
        for trial in trials:
            L = trial["L"]
            per_trial_rows.append(
                {
                    "L": int(L),
                    "cv_r_max": float(trial["cv_r"].max().item()),
                    "cv_r_final": float(trial["cv_r"][-1].item()),
                    "cv_n_max": float(trial["cv_n"].max().item()),
                    "cv_n_final": float(trial["cv_n"][-1].item()),
                    "q_err_max": float(trial["q_err_max"]),
                }
            )
        (run.root / "per_trial_summary.json").write_text(
            json.dumps(per_trial_rows, indent=2) + "\n", encoding="utf-8"
        )

        # --- Acceptance ---
        cv_r_max_all = max(t["cv_r"].max().item() for t in trials)
        cv_n_final_min = min(t["cv_n"][-1].item() for t in trials)
        q_err_max_all = max(t["q_err_max"] for t in trials)

        c1_ok = cv_r_max_all <= cfg.commutant_tol
        contrast_ok = cv_n_final_min >= cfg.commutant_tol * cfg.naive_contrast_ratio
        c2_ok = q_err_max_all <= cfg.c2_tol

        # MC-Haar consistency: at the largest N, every checkpoint must be
        # below ``mc_haar_tol``.
        N_max = max(cfg.mc_n_samples_list) if cfg.mc_n_samples_list else 0
        mc_rows_at_Nmax = [r for r in mc_rows if r["N"] == N_max]
        mc_worst = (
            max((r["rel_err_frobenius"] for r in mc_rows_at_Nmax),
                default=float("inf"))
        )
        mc_ok = bool(mc_rows_at_Nmax) and mc_worst <= cfg.mc_haar_tol

        ctx.record_compute_proxy(float(sweep_wall + t_mc))
        ctx.record_extra("cv_r_max_all", cv_r_max_all)
        ctx.record_extra("cv_n_final_min", cv_n_final_min)
        ctx.record_extra("q_err_max_all", q_err_max_all)
        ctx.record_extra("per_trial", per_trial_rows)
        ctx.record_extra("mc_consistency_rows", mc_rows)
        ctx.record_extra("mc_worst_at_N_max", mc_worst)

        status_parts: list[str] = []
        status_parts.append(
            "c1_commutant_ok" if c1_ok else
            f"c1_commutant_violated(max={cv_r_max_all:.2e})"
        )
        status_parts.append(
            "c1_contrast_ok" if contrast_ok else
            f"c1_contrast_weak(naive_final_min={cv_n_final_min:.2e})"
        )
        status_parts.append(
            "c2_closure_ok" if c2_ok else
            f"c2_closure_violated(q_err_max={q_err_max_all:.2e})"
        )
        status_parts.append(
            "c1_mc_haar_ok" if mc_ok else
            f"c1_mc_haar_violated(worst_rel_err={mc_worst:.2e})"
        )
        status = "+".join(status_parts)

        ctx.write_summary(
            {
                "plan_reference": (
                    "EXPERIMENT_PLAN_FINAL.MD §7.1 (C1) and §7.2 (C2)"
                ),
                "category": (
                    "operator-level exact theorem-C validation — no learned "
                    "architecture. C1 validates band-RRS commutant closure "
                    "from symmetric initialization; C2 validates the "
                    "grouped scalar dynamics against the matrix path. C2 "
                    "is only interpreted AFTER C1 acceptance passes."
                ),
                "interpretation": (
                    "C1: the R-averaged reduced-Γ recursion starting from "
                    "Γ(0) = 0 preserves the block commutant C(B) exactly "
                    "at machine precision (float64 eps-scale). The naive "
                    "per-F-mode recursion (no R-averaging) is tracked "
                    "alongside as a NEGATIVE CONTROL ONLY — under "
                    "within-block heterogeneity κ_b > 1 it leaves C(B) "
                    "by many orders of magnitude, confirming R-averaging "
                    "is what enforces the theorem-C commutant closure. "
                    "The MC-Haar consistency diagnostic further confirms "
                    "that the algebraic commutant projection used by C1 "
                    "equals the intended band-Haar population average "
                    "(||MC − alg||_F → 0 as N → ∞ at 1/√N rate). "
                    "C2: the grouped scalar ODE on q_b(t) agrees to float "
                    "eps with the block-scalars extracted from the matrix "
                    "path. C2 is interpreted only AFTER C1 acceptance "
                    "passes. The effective reduced dynamics is fully "
                    "captured by a length-n_blocks vector trajectory."
                ),
                "device": str(device),
                "D": cfg.D,
                "partition": {
                    "kind": cfg.partition_kind,
                    "m": cfg.partition_m,
                    "n_blocks": partition.n_blocks,
                    "sizes": list(partition.sizes),
                },
                "n_L": len(cfg.L_list),
                "L_list": list(cfg.L_list),
                "T": cfg.T,
                "eta": cfg.eta,
                "status": status,
                "commutant_tol": cfg.commutant_tol,
                "c2_tol": cfg.c2_tol,
                "naive_contrast_ratio": cfg.naive_contrast_ratio,
                "mc_haar_tol": cfg.mc_haar_tol,
                "mc_n_samples_list": list(cfg.mc_n_samples_list),
                "mc_checkpoint_fractions": list(cfg.mc_checkpoint_fractions),
                "cv_r_max_all": cv_r_max_all,
                "cv_n_final_min": cv_n_final_min,
                "q_err_max_all": q_err_max_all,
                "mc_worst_at_N_max": float(mc_worst),
                "mc_rows": mc_rows,
                "per_trial": per_trial_rows,
                "sweep_wallclock_seconds": round(sweep_wall, 3),
                "mc_diagnostic_wallclock_seconds": round(t_mc, 3),
            }
        )

        print()
        print("=" * 72)
        print(f" C1+C2 commutant closure on {device}")
        print(
            f"   C1 cv_r max across (t, L)   = {cv_r_max_all:.3e}  "
            f"{'OK' if c1_ok else 'FAIL'} "
            f"(tol = {cfg.commutant_tol:.1e})"
        )
        print(
            f"   C1 contrast (min naive cv)  = {cv_n_final_min:.3e}  "
            f"{'OK' if contrast_ok else 'WEAK'} "
            f"(required >= {cfg.commutant_tol * cfg.naive_contrast_ratio:.1e})"
        )
        print(
            f"   C2 q_err max across (t, b)  = {q_err_max_all:.3e}  "
            f"{'OK' if c2_ok else 'FAIL'} "
            f"(tol = {cfg.c2_tol:.1e})"
        )
        print(
            f"   C1 MC-Haar worst rel_err @ N = {N_max} "
            f"= {mc_worst:.3e}  "
            f"{'OK' if mc_ok else 'FAIL'} "
            f"(tol = {cfg.mc_haar_tol:.1e})"
        )
        if c1_ok and c2_ok and mc_ok and contrast_ok:
            print(
                "   Interpretation: band-RRS commutant closure, grouped "
                "dynamics, and MC-Haar consistency confirmed. Proceed to C3."
            )
        else:
            print(
                "   Interpretation: closure test FAILED. Do not interpret "
                "downstream figures until the implementation is corrected."
            )
        print("=" * 72)

        if not c1_ok or not contrast_ok or not c2_ok or not mc_ok:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
