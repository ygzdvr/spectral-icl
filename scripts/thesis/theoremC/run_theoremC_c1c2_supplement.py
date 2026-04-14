"""Experiment C1+C2 supplement: theorem-objects not directly validated by
:mod:`run_theoremC_commutant_closure`.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §7.1–§7.2 (supplements C1+C2).
Chapter reference: Theorem-C chapter draft (``thesis/theorem_c.txt``).

Purpose
-------
The main C1+C2 script validates *commutant closure* of the R-averaged
recursion (Prop. 3.7) and the *grouped ODE* against the matrix block-scalar
path (eq. 3.3 of Theorem 3.8). This supplement fills five remaining gaps:

1. **Direct Lemma 3.5 invariance test.** Verify ``E_L(Q) = E_L(UQU*)`` by
   paired Monte Carlo for ``U`` sampled from the band-rotation group
   ``G_B`` and for ``Q`` both inside and outside the commutant.
2. **Direct grouped-loss formula verification (eq. 3.2).** At 50 log-spaced
   checkpoints along the R-averaged trajectory, evaluate the loss two
   independent ways — an explicit P×P matrix-product computation and the
   block-scalar sum (3.2) — and verify machine-precision agreement.
3. **Induced metric (Theorem 3.8 step 5).** Check
   ``||Q(t)||_F^2 = Σ_b m_b q_b(t)^2`` at the same checkpoints.
4. **Corollary 3.9 endpoint recovery.** Run the same band-RRS machinery at
   (a) all-singleton partition (``M = P``, recovers the fixed-basis modewise
   closure of Theorem B), (b) single-band partition (``M = 1``, recovers
   scalar isotropic dynamics), (c) the main ``m = 8`` partition, and
   compare the matrix trajectories against independently-implemented
   modewise / scalar ODE integrators.
5. **Unequal-block partition.** Re-run the C1 commutant-violation and C2
   grouped-ODE tests on a genuinely unequal block partition
   ``(4, 4, 8, 16, 32)`` summing to ``D = 64``, to confirm the theorem holds
   for arbitrary partitions.

Output contract
---------------
The canonical run directory
``outputs/thesis/theoremC/run_theoremC_c1c2_supplement/<run_id>/``
contains:

- ``figures/``        — PNG figures (``item1_*``, ``item2_*``, ``item4_*``,
  ``item5_*``).
- ``pdfs/``           — LaTeX-embeddable PDFs of the same figures.
- ``config.json``     — exact ``C1C2SupplementConfig``.
- ``metadata.json``   — run_id, git commit, env, timings, compute proxy.
- ``summary.txt``     — canonical :class:`RunContext` summary.
- ``c1c2_supplement_summary.txt`` — human-readable per-item pass/fail
  summary written in addition to the canonical ``summary.txt`` per the
  user's explicit output contract.
- ``per_item_summary.json`` — structured per-item worst-case errors.

Acceptance
----------
- Items 2, 3, 4, 5 (exact algebraic tests) must pass at ``tol_exact =
  1e-12`` (``float64`` eps-scale with margin).
- Item 1 (Monte-Carlo-based Lemma 3.5 test) uses a paired-MC tolerance
  ``5 · std(paired_diff) / sqrt(N_mc)`` per ``(Q, U)`` pair (~5σ confidence
  under the null hypothesis ``E_L(Q) = E_L(UQU*)``).

Run
---
::

    python -u scripts/thesis/theoremC/run_theoremC_c1c2_supplement.py \\
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
from scripts.thesis.utils.partitions import (
    BlockPartition,
    equal_blocks,
    mass_preserving_block_spectrum,
    mass_preserving_block_task,
)
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
class C1C2SupplementConfig:
    """Frozen configuration for the C1+C2 supplement. Defaults mirror the
    main C1+C2 experiment on the ``m = 8`` partition, with added sweeps for
    items 1, 4, and 5.
    """

    # Ambient operator size + main partition (same as C1+C2).
    D: int = 64
    main_partition_m: int = 8
    block_means_lam: tuple[float, ...] = (
        1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
    )
    block_kappas_lam: tuple[float, ...] = (
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    )
    block_means_omega: tuple[float, ...] = (
        1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
    )
    block_kappas_omega: tuple[float, ...] = (
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    )
    xi_shape: str = "linear"
    spectral_basis_kind: str = "dct2"

    # Shared recursion knobs (match the main C1+C2 script).
    L_list: tuple[int, ...] = (1, 2, 4, 8)
    T: int = 5000
    eta: float = 5e-3

    # Item-1 (Lemma 3.5) knobs.
    item1_n_U: int = 20
    item1_n_mc: int = 10000
    item1_mc_batch: int = 500
    item1_tol_sigmas: float = 5.0
    item1_seed: int = 12345
    item1_random_Q_norm: float = 1.0
    item1_figure_L: int = 4  # which L to show the bar chart for

    # Item-2 / item-3 checkpoint grid.
    n_checkpoints: int = 50

    # Item-4 (Cor 3.9 endpoints) knobs.
    item4_L: int = 4
    item4_plot_mode_indices: tuple[int, ...] = (0, 8, 16, 24, 32, 40, 48, 56)

    # Item-5 (unequal partition) knobs.
    item5_block_sizes: tuple[int, ...] = (4, 4, 8, 16, 32)
    item5_block_means_lam: tuple[float, ...] = (1.0, 0.85, 0.7, 0.55, 0.4)
    item5_block_kappas_lam: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0)
    item5_block_means_omega: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6)
    item5_block_kappas_omega: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0)
    item5_L: int = 4

    # Tolerances.
    tol_exact: float = 1e-12

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Generator / partition helpers
# ---------------------------------------------------------------------------


def _build_main_g2_config(cfg: C1C2SupplementConfig) -> G2Config:
    n_blocks = cfg.D // cfg.main_partition_m
    if len(cfg.block_means_lam) != n_blocks:
        raise ValueError(
            f"block_means_lam length {len(cfg.block_means_lam)} != n_blocks = "
            f"D // m = {n_blocks}"
        )
    return G2Config(
        D=cfg.D,
        partition_kind="equal",
        partition_params={"m": cfg.main_partition_m},
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


def _build_unequal_partition(
    D: int, block_sizes: tuple[int, ...]
) -> BlockPartition:
    if sum(block_sizes) != D:
        raise ValueError(
            f"block_sizes {block_sizes} sum to {sum(block_sizes)} != D = {D}"
        )
    blocks: list[tuple[int, ...]] = []
    cursor = 0
    for m in block_sizes:
        blocks.append(tuple(range(cursor, cursor + m)))
        cursor += m
    return BlockPartition(D=D, blocks=tuple(blocks))


def _block_mean_vector(
    v: torch.Tensor, partition: BlockPartition
) -> torch.Tensor:
    """Length-``D`` vector with within-block means. Equivalent to
    ``diag(Π_C(diag(v)))`` but without the matrix round-trip.
    """
    u = torch.zeros_like(v)
    for block in partition.blocks:
        idx = list(block)
        u[idx] = v[idx].mean()
    return u


def _per_index_from_q(
    q: torch.Tensor, partition: BlockPartition
) -> torch.Tensor:
    """Length-``D`` tensor whose entry at index ``i`` is ``q_{b(i)}``."""
    out = torch.zeros(partition.D, dtype=q.dtype)
    for b_idx, block in enumerate(partition.blocks):
        idx = list(block)
        out[idx] = q[b_idx]
    return out


# ---------------------------------------------------------------------------
# Recursions (copied structurally from the main C1+C2 script so the
# supplement has no inter-script dependency on private helpers)
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
        r_avg = _block_mean_vector(naive, partition)
        gamma = gamma + eta * r_avg
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
    n_blocks = partition.n_blocks
    q = torch.zeros(n_blocks, dtype=torch.float64)
    traj = torch.zeros(T + 1, n_blocks, dtype=torch.float64)
    lam64 = lam.to(torch.float64)
    omega64 = omega.to(torch.float64)
    exponent = 2 * int(L) - 1
    block_idx_lists: list[list[int]] = [list(b) for b in partition.blocks]
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


def _evolve_per_mode(
    lam: torch.Tensor,
    omega: torch.Tensor,
    *,
    L: int,
    eta: float,
    T: int,
) -> torch.Tensor:
    """Independent per-mode ODE integrator used for item-4 (a):
    ``δq_i = η · ω_i · λ_i² · (1 − λ_i q_i / L)^{2L−1}``.

    Written independently from :func:`_evolve_matrix_r_averaged` so that the
    item-4 singleton-partition test is a bona-fide cross-code comparison.
    """
    P = int(lam.shape[0])
    q = torch.zeros(P, dtype=torch.float64)
    traj = torch.zeros(T + 1, P, dtype=torch.float64)
    lam64 = lam.to(torch.float64)
    omega64 = omega.to(torch.float64)
    exponent = 2 * int(L) - 1
    for t in range(T):
        residual = 1.0 - lam64 * q / int(L)
        delta = omega64 * lam64.pow(2) * residual.pow(exponent)
        q = q + eta * delta
        traj[t + 1] = q
    return traj


def _evolve_scalar_isotropic(
    lam: torch.Tensor,
    omega: torch.Tensor,
    *,
    L: int,
    eta: float,
    T: int,
) -> torch.Tensor:
    """Independent scalar ODE integrator used for item-4 (b):
    ``δq = η · (1/P) · Σ_i ω_i λ_i² (1 − λ_i q / L)^{2L−1}``.

    This matches the discretization convention used by
    :func:`_evolve_matrix_r_averaged` on a single-band partition (where the
    per-index update is the within-block mean).
    """
    P = int(lam.shape[0])
    q = torch.zeros((), dtype=torch.float64)
    traj = torch.zeros(T + 1, dtype=torch.float64)
    lam64 = lam.to(torch.float64)
    omega64 = omega.to(torch.float64)
    exponent = 2 * int(L) - 1
    for t in range(T):
        residual = 1.0 - lam64 * q / int(L)
        delta = (omega64 * lam64.pow(2) * residual.pow(exponent)).mean()
        q = q + eta * delta
        traj[t + 1] = q
    return traj


# ---------------------------------------------------------------------------
# Monte Carlo Haar sampling (same construction as the main script)
# ---------------------------------------------------------------------------


def _sample_block_haar_many(
    partition: BlockPartition,
    n_samples: int,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    D = partition.D
    R = torch.zeros(n_samples, D, D, dtype=dtype)
    for block in partition.blocks:
        m = len(block)
        idx = torch.tensor(list(block), dtype=torch.long)
        A = torch.randn(n_samples, m, m, generator=generator, dtype=dtype)
        Q_raw, R_upper = torch.linalg.qr(A)
        d = torch.sign(torch.diagonal(R_upper, dim1=-2, dim2=-1))
        d = torch.where(d == 0, torch.ones_like(d), d)
        Q = Q_raw * d.unsqueeze(-2)
        R[:, idx[:, None], idx[None, :]] = Q
    return R


# ---------------------------------------------------------------------------
# Item 1: Lemma 3.5 invariance test
# ---------------------------------------------------------------------------


def _loss_per_sample_batched(
    Q_fp: torch.Tensor,
    lam: torch.Tensor,
    omega: torch.Tensor,
    R_samples: torch.Tensor,
    L: int,
    batch: int,
) -> torch.Tensor:
    """Compute per-sample population-loss integrand

        ℓ_n(Q) = (1/P) · Tr[ Ω_{R_n} · ((I − Q T_{R_n} / L)^L)^T · T_{R_n}
                              · (I − Q T_{R_n} / L)^L ],

    for each of the ``N`` block-Haar samples ``R_n``, in F_P basis where
    ``T_R = R diag(λ) R^T`` and ``Ω_R = R diag(ω) R^T``.

    Returns a length-``N`` tensor (float64, CPU).
    """
    N = int(R_samples.shape[0])
    P = int(lam.shape[0])
    L_int = int(L)
    losses = torch.zeros(N, dtype=torch.float64)
    I_P = torch.eye(P, dtype=torch.float64)
    lam_row = lam.view(1, 1, P)
    om_row = omega.view(1, 1, P)
    for start in range(0, N, batch):
        stop = min(N, start + batch)
        R = R_samples[start:stop]                          # (b, P, P)
        # T_R = R · diag(λ) · R^T -> computed via elementwise scaling.
        T_R = torch.matmul(R * lam_row, R.transpose(-1, -2))     # (b, P, P)
        Om_R = torch.matmul(R * om_row, R.transpose(-1, -2))     # (b, P, P)
        Q_batch = Q_fp.unsqueeze(0).expand(stop - start, -1, -1)
        M = I_P.unsqueeze(0) - torch.matmul(Q_batch, T_R) / L_int
        M_pow = M.clone()
        for _ in range(L_int - 1):
            M_pow = torch.matmul(M_pow, M)
        inner = torch.matmul(
            torch.matmul(M_pow.transpose(-1, -2), T_R), M_pow
        )
        # Tr[Ω_R · inner] per-sample, divided by P.
        losses[start:stop] = (
            torch.einsum("nij,nji->n", Om_R, inner) / P
        ).to(torch.float64)
    return losses


def _build_test_Qs(
    cfg: C1C2SupplementConfig,
    partition: BlockPartition,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Build two test Q matrices in F_P basis: (1) a random symmetric
    matrix that is NOT in the commutant, and (2) a random block-scalar
    matrix that IS in the commutant (sanity check).
    """
    P = partition.D
    # Random symmetric Q (not in commutant).
    A = torch.randn(P, P, generator=generator, dtype=torch.float64)
    Q_random = 0.5 * (A + A.T)
    # Normalize to a controlled Frobenius scale so loss magnitudes are O(1).
    frob = Q_random.norm()
    if float(frob) > 0:
        Q_random = Q_random * (cfg.item1_random_Q_norm / float(frob))

    # Random block-scalar Q (in commutant).
    n_blocks = partition.n_blocks
    q_rand = torch.randn(n_blocks, generator=generator, dtype=torch.float64)
    # Scale so ||Q_commutant||_F ~ item1_random_Q_norm too.
    q_per_idx = _per_index_from_q(q_rand, partition)
    Q_commutant = torch.diag(q_per_idx)
    frob_c = Q_commutant.norm()
    if float(frob_c) > 0:
        Q_commutant = Q_commutant * (cfg.item1_random_Q_norm / float(frob_c))
    return {"random": Q_random, "commutant": Q_commutant}


def _run_item1(
    cfg: C1C2SupplementConfig,
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
) -> dict[str, Any]:
    """Paired Monte-Carlo evaluation of ``E_L(Q)`` vs ``E_L(UQU*)`` for 20
    block-Haar ``U``, for both ``Q ∈ C(B)`` and ``Q ∉ C(B)``, at every L.

    Tolerance per ``(Q, U)`` pair: ``item1_tol_sigmas ·
    std(paired_diff) / sqrt(N_mc)``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.item1_seed))

    Qs = _build_test_Qs(cfg, partition, gen)

    # Pre-sample the 20 ``U`` elements of G_B (in F_P basis ⇒ block-Haar).
    U_samples = _sample_block_haar_many(
        partition, cfg.item1_n_U, gen, dtype=torch.float64
    )  # (n_U, P, P)

    # Pre-sample the N_mc block-Haar R's used by all loss evaluations. Using
    # the SAME R samples for every loss evaluation is the paired-MC design:
    # the Lemma-3.5 null reduces to a single random variable whose paired
    # variance controls the tolerance.
    R_samples = _sample_block_haar_many(
        partition, cfg.item1_n_mc, gen, dtype=torch.float64
    )

    rows: list[dict[str, Any]] = []
    bar_data: dict[str, Any] | None = None

    for L in cfg.L_list:
        L_int = int(L)
        for Q_label, Q_fp in Qs.items():
            # Baseline E_L(Q) (per-sample losses for paired comparison).
            losses_Q = _loss_per_sample_batched(
                Q_fp, lam, omega, R_samples, L_int, int(cfg.item1_mc_batch)
            )
            mean_Q = float(losses_Q.mean().item())

            u_diffs: list[float] = []
            u_means: list[float] = []
            u_tols: list[float] = []
            u_ok: list[bool] = []
            for u_idx in range(cfg.item1_n_U):
                U = U_samples[u_idx]
                UQU = U @ Q_fp @ U.T
                losses_UQU = _loss_per_sample_batched(
                    UQU, lam, omega, R_samples, L_int,
                    int(cfg.item1_mc_batch),
                )
                mean_UQU = float(losses_UQU.mean().item())
                diff = mean_Q - mean_UQU

                # Paired-MC standard error of the difference.
                paired = losses_Q - losses_UQU
                se = float(paired.std(unbiased=True).item()) / (
                    cfg.item1_n_mc ** 0.5
                )
                tol = float(cfg.item1_tol_sigmas) * se
                # Fallback: when paired variance is below eps (Q ∈ C(B) and
                # U ∈ G_B commute ⇒ UQU* = Q ⇒ paired diff ≡ 0), accept
                # anything at machine precision.
                if tol < cfg.tol_exact:
                    tol = cfg.tol_exact
                ok = abs(diff) <= tol
                u_diffs.append(float(diff))
                u_means.append(mean_UQU)
                u_tols.append(tol)
                u_ok.append(bool(ok))

            all_ok = bool(all(u_ok))
            worst_idx = int(np.argmax(np.abs(np.asarray(u_diffs))))
            row = {
                "L": L_int,
                "Q": Q_label,
                "n_U": int(cfg.item1_n_U),
                "baseline_loss": mean_Q,
                "max_abs_diff": float(
                    np.max(np.abs(np.asarray(u_diffs)))
                ),
                "worst_tol_at_max": float(u_tols[worst_idx]),
                "worst_sigmas_at_max": (
                    float(u_diffs[worst_idx])
                    / max(u_tols[worst_idx] / cfg.item1_tol_sigmas, 1e-30)
                ),
                "all_within_tol": all_ok,
                "u_means": u_means,
                "u_diffs": u_diffs,
                "u_tols": u_tols,
            }
            rows.append(row)
            if (
                bar_data is None
                and L_int == int(cfg.item1_figure_L)
                and Q_label == "random"
            ):
                bar_data = {
                    "L": L_int,
                    "Q_label": Q_label,
                    "mean_Q": mean_Q,
                    "u_means": list(u_means),
                    "u_diffs": list(u_diffs),
                    "u_tols": list(u_tols),
                }
    return {"rows": rows, "bar_data": bar_data}


# ---------------------------------------------------------------------------
# Item 2 + Item 3: grouped-loss formula and induced-metric verification
# ---------------------------------------------------------------------------


def _matrix_loss_block_scalar(
    q: torch.Tensor,
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
    L: int,
) -> float:
    """"Matrix loss" path: compute the closed-form population loss for a
    block-scalar ``Q`` via an explicit P×P matrix-multiplication sequence
    in F_P basis (where ``Q = D_q``, ``T = diag(λ)``, ``Ω = diag(ω)``). This
    does NOT shortcut via the diagonality of ``D_q Λ``; it builds the full
    P×P matrices and raises ``(I − D_q Λ / L)`` to power ``L`` by repeated
    ``torch.matmul``.

    The population block-Haar average has already been carried out
    analytically (Theorem 3.8: all ``R`` cancel when ``Q ∈ C(B)``), so this
    is the deterministic "ground truth" matrix expression.
    """
    P = int(lam.shape[0])
    L_int = int(L)
    q_per_idx = _per_index_from_q(q, partition)
    D_q = torch.diag(q_per_idx)                 # (P, P)
    Lam = torch.diag(lam.to(torch.float64))     # (P, P)
    Om = torch.diag(omega.to(torch.float64))    # (P, P)
    I_P = torch.eye(P, dtype=torch.float64)
    M = I_P - D_q @ Lam / L_int
    M_pow = M
    for _ in range(L_int - 1):
        M_pow = M_pow @ M
    inner = M_pow.T @ Lam @ M_pow
    return float(torch.trace(Om @ inner).item()) / P


def _grouped_formula_loss(
    q: torch.Tensor,
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
    L: int,
) -> float:
    """Evaluate equation (3.2) of Theorem 3.8 directly."""
    P = partition.D
    L_int = int(L)
    total = 0.0
    lam64 = lam.to(torch.float64)
    om64 = omega.to(torch.float64)
    for b_idx, block in enumerate(partition.blocks):
        idx = list(block)
        lam_b = lam64[idx]
        om_b = om64[idx]
        residual = 1.0 - lam_b * q[b_idx] / L_int
        total += float((om_b * lam_b * residual.pow(2 * L_int)).sum().item())
    return total / P


def _log_spaced_checkpoints(T: int, n: int) -> np.ndarray:
    """Return up to ``n`` unique integer checkpoints in ``[1, T]`` spaced
    log-uniformly. Always includes ``T``.
    """
    if n < 1 or T < 1:
        return np.asarray([T], dtype=np.int64)
    raw = np.unique(np.round(np.geomspace(1.0, float(T), n)).astype(np.int64))
    raw = raw[(raw >= 1) & (raw <= T)]
    if raw.size == 0 or raw[-1] != T:
        raw = np.concatenate([raw, np.asarray([T], dtype=np.int64)])
    return np.unique(raw)


def _run_item2_item3(
    cfg: C1C2SupplementConfig,
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
) -> dict[str, Any]:
    """For each L, evolve the R-averaged matrix recursion on the main
    partition, then at ``n_checkpoints`` log-spaced steps extract the
    block-scalar coordinates ``q_b`` and evaluate:

      (2a) matrix_loss  — explicit P×P matrix path,
      (2b) grouped_loss — equation (3.2) direct sum,
      (3)  metric check  — ||Q(t)||_F^2 vs Σ_b m_b q_b^2.
    """
    checkpoints = _log_spaced_checkpoints(cfg.T, cfg.n_checkpoints)
    sizes = np.asarray(partition.sizes, dtype=np.float64)
    sizes_t = torch.tensor(sizes, dtype=torch.float64)

    per_L: dict[int, dict[str, np.ndarray]] = {}
    per_L_stats: list[dict[str, Any]] = []

    for L in cfg.L_list:
        gamma_traj, _cv = _evolve_matrix_r_averaged(
            lam, omega, partition, L=int(L), eta=cfg.eta, T=cfg.T
        )
        checkpoint_list = checkpoints.tolist()
        matrix_losses = np.zeros(len(checkpoint_list), dtype=np.float64)
        grouped_losses = np.zeros(len(checkpoint_list), dtype=np.float64)
        metric_lhs = np.zeros(len(checkpoint_list), dtype=np.float64)
        metric_rhs = np.zeros(len(checkpoint_list), dtype=np.float64)
        for i, step in enumerate(checkpoint_list):
            gamma_t = gamma_traj[int(step)]
            q_t = extract_block_scalars(torch.diag(gamma_t), partition)
            matrix_losses[i] = _matrix_loss_block_scalar(
                q_t, lam, omega, partition, int(L)
            )
            grouped_losses[i] = _grouped_formula_loss(
                q_t, lam, omega, partition, int(L)
            )
            # ||diag(gamma_t)||_F^2 = ||gamma_t||_2^2.
            metric_lhs[i] = float(gamma_t.pow(2).sum().item())
            metric_rhs[i] = float(
                (sizes_t * q_t.pow(2)).sum().item()
            )
        abs_err_loss = np.abs(matrix_losses - grouped_losses)
        rel_err_loss = abs_err_loss / np.maximum(
            np.abs(matrix_losses), 1e-300
        )
        rel_err_metric = np.abs(metric_lhs - metric_rhs) / np.maximum(
            np.abs(metric_lhs), 1e-300
        )
        per_L[int(L)] = {
            "checkpoints": np.asarray(checkpoint_list, dtype=np.int64),
            "matrix_loss": matrix_losses,
            "grouped_loss": grouped_losses,
            "rel_err_loss": rel_err_loss,
            "metric_lhs": metric_lhs,
            "metric_rhs": metric_rhs,
            "rel_err_metric": rel_err_metric,
        }
        per_L_stats.append(
            {
                "L": int(L),
                "max_rel_err_loss": float(rel_err_loss.max()),
                "max_abs_err_loss": float(abs_err_loss.max()),
                "max_rel_err_metric": float(rel_err_metric.max()),
            }
        )
    item2_max_rel = max(s["max_rel_err_loss"] for s in per_L_stats)
    item3_max_rel = max(s["max_rel_err_metric"] for s in per_L_stats)
    return {
        "per_L": per_L,
        "per_L_stats": per_L_stats,
        "item2_max_rel_err": float(item2_max_rel),
        "item3_max_rel_err": float(item3_max_rel),
    }


# ---------------------------------------------------------------------------
# Item 4: Corollary 3.9 endpoint recovery
# ---------------------------------------------------------------------------


def _run_item4(
    cfg: C1C2SupplementConfig,
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition_main: BlockPartition,
) -> dict[str, Any]:
    """Run the three partition regimes at ``item4_L`` and compare
    matrix-level R-averaged trajectory against the independently-
    implemented modewise / scalar ODE integrator.
    """
    L = int(cfg.item4_L)
    T = cfg.T
    eta = cfg.eta
    D = cfg.D

    # (a) All singletons.
    part_single = equal_blocks(D, 1)
    gamma_a, _ = _evolve_matrix_r_averaged(
        lam, omega, part_single, L=L, eta=eta, T=T
    )
    # With singleton blocks, the matrix γ_i trajectory IS the per-mode q_i
    # trajectory; compare against the independent per-mode ODE.
    q_per_mode = _evolve_per_mode(lam, omega, L=L, eta=eta, T=T)
    abs_err_a = float((gamma_a - q_per_mode).abs().max().item())
    rel_err_a = float(
        ((gamma_a - q_per_mode).abs()
         / (q_per_mode.abs().max() + 1e-300)).max().item()
    )

    # (b) Single band.
    part_single_band = BlockPartition(D=D, blocks=(tuple(range(D)),))
    gamma_b, _ = _evolve_matrix_r_averaged(
        lam, omega, part_single_band, L=L, eta=eta, T=T
    )
    # With one band, every γ_i equals the scalar q; pick index 0 for the
    # trajectory and verify against the independent scalar ODE.
    q_scalar_matrix = gamma_b[:, 0]
    q_scalar_ode = _evolve_scalar_isotropic(
        lam, omega, L=L, eta=eta, T=T
    )
    abs_err_b = float((q_scalar_matrix - q_scalar_ode).abs().max().item())
    rel_err_b = float(
        ((q_scalar_matrix - q_scalar_ode).abs()
         / (q_scalar_ode.abs().max() + 1e-300)).max().item()
    )
    # Also verify internal consistency: all γ_i should be equal inside the
    # single band, i.e. the γ matrix path is scalar-isotropic at every t.
    within_band_spread = float(
        (gamma_b - gamma_b.mean(dim=1, keepdim=True)).abs().max().item()
    )

    # (c) Original m = 8 partition.
    gamma_c, _ = _evolve_matrix_r_averaged(
        lam, omega, partition_main, L=L, eta=eta, T=T
    )
    q_c_ode = _evolve_grouped_ode(
        lam, omega, partition_main, L=L, eta=eta, T=T
    )
    # Extract block scalars from the matrix path and compare with the
    # grouped ODE (this is the same check the main C1+C2 script performs,
    # redone here at L = item4_L for completeness).
    q_c_mat = torch.zeros_like(q_c_ode)
    for t in range(T + 1):
        q_c_mat[t] = extract_block_scalars(
            torch.diag(gamma_c[t]), partition_main
        )
    abs_err_c = float((q_c_mat - q_c_ode).abs().max().item())

    return {
        "L": L,
        "singleton": {
            "gamma_matrix": gamma_a,
            "q_per_mode": q_per_mode,
            "abs_err": abs_err_a,
            "rel_err": rel_err_a,
        },
        "single_band": {
            "q_matrix": q_scalar_matrix,
            "q_ode": q_scalar_ode,
            "within_band_spread": within_band_spread,
            "abs_err": abs_err_b,
            "rel_err": rel_err_b,
        },
        "m8": {
            "q_matrix": q_c_mat,
            "q_ode": q_c_ode,
            "abs_err": abs_err_c,
        },
    }


# ---------------------------------------------------------------------------
# Item 5: unequal partition
# ---------------------------------------------------------------------------


def _run_item5(
    cfg: C1C2SupplementConfig,
) -> dict[str, Any]:
    """Run the C1 commutant-violation and C2 grouped-ODE tests on the
    unequal partition ``item5_block_sizes``.
    """
    partition = _build_unequal_partition(cfg.D, cfg.item5_block_sizes)
    block_means_lam = torch.tensor(
        cfg.item5_block_means_lam, dtype=torch.float64
    )
    block_kappas_lam = torch.tensor(
        cfg.item5_block_kappas_lam, dtype=torch.float64
    )
    block_means_om = torch.tensor(
        cfg.item5_block_means_omega, dtype=torch.float64
    )
    block_kappas_om = torch.tensor(
        cfg.item5_block_kappas_omega, dtype=torch.float64
    )
    lam = mass_preserving_block_spectrum(
        partition, block_means_lam, block_kappas_lam,
        xi_shape=cfg.xi_shape, dtype=torch.float64,
    )
    omega = mass_preserving_block_task(
        partition, block_means_om, block_kappas_om,
        xi_shape=cfg.xi_shape, dtype=torch.float64,
    )

    L = int(cfg.item5_L)
    gamma_r, cv_r = _evolve_matrix_r_averaged(
        lam, omega, partition, L=L, eta=cfg.eta, T=cfg.T
    )
    q_ode = _evolve_grouped_ode(
        lam, omega, partition, L=L, eta=cfg.eta, T=cfg.T
    )
    q_mat = torch.zeros_like(q_ode)
    for t in range(cfg.T + 1):
        q_mat[t] = extract_block_scalars(torch.diag(gamma_r[t]), partition)

    return {
        "partition": partition,
        "lam": lam,
        "omega": omega,
        "L": L,
        "gamma_r": gamma_r,
        "cv_r": cv_r,
        "q_ode": q_ode,
        "q_mat": q_mat,
        "cv_max": float(cv_r.max().item()),
        "q_err_max": float((q_mat - q_ode).abs().max().item()),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_item1(
    cfg: C1C2SupplementConfig,
    item1: dict[str, Any],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    bar = item1["bar_data"]
    if bar is None:
        return
    u_means = np.asarray(bar["u_means"])
    u_tols = np.asarray(bar["u_tols"])
    mean_Q = float(bar["mean_Q"])
    idx = np.arange(len(u_means))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(idx, u_means, color="#3b6ea8", edgecolor="k", lw=0.6,
           label=r"$\widehat{E}_L(UQU^\top)$ (paired MC)")
    ax.errorbar(idx, u_means, yerr=u_tols, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=2.5,
                label=f"± {cfg.item1_tol_sigmas:.0f}σ paired MC tol")
    ax.axhline(mean_Q, color="red", lw=1.1, ls="--",
               label=r"$\widehat{E}_L(Q)$ (baseline)")
    ax.set_xlabel("U sample index")
    ax.set_ylabel(r"population loss estimate $\widehat{E}_L$")
    ax.set_title(
        f"Item 1 — Lemma 3.5 band-rotation invariance (L = {bar['L']}, "
        f"random Q): paired MC with N = {cfg.item1_n_mc}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    save_both(fig, run_dir, "item1_lemma35_invariance_bar")
    plt.close(fig)


def _plot_item2(
    cfg: C1C2SupplementConfig,
    item23: dict[str, Any],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    L_colors = sequential_colors(len(cfg.L_list), palette="rocket")

    # (i) Loss curves: matrix vs grouped formula.
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for color, L in zip(L_colors, cfg.L_list):
        data = item23["per_L"][int(L)]
        t = data["checkpoints"].astype(float)
        ax.plot(t, data["matrix_loss"], color=color, lw=1.4,
                label=f"matrix  L = {int(L)}")
        overlay_reference(
            ax, t, data["grouped_loss"],
            label=None, style="--", color="black", lw=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel(r"$E_L(Q(t))$")
    ax.set_title(
        "Item 2 — matrix loss (colored, solid) vs grouped formula eq. (3.2) "
        "(dashed black); visually indistinguishable",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "item2_loss_curves")
    plt.close(fig)

    # (ii) Relative error vs t.
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    floor = cfg.tol_exact * 1e-3
    for color, L in zip(L_colors, cfg.L_list):
        data = item23["per_L"][int(L)]
        t = data["checkpoints"].astype(float)
        err = np.where(data["rel_err_loss"] > floor,
                       data["rel_err_loss"], floor)
        ax.plot(t, err, color=color, lw=1.3, label=f"L = {int(L)}")
    ax.axhline(cfg.tol_exact, color="red", lw=0.8, ls="--",
               label=f"tol = {cfg.tol_exact:.0e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel(
        r"$|E_L^{\mathrm{mat}} - E_L^{\mathrm{grp}}| "
        r"/ |E_L^{\mathrm{mat}}|$"
    )
    ax.set_title(
        "Item 2 — relative error between matrix-loss and grouped-formula "
        "paths (expected ≈ float eps)",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "item2_relative_error")
    plt.close(fig)


def _plot_item4(
    cfg: C1C2SupplementConfig,
    item4: dict[str, Any],
    run_dir: ThesisRunDir,
    partition_main: BlockPartition,
) -> None:
    import matplotlib.pyplot as plt

    T = cfg.T
    t_axis = np.arange(1, T + 1, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    # (a) Singletons — 8 representative modes.
    mode_indices = [int(i) for i in cfg.item4_plot_mode_indices]
    colors_a = sequential_colors(len(mode_indices), palette="rocket")
    gamma_a = item4["singleton"]["gamma_matrix"].numpy()
    q_per_mode = item4["singleton"]["q_per_mode"].numpy()
    for color, k in zip(colors_a, mode_indices):
        y_mat = gamma_a[1:, k]
        y_ode = q_per_mode[1:, k]
        y_mat_safe = np.where(y_mat > 0, y_mat, np.nan)
        y_ode_safe = np.where(y_ode > 0, y_ode, np.nan)
        axes[0].plot(t_axis, y_mat_safe, color=color, lw=1.3,
                     label=f"k = {k}")
        overlay_reference(
            axes[0], t_axis, y_ode_safe,
            label=None, style="--", color="black", lw=0.9,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step t")
    axes[0].set_ylabel(r"mode coordinate $q_k(t)$")
    axes[0].set_title(
        f"(a) all singletons  M = P = {cfg.D}\n"
        "matrix (solid) vs per-mode ODE (dashed)",
        fontsize=9,
    )
    axes[0].legend(fontsize=7, loc="best", ncol=2)

    # (c) m = 8 — eight block scalars.
    colors_c = sequential_colors(partition_main.n_blocks, palette="rocket")
    q_c_mat = item4["m8"]["q_matrix"].numpy()
    q_c_ode = item4["m8"]["q_ode"].numpy()
    for color, b in zip(colors_c, range(partition_main.n_blocks)):
        y_mat = q_c_mat[1:, b]
        y_ode = q_c_ode[1:, b]
        y_mat = np.where(y_mat > 0, y_mat, np.nan)
        y_ode = np.where(y_ode > 0, y_ode, np.nan)
        axes[1].plot(t_axis, y_mat, color=color, lw=1.3,
                     label=f"b = {b}")
        overlay_reference(
            axes[1], t_axis, y_ode,
            label=None, style="--", color="black", lw=0.9,
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("step t")
    axes[1].set_ylabel(r"block scalar $q_b(t)$")
    axes[1].set_title(
        f"(b) main partition  m = {cfg.main_partition_m}, "
        f"M = {partition_main.n_blocks}\n"
        "matrix (solid) vs grouped ODE (dashed)",
        fontsize=9,
    )
    axes[1].legend(fontsize=7, loc="best", ncol=2)

    # (b) Single band — one scalar.
    q_s_mat = item4["single_band"]["q_matrix"].numpy()
    q_s_ode = item4["single_band"]["q_ode"].numpy()
    y_mat = np.where(q_s_mat[1:] > 0, q_s_mat[1:], np.nan)
    y_ode = np.where(q_s_ode[1:] > 0, q_s_ode[1:], np.nan)
    axes[2].plot(t_axis, y_mat, color="#3b6ea8", lw=1.4,
                 label="matrix R-averaged")
    overlay_reference(
        axes[2], t_axis, y_ode,
        label="scalar ODE", style="--", color="black", lw=1.0,
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("step t")
    axes[2].set_ylabel(r"scalar coordinate $q(t)$")
    axes[2].set_title(
        f"(c) single band  M = 1, P = {cfg.D}\n"
        "matrix (solid) vs scalar ODE (dashed)",
        fontsize=9,
    )
    axes[2].legend(fontsize=8, loc="best")

    fig.suptitle(
        "Item 4 — Corollary 3.9 endpoint recovery "
        f"(L = {cfg.item4_L}):  modewise (M=P) ↔ grouped (M=8) ↔ scalar (M=1)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_both(fig, run_dir, "item4_endpoint_recovery")
    plt.close(fig)


def _plot_item5(
    cfg: C1C2SupplementConfig,
    item5: dict[str, Any],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    partition: BlockPartition = item5["partition"]
    L = int(item5["L"])
    T = cfg.T

    # CV trajectory.
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    floor = cfg.tol_exact * 1e-3
    t_axis = np.arange(1, T + 1, dtype=float)
    cv = item5["cv_r"].numpy()[1:]
    cv_plot = np.where(cv > floor, cv, floor)
    ax.plot(t_axis, cv_plot, color="#3b6ea8", lw=1.3,
            label=f"R-averaged (unequal blocks, L = {L})")
    ax.axhline(cfg.tol_exact, color="red", lw=0.8, ls="--",
               label=f"tol = {cfg.tol_exact:.0e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel(
        r"commutant violation $\|\Gamma - \Pi_C(\Gamma)\|_F^2/\|\Gamma\|_F^2$"
    )
    ax.set_title(
        f"Item 5 — unequal partition {tuple(partition.sizes)}: "
        "commutant closure (expected ≈ float eps)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "item5_unequal_commutant_trajectory")
    plt.close(fig)

    # Grouped trajectories.
    n_blocks = partition.n_blocks
    colors = sequential_colors(n_blocks, palette="rocket")
    q_mat = item5["q_mat"].numpy()
    q_ode = item5["q_ode"].numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for b, color in enumerate(colors):
        y_mat = q_mat[1:, b]
        y_ode = q_ode[1:, b]
        y_mat = np.where(y_mat > 0, y_mat, np.nan)
        y_ode = np.where(y_ode > 0, y_ode, np.nan)
        ax.plot(t_axis, y_mat, color=color, lw=1.4,
                label=f"b = {b}  (m = {partition.sizes[b]})")
        overlay_reference(
            ax, t_axis, y_ode, label=None,
            style="--", color="black", lw=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel(r"block scalar $q_b(t)$")
    ax.set_title(
        f"Item 5 — unequal partition {tuple(partition.sizes)}: "
        "matrix $q_b$ (solid) vs grouped ODE (dashed)",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="best", frameon=True)
    fig.tight_layout()
    save_both(fig, run_dir, "item5_unequal_grouped_trajectories")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary text writer
# ---------------------------------------------------------------------------


def _write_supplement_summary(
    run_dir: ThesisRunDir,
    cfg: C1C2SupplementConfig,
    per_item: dict[str, Any],
) -> Path:
    """Write the human-readable ``c1c2_supplement_summary.txt`` required by
    the user-specified output contract.
    """
    path = run_dir.root / "c1c2_supplement_summary.txt"
    lines: list[str] = []
    lines.append("Experiment C1+C2 supplement — per-item acceptance summary")
    lines.append("=" * 72)
    lines.append(
        "Plan ref: EXPERIMENT_PLAN_FINAL.MD §7.1–§7.2 (supplements C1+C2)."
    )
    lines.append(
        "Theorem ref: thesis/theorem_c.txt — Lemma 3.5, Theorem 3.8, "
        "Corollary 3.9."
    )
    lines.append(
        f"Config: D = {cfg.D}, main partition m = {cfg.main_partition_m}, "
        f"T = {cfg.T}, eta = {cfg.eta}"
    )
    lines.append("")

    def _mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    # Item 1
    i1 = per_item["item1"]
    lines.append("Item 1 — Lemma 3.5 band-rotation invariance (paired MC)")
    lines.append(
        f"  tol per pair: {cfg.item1_tol_sigmas:.1f}σ with "
        f"N_mc = {cfg.item1_n_mc} (~ const / sqrt(N_mc))"
    )
    lines.append(
        f"  n_U per (Q, L) cell: {cfg.item1_n_U}"
    )
    for row in i1["rows"]:
        lines.append(
            f"    L = {row['L']}  Q = {row['Q']:>9s}  "
            f"max |diff| = {row['max_abs_diff']:.3e}  "
            f"tol@worst = {row['worst_tol_at_max']:.3e}  "
            f"{_mark(row['all_within_tol'])}"
        )
    lines.append(
        f"  → overall: {_mark(i1['all_ok'])}  "
        f"(worst |diff| = {i1['worst_abs_diff']:.3e})"
    )
    lines.append("")

    # Item 2
    i2 = per_item["item2"]
    lines.append("Item 2 — grouped-loss formula eq. (3.2) direct verification")
    lines.append(f"  tol: {cfg.tol_exact:.0e}")
    for s in i2["per_L_stats"]:
        lines.append(
            f"    L = {s['L']}  max rel err = {s['max_rel_err_loss']:.3e}  "
            f"max abs err = {s['max_abs_err_loss']:.3e}"
        )
    lines.append(
        f"  → overall: {_mark(i2['ok'])}  "
        f"(worst rel err = {i2['worst_rel_err']:.3e})"
    )
    lines.append("")

    # Item 3
    i3 = per_item["item3"]
    lines.append("Item 3 — induced metric ‖Q‖_F² = Σ_b m_b q_b²")
    lines.append(f"  tol: {cfg.tol_exact:.0e}")
    for s in i3["per_L_stats"]:
        lines.append(
            f"    L = {s['L']}  max rel err = {s['max_rel_err_metric']:.3e}"
        )
    lines.append(
        f"  → overall: {_mark(i3['ok'])}  "
        f"(worst rel err = {i3['worst_rel_err']:.3e})"
    )
    lines.append("")

    # Item 4
    i4 = per_item["item4"]
    lines.append(
        f"Item 4 — Corollary 3.9 endpoint recovery (L = {cfg.item4_L})"
    )
    lines.append(f"  tol: {cfg.tol_exact:.0e}")
    lines.append(
        f"    (a) singletons   max |γ(t) − q_per_mode| = "
        f"{i4['singleton']['abs_err']:.3e}  (rel = "
        f"{i4['singleton']['rel_err']:.3e})  "
        f"{_mark(i4['singleton']['ok'])}"
    )
    lines.append(
        f"    (b) single band  max |γ(t) − q_scalar| = "
        f"{i4['single_band']['abs_err']:.3e}  (rel = "
        f"{i4['single_band']['rel_err']:.3e})  "
        f"within_band_spread = {i4['single_band']['within_band_spread']:.3e}  "
        f"{_mark(i4['single_band']['ok'])}"
    )
    lines.append(
        f"    (c) m = {cfg.main_partition_m:>2d}         max |q_mat − q_ode| = "
        f"{i4['m8']['abs_err']:.3e}  {_mark(i4['m8']['ok'])}"
    )
    lines.append(f"  → overall: {_mark(i4['ok'])}")
    lines.append("")

    # Item 5
    i5 = per_item["item5"]
    lines.append(
        f"Item 5 — unequal partition {tuple(i5['block_sizes'])}"
    )
    lines.append(f"  tol: {cfg.tol_exact:.0e}")
    lines.append(
        f"    max commutant violation = {i5['cv_max']:.3e}  "
        f"{_mark(i5['cv_ok'])}"
    )
    lines.append(
        f"    max |q_mat − q_ode|     = {i5['q_err_max']:.3e}  "
        f"{_mark(i5['q_err_ok'])}"
    )
    lines.append(f"  → overall: {_mark(i5['ok'])}")
    lines.append("")

    lines.append("=" * 72)
    lines.append(f"Top-line status: {_mark(per_item['all_ok'])}")
    lines.append(
        "Category: operator-level exact theorem-C supplement — no learned "
        "architecture. Items 2/3/4/5 are algebraic; item 1 is a paired-MC "
        "Lemma-3.5 test at ~5σ confidence."
    )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI + driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment C1+C2 supplement (plan §7.1–§7.2): Lemma 3.5 "
            "invariance, grouped-loss formula, induced metric, Corollary "
            "3.9 endpoints, unequal partition."
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
    p.add_argument("--main-partition-m", type=int, default=None)
    p.add_argument("--L-list", type=str, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--item1-n-mc", type=int, default=None)
    p.add_argument("--item1-n-U", type=int, default=None)
    p.add_argument("--n-checkpoints", type=int, default=None)
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> C1C2SupplementConfig:
    base = C1C2SupplementConfig()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.D is not None:
        overrides["D"] = int(args.D)
    if args.main_partition_m is not None:
        overrides["main_partition_m"] = int(args.main_partition_m)
    if args.L_list is not None:
        overrides["L_list"] = _parse_list_ints(args.L_list)
    if args.T is not None:
        overrides["T"] = int(args.T)
    if args.eta is not None:
        overrides["eta"] = float(args.eta)
    if args.item1_n_mc is not None:
        overrides["item1_n_mc"] = int(args.item1_n_mc)
    if args.item1_n_U is not None:
        overrides["item1_n_U"] = int(args.item1_n_U)
    if args.n_checkpoints is not None:
        overrides["n_checkpoints"] = int(args.n_checkpoints)
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
    print(f"[C1C2-SUP] device = {device}")

    # Build the main G2 operator (same spectra as the main C1+C2 experiment).
    g2_cfg = _build_main_g2_config(cfg)
    op = g2_generate_operator(g2_cfg)
    lam = op["Lambda"].to(torch.float64)
    omega = op["Omega"].to(torch.float64)
    partition_main: BlockPartition = op["partition"]
    print(
        f"[C1C2-SUP] D={cfg.D}  main partition=equal×{cfg.main_partition_m}  "
        f"n_blocks={partition_main.n_blocks}"
    )

    run = ThesisRunDir(__file__, phase="theoremC")
    with RunContext(
        run,
        config=cfg,
        seeds=[int(cfg.item1_seed)],
        notes=(
            "C1+C2 supplement: Lemma 3.5 invariance (item 1), grouped loss "
            "formula (item 2), induced metric (item 3), Corollary 3.9 "
            "endpoints (item 4), unequal partition (item 5)."
        ),
    ) as ctx:
        apply_thesis_style()
        t_overall = time.perf_counter()

        # ---- Item 1 ----
        print("[C1C2-SUP] item 1 — Lemma 3.5 paired-MC invariance ...")
        t0 = time.perf_counter()
        item1_raw = _run_item1(cfg, lam, omega, partition_main)
        item1_dt = time.perf_counter() - t0
        ctx.record_step_time(item1_dt)
        worst_diff = max(r["max_abs_diff"] for r in item1_raw["rows"])
        all_ok_1 = all(r["all_within_tol"] for r in item1_raw["rows"])
        print(
            f"[C1C2-SUP] item 1 done in {item1_dt:.1f} s — "
            f"worst |diff| = {worst_diff:.3e}  "
            f"{'OK' if all_ok_1 else 'FAIL'}"
        )
        for row in item1_raw["rows"]:
            print(
                f"   L={row['L']}  Q={row['Q']:>9s}  "
                f"max|diff|={row['max_abs_diff']:.3e}  "
                f"tol@worst={row['worst_tol_at_max']:.3e}  "
                f"all_ok={row['all_within_tol']}"
            )
        item1_block = {
            "rows": item1_raw["rows"],
            "bar_data": item1_raw["bar_data"],
            "worst_abs_diff": float(worst_diff),
            "all_ok": bool(all_ok_1),
        }

        # ---- Items 2 + 3 ----
        print("[C1C2-SUP] items 2+3 — grouped loss formula + induced metric ...")
        t0 = time.perf_counter()
        item23 = _run_item2_item3(cfg, lam, omega, partition_main)
        item23_dt = time.perf_counter() - t0
        ctx.record_step_time(item23_dt)
        item2_ok = item23["item2_max_rel_err"] <= cfg.tol_exact
        item3_ok = item23["item3_max_rel_err"] <= cfg.tol_exact
        print(
            f"[C1C2-SUP] items 2+3 done in {item23_dt:.1f} s — "
            f"item2 worst rel err = {item23['item2_max_rel_err']:.3e} "
            f"({'OK' if item2_ok else 'FAIL'})  "
            f"item3 worst rel err = {item23['item3_max_rel_err']:.3e} "
            f"({'OK' if item3_ok else 'FAIL'})"
        )
        item2_block = {
            "per_L_stats": item23["per_L_stats"],
            "worst_rel_err": item23["item2_max_rel_err"],
            "ok": bool(item2_ok),
        }
        item3_block = {
            "per_L_stats": item23["per_L_stats"],
            "worst_rel_err": item23["item3_max_rel_err"],
            "ok": bool(item3_ok),
        }

        # ---- Item 4 ----
        print("[C1C2-SUP] item 4 — Corollary 3.9 endpoint recovery ...")
        t0 = time.perf_counter()
        item4_raw = _run_item4(cfg, lam, omega, partition_main)
        item4_dt = time.perf_counter() - t0
        ctx.record_step_time(item4_dt)
        singleton_ok = item4_raw["singleton"]["abs_err"] <= cfg.tol_exact
        single_band_ok = (
            item4_raw["single_band"]["abs_err"] <= cfg.tol_exact
            and item4_raw["single_band"]["within_band_spread"]
            <= cfg.tol_exact
        )
        m8_ok = item4_raw["m8"]["abs_err"] <= cfg.tol_exact
        item4_ok = singleton_ok and single_band_ok and m8_ok
        print(
            f"[C1C2-SUP] item 4 done in {item4_dt:.1f} s — "
            f"singletons {item4_raw['singleton']['abs_err']:.3e} "
            f"({'OK' if singleton_ok else 'FAIL'})  "
            f"single-band {item4_raw['single_band']['abs_err']:.3e} "
            f"({'OK' if single_band_ok else 'FAIL'})  "
            f"m=8 {item4_raw['m8']['abs_err']:.3e} "
            f"({'OK' if m8_ok else 'FAIL'})"
        )
        item4_block = {
            "L": int(cfg.item4_L),
            "singleton": {
                "abs_err": float(item4_raw["singleton"]["abs_err"]),
                "rel_err": float(item4_raw["singleton"]["rel_err"]),
                "ok": bool(singleton_ok),
            },
            "single_band": {
                "abs_err": float(item4_raw["single_band"]["abs_err"]),
                "rel_err": float(item4_raw["single_band"]["rel_err"]),
                "within_band_spread": float(
                    item4_raw["single_band"]["within_band_spread"]
                ),
                "ok": bool(single_band_ok),
            },
            "m8": {
                "abs_err": float(item4_raw["m8"]["abs_err"]),
                "ok": bool(m8_ok),
            },
            "ok": bool(item4_ok),
        }

        # ---- Item 5 ----
        print("[C1C2-SUP] item 5 — unequal partition ...")
        t0 = time.perf_counter()
        item5_raw = _run_item5(cfg)
        item5_dt = time.perf_counter() - t0
        ctx.record_step_time(item5_dt)
        cv_ok_5 = item5_raw["cv_max"] <= cfg.tol_exact
        q_err_ok_5 = item5_raw["q_err_max"] <= cfg.tol_exact
        item5_ok = cv_ok_5 and q_err_ok_5
        print(
            f"[C1C2-SUP] item 5 done in {item5_dt:.1f} s — "
            f"cv_max = {item5_raw['cv_max']:.3e} "
            f"({'OK' if cv_ok_5 else 'FAIL'})  "
            f"q_err_max = {item5_raw['q_err_max']:.3e} "
            f"({'OK' if q_err_ok_5 else 'FAIL'})"
        )
        item5_block = {
            "block_sizes": list(item5_raw["partition"].sizes),
            "L": int(item5_raw["L"]),
            "cv_max": float(item5_raw["cv_max"]),
            "q_err_max": float(item5_raw["q_err_max"]),
            "cv_ok": bool(cv_ok_5),
            "q_err_ok": bool(q_err_ok_5),
            "ok": bool(item5_ok),
        }

        # ---- Figures ----
        _plot_item1(cfg, item1_raw, run)
        _plot_item2(cfg, item23, run)
        _plot_item4(cfg, item4_raw, run, partition_main)
        _plot_item5(cfg, item5_raw, run)

        # ---- NPZ payload ----
        npz_payload: dict[str, np.ndarray] = {
            "Lambda": lam.detach().cpu().numpy(),
            "Omega": omega.detach().cpu().numpy(),
            "main_block_sizes": np.asarray(
                partition_main.sizes, dtype=np.int64
            ),
            "L_list": np.asarray(cfg.L_list, dtype=np.int64),
        }
        for L in cfg.L_list:
            data = item23["per_L"][int(L)]
            npz_payload[f"L{int(L)}__checkpoints"] = data["checkpoints"]
            npz_payload[f"L{int(L)}__matrix_loss"] = data["matrix_loss"]
            npz_payload[f"L{int(L)}__grouped_loss"] = data["grouped_loss"]
            npz_payload[f"L{int(L)}__rel_err_loss"] = data["rel_err_loss"]
            npz_payload[f"L{int(L)}__metric_lhs"] = data["metric_lhs"]
            npz_payload[f"L{int(L)}__metric_rhs"] = data["metric_rhs"]
            npz_payload[f"L{int(L)}__rel_err_metric"] = data[
                "rel_err_metric"
            ]
        npz_payload["item4_singleton_gamma_final"] = (
            item4_raw["singleton"]["gamma_matrix"][-1].numpy()
        )
        npz_payload["item4_singleton_q_per_mode_final"] = (
            item4_raw["singleton"]["q_per_mode"][-1].numpy()
        )
        npz_payload["item4_single_band_q_matrix"] = (
            item4_raw["single_band"]["q_matrix"].numpy()
        )
        npz_payload["item4_single_band_q_ode"] = (
            item4_raw["single_band"]["q_ode"].numpy()
        )
        npz_payload["item4_m8_q_mat_final"] = (
            item4_raw["m8"]["q_matrix"][-1].numpy()
        )
        npz_payload["item4_m8_q_ode_final"] = (
            item4_raw["m8"]["q_ode"][-1].numpy()
        )
        npz_payload["item5_lam"] = item5_raw["lam"].numpy()
        npz_payload["item5_omega"] = item5_raw["omega"].numpy()
        npz_payload["item5_block_sizes"] = np.asarray(
            item5_raw["partition"].sizes, dtype=np.int64
        )
        npz_payload["item5_cv_r"] = item5_raw["cv_r"].numpy()
        npz_payload["item5_q_mat_final"] = item5_raw["q_mat"][-1].numpy()
        npz_payload["item5_q_ode_final"] = item5_raw["q_ode"][-1].numpy()
        np.savez_compressed(run.npz_path("c1c2_supplement"), **npz_payload)

        # ---- Per-item JSON + human-readable summary ----
        per_item = {
            "item1": item1_block,
            "item2": item2_block,
            "item3": item3_block,
            "item4": item4_block,
            "item5": item5_block,
        }
        all_ok = (
            item1_block["all_ok"]
            and item2_block["ok"]
            and item3_block["ok"]
            and item4_block["ok"]
            and item5_block["ok"]
        )
        per_item["all_ok"] = bool(all_ok)

        (run.root / "per_item_summary.json").write_text(
            json.dumps(per_item, indent=2) + "\n", encoding="utf-8"
        )
        supp_path = _write_supplement_summary(run, cfg, per_item)

        # ---- RunContext summary ----
        total_wall = time.perf_counter() - t_overall
        ctx.record_compute_proxy(float(total_wall))
        ctx.record_extra("worst_item1_abs_diff", float(worst_diff))
        ctx.record_extra(
            "worst_item2_rel_err", float(item23["item2_max_rel_err"])
        )
        ctx.record_extra(
            "worst_item3_rel_err", float(item23["item3_max_rel_err"])
        )
        ctx.record_extra("item4", item4_block)
        ctx.record_extra("item5", item5_block)

        status = (
            "item1_" + ("ok" if item1_block["all_ok"] else "fail")
            + "+item2_" + ("ok" if item2_block["ok"] else "fail")
            + "+item3_" + ("ok" if item3_block["ok"] else "fail")
            + "+item4_" + ("ok" if item4_block["ok"] else "fail")
            + "+item5_" + ("ok" if item5_block["ok"] else "fail")
        )
        ctx.write_summary(
            {
                "plan_reference": (
                    "EXPERIMENT_PLAN_FINAL.MD §7.1–§7.2 (supplements the main "
                    "C1+C2 script)"
                ),
                "theorem_reference": (
                    "thesis/theorem_c.txt: Lemma 3.5 (band-rotation "
                    "invariance of the population loss), Theorem 3.8 "
                    "(exact grouped closure, grouped loss formula eq. 3.2, "
                    "induced metric, grouped gradient-flow ODE eq. 3.3), "
                    "Corollary 3.9 (endpoint recovery: singletons ↔ "
                    "modewise; single band ↔ scalar isotropic)."
                ),
                "category": (
                    "operator-level exact theorem-C supplement — no learned "
                    "architecture. Items 2/3/4/5 are algebraic "
                    "(machine-precision acceptance); item 1 is a paired-MC "
                    "Lemma-3.5 test with tolerance "
                    "item1_tol_sigmas · std(paired_diff) / sqrt(N_mc)."
                ),
                "interpretation": (
                    "Item 1 confirms Lemma 3.5 (band-rotation invariance of "
                    "E_L) holds for both Q ∈ C(B) and Q ∉ C(B) under paired "
                    "MC over N_mc block-Haar samples, at ~5σ confidence. "
                    "Item 2 confirms that equation (3.2) is the correct "
                    "closed form of the matrix population loss for "
                    "block-scalar Q, by independent P×P matrix-product "
                    "evaluation vs direct modewise sum. Item 3 confirms "
                    "the Frobenius induced-metric identity ‖Q‖_F² = "
                    "Σ_b m_b q_b² from Theorem 3.8 step 5. Item 4 confirms "
                    "Corollary 3.9 endpoint recovery: the band-RRS "
                    "grouped machinery specializes to Theorem-B modewise "
                    "dynamics at M = P and to scalar isotropic dynamics at "
                    "M = 1. Item 5 confirms the theorem holds for a "
                    "non-uniform block partition (4,4,8,16,32)."
                ),
                "device": str(device),
                "D": cfg.D,
                "main_partition_m": cfg.main_partition_m,
                "n_blocks_main": partition_main.n_blocks,
                "L_list": list(cfg.L_list),
                "T": cfg.T,
                "eta": cfg.eta,
                "n_checkpoints": cfg.n_checkpoints,
                "item1_n_U": cfg.item1_n_U,
                "item1_n_mc": cfg.item1_n_mc,
                "item1_tol_sigmas": cfg.item1_tol_sigmas,
                "item4_L": cfg.item4_L,
                "item5_block_sizes": list(cfg.item5_block_sizes),
                "tol_exact": cfg.tol_exact,
                "status": status,
                "all_ok": bool(all_ok),
                "worst_item1_abs_diff": float(worst_diff),
                "worst_item2_rel_err": float(
                    item23["item2_max_rel_err"]
                ),
                "worst_item3_rel_err": float(
                    item23["item3_max_rel_err"]
                ),
                "item4": item4_block,
                "item5": item5_block,
                "total_wallclock_seconds": round(float(total_wall), 3),
                "supplement_summary_path": str(supp_path),
            }
        )

        # ---- Top-line print ----
        print()
        print("=" * 72)
        print(f" C1+C2 supplement on {device}")
        print(
            f"   Item 1 (Lemma 3.5)   worst |diff|   = {worst_diff:.3e}  "
            f"{'OK' if item1_block['all_ok'] else 'FAIL'}"
        )
        print(
            f"   Item 2 (eq. 3.2)    worst rel err  = "
            f"{item23['item2_max_rel_err']:.3e}  "
            f"{'OK' if item2_block['ok'] else 'FAIL'} "
            f"(tol = {cfg.tol_exact:.0e})"
        )
        print(
            f"   Item 3 (metric)     worst rel err  = "
            f"{item23['item3_max_rel_err']:.3e}  "
            f"{'OK' if item3_block['ok'] else 'FAIL'} "
            f"(tol = {cfg.tol_exact:.0e})"
        )
        print(
            f"   Item 4 (Cor 3.9)    "
            f"singletons = {item4_block['singleton']['abs_err']:.3e}  "
            f"single-band = {item4_block['single_band']['abs_err']:.3e}  "
            f"m=8 = {item4_block['m8']['abs_err']:.3e}  "
            f"{'OK' if item4_block['ok'] else 'FAIL'}"
        )
        print(
            f"   Item 5 (unequal)    cv_max = {item5_block['cv_max']:.3e}  "
            f"q_err_max = {item5_block['q_err_max']:.3e}  "
            f"{'OK' if item5_block['ok'] else 'FAIL'}"
        )
        print()
        print(f"   supplement summary: {supp_path}")
        print("=" * 72)

        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
