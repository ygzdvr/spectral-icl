"""§9.2 architecture-aligned structured-mask / SSD suite on band-RRS data.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §9.2. Theorem references:
``thesis/theorem_a.txt`` Proposition 3 (SSD realization) and
``thesis/theorem_c.txt`` Theorem 1 (grouped commutant closure),
Corollary (grouped ODE / grouped-loss formula), Proposition 4
(finite-depth grouped closure).

Role
----
This is the first §9.2 (structured-mask) architecture-aligned script.
Two purposes:

1. Verify G2 **sampled-context** mode works correctly when paired with a
   trainable reduced-Γ model (G2 sampled is used here for the first
   time; any bugs surface now rather than in §9.3 hybrid experiments).
2. Test theorem-C grouped-closure predictions on band-RRS sampled
   contexts using a **spectral-only** reduced-Γ model. The block-Haar
   rotation symmetry of the population band-RRS loss forces the
   learned γ into the commutant class (block-scalar structure per
   Fourier-basis band). This experiment asks whether the trainable
   architecture, under gradient flow on sampled-context band-RRS data,
   reproduces the theorem-C population prediction.

Model
-----
Reduced-Γ GD-compatible forward pass, with a single tied
``γ ∈ ℝ^D`` parameterizing the circulant ``Γ = F^T · diag(γ) · F``.
Mathematically equivalent to the SSD realization from Proposition 3
(theorem-A tier already verified the algebraic equivalence in A1–A4;
this script trains the reduced form directly). For each batch of
sampled contexts ``X ∈ ℝ^{B × D × P}``, ``X_query ∈ ℝ^{B × D × K}``:

    G_c       = (1 / P) · (F · X_c)^T · diag(γ) · (F · X_c)         ∈ ℝ^{P × P}
    G_c^⋆    = (1 / P) · (F · X_c^⋆)^T · diag(γ) · (F · X_c)         ∈ ℝ^{K × P}
    r^ℓ+1     = r^ℓ − (1 / L) · G_c · r^ℓ,                           r^0 = y_train
    f         = (1 / L) · G_c^⋆ · Σ_{ℓ=0..L−1} r^ℓ
    MSE       = ‖f − y_query‖² / (B · K)

Parameter count is ``D`` regardless of ``L``. Train with Adam.

Data (G2 sampled-context, band-RRS)
-----------------------------------
Per training step, a **fresh** batch of ``B`` contexts is sampled. Each
context draws its own block-Haar rotation ``R_c = ⊕_b R_{c,b}`` where
``R_{c,b} ∼ Haar(O(m_b))`` independently. The feature-space covariance

    Σ_{R_c} = F^T · R_c · diag(Λ) · R_c^T · F

is used for BOTH the training positions (P) and the query positions
(K) — same context, same rotation. Teacher ``β_c ∼ N(0, F^T · R_c ·
diag(Ω) · R_c^T · F)`` (same rotation for β as for X). Labels noiseless:
``y = β^T · x / √D`` per position.

This is the load-bearing mechanism of theorem C: the rotation changes
PER context, so the model must learn a γ that works across ALL
rotations in expectation. The population loss is band-rotation
invariant, and gradient flow from ``γ(0) = 0`` therefore lives in the
block-commutant class.

Sweep
-----
``D = P = 64``, ``K = 16``, ``L = 4`` (fixed depth; no depth sweep in
this script — depth irrelevance was addressed by §9.1).

Band size ``m ∈ {1, 2, 4, 8, 16}`` (bands of ``M = D/m ∈ {64, 32, 16,
8, 4}``). Within-band condition number ``κ ∈ {1.0, 1.5, 2.0, 3.0, 5.0,
10.0}``. 4 seeds per ``(m, κ)`` cell.

Spectrum construction (via G2's mass-preserving convention):
``λ_{b,j} = κ_b^{ξ_j} / mean_u(κ_b^{ξ_u})``, ``ξ_j`` linear within each
block. This preserves the block mean at ``1.0`` for every κ and gives
within-block ``λ_max / λ_min = κ`` exactly. Task weight ``ω_i = 1`` for
all modes.

Training: Adam, lr = 1e-2, batch = 64, 30000 steps, σ = 0 (noiseless),
label_norm = sqrt_D.

Theorem-C grouped-closure prediction
------------------------------------
For each block ``B_b`` with eigenvalues ``{λ_i}_{i ∈ B_b}`` and weights
``{ω_i}_{i ∈ B_b}``, the Proposition-4 grouped-closure optimum is

    q_b^⋆ = arg min_q  Σ_{i ∈ B_b} ω_i · λ_i · (1 − λ_i · q / L)^{2L}
    E_b^⋆ = Σ_{i ∈ B_b} ω_i · λ_i · (1 − λ_i · q_b^⋆ / L)^{2L}
    E⋆     = (1 / D) · Σ_b E_b^⋆

Computed per (m, κ) cell via scipy.optimize.minimize_scalar.

Primary figures
---------------
1. ``heterogeneity_phase_diagram`` (HEADLINE) — two heatmaps side by
   side: architecture final loss vs (m, κ), and theorem-C grouped
   optimum E⋆ vs (m, κ), shared color scale.
2. ``loss_vs_kappa_at_fixed_m`` — line plot, one curve per m,
   theorem-C dashed overlay.
3. ``loss_vs_step_selected_cells`` — convergence curves at
   (m=1, κ=1), (m=4, κ=3), (m=16, κ=10).
4. ``learned_filter_band_structure`` — γ_k vs mode k, colored by band.
   At κ=1 should approximate L/λ_k; at κ > 1 with m > 1 should exhibit
   block-scalar structure.
5. ``stationary_bridge_m1`` — relative error |γ − L/λ_k| / |L/λ_k| at
   m=1 across κ. Validates G2 + forward pass.

Acceptance (qualitative)
------------------------
1. No NaN across all 120 cells.
2. Stationary bridge at (m=1, κ=1): mean final loss ≤ 0.05.
3. Heterogeneity ordering: at each m ≥ 2, final loss non-decreasing
   in κ (for κ ≥ 2).
4. Band-size ordering: at each κ ≥ 2, final loss non-decreasing in m.
5. Theory-architecture correlation: Spearman ρ ≥ 0.7 between arch
   final loss and theorem-C E⋆.
6. Learned-filter bridge: at (m=1, κ=1), relative error
   ``|γ_k − L/λ_k| / (L/λ_k) ≤ 0.3`` per mode.
7. Block-scalar structure: at m ≥ 4, κ ≥ 3, the within-band std of
   learned γ divided by the between-band-means std is ≤ 0.3.

Process
-------
Sampled on-GPU to avoid CPU↔GPU transfer per step. 30k steps × 120
cells at D=P=64, L=4; SLURM 2h (may require rerun at reduced steps).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from scripts.thesis.utils.data_generators import G2Config, g2_generate_operator
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
class ArchSSDBandRRSConfig:
    # Geometry
    D: int = 64
    P: int = 64
    K: int = 16
    L: int = 4

    # Sweep
    m_list: tuple[int, ...] = (1, 2, 4, 8, 16)
    kappa_list: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    seed_list: tuple[int, ...] = (0, 1, 2, 3)

    # Training
    # 10k (bumped DOWN from 30k) — the pool-based Haar sampler brings
    # step cost to ~2.7 ms, but 30k × 120 cells still exceeds the 2h
    # SLURM wall. 10k is enough for the grouped-closure phase-diagram
    # pattern matching (which is §9.2's target), and (m=1, κ=1)
    # stationary bridge already converges by then.
    train_steps: int = 10000
    # Pool size: number of block-Haar matrices precomputed per cell,
    # sampled uniformly per step. 4096 >> batch_contexts so statistical
    # independence is preserved over 30k·batch_contexts sample draws.
    haar_pool_size: int = 4096
    batch_contexts: int = 64
    learning_rate: float = 1e-2
    optimizer: str = "adam"
    weight_decay: float = 0.0
    log_every: int = 100
    label_norm: str = "sqrt_D"
    sigma: float = 0.0
    final_loss_window: int = 20
    init_scale: float = 0.0
    spectral_basis_kind: str = "dct2"

    # Selected cells for diagnostic figures
    convergence_viz_cells: tuple[tuple[int, float], ...] = (
        (1, 1.0), (4, 3.0), (16, 10.0),
    )
    filter_viz_cells: tuple[tuple[int, float], ...] = (
        (1, 1.0), (1, 10.0), (4, 3.0), (8, 3.0), (16, 10.0),
    )

    # Acceptance thresholds
    stationary_bridge_loss_thresh: float = 0.05
    filter_bridge_rel_err_thresh: float = 0.3
    block_scalar_ratio_thresh: float = 0.3
    spearman_rho_min: float = 0.7

    # Misc
    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# On-GPU block-Haar sampling (QR on Gaussian with diagonal-sign correction)
# ---------------------------------------------------------------------------


def _gpu_block_haar_batch(
    n_contexts: int, block_sizes: list[int],
    gen: torch.Generator, dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Return (n_contexts, D, D) block-diagonal orthogonal matrices sampled
    on the target device. Each block is Haar-uniform on O(m_b) via QR of
    a Gaussian with diagonal-sign correction of R_qr.
    """
    D = int(sum(block_sizes))
    if not block_sizes:
        return torch.zeros(n_contexts, 0, 0, dtype=dtype, device=device)

    all_same = len(set(block_sizes)) == 1
    if all_same:
        m = int(block_sizes[0])
        n_blocks = len(block_sizes)
        if m == 1:
            sign = (
                torch.randint(
                    0, 2, (n_contexts, D),
                    generator=gen, device=device,
                ).to(dtype) * 2.0 - 1.0
            )
            return torch.diag_embed(sign)
        # Batched QR: one call for all (n_contexts · n_blocks) blocks.
        A = torch.randn(
            n_contexts * n_blocks, m, m,
            generator=gen, dtype=dtype, device=device,
        )
        Q, R_qr = torch.linalg.qr(A)
        diag_R = torch.diagonal(R_qr, dim1=-2, dim2=-1)
        sign = torch.where(
            diag_R == 0, torch.ones_like(diag_R), diag_R.sign(),
        )
        Q = Q * sign.unsqueeze(-2)                          # (n·b, m, m)
        Q = Q.view(n_contexts, n_blocks, m, m)
        # Assemble block-diagonal via index_put (contiguous blocks).
        R = torch.zeros(n_contexts, D, D, dtype=dtype, device=device)
        for b in range(n_blocks):
            R[:, b * m:(b + 1) * m, b * m:(b + 1) * m] = Q[:, b]
        return R

    # Fallback: unequal block sizes (not used in this script's default
    # sweep, but preserved for flexibility).
    R = torch.zeros(n_contexts, D, D, dtype=dtype, device=device)
    offset = 0
    for m_b in block_sizes:
        if m_b == 1:
            sign = (
                torch.randint(0, 2, (n_contexts, 1, 1),
                              generator=gen, device=device).to(dtype) * 2.0 - 1.0
            )
            R[:, offset:offset + 1, offset:offset + 1] = sign
        else:
            A = torch.randn(n_contexts, m_b, m_b, generator=gen, dtype=dtype, device=device)
            Q, R_qr = torch.linalg.qr(A)
            diag_R = torch.diagonal(R_qr, dim1=-2, dim2=-1)
            sign = torch.where(
                diag_R == 0, torch.ones_like(diag_R), diag_R.sign()
            )
            Q = Q * sign.unsqueeze(-2)
            R[:, offset:offset + m_b, offset:offset + m_b] = Q
        offset += m_b
    return R


def _build_haar_pool(
    pool_size: int, block_sizes: list[int],
    gen: torch.Generator, dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Precompute a pool of ``pool_size`` block-Haar matrices on-device.
    Returned tensor has shape ``(pool_size, D, D)``. Per step, draw a
    uniformly-random batch of indices into this pool — dramatically
    faster than running ``torch.linalg.qr`` every step."""
    return _gpu_block_haar_batch(pool_size, block_sizes, gen, dtype, device)


def _sample_g2_batch_from_pool(
    pool: torch.Tensor,                 # (pool_size, D, D)
    Lambda: torch.Tensor,               # (D,)
    Omega: torch.Tensor,                # (D,)
    F: torch.Tensor,                    # (D, D) real orthogonal spectral basis
    P: int, K: int, B: int,
    norm_factor: float,
    gen: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of G2 band-RRS contexts using a precomputed
    Haar pool. Returns (X_train, X_query, y_train, y_query) with shapes
    (B, D, P), (B, D, K), (B, P), (B, K). The same R_c is used for
    X_train, X_query, and β per context."""
    D = int(Lambda.shape[0])
    idx = torch.randint(0, int(pool.shape[0]), (B,), generator=gen, device=device)
    R = pool.index_select(0, idx)                               # (B, D, D)
    lam_sqrt = Lambda.sqrt().to(dtype=dtype, device=device)
    z = torch.randn(B, D, P + K, generator=gen, dtype=dtype, device=device)
    z = z * lam_sqrt.view(1, D, 1)
    tilde_x = torch.einsum("bij,bjn->bin", R, z)                # (B, D, P+K)
    x = torch.einsum("ij,bjn->bin", F.T, tilde_x)

    om_sqrt = Omega.sqrt().to(dtype=dtype, device=device)
    z_b = torch.randn(B, D, generator=gen, dtype=dtype, device=device)
    z_b = z_b * om_sqrt.view(1, D)
    tilde_b = torch.einsum("bij,bj->bi", R, z_b)
    beta = torch.einsum("ij,bj->bi", F.T, tilde_b)

    y_full = torch.einsum("bd,bdn->bn", beta, x) / float(math.sqrt(norm_factor))
    return (
        x[:, :, :P].contiguous(),
        x[:, :, P:P + K].contiguous(),
        y_full[:, :P].contiguous(),
        y_full[:, P:].contiguous(),
    )


# Kept for backward compatibility with the smoke-test script; not used
# in the main training loop (too slow per step).
def _sample_g2_batch(
    block_sizes: list[int],
    Lambda: torch.Tensor,
    Omega: torch.Tensor,
    F: torch.Tensor,
    P: int, K: int, B: int,
    norm_factor: float,
    gen: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-pooled variant (fresh Haar per call). Retained for tests."""
    D = int(Lambda.shape[0])
    R = _gpu_block_haar_batch(B, block_sizes, gen, dtype, device)
    lam_sqrt = Lambda.sqrt().to(dtype=dtype, device=device)
    z = torch.randn(B, D, P + K, generator=gen, dtype=dtype, device=device)
    z = z * lam_sqrt.view(1, D, 1)
    tilde_x = torch.einsum("bij,bjn->bin", R, z)
    x = torch.einsum("ij,bjn->bin", F.T, tilde_x)
    om_sqrt = Omega.sqrt().to(dtype=dtype, device=device)
    z_b = torch.randn(B, D, generator=gen, dtype=dtype, device=device)
    z_b = z_b * om_sqrt.view(1, D)
    tilde_b = torch.einsum("bij,bj->bi", R, z_b)
    beta = torch.einsum("ij,bj->bi", F.T, tilde_b)
    y_full = torch.einsum("bd,bdn->bn", beta, x) / float(math.sqrt(norm_factor))
    return (
        x[:, :, :P].contiguous(),
        x[:, :, P:P + K].contiguous(),
        y_full[:, :P].contiguous(),
        y_full[:, P:].contiguous(),
    )


# ---------------------------------------------------------------------------
# Reduced-Γ spectral-only model (GD-compatible forward)
# ---------------------------------------------------------------------------


class ReducedGammaSpectralFilter(nn.Module):
    """Single tied γ ∈ ℝ^D parameterizing circulant Γ = F^T diag(γ) F.

    The GD-compatible reduced-Γ recursion operates on per-context X:

        G = (1/P) (F·X)^T diag(γ) (F·X)
        r^{ℓ+1} = r^ℓ − (1/L) G r^ℓ,     r^0 = y_train
        f       = (1/L) G^⋆ Σ_{ℓ=0..L−1} r^ℓ

    where G^⋆ = (1/P) (F·X_query)^T diag(γ) (F·X).
    """

    def __init__(
        self, D: int, F: torch.Tensor, L: int, *,
        init_scale: float = 0.0, dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.D = int(D)
        self.L = int(L)
        if float(init_scale) == 0.0:
            init = torch.zeros(self.D, dtype=dtype)
        else:
            init = float(init_scale) * torch.randn(self.D, dtype=dtype)
        self.gamma = nn.Parameter(init)
        self.register_buffer("F", F.to(dtype))

    def forward(
        self, X: torch.Tensor, X_query: torch.Tensor, y_train: torch.Tensor,
    ) -> torch.Tensor:
        """X:(B,D,P), X_query:(B,D,K), y_train:(B,P) → (B,K).

        Feature-space forward — avoids forming the (B, P, P) matrix G.
        Cost per step is O(L · B · D · P), linear in P (vs quadratic if
        G were assembled). At P = 256 this is ~16× cheaper.

        Identity used: for any r ∈ ℝ^P,
            (X̃^T diag(γ) X̃) · r  =  X̃^T · (γ ⊙ (X̃ · r)).
        Each layer therefore needs two D×P einsums and one (D,)-shaped
        element-wise multiply, not a (P, P) matmul.
        """
        inv_L = 1.0 / float(self.L)
        P = int(X.shape[2])
        inv_LP = 1.0 / (float(self.L) * float(P))
        tilde_X = torch.einsum("ij,bjn->bin", self.F, X)           # (B, D, P)
        tilde_Xq = torch.einsum("ij,bjn->bin", self.F, X_query)    # (B, D, K)

        r = y_train
        accumulator = torch.zeros_like(y_train)
        for _ell in range(self.L):
            accumulator = accumulator + r
            # z = tilde_X · r       shape (B, D), cost O(BDP)
            z = torch.einsum("bdn,bn->bd", tilde_X, r)
            z = self.gamma.view(1, -1) * z                          # (B, D)
            # g = tilde_X^T · z     shape (B, P), cost O(BDP)
            g = torch.einsum("bdn,bd->bn", tilde_X, z)
            r = r - g * inv_LP

        # f = (1/(L·P)) · tilde_Xq^T · diag(γ) · tilde_X · accumulator
        z = torch.einsum("bdn,bn->bd", tilde_X, accumulator)         # (B, D)
        z = self.gamma.view(1, -1) * z                                # (B, D)
        f = torch.einsum("bdk,bd->bk", tilde_Xq, z) * inv_LP          # (B, K)
        return f

    def final_gamma(self) -> torch.Tensor:
        return self.gamma.detach().clone()


# ---------------------------------------------------------------------------
# Theorem-C grouped-closure optimum (Proposition 4)
# ---------------------------------------------------------------------------


def _golden_section_minimize(
    f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200,
) -> tuple[float, float]:
    """Minimize unimodal f on [a, b] by golden-section search. Returns (x⋆, f(x⋆))."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618
    resphi = 1.0 - phi                   # ≈ 0.382
    x1 = a + resphi * (b - a)
    x2 = a + phi * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if f1 < f2:
            b = x2
            x2, f2 = x1, f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = a + phi * (b - a)
            f2 = f(x2)
    if f1 < f2:
        return x1, f1
    return x2, f2


def _grouped_closure_E_star(
    Lambda: np.ndarray, Omega: np.ndarray,
    block_sizes: list[int], L: int,
) -> tuple[float, np.ndarray, list[float]]:
    """Compute theorem-C grouped-closure optimum E⋆ and per-block q_b⋆.

    Returns (E_star_total, E_star_per_block, q_star_per_block).
    E_star_total = (1/D) Σ_b E_b⋆. Inner minimization via golden-section
    on a bounded interval [0, 4 L / λ_max] (safe for unimodal block
    objective).
    """
    D = int(Lambda.shape[0])
    L = int(L)
    E_per_block: list[float] = []
    q_per_block: list[float] = []
    offset = 0
    for m in block_sizes:
        idx = np.arange(offset, offset + m, dtype=int)
        lam = Lambda[idx]
        om = Omega[idx]
        lam_max = float(lam.max()) if lam.size else 1.0

        def f_block(q: float, lam=lam, om=om, L=L) -> float:
            residual = 1.0 - lam * float(q) / float(L)
            return float(np.sum(om * lam * residual ** (2 * L)))

        q_star, E_b_star = _golden_section_minimize(
            f_block, 0.0, 4.0 * L / max(lam_max, 1e-30),
        )
        E_per_block.append(float(E_b_star))
        q_per_block.append(float(q_star))
        offset += m
    E_star_total = float(np.sum(E_per_block) / float(D))
    return E_star_total, np.asarray(E_per_block, dtype=float), q_per_block


# ---------------------------------------------------------------------------
# G2 operator setup per (m, κ) cell
# ---------------------------------------------------------------------------


def _build_g2_operator(
    cfg: ArchSSDBandRRSConfig, m: int, kappa: float,
) -> dict[str, Any]:
    """Build (Lambda, Omega, F, block_sizes) for a given (m, κ) cell via
    G2's public operator API. ω uniform (mean 1, κ 1 per block). λ has
    within-band κ and block mean 1."""
    if cfg.D % m != 0:
        raise ValueError(f"D={cfg.D} must be divisible by m={m}")
    n_blocks = cfg.D // m
    block_means_lam = tuple(1.0 for _ in range(n_blocks))
    block_kappas_lam = tuple(float(kappa) for _ in range(n_blocks))
    block_means_omega = tuple(1.0 for _ in range(n_blocks))
    block_kappas_omega = tuple(1.0 for _ in range(n_blocks))
    g2_cfg = G2Config(
        D=int(cfg.D),
        partition_kind="equal",
        partition_params={"m": int(m)},
        block_means_lam=block_means_lam,
        block_kappas_lam=block_kappas_lam,
        block_means_omega=block_means_omega,
        block_kappas_omega=block_kappas_omega,
        xi_shape="linear",
        spectral_basis_kind=cfg.spectral_basis_kind,
        label_norm=cfg.label_norm,
        sigma=float(cfg.sigma),
        seeds={"R": 0, "x": 1, "beta": 2, "noise": 3},  # dummies
        dtype=cfg.dtype,
    )
    op = g2_generate_operator(g2_cfg)
    partition = op["partition"]
    block_sizes = [len(b) for b in partition.blocks]
    return {
        "Lambda": op["Lambda"],
        "Omega": op["Omega"],
        "F": op["F"],
        "block_sizes": block_sizes,
        "partition": partition,
    }


# ---------------------------------------------------------------------------
# Per-cell training
# ---------------------------------------------------------------------------


def _train_one(
    cfg: ArchSSDBandRRSConfig,
    m: int, kappa: float, seed: int,
    Lambda: torch.Tensor, Omega: torch.Tensor, F: torch.Tensor,
    block_sizes: list[int],
    device: torch.device,
) -> dict[str, Any]:
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    norm_factor = float(cfg.D) if cfg.label_norm == "sqrt_D" else float(cfg.P)
    torch.manual_seed(int(seed) * 997 + 17 * int(m) + 31 * int(1000 * kappa))
    model = ReducedGammaSpectralFilter(
        D=cfg.D, F=F, L=cfg.L,
        init_scale=float(cfg.init_scale), dtype=dtype,
    ).to(device)
    if cfg.optimizer == "adam":
        opt = torch.optim.Adam(
            model.parameters(), lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
    else:
        raise ValueError(f"unknown optimizer: {cfg.optimizer!r}")

    gen = torch.Generator(device=device)
    gen.manual_seed(
        int(seed) * 7919 + 31 * int(m) + 59 * int(1000 * kappa) + 137
    )

    Lambda_dev = Lambda.to(dtype=dtype, device=device)
    Omega_dev = Omega.to(dtype=dtype, device=device)

    # Precompute a per-cell Haar pool once (QR is slow; pool indexing
    # is fast). Use a separate generator for pool construction so the
    # per-step generator state is independent.
    gen_pool = torch.Generator(device=device)
    gen_pool.manual_seed(
        int(seed) * 104729 + 31 * int(m) + 59 * int(1000 * kappa)
    )
    pool = _build_haar_pool(
        int(cfg.haar_pool_size), block_sizes, gen_pool, dtype, device,
    )

    loss_steps: list[int] = []
    loss_values: list[float] = []
    nan_failure = False
    t0 = time.perf_counter()
    for step in range(int(cfg.train_steps)):
        X, X_q, y_tr, y_q = _sample_g2_batch_from_pool(
            pool, Lambda_dev, Omega_dev, model.F,
            cfg.P, cfg.K, int(cfg.batch_contexts),
            norm_factor, gen, dtype, device,
        )
        opt.zero_grad()
        y_pred = model(X, X_q, y_tr)
        loss = ((y_pred - y_q) ** 2).mean()
        if not torch.isfinite(loss):
            nan_failure = True
            print(
                f"   [m={m}, κ={kappa}, seed={seed}] NaN loss at step {step}; "
                "aborting cell."
            )
            break
        loss.backward()
        opt.step()
        if (step % int(cfg.log_every) == 0) or step == cfg.train_steps - 1:
            loss_steps.append(int(step))
            loss_values.append(float(loss.item()))
    t_train = time.perf_counter() - t0
    gamma_final = model.final_gamma().cpu()

    return {
        "m": int(m),
        "kappa": float(kappa),
        "seed": int(seed),
        "loss_steps": loss_steps,
        "loss_values": loss_values,
        "gamma_final": gamma_final,
        "nan_failure": bool(nan_failure),
        "train_seconds": float(t_train),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_heterogeneity_phase_diagram(
    cfg: ArchSSDBandRRSConfig,
    arch_loss_mat: np.ndarray,     # (n_m, n_kappa)
    theory_E_mat: np.ndarray,       # (n_m, n_kappa)
    run_dir: ThesisRunDir,
) -> None:
    """Headline figure: side-by-side architecture vs theory heatmaps."""
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    vmin = float(
        max(min(arch_loss_mat[arch_loss_mat > 0].min(), theory_E_mat[theory_E_mat > 0].min()), 1e-30)
    )
    vmax = float(max(arch_loss_mat.max(), theory_E_mat.max()))
    extent = [0, len(cfg.kappa_list), 0, len(cfg.m_list)]
    imL = axL.imshow(
        arch_loss_mat, origin="lower", aspect="auto",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap="rocket_r", extent=extent,
    )
    imR = axR.imshow(
        theory_E_mat, origin="lower", aspect="auto",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap="rocket_r", extent=extent,
    )
    for ax, im, title in (
        (axL, imL, "architecture final loss"),
        (axR, imR, "theorem-C grouped optimum $E^\\star$"),
    ):
        ax.set_xticks(np.arange(len(cfg.kappa_list)) + 0.5)
        ax.set_xticklabels([f"{k:.1f}" for k in cfg.kappa_list])
        ax.set_yticks(np.arange(len(cfg.m_list)) + 0.5)
        ax.set_yticklabels([str(m) for m in cfg.m_list])
        ax.set_xlabel(r"within-band condition number $\kappa$")
        ax.set_ylabel(r"band size $m$")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        r"§9.2 HEADLINE: band-RRS heterogeneity phase diagram "
        r"(architecture vs theorem-C, D=P=$" + str(cfg.D) + r"$, L=$" + str(cfg.L) + r"$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "heterogeneity_phase_diagram")
    plt.close(fig)


def _plot_loss_vs_kappa_at_fixed_m(
    cfg: ArchSSDBandRRSConfig,
    arch_loss_mat: np.ndarray,
    arch_loss_se_mat: np.ndarray,
    theory_E_mat: np.ndarray,
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    colors = sequential_colors(len(cfg.m_list), palette="rocket")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for color, mi, m in zip(colors, range(len(cfg.m_list)), cfg.m_list):
        mean = arch_loss_mat[mi]
        se = arch_loss_se_mat[mi]
        ax.plot(
            cfg.kappa_list, mean, color=color, lw=1.4, marker="o", ms=4.5,
            label=rf"$m = {m}$ (arch)",
        )
        ax.fill_between(
            cfg.kappa_list, np.clip(mean - se, 1e-30, None), mean + se,
            color=color, alpha=0.18, lw=0,
        )
        ax.plot(
            cfg.kappa_list, theory_E_mat[mi],
            color=color, lw=1.0, ls="--",
        )
    ax.set_xlabel(r"within-band condition number $\kappa$")
    ax.set_ylabel(r"final loss")
    ax.set_yscale("log")
    ax.set_title(
        r"§9.2 loss vs $\kappa$ at fixed $m$ (architecture solid, "
        r"theorem-C $E^\star$ dashed)",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    save_both(fig, run_dir, "loss_vs_kappa_at_fixed_m")
    plt.close(fig)


def _plot_loss_vs_step_selected_cells(
    cfg: ArchSSDBandRRSConfig,
    runs_by_cell_seed: dict[tuple[int, float, int], dict[str, Any]],
    theory_by_cell: dict[tuple[int, float], float],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = sequential_colors(len(cfg.convergence_viz_cells), palette="rocket")
    for color, (m, kap) in zip(colors, cfg.convergence_viz_cells):
        curves = []
        steps_ref = None
        for seed in cfg.seed_list:
            r = runs_by_cell_seed.get((int(m), float(kap), int(seed)))
            if r is None:
                continue
            if steps_ref is None:
                steps_ref = np.asarray(r["loss_steps"], dtype=int)
            curves.append(np.asarray(r["loss_values"], dtype=float))
        if not curves:
            continue
        arr = np.stack(curves, axis=0)
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(
            steps_ref, np.maximum(mean, 1e-30),
            color=color, lw=1.2,
            label=rf"arch $(m={m}, \kappa={kap:.1f})$",
        )
        ax.fill_between(
            steps_ref, np.maximum(mean - se, 1e-30), mean + se,
            color=color, alpha=0.18, lw=0,
        )
        E = theory_by_cell.get((int(m), float(kap)), 0.0)
        if E > 0.0:
            ax.axhline(
                E, color=color, ls="--", lw=0.8,
                label=rf"$E^\star (m={m}, \kappa={kap:.1f})$",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"training step")
    ax.set_ylabel(r"loss")
    ax.set_title("§9.2 loss vs step at selected $(m, \\kappa)$", fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=1)
    fig.tight_layout()
    save_both(fig, run_dir, "loss_vs_step_selected_cells")
    plt.close(fig)


def _plot_learned_filter_band_structure(
    cfg: ArchSSDBandRRSConfig,
    runs_by_cell_seed: dict[tuple[int, float, int], dict[str, Any]],
    operators_by_cell: dict[tuple[int, float], dict[str, Any]],
    q_star_by_cell: dict[tuple[int, float], list[float]],
    run_dir: ThesisRunDir,
) -> None:
    """Plot learned γ_k vs mode k for selected (m, κ) cells, colored by
    band. Overlay per-band q⋆ from theorem-C (dashed horizontal per
    band). At (m=1, κ=1), γ should approximate L/λ_k."""
    import matplotlib.pyplot as plt

    n_cells = len(cfg.filter_viz_cells)
    ncols = min(3, n_cells)
    nrows = int(math.ceil(n_cells / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), sharex=False,
    )
    axes_flat = np.asarray(axes).reshape(-1)
    k_axis = np.arange(cfg.D)

    for idx, (m, kap) in enumerate(cfg.filter_viz_cells):
        ax = axes_flat[idx]
        op = operators_by_cell.get((int(m), float(kap)))
        if op is None:
            ax.axis("off")
            continue
        block_sizes = op["block_sizes"]
        Lambda_np = op["Lambda"].detach().cpu().numpy().astype(float)
        target_stationary = float(cfg.L) / np.maximum(Lambda_np, 1e-30)

        # Per-seed learned γ, then mean across seeds.
        gammas = []
        for seed in cfg.seed_list:
            r = runs_by_cell_seed.get((int(m), float(kap), int(seed)))
            if r is None or r["nan_failure"]:
                continue
            gammas.append(r["gamma_final"].numpy().astype(float))
        if not gammas:
            ax.axis("off")
            continue
        gamma_mean = np.stack(gammas, axis=0).mean(axis=0)

        # Color modes by band.
        n_blocks = len(block_sizes)
        band_colors = sequential_colors(max(n_blocks, 2), palette="rocket")
        offset = 0
        for b, msize in enumerate(block_sizes):
            idxs = np.arange(offset, offset + msize, dtype=int)
            ax.plot(
                idxs, gamma_mean[idxs],
                color=band_colors[b % len(band_colors)],
                lw=0, marker="o", ms=4,
                label=rf"band {b}" if (b < 4 and n_blocks <= 8) else None,
            )
            # theorem-C q_b⋆ horizontal (per band).
            q_stars = q_star_by_cell.get((int(m), float(kap)))
            if q_stars is not None and b < len(q_stars):
                ax.hlines(
                    q_stars[b], offset - 0.5, offset + msize - 0.5,
                    color=band_colors[b % len(band_colors)],
                    lw=1.0, ls="--",
                )
            offset += msize

        # Stationary reference at m=1 κ=1 only.
        if m == 1 and abs(kap - 1.0) < 1e-9:
            ax.plot(
                k_axis, target_stationary,
                color="black", lw=1.0, ls=":", label=r"$L/\lambda_k$",
            )

        ax.set_xlabel(r"mode index $k$")
        ax.set_ylabel(r"learned $\gamma_k$")
        ax.set_title(rf"$m = {m},\ \kappa = {kap:.1f}$", fontsize=10)
        ax.legend(fontsize=7, loc="best")
    for j in range(n_cells, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(
        "§9.2 learned γ vs mode index (band-colored; "
        r"dashed = theorem-C $q_b^\star$ per band)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "learned_filter_band_structure")
    plt.close(fig)


def _plot_alpha_comparison_phase_diagram(
    cfg: ArchSSDBandRRSConfig,
    arch_loss_mat_current: np.ndarray,
    theory_E_mat: np.ndarray,
    alpha1_artifact_path: str,
    run_dir: ThesisRunDir,
) -> None:
    """Three-panel figure: α=1 arch heatmap | α=current arch heatmap |
    theorem-C E⋆ heatmap. All three share a log color scale so the
    emergence of the theorem-C grouped-closure pattern at higher α is
    directly visible.

    Loads the α=1 architecture loss matrix from a prior canonical run's
    ``arch_ssd_band_rrs.npz`` file. If the (m_list, kappa_list) geometry
    disagrees with the current run, the alignment is best-effort — this
    figure is for qualitative comparison.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    try:
        with np.load(alpha1_artifact_path, allow_pickle=True) as raw:
            alpha1_arch = np.asarray(raw["arch_loss_mat"])
            alpha1_m_list = [int(x) for x in np.asarray(raw["m_list"]).tolist()]
            alpha1_kappa_list = [float(x) for x in np.asarray(raw["kappa_list"]).tolist()]
    except Exception as e:
        print(
            f"[alpha-compare] could not load α=1 artifact "
            f"{alpha1_artifact_path}: {e}"
        )
        return

    if (
        alpha1_m_list != list(cfg.m_list)
        or alpha1_kappa_list != list(cfg.kappa_list)
    ):
        print(
            f"[alpha-compare] α=1 geometry {alpha1_m_list}×{alpha1_kappa_list}"
            f" differs from current {list(cfg.m_list)}×{list(cfg.kappa_list)}; "
            "skipping α comparison."
        )
        return

    alpha_current = float(cfg.P) / float(cfg.D)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    axL, axM, axR = axes
    pos_vals = np.concatenate([
        arch_loss_mat_current[arch_loss_mat_current > 0],
        alpha1_arch[alpha1_arch > 0],
        theory_E_mat[theory_E_mat > 0],
    ])
    vmin = float(pos_vals.min()) if pos_vals.size else 1e-30
    vmax = float(pos_vals.max()) if pos_vals.size else 1.0
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)
    cmap = "rocket_r"
    extent = [0, len(cfg.kappa_list), 0, len(cfg.m_list)]
    imL = axL.imshow(
        alpha1_arch, origin="lower", aspect="auto",
        norm=norm, cmap=cmap, extent=extent,
    )
    imM = axM.imshow(
        arch_loss_mat_current, origin="lower", aspect="auto",
        norm=norm, cmap=cmap, extent=extent,
    )
    imR = axR.imshow(
        theory_E_mat, origin="lower", aspect="auto",
        norm=norm, cmap=cmap, extent=extent,
    )
    for ax, im, title in (
        (axL, imL, rf"architecture at $\alpha = 1$"),
        (axM, imM, rf"architecture at $\alpha = {alpha_current:.0f}$"),
        (axR, imR, r"theorem-C $E^\star$ (population)"),
    ):
        ax.set_xticks(np.arange(len(cfg.kappa_list)) + 0.5)
        ax.set_xticklabels([f"{k:.1f}" for k in cfg.kappa_list])
        ax.set_yticks(np.arange(len(cfg.m_list)) + 0.5)
        ax.set_yticklabels([str(m) for m in cfg.m_list])
        ax.set_xlabel(r"within-band condition number $\kappa$")
        ax.set_ylabel(r"band size $m$")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        r"§9.2 α comparison: architecture loss at $\alpha = 1$ vs "
        rf"$\alpha = {alpha_current:.0f}$ vs theorem-C $E^\star$  "
        r"(shared log color scale)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_both(fig, run_dir, "alpha_comparison_phase_diagram")
    plt.close(fig)


def _plot_stationary_bridge_m1(
    cfg: ArchSSDBandRRSConfig,
    runs_by_cell_seed: dict[tuple[int, float, int], dict[str, Any]],
    operators_by_cell: dict[tuple[int, float], dict[str, Any]],
    run_dir: ThesisRunDir,
) -> None:
    """At m=1 only, |γ − L/λ| / (L/λ) per mode across κ values.
    Validates that G2 sampled + reduced-Γ forward reaches the theorem-B
    stationary fixed point when there is no heterogeneity grouping."""
    import matplotlib.pyplot as plt

    k_axis = np.arange(cfg.D)
    colors = sequential_colors(len(cfg.kappa_list), palette="rocket")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for color, kap in zip(colors, cfg.kappa_list):
        op = operators_by_cell.get((1, float(kap)))
        if op is None:
            continue
        Lambda_np = op["Lambda"].detach().cpu().numpy().astype(float)
        target = float(cfg.L) / np.maximum(Lambda_np, 1e-30)
        rel_errs = []
        for seed in cfg.seed_list:
            r = runs_by_cell_seed.get((1, float(kap), int(seed)))
            if r is None or r["nan_failure"]:
                continue
            g = r["gamma_final"].numpy().astype(float)
            rel_errs.append(np.abs(g - target) / np.maximum(np.abs(target), 1e-30))
        if not rel_errs:
            continue
        arr = np.stack(rel_errs, axis=0)
        mean = arr.mean(axis=0)
        ax.plot(
            k_axis, mean, color=color, lw=1.2, marker="o", ms=3.0,
            label=rf"$\kappa = {kap:.1f}$",
        )
    ax.axhline(
        cfg.filter_bridge_rel_err_thresh,
        color="black", ls="--", lw=0.8,
        label=rf"acceptance $\leq {cfg.filter_bridge_rel_err_thresh:.2f}$",
    )
    ax.set_xlabel(r"mode index $k$")
    ax.set_ylabel(r"$|\gamma_k - L/\lambda_k| / (L/\lambda_k)$")
    ax.set_yscale("log")
    ax.set_title(
        "§9.2 stationary bridge diagnostic at $m = 1$ "
        r"(G2 singleton-block limit; all $\kappa$ should look identical)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "stationary_bridge_m1")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI and driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_list_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "§9.2 architecture-aligned structured-mask SSD experiment on "
            "band-RRS sampled contexts. Spectral-only reduced-Γ model."
        )
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda", "auto"))
    p.add_argument("--dtype", type=str, default="float64", choices=("float32", "float64"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--P", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--D", type=int, default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--m-list", type=str, default=None)
    p.add_argument("--kappa-list", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--stationary-bridge-thresh", type=float, default=None)
    p.add_argument("--filter-bridge-thresh", type=float, default=None)
    p.add_argument("--block-scalar-thresh", type=float, default=None)
    p.add_argument("--spearman-rho-min", type=float, default=None)
    p.add_argument(
        "--alpha1-artifact", type=str, default=None,
        help="Absolute path to an existing α=1 arch_ssd_band_rrs.npz "
             "file; when supplied, a three-panel comparison figure is "
             "emitted showing α=1 arch vs α=current arch vs theorem-C E⋆.",
    )
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> ArchSSDBandRRSConfig:
    base = ArchSSDBandRRSConfig()
    o: dict[str, Any] = {"dtype": args.dtype, "device": args.device}
    if args.train_steps is not None:
        o["train_steps"] = int(args.train_steps)
    if args.P is not None:
        o["P"] = int(args.P)
    if args.K is not None:
        o["K"] = int(args.K)
    if args.D is not None:
        o["D"] = int(args.D)
    if args.L is not None:
        o["L"] = int(args.L)
    if args.m_list is not None:
        o["m_list"] = _parse_list_ints(args.m_list)
    if args.kappa_list is not None:
        o["kappa_list"] = _parse_list_floats(args.kappa_list)
    if args.seeds is not None:
        o["seed_list"] = _parse_list_ints(args.seeds)
    if args.stationary_bridge_thresh is not None:
        o["stationary_bridge_loss_thresh"] = float(args.stationary_bridge_thresh)
    if args.filter_bridge_thresh is not None:
        o["filter_bridge_rel_err_thresh"] = float(args.filter_bridge_thresh)
    if args.block_scalar_thresh is not None:
        o["block_scalar_ratio_thresh"] = float(args.block_scalar_thresh)
    if args.spearman_rho_min is not None:
        o["spearman_rho_min"] = float(args.spearman_rho_min)
    return replace(base, **o)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    n = x.size
    if n < 2:
        return 0.0
    mx = rx.mean(); my = ry.mean()
    num = float(np.sum((rx - mx) * (ry - my)))
    den = float(np.sqrt(np.sum((rx - mx) ** 2) * np.sum((ry - my) ** 2)))
    return num / max(den, 1e-30)


def main() -> int:
    args = _cli()
    if args.no_show:
        matplotlib.use("Agg")

    cfg = _config_from_cli(args)
    device = _resolve_device(cfg.device)
    print(f"[arch-§9.2] device = {device}")

    run = ThesisRunDir(__file__, phase="architectures")
    with RunContext(
        run,
        config=cfg,
        seeds=list(cfg.seed_list),
        notes=(
            "§9.2 architecture-aligned structured-mask / SSD experiment on "
            "G2 sampled band-RRS contexts. Spectral-only reduced-Γ forward "
            "with tied weights. Heterogeneity phase-diagram vs theorem-C."
        ),
    ) as ctx:
        apply_thesis_style()

        # ---- Build operator objects per (m, κ) once ------------------
        operators_by_cell: dict[tuple[int, float], dict[str, Any]] = {}
        theory_by_cell: dict[tuple[int, float], float] = {}
        theory_per_block_by_cell: dict[tuple[int, float], np.ndarray] = {}
        q_star_by_cell: dict[tuple[int, float], list[float]] = {}
        for m in cfg.m_list:
            for kap in cfg.kappa_list:
                op = _build_g2_operator(cfg, int(m), float(kap))
                operators_by_cell[(int(m), float(kap))] = op
                Lambda_np = op["Lambda"].detach().cpu().numpy().astype(float)
                Omega_np = op["Omega"].detach().cpu().numpy().astype(float)
                E, E_per_b, q_per_b = _grouped_closure_E_star(
                    Lambda_np, Omega_np, op["block_sizes"], cfg.L,
                )
                theory_by_cell[(int(m), float(kap))] = E
                theory_per_block_by_cell[(int(m), float(kap))] = E_per_b
                q_star_by_cell[(int(m), float(kap))] = q_per_b
                print(
                    f"[arch-§9.2] m={m:>2d} κ={kap:>4.1f}  "
                    f"blocks={len(op['block_sizes'])}  "
                    f"λ range=[{Lambda_np.min():.3e}, {Lambda_np.max():.3e}]  "
                    f"E⋆={E:.3e}"
                )

        # ---- Training sweep ------------------------------------------
        runs_by_cell_seed: dict[tuple[int, float, int], dict[str, Any]] = {}
        n_total = len(cfg.m_list) * len(cfg.kappa_list) * len(cfg.seed_list)
        idx = 0
        t_sweep0 = time.perf_counter()
        for m in cfg.m_list:
            for kap in cfg.kappa_list:
                op = operators_by_cell[(int(m), float(kap))]
                for seed in cfg.seed_list:
                    idx += 1
                    t_cell = time.perf_counter()
                    r = _train_one(
                        cfg, int(m), float(kap), int(seed),
                        op["Lambda"], op["Omega"], op["F"],
                        op["block_sizes"], device,
                    )
                    dt = time.perf_counter() - t_cell
                    ctx.record_step_time(dt)
                    runs_by_cell_seed[(int(m), float(kap), int(seed))] = r
                    tail = r["loss_values"][-cfg.final_loss_window:]
                    final = float(np.mean(tail)) if tail else float("nan")
                    init = float(r["loss_values"][0]) if r["loss_values"] else float("nan")
                    print(
                        f"[{idx:>3d}/{n_total}] "
                        f"m={m:>2d} κ={kap:>4.1f} seed={seed}  "
                        f"init≈{init:.3e}  final≈{final:.3e}  "
                        f"nan={r['nan_failure']}  ({dt:.2f} s)"
                    )
        sweep_wall = time.perf_counter() - t_sweep0

        # ---- Aggregate per cell --------------------------------------
        arch_loss_mat = np.zeros((len(cfg.m_list), len(cfg.kappa_list)), dtype=float)
        arch_loss_se_mat = np.zeros_like(arch_loss_mat)
        theory_E_mat = np.zeros_like(arch_loss_mat)
        per_cell_finals: dict[tuple[int, float], np.ndarray] = {}
        for mi, m in enumerate(cfg.m_list):
            for ki, kap in enumerate(cfg.kappa_list):
                vals = []
                for seed in cfg.seed_list:
                    r = runs_by_cell_seed[(int(m), float(kap), int(seed))]
                    tail = r["loss_values"][-cfg.final_loss_window:]
                    vals.append(float(np.mean(tail)) if tail else float("nan"))
                arr = np.asarray(vals)
                per_cell_finals[(int(m), float(kap))] = arr
                arch_loss_mat[mi, ki] = float(arr.mean())
                if arr.size > 1:
                    arch_loss_se_mat[mi, ki] = float(
                        arr.std(ddof=1) / np.sqrt(arr.size)
                    )
                theory_E_mat[mi, ki] = theory_by_cell[(int(m), float(kap))]

        # ---- Acceptance gates ----------------------------------------
        nan_count = sum(1 for r in runs_by_cell_seed.values() if r["nan_failure"])
        no_nan = nan_count == 0

        # (2) stationary bridge loss at (m=1, κ=1).
        key11 = (1, 1.0)
        stat_bridge_loss = float(np.mean(per_cell_finals[key11]))
        bridge_ok = stat_bridge_loss <= cfg.stationary_bridge_loss_thresh

        # (3) heterogeneity ordering: at each m ≥ 2, non-decreasing in κ for κ ≥ 2.
        heterogeneity_violations: list[dict[str, Any]] = []
        kappa_from2_idx = [
            ki for ki, kap in enumerate(cfg.kappa_list) if kap >= 2.0 - 1e-9
        ]
        for mi, m in enumerate(cfg.m_list):
            if m < 2:
                continue
            for a, b in zip(kappa_from2_idx[:-1], kappa_from2_idx[1:]):
                if arch_loss_mat[mi, b] < arch_loss_mat[mi, a] - 1e-4:
                    heterogeneity_violations.append({
                        "m": int(m),
                        "kappa_lo": float(cfg.kappa_list[a]),
                        "kappa_hi": float(cfg.kappa_list[b]),
                        "loss_lo": float(arch_loss_mat[mi, a]),
                        "loss_hi": float(arch_loss_mat[mi, b]),
                    })
        heterogeneity_ok = not heterogeneity_violations

        # (4) band-size ordering: at each κ ≥ 2, non-decreasing in m.
        band_size_violations: list[dict[str, Any]] = []
        for ki, kap in enumerate(cfg.kappa_list):
            if kap < 2.0 - 1e-9:
                continue
            for a, b in zip(range(len(cfg.m_list) - 1), range(1, len(cfg.m_list))):
                if arch_loss_mat[b, ki] < arch_loss_mat[a, ki] - 1e-4:
                    band_size_violations.append({
                        "kappa": float(kap),
                        "m_lo": int(cfg.m_list[a]),
                        "m_hi": int(cfg.m_list[b]),
                        "loss_lo": float(arch_loss_mat[a, ki]),
                        "loss_hi": float(arch_loss_mat[b, ki]),
                    })
        band_size_ok = not band_size_violations

        # (5) Spearman correlation arch vs theory across all cells.
        x_flat = arch_loss_mat.reshape(-1)
        y_flat = theory_E_mat.reshape(-1)
        spearman = _spearman_rank_corr(x_flat, y_flat)
        correlation_ok = spearman >= cfg.spearman_rho_min

        # (6) learned filter bridge at (m=1, κ=1).
        op11 = operators_by_cell[(1, 1.0)]
        Lambda_np_11 = op11["Lambda"].detach().cpu().numpy().astype(float)
        target_11 = float(cfg.L) / np.maximum(Lambda_np_11, 1e-30)
        per_mode_errs_11 = []
        for seed in cfg.seed_list:
            r = runs_by_cell_seed.get((1, 1.0, int(seed)))
            if r is None or r["nan_failure"]:
                continue
            g = r["gamma_final"].numpy().astype(float)
            rel_err = np.abs(g - target_11) / np.maximum(np.abs(target_11), 1e-30)
            per_mode_errs_11.append(rel_err)
        filter_bridge_max = float(
            np.stack(per_mode_errs_11, axis=0).mean(axis=0).max()
        ) if per_mode_errs_11 else float("inf")
        filter_bridge_ok = filter_bridge_max <= cfg.filter_bridge_rel_err_thresh

        # (7) block-scalar structure at m ≥ 4, κ ≥ 3.
        block_scalar_records: list[dict[str, Any]] = []
        block_scalar_ok = True
        for m in cfg.m_list:
            if m < 4:
                continue
            for kap in cfg.kappa_list:
                if kap < 3.0 - 1e-9:
                    continue
                op = operators_by_cell[(int(m), float(kap))]
                block_sizes = op["block_sizes"]
                # Mean γ across seeds.
                gammas = [
                    runs_by_cell_seed[(int(m), float(kap), int(s))]["gamma_final"].numpy()
                    for s in cfg.seed_list
                    if not runs_by_cell_seed[(int(m), float(kap), int(s))]["nan_failure"]
                ]
                if not gammas:
                    continue
                g_mean = np.stack(gammas, axis=0).mean(axis=0)
                # Within-band std + between-band-means std.
                within_stds = []
                means = []
                offset = 0
                for msize in block_sizes:
                    chunk = g_mean[offset:offset + msize]
                    within_stds.append(float(chunk.std(ddof=0)))
                    means.append(float(chunk.mean()))
                    offset += msize
                mean_within = float(np.mean(within_stds))
                std_between = float(np.std(means, ddof=0))
                ratio = mean_within / max(std_between, 1e-30)
                block_scalar_records.append({
                    "m": int(m), "kappa": float(kap),
                    "mean_within_std": mean_within,
                    "between_band_means_std": std_between,
                    "ratio": ratio,
                })
                if ratio > cfg.block_scalar_ratio_thresh:
                    block_scalar_ok = False

        all_ok = (
            no_nan and bridge_ok and heterogeneity_ok
            and band_size_ok and correlation_ok and filter_bridge_ok
            and block_scalar_ok
        )

        ctx.record_compute_proxy(float(sweep_wall))
        ctx.record_extra("nan_count", int(nan_count))
        ctx.record_extra("stationary_bridge_loss", stat_bridge_loss)
        ctx.record_extra("heterogeneity_violations", heterogeneity_violations[:10])
        ctx.record_extra("band_size_violations", band_size_violations[:10])
        ctx.record_extra("spearman_rho", float(spearman))
        ctx.record_extra("filter_bridge_rel_err_max", float(filter_bridge_max))
        ctx.record_extra(
            "block_scalar_records",
            [{k: float(v) if isinstance(v, (float, int)) else v for k, v in r.items()} for r in block_scalar_records],
        )

        # ---- Figures -------------------------------------------------
        _plot_heterogeneity_phase_diagram(
            cfg, arch_loss_mat, theory_E_mat, run,
        )
        _plot_loss_vs_kappa_at_fixed_m(
            cfg, arch_loss_mat, arch_loss_se_mat, theory_E_mat, run,
        )
        _plot_loss_vs_step_selected_cells(
            cfg, runs_by_cell_seed, theory_by_cell, run,
        )
        _plot_learned_filter_band_structure(
            cfg, runs_by_cell_seed, operators_by_cell, q_star_by_cell, run,
        )
        _plot_stationary_bridge_m1(
            cfg, runs_by_cell_seed, operators_by_cell, run,
        )
        if args.alpha1_artifact:
            _plot_alpha_comparison_phase_diagram(
                cfg, arch_loss_mat, theory_E_mat,
                args.alpha1_artifact, run,
            )

        # ---- NPZ dump ------------------------------------------------
        npz_payload: dict[str, Any] = {
            "D": int(cfg.D), "P": int(cfg.P), "K": int(cfg.K), "L": int(cfg.L),
            "m_list": np.asarray(cfg.m_list, dtype=np.int64),
            "kappa_list": np.asarray(cfg.kappa_list, dtype=np.float64),
            "seed_list": np.asarray(cfg.seed_list, dtype=np.int64),
            "arch_loss_mat": arch_loss_mat,
            "arch_loss_se_mat": arch_loss_se_mat,
            "theory_E_mat": theory_E_mat,
            "spearman_rho": float(spearman),
        }
        for (m, kap), op in operators_by_cell.items():
            key = f"m{m}_k{kap:.2f}"
            npz_payload[f"{key}__Lambda"] = op["Lambda"].detach().cpu().numpy()
            npz_payload[f"{key}__Omega"] = op["Omega"].detach().cpu().numpy()
            npz_payload[f"{key}__block_sizes"] = np.asarray(op["block_sizes"], dtype=np.int64)
            npz_payload[f"{key}__q_star"] = np.asarray(
                q_star_by_cell[(int(m), float(kap))], dtype=np.float64
            )
            npz_payload[f"{key}__E_per_block"] = theory_per_block_by_cell[(int(m), float(kap))]
        for (m, kap, seed), r in runs_by_cell_seed.items():
            key = f"m{m}_k{kap:.2f}_seed{seed}"
            npz_payload[f"{key}__gamma_final"] = r["gamma_final"].numpy()
            npz_payload[f"{key}__loss_steps"] = np.asarray(r["loss_steps"], dtype=np.int64)
            npz_payload[f"{key}__loss_values"] = np.asarray(r["loss_values"], dtype=np.float64)
        np.savez(run.npz_path("arch_ssd_band_rrs"), **npz_payload)

        per_cell_rows = []
        for (m, kap, seed), r in runs_by_cell_seed.items():
            tail = r["loss_values"][-cfg.final_loss_window:]
            final = float(np.mean(tail)) if tail else float("nan")
            init = float(r["loss_values"][0]) if r["loss_values"] else float("nan")
            per_cell_rows.append({
                "m": int(m), "kappa": float(kap), "seed": int(seed),
                "initial_loss": init, "final_loss": final,
                "theorem_C_E_star": theory_by_cell[(int(m), float(kap))],
                "nan_failure": bool(r["nan_failure"]),
                "train_seconds": float(r["train_seconds"]),
            })
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(per_cell_rows, indent=2)
        )

        # ---- Terminal summary ----------------------------------------
        print()
        print("=" * 72)
        print(f" §9.2 architecture-aligned structured-mask/SSD on {device}")
        print(
            f"   N cells = {len(runs_by_cell_seed)}; "
            f"train_steps = {cfg.train_steps}; batch = {cfg.batch_contexts}; "
            f"L = {cfg.L}; D = P = {cfg.D}"
        )
        print(f"   no NaN: {no_nan}  (count = {nan_count})")
        print(
            f"   stationary bridge at (m=1, κ=1): "
            f"loss = {stat_bridge_loss:.3e}  "
            f"(gate ≤ {cfg.stationary_bridge_loss_thresh}: "
            f"{'OK' if bridge_ok else 'WEAK'})"
        )
        print(
            f"   heterogeneity ordering (non-decreasing in κ for κ≥2, m≥2): "
            f"{'OK' if heterogeneity_ok else 'WEAK'}  "
            f"({len(heterogeneity_violations)} violations)"
        )
        print(
            f"   band-size ordering (non-decreasing in m for κ≥2): "
            f"{'OK' if band_size_ok else 'WEAK'}  "
            f"({len(band_size_violations)} violations)"
        )
        print(
            f"   Spearman(arch, theory) ρ = {spearman:.3f}  "
            f"(gate ≥ {cfg.spearman_rho_min}: "
            f"{'OK' if correlation_ok else 'WEAK'})"
        )
        print(
            f"   filter bridge at (m=1, κ=1) max rel err = {filter_bridge_max:.3f}  "
            f"(gate ≤ {cfg.filter_bridge_rel_err_thresh}: "
            f"{'OK' if filter_bridge_ok else 'WEAK'})"
        )
        print(
            f"   block-scalar structure at m≥4, κ≥3: "
            f"{'OK' if block_scalar_ok else 'WEAK'}  "
            f"({len(block_scalar_records)} cells checked)"
        )
        print()
        print("   final loss matrix (rows m, cols κ):")
        header = "     " + "  ".join(f"κ={k:>4.1f}" for k in cfg.kappa_list)
        print(header)
        for mi, m in enumerate(cfg.m_list):
            row = f"   m={m:>2d} " + "  ".join(
                f"{arch_loss_mat[mi, ki]:.3e}" for ki in range(len(cfg.kappa_list))
            )
            print(row)
        print()
        print("   theorem-C E⋆ matrix (rows m, cols κ):")
        print(header)
        for mi, m in enumerate(cfg.m_list):
            row = f"   m={m:>2d} " + "  ".join(
                f"{theory_E_mat[mi, ki]:.3e}" for ki in range(len(cfg.kappa_list))
            )
            print(row)
        print("=" * 72)

        # ---- Final summary -------------------------------------------
        ctx.write_summary({
            "plan_reference": (
                "EXPERIMENT_PLAN_FINAL.MD §9.2 (architecture-aligned "
                "structured-mask/SSD suite); theorem refs: "
                "thesis/theorem_a.txt Proposition 3 (SSD realization); "
                "thesis/theorem_c.txt Proposition 4 (grouped closure)."
            ),
            "phase": "Phase IV — architecture-aligned validation layer",
            "category": (
                "G2 sampled-context band-RRS with trainable spectral-only "
                "reduced-Γ model. NOT an exact theorem verification; "
                "qualitative architecture-level pattern matching against "
                "theorem-C grouped closure."
            ),
            "framing": (
                "This experiment validates (a) that G2 sampled-context mode "
                "works correctly with a trainable model, and (b) that "
                "theorem-C grouped-closure predictions hold in the "
                "architecture-aligned setting. The spectral-only reduced-Γ "
                "model, trained on sampled band-RRS contexts with "
                "per-context independent block-Haar rotations, should "
                "reproduce the theorem-C phase diagram qualitatively: "
                "larger band size m and larger within-band condition "
                "number κ create higher spectral-only loss floors. The "
                "learned γ should exhibit block-scalar structure at m ≥ 4, "
                "κ ≥ 3 (commutant closure manifest at the architecture "
                "level)."
            ),
            "architecture": (
                "Reduced-Γ GD-compatible forward pass (equivalent to the "
                "theorem-A SSD realization). Single tied γ ∈ ℝ^D in the "
                "DCT-II spectral basis. Parameter count = D regardless of "
                "L. Initialization γ ≡ 0 matches theorem-C γ(0) = 0 "
                "boundary."
            ),
            "task": (
                "G2 band-RRS sampled contexts. Per training step, fresh "
                f"batch of {cfg.batch_contexts} contexts with "
                "per-context block-Haar rotations. Block mean λ = 1 "
                "uniformly; within-band κ swept. ω ≡ 1. σ = 0 (noiseless). "
                f"label_norm = {cfg.label_norm}."
            ),
            "device": str(device),
            "geometry": {
                "D": int(cfg.D), "P": int(cfg.P),
                "K": int(cfg.K), "L": int(cfg.L),
                "batch_contexts": int(cfg.batch_contexts),
                "train_steps": int(cfg.train_steps),
                "log_every": int(cfg.log_every),
            },
            "sweep": {
                "m_list": list(cfg.m_list),
                "kappa_list": list(cfg.kappa_list),
                "seed_list": list(cfg.seed_list),
                "n_cells": int(len(runs_by_cell_seed)),
            },
            "optimizer": cfg.optimizer,
            "learning_rate": float(cfg.learning_rate),
            "label_norm": cfg.label_norm,
            "status": (
                "all_ok" if all_ok else (
                    ("no_nan_ok" if no_nan else "NaN_FAILURE") + "+" +
                    ("bridge_ok" if bridge_ok else "bridge_FAIL") + "+" +
                    ("heterogeneity_ok" if heterogeneity_ok else "heterogeneity_FAIL") + "+" +
                    ("band_size_ok" if band_size_ok else "band_size_FAIL") + "+" +
                    ("correlation_ok" if correlation_ok else "correlation_FAIL") + "+" +
                    ("filter_bridge_ok" if filter_bridge_ok else "filter_bridge_FAIL") + "+" +
                    ("block_scalar_ok" if block_scalar_ok else "block_scalar_FAIL")
                )
            ),
            "gates": {
                "no_nan": bool(no_nan), "nan_count": int(nan_count),
                "stationary_bridge_loss": float(stat_bridge_loss),
                "stationary_bridge_threshold": float(cfg.stationary_bridge_loss_thresh),
                "heterogeneity_ok": bool(heterogeneity_ok),
                "n_heterogeneity_violations": int(len(heterogeneity_violations)),
                "band_size_ok": bool(band_size_ok),
                "n_band_size_violations": int(len(band_size_violations)),
                "spearman_rho": float(spearman),
                "spearman_rho_threshold": float(cfg.spearman_rho_min),
                "filter_bridge_rel_err_max": float(filter_bridge_max),
                "filter_bridge_threshold": float(cfg.filter_bridge_rel_err_thresh),
                "block_scalar_ok": bool(block_scalar_ok),
                "block_scalar_ratio_threshold": float(cfg.block_scalar_ratio_thresh),
                "block_scalar_records": [
                    {str(k): (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in r.items()}
                    for r in block_scalar_records
                ],
            },
            "sweep_wallclock_seconds": round(float(sweep_wall), 3),
        })

        if not all_ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
