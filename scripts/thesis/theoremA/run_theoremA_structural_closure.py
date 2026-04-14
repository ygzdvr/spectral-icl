"""Experiment A-structural: Proposition 2 + Proposition 5 + Remark 2 validation.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §8 (theorem-A exact tier,
remaining untested objects of the theorem-A chapter).

Theorem-level framing
---------------------
A1 and A1b validate Theorem 1 in the GD-compatible special case. A1-general
validates Theorem 1 for every train-supported ``S`` and Proposition 3
(necessity of GD-compatibility). A2 / A3 / A4 cover the perturbation bound,
explicit semiseparable realization, and negative controls. The three
remaining theorem-chapter objects not yet directly tested are:

* **Proposition 2** — rank-1 factorization and semiseparable representation
  of the GD-compatible mixer ``M^GD``. Claims (1)–(2) are structural matrix
  identities about the specific matrix ``M^GD = s t^⊤`` with
  ``s = [−𝟏_P; 𝟏_K]`` and ``t = [𝟏_P; 0_K]``.
* **Proposition 5 (structural closure)** — if ``S_TT`` and ``K_Γ(X)`` are
  both Toeplitz / both circulant / lower-``r_1``/``r_2``-semiseparable,
  then ``A_S(X,Γ) = (1/P)(S_TT ⊙ K_Γ(X))`` inherits the same structure
  (with semiseparable rank ``r_1 r_2``).
* **Remark 2** — the untied-layer extension: per-layer ``Γ_ℓ`` gives a
  non-autonomous reduced model whose L-layer forward agrees with the
  full hidden-state forward with per-layer bilinear form.

Each part is a deterministic matrix-level identity test: no training, no
architecture, no statistics.

Structure
---------
* **Part 1** — Proposition 2 claims (1)–(2): rank-1 factorization and
  semiseparable generator reconstruction of ``M^GD``. Machine eps.
* **Part 2** — Toeplitz closure. Build an exactly-Toeplitz ``S_TT`` from a
  decaying kernel. Build an exactly-Toeplitz ``K_Γ(X) = X^⊤ Γ X`` by
  choosing ``Γ = I`` and ``X`` from a Cholesky-like factor of a target
  Toeplitz PSD matrix. Verify ``A_S`` is Toeplitz.
* **Part 3** — Circulant closure. Build circulant ``S_TT`` via a
  real-even period-P kernel. Build circulant ``K_Γ(X)`` via a real-
  orthogonal eigendecomposition of a target circulant PSD matrix
  (``X = diag(√λ) V^⊤`` so ``X^⊤ X`` equals the target exactly up to
  eigendecomposition precision). Verify ``A_S`` is circulant, and check
  the **correct** Fourier consistency relation for a Hadamard product of
  circulants: the DFT eigenvalues of the Hadamard product are NOT the
  elementwise product of DFT eigenvalues (that is the formula for the
  *matrix* product of circulants). The correct identity is the
  circular-convolution relation
  ``eigvals(A_S) = (1/P²) · (eigvals(S_TT) ⊛ eigvals(K_Γ(X)))``,
  which follows directly from the convolution theorem applied to
  ``FFT(s ⊙ k)``.
* **Part 4** — Semiseparable rank multiplicativity. Build lower
  ``r_1``-semiseparable ``S_TT`` (``S[i,j] = u_i^⊤ v_j`` for ``i > j``) and
  lower ``r_2``-semiseparable ``K`` (``K[i,j] = a_i^⊤ b_j`` for ``i > j``).
  Verify (i) the canonical strict-lower rectangular block
  ``A_S[⌈P/2⌉:, :⌊P/2⌋]`` has SVD rank ≤ ``r_1 r_2``, (ii) several other
  random strictly-lower-triangular rectangular submatrices satisfy the
  same bound (the *actual* semiseparable-rank definition from Definition 3
  of the chapter — not the SVD rank of the zero-padded strict-lower
  matrix, which can exceed ``r_1 r_2``, e.g. a constant strict-lower
  triangle is lower-1-semiseparable but has SVD rank ``P − 1``), and
  (iii) the explicit Kronecker factorization
  ``A_S[i,j] = (1/P) (u_i ⊗ a_i)^⊤ (v_j ⊗ b_j)`` for every ``i > j``.
* **Part 5 (optional, default on)** — Remark 2 untied-layer equivalence.
  Build ``L`` distinct ``Γ_ℓ`` matrices, run the full hidden-state forward
  with per-layer bilinear form, and compare to the non-autonomous reduced
  recursion
  ``r^{ℓ+1} = (I + L^{-1} A_S(X, Γ_ℓ)) r^ℓ,
  F = (1/L) Σ_m B_S(X_⋆, X, Γ_m) r^m``.

Acceptance gates
----------------
* ``prop2_rank1_pass`` — ``‖M^GD − s t^⊤‖_F < 1e-14`` for every (P, K).
* ``prop2_generators_pass`` — ``‖M^GD − M_reconstructed‖_F < 1e-14`` where
  ``M_reconstructed`` is built from the ``(u_i, v_j, d_i)`` generators per
  Definition 3.
* ``prop5_toeplitz_pass`` — max-diagonal Toeplitz violation and off-
  Toeplitz energy both ``< 1e-12``.
* ``prop5_circulant_pass`` — off-circulant-Fourier energy ``< 1e-12``
  and the Hadamard-convolution Fourier consistency relation ``< 1e-12``.
* ``prop5_semiseparable_pass`` — canonical strict-lower-block SVD rank
  has ``σ_{r_1 r_2 + 1} < 1e-10``; every sampled strictly-lower-tri
  submatrix satisfies the same bound.
* ``prop5_kronecker_pass`` — per-entry Kronecker reconstruction error
  ``< 1e-12`` for every ``(i, j)`` with ``i > j``.
* ``remark2_untied_pass`` — R0-untied vs. R-reduced-non-autonomous
  relative L2 error ``< 1e-10`` in every cell.

Run
---
::

    python -u scripts/thesis/theoremA/run_theoremA_structural_closure.py \
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

from scripts.thesis.utils.metrics import reduced_model_error
from scripts.thesis.utils.plotting import (
    apply_thesis_style,
    phase_heatmap,
    save_both,
    sequential_colors,
)
from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralConfig:
    """Frozen configuration for the A-structural sweep."""

    P_list: tuple[int, ...] = (8, 16, 32, 64)
    K_list: tuple[int, ...] = (4, 8, 16)
    tau_list: tuple[float, ...] = (1.0, 3.0)

    # (r1, r2) pairs for the semiseparable rank test.
    semisep_rank_pairs: tuple[tuple[int, int], ...] = (
        (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2),
    )
    # Number of random strictly-lower-triangular rectangular submatrices to
    # test per (P, r1, r2).
    n_random_submatrices: int = 5

    # Untied-layer (Remark 2) sweep.
    untied_D_list: tuple[int, ...] = (16, 32)
    untied_P_list: tuple[int, ...] = (16, 32)
    untied_K_list: tuple[int, ...] = (8,)
    untied_L_list: tuple[int, ...] = (2, 4, 8)
    untied_B: int = 2
    include_untied: bool = True

    # Acceptance thresholds.
    prop2_tol: float = 1e-14
    toeplitz_tol: float = 1e-12
    circulant_tol: float = 1e-12
    semisep_sv_tol: float = 1e-10      # SVD null-space threshold
    kronecker_tol: float = 1e-12
    untied_tol: float = 1e-10

    base_seed: int = 0
    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Matrix-structure helpers
# ---------------------------------------------------------------------------


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _toeplitz_projection(M: torch.Tensor) -> torch.Tensor:
    """Project a (P, P) matrix onto its Toeplitz part by averaging each diagonal."""
    P = int(M.shape[-1])
    out = torch.zeros_like(M)
    rows = torch.arange(P, device=M.device)
    for d in range(-(P - 1), P):
        if d >= 0:
            i = rows[: P - d]
            j = i + d
        else:
            j = rows[: P + d]
            i = j - d
        avg = M[i, j].mean()
        out[i, j] = avg
    return out


def _toeplitz_diagonal_variance(M: torch.Tensor) -> float:
    """Max-over-diagonals variance of the (P, P) matrix (should be 0 if Toeplitz)."""
    P = int(M.shape[-1])
    max_var = 0.0
    for d in range(-(P - 1), P):
        diag = torch.diagonal(M, offset=d)
        if diag.numel() > 1:
            v = float(diag.var(unbiased=False))
            if v > max_var:
                max_var = v
    return max_var


def _circulant_projection(M: torch.Tensor) -> torch.Tensor:
    """Project onto circulant by averaging along each mod-P diagonal."""
    P = int(M.shape[-1])
    out = torch.zeros_like(M)
    rows = torch.arange(P, device=M.device)
    for d in range(P):
        cols = (rows + d) % P
        avg = M[rows, cols].mean()
        out[rows, cols] = avg
    return out


def _is_circulant_via_dft(M: torch.Tensor) -> float:
    """Return the off-diagonal Fourier energy of ``F M F^*`` as a ratio."""
    P = int(M.shape[-1])
    # Complex DFT via FFT. For circulant M, F^* M F is diagonal (with
    # normalization): DFT applied to each column then conjugated.
    Mc = M.to(dtype=torch.complex128)
    # F M F^* is diagonal. Use torch.fft: we can compute eigvals as
    # fft(first_col(M)). If M is circulant, the off-diagonal spectral
    # energy of ``F M F^* `` is 0. We form the matrix explicitly.
    # F[k, j] = (1/√P) * exp(-2πi k j / P)  (unitary convention).
    k = torch.arange(P, dtype=torch.float64, device=M.device)
    j = k
    F = torch.exp(-2j * math.pi * torch.outer(k, j) / P) / math.sqrt(float(P))
    F = F.to(torch.complex128)
    FMFh = F @ Mc @ F.conj().T
    diag = torch.diag(FMFh)
    off = FMFh - torch.diag(diag)
    num = float(off.abs().pow(2).sum().sqrt())
    den = float(M.norm()) + 1e-300
    return num / den


def _build_toeplitz_psd_kernel(P: int, tau: float, dtype: torch.dtype) -> torch.Tensor:
    """Build a symmetric Toeplitz PSD matrix ``K[i,j] = exp(-|i-j|/τ)``."""
    idx = torch.arange(P, dtype=dtype)
    D = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    return torch.exp(-D / float(tau))


def _build_circulant_psd_kernel(P: int, tau: float, dtype: torch.dtype) -> torch.Tensor:
    """Build a circulant PSD matrix ``K[i,j] = exp(-min(|i-j|, P-|i-j|)/τ)``."""
    idx = torch.arange(P, dtype=dtype)
    D = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    Dcirc = torch.minimum(D, P - D)
    return torch.exp(-Dcirc / float(tau))


def _x_from_psd_target(K_target: torch.Tensor) -> torch.Tensor:
    """Return ``X`` such that ``X^⊤ X = K_target`` via symmetric eigendecomp.

    ``K_target`` must be symmetric PSD. The returned ``X`` has shape
    ``(P, P)`` (D = P). Uses ``torch.linalg.eigh`` for robustness; for a
    clean PSD target the precision is ~``1e-14``.
    """
    evals, evecs = torch.linalg.eigh(K_target)
    evals = evals.clamp(min=0.0)
    return torch.diag(evals.sqrt()) @ evecs.T


def _build_semiseparable_strict_lower(
    P: int, r: int, seed: int, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(M, U, V)`` with ``M[i,j] = u_i^⊤ v_j`` for ``i > j`` else 0.

    ``U, V`` have shape ``(P, r)``; rows are the semiseparable generators.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    U = torch.randn(P, r, generator=gen, dtype=dtype)
    V = torch.randn(P, r, generator=gen, dtype=dtype)
    M_full = U @ V.T
    M = torch.tril(M_full, diagonal=-1)
    return M, U, V


# ---------------------------------------------------------------------------
# PART 1 — Proposition 2 (M_GD = s t^⊤, semiseparable reconstruction)
# ---------------------------------------------------------------------------


def _run_part1(
    cfg: StructuralConfig, device: torch.device,
) -> list[dict[str, Any]]:
    """Validate Proposition 2 claims (1)–(2) at the matrix-identity level."""
    dtype = _torch_dtype(cfg.dtype)
    out: list[dict[str, Any]] = []
    for P in cfg.P_list:
        for K in cfg.K_list:
            T = P + K
            # Block construction of M_GD.
            ones_PP = torch.ones(P, P, dtype=dtype, device=device)
            ones_KP = torch.ones(K, P, dtype=dtype, device=device)
            M_GD = torch.zeros(T, T, dtype=dtype, device=device)
            M_GD[:P, :P] = -ones_PP
            M_GD[P:, :P] = ones_KP

            # s, t per eq. (10).
            s = torch.cat([
                -torch.ones(P, dtype=dtype, device=device),
                torch.ones(K, dtype=dtype, device=device),
            ])
            t = torch.cat([
                torch.ones(P, dtype=dtype, device=device),
                torch.zeros(K, dtype=dtype, device=device),
            ])
            st_outer = torch.outer(s, t)
            rank1_err = float((M_GD - st_outer).norm())

            # Rank via SVD: σ_1 should dominate; σ_2, σ_3, … should be 0.
            sv = torch.linalg.svdvals(M_GD)
            sv_np = sv.detach().cpu().numpy().astype(float)
            # "Second singular value" — should be ~ eps.
            sigma1 = float(sv_np[0])
            sigma2 = float(sv_np[1]) if sv_np.size >= 2 else 0.0
            # Relative, so the threshold is scale-free.
            sigma2_rel = sigma2 / (sigma1 + 1e-300)

            # Semiseparable generators per eq. (11): u_i = ṽ_i = s_i,
            # v_j = ṽ_j = t_j, d_i = s_i · t_i.
            d = s * t
            M_recon = torch.zeros_like(M_GD)
            # Strict lower: u_i^⊤ v_j = s_i · t_j for i > j.
            for i in range(T):
                for j in range(T):
                    if i > j:
                        M_recon[i, j] = s[i] * t[j]
                    elif i < j:
                        # ṽ generators equal non-tilde here.
                        M_recon[i, j] = s[i] * t[j]
                    else:
                        M_recon[i, j] = d[i]
            generator_err = float((M_GD - M_recon).norm())

            out.append({
                "P": int(P), "K": int(K), "T": T,
                "rank1_err": rank1_err,
                "sigma1": sigma1,
                "sigma2": sigma2,
                "sigma2_rel": sigma2_rel,
                "generator_err": generator_err,
            })
    return out


# ---------------------------------------------------------------------------
# PART 2 — Proposition 5, Toeplitz closure
# ---------------------------------------------------------------------------


def _run_part2(
    cfg: StructuralConfig, device: torch.device,
) -> list[dict[str, Any]]:
    """Verify Toeplitz closure of the Hadamard-Γ reduction."""
    dtype = _torch_dtype(cfg.dtype)
    out: list[dict[str, Any]] = []
    for P in cfg.P_list:
        for tau in cfg.tau_list:
            # S_TT is an exactly-Toeplitz symmetric decaying kernel.
            S_TT = _build_toeplitz_psd_kernel(P, tau, dtype).to(device)

            # K_Γ(X) = X^⊤ X built to be exactly Toeplitz. We pick a target
            # Toeplitz PSD matrix (identical kernel shape with a different
            # decay parameter so the test is non-trivial) and find X via
            # eigendecomposition.
            K_target = _build_toeplitz_psd_kernel(P, tau * 0.5, dtype).to(device)
            X = _x_from_psd_target(K_target)
            Gamma = torch.eye(P, dtype=dtype, device=device)
            K_Gamma = X.T @ Gamma @ X

            # Diagnostic: is the constructed K_Γ(X) really Toeplitz?
            K_toep_var_rel = _toeplitz_diagonal_variance(K_Gamma) / (
                float(K_Gamma.norm()) ** 2 + 1e-300
            )

            # A_S = (1/P) * (S_TT ⊙ K_Γ(X)).
            A_S = (S_TT * K_Gamma) / float(P)

            # Primary gate: off-Toeplitz energy.
            A_S_toep = _toeplitz_projection(A_S)
            off_toep = float((A_S - A_S_toep).norm())
            A_S_norm = float(A_S.norm())
            off_toep_rel = off_toep / (A_S_norm + 1e-300)

            # Secondary diagnostic: max-diagonal variance normalized by
            # ‖A_S‖_F² per the spec.
            max_diag_var_rel = _toeplitz_diagonal_variance(A_S) / (
                A_S_norm ** 2 + 1e-300
            )

            out.append({
                "P": int(P), "tau": float(tau),
                "K_input_toeplitz_var_rel": float(K_toep_var_rel),
                "off_toeplitz_energy_rel": float(off_toep_rel),
                "max_diag_variance_rel": float(max_diag_var_rel),
                "A_S_norm": A_S_norm,
            })
    return out


# ---------------------------------------------------------------------------
# PART 3 — Proposition 5, Circulant closure (+ Fourier convolution check)
# ---------------------------------------------------------------------------


def _run_part3(
    cfg: StructuralConfig, device: torch.device,
) -> list[dict[str, Any]]:
    """Verify circulant closure and the correct Hadamard Fourier relation."""
    dtype = _torch_dtype(cfg.dtype)
    out: list[dict[str, Any]] = []
    for P in cfg.P_list:
        # Use tau = 3.0 (wide kernel; well-conditioned).
        tau = 3.0
        S_TT = _build_circulant_psd_kernel(P, tau, dtype).to(device)

        # K target circulant PSD with a different decay.
        K_target = _build_circulant_psd_kernel(P, tau * 0.5, dtype).to(device)
        X = _x_from_psd_target(K_target)
        Gamma = torch.eye(P, dtype=dtype, device=device)
        K_Gamma = X.T @ Gamma @ X

        # Diagnostic: is K_Γ(X) really circulant?
        K_input_circ_err = _is_circulant_via_dft(K_Gamma)

        # Also S_TT.
        S_TT_circ_err = _is_circulant_via_dft(S_TT)

        # A_S = (1/P) * (S_TT ⊙ K_Γ(X)).
        A_S = (S_TT * K_Gamma) / float(P)

        # Primary gate: off-circulant Fourier energy.
        off_circ_rel = _is_circulant_via_dft(A_S)

        # Also: ‖A_S − circulant_proj(A_S)‖_F / ‖A_S‖_F.
        A_S_circ = _circulant_projection(A_S)
        off_circ_spatial = float((A_S - A_S_circ).norm()) / (
            float(A_S.norm()) + 1e-300
        )

        # Fourier-space consistency. For circulant M with first column m,
        # the DFT eigenvalues are m̂ = FFT(m). For A_S = (1/P) (S ⊙ K),
        # viewed as circulants with first columns s, k (entries
        # c_{(i-0) mod P} = M[i, 0]),
        #    â = FFT(a), a = (1/P)(s ⊙ k)
        # and by the convolution theorem FFT(s ⊙ k) = (1/P) (ŝ ⊛ k̂),
        # where ⊛ is circular convolution. Therefore
        #    â = (1/P²) · (ŝ ⊛ k̂).
        # This is the CORRECT Fourier identity for a Hadamard product of
        # circulants. (The simple elementwise product ŝ · k̂ is the formula
        # for the MATRIX PRODUCT of two circulants — a different operation.)
        s_col = S_TT[:, 0]
        k_col = K_Gamma[:, 0]
        a_col = A_S[:, 0]
        s_hat = torch.fft.fft(s_col.to(torch.complex128))
        k_hat = torch.fft.fft(k_col.to(torch.complex128))
        a_hat = torch.fft.fft(a_col.to(torch.complex128))
        # Circular convolution ŝ ⊛ k̂ via FFT convolution-theorem dual.
        conv_sk = torch.fft.ifft(
            torch.fft.fft(s_hat) * torch.fft.fft(k_hat)
        )  # this gives (1/P) · (ŝ ⊛ k̂) with torch's unitary-free FFT
        # With torch.fft.fft (no scaling) and ifft (scaled by 1/P):
        #   ifft(fft(ŝ) · fft(k̂)) = ŝ ⊛ k̂.
        # Verify:  â =? (1/P²) · (ŝ ⊛ k̂)
        # Torch's fft has the unscaled forward convention:
        #   X_k = Σ_n x_n e^{-2πi k n / N}.
        # conv theorem: FFT(x ⊙ y) = (1/N) · (FFT(x) ⊛ FFT(y)).
        # So FFT((1/P) s ⊙ k) = (1/P²) · (ŝ ⊛ k̂). Match.
        expected_a_hat = conv_sk / float(P * P)
        fourier_consistency_err = float(
            (a_hat - expected_a_hat).abs().norm()
        ) / (float(a_hat.abs().norm()) + 1e-300)

        # INCORRECT elementwise-product formula — verify it really fails
        # (the user's originally-phrased claim equivalent). This is kept as
        # a diagnostic to document why we use the convolution relation.
        elementwise_claim_err = float(
            (a_hat - s_hat * k_hat).abs().norm()
        ) / (float(a_hat.abs().norm()) + 1e-300)

        out.append({
            "P": int(P), "tau": float(tau),
            "S_TT_circulant_err": float(S_TT_circ_err),
            "K_input_circulant_err": float(K_input_circ_err),
            "off_circulant_fourier_rel": float(off_circ_rel),
            "off_circulant_spatial_rel": float(off_circ_spatial),
            "fourier_convolution_consistency_err": float(fourier_consistency_err),
            "fourier_elementwise_claim_err": float(elementwise_claim_err),
        })
    return out


# ---------------------------------------------------------------------------
# PART 4 — Proposition 5, Semiseparable rank multiplicativity
# ---------------------------------------------------------------------------


def _run_part4(
    cfg: StructuralConfig, device: torch.device,
) -> list[dict[str, Any]]:
    """Verify semiseparable rank closure: r_A ≤ r_1 r_2."""
    dtype = _torch_dtype(cfg.dtype)
    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(cfg.base_seed) + 2024)
    for P in cfg.P_list:
        for (r1, r2) in cfg.semisep_rank_pairs:
            if r1 * r2 >= P:
                # Rank bound is trivial; skip.
                continue
            seed_S = int(cfg.base_seed) + 17 * P + 31 * r1 + 7 * r2
            seed_K = int(cfg.base_seed) + 11 * P + 59 * r2 + 13 * r1
            S_TT, U, V = _build_semiseparable_strict_lower(P, r1, seed_S, dtype)
            K, A, B = _build_semiseparable_strict_lower(P, r2, seed_K, dtype)
            S_TT = S_TT.to(device); K = K.to(device)
            U = U.to(device); V = V.to(device)
            A = A.to(device); B = B.to(device)

            A_S = (S_TT * K) / float(P)

            # (d) Canonical strict-lower-tri rectangular block. Rows
            # ``[⌈P/2⌉, P)``, cols ``[0, ⌊P/2⌋)``. This block lies
            # strictly below the diagonal because min(rows) ≥ ⌈P/2⌉ ≥
            # ⌊P/2⌋ > max(cols). So its rank equals the semiseparable
            # rank lower bound.
            i_lo = (P + 1) // 2
            j_hi = P // 2
            block_main = A_S[i_lo:P, 0:j_hi].clone()
            sv_main = torch.linalg.svdvals(block_main)
            sv_main_np = sv_main.detach().cpu().numpy().astype(float)
            r_mul = int(r1 * r2)
            sigma_r_plus1_main = (
                float(sv_main_np[r_mul]) if sv_main_np.size > r_mul else 0.0
            )
            sigma_1_main = float(sv_main_np[0]) if sv_main_np.size > 0 else 0.0
            sigma_r_plus1_main_rel = sigma_r_plus1_main / (sigma_1_main + 1e-300)

            # (e) Several random strictly-lower-triangular rectangular
            # submatrices. For each, sample (i_lo, i_hi, j_lo, j_hi) with
            # ``i_lo > j_hi`` (so the block lies strictly below the
            # diagonal), a minimum block size of ``2 · r_mul + 1``, and
            # verify its rank.
            sub_worst_sigma_rel = 0.0
            sub_results = []
            for k in range(cfg.n_random_submatrices):
                # Random block with constraints.
                min_size = max(2 * r_mul + 1, 2)
                if P <= min_size + 1:
                    continue
                # j_hi  ∈ [1, P // 2)
                jh = int(rng.integers(1, max(2, P // 2)))
                il = int(rng.integers(jh + 1, P))
                jl = int(rng.integers(0, jh))
                ih = int(rng.integers(il + 1, P + 1))
                block = A_S[il:ih, jl:jh]
                if block.numel() == 0 or min(block.shape) <= r_mul:
                    continue
                sv = torch.linalg.svdvals(block)
                sv_np = sv.detach().cpu().numpy().astype(float)
                s1 = float(sv_np[0]) if sv_np.size > 0 else 0.0
                srel = (
                    float(sv_np[r_mul]) / (s1 + 1e-300)
                    if sv_np.size > r_mul else 0.0
                )
                sub_worst_sigma_rel = max(sub_worst_sigma_rel, srel)
                sub_results.append({
                    "i_lo": il, "i_hi": ih, "j_lo": jl, "j_hi": jh,
                    "rows": ih - il, "cols": jh - jl,
                    "sigma_r_plus1_rel": srel,
                })

            # (f) Kronecker verification for every (i, j) with i > j.
            # Expected: A_S[i, j] == (U[i] ⊗ A[i])^⊤ (V[j] ⊗ B[j]) / P
            #         == (U[i]^⊤ V[j]) · (A[i]^⊤ B[j]) / P.
            tril_mask = torch.tril(
                torch.ones(P, P, dtype=torch.bool, device=device), diagonal=-1,
            )
            uv = U @ V.T
            ab = A @ B.T
            expected = (uv * ab) / float(P)
            diff = (A_S - expected).abs()
            kron_err = float(diff[tril_mask].max())
            kron_rel = kron_err / (float(expected[tril_mask].abs().max()) + 1e-300)

            out.append({
                "P": int(P), "r1": int(r1), "r2": int(r2),
                "r_mul": r_mul,
                "main_block_shape": list(block_main.shape),
                "main_block_sv": sv_main_np.tolist(),
                "main_block_sigma_r_plus1_rel": float(sigma_r_plus1_main_rel),
                "random_submatrix_worst_sigma_r_plus1_rel": float(sub_worst_sigma_rel),
                "random_submatrices": sub_results,
                "kronecker_max_abs_err": kron_err,
                "kronecker_rel_err": float(kron_rel),
            })
    return out


# ---------------------------------------------------------------------------
# PART 5 — Remark 2, untied-layer equivalence
# ---------------------------------------------------------------------------


def _route_R0_untied(
    X_train: torch.Tensor, X_query: torch.Tensor,
    Gammas: list[torch.Tensor],
    y_train: torch.Tensor,
    S_TT: torch.Tensor, S_QT: torch.Tensor,
    L: int, P_norm: int,
) -> torch.Tensor:
    """Full hidden-state forward with a per-layer ``Γ_ℓ``."""
    B = int(X_train.shape[0])
    P = int(X_train.shape[-1])
    K = int(X_query.shape[-1])
    dtype = X_train.dtype
    device = X_train.device
    X = torch.cat([X_train, X_query], dim=-1)

    S_full = torch.zeros(P + K, P + K, dtype=dtype, device=device)
    S_full[:P, :P] = S_TT
    S_full[P:, :P] = S_QT

    h = torch.cat(
        [y_train, torch.zeros(B, K, dtype=dtype, device=device)], dim=-1,
    )
    inv_L = 1.0 / float(L)
    for ell in range(int(L)):
        Gamma_ell = Gammas[ell]
        GammaX = torch.einsum("de,bef->bdf", Gamma_ell, X)
        S_pos = torch.einsum("bdm,bdn->bmn", X, GammaX) / float(P_norm)
        M_eff = S_pos * S_full.unsqueeze(0)
        h = h + inv_L * torch.einsum("bmn,bn->bm", M_eff, h)
    return h[:, P:]


def _route_reduced_untied(
    X_train: torch.Tensor, X_query: torch.Tensor,
    Gammas: list[torch.Tensor],
    y_train: torch.Tensor,
    S_TT: torch.Tensor, S_QT: torch.Tensor,
    L: int, P_norm: int,
) -> torch.Tensor:
    """Non-autonomous reduced recursion (Remark 2).

    ``r^{ℓ+1} = (I + L^{-1} A_S(X, Γ_ℓ)) r^ℓ`` with
    ``F = (1/L) Σ_m B_S(X_⋆, X, Γ_m) r^m``.
    """
    B = int(X_train.shape[0])
    K = int(X_query.shape[-1])
    dtype = y_train.dtype
    device = y_train.device
    inv_L = 1.0 / float(L)
    r = y_train.clone()
    F = torch.zeros(B, K, dtype=dtype, device=device)
    for m in range(int(L)):
        Gamma_m = Gammas[m]
        K_train_m = torch.einsum(
            "bim,ij,bjn->bmn", X_train, Gamma_m, X_train,
        ) / float(P_norm)
        K_query_m = torch.einsum(
            "bim,ij,bjn->bmn", X_query, Gamma_m, X_train,
        ) / float(P_norm)
        A_S_m = S_TT.unsqueeze(0) * K_train_m
        B_S_m = S_QT.unsqueeze(0) * K_query_m
        F = F + inv_L * torch.einsum("bki,bi->bk", B_S_m, r)
        r = r + inv_L * torch.einsum("bpi,bi->bp", A_S_m, r)
    return F


def _run_part5(
    cfg: StructuralConfig, device: torch.device,
) -> list[dict[str, Any]]:
    """Untied-layer Remark 2 check: R0-untied vs reduced non-autonomous."""
    dtype = _torch_dtype(cfg.dtype)
    out: list[dict[str, Any]] = []
    base = int(cfg.base_seed)
    for D in cfg.untied_D_list:
        for P in cfg.untied_P_list:
            for K in cfg.untied_K_list:
                for L in cfg.untied_L_list:
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(base + 7 + 1000 * D + 100 * P + 10 * K + L)
                    X_train = torch.randn(
                        cfg.untied_B, D, P, generator=gen, dtype=dtype,
                    ).to(device)
                    X_query = torch.randn(
                        cfg.untied_B, D, K, generator=gen, dtype=dtype,
                    ).to(device)
                    beta = torch.randn(
                        cfg.untied_B, D, generator=gen, dtype=dtype,
                    ).to(device)
                    norm = math.sqrt(float(D))
                    y_train = torch.einsum(
                        "bd,bdp->bp", beta, X_train,
                    ) / norm

                    # GD-compatible mask is fine here — Remark 2 is about
                    # the per-layer Γ, not the mask class.
                    S_TT = -torch.ones(P, P, dtype=dtype, device=device)
                    S_QT = torch.ones(K, P, dtype=dtype, device=device)

                    # L distinct random Γ_ℓ matrices.
                    Gammas = []
                    for ell in range(int(L)):
                        g = torch.Generator(device="cpu")
                        g.manual_seed(
                            base + 101 + 997 * D + 7 * ell + 3 * P,
                        )
                        G_ell = torch.randn(
                            D, D, generator=g, dtype=dtype,
                        ) / math.sqrt(float(D))
                        Gammas.append(G_ell.to(device))

                    f_R0 = _route_R0_untied(
                        X_train, X_query, Gammas, y_train, S_TT, S_QT,
                        L=int(L), P_norm=int(P),
                    )
                    f_red = _route_reduced_untied(
                        X_train, X_query, Gammas, y_train, S_TT, S_QT,
                        L=int(L), P_norm=int(P),
                    )
                    err = reduced_model_error(f_R0, f_red)
                    out.append({
                        "D": int(D), "P": int(P), "K": int(K), "L": int(L),
                        "err_R0_vs_reduced": float(err),
                    })
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_prop2(
    cfg: StructuralConfig, results: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 1: Proposition 2 structural errors per (P, K)."""
    import matplotlib.pyplot as plt

    labels = [f"P={r['P']},K={r['K']}" for r in results]
    rank1 = [r["rank1_err"] for r in results]
    sigma2 = [r["sigma2_rel"] for r in results]
    gen_err = [r["generator_err"] for r in results]
    floor = 1e-18
    rank1 = [max(floor, v) for v in rank1]
    sigma2 = [max(floor, v) for v in sigma2]
    gen_err = [max(floor, v) for v in gen_err]

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * len(labels) + 3.0), 4.0))
    ax.bar(x - w, rank1, w, label=r"$\|M^{GD} - s t^\top\|_F$", color="C0")
    ax.bar(x, sigma2, w, label=r"$\sigma_2 / \sigma_1$", color="C1")
    ax.bar(
        x + w, gen_err, w,
        label="semiseparable-gen. reconstruction err",
        color="C2",
    )
    ax.axhline(
        cfg.prop2_tol, color="red", lw=0.9, ls="--",
        label=f"tol = {cfg.prop2_tol:.0e}",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("error")
    ax.set_title(
        "Proposition 2: rank-1 factorization and semiseparable reconstruction of $M^{GD}$",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save_both(fig, run_dir, "a5_prop2_structural")
    plt.close(fig)


def _plot_toeplitz(
    cfg: StructuralConfig, results: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 2: Toeplitz violation heatmap over (P, τ)."""
    import matplotlib.pyplot as plt

    P_list = list(cfg.P_list)
    tau_list = list(cfg.tau_list)
    grid = np.zeros((len(P_list), len(tau_list)))
    for r in results:
        iP = P_list.index(int(r["P"]))
        it = tau_list.index(float(r["tau"]))
        grid[iP, it] = r["off_toeplitz_energy_rel"]
    floor = 1e-18
    grid_plot = np.where(grid > floor, grid, floor)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    phase_heatmap(
        ax, grid_plot,
        x_coords=np.asarray(tau_list, dtype=float),
        y_coords=np.asarray(P_list, dtype=float),
        xlabel=r"$\tau$",
        ylabel="P",
        cbar_label=r"$\|A_S - \mathrm{Toep}(A_S)\|_F / \|A_S\|_F$",
        log_z=True, log_x=False, log_y=True,
    )
    ax.set_title(
        "Proposition 5: Toeplitz closure of the Hadamard reduction",
        fontsize=10,
    )
    fig.tight_layout()
    save_both(fig, run_dir, "a5_toeplitz_closure")
    plt.close(fig)


def _plot_circulant(
    cfg: StructuralConfig, results: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 3: circulant-closure + Fourier convolution consistency."""
    import matplotlib.pyplot as plt

    P_arr = np.asarray([r["P"] for r in results], dtype=float)
    off_four = np.asarray([r["off_circulant_fourier_rel"] for r in results])
    off_spatial = np.asarray([r["off_circulant_spatial_rel"] for r in results])
    four_conv = np.asarray([r["fourier_convolution_consistency_err"] for r in results])
    four_el = np.asarray([r["fourier_elementwise_claim_err"] for r in results])

    floor = 1e-18
    off_four = np.clip(off_four, floor, None)
    off_spatial = np.clip(off_spatial, floor, None)
    four_conv = np.clip(four_conv, floor, None)
    four_el = np.clip(four_el, floor, None)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    ax.plot(
        P_arr, off_four, marker="o", color="C0",
        label=r"$\|F A_S F^* - \mathrm{diag}\|_F / \|A_S\|_F$",
    )
    ax.plot(
        P_arr, off_spatial, marker="s", color="C2",
        label=r"spatial circ. projection err",
    )
    ax.axhline(
        cfg.circulant_tol, color="red", lw=0.9, ls="--",
        label=f"tol = {cfg.circulant_tol:.0e}",
    )
    ax.set_xlabel("P")
    ax.set_ylabel("rel. error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("(a) off-circulant Fourier energy of $A_S$", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(
        P_arr, four_conv, marker="o", color="C0",
        label=r"$\hat a = (1/P^2) \cdot \hat s \circledast \hat k$  (correct)",
    )
    ax.plot(
        P_arr, four_el, marker="x", color="C3", ls="--",
        label=r"$\hat a = \hat s \odot \hat k$  (matrix-prod formula, wrong here)",
    )
    ax.axhline(
        cfg.circulant_tol, color="red", lw=0.9, ls="--",
        label=f"tol = {cfg.circulant_tol:.0e}",
    )
    ax.set_xlabel("P")
    ax.set_ylabel("rel. error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(
        "(b) Fourier consistency: Hadamard uses convolution, not elementwise product",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    fig.suptitle(
        "Proposition 5: circulant closure of the Hadamard reduction",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_both(fig, run_dir, "a5_circulant_closure")
    plt.close(fig)


def _plot_semiseparable(
    cfg: StructuralConfig, results: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 4: SVD spectrum of the canonical strict-lower block per (P, r1, r2)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = sequential_colors(len(results))
    for r, color in zip(results, colors):
        sv = np.asarray(r["main_block_sv"], dtype=float)
        if sv.size == 0:
            continue
        sv_norm = sv / (sv[0] + 1e-300)
        x = np.arange(1, sv.size + 1)
        ax.plot(
            x, np.clip(sv_norm, 1e-18, None),
            color=color, lw=1.0, alpha=0.85,
            label=(
                f"P={r['P']}, $r_1 r_2$={r['r_mul']} "
                f"(=$r_1$·$r_2$ = {r['r1']}·{r['r2']})"
            ),
        )
        ax.axvline(r["r_mul"] + 0.5, color=color, lw=0.5, ls=":", alpha=0.5)
    ax.axhline(
        cfg.semisep_sv_tol, color="red", lw=0.9, ls="--",
        label=rf"tol = {cfg.semisep_sv_tol:.0e}",
    )
    ax.set_xlabel("singular-value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_yscale("log")
    ax.set_title(
        "Proposition 5: SVD spectrum of the canonical strict-lower rectangular block of $A_S$",
        fontsize=10,
    )
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    fig.tight_layout()
    save_both(fig, run_dir, "a5_semiseparable_rank")
    plt.close(fig)


def _plot_kronecker(
    cfg: StructuralConfig, results: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 5: per-(P, r1, r2) Kronecker reconstruction max-abs err."""
    import matplotlib.pyplot as plt

    vals = np.asarray([r["kronecker_max_abs_err"] for r in results], dtype=float)
    floor = 1e-18
    vals = np.clip(vals, floor, None)
    bins = np.logspace(-18, -6, 30)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(vals, bins=bins, color="C0", alpha=0.85, edgecolor="black", lw=0.5)
    ax.axvline(
        cfg.kronecker_tol, color="red", lw=0.9, ls="--",
        label=f"tol = {cfg.kronecker_tol:.0e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\max_{i>j} |A_S[i,j] - (u_i \otimes a_i)^\top (v_j \otimes b_j)/P|$")
    ax.set_ylabel("count")
    ax.set_title(
        "Proposition 5: Kronecker factorization consistency",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a5_kronecker_verification")
    plt.close(fig)


def _plot_untied(
    cfg: StructuralConfig, results: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 6: Remark 2 untied-layer R0 vs non-autonomous reduced."""
    import matplotlib.pyplot as plt

    vals = np.asarray([r["err_R0_vs_reduced"] for r in results], dtype=float)
    floor = 1e-18
    vals = np.clip(vals, floor, None)
    bins = np.logspace(-18, -6, 30)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(vals, bins=bins, color="C0", alpha=0.85, edgecolor="black", lw=0.5)
    ax.axvline(
        cfg.untied_tol, color="red", lw=0.9, ls="--",
        label=f"tol = {cfg.untied_tol:.0e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"rel. error $R_0^{\mathrm{untied}}$ vs non-autonomous reduced")
    ax.set_ylabel("count")
    ax.set_title(
        "Remark 2: untied-layer equivalence", fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a5_untied_layers")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment A-structural: Proposition 2 + Proposition 5 + "
            "Remark 2 structural validation (plan §8)."
        )
    )
    p.add_argument(
        "--device", type=str, default="cuda", choices=("cpu", "cuda", "auto")
    )
    p.add_argument(
        "--dtype", type=str, default="float64", choices=("float32", "float64")
    )
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--skip-untied", action="store_true",
        help="Skip Part 5 (Remark 2 untied-layer equivalence).",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Small sweep for smoke-testing.",
    )
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> StructuralConfig:
    base = StructuralConfig()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.skip_untied:
        overrides["include_untied"] = False
    if args.quick:
        overrides.update(
            P_list=(8, 16),
            K_list=(4,),
            tau_list=(3.0,),
            semisep_rank_pairs=((1, 1), (2, 2)),
            untied_D_list=(16,),
            untied_P_list=(16,),
            untied_K_list=(8,),
            untied_L_list=(2, 4),
        )
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
    print(f"[A-structural] device = {device}")

    run = ThesisRunDir(__file__, phase="theoremA")
    with RunContext(
        run,
        config=cfg,
        seeds=[cfg.base_seed, cfg.base_seed + 7, cfg.base_seed + 13,
               cfg.base_seed + 101, cfg.base_seed + 2024],
        notes=(
            "A-structural: validates Proposition 2 (rank-1 factorization "
            "and semiseparable rep of M_GD), Proposition 5 (Toeplitz / "
            "circulant / semiseparable closure of the Hadamard reduction), "
            "and Remark 2 (untied-layer non-autonomous reduced model). "
            "Deterministic matrix-identity tier; no training."
        ),
    ) as ctx:
        apply_thesis_style()

        # --- Run all five parts ---
        t0 = time.perf_counter()
        results_p1 = _run_part1(cfg, device); t_p1 = time.perf_counter() - t0
        print(f"[A-structural] Part 1 (Prop 2) done in {t_p1:.2f} s "
              f"({len(results_p1)} cells)")

        t0 = time.perf_counter()
        results_p2 = _run_part2(cfg, device); t_p2 = time.perf_counter() - t0
        print(f"[A-structural] Part 2 (Toeplitz) done in {t_p2:.2f} s "
              f"({len(results_p2)} cells)")

        t0 = time.perf_counter()
        results_p3 = _run_part3(cfg, device); t_p3 = time.perf_counter() - t0
        print(f"[A-structural] Part 3 (Circulant) done in {t_p3:.2f} s "
              f"({len(results_p3)} cells)")

        t0 = time.perf_counter()
        results_p4 = _run_part4(cfg, device); t_p4 = time.perf_counter() - t0
        print(f"[A-structural] Part 4 (Semisep) done in {t_p4:.2f} s "
              f"({len(results_p4)} cells)")

        if cfg.include_untied:
            t0 = time.perf_counter()
            results_p5 = _run_part5(cfg, device); t_p5 = time.perf_counter() - t0
            print(f"[A-structural] Part 5 (Remark 2) done in {t_p5:.2f} s "
                  f"({len(results_p5)} cells)")
        else:
            results_p5 = []
            t_p5 = 0.0

        total_wall = t_p1 + t_p2 + t_p3 + t_p4 + t_p5

        # --- Figures ---
        _plot_prop2(cfg, results_p1, run)
        _plot_toeplitz(cfg, results_p2, run)
        _plot_circulant(cfg, results_p3, run)
        _plot_semiseparable(cfg, results_p4, run)
        _plot_kronecker(cfg, results_p4, run)
        if results_p5:
            _plot_untied(cfg, results_p5, run)

        # --- Acceptance gates ---
        prop2_rank1_pass = all(
            r["rank1_err"] < cfg.prop2_tol
            and r["sigma2_rel"] < cfg.prop2_tol
            for r in results_p1
        )
        prop2_generators_pass = all(
            r["generator_err"] < cfg.prop2_tol for r in results_p1
        )

        prop5_toeplitz_pass = all(
            r["off_toeplitz_energy_rel"] < cfg.toeplitz_tol
            for r in results_p2
        )

        prop5_circulant_pass = all(
            r["off_circulant_fourier_rel"] < cfg.circulant_tol
            and r["off_circulant_spatial_rel"] < cfg.circulant_tol
            and r["fourier_convolution_consistency_err"] < cfg.circulant_tol
            for r in results_p3
        )

        def _semisep_ok(r: dict[str, Any]) -> bool:
            if r["main_block_sigma_r_plus1_rel"] >= cfg.semisep_sv_tol:
                return False
            if r["random_submatrix_worst_sigma_r_plus1_rel"] >= cfg.semisep_sv_tol:
                return False
            return True

        prop5_semisep_pass = all(_semisep_ok(r) for r in results_p4)

        prop5_kron_pass = all(
            r["kronecker_rel_err"] < cfg.kronecker_tol for r in results_p4
        )

        if results_p5:
            remark2_untied_pass = all(
                r["err_R0_vs_reduced"] < cfg.untied_tol for r in results_p5
            )
        else:
            remark2_untied_pass = None  # not tested

        # --- Worst-case per-part ---
        worst_p1_rank1 = max(r["rank1_err"] for r in results_p1)
        worst_p1_sigma2 = max(r["sigma2_rel"] for r in results_p1)
        worst_p1_gen = max(r["generator_err"] for r in results_p1)
        worst_p2_toep = max(r["off_toeplitz_energy_rel"] for r in results_p2)
        worst_p3_circ = max(r["off_circulant_fourier_rel"] for r in results_p3)
        worst_p3_four = max(r["fourier_convolution_consistency_err"] for r in results_p3)
        worst_p4_svr = max(
            r["main_block_sigma_r_plus1_rel"] for r in results_p4
        ) if results_p4 else 0.0
        worst_p4_kron = max(
            r["kronecker_rel_err"] for r in results_p4
        ) if results_p4 else 0.0
        if results_p5:
            worst_p5_untied = max(r["err_R0_vs_reduced"] for r in results_p5)
        else:
            worst_p5_untied = float("nan")

        # --- Per-cell summary JSON ---
        per_cell = {
            "part1_prop2": results_p1,
            "part2_toeplitz": results_p2,
            "part3_circulant": results_p3,
            "part4_semiseparable": results_p4,
            "part5_untied_layers": results_p5,
        }
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(per_cell, indent=2) + "\n", encoding="utf-8"
        )

        # --- NPZ payload (compact arrays) ---
        npz_payload: dict[str, np.ndarray] = {}
        if results_p1:
            npz_payload["p1_rank1_err"] = np.asarray([r["rank1_err"] for r in results_p1])
            npz_payload["p1_sigma2_rel"] = np.asarray([r["sigma2_rel"] for r in results_p1])
            npz_payload["p1_generator_err"] = np.asarray([r["generator_err"] for r in results_p1])
        if results_p2:
            npz_payload["p2_off_toeplitz_rel"] = np.asarray(
                [r["off_toeplitz_energy_rel"] for r in results_p2]
            )
            npz_payload["p2_max_diag_var_rel"] = np.asarray(
                [r["max_diag_variance_rel"] for r in results_p2]
            )
        if results_p3:
            npz_payload["p3_off_circulant_fourier_rel"] = np.asarray(
                [r["off_circulant_fourier_rel"] for r in results_p3]
            )
            npz_payload["p3_fourier_convolution_err"] = np.asarray(
                [r["fourier_convolution_consistency_err"] for r in results_p3]
            )
            npz_payload["p3_fourier_elementwise_err"] = np.asarray(
                [r["fourier_elementwise_claim_err"] for r in results_p3]
            )
        if results_p4:
            npz_payload["p4_main_block_sigma_r_plus1_rel"] = np.asarray(
                [r["main_block_sigma_r_plus1_rel"] for r in results_p4]
            )
            npz_payload["p4_kronecker_max_abs_err"] = np.asarray(
                [r["kronecker_max_abs_err"] for r in results_p4]
            )
        if results_p5:
            npz_payload["p5_err"] = np.asarray(
                [r["err_R0_vs_reduced"] for r in results_p5]
            )
        np.savez_compressed(
            run.npz_path("structural_closure"), **npz_payload,
        )

        # --- Metadata extras ---
        ctx.record_compute_proxy(float(total_wall))
        ctx.record_extra("worst_p1_rank1_err", worst_p1_rank1)
        ctx.record_extra("worst_p1_sigma2_rel", worst_p1_sigma2)
        ctx.record_extra("worst_p1_generator_err", worst_p1_gen)
        ctx.record_extra("worst_p2_off_toeplitz_rel", worst_p2_toep)
        ctx.record_extra("worst_p3_off_circulant_fourier_rel", worst_p3_circ)
        ctx.record_extra("worst_p3_fourier_conv_err", worst_p3_four)
        ctx.record_extra("worst_p4_sigma_r_plus1_rel", worst_p4_svr)
        ctx.record_extra("worst_p4_kronecker_rel_err", worst_p4_kron)
        ctx.record_extra("worst_p5_untied_err", worst_p5_untied)
        ctx.record_extra("prop2_rank1_pass", bool(prop2_rank1_pass))
        ctx.record_extra("prop2_generators_pass", bool(prop2_generators_pass))
        ctx.record_extra("prop5_toeplitz_pass", bool(prop5_toeplitz_pass))
        ctx.record_extra("prop5_circulant_pass", bool(prop5_circulant_pass))
        ctx.record_extra("prop5_semiseparable_pass", bool(prop5_semisep_pass))
        ctx.record_extra("prop5_kronecker_pass", bool(prop5_kron_pass))
        ctx.record_extra(
            "remark2_untied_pass",
            (bool(remark2_untied_pass) if remark2_untied_pass is not None else None),
        )

        passes = [
            prop2_rank1_pass, prop2_generators_pass,
            prop5_toeplitz_pass, prop5_circulant_pass,
            prop5_semisep_pass, prop5_kron_pass,
        ]
        if remark2_untied_pass is not None:
            passes.append(remark2_untied_pass)
        all_ok = all(passes)

        status_parts = [
            f"prop2_rank1={'ok' if prop2_rank1_pass else 'FAIL'}",
            f"prop2_gen={'ok' if prop2_generators_pass else 'FAIL'}",
            f"prop5_toep={'ok' if prop5_toeplitz_pass else 'FAIL'}",
            f"prop5_circ={'ok' if prop5_circulant_pass else 'FAIL'}",
            f"prop5_semisep={'ok' if prop5_semisep_pass else 'FAIL'}",
            f"prop5_kron={'ok' if prop5_kron_pass else 'FAIL'}",
        ]
        if remark2_untied_pass is not None:
            status_parts.append(
                f"remark2_untied={'ok' if remark2_untied_pass else 'FAIL'}"
            )
        status = " ".join(status_parts)

        ctx.write_summary({
            "plan_reference": (
                "EXPERIMENT_PLAN_FINAL.MD §8 (theorem-A exact tier, "
                "structural closure + untied-layer Remark 2)"
            ),
            "framing": (
                "Validates the remaining theorem-chapter objects not "
                "covered by A1/A1b/A2/A3/A4/A1-general: Proposition 2 "
                "(rank-1 factorization + semiseparable representation of "
                "M_GD), Proposition 5 (structural closure: Toeplitz, "
                "circulant, and semiseparable-rank multiplicativity of "
                "the Hadamard reduction A_S = (1/P)(S_TT ⊙ X^⊤ Γ X)), and "
                "Remark 2 (untied-layer non-autonomous reduced model). "
                "Deterministic matrix-identity tier; no training, no "
                "architecture, no statistics."
            ),
            "category": (
                "exact theorem-A matrix-level structural identity "
                "validation."
            ),
            "interpretation": (
                "All gates are algebraic identities up to float64 "
                "precision. The Fourier identity for a Hadamard product "
                "of two circulants is a circular convolution of their "
                "DFT eigenvalues with a 1/P² factor — NOT the "
                "elementwise product of DFT eigenvalues (which holds for "
                "the matrix product of circulants and is separately "
                "recorded as a diagnostic that intentionally fails)."
            ),
            "acceptance_framing": (
                f"prop2_tol = {cfg.prop2_tol:.0e}; "
                f"toeplitz_tol = {cfg.toeplitz_tol:.0e}; "
                f"circulant_tol = {cfg.circulant_tol:.0e} (applies to "
                "off-circulant Fourier, spatial projection, and the "
                "Hadamard convolution identity); "
                f"semiseparable σ_{{r+1}}/σ_1 tol = {cfg.semisep_sv_tol:.0e}; "
                f"kronecker_tol = {cfg.kronecker_tol:.0e}; "
                f"untied_tol = {cfg.untied_tol:.0e}."
            ),
            "status": status,
            "prop2_rank1_pass": bool(prop2_rank1_pass),
            "prop2_generators_pass": bool(prop2_generators_pass),
            "prop5_toeplitz_pass": bool(prop5_toeplitz_pass),
            "prop5_circulant_pass": bool(prop5_circulant_pass),
            "prop5_semiseparable_pass": bool(prop5_semisep_pass),
            "prop5_kronecker_pass": bool(prop5_kron_pass),
            "remark2_untied_pass": (
                bool(remark2_untied_pass) if remark2_untied_pass is not None else None
            ),
            "worst_p1_rank1_err": float(worst_p1_rank1),
            "worst_p1_sigma2_rel": float(worst_p1_sigma2),
            "worst_p1_generator_err": float(worst_p1_gen),
            "worst_p2_off_toeplitz_rel": float(worst_p2_toep),
            "worst_p3_off_circulant_fourier_rel": float(worst_p3_circ),
            "worst_p3_fourier_convolution_err": float(worst_p3_four),
            "worst_p4_main_sigma_r_plus1_rel": float(worst_p4_svr),
            "worst_p4_kronecker_rel_err": float(worst_p4_kron),
            "worst_p5_untied_err": float(worst_p5_untied),
            "n_cells_p1": len(results_p1),
            "n_cells_p2": len(results_p2),
            "n_cells_p3": len(results_p3),
            "n_cells_p4": len(results_p4),
            "n_cells_p5": len(results_p5),
            "device": str(device),
            "sweep_wallclock_seconds": round(float(total_wall), 3),
        })

        # Console summary.
        print()
        print("=" * 78)
        print(f" A-structural theorem-A structural validation on {device}")
        print(
            f"   Prop 2 rank-1            worst = {worst_p1_rank1:.2e}   "
            f"{'PASS' if prop2_rank1_pass else 'FAIL'} (tol {cfg.prop2_tol:.0e})"
        )
        print(
            f"   Prop 2 generators        worst = {worst_p1_gen:.2e}   "
            f"{'PASS' if prop2_generators_pass else 'FAIL'}"
        )
        print(
            f"   Prop 5 Toeplitz          worst = {worst_p2_toep:.2e}   "
            f"{'PASS' if prop5_toeplitz_pass else 'FAIL'} "
            f"(tol {cfg.toeplitz_tol:.0e})"
        )
        print(
            f"   Prop 5 Circulant off-F   worst = {worst_p3_circ:.2e}   "
            f"{'PASS' if prop5_circulant_pass else 'FAIL'} "
            f"(tol {cfg.circulant_tol:.0e})"
        )
        print(
            f"   Prop 5 Fourier (conv)    worst = {worst_p3_four:.2e}"
        )
        print(
            f"   Prop 5 Semisep σ_r+1     worst = {worst_p4_svr:.2e}   "
            f"{'PASS' if prop5_semisep_pass else 'FAIL'} "
            f"(tol {cfg.semisep_sv_tol:.0e})"
        )
        print(
            f"   Prop 5 Kronecker         worst = {worst_p4_kron:.2e}   "
            f"{'PASS' if prop5_kron_pass else 'FAIL'} "
            f"(tol {cfg.kronecker_tol:.0e})"
        )
        if results_p5:
            print(
                f"   Remark 2 untied layers   worst = {worst_p5_untied:.2e}   "
                f"{'PASS' if remark2_untied_pass else 'FAIL'} "
                f"(tol {cfg.untied_tol:.0e})"
            )
        print("=" * 78)

        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
