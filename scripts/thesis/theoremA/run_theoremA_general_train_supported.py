"""Experiment A1-general: theorem-A exact reduction for ALL train-supported mixers.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §8.1 (extension beyond the
GD-compatible special case verified by A1 / A1b).

Theorem-level framing
---------------------
A1 and A1b validate Theorem 1 (``thm:theoremA_exact_SGamma``) and Corollary 1
(``cor:theoremA_GD_recovery``) in the **GD-compatible special case** only.
This script extends the verification to Theorem 1 in its **full generality**,
covering every train-supported structured mixer, and simultaneously validates
Proposition 3 (``prop:theoremA_necessity``): the reduced-Γ feature-space
predictor is recovered **if and only if** the mixer is GD-compatible.

The theorem chapter statements tested here:

* **Theorem 1** — for any train-supported ``S`` (``S_TQ = 0``, ``S_QQ = 0``),
  defining ``A_S(X,Γ) = (1/P)(S_TT ⊙ X^⊤ Γ X)`` and
  ``B_S(X_⋆,X,Γ) = (1/P)(S_QT ⊙ X_⋆^⊤ Γ X)``, the L-layer structured forward
  pass equals the reduced ``(A_S, B_S)`` recursion
  ``r^{ℓ+1} = (I + L^{-1} A_S) r^ℓ`` with query readout
  ``F = (1/L) B_S ∑_{ℓ=0..L-1} (I + L^{-1} A_S)^ℓ y``.

* **Corollary 1** — when ``S`` is GD-compatible the reduced representation
  collapses to the feature-space reduced-Γ predictor
  ``F = (1/LP) X_⋆^⊤ Γ ∑ (I − L^{-1} Σ̂ Γ)^ℓ X y``.

* **Proposition 3** — the converse: if the reduced-Γ predictor coincides with
  the structured forward pass for **every** ``(X, X_⋆, y, Γ)``, then ``S``
  must be GD-compatible. The experimental contrapositive: for **any** non-GD
  train-supported ``S``, the reduced-Γ predictor (R3) must differ from the
  true ``(A_S, B_S)`` recursion (R2) by a non-trivial amount.

Sweep axes
----------
Four structurally distinct train-supported mask types are tested:

1. ``gd_compatible`` — ``S_TT = −𝟏𝟏^⊤``, ``S_QT = +𝟏𝟏^⊤`` (control).
2. ``lower_triangular`` — ``S_TT`` with ``−1`` on/below diagonal, ``0`` above;
   ``S_QT = 𝟏𝟏^⊤``. Train-supported but **not** GD-compatible.
3. ``random_dense`` — both ``S_TT`` and ``S_QT`` sampled iid ``N(0, 1)``.
   Train-supported but not GD-compatible.
4. ``near_gd`` — ``S_TT = −𝟏𝟏^⊤ + ε E``, ``S_QT = 𝟏𝟏^⊤ + ε E'`` with ``E, E'``
   iid ``N(0, 1)`` and ``ε ∈ {0.01, 0.1, 0.5}``. Not GD-compatible for ``ε>0``;
   exhibits smooth degradation of Corollary 1 as ``ε → 0``.

Three Γ types (Γ :=  α_v W_x^⊤ W_q^⊤ W_k W_x is not assumed symmetric by
theorem A; ``random_nonsymmetric`` is the load-bearing test that the bilinear
form ``x^⊤ Γ x'`` is implemented with the correct ``W_q / W_k`` ordering):

1. ``identity`` — ``Γ = I``.
2. ``random_symmetric`` — ``Γ = A^⊤ A`` for random ``A``.
3. ``random_nonsymmetric`` — dense random ``Γ`` with no symmetry constraint.

Two covariance settings:

1. ``isotropic`` — ``Σ = Ω = I``.
2. ``structured`` — ``Σ = diag(k^{−1})`` and ``Ω = diag(k^{−0.5})``.

For each ``(mask, γ, σ)`` family, sweep ``D × P × K × L``.

Four prediction routes per cell
-------------------------------
* **R0 — Full hidden-state forward.** Builds the full ``(P+K)×(P+K)`` bilinear
  score ``[x_μ^⊤ Γ x_ν / P]`` from ``X = [X_train | X_query]``, applies the
  train-supported mask ``S``, and runs ``L`` explicit residual-stream layer
  updates on a length-``(P+K)`` scalar hidden channel. Does **not** consume
  ``A_S, B_S``.
* **R1 — Iterative reduced recursion.** Computes ``A_S, B_S`` from the
  theorem formulas and iterates ``r^{ℓ+1} = (I + L^{-1} A_S) r^ℓ``,
  accumulating ``Σ T^ℓ y`` via vector updates.
* **R2 — Closed-form reduced recursion.** Same ``A_S, B_S`` but materializes
  ``T^ℓ`` as matrix powers and forms the matrix geometric sum
  ``M = I + T + T^2 + ⋯ + T^{L−1}`` explicitly, then applies ``(1/L) B_S M y``.
  R1 and R2 are structurally different code paths; their agreement is a
  consistency cross-check for the reduced recursion.
* **R3 — Feature-space reduced-Γ predictor (GD-only).** The preconditioned-GD
  formula ``f(x_⋆) = (1/LP) x_⋆^⊤ Γ ∑ (I − L^{-1} Σ̂ Γ)^ℓ X y``. Knows nothing
  about the mask ``S``. For a GD-compatible ``S`` this equals R0/R1/R2;
  for any other train-supported ``S`` it generically does not (Proposition 3).

Acceptance gates
----------------
* **theorem1_pass** — for every cell (all mask types): ``max(err_R0_R1,
  err_R0_R2, err_R1_R2) ≤ 1e-10``. Theorem 1 holds for every train-supported
  ``S``.
* **corollary1_pass** — for ``gd_compatible`` cells only: ``max(err_R0_R3,
  err_R2_R3) ≤ 1e-10``. Corollary 1's reduced-Γ collapse.
* **necessity_pass** — for every non-GD mask kind, the **maximum**
  ``err_R2_R3`` across that kind's cells exceeds ``1e-3``. This is the
  experimental contrapositive to Proposition 3: Proposition 3 requires the
  structured forward pass to coincide with the reduced-Γ predictor for
  *every* ``(L, D, X, X_⋆, y, Γ)``; the contrapositive therefore requires
  witnessing at least one ``(L, D, X, X_⋆, y, Γ)`` where the two differ by a
  non-trivial amount — not every cell. (Per-cell caveat: at ``L=1`` the
  reduced prediction ``F = B_S y`` only touches ``S_QT``; for mask kinds
  that leave ``S_QT = 𝟏_K 𝟏_P^⊤`` unchanged — e.g. ``lower_triangular`` —
  R2 coincides with R3 at ``L=1`` regardless of ``S_TT``. This is
  consistent with Proposition 3 and is recorded as a diagnostic only.)
* **near-GD interpolation** — ``err_R2_R3`` is monotone in ``ε`` (diagnostic).

Run
---
::

    python -u scripts/thesis/theoremA/run_theoremA_general_train_supported.py \
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

from scripts.thesis.utils.data_generators import GAConfig, ga_generate
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


_MASK_SETTINGS: tuple[tuple[str, float | None], ...] = (
    ("gd_compatible", None),
    ("lower_triangular", None),
    ("random_dense", None),
    ("near_gd", 0.01),
    ("near_gd", 0.1),
    ("near_gd", 0.5),
)

_GAMMA_KINDS: tuple[str, ...] = (
    "identity",
    "random_symmetric",
    "random_nonsymmetric",
)

_SIGMA_KINDS: tuple[str, ...] = (
    "isotropic",
    "structured",
)


@dataclass(frozen=True)
class GeneralA1Config:
    """Frozen configuration for the generalized A1 sweep.

    Full cell count = ``|mask_settings| × |gamma_kinds| × |sigma_kinds|
    × |D| × |P| × |K| × |L|`` = 6 × 3 × 2 × 4 × 3 × 2 × 4 = 1728. Each cell
    is a few small matmuls so the full sweep finishes in well under a
    minute on CPU or GPU.
    """

    D_list: tuple[int, ...] = (8, 16, 32, 64)
    P_list: tuple[int, ...] = (8, 16, 32)
    K_list: tuple[int, ...] = (4, 8)
    L_list: tuple[int, ...] = (1, 2, 4, 8)

    mask_settings: tuple[tuple[str, float | None], ...] = _MASK_SETTINGS
    gamma_kinds: tuple[str, ...] = _GAMMA_KINDS
    sigma_kinds: tuple[str, ...] = _SIGMA_KINDS

    label_norm: str = "sqrt_D"   # theorem-A default per plan §3
    sigma_noise: float = 0.0
    B: int = 2
    base_seed: int = 0

    # Acceptance.
    machine_eps_tol: float = 1e-10
    necessity_threshold: float = 1e-3

    # Figure slices.
    heatmap_K: int = 8
    heatmap_L: int = 4
    heatmap_gamma_kind: str = "random_nonsymmetric"
    heatmap_sigma_kind: str = "structured"

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Σ / Ω / Γ / S construction
# ---------------------------------------------------------------------------


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _build_sigma_omega_specs(
    D: int, sigma_kind: str, dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (Sigma_kind, Sigma_params, Omega_kind, Omega_params) payloads for GA.

    Returns two dicts ``{"kind": ..., "params": {...}}``; caller passes them
    into GAConfig.
    """
    if sigma_kind == "isotropic":
        return (
            {"kind": "isotropic", "params": {}},
            {"kind": "isotropic", "params": {}},
        )
    if sigma_kind == "structured":
        k = torch.arange(1, D + 1, dtype=dtype)
        sigma_spec = (k.to(dtype=torch.float64)) ** (-1.0)       # λ_Σ[k] = k^{-1}
        omega_spec = (k.to(dtype=torch.float64)) ** (-0.5)       # λ_Ω[k] = k^{-0.5}
        return (
            {"kind": "diag_spectrum", "params": {"spec": sigma_spec.to(dtype)}},
            {"kind": "diag_spectrum", "params": {"spec": omega_spec.to(dtype)}},
        )
    raise ValueError(f"unknown sigma_kind: {sigma_kind!r}")


def _build_gamma(
    D: int, gamma_kind: str, seed: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Construct a (D, D) Γ matrix per the requested kind.

    Uses an explicit ``torch.Generator`` seed so Γ is deterministic per
    ``(D, gamma_kind, seed)``.
    """
    if gamma_kind == "identity":
        return torch.eye(D, dtype=dtype)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if gamma_kind == "random_symmetric":
        A = torch.randn(D, D, generator=gen, dtype=dtype)
        Gamma = A.T @ A / float(D)               # symmetric PSD, O(1) scale
        return Gamma.contiguous()
    if gamma_kind == "random_nonsymmetric":
        Gamma = torch.randn(D, D, generator=gen, dtype=dtype) / math.sqrt(float(D))
        return Gamma.contiguous()
    raise ValueError(f"unknown gamma_kind: {gamma_kind!r}")


def _build_S_blocks(
    P: int, K: int, mask_kind: str, epsilon: float | None, seed: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(S_TT, S_QT)`` for the requested train-supported mask.

    Returns two tensors of shapes ``(P, P)`` and ``(K, P)``. By construction
    both ``S_TQ`` and ``S_QQ`` are zero, so the full ``S ∈ ℝ^{(P+K)×(P+K)}``
    assembled as ``[[S_TT, 0], [S_QT, 0]]`` is always train-supported.
    """
    ones_PP = torch.ones(P, P, dtype=dtype)
    ones_KP = torch.ones(K, P, dtype=dtype)
    if mask_kind == "gd_compatible":
        return -ones_PP, ones_KP
    if mask_kind == "lower_triangular":
        # S_TT = -1 on and below diagonal, 0 above. S_QT = +1 (as in GD).
        S_TT = -torch.tril(ones_PP)
        return S_TT, ones_KP
    if mask_kind == "random_dense":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        S_TT = torch.randn(P, P, generator=gen, dtype=dtype)
        S_QT = torch.randn(K, P, generator=gen, dtype=dtype)
        return S_TT, S_QT
    if mask_kind == "near_gd":
        if epsilon is None:
            raise ValueError("near_gd requires epsilon")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        E_TT = torch.randn(P, P, generator=gen, dtype=dtype)
        E_QT = torch.randn(K, P, generator=gen, dtype=dtype)
        S_TT = -ones_PP + float(epsilon) * E_TT
        S_QT = ones_KP + float(epsilon) * E_QT
        return S_TT, S_QT
    raise ValueError(f"unknown mask_kind: {mask_kind!r}")


# ---------------------------------------------------------------------------
# Four forward routes
# ---------------------------------------------------------------------------


def _assemble_S_full(
    S_TT: torch.Tensor, S_QT: torch.Tensor, P: int, K: int,
) -> torch.Tensor:
    """Assemble full ``(P+K)×(P+K)`` train-supported S with zero query columns."""
    dtype = S_TT.dtype
    device = S_TT.device
    S = torch.zeros(P + K, P + K, dtype=dtype, device=device)
    S[:P, :P] = S_TT
    S[P:, :P] = S_QT
    return S


def _route_R0_full_hidden_state(
    X_train: torch.Tensor,          # (B, D, P)
    X_query: torch.Tensor,          # (B, D, K)
    Gamma: torch.Tensor,            # (D, D)
    y_train: torch.Tensor,          # (B, P)
    S_TT: torch.Tensor,             # (P, P)
    S_QT: torch.Tensor,             # (K, P)
    L: int,
    P_norm: int,
) -> torch.Tensor:
    """Full-sequence ``(P+K)``-length residual stream with general train-supported S.

    Implements Lemma 1 + Theorem 1 exactly: scalar channel updates
    ``Δ_μ^{ℓ+1} = Δ_μ^ℓ + (1/(LP)) ∑_ν S_{μν} (x_μ^⊤ Γ x_ν) Δ_ν^ℓ`` on the
    full-sequence tensor. The bilinear form ``x_μ^⊤ Γ x_ν`` is computed as
    ``X^⊤ Γ X`` with the Γ-ordering required by the theorem (Γ is NOT
    symmetrized).
    """
    B = int(X_train.shape[0])
    P = int(X_train.shape[-1])
    K = int(X_query.shape[-1])
    dtype = X_train.dtype
    device = X_train.device

    X = torch.cat([X_train, X_query], dim=-1)                     # (B, D, P+K)
    # S_pos[μ, ν] = x_μ^⊤ Γ x_ν / P. Critical: Γ acts on the right index, X^⊤
    # on the left — i.e. (X^⊤)(Γ)(X), not a symmetrized version.
    GammaX = torch.einsum("de,bef->bdf", Gamma, X)                # (B, D, P+K)
    S_pos = torch.einsum("bdm,bdn->bmn", X, GammaX) / float(P_norm)

    S_full = _assemble_S_full(S_TT, S_QT, P, K)                   # (P+K, P+K)
    M_eff = S_pos * S_full.unsqueeze(0)                           # (B, P+K, P+K)

    h = torch.cat(
        [y_train, torch.zeros(B, K, dtype=dtype, device=device)],
        dim=-1,
    )  # (B, P+K)
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        h = h + inv_L * torch.einsum("bmn,bn->bm", M_eff, h)
    return h[:, P:]


def _build_A_B_from_S(
    X_train: torch.Tensor, X_query: torch.Tensor, Gamma: torch.Tensor,
    S_TT: torch.Tensor, S_QT: torch.Tensor, P_norm: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``A_S = (1/P)(S_TT ⊙ X^⊤ Γ X)`` and
    ``B_S = (1/P)(S_QT ⊙ X_⋆^⊤ Γ X)`` per Theorem 1.
    """
    K_train = torch.einsum("bim,ij,bjn->bmn", X_train, Gamma, X_train) / float(P_norm)
    K_query = torch.einsum("bim,ij,bjn->bmn", X_query, Gamma, X_train) / float(P_norm)
    A_S = S_TT.unsqueeze(0) * K_train                             # (B, P, P)
    B_S = S_QT.unsqueeze(0) * K_query                             # (B, K, P)
    return A_S, B_S


def _route_R1_iterative_reduced(
    A_S: torch.Tensor, B_S: torch.Tensor, y_train: torch.Tensor, L: int,
) -> torch.Tensor:
    """R1: accumulate ``∑ T^ℓ y`` via vector iteration; ``F = (1/L) B_S sum``."""
    z = y_train.clone()
    sum_T_y = torch.zeros_like(z)
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        sum_T_y = sum_T_y + z
        z = z + inv_L * torch.einsum("bpi,bi->bp", A_S, z)
    return inv_L * torch.einsum("bki,bi->bk", B_S, sum_T_y)


def _route_R2_matrix_power_reduced(
    A_S: torch.Tensor, B_S: torch.Tensor, y_train: torch.Tensor, L: int,
) -> torch.Tensor:
    """R2: materialize matrix powers ``T^ℓ`` and form the geometric sum.

    Structurally distinct from R1: builds ``T, T^2, …, T^{L-1}`` explicitly
    and sums them before applying to y. Agreement with R1 to float eps
    cross-checks the reduced recursion implementation.
    """
    B, P, _ = A_S.shape
    dtype = A_S.dtype
    device = A_S.device
    inv_L = 1.0 / float(L)
    I_P = torch.eye(P, dtype=dtype, device=device).expand(B, P, P).contiguous()
    T = I_P + A_S * inv_L                                        # (B, P, P)
    M_sum = I_P.clone()                                          # l = 0 term
    T_pow = I_P.clone()
    for _ in range(int(L) - 1):
        T_pow = torch.einsum("bij,bjk->bik", T_pow, T)
        M_sum = M_sum + T_pow
    sum_T_y = torch.einsum("bpi,bi->bp", M_sum, y_train)
    return inv_L * torch.einsum("bki,bi->bk", B_S, sum_T_y)


def _route_R3_feature_space_gamma(
    X_train: torch.Tensor, X_query: torch.Tensor, Gamma: torch.Tensor,
    y_train: torch.Tensor, L: int, P_norm: int,
) -> torch.Tensor:
    """R3: feature-space preconditioned-GD reduced-Γ predictor.

    ``f(x_⋆) = (1/LP) x_⋆^⊤ Γ ∑ (I − L^{-1} Σ̂ Γ)^ℓ X y``.

    This formula knows nothing about the mask S — it always implements the
    GD-compatible predictor. For a GD-compatible S it equals R0/R1/R2; for
    any non-GD train-supported S it generically does not (Proposition 3).
    """
    B = int(X_train.shape[0])
    D = int(X_train.shape[1])
    inv_L = 1.0 / float(L)
    inv_P = 1.0 / float(P_norm)
    w = torch.zeros(B, D, dtype=y_train.dtype, device=y_train.device)
    for _ in range(int(L)):
        Xt_w = torch.einsum("bdp,bd->bp", X_train, w)
        r = y_train - Xt_w
        Xr = torch.einsum("bdp,bp->bd", X_train, r)
        GXr = torch.einsum("de,be->bd", Gamma, Xr)
        w = w + inv_L * inv_P * GXr
    return torch.einsum("bdk,bd->bk", X_query, w)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def _run_trial(
    cfg: GeneralA1Config,
    mask_kind: str, epsilon: float | None,
    gamma_kind: str,
    sigma_kind: str,
    D: int, P: int, K: int, L: int,
    device: torch.device,
) -> dict[str, Any]:
    dtype = _torch_dtype(cfg.dtype)
    # Seeds derived from all sweep axes. Γ seed depends only on (D, γ) so Γ
    # is consistent across P/K/L for a fixed Γ family. S seed depends on
    # (P, K, mask, ε) so masks are consistent across D/γ/σ.
    seed_x = (
        int(cfg.base_seed) + 1
        + 10_000 * D + 1_000 * P + 100 * K + 10 * L
        + hash((mask_kind, str(epsilon), gamma_kind, sigma_kind)) % 10_000
    )
    seed_beta = seed_x + 7
    seed_noise = seed_x + 13
    seed_gamma = int(cfg.base_seed) + 101 + 23 * D + 7 * hash(gamma_kind) % 10_000
    seed_S = int(cfg.base_seed) + 31 + 11 * P + 3 * K + 7 * hash(
        (mask_kind, str(epsilon))
    ) % 10_000

    Sigma_d, Omega_d = _build_sigma_omega_specs(int(D), sigma_kind, dtype)
    Gamma_cpu = _build_gamma(int(D), gamma_kind, seed=seed_gamma, dtype=dtype)

    # Call GA to sample (X_train, X_query, β, y_train) with the requested
    # Σ, Ω, Γ. We discard GA's mask / A_S_GD / B_S_GD — we build our own
    # ``S_TT, S_QT`` for the general train-supported mask.
    g = GAConfig(
        D=int(D), P=int(P), K=int(K), B=int(cfg.B),
        Sigma_kind=Sigma_d["kind"], Sigma_params=Sigma_d["params"],
        Omega_kind=Omega_d["kind"], Omega_params=Omega_d["params"],
        Gamma_kind="full_matrix", Gamma_params={"matrix": Gamma_cpu},
        label_norm=cfg.label_norm,
        sigma=float(cfg.sigma_noise),
        mask_kind="gd_compatible",                # irrelevant here
        L=int(L),
        return_feature_space=False,
        seeds={"x": seed_x, "beta": seed_beta, "noise": seed_noise, "mask": 0},
        dtype=cfg.dtype,
        device="cpu",
    )
    op = ga_generate(g)
    X_train = op["X_train"].to(device)
    X_query = op["X_query"].to(device)
    y_train = op["y_train"].to(device)
    Gamma = op["Gamma"].to(device)

    # Build custom S blocks for this mask kind.
    S_TT, S_QT = _build_S_blocks(
        int(P), int(K), mask_kind, epsilon, seed=seed_S, dtype=dtype,
    )
    S_TT = S_TT.to(device)
    S_QT = S_QT.to(device)

    # Reduced operators derived from the theorem formulas.
    A_S, B_S = _build_A_B_from_S(X_train, X_query, Gamma, S_TT, S_QT, P_norm=int(P))

    # Forward routes.
    t0 = time.perf_counter()
    f_R0 = _route_R0_full_hidden_state(
        X_train, X_query, Gamma, y_train, S_TT, S_QT, L=int(L), P_norm=int(P),
    )
    t_R0 = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R1 = _route_R1_iterative_reduced(A_S, B_S, y_train, L=int(L))
    t_R1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R2 = _route_R2_matrix_power_reduced(A_S, B_S, y_train, L=int(L))
    t_R2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R3 = _route_R3_feature_space_gamma(
        X_train, X_query, Gamma, y_train, L=int(L), P_norm=int(P),
    )
    t_R3 = time.perf_counter() - t0

    err_R0_R1 = reduced_model_error(f_R0, f_R1)
    err_R0_R2 = reduced_model_error(f_R0, f_R2)
    err_R1_R2 = reduced_model_error(f_R1, f_R2)
    err_R0_R3 = reduced_model_error(f_R0, f_R3)
    err_R2_R3 = reduced_model_error(f_R2, f_R3)

    theorem1_err = max(
        float(err_R0_R1), float(err_R0_R2), float(err_R1_R2)
    )

    return {
        "mask_kind": mask_kind,
        "epsilon": (None if epsilon is None else float(epsilon)),
        "gamma_kind": gamma_kind,
        "sigma_kind": sigma_kind,
        "D": int(D), "P": int(P), "K": int(K), "L": int(L),
        "err_R0_R1": float(err_R0_R1),
        "err_R0_R2": float(err_R0_R2),
        "err_R1_R2": float(err_R1_R2),
        "err_R0_R3": float(err_R0_R3),
        "err_R2_R3": float(err_R2_R3),
        "theorem1_err": theorem1_err,
        "t_R0_seconds": float(t_R0),
        "t_R1_seconds": float(t_R1),
        "t_R2_seconds": float(t_R2),
        "t_R3_seconds": float(t_R3),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


_MASK_LABELS = {
    ("gd_compatible", None): "gd_compatible",
    ("lower_triangular", None): "lower_triangular",
    ("random_dense", None): "random_dense",
    ("near_gd", 0.01): r"near_gd ($\epsilon$=0.01)",
    ("near_gd", 0.1): r"near_gd ($\epsilon$=0.1)",
    ("near_gd", 0.5): r"near_gd ($\epsilon$=0.5)",
}


def _mask_key(trial: dict[str, Any]) -> tuple[str, float | None]:
    return (trial["mask_kind"], trial["epsilon"])


def _plot_theorem1_errors(
    cfg: GeneralA1Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 1: histogram of max-of-Theorem-1 errors across all cells."""
    import matplotlib.pyplot as plt

    floor = 1e-18
    bins = np.logspace(-18, -6, 40)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    mask_settings = list(cfg.mask_settings)
    colors = sequential_colors(len(mask_settings))
    for (mk, eps), color in zip(mask_settings, colors):
        sub = [t for t in trials if _mask_key(t) == (mk, eps)]
        if not sub:
            continue
        vals = np.clip([t["theorem1_err"] for t in sub], floor, None)
        ax.hist(
            vals, bins=bins, alpha=0.55, label=_MASK_LABELS[(mk, eps)],
            edgecolor="black", lw=0.3, color=color,
        )
    ax.axvline(
        cfg.machine_eps_tol, color="red", lw=1.0, ls="--",
        label=f"theorem1 tol = {cfg.machine_eps_tol:.0e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"max rel. error across $\{R_0\text{–}R_1, R_0\text{–}R_2, R_1\text{–}R_2\}$")
    ax.set_ylabel("count")
    ax.set_title(
        f"A1-general: Theorem 1 exactness across {len(trials)} cells "
        "(all train-supported S)",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    save_both(fig, run_dir, "a1_general_theorem1_errors")
    plt.close(fig)


def _plot_necessity(
    cfg: GeneralA1Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 2: err_R2_R3 grouped by mask setting (validates Proposition 3)."""
    import matplotlib.pyplot as plt

    floor = 1e-18
    mask_settings = list(cfg.mask_settings)
    data = []
    labels = []
    for (mk, eps) in mask_settings:
        sub = [t for t in trials if _mask_key(t) == (mk, eps)]
        if not sub:
            continue
        vals = np.clip([t["err_R2_R3"] for t in sub], floor, None)
        data.append(vals)
        labels.append(_MASK_LABELS[(mk, eps)])

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    positions = np.arange(1, len(data) + 1)
    # Boxplot without outlier dots; jittered scatter on top.
    bp = ax.boxplot(
        data, positions=positions, widths=0.55, showfliers=False,
        medianprops={"color": "black"},
    )
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            positions[i] + jitter, vals, s=8, alpha=0.35, color="C0",
            edgecolors="none",
        )
    ax.axhline(
        cfg.machine_eps_tol, color="green", lw=0.9, ls=":",
        label=rf"corollary1 tol = {cfg.machine_eps_tol:.0e}",
    )
    ax.axhline(
        cfg.necessity_threshold, color="red", lw=0.9, ls="--",
        label=rf"necessity floor = {cfg.necessity_threshold:.0e}",
    )
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"rel. error ($R_2$ vs $R_3$)")
    ax.set_title(
        "A1-general necessity (Prop. 3): $R_2 \\neq R_3$ for non-GD masks",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_both(fig, run_dir, "a1_general_necessity")
    plt.close(fig)


def _plot_near_gd_interpolation(
    cfg: GeneralA1Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 3: err_R2_R3 vs ε for the near_gd family, broken out by γ-kind."""
    import matplotlib.pyplot as plt

    near = [t for t in trials if t["mask_kind"] == "near_gd"]
    if not near:
        return
    eps_values = sorted({float(t["epsilon"]) for t in near})
    gamma_kinds = list(cfg.gamma_kinds)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    colors = sequential_colors(len(gamma_kinds))
    for gk, color in zip(gamma_kinds, colors):
        med = []
        lo = []
        hi = []
        for e in eps_values:
            sub = [
                t["err_R2_R3"] for t in near
                if float(t["epsilon"]) == e and t["gamma_kind"] == gk
            ]
            arr = np.asarray(sub, dtype=float) if sub else np.array([np.nan])
            med.append(float(np.nanmedian(arr)))
            lo.append(float(np.nanquantile(arr, 0.1)))
            hi.append(float(np.nanquantile(arr, 0.9)))
        ax.plot(
            eps_values, med, color=color, marker="o", lw=1.5, ms=5.0,
            label=gk,
        )
        ax.fill_between(eps_values, lo, hi, color=color, alpha=0.15)
    ax.axhline(
        cfg.necessity_threshold, color="red", lw=0.9, ls="--",
        label=rf"necessity floor = {cfg.necessity_threshold:.0e}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"mask perturbation $\epsilon$")
    ax.set_ylabel(r"rel. error ($R_2$ vs $R_3$)")
    ax.set_title(
        "A1-general near-GD interpolation: degradation of Corollary 1 with $\\epsilon$",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a1_general_near_gd_interpolation")
    plt.close(fig)


def _plot_gamma_sensitivity(
    cfg: GeneralA1Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 4: GD-compatible Theorem-1 error across γ-kinds (W_q/W_k ordering check)."""
    import matplotlib.pyplot as plt

    floor = 1e-18
    gd_trials = [t for t in trials if t["mask_kind"] == "gd_compatible"]
    if not gd_trials:
        return
    gamma_kinds = list(cfg.gamma_kinds)
    bins = np.logspace(-18, -6, 40)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = sequential_colors(len(gamma_kinds))
    for gk, color in zip(gamma_kinds, colors):
        sub = [t for t in gd_trials if t["gamma_kind"] == gk]
        if not sub:
            continue
        vals = np.clip([t["theorem1_err"] for t in sub], floor, None)
        ax.hist(
            vals, bins=bins, alpha=0.55, label=gk,
            edgecolor="black", lw=0.3, color=color,
        )
    ax.axvline(
        cfg.machine_eps_tol, color="red", lw=1.0, ls="--",
        label=f"tol = {cfg.machine_eps_tol:.0e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"max rel. error across $\{R_0\text{–}R_1, R_0\text{–}R_2, R_1\text{–}R_2\}$")
    ax.set_ylabel("count")
    ax.set_title(
        "A1-general $\\Gamma$ sensitivity (GD-compatible): all $\\gamma$ families must match",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a1_general_gamma_sensitivity")
    plt.close(fig)


def _plot_error_vs_L(
    cfg: GeneralA1Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 5: max(err_R0_R1, err_R0_R2, err_R1_R2) vs L, broken out by mask."""
    import matplotlib.pyplot as plt

    floor = 1e-18
    L_list = list(cfg.L_list)
    mask_settings = list(cfg.mask_settings)
    colors = sequential_colors(len(mask_settings))

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for (mk, eps), color in zip(mask_settings, colors):
        sub_all = [t for t in trials if _mask_key(t) == (mk, eps)]
        if not sub_all:
            continue
        y_vals = []
        for L in L_list:
            sub = [t for t in sub_all if int(t["L"]) == int(L)]
            if not sub:
                y_vals.append(np.nan)
                continue
            y_vals.append(max(t["theorem1_err"] for t in sub))
        y_plot = np.clip(np.asarray(y_vals, dtype=float), floor, None)
        ax.plot(
            L_list, y_plot, color=color, marker="o", ms=4.5, lw=1.3,
            label=_MASK_LABELS[(mk, eps)],
        )
    ax.axhline(
        cfg.machine_eps_tol, color="red", lw=0.9, ls="--",
        label=f"tol = {cfg.machine_eps_tol:.0e}",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("depth L")
    ax.set_ylabel(r"max Theorem-1 rel. error across (D, P, K, $\gamma$, $\sigma$)")
    ax.set_title(
        "A1-general error growth with depth L", fontsize=10,
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    save_both(fig, run_dir, "a1_general_error_vs_L")
    plt.close(fig)


def _plot_heatmap(
    cfg: GeneralA1Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Figure 6: per-mask-setting max Theorem-1 error over (D, P) at fixed (K, L).

    Pinned to ``(gamma_kind='random_nonsymmetric', sigma_kind='structured')``
    by default — the hardest cell of the sweep. If the user changed the
    config axes, the heatmap picks the configured representative cell.
    """
    import matplotlib.pyplot as plt

    K = int(cfg.heatmap_K)
    L = int(cfg.heatmap_L)
    if K not in cfg.K_list or L not in cfg.L_list:
        return
    gk = cfg.heatmap_gamma_kind
    sk = cfg.heatmap_sigma_kind
    if gk not in cfg.gamma_kinds or sk not in cfg.sigma_kinds:
        return

    mask_settings = list(cfg.mask_settings)
    D_list = list(cfg.D_list)
    P_list = list(cfg.P_list)
    floor = 1e-18

    n = len(mask_settings)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.3 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    for idx, (mk, eps) in enumerate(mask_settings):
        ax = axes[idx // n_cols, idx % n_cols]
        grid = np.full((len(D_list), len(P_list)), floor, dtype=float)
        for t in trials:
            if _mask_key(t) != (mk, eps): continue
            if t["gamma_kind"] != gk or t["sigma_kind"] != sk: continue
            if int(t["K"]) != K or int(t["L"]) != L: continue
            i = D_list.index(int(t["D"]))
            j = P_list.index(int(t["P"]))
            grid[i, j] = max(grid[i, j], t["theorem1_err"])
        grid_plot = np.where(grid > floor, grid, floor)
        phase_heatmap(
            ax, grid_plot,
            x_coords=np.asarray(P_list, dtype=float),
            y_coords=np.asarray(D_list, dtype=float),
            xlabel="P", ylabel="D",
            cbar_label=r"theorem-1 err",
            log_z=True, log_x=True, log_y=True,
        )
        ax.set_title(_MASK_LABELS[(mk, eps)], fontsize=9)
    # Hide unused subplots.
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle(
        f"A1-general Theorem-1 error by mask kind ($\\gamma$={gk}, $\\sigma$={sk}, K={K}, L={L})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_both(fig, run_dir, "a1_general_heatmap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment A1-general: Theorem-1 exactness across all "
            "train-supported masks + Proposition-3 necessity of "
            "GD-compatibility (plan §8.1 extension)."
        )
    )
    p.add_argument(
        "--device", type=str, default="cuda", choices=("cpu", "cuda", "auto")
    )
    p.add_argument(
        "--dtype", type=str, default="float64", choices=("float32", "float64")
    )
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--D-list", type=str, default=None)
    p.add_argument("--P-list", type=str, default=None)
    p.add_argument("--K-list", type=str, default=None)
    p.add_argument("--L-list", type=str, default=None)
    p.add_argument(
        "--quick", action="store_true",
        help="Tiny sweep for smoke-testing (one cell per family).",
    )
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> GeneralA1Config:
    base = GeneralA1Config()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.D_list is not None:
        overrides["D_list"] = _parse_list_ints(args.D_list)
    if args.P_list is not None:
        overrides["P_list"] = _parse_list_ints(args.P_list)
    if args.K_list is not None:
        overrides["K_list"] = _parse_list_ints(args.K_list)
    if args.L_list is not None:
        overrides["L_list"] = _parse_list_ints(args.L_list)
    if args.quick:
        overrides.update(
            D_list=(16,), P_list=(16,), K_list=(4,), L_list=(1, 4),
            heatmap_K=4, heatmap_L=4,
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


def _compute_aggregates(
    cfg: GeneralA1Config, trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-axis worst-case summaries used in summary.txt."""
    per_mask: dict[str, dict[str, float]] = {}
    for (mk, eps) in cfg.mask_settings:
        sub = [t for t in trials if _mask_key(t) == (mk, eps)]
        if not sub:
            continue
        key = f"{mk}" if eps is None else f"{mk}:eps={eps}"
        per_mask[key] = {
            "n_cells": len(sub),
            "worst_theorem1_err": max(t["theorem1_err"] for t in sub),
            "worst_R2_R3": max(t["err_R2_R3"] for t in sub),
            "min_R2_R3": min(t["err_R2_R3"] for t in sub),
            "median_R2_R3": float(np.median([t["err_R2_R3"] for t in sub])),
            "worst_R0_R3": max(t["err_R0_R3"] for t in sub),
        }

    per_gamma: dict[str, float] = {}
    for gk in cfg.gamma_kinds:
        sub = [t for t in trials if t["gamma_kind"] == gk]
        if sub:
            per_gamma[gk] = max(t["theorem1_err"] for t in sub)

    per_sigma: dict[str, float] = {}
    for sk in cfg.sigma_kinds:
        sub = [t for t in trials if t["sigma_kind"] == sk]
        if sub:
            per_sigma[sk] = max(t["theorem1_err"] for t in sub)

    return {
        "per_mask": per_mask,
        "per_gamma": per_gamma,
        "per_sigma": per_sigma,
    }


def main() -> int:
    args = _cli()
    if args.no_show:
        matplotlib.use("Agg")

    cfg = _config_from_cli(args)
    device = _resolve_device(cfg.device)
    print(f"[A1-general] device = {device}")

    run = ThesisRunDir(__file__, phase="theoremA")
    with RunContext(
        run,
        config=cfg,
        seeds=[cfg.base_seed, cfg.base_seed + 7,
               cfg.base_seed + 13, cfg.base_seed + 19,
               cfg.base_seed + 31, cfg.base_seed + 101],
        notes=(
            "A1-general: Theorem-1 exactness for every train-supported "
            "structured mixer (4 mask kinds including near-GD interpolation, "
            "3 Γ kinds including random-nonsymmetric, 2 covariance settings) "
            "plus Proposition-3 necessity check that the reduced-Γ predictor "
            "R3 genuinely fails outside the GD-compatible class."
        ),
    ) as ctx:
        apply_thesis_style()

        trials: list[dict[str, Any]] = []
        n_total = (
            len(cfg.mask_settings) * len(cfg.gamma_kinds) * len(cfg.sigma_kinds)
            * len(cfg.D_list) * len(cfg.P_list)
            * len(cfg.K_list) * len(cfg.L_list)
        )
        idx = 0
        t_sweep_start = time.perf_counter()
        for (mk, eps) in cfg.mask_settings:
            for gk in cfg.gamma_kinds:
                for sk in cfg.sigma_kinds:
                    for D in cfg.D_list:
                        for P in cfg.P_list:
                            for K in cfg.K_list:
                                for L in cfg.L_list:
                                    idx += 1
                                    t0 = time.perf_counter()
                                    trial = _run_trial(
                                        cfg, mk, eps, gk, sk,
                                        int(D), int(P), int(K), int(L), device,
                                    )
                                    dt = time.perf_counter() - t0
                                    ctx.record_step_time(dt)
                                    if idx <= 5 or idx % 200 == 0 or idx == n_total:
                                        eps_s = "-" if eps is None else f"{eps:g}"
                                        print(
                                            f"[{idx:>5d}/{n_total}] "
                                            f"mask={mk:<16s} eps={eps_s:<5s} "
                                            f"γ={gk:<22s} σ={sk:<10s} "
                                            f"D={int(D):>3d} P={int(P):>3d} "
                                            f"K={int(K):>2d} L={int(L):>2d}  "
                                            f"T1={trial['theorem1_err']:.2e} "
                                            f"R2R3={trial['err_R2_R3']:.2e} "
                                            f"({dt*1000:.1f} ms)"
                                        )
                                    trials.append(trial)
        sweep_wall = time.perf_counter() - t_sweep_start
        print(
            f"[A1-general] swept {len(trials)} cells in "
            f"{sweep_wall:.2f} s"
        )

        # --- Figures ---
        _plot_theorem1_errors(cfg, trials, run)
        _plot_necessity(cfg, trials, run)
        _plot_near_gd_interpolation(cfg, trials, run)
        _plot_gamma_sensitivity(cfg, trials, run)
        _plot_error_vs_L(cfg, trials, run)
        _plot_heatmap(cfg, trials, run)

        # --- Per-cell JSON ---
        rows = [
            {
                "mask_kind": t["mask_kind"],
                "epsilon": t["epsilon"],
                "gamma_kind": t["gamma_kind"],
                "sigma_kind": t["sigma_kind"],
                "D": t["D"], "P": t["P"], "K": t["K"], "L": t["L"],
                "err_R0_R1": float(t["err_R0_R1"]),
                "err_R0_R2": float(t["err_R0_R2"]),
                "err_R1_R2": float(t["err_R1_R2"]),
                "err_R0_R3": float(t["err_R0_R3"]),
                "err_R2_R3": float(t["err_R2_R3"]),
                "theorem1_err": float(t["theorem1_err"]),
            }
            for t in trials
        ]
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

        # --- Save npz of full error tensors ---
        D_list = list(cfg.D_list); P_list = list(cfg.P_list)
        K_list = list(cfg.K_list); L_list = list(cfg.L_list)
        gamma_kinds = list(cfg.gamma_kinds); sigma_kinds = list(cfg.sigma_kinds)
        mask_settings = list(cfg.mask_settings)
        mask_labels = [
            mk if eps is None else f"{mk}:eps={eps}"
            for (mk, eps) in mask_settings
        ]
        shape = (
            len(mask_settings), len(gamma_kinds), len(sigma_kinds),
            len(D_list), len(P_list), len(K_list), len(L_list),
        )
        err_R0_R1_grid = np.zeros(shape)
        err_R0_R2_grid = np.zeros(shape)
        err_R1_R2_grid = np.zeros(shape)
        err_R0_R3_grid = np.zeros(shape)
        err_R2_R3_grid = np.zeros(shape)
        for t in trials:
            im = mask_settings.index((t["mask_kind"], t["epsilon"]))
            ig = gamma_kinds.index(t["gamma_kind"])
            isg = sigma_kinds.index(t["sigma_kind"])
            iD = D_list.index(int(t["D"]))
            iP = P_list.index(int(t["P"]))
            iK = K_list.index(int(t["K"]))
            iL = L_list.index(int(t["L"]))
            idx_tup = (im, ig, isg, iD, iP, iK, iL)
            err_R0_R1_grid[idx_tup] = t["err_R0_R1"]
            err_R0_R2_grid[idx_tup] = t["err_R0_R2"]
            err_R1_R2_grid[idx_tup] = t["err_R1_R2"]
            err_R0_R3_grid[idx_tup] = t["err_R0_R3"]
            err_R2_R3_grid[idx_tup] = t["err_R2_R3"]
        np.savez_compressed(
            run.npz_path("general_train_supported"),
            mask_labels=np.asarray(mask_labels),
            gamma_kinds=np.asarray(gamma_kinds),
            sigma_kinds=np.asarray(sigma_kinds),
            D_list=np.asarray(D_list, dtype=np.int64),
            P_list=np.asarray(P_list, dtype=np.int64),
            K_list=np.asarray(K_list, dtype=np.int64),
            L_list=np.asarray(L_list, dtype=np.int64),
            err_R0_R1_grid=err_R0_R1_grid,
            err_R0_R2_grid=err_R0_R2_grid,
            err_R1_R2_grid=err_R1_R2_grid,
            err_R0_R3_grid=err_R0_R3_grid,
            err_R2_R3_grid=err_R2_R3_grid,
        )

        # --- Acceptance gates ---
        worst_theorem1 = max(t["theorem1_err"] for t in trials)
        theorem1_pass = worst_theorem1 <= cfg.machine_eps_tol

        gd_trials = [t for t in trials if t["mask_kind"] == "gd_compatible"]
        if gd_trials:
            worst_R2_R3_gd = max(t["err_R2_R3"] for t in gd_trials)
            worst_R0_R3_gd = max(t["err_R0_R3"] for t in gd_trials)
            corollary1_pass = (
                worst_R2_R3_gd <= cfg.machine_eps_tol
                and worst_R0_R3_gd <= cfg.machine_eps_tol
            )
        else:
            worst_R2_R3_gd = float("nan")
            worst_R0_R3_gd = float("nan")
            corollary1_pass = False

        non_gd_trials = [t for t in trials if t["mask_kind"] != "gd_compatible"]
        if non_gd_trials:
            min_R2_R3_non_gd = min(t["err_R2_R3"] for t in non_gd_trials)
            worst_R2_R3_non_gd = max(t["err_R2_R3"] for t in non_gd_trials)
            # Necessity (contrapositive of Prop. 3): for EACH non-GD mask
            # kind, there must EXIST a cell where R2 and R3 disagree by at
            # least ``necessity_threshold``. Per-cell uniformity is not
            # required — at L=1 with S_QT = 1_K 1_P^⊤ any non-GD mask that
            # only perturbs S_TT gives err_R2_R3 ≡ 0 (since the L=1 readout
            # ``F = B_S · y`` is independent of S_TT).
            per_mask_max_r2r3: dict[tuple[str, float | None], float] = {}
            for (mk, eps) in cfg.mask_settings:
                if mk == "gd_compatible":
                    continue
                sub = [
                    t["err_R2_R3"] for t in trials
                    if _mask_key(t) == (mk, eps)
                ]
                if sub:
                    per_mask_max_r2r3[(mk, eps)] = max(sub)
            necessity_pass = (
                len(per_mask_max_r2r3) > 0
                and all(v >= cfg.necessity_threshold for v in per_mask_max_r2r3.values())
            )
        else:
            min_R2_R3_non_gd = float("nan")
            worst_R2_R3_non_gd = float("nan")
            necessity_pass = False

        # Near-GD monotonicity diagnostic.
        near_gd_monotone: dict[str, bool] = {}
        eps_vals = sorted({
            float(t["epsilon"]) for t in trials if t["mask_kind"] == "near_gd"
        })
        for gk in cfg.gamma_kinds:
            for sk in cfg.sigma_kinds:
                medians = []
                for e in eps_vals:
                    sub = [
                        t["err_R2_R3"] for t in trials
                        if t["mask_kind"] == "near_gd"
                        and float(t["epsilon"]) == e
                        and t["gamma_kind"] == gk
                        and t["sigma_kind"] == sk
                    ]
                    medians.append(float(np.median(sub)) if sub else float("nan"))
                # Monotone-nondecreasing in ε (within a small slack).
                ok = all(
                    medians[i + 1] >= medians[i] * 0.5
                    for i in range(len(medians) - 1)
                ) if len(medians) >= 2 else True
                near_gd_monotone[f"{gk}|{sk}"] = bool(ok)

        agg = _compute_aggregates(cfg, trials)

        ctx.record_compute_proxy(float(sweep_wall))
        ctx.record_extra("worst_theorem1_err", worst_theorem1)
        ctx.record_extra("worst_R2_R3_gd", worst_R2_R3_gd)
        ctx.record_extra("worst_R0_R3_gd", worst_R0_R3_gd)
        ctx.record_extra("min_R2_R3_non_gd", min_R2_R3_non_gd)
        ctx.record_extra("worst_R2_R3_non_gd", worst_R2_R3_non_gd)
        ctx.record_extra("theorem1_pass", bool(theorem1_pass))
        ctx.record_extra("corollary1_pass", bool(corollary1_pass))
        ctx.record_extra("necessity_pass", bool(necessity_pass))
        ctx.record_extra("near_gd_monotone", near_gd_monotone)
        ctx.record_extra("per_mask_aggregate", agg["per_mask"])
        ctx.record_extra("per_gamma_worst_theorem1", agg["per_gamma"])
        ctx.record_extra("per_sigma_worst_theorem1", agg["per_sigma"])

        status_parts = [
            f"theorem1={'ok' if theorem1_pass else f'FAIL({worst_theorem1:.2e})'}",
            f"corollary1={'ok' if corollary1_pass else 'FAIL'}",
            f"necessity={'ok' if necessity_pass else 'FAIL(per-mask max < floor)'}",
        ]
        status = " ".join(status_parts)

        ctx.write_summary(
            {
                "plan_reference": (
                    "EXPERIMENT_PLAN_FINAL.MD §8.1 "
                    "(A1 generalization to full train-supported theorem)"
                ),
                "framing": (
                    "A1 / A1b validate Theorem 1 and Corollary 1 in the "
                    "GD-compatible special case only. A1-general extends "
                    "Theorem-1 verification to ALL train-supported mixers "
                    "(4 mask kinds: gd_compatible, lower_triangular, "
                    "random_dense, near_gd at ε ∈ {0.01, 0.1, 0.5}), ALL "
                    "Γ kinds (identity, random_symmetric, "
                    "random_nonsymmetric — the latter is the load-bearing "
                    "W_q/W_k-ordering check), and TWO covariance settings "
                    "(isotropic, structured k^{-1}/k^{-0.5})."
                ),
                "category": (
                    "exact theorem-A operator-level forward-pass "
                    "equivalence test — Theorem 1 (full generality) and "
                    "Proposition 3 (necessity of GD-compatibility). "
                    "Deterministic, no training, no architecture search."
                ),
                "interpretation": (
                    "Theorem 1 holds for every train-supported S: R0 "
                    "(full hidden-state forward), R1 (iterative reduced "
                    "AB), and R2 (matrix-power reduced AB) must all agree "
                    "to float64 machine precision. Corollary 1 collapses "
                    "this to the reduced-Γ predictor R3 only under "
                    "GD-compatibility. Proposition 3's contrapositive — "
                    "R3 genuinely fails outside the GD-compatible class — "
                    "is validated by requiring, for each non-GD mask "
                    "kind, that max_cells err_R2_R3 ≥ 1e-3 (i.e. there "
                    "EXISTS at least one witness cell). Per-cell "
                    "uniformity is not required: Proposition 3's "
                    "hypothesis is a universal quantifier over (L, D, X, "
                    "X_⋆, Γ), so the contrapositive only needs one "
                    "counterexample. At L=1 any mask with S_QT = 1_K 1_P^⊤ "
                    "gives R2 ≡ R3 regardless of S_TT and is consistent "
                    "with the theorem. The near-GD interpolation at ε ∈ "
                    "{0.01, 0.1, 0.5} documents how the reduced-Γ collapse "
                    "degrades smoothly with mask deviation; this is "
                    "diagnostic only, not a theorem."
                ),
                "empirical_route": (
                    "R0 built from the full (P+K)×(P+K) bilinear score "
                    "x_μ^⊤ Γ x_ν with Γ applied on the right — never "
                    "symmetrized — and the general train-supported S "
                    "assembled as [[S_TT, 0],[S_QT, 0]]. R1 iterates the "
                    "r-recursion r^{ℓ+1} = (I + L⁻¹A_S) r^ℓ and "
                    "accumulates Σ T^ℓ y via vector updates. R2 "
                    "materializes T^ℓ as matrix powers and forms the "
                    "matrix geometric sum before applying (1/L) B_S. R3 "
                    "is the feature-space reduced-Γ formula — knows "
                    "nothing about S."
                ),
                "acceptance_framing": (
                    "theorem1_pass: max(err_R0_R1, err_R0_R2, err_R1_R2) "
                    "≤ 1e-10 in every cell. corollary1_pass: "
                    "max(err_R0_R3, err_R2_R3) ≤ 1e-10 in every "
                    "gd_compatible cell. necessity_pass: for each non-GD "
                    "mask kind, max_cells err_R2_R3 ≥ "
                    f"{cfg.necessity_threshold:.0e}. The max (not min) is "
                    "required because Proposition 3's contrapositive only "
                    "requires SOME (L, D, X, X_⋆, Γ) witness of R2 ≠ R3; "
                    "at L=1 any mask with S_QT = 1_K 1_P^⊤ gives R2 ≡ R3 "
                    "regardless of S_TT, and this is consistent with the "
                    "theorem."
                ),
                "device": str(device),
                "n_cells": len(trials),
                "mask_settings": [
                    {"kind": mk, "epsilon": eps}
                    for (mk, eps) in cfg.mask_settings
                ],
                "gamma_kinds": list(cfg.gamma_kinds),
                "sigma_kinds": list(cfg.sigma_kinds),
                "D_list": list(cfg.D_list),
                "P_list": list(cfg.P_list),
                "K_list": list(cfg.K_list),
                "L_list": list(cfg.L_list),
                "B": cfg.B,
                "label_norm": cfg.label_norm,
                "machine_eps_tol": cfg.machine_eps_tol,
                "necessity_threshold": cfg.necessity_threshold,
                "worst_theorem1_err": float(worst_theorem1),
                "worst_R2_R3_gd": float(worst_R2_R3_gd),
                "worst_R0_R3_gd": float(worst_R0_R3_gd),
                "min_R2_R3_non_gd": float(min_R2_R3_non_gd),
                "worst_R2_R3_non_gd": float(worst_R2_R3_non_gd),
                "theorem1_pass": bool(theorem1_pass),
                "corollary1_pass": bool(corollary1_pass),
                "necessity_pass": bool(necessity_pass),
                "near_gd_monotone": near_gd_monotone,
                "per_mask_aggregate": agg["per_mask"],
                "per_gamma_worst_theorem1": agg["per_gamma"],
                "per_sigma_worst_theorem1": agg["per_sigma"],
                "status": status,
                "sweep_wallclock_seconds": round(float(sweep_wall), 3),
            }
        )

        # --- Console summary ---
        print()
        print("=" * 78)
        print(f" A1-general theorem-A exactness over {len(trials)} cells on {device}")
        print(
            f"   Theorem 1  worst theorem1_err = {worst_theorem1:.3e}   "
            f"{'PASS' if theorem1_pass else 'FAIL'}   "
            f"(tol {cfg.machine_eps_tol:.0e})"
        )
        print(
            f"   Corollary 1 (GD only)  worst R2_R3 = {worst_R2_R3_gd:.3e}, "
            f"worst R0_R3 = {worst_R0_R3_gd:.3e}   "
            f"{'PASS' if corollary1_pass else 'FAIL'}"
        )
        print(
            f"   Proposition 3 necessity  worst R2_R3 (non-GD) = "
            f"{worst_R2_R3_non_gd:.3e}, min (diag.) = "
            f"{min_R2_R3_non_gd:.3e}   "
            f"{'PASS' if necessity_pass else 'FAIL'}   "
            f"(per-mask max must exceed {cfg.necessity_threshold:.0e})"
        )
        print("   Per-mask worst Theorem-1 error:")
        for key, stats in agg["per_mask"].items():
            print(
                f"     {key:<34s}  theorem1 = {stats['worst_theorem1_err']:.2e}  "
                f"R2_R3 median = {stats['median_R2_R3']:.2e}  "
                f"worst R2_R3 = {stats['worst_R2_R3']:.2e}"
            )
        print("   Per-γ worst Theorem-1 error:")
        for gk, v in agg["per_gamma"].items():
            print(f"     {gk:<24s}  {v:.2e}")
        print("   Per-σ worst Theorem-1 error:")
        for sk, v in agg["per_sigma"].items():
            print(f"     {sk:<24s}  {v:.2e}")
        print("=" * 78)

        all_ok = theorem1_pass and corollary1_pass and necessity_pass
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
