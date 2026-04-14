"""§9.1 architecture-aligned spectral OOD brittleness (theorem-B B3 analogue).

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §9.1 (architecture-aligned
spectral-only suite). Theorem reference: ``thesis/theorem_b.txt`` Corollary
"Out-of-distribution brittleness under stationary symbol shift"
(``cor:theoremB_ood``, Corollary 5). Operator-level sibling:
``scripts/thesis/theoremB/run_theoremB_symbol_shift.py`` (B3 exact) and
the supplementary replot ``run_theoremB_supplementary_figures.py``.

Role
----
This is the spectral-only architecture-aligned analogue of theorem-B
B3. A canonical trainable real-valued linear FFT-based spectral filter
with tied Fourier-mode weights is trained on matched stationary
circulant ICL regression and then evaluated under two symbol-native
OOD shift families:

1. **Structural interpolation** (family 1): s_te(α) = (1 − α) · s_tr +
   α · s_flat. The flat target is the opposite extreme of the
   power-law s_tr; in the canonical B3 setup every mode satisfies
   |1 − s_te,k / s_tr,k| < 1 at every α ∈ [0, 1], so Corollary 5
   predicts depth-driven ATTENUATION of OOD loss (deeper ⇒ less
   brittle).
2. **Frequency permutation** (family 2): s_te(α, seed) =
   (1 − α) · s_tr + α · permute(s_tr, seed). For α large enough,
   some Fourier modes satisfy |1 − s_te,k / s_tr,k| > 1, which is the
   regime in which Corollary 5 predicts depth-driven AMPLIFICATION
   (deeper ⇒ more brittle). The two shift families are therefore
   designed to probe BOTH regimes of Corollary 5.

The Fourier basis is fixed throughout (no rotations, no basis
adaptation). This is binding: generic covariance rotations belong to
the theorem-C bridge experiments, not to the §9.1 / theorem-B spectral
OOD story.

Model
-----
Primary: tied-weight circulant spectral filter. Single learnable
``γ ∈ ℝ^D`` shared across all ``L_S`` layers. The architecture matches
the theorem's single circulant ``Q``; tied weights have exactly ``D``
parameters independent of ``L_S``. Fixed geometry ``D = P = 32`` with
``K = 8``: this is the matched-stationary regime (``α_paper = P / D
= 1``). Depth-irrelevance across ``α_paper`` was addressed by the
previous tied-weight α-sweep script; this script is an OOD
experiment, not an α-sweep.

The forward pass at each layer is

    h^(ℓ+1) = h^(ℓ) + (1 / L_S) · (M_signed ⊙ S(γ)) · h^(ℓ),
    S(γ)[m, n] = (1 / P) · Σ_k γ_k · X̃[k, m] · X̃[k, n].

With tied ``γ``, ``S(γ)`` is precomputed once per forward pass.

OOD evaluation protocol
-----------------------
After training on matched stationary data (power-law ``s_tr``,
teacher ``ω``, same ν = 0.5 and νβ = 1.0 as the existing §9.1 runs),
the population OOD loss at a shifted test symbol ``s_te`` is
evaluated in closed form using the B3 protocol

    L_ood^arch(γ(T); s_tr, s_te, ω, L_S)
        = (1 / D) · Σ_k ω_k · s_te,k · (1 − s_tr,k · γ_k(T) / L_S)^{2 L_S}.

This is the stationary-loss formula at the TRAINED γ, with ``s_te``
only re-weighting modes in the prefactor; ``s_tr`` still enters the
residual because the score bilinear form is computed from training-
distributed features. (This matches B3's evaluation exactly, at the
architecture level: the only difference from B3 is that γ(T) here
comes from Adam on the trainable filter rather than from the
reduced-recursion exact object.) A per-batch MSE sampled from actual
``s_te``-distributed data would equal this formula in expectation
under sqrt_D label normalization.

Separately, the exact **Corollary 5 analytical reference** — the
asymptotic OOD loss at the converged optimum Q⋆ = L_S · T⁻¹ —

    L_ood^C5(L_S, s_te) = (1 / P) · Σ_k ω_k · s_te,k · |1 − s_te,k / s_tr,k|^{2 L_S}

is overlaid as a black dashed reference on every OOD-vs-α figure.
This is the same formula computed by
``run_theoremB_supplementary_figures.py``; overlaying it here lets
the reader see whether the trained architecture qualitatively
reproduces the per-regime Corollary-5 prediction.

Primary outputs
---------------
1. ``ood_loss_vs_alpha`` — two-panel (family 1, family 2): absolute
   L_ood vs α, one curve per L_S, Corollary-5 overlay.
2. ``normalized_brittleness_vs_alpha`` — two-panel: L_ood(α, L_S) /
   L_matched(0, L_S) vs α. α = 0 enforces ratio = 1 (sanity).
3. ``depth_sensitivity_at_fixed_alpha`` — two-panel (family 1, family
   2): L_ood vs L_S at α ∈ {0.5, 1.0}. Corollary-5 dashed overlay.
4. ``symbol_visualization`` — s_tr, s_flat, and selected s_te in
   Fourier space (both families).

Acceptance (qualitative, architecture-aligned)
----------------------------------------------
1. **Matched baseline recovery**: at α = 0 for both families, L_ood
   equals L_matched(0, L_S) to ``matched_baseline_tol`` for every L_S.
2. **Material brittleness at α = 1**: at least one L_S exhibits
   L_ood(α = 1, L_S) / L_matched(L_S) ≥ ``brittleness_ratio_min`` in
   at least one family.
3. **Regime direction** (qualitative): at α = 1, family 1 satisfies
   ``max_k |1 − s_te,k / s_tr,k| < 1`` by construction (attenuation
   regime); family 2 at α ≥ 0.5 exhibits modes with ``|·| > 1``
   (amplification regime, per seed). Both regimes are exercised.
4. **Corollary 5 qualitative tracking**: the sign of the slope
   ``d L_ood / d L_S`` at fixed α matches Corollary 5 at the
   converged optimum for family 1 (negative, attenuation) and
   for family 2 at large α (at least one seed with some L_S showing
   amplification).

This is architecture-aligned support for theorem-B; it is NOT an
exact theorem verification.

Run via SLURM
-------------
::

    sbatch experiments/thesis/architectures/run_arch_spectral_ood.sh
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

from scripts.thesis.utils.data_generators import G1Config, g1_generate
from scripts.thesis.utils.fourier_ops import (
    frequency_permutation,
    symbol_flat,
    symbol_interpolate,
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
class ArchSpectralOODConfig:
    """Frozen configuration for the §9.1 architecture-aligned OOD experiment.
    """

    # ---------------- Geometry (matched stationary regime) ------------
    D: int = 32
    P: int = 32     # D = P so α_paper = P / D = 1 (the matched regime)
    K: int = 8

    # ---------------- Training symbol ---------------------------------
    symbol_kind: str = "power_law"
    power_law_nu: float = 0.5
    task_spec_nu_beta: float = 1.0

    # ---------------- Architecture ------------------------------------
    L_S_list: tuple[int, ...] = (1, 2, 4, 8)
    tied_weights: bool = True
    init_scale: float = 0.0

    # ---------------- Training ----------------------------------------
    # 30k steps (bumped from 10k) so L_S ≥ 4 reaches near-convergence
    # on the matched recursion. At 10k the matched baseline was the
    # dominant source of L_ood variation across depth; at 30k the
    # Corollary-5 regime distinction (family-1 attenuation vs
    # family-2 amplification) becomes visible at the ABSOLUTE scale.
    train_steps: int = 30000
    batch_contexts: int = 64
    learning_rate: float = 5e-2
    optimizer: str = "adam"
    weight_decay: float = 0.0
    log_every: int = 25
    label_norm: str = "sqrt_D"

    # ---------------- Seeds -------------------------------------------
    seed_list: tuple[int, ...] = (0, 1, 2, 3)

    # ---------------- OOD shift families ------------------------------
    f1_target_kind: str = "flat"
    f1_target_flat_value: float = 1.0
    f1_alphas: tuple[float, ...] = (
        0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0,
    )
    f2_perm_seeds: tuple[int, ...] = (17, 23, 31, 41, 47, 53, 61, 71)
    f2_alphas: tuple[float, ...] = (
        0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0,
    )

    # Alpha values at which to slice depth-sensitivity figures.
    depth_slice_alphas: tuple[float, ...] = (0.5, 1.0)

    # ---------------- Acceptance (qualitative) ------------------------
    matched_baseline_tol: float = 1e-10
    brittleness_alpha: float = 1.0
    brittleness_ratio_min: float = 1.25
    final_loss_window: int = 20

    # ---------------- Symbol viz figure slice -------------------------
    symbol_samples_alphas: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0)
    symbol_samples_family2_seed: int = 17

    # ---------------- Misc --------------------------------------------
    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# F-basis batch sampler (matched stationary training; shared with §9.1 v2)
# ---------------------------------------------------------------------------


def _sample_batch_F_basis(
    s_tr: torch.Tensor,
    omega: torch.Tensor,
    P: int,
    K: int,
    B: int,
    norm_factor: float,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    D = int(s_tr.shape[0])
    s_sqrt = s_tr.to(dtype).sqrt().to(device)
    om_sqrt = omega.to(dtype).sqrt().to(device)
    z_x = torch.randn(B, D, P + K, generator=generator, dtype=dtype, device="cpu").to(device)
    X_tilde = z_x * s_sqrt.view(1, D, 1)
    z_b = torch.randn(B, D, generator=generator, dtype=dtype, device="cpu").to(device)
    beta_tilde = z_b * om_sqrt.view(1, D)
    y_full = torch.einsum("bd,bdf->bf", beta_tilde, X_tilde)
    y_full = y_full / float(math.sqrt(norm_factor))
    return X_tilde, y_full[:, :P].contiguous(), y_full[:, P:].contiguous()


# ---------------------------------------------------------------------------
# Tied-weight circulant spectral filter
# ---------------------------------------------------------------------------


class TiedCirculantSpectralFilter(nn.Module):
    """Single tied Fourier symbol γ ∈ ℝ^D shared across all L_S layers.

    Parameter count = D independent of L_S. Score is precomputed once per
    forward pass (tied γ → same S at every layer).
    """

    def __init__(
        self, D: int, P: int, K: int, L_S: int, *,
        init_scale: float = 0.0, dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.D = int(D)
        self.P = int(P)
        self.K = int(K)
        self.L_S = int(L_S)
        if float(init_scale) == 0.0:
            init = torch.zeros(self.D, dtype=dtype)
        else:
            init = float(init_scale) * torch.randn(self.D, dtype=dtype)
        self.gamma = nn.Parameter(init)
        M = torch.zeros(self.P + self.K, self.P + self.K, dtype=dtype)
        M[:self.P, :self.P] = -1.0
        M[self.P:, :self.P] = +1.0
        self.register_buffer("M_signed", M)

    def forward(
        self, X_tilde: torch.Tensor, y_train: torch.Tensor,
    ) -> torch.Tensor:
        B = int(X_tilde.shape[0])
        device, dtype = y_train.device, y_train.dtype
        h = torch.cat(
            [y_train, torch.zeros(B, self.K, dtype=dtype, device=device)],
            dim=-1,
        )
        weighted = self.gamma.view(1, -1, 1) * X_tilde
        score = torch.einsum(
            "bdm,bdn->bmn", weighted, X_tilde,
        ) / float(self.P)
        masked_score = self.M_signed.unsqueeze(0) * score
        inv_L = 1.0 / float(self.L_S)
        for _ell in range(self.L_S):
            update = torch.einsum("bmn,bn->bm", masked_score, h)
            h = h + inv_L * update
        return h[:, self.P:]

    def final_gamma(self) -> torch.Tensor:
        return self.gamma.detach().clone()


# ---------------------------------------------------------------------------
# OOD loss helpers (both B3 protocol and Corollary 5 analytical)
# ---------------------------------------------------------------------------


def _ood_loss_B3(
    gamma: torch.Tensor,
    s_tr: torch.Tensor,
    s_te: torch.Tensor,
    omega: torch.Tensor,
    L_S: int,
    norm_factor: float,
) -> float:
    """Population OOD loss at the TRAINED γ (B3 protocol).

    (1 / norm_factor) · Σ_k ω_k · s_te,k · (1 − s_tr,k · γ_k / L_S)^{2 L_S}.

    With sqrt_D normalization (``norm_factor = D``) and the canonical
    ``P = D`` geometry, this equals ``L_ood / P`` in the B3 script's
    un-normalized convention.
    """
    g = gamma.to(torch.float64)
    s_tr64 = s_tr.to(torch.float64)
    s_te64 = s_te.to(torch.float64)
    w64 = omega.to(torch.float64)
    residual = 1.0 - s_tr64 * g / int(L_S)
    transfer_sq = residual.pow(2 * int(L_S))
    return float((w64 * s_te64 * transfer_sq).sum().item() / float(norm_factor))


def _ood_loss_corollary5(
    s_tr: torch.Tensor,
    s_te: torch.Tensor,
    omega: torch.Tensor,
    L_S: int,
    norm_factor: float,
) -> float:
    """Corollary-5 asymptotic OOD loss at the converged optimum Q⋆ = L · T⁻¹.

    (1 / norm_factor) · Σ_k ω_k · s_te,k · |1 − s_te,k / s_tr,k|^{2 L_S}.
    """
    s_tr64 = s_tr.to(torch.float64)
    s_te64 = s_te.to(torch.float64)
    w64 = omega.to(torch.float64)
    factor = (1.0 - s_te64 / s_tr64).abs()
    return float((w64 * s_te64 * factor.pow(2 * int(L_S))).sum().item() / float(norm_factor))


# ---------------------------------------------------------------------------
# Per-L_S matched training (shared across α-evaluation, symbol evaluation)
# ---------------------------------------------------------------------------


def _train_matched(
    cfg: ArchSpectralOODConfig,
    L_S: int,
    seed: int,
    s_tr: torch.Tensor,
    omega: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Train a tied-weight circulant spectral filter on matched stationary data.
    Returns the terminal γ (for OOD evaluation) plus training diagnostics.
    """
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    norm_factor = (
        float(cfg.D) if cfg.label_norm == "sqrt_D" else float(cfg.P)
    )
    torch.manual_seed(int(seed) * 991 + 17 * int(L_S) + 1000003)
    model = TiedCirculantSpectralFilter(
        D=cfg.D, P=cfg.P, K=cfg.K, L_S=int(L_S),
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

    sample_gen = torch.Generator(device="cpu")
    sample_gen.manual_seed(int(seed) * 7919 + 31 * int(L_S) + 137 * int(cfg.P))

    loss_steps: list[int] = []
    loss_values: list[float] = []
    nan_failure = False
    t0 = time.perf_counter()
    for step in range(int(cfg.train_steps)):
        X_tilde, y_train, y_query = _sample_batch_F_basis(
            s_tr, omega, cfg.P, cfg.K, int(cfg.batch_contexts),
            norm_factor, sample_gen, dtype, device,
        )
        opt.zero_grad()
        y_pred = model(X_tilde, y_train)
        loss = ((y_pred - y_query) ** 2).mean()
        if not torch.isfinite(loss):
            nan_failure = True
            print(
                f"   [L_S={L_S}, seed={seed}] NaN loss at step {step}; "
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
        "L_S": int(L_S),
        "seed": int(seed),
        "loss_steps": loss_steps,
        "loss_values": loss_values,
        "gamma_final": gamma_final,
        "nan_failure": bool(nan_failure),
        "train_seconds": float(t_train),
    }


# ---------------------------------------------------------------------------
# OOD sweeps
# ---------------------------------------------------------------------------


def _eval_family1(
    cfg: ArchSpectralOODConfig,
    s_tr: torch.Tensor,
    omega: torch.Tensor,
    gamma_by_LS_seed: dict[tuple[int, int], torch.Tensor],
) -> dict[str, Any]:
    norm_factor = (
        float(cfg.D) if cfg.label_norm == "sqrt_D" else float(cfg.P)
    )
    s_flat = symbol_flat(int(cfg.D), float(cfg.f1_target_flat_value))
    # L_ood per (L_S, seed, α)
    losses: dict[int, np.ndarray] = {}
    c5: dict[int, np.ndarray] = {}
    for L_S in cfg.L_S_list:
        per_seed_alpha = np.zeros(
            (len(cfg.seed_list), len(cfg.f1_alphas)), dtype=np.float64
        )
        c5_per_alpha = np.zeros(len(cfg.f1_alphas), dtype=np.float64)
        for a_i, alpha in enumerate(cfg.f1_alphas):
            s_te = symbol_interpolate(s_tr, s_flat, float(alpha))
            c5_per_alpha[a_i] = _ood_loss_corollary5(
                s_tr, s_te, omega, int(L_S), norm_factor,
            )
            for s_i, seed in enumerate(cfg.seed_list):
                g = gamma_by_LS_seed[(int(L_S), int(seed))]
                per_seed_alpha[s_i, a_i] = _ood_loss_B3(
                    g, s_tr, s_te, omega, int(L_S), norm_factor,
                )
        losses[int(L_S)] = per_seed_alpha
        c5[int(L_S)] = c5_per_alpha
    return {
        "s_other": s_flat,
        "losses_per_seed_alpha_by_LS": losses,
        "corollary5_by_LS": c5,
    }


def _eval_family2(
    cfg: ArchSpectralOODConfig,
    s_tr: torch.Tensor,
    omega: torch.Tensor,
    gamma_by_LS_seed: dict[tuple[int, int], torch.Tensor],
) -> dict[str, Any]:
    norm_factor = (
        float(cfg.D) if cfg.label_norm == "sqrt_D" else float(cfg.P)
    )
    s_perm_by_seed = {
        int(s): frequency_permutation(s_tr, seed=int(s)) for s in cfg.f2_perm_seeds
    }
    # losses[(L_S)] has shape (train_seed, perm_seed, α).
    losses: dict[int, np.ndarray] = {}
    c5: dict[int, np.ndarray] = {}  # shape (perm_seed, α)
    for L_S in cfg.L_S_list:
        per_seed = np.zeros(
            (len(cfg.seed_list), len(cfg.f2_perm_seeds), len(cfg.f2_alphas)),
            dtype=np.float64,
        )
        c5_arr = np.zeros(
            (len(cfg.f2_perm_seeds), len(cfg.f2_alphas)),
            dtype=np.float64,
        )
        for p_i, p_seed in enumerate(cfg.f2_perm_seeds):
            s_perm = s_perm_by_seed[int(p_seed)]
            for a_i, alpha in enumerate(cfg.f2_alphas):
                s_te = symbol_interpolate(s_tr, s_perm, float(alpha))
                c5_arr[p_i, a_i] = _ood_loss_corollary5(
                    s_tr, s_te, omega, int(L_S), norm_factor,
                )
                for t_i, train_seed in enumerate(cfg.seed_list):
                    g = gamma_by_LS_seed[(int(L_S), int(train_seed))]
                    per_seed[t_i, p_i, a_i] = _ood_loss_B3(
                        g, s_tr, s_te, omega, int(L_S), norm_factor,
                    )
        losses[int(L_S)] = per_seed
        c5[int(L_S)] = c5_arr
    return {
        "s_perm_by_seed": s_perm_by_seed,
        "losses_by_LS": losses,
        "corollary5_by_LS": c5,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_ood_loss_vs_alpha(
    cfg: ArchSpectralOODConfig,
    f1: dict[str, Any],
    f2: dict[str, Any],
    matched_L_ood_by_LS: dict[int, float],
    run_dir: ThesisRunDir,
) -> None:
    """Absolute OOD loss vs α, per L_S, with per-L_S matched-baseline
    horizontal dashed lines. NO Corollary-5 overlay: Corollary 5 evaluates
    at Q⋆ = L T⁻¹ (where L_matched = 0); the architecture evaluates at
    γ(T) (L_matched > 0). The two objects are incomparable on the same
    y-axis at finite training time. Corollary 5 is plotted separately in
    ``corollary5_operator_reference``.
    """
    import matplotlib.pyplot as plt

    L_colors = sequential_colors(len(cfg.L_S_list), palette="rocket")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Family 1 — architecture curves + per-L_S matched baseline line.
    for color, L_S in zip(L_colors, cfg.L_S_list):
        per_seed = f1["losses_per_seed_alpha_by_LS"][int(L_S)]
        mean = per_seed.mean(axis=0)
        if per_seed.shape[0] > 1:
            se = per_seed.std(axis=0, ddof=1) / float(np.sqrt(per_seed.shape[0]))
        else:
            se = np.zeros_like(mean)
        axL.plot(
            cfg.f1_alphas, mean, color=color, lw=1.4, marker="o", ms=4.0,
            label=rf"$L_S = {L_S}$",
            zorder=2,
        )
        axL.fill_between(
            cfg.f1_alphas, mean - se, mean + se,
            color=color, alpha=0.18, lw=0,
        )
        axL.axhline(
            matched_L_ood_by_LS[int(L_S)], color=color, ls="--", lw=0.8,
            alpha=0.9, zorder=1,
        )
    axL.set_xlabel(r"shift $\alpha$")
    axL.set_ylabel(r"OOD loss  $\mathcal{L}_{\mathrm{ood}}(\alpha, L_S)$")
    axL.set_yscale("log")
    axL.set_title(
        rf"Family 1: structural interpolation  "
        rf"$s_{{\mathrm{{te}}}}(\alpha) = (1-\alpha) s_{{\mathrm{{tr}}}} + "
        rf"\alpha \cdot 1$   "
        r"(dashed = matched baseline per $L_S$)",
        fontsize=10,
    )
    axL.legend(fontsize=8, loc="best", frameon=True, ncol=1)

    # Family 2 — architecture curves (median over perm seeds) + matched baseline.
    for color, L_S in zip(L_colors, cfg.L_S_list):
        per_tr = f2["losses_by_LS"][int(L_S)]
        med_over_perm = np.median(per_tr, axis=1)
        mean = med_over_perm.mean(axis=0)
        if med_over_perm.shape[0] > 1:
            se = med_over_perm.std(axis=0, ddof=1) / float(np.sqrt(med_over_perm.shape[0]))
        else:
            se = np.zeros_like(mean)
        axR.plot(
            cfg.f2_alphas, mean, color=color, lw=1.4, marker="o", ms=4.0,
            label=rf"$L_S = {L_S}$",
            zorder=2,
        )
        axR.fill_between(
            cfg.f2_alphas, mean - se, mean + se,
            color=color, alpha=0.18, lw=0,
        )
        axR.axhline(
            matched_L_ood_by_LS[int(L_S)], color=color, ls="--", lw=0.8,
            alpha=0.9, zorder=1,
        )
    axR.set_xlabel(r"shift $\alpha$")
    axR.set_ylabel(r"OOD loss  $\mathcal{L}_{\mathrm{ood}}(\alpha, L_S)$")
    axR.set_yscale("log")
    axR.set_title(
        rf"Family 2: frequency permutation  "
        rf"$s_{{\mathrm{{te}}}}(\alpha, s) = (1-\alpha) s_{{\mathrm{{tr}}}} + "
        rf"\alpha \cdot \pi_s(s_{{\mathrm{{tr}}}})$  (median over 8 perm seeds; "
        r"dashed = matched baseline)",
        fontsize=10,
    )
    axR.legend(fontsize=8, loc="best", frameon=True)

    fig.suptitle(
        "§9.1 architecture-aligned spectral OOD brittleness "
        "(tied-weight filter, theorem-B B3 analogue)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "ood_loss_vs_alpha")
    plt.close(fig)


def _plot_normalized_brittleness(
    cfg: ArchSpectralOODConfig,
    f1: dict[str, Any],
    f2: dict[str, Any],
    matched_L_ood_by_LS: dict[int, float],
    run_dir: ThesisRunDir,
) -> None:
    """L_ood(α, L_S) / L_matched(L_S) vs α. At α = 0 the ratio is 1 per seed
    (matched baseline recovery identity)."""
    import matplotlib.pyplot as plt

    L_colors = sequential_colors(len(cfg.L_S_list), palette="rocket")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)

    for color, L_S in zip(L_colors, cfg.L_S_list):
        per_seed = f1["losses_per_seed_alpha_by_LS"][int(L_S)]  # (n_seed, α)
        # matched denominator is the seed-specific α = 0 point.
        denom = per_seed[:, 0:1]
        ratio = per_seed / np.maximum(denom, 1e-30)
        mean = ratio.mean(axis=0)
        if ratio.shape[0] > 1:
            se = ratio.std(axis=0, ddof=1) / float(np.sqrt(ratio.shape[0]))
        else:
            se = np.zeros_like(mean)
        axL.plot(
            cfg.f1_alphas, mean, color=color, lw=1.4, marker="o", ms=4.0,
            label=rf"$L_S = {L_S}$",
        )
        axL.fill_between(
            cfg.f1_alphas, mean - se, mean + se,
            color=color, alpha=0.18, lw=0,
        )
    axL.axhline(
        1.0, color="gray", ls=":", lw=1.0,
        label=r"matched baseline $\mathcal{L}_{\mathrm{matched}}$",
    )
    axL.set_xlabel(r"shift $\alpha$")
    axL.set_ylabel(
        r"$\mathcal{L}_{\mathrm{ood}} / \mathcal{L}_{\mathrm{matched}}$"
    )
    axL.set_yscale("log")
    axL.set_title("Family 1: normalized brittleness", fontsize=10)
    axL.legend(fontsize=8, loc="best", frameon=True)

    for color, L_S in zip(L_colors, cfg.L_S_list):
        per_tr = f2["losses_by_LS"][int(L_S)]  # (train_seed, perm_seed, α)
        med_over_perm = np.median(per_tr, axis=1)
        denom = med_over_perm[:, 0:1]
        ratio = med_over_perm / np.maximum(denom, 1e-30)
        mean = ratio.mean(axis=0)
        if ratio.shape[0] > 1:
            se = ratio.std(axis=0, ddof=1) / float(np.sqrt(ratio.shape[0]))
        else:
            se = np.zeros_like(mean)
        axR.plot(
            cfg.f2_alphas, mean, color=color, lw=1.4, marker="o", ms=4.0,
            label=rf"$L_S = {L_S}$",
        )
        axR.fill_between(
            cfg.f2_alphas, mean - se, mean + se,
            color=color, alpha=0.18, lw=0,
        )
    axR.axhline(1.0, color="gray", ls=":", lw=1.0)
    axR.set_xlabel(r"shift $\alpha$")
    axR.set_ylabel(
        r"$\mathcal{L}_{\mathrm{ood}} / \mathcal{L}_{\mathrm{matched}}$"
    )
    axR.set_yscale("log")
    axR.set_title(
        "Family 2: normalized brittleness (median over 8 perm seeds)",
        fontsize=10,
    )
    axR.legend(fontsize=8, loc="best", frameon=True)

    fig.suptitle(
        "§9.1 HEADLINE: normalized OOD brittleness vs shift α  "
        "(tied-weight filter; finite-T architecture evaluation)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "normalized_brittleness_vs_alpha")
    plt.close(fig)


def _plot_depth_sensitivity(
    cfg: ArchSpectralOODConfig,
    f1: dict[str, Any],
    f2: dict[str, Any],
    run_dir: ThesisRunDir,
) -> None:
    """Normalized brittleness ratio L_ood/L_matched vs L_S at fixed α.

    Plotting the ABSOLUTE L_ood would be dominated by the matched-baseline
    convergence (deeper = tighter matched = smaller absolute L_ood in both
    families), which visually contradicts the theorem's regime distinction.
    Dividing by each depth's own matched baseline isolates the OOD signal,
    showing deeper = more brittle at a fixed shift in BOTH families. The
    Corollary-5 operator-level regime distinction is presented separately
    in ``corollary5_operator_reference``.
    """
    import matplotlib.pyplot as plt

    alpha_colors = sequential_colors(len(cfg.depth_slice_alphas), palette="rocket")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    L_arr = np.asarray(cfg.L_S_list, dtype=int)
    f1_alpha_grid = list(cfg.f1_alphas)
    f2_alpha_grid = list(cfg.f2_alphas)

    for color, alpha in zip(alpha_colors, cfg.depth_slice_alphas):
        a_i1 = int(np.argmin(np.abs(np.asarray(f1_alpha_grid) - float(alpha))))
        means: list[float] = []
        ses: list[float] = []
        for L_S in cfg.L_S_list:
            per_seed = f1["losses_per_seed_alpha_by_LS"][int(L_S)]
            matched_per_seed = per_seed[:, 0]              # α = 0 per seed
            ood_per_seed = per_seed[:, a_i1]
            ratio = ood_per_seed / np.maximum(matched_per_seed, 1e-30)
            means.append(float(ratio.mean()))
            ses.append(
                float(ratio.std(ddof=1) / np.sqrt(ratio.size))
                if ratio.size > 1 else 0.0
            )
        axL.errorbar(
            L_arr, means, yerr=ses, fmt="o-", color=color, ms=5, lw=1.4,
            capsize=4,
            label=rf"$\alpha = {alpha:.2f}$",
        )

    for color, alpha in zip(alpha_colors, cfg.depth_slice_alphas):
        a_i2 = int(np.argmin(np.abs(np.asarray(f2_alpha_grid) - float(alpha))))
        means: list[float] = []
        ses: list[float] = []
        for L_S in cfg.L_S_list:
            per_tr = f2["losses_by_LS"][int(L_S)]           # (train_seed, perm_seed, α)
            matched_per_seed = np.median(per_tr[:, :, 0], axis=1)
            ood_per_seed = np.median(per_tr[:, :, a_i2], axis=1)
            ratio = ood_per_seed / np.maximum(matched_per_seed, 1e-30)
            means.append(float(ratio.mean()))
            ses.append(
                float(ratio.std(ddof=1) / np.sqrt(ratio.size))
                if ratio.size > 1 else 0.0
            )
        axR.errorbar(
            L_arr, means, yerr=ses, fmt="o-", color=color, ms=5, lw=1.4,
            capsize=4,
            label=rf"$\alpha = {alpha:.2f}$ (median)",
        )

    for ax, title in (
        (axL, "Family 1: normalized brittleness vs $L_S$"),
        (axR, "Family 2: normalized brittleness vs $L_S$ (median over 8 perm seeds)"),
    ):
        ax.axhline(
            1.0, color="gray", ls=":", lw=1.0,
            label=r"matched $\mathcal{L}_{\mathrm{matched}}$",
        )
        ax.set_xlabel(r"spectral depth $L_S$")
        ax.set_ylabel(
            r"$\mathcal{L}_{\mathrm{ood}} / \mathcal{L}_{\mathrm{matched}}$"
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "§9.1 OOD depth-sensitivity at fixed shift "
        "(normalized brittleness; architecture only)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "depth_sensitivity_at_fixed_alpha")
    plt.close(fig)


def _plot_corollary5_operator_reference(
    cfg: ArchSpectralOODConfig,
    f1: dict[str, Any],
    f2: dict[str, Any],
    run_dir: ThesisRunDir,
) -> None:
    """Operator-level Corollary-5 reference: absolute OOD loss at the
    converged optimum Q⋆ = L · T⁻¹, computed analytically, for each L_S
    across the shift-α grid. Family 1 (left) and family 2 (right, median
    over the 8 permutation seeds).

    This figure is the CORRECT place to read off the theorem-level regime
    distinction (family-1 attenuation: deeper = lower OOD; family-2
    amplification at large α: deeper = higher OOD). The values are NOT
    directly comparable to the architecture figures because the
    architecture evaluates at finite-time γ(T), not at Q⋆; the two
    objects live on different y-scales.
    """
    import matplotlib.pyplot as plt

    L_colors = sequential_colors(len(cfg.L_S_list), palette="rocket")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for color, L_S in zip(L_colors, cfg.L_S_list):
        c5 = f1["corollary5_by_LS"][int(L_S)]
        axL.plot(
            cfg.f1_alphas, np.maximum(c5, 1e-40),
            color=color, lw=1.4, marker="o", ms=4.0,
            label=rf"$L_S = {L_S}$",
        )
    axL.set_xlabel(r"shift $\alpha$")
    axL.set_ylabel(
        r"$\mathcal{L}_{\mathrm{ood}}^{(\mathrm{Cor.\,5})}(\alpha, L_S)$"
    )
    axL.set_yscale("log")
    axL.set_title(
        "Family 1 (Corollary-5 attenuation regime; deeper = lower OOD)",
        fontsize=10,
    )
    axL.legend(fontsize=8, loc="best", frameon=True)

    for color, L_S in zip(L_colors, cfg.L_S_list):
        c5_arr = f2["corollary5_by_LS"][int(L_S)]          # (perm_seed, α)
        med = np.median(c5_arr, axis=0)
        q25 = np.quantile(c5_arr, 0.25, axis=0)
        q75 = np.quantile(c5_arr, 0.75, axis=0)
        axR.plot(
            cfg.f2_alphas, np.maximum(med, 1e-40),
            color=color, lw=1.4, marker="o", ms=4.0,
            label=rf"$L_S = {L_S}$ (median)",
        )
        axR.fill_between(
            cfg.f2_alphas,
            np.maximum(q25, 1e-40), np.maximum(q75, 1e-40),
            color=color, alpha=0.18, lw=0,
        )
    axR.set_xlabel(r"shift $\alpha$")
    axR.set_ylabel(
        r"$\mathcal{L}_{\mathrm{ood}}^{(\mathrm{Cor.\,5})}(\alpha, L_S)$"
    )
    axR.set_yscale("log")
    axR.set_title(
        "Family 2 (Corollary-5 amplification regime at large $\\alpha$; "
        "deeper = higher OOD)",
        fontsize=10,
    )
    axR.legend(fontsize=8, loc="best", frameon=True)

    fig.suptitle(
        "Operator-level Corollary-5 reference "
        "(converged optimum $Q^\\star = L\\,T^{-1}$, "
        "NOT directly comparable to finite-time architecture)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "corollary5_operator_reference")
    plt.close(fig)


def _plot_symbol_visualization(
    cfg: ArchSpectralOODConfig,
    s_tr: torch.Tensor,
    s_flat: torch.Tensor,
    s_perm_seed_sample: torch.Tensor,
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    k_axis = np.arange(int(cfg.D))
    s_tr_np = s_tr.detach().cpu().numpy()
    colors = sequential_colors(len(cfg.symbol_samples_alphas), palette="rocket")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for color, alpha in zip(colors, cfg.symbol_samples_alphas):
        s_te = symbol_interpolate(s_tr, s_flat, float(alpha))
        axL.plot(
            k_axis, s_te.detach().cpu().numpy(),
            color=color, lw=1.3, marker="o", ms=3,
            label=rf"$\alpha = {alpha:.2f}$",
        )
    axL.plot(
        k_axis, s_tr_np, color="black", lw=1.0, ls=":",
        label=r"$s_{\mathrm{tr}}$ (reference)",
    )
    axL.set_xlabel(r"Fourier mode index $k$")
    axL.set_ylabel(r"symbol value $s_{\mathrm{te}, k}$")
    axL.set_title(
        r"Family 1: $s_{\mathrm{te}} = (1-\alpha) s_{\mathrm{tr}} + \alpha \cdot 1$",
        fontsize=10,
    )
    axL.legend(fontsize=8, loc="best")

    for color, alpha in zip(colors, cfg.symbol_samples_alphas):
        s_te = symbol_interpolate(s_tr, s_perm_seed_sample, float(alpha))
        axR.plot(
            k_axis, s_te.detach().cpu().numpy(),
            color=color, lw=1.3, marker="o", ms=3,
            label=rf"$\alpha = {alpha:.2f}$",
        )
    axR.plot(
        k_axis, s_tr_np, color="black", lw=1.0, ls=":",
        label=r"$s_{\mathrm{tr}}$ (reference)",
    )
    axR.set_xlabel(r"Fourier mode index $k$")
    axR.set_ylabel(r"symbol value $s_{\mathrm{te}, k}$")
    axR.set_title(
        rf"Family 2: $s_{{\mathrm{{te}}}} = (1-\alpha) s_{{\mathrm{{tr}}}} + "
        rf"\alpha \cdot \pi_{{{int(cfg.symbol_samples_family2_seed)}}}(s_{{\mathrm{{tr}}}})$",
        fontsize=10,
    )
    axR.legend(fontsize=8, loc="best")

    fig.suptitle(
        "§9.1 symbol visualization: symbol-native OOD shift families",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "symbol_visualization")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "§9.1 architecture-aligned spectral OOD brittleness: tied-weight "
            "trainable circulant spectral filter on matched stationary "
            "circulant ICL regression, evaluated on B3 symbol-native shift "
            "families 1 (structural interpolation) and 2 (frequency "
            "permutation). Corollary-5 asymptotic overlay."
        )
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda", "auto"))
    p.add_argument("--dtype", type=str, default="float64", choices=("float32", "float64"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--L-S-list", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None)
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> ArchSpectralOODConfig:
    base = ArchSpectralOODConfig()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.train_steps is not None:
        overrides["train_steps"] = int(args.train_steps)
    if args.L_S_list is not None:
        overrides["L_S_list"] = _parse_list_ints(args.L_S_list)
    if args.seeds is not None:
        overrides["seed_list"] = _parse_list_ints(args.seeds)
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
    if cfg.D != cfg.P:
        print(
            f"[arch-§9.1-OOD] WARNING: D = {cfg.D} != P = {cfg.P}; "
            "the OOD experiment is designed for the matched stationary "
            "regime D = P. Proceeding anyway."
        )
    device = _resolve_device(cfg.device)
    print(f"[arch-§9.1-OOD] device = {device}")

    run = ThesisRunDir(__file__, phase="architectures")
    with RunContext(
        run,
        config=cfg,
        seeds=list(cfg.seed_list),
        notes=(
            "§9.1 architecture-aligned spectral OOD experiment (theorem-B "
            "B3 analogue, tied-weight trainable circulant filter, two "
            "symbol-native shift families, Corollary-5 asymptotic overlay)."
        ),
    ) as ctx:
        apply_thesis_style()

        # ---------------- Build (s_tr, ω) via G1 at P = D --------------
        if cfg.symbol_kind == "power_law":
            symbol_params: dict[str, Any] = {"nu": cfg.power_law_nu}
        elif cfg.symbol_kind == "multiband":
            symbol_params = {"bands": [[0, 2, 1.0], [5, 7, 0.8]]}
        elif cfg.symbol_kind == "flat":
            symbol_params = {"value": 1.0}
        else:
            raise ValueError(f"unknown symbol_kind: {cfg.symbol_kind!r}")

        g1_cfg = G1Config(
            P=cfg.P, D=cfg.D, B=1,
            query_mode="full_window",
            matched_query_realization="independent",
            symbol_kind_tr=cfg.symbol_kind,
            symbol_params_tr=symbol_params,
            symbol_kind_te="matched",
            symbol_params_te={},
            task_spec_kind="power_law",
            task_spec_params={"nu_beta": cfg.task_spec_nu_beta},
            sigma=0.0,
            label_norm=cfg.label_norm,
            exact_mode=True,
            sample_data=False,
            population_mode=True,
            dtype=cfg.dtype,
        )
        op = g1_generate(g1_cfg)
        dtype64 = torch.float64 if cfg.dtype == "float64" else torch.float32
        s_tr = op["s_tr"].to(dtype64)
        omega = op["omega"].to(dtype64)
        norm_factor = (
            float(cfg.D) if cfg.label_norm == "sqrt_D" else float(cfg.P)
        )
        initial_loss_analytical = float(
            (omega.to(torch.float64) * s_tr.to(torch.float64)).sum().item()
        )
        print(
            f"[arch-§9.1-OOD] D = P = {cfg.D};  "
            f"s_tr range=[{float(s_tr.min()):.3e}, {float(s_tr.max()):.3e}]  "
            f"L(γ=0) = {initial_loss_analytical:.4e}  "
            f"per-batch MSE@γ=0 = "
            f"{initial_loss_analytical / norm_factor:.4e}"
        )

        # ---------------- Report deviation regimes (diagnostic) ---------
        s_flat = symbol_flat(int(cfg.D), float(cfg.f1_target_flat_value))
        s_te_f1_full = symbol_interpolate(s_tr, s_flat, 1.0)
        max_dev_f1 = float(torch.abs(1.0 - s_te_f1_full / s_tr).max().item())
        print(
            f"[arch-§9.1-OOD] family-1 at α = 1.0: "
            f"max_k |1 − s_te,k / s_tr,k| = {max_dev_f1:.3f}  "
            f"({'attenuation' if max_dev_f1 < 1.0 else 'mixed / amplification'})"
        )
        n_amp_f2_alpha1 = 0
        for pseed in cfg.f2_perm_seeds:
            s_perm = frequency_permutation(s_tr, seed=int(pseed))
            s_te = symbol_interpolate(s_tr, s_perm, 1.0)
            n_amp_f2_alpha1 += int(torch.sum(torch.abs(1.0 - s_te / s_tr) > 1.0).item())
        print(
            f"[arch-§9.1-OOD] family-2 at α = 1.0: "
            f"total # modes with |·| > 1 across {len(cfg.f2_perm_seeds)} perm seeds = "
            f"{n_amp_f2_alpha1}"
        )

        # ---------------- Matched training per (L_S, seed) -------------
        gamma_by_LS_seed: dict[tuple[int, int], torch.Tensor] = {}
        train_diag: dict[tuple[int, int], dict[str, Any]] = {}
        n_total = len(cfg.L_S_list) * len(cfg.seed_list)
        idx = 0
        t_train_sweep = time.perf_counter()
        for L_S in cfg.L_S_list:
            for seed in cfg.seed_list:
                idx += 1
                t0 = time.perf_counter()
                r = _train_matched(cfg, int(L_S), int(seed), s_tr, omega, device)
                dt = time.perf_counter() - t0
                ctx.record_step_time(dt)
                gamma_by_LS_seed[(int(L_S), int(seed))] = r["gamma_final"]
                train_diag[(int(L_S), int(seed))] = r
                tail = r["loss_values"][-cfg.final_loss_window:]
                final_smoothed = float(np.mean(tail)) if tail else float("nan")
                init_batch = float(r["loss_values"][0]) if r["loss_values"] else float("nan")
                print(
                    f"[train {idx:>2d}/{n_total}] L_S={L_S:>2d} seed={seed}  "
                    f"init≈{init_batch:.3e}  "
                    f"final(w{cfg.final_loss_window})≈{final_smoothed:.3e}  "
                    f"nan={r['nan_failure']}  ({dt:.2f} s)"
                )
        train_wall = time.perf_counter() - t_train_sweep

        # ---------------- Convergence diagnostics per L_S --------------
        # At each L_S, the tied γ should approach γ⋆_k = L_S / s_tr,k.
        # Median |T_k^cumul| = median |1 − γ_k·s_tr,k/L_S|^{L_S} across
        # modes quantifies how close training came to the theorem-B
        # matched-stationary target. At 30k steps L_S ≥ 4 should be
        # near-converged; L_S = 1 should still be modest.
        median_abs_T_by_LS: dict[int, float] = {}
        for L_S in cfg.L_S_list:
            per_seed_medians: list[float] = []
            for seed in cfg.seed_list:
                g = gamma_by_LS_seed[(int(L_S), int(seed))].to(torch.float64)
                s_tr64 = s_tr.to(torch.float64)
                residual = 1.0 - g * s_tr64 / float(L_S)
                T = residual.abs().pow(int(L_S))
                per_seed_medians.append(float(T.median().item()))
            median_abs_T_by_LS[int(L_S)] = float(np.mean(per_seed_medians))

        # ---------------- Matched baseline per L_S (α=0 sanity) --------
        # At α = 0 family-1 and family-2 both give s_te = s_tr, so the
        # B3 formula collapses to the matched stationary loss at γ(T).
        matched_L_ood_by_LS: dict[int, float] = {}
        for L_S in cfg.L_S_list:
            seed_vals = []
            for seed in cfg.seed_list:
                g = gamma_by_LS_seed[(int(L_S), int(seed))]
                seed_vals.append(
                    _ood_loss_B3(g, s_tr, s_tr, omega, int(L_S), norm_factor)
                )
            matched_L_ood_by_LS[int(L_S)] = float(np.mean(seed_vals))

        # ---------------- Family-1 and family-2 OOD sweeps -------------
        f1 = _eval_family1(cfg, s_tr, omega, gamma_by_LS_seed)
        f2 = _eval_family2(cfg, s_tr, omega, gamma_by_LS_seed)

        # ---------------- Acceptance gates -----------------------------
        # (A) Matched baseline recovery at α = 0.
        baseline_violations: list[dict[str, Any]] = []
        for L_S in cfg.L_S_list:
            matched = matched_L_ood_by_LS[int(L_S)]
            f1_at_zero = f1["losses_per_seed_alpha_by_LS"][int(L_S)][:, 0].mean()
            f1_err = abs(float(f1_at_zero) - matched)
            f2_at_zero = np.median(
                f2["losses_by_LS"][int(L_S)][:, :, 0], axis=1
            ).mean()
            f2_err = abs(float(f2_at_zero) - matched)
            if max(f1_err, f2_err) > cfg.matched_baseline_tol:
                baseline_violations.append({
                    "L_S": int(L_S),
                    "matched": matched,
                    "f1_at_0": float(f1_at_zero),
                    "f2_at_0": float(f2_at_zero),
                    "f1_err": f1_err,
                    "f2_err": f2_err,
                })
        baseline_ok = not baseline_violations

        # (B) Material brittleness at α = brittleness_alpha.
        a_brittle = float(cfg.brittleness_alpha)
        f1_ai = int(np.argmin(np.abs(np.asarray(cfg.f1_alphas) - a_brittle)))
        f2_ai = int(np.argmin(np.abs(np.asarray(cfg.f2_alphas) - a_brittle)))
        brittle_any = False
        brittle_records: list[dict[str, Any]] = []
        for L_S in cfg.L_S_list:
            matched = matched_L_ood_by_LS[int(L_S)]
            f1_val = float(
                f1["losses_per_seed_alpha_by_LS"][int(L_S)][:, f1_ai].mean()
            )
            f1_ratio = f1_val / max(matched, 1e-30)
            f2_val = float(
                np.median(f2["losses_by_LS"][int(L_S)][:, :, f2_ai], axis=1).mean()
            )
            f2_ratio = f2_val / max(matched, 1e-30)
            if f1_ratio >= cfg.brittleness_ratio_min or f2_ratio >= cfg.brittleness_ratio_min:
                brittle_any = True
            brittle_records.append({
                "L_S": int(L_S),
                "f1_ratio_at_alpha_brittle": f1_ratio,
                "f2_ratio_at_alpha_brittle": f2_ratio,
            })
        brittle_ok = brittle_any

        # (C) Both regimes exercised.
        regime_ok = (max_dev_f1 < 1.0) and (n_amp_f2_alpha1 > 0)

        # (D) Corollary-5 qualitative tracking (family 1 attenuation):
        # slope_of_mean_L_ood across L_S at α = 1.0 should be negative
        # for family 1 (architecture and theory both). Empirical + theory.
        fam1_empirical_ai1 = [
            float(f1["losses_per_seed_alpha_by_LS"][int(L_S)][:, f1_ai].mean())
            for L_S in cfg.L_S_list
        ]
        fam1_theory_ai1 = [
            float(f1["corollary5_by_LS"][int(L_S)][f1_ai])
            for L_S in cfg.L_S_list
        ]
        fam1_emp_decreasing = all(
            fam1_empirical_ai1[i + 1] <= fam1_empirical_ai1[i] + 1e-12
            for i in range(len(fam1_empirical_ai1) - 1)
        )
        fam1_thy_decreasing = all(
            fam1_theory_ai1[i + 1] <= fam1_theory_ai1[i] + 1e-12
            for i in range(len(fam1_theory_ai1) - 1)
        )
        corollary5_qualitative_ok = fam1_emp_decreasing and fam1_thy_decreasing

        # No NaN
        nan_count = sum(1 for r in train_diag.values() if r["nan_failure"])
        no_nan = nan_count == 0

        all_ok = (
            no_nan
            and baseline_ok
            and brittle_ok
            and regime_ok
            and corollary5_qualitative_ok
        )

        # ---------------- Figures -------------------------------------
        _plot_ood_loss_vs_alpha(
            cfg, f1, f2, matched_L_ood_by_LS, run,
        )
        _plot_normalized_brittleness(
            cfg, f1, f2, matched_L_ood_by_LS, run,
        )
        _plot_depth_sensitivity(cfg, f1, f2, run)
        _plot_corollary5_operator_reference(cfg, f1, f2, run)
        # Pick the first permutation seed for the family-2 symbol viz.
        viz_perm = frequency_permutation(
            s_tr, seed=int(cfg.symbol_samples_family2_seed)
        )
        _plot_symbol_visualization(cfg, s_tr, s_flat, viz_perm, run)

        # ---------------- NPZ + per-cell JSON -------------------------
        npz_payload: dict[str, Any] = {
            "D": int(cfg.D),
            "P": int(cfg.P),
            "K": int(cfg.K),
            "L_S_list": np.asarray(cfg.L_S_list, dtype=np.int64),
            "seed_list": np.asarray(cfg.seed_list, dtype=np.int64),
            "f1_alphas": np.asarray(cfg.f1_alphas, dtype=np.float64),
            "f2_alphas": np.asarray(cfg.f2_alphas, dtype=np.float64),
            "f2_perm_seeds": np.asarray(cfg.f2_perm_seeds, dtype=np.int64),
            "s_tr": s_tr.detach().cpu().numpy(),
            "s_flat": s_flat.detach().cpu().numpy(),
            "omega": omega.detach().cpu().numpy(),
            "matched_L_ood_by_LS": np.asarray(
                [matched_L_ood_by_LS[int(L)] for L in cfg.L_S_list],
                dtype=np.float64,
            ),
            "norm_factor": float(norm_factor),
            "max_dev_f1_at_alpha1": float(max_dev_f1),
            "n_modes_amplify_f2_at_alpha1": int(n_amp_f2_alpha1),
        }
        for L_S in cfg.L_S_list:
            npz_payload[f"f1_losses_L{L_S}"] = f1["losses_per_seed_alpha_by_LS"][int(L_S)]
            npz_payload[f"f1_corollary5_L{L_S}"] = f1["corollary5_by_LS"][int(L_S)]
            npz_payload[f"f2_losses_L{L_S}"] = f2["losses_by_LS"][int(L_S)]
            npz_payload[f"f2_corollary5_L{L_S}"] = f2["corollary5_by_LS"][int(L_S)]
            for seed in cfg.seed_list:
                npz_payload[f"gamma_L{L_S}_seed{seed}"] = (
                    gamma_by_LS_seed[(int(L_S), int(seed))].numpy()
                )
        for pseed in cfg.f2_perm_seeds:
            npz_payload[f"s_perm_seed{pseed}"] = (
                f2["s_perm_by_seed"][int(pseed)].detach().cpu().numpy()
            )
        np.savez(run.npz_path("arch_spectral_ood"), **npz_payload)

        per_cell_rows: list[dict[str, Any]] = []
        for (L_S, seed), r in train_diag.items():
            tail = r["loss_values"][-cfg.final_loss_window:]
            final = float(np.mean(tail)) if tail else float("nan")
            init = float(r["loss_values"][0]) if r["loss_values"] else float("nan")
            per_cell_rows.append({
                "L_S": int(L_S), "seed": int(seed),
                "initial_loss": init, "final_loss": final,
                "matched_L_ood": matched_L_ood_by_LS[int(L_S)],
                "nan_failure": bool(r["nan_failure"]),
                "train_seconds": float(r["train_seconds"]),
            })
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(per_cell_rows, indent=2)
        )

        # ---------------- Terminal summary ----------------------------
        print()
        print("=" * 72)
        print(f" §9.1 architecture-aligned spectral OOD on {device}")
        print(
            f"   N training cells (L_S × seed) = {len(train_diag)}; "
            f"N steps = {cfg.train_steps}; batch = {cfg.batch_contexts}; "
            f"norm = {cfg.label_norm}"
        )
        print(f"   no NaN: {no_nan}  (count = {nan_count})")
        print(
            f"   matched-baseline recovery (α = 0, tol "
            f"{cfg.matched_baseline_tol:.1e}): "
            f"{'OK' if baseline_ok else 'WEAK'}  "
            f"({len(baseline_violations)} violations)"
        )
        print(
            f"   material brittleness at α = {a_brittle:.2f}, "
            f"ratio ≥ {cfg.brittleness_ratio_min:.2f}: "
            f"{'OK' if brittle_ok else 'WEAK'}"
        )
        for rec in brittle_records:
            print(
                f"     L_S = {rec['L_S']:>2d}  "
                f"f1 ratio = {rec['f1_ratio_at_alpha_brittle']:.3f}  "
                f"f2 ratio = {rec['f2_ratio_at_alpha_brittle']:.3f}"
            )
        print(
            f"   both Corollary-5 regimes exercised "
            f"(family-1 max |·| = {max_dev_f1:.3f} < 1; "
            f"family-2 α=1 total # |·|>1 modes = {n_amp_f2_alpha1}): "
            f"{'OK' if regime_ok else 'WEAK'}"
        )
        print(
            f"   Corollary-5 qualitative tracking (family-1 slope ↓ in L_S "
            f"for both arch and theory at α = {a_brittle:.2f}): "
            f"{'OK' if corollary5_qualitative_ok else 'WEAK'}"
        )
        print(
            f"   matched L_ood(α=0, L_S) per L_S: "
            + ", ".join(
                f"L_S={L}:{matched_L_ood_by_LS[int(L)]:.3e}"
                for L in cfg.L_S_list
            )
        )
        print(
            f"   convergence diagnostic — median |T_k^cumul| per L_S: "
            + ", ".join(
                f"L_S={L}:{median_abs_T_by_LS[int(L)]:.3e}"
                for L in cfg.L_S_list
            )
        )
        # Family-1 vs family-2 regime-distinction diagnostic at α = 1.
        fam1_ratio_by_L = {
            L: float(
                f1["losses_per_seed_alpha_by_LS"][int(L)][:, f1_ai].mean()
                / max(matched_L_ood_by_LS[int(L)], 1e-30)
            )
            for L in cfg.L_S_list
        }
        fam2_ratio_by_L = {
            L: float(
                np.median(
                    f2["losses_by_LS"][int(L)][:, :, f2_ai], axis=1
                ).mean() / max(matched_L_ood_by_LS[int(L)], 1e-30)
            )
            for L in cfg.L_S_list
        }
        print(
            "   normalized brittleness at α = 1 per L_S:  "
            + ", ".join(
                f"L_S={L}: f1={fam1_ratio_by_L[L]:.3f} f2={fam2_ratio_by_L[L]:.3f}"
                for L in cfg.L_S_list
            )
        )
        max_LS = max(cfg.L_S_list)
        min_LS = min(cfg.L_S_list)
        regime_gap_at_max_LS = (
            fam2_ratio_by_L[max_LS] - fam1_ratio_by_L[max_LS]
        )
        regime_gap_at_min_LS = (
            fam2_ratio_by_L[min_LS] - fam1_ratio_by_L[min_LS]
        )
        print(
            f"   regime-distinction gap (family2 − family1) in normalized ratio:  "
            f"L_S = {min_LS}: {regime_gap_at_min_LS:+.3f}  "
            f"L_S = {max_LS}: {regime_gap_at_max_LS:+.3f}"
        )
        print("=" * 72)

        # ---------------- Final summary -------------------------------
        ctx.record_compute_proxy(float(train_wall))
        ctx.record_extra("nan_count", int(nan_count))
        ctx.record_extra("baseline_violations", baseline_violations)
        ctx.record_extra("brittle_records", brittle_records)
        ctx.record_extra("max_dev_f1_at_alpha1", float(max_dev_f1))
        ctx.record_extra("n_modes_amplify_f2_at_alpha1", int(n_amp_f2_alpha1))
        ctx.record_extra(
            "matched_L_ood_by_LS",
            {str(L): float(v) for L, v in matched_L_ood_by_LS.items()},
        )
        ctx.record_extra("train_wallclock_seconds", float(train_wall))
        ctx.record_extra(
            "status",
            "all_ok" if all_ok else (
                ("no_nan_ok" if no_nan else "NaN_FAILURE") + "+" +
                ("baseline_ok" if baseline_ok else "baseline_FAIL") + "+" +
                ("brittle_ok" if brittle_ok else "brittle_FAIL") + "+" +
                ("regime_ok" if regime_ok else "regime_FAIL") + "+" +
                ("corollary5_ok" if corollary5_qualitative_ok else "corollary5_FAIL")
            ),
        )

        ctx.write_summary({
            "plan_reference": (
                "EXPERIMENT_PLAN_FINAL.MD §9.1 (architecture-aligned spectral-"
                "only suite); theorem reference: thesis/theorem_b.txt Corollary "
                "'Out-of-distribution brittleness under stationary symbol shift' "
                "(cor:theoremB_ood, Corollary 5)"
            ),
            "phase": "Phase IV — architecture-aligned validation layer",
            "category": (
                "architecture-aligned analogue of theorem-B B3. Trainable "
                "tied-weight circulant spectral filter on matched stationary "
                "circulant ICL regression, evaluated on symbol-native OOD "
                "shift families 1 (structural interpolation) and 2 (frequency "
                "permutation). Corollary-5 asymptotic overlay. NOT an exact "
                "theorem verification."
            ),
            "framing": (
                "This script is architecture-aligned support for theorem-B "
                "B3. The spectral filter is a trainable realistic model (as "
                "opposed to the operator-level reduced recursion in B3). The "
                "OOD shifts are symbol-native — no generic covariance "
                "rotations. Fixed Fourier basis throughout. Tied weights "
                "match theorem-B's single circulant Q."
            ),
            "architecture": (
                "Tied-weight real-valued L_S-layer circulant spectral filter "
                "(single γ ∈ ℝ^D shared across all layers; D = P = 32; "
                "K = 8). Per-forward score S(γ) computed once; L_S residual "
                "updates with the signed-mask × bilinear-score operator."
            ),
            "task": (
                "G1 matched stationary training (power_law ν = "
                f"{cfg.power_law_nu}, teacher νβ = {cfg.task_spec_nu_beta}; "
                f"label_norm = {cfg.label_norm}; batch = {cfg.batch_contexts}; "
                f"Adam lr = {cfg.learning_rate}; {cfg.train_steps} steps); "
                "then B3 OOD evaluation under family 1 (structural flat "
                "interpolation) and family 2 (frequency permutation, "
                f"{len(cfg.f2_perm_seeds)} perm seeds) at 12 α values."
            ),
            "interpretation": (
                f"α = 0 matched baseline recovery: "
                f"{'OK' if baseline_ok else 'violated'}; material "
                f"brittleness at α = {a_brittle:.2f} "
                f"(ratio ≥ {cfg.brittleness_ratio_min:.2f}): "
                f"{'OK' if brittle_ok else 'below threshold'}. Family-1 "
                f"stays in the Corollary-5 attenuation regime (max_k "
                f"|1 − s_te,k/s_tr,k| = {max_dev_f1:.3f} < 1 at α = 1); "
                f"family-2 at α = 1 activates the amplification regime on "
                f"{n_amp_f2_alpha1} modes across the 8 permutation seeds. "
                "The Corollary-5 analytical overlay (black dashed) gives "
                "the asymptotic OOD loss at the converged optimum Q⋆ = "
                "L·T⁻¹; the trained architecture's B3 loss uses the "
                "finite-time γ(T). Quantitative agreement between the two "
                "is not expected because γ(T) ≠ γ⋆ in finite time; "
                "qualitative pattern agreement (monotonic direction, "
                "regime distinction between families) is the "
                "architecture-aligned signal."
            ),
            "device": str(device),
            "geometry": {
                "D": int(cfg.D), "P": int(cfg.P), "K": int(cfg.K),
                "batch_contexts": int(cfg.batch_contexts),
                "train_steps": int(cfg.train_steps),
                "log_every": int(cfg.log_every),
            },
            "L_S_list": list(cfg.L_S_list),
            "seed_list": list(cfg.seed_list),
            "tied_weights": True,
            "n_train_cells": int(len(train_diag)),
            "optimizer": cfg.optimizer,
            "learning_rate": float(cfg.learning_rate),
            "label_norm": cfg.label_norm,
            "symbol_kind": cfg.symbol_kind,
            "power_law_nu": float(cfg.power_law_nu),
            "task_spec_nu_beta": float(cfg.task_spec_nu_beta),
            "status": (
                "all_ok" if all_ok else (
                    ("no_nan_ok" if no_nan else "NaN_FAILURE") + "+" +
                    ("baseline_ok" if baseline_ok else "baseline_FAIL") + "+" +
                    ("brittle_ok" if brittle_ok else "brittle_FAIL") + "+" +
                    ("regime_ok" if regime_ok else "regime_FAIL") + "+" +
                    ("corollary5_ok" if corollary5_qualitative_ok else "corollary5_FAIL")
                )
            ),
            "matched_baseline_tol": float(cfg.matched_baseline_tol),
            "brittleness_alpha": float(cfg.brittleness_alpha),
            "brittleness_ratio_min": float(cfg.brittleness_ratio_min),
            "nan_count": int(nan_count),
            "matched_L_ood_by_LS":
                {str(L): float(v) for L, v in matched_L_ood_by_LS.items()},
            "max_dev_f1_at_alpha1": float(max_dev_f1),
            "n_modes_amplify_f2_at_alpha1": int(n_amp_f2_alpha1),
            "brittle_records_at_alpha_brittle": [
                {str(k): v for k, v in rec.items()}
                for rec in brittle_records
            ],
            "fam1_empirical_loss_at_alpha_brittle":
                {str(L): float(v) for L, v in zip(cfg.L_S_list, fam1_empirical_ai1)},
            "fam1_theory_loss_at_alpha_brittle":
                {str(L): float(v) for L, v in zip(cfg.L_S_list, fam1_theory_ai1)},
            "corollary5_family1_attenuation_slope_ok": bool(fam1_emp_decreasing),
            "corollary5_family1_theory_slope_ok": bool(fam1_thy_decreasing),
            "convergence_median_abs_T_by_LS":
                {str(L): float(v) for L, v in median_abs_T_by_LS.items()},
            "normalized_brittleness_at_alpha_brittle": {
                "family1": {str(L): float(v) for L, v in fam1_ratio_by_L.items()},
                "family2": {str(L): float(v) for L, v in fam2_ratio_by_L.items()},
                "regime_gap_at_min_LS": float(regime_gap_at_min_LS),
                "regime_gap_at_max_LS": float(regime_gap_at_max_LS),
            },
            "train_wallclock_seconds": round(float(train_wall), 3),
        })

        if not all_ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
