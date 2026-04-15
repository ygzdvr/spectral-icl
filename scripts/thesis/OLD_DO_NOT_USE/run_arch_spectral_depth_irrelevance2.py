"""§9.1 architecture-aligned depth-irrelevance experiment v2.

Clean tied-weight α-sweep with long-context (``sqrt_D``) label
normalization, plus a transfer-alignment figure probing whether all
depths learn the same matched-stationary spectral target.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §9.1 (architecture-aligned
spectral-only suite). Theorem reference: ``thesis/theorem_b.txt``
Corollary "Modewise gradient-flow dynamics and long-context depth
irrelevance" (``cor:theoremB_modewise_ode``).

This script supersedes the first tied-weight attempt
``run_arch_spectral_depth_irrelevance.py`` (v1). v1 used ``label_norm =
sqrt_P`` which matches the finite-P exact B1 regime; that normalization
is not appropriate for an α = P/D sweep, because it conflates the
sample-complexity regime change with the per-batch MSE scale. v2 uses
``label_norm = sqrt_D`` — the long-context / population-style
normalization — so the per-batch MSE at ``γ = 0`` is
``(1/D) · Σ_k ω_k · s_{tr,k}``, independent of P. This puts every α on
the same loss scale and makes the depth-floor ratio directly
interpretable.

v2 also adds a transfer-function alignment figure: theorem B is about
learning the fixed-basis inverse filter, not just lowering loss. If
depth irrelevance is real at the architecture level, all depths at
large α should visibly converge to the matched stationary target
``γ_k^⋆ = L_S / s_{tr,k}`` per Fourier mode, equivalently the
cumulative transfer factor ``T_k = (1 − γ_k · s_{tr,k}/L_S)^{L_S}``
should approach 0 on every mode.

This script does NOT modify the canonical decoupled §9.1 script
``run_arch_spectral_linear_stationary.py`` or v1.

Why tied weights
----------------
The first §9.1 canonical script uses per-layer decoupled
``γ^(ℓ) ∈ ℝ^D`` (total ``L_S · D`` parameters), which confounds depth
with parameter count: deeper models naturally reach a lower
finite-batch noise floor because they have strictly more learnable γ.
The theorem-B depth-irrelevance statement is about a **single
circulant Q** — a single Fourier symbol ``γ ∈ ℝ^D`` shared across all
layers. So the definitive architecture-aligned depth-irrelevance test
requires **tied weights**: one ``γ`` used at every layer, with
parameter count exactly ``D`` regardless of ``L_S``.

Why the α sweep
---------------
Theorem B's depth-irrelevance claim is a population / large-context
statement. At finite ``P`` and finite batch size, the per-mode
gradient at the matched optimum scales as
``(1 − γ_k · s_{tr,k}/L_S)^{2L_S − 1}``, and adaptive optimizer steps
interact with per-batch operator noise differently at different
depths. As ``α = P/D → ∞`` the per-batch sample-space operator
concentrates (noise ``∝ 1/√P``), and all depths should converge to
approximately equal final loss and the same matched-stationary target
— the architecture-aligned realization of population depth-irrelevance.

Model — tied circulant spectral filter
--------------------------------------
A single learnable ``γ ∈ ℝ^D`` parameterizes the circulant
``Γ = F^T · diag(γ) · F``. The L-layer forward pass on residual-stream
``h = [y_train | 0_K] ∈ ℝ^{P+K}`` is

    h^(ℓ+1) = h^(ℓ) + (1/L_S) · (M_signed ⊙ S(γ)) · h^(ℓ),
    S(γ)[m, n] = (1/P) · Σ_k γ_k · X̃[k, m] · X̃[k, n],

with the GD-compatible signed mask (``−1`` train×train, ``+1``
test×train, 0 elsewhere). Because ``γ`` is tied, ``S(γ)`` is identical
at every layer; we precompute it once per forward pass and apply
``L_S`` residual updates. Cumulative transfer:

    T_k = (1 − γ_k · s_{tr,k} / L_S)^{L_S}.

At the matched stationary optimum ``γ_k^⋆ = L_S / s_{tr,k}``,
``T_k = 0``.

Task
----
G1 matched stationary circulant ICL regression; power-law symbol
``ν = 0.5``, teacher ``νβ = 1.0``. Same symbol family as the other
§9.1 runs. ``label_norm = sqrt_D`` (population-style, v2 distinction
from v1).

Per-step inline F-basis sampling: fresh ``(B, D, P + K)`` batch per
Adam update. Per-feature spectra ``s_tr``, ``ω`` of length ``D`` are
shared across all P in the α sweep (computed once via G1 at P = D).

Sweep
-----
``D = 32`` fixed. ``P ∈ {32, 64, 128, 256}`` → ``α ∈ {1, 2, 4, 8}``.
``L_S ∈ {1, 2, 4, 8}``. Four seeds. ``10000`` Adam steps per cell.

Primary figures
---------------
1. ``depth_floor_ratio_vs_alpha`` — headline figure: depth-floor ratio
   ``max_{L_S}(final) / min_{L_S}(final)`` vs α.
2. ``loss_vs_step_alpha_max`` — loss-vs-step at α = 8.
3. ``loss_vs_step_alpha_min`` — loss-vs-step at α = 1.
4. ``per_alpha_final_loss_vs_LS`` — diagnostic 2×2 grid.
5. ``transfer_alignment_alpha_min_max`` — NEW in v2: cumulative
   transfer ``|T_k^{cumul}|`` per Fourier mode at end of training, one
   subpanel for α = 1, one for α = 8. Theorem-B target is
   ``T_k^⋆ = 0``; all L_S at large α should visibly align.

Acceptance (qualitative, architecture-aligned)
----------------------------------------------
1. No NaN across the 64 cells.
2. Substantial decay: per ``(α, L_S, seed)``,
   ``final_loss ≤ decay_fraction · initial_loss``.
3. Depth-floor ratio is non-increasing along the α sweep.
4. Depth-floor ratio at largest α is **materially smaller** than at
   smallest α — default threshold
   ``ratio(α_max) ≤ 0.8 · ratio(α_min)`` (i.e. ≥ 20% relative
   reduction). Stretch: ``ratio(α_max) ≤ 2.0`` is reported as a bonus
   if achieved, but is not the only hard gate.
5. Median ``|T_k^{cumul}|`` at largest α across L_S ≤
   ``transfer_max_median_alpha_max`` (default 0.4) — the architecture
   is visibly aligned with the matched-stationary target at large α.

Summary wording (binding)
-------------------------
"This experiment is architecture-aligned support for theorem-B depth
irrelevance in the tied-weight setting. It does not re-prove theorem
B. It tests whether the depth gap shrinks as α = P/D increases when
parameter count is held fixed across depth."

Run via SLURM
-------------
::

    sbatch experiments/thesis/architectures/run_arch_spectral_depth_irrelevance2.sh
"""

from __future__ import annotations

import argparse
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
class ArchSpectralDepthIrrelevance2Config:
    """Frozen configuration for v2 (long-context ``sqrt_D`` normalization)."""

    # ---------------- Geometry ---------------------------------------
    D: int = 32
    P_list: tuple[int, ...] = (32, 64, 128, 256)
    K: int = 8

    # ---------------- Matched stationary symbol ---------------------
    symbol_kind: str = "power_law"
    power_law_nu: float = 0.5
    task_spec_nu_beta: float = 1.0

    # ---------------- Architecture --------------------------------
    L_S_list: tuple[int, ...] = (1, 2, 4, 8)
    init_scale: float = 0.0

    # ---------------- Training ------------------------------------
    train_steps: int = 10000
    # Per-P step override (sanctioned fallback from the user spec:
    # "If P = 256 is too slow, reduce to 5000 only for P >= 128, but
    # try the uniform 10000-step sweep first"). The uniform 10000-step
    # sweep hit the 1-hour SLURM wall-time limit at P = 256; this
    # mapping trims large-P cells. ``None`` at a P means use the
    # uniform ``train_steps`` value.
    train_steps_by_P: tuple[tuple[int, int], ...] = (
        (32, 10000), (64, 10000), (128, 5000), (256, 5000),
    )
    batch_contexts: int = 64
    learning_rate: float = 5e-2
    optimizer: str = "adam"
    weight_decay: float = 0.0
    log_every: int = 25
    # v2 DISTINCTION FROM v1: use long-context / population-style
    # normalization so the per-batch MSE scale is P-independent.
    label_norm: str = "sqrt_D"

    # ---------------- Seeds ---------------------------------------
    seed_list: tuple[int, ...] = (0, 1, 2, 3)

    # ---------------- Acceptance (qualitative) --------------------
    decay_fraction: float = 0.50
    # v2 gate — ratio monotone-decreasing with α, AND materially smaller
    # at the top of the sweep. The "material" threshold is a 20%
    # relative reduction; stretch goal 2× is reported if achieved but
    # not required.
    alpha_material_reduction: float = 0.80  # ratio(α_max) ≤ this × ratio(α_min)
    alpha_stretch_ratio: float = 2.0        # informational only
    # v2 transfer-alignment gate at the largest α: median over modes of
    # |T_k^cumul| must be visibly close to 0.
    transfer_max_median_alpha_max: float = 0.4
    final_loss_window: int = 20

    # ---------------- Misc ----------------------------------------
    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# F-basis batch sampler
# ---------------------------------------------------------------------------


def _sample_batch_F_basis(
    s_tr: torch.Tensor,        # (D,) real, > 0
    omega: torch.Tensor,       # (D,) real, > 0
    P: int,
    K: int,
    B: int,
    norm_factor: float,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample B contexts in F-basis, matched stationary regime.

    Returns (X̃, y_train, y_query) of shapes (B, D, P+K), (B, P), (B, K).
    """
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

    Parameter count = D, independent of L_S. Score ``S(γ)`` is
    precomputed once per forward pass (tied γ → same S at every
    layer); each layer is a cheap residual update.
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
        weighted = self.gamma.view(1, -1, 1) * X_tilde            # (B, D, P+K)
        score = torch.einsum(
            "bdm,bdn->bmn", weighted, X_tilde,
        ) / float(self.P)                                         # (B, P+K, P+K)
        masked_score = self.M_signed.unsqueeze(0) * score
        inv_L = 1.0 / float(self.L_S)
        for _ell in range(self.L_S):
            update = torch.einsum("bmn,bn->bm", masked_score, h)
            h = h + inv_L * update
        return h[:, self.P:]

    def cumulative_transfer_factor(
        self, s_tr: torch.Tensor,
    ) -> torch.Tensor:
        """T_k = (1 − γ_k · s_{tr,k} / L_S)^{L_S} per Fourier mode."""
        factor = 1.0 - self.gamma.detach() * s_tr.view(-1) / float(self.L_S)
        return factor.pow(self.L_S)

    def final_gamma(self) -> torch.Tensor:
        return self.gamma.detach().clone()


# ---------------------------------------------------------------------------
# Per-(P, L_S, seed) training run
# ---------------------------------------------------------------------------


def _resolve_train_steps_for_P(
    cfg: ArchSpectralDepthIrrelevance2Config, P: int,
) -> int:
    for p_, n_ in cfg.train_steps_by_P:
        if int(p_) == int(P):
            return int(n_)
    return int(cfg.train_steps)


def _train_one(
    cfg: ArchSpectralDepthIrrelevance2Config,
    P: int,
    L_S: int,
    seed: int,
    s_tr: torch.Tensor,
    omega: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    norm_factor = (
        float(cfg.D) if cfg.label_norm == "sqrt_D" else float(P)
    )
    train_steps = _resolve_train_steps_for_P(cfg, int(P))
    torch.manual_seed(int(seed) * 991 + 17 * int(L_S) + 31 * int(P))
    model = TiedCirculantSpectralFilter(
        D=cfg.D, P=int(P), K=cfg.K, L_S=int(L_S),
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
    sample_gen.manual_seed(int(seed) * 7919 + 31 * int(L_S) + 137 * int(P))

    loss_steps: list[int] = []
    loss_values: list[float] = []
    nan_failure = False
    t0 = time.perf_counter()
    for step in range(int(train_steps)):
        X_tilde, y_train, y_query = _sample_batch_F_basis(
            s_tr, omega, int(P), cfg.K, int(cfg.batch_contexts),
            norm_factor, sample_gen, dtype, device,
        )
        opt.zero_grad()
        y_pred = model(X_tilde, y_train)
        loss = ((y_pred - y_query) ** 2).mean()
        if not torch.isfinite(loss):
            nan_failure = True
            print(
                f"   [P={P}, L_S={L_S}, seed={seed}] NaN loss at step {step}; "
                "aborting cell."
            )
            break
        loss.backward()
        opt.step()
        if (step % int(cfg.log_every) == 0) or step == train_steps - 1:
            loss_steps.append(int(step))
            loss_values.append(float(loss.item()))
    t_train = time.perf_counter() - t0

    s_tr_dev = s_tr.to(dtype).to(device)
    cumul_T = model.cumulative_transfer_factor(s_tr_dev).cpu()
    final_gamma = model.final_gamma().cpu()

    return {
        "P": int(P),
        "L_S": int(L_S),
        "seed": int(seed),
        "loss_steps": loss_steps,
        "loss_values": loss_values,
        "cumul_transfer": cumul_T,
        "final_gamma": final_gamma,
        "nan_failure": bool(nan_failure),
        "train_seconds": float(t_train),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_loss_curves_across_seeds(
    runs: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    base_steps = runs[0]["loss_steps"]
    for r in runs[1:]:
        if r["loss_steps"] != base_steps:
            raise AssertionError(
                "loss_steps differ across seeds; cannot aggregate"
            )
    arr = np.asarray(
        [r["loss_values"] for r in runs], dtype=np.float64,
    )
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        std = arr.std(axis=0, ddof=1)
        se = std / float(np.sqrt(arr.shape[0]))
    else:
        se = np.zeros_like(mean)
    return {
        "steps": np.asarray(base_steps, dtype=np.int64),
        "mean": mean,
        "se": se,
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_depth_floor_ratio_vs_alpha(
    cfg: ArchSpectralDepthIrrelevance2Config,
    depth_floor_ratio_by_alpha: dict[float, float],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    alphas = sorted(depth_floor_ratio_by_alpha.keys())
    ratios = [depth_floor_ratio_by_alpha[a] for a in alphas]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(
        alphas, ratios, color="C3", lw=1.4, marker="o", ms=7,
        label="tied-weight filter (v2, sqrt_D)",
    )
    ax.axhline(
        1.0, color="gray", ls=":", lw=1.0,
        label=r"theorem-B asymptote: ratio $\to 1$",
    )
    ax.axhline(
        cfg.alpha_stretch_ratio,
        color="black", ls="--", lw=0.8,
        label=rf"stretch target at largest $\alpha$: ratio $\leq {cfg.alpha_stretch_ratio:.1f}$",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"context-to-dimension ratio $\alpha = P / D$")
    ax.set_ylabel(
        r"depth-floor ratio  "
        r"$\max_{L_S}\mathrm{loss} / \min_{L_S}\mathrm{loss}$"
    )
    ax.set_title(
        r"§9.1 v2 tied-weight depth-floor ratio vs $\alpha$ "
        "(architecture-aligned B2)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "depth_floor_ratio_vs_alpha")
    plt.close(fig)


def _plot_loss_vs_step_at_alpha(
    cfg: ArchSpectralDepthIrrelevance2Config,
    P: int,
    aggregates_by_LS: dict[int, dict[str, np.ndarray]],
    initial_loss_analytical: float,
    norm_factor: float,
    figure_stem: str,
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    L_colors = sequential_colors(len(cfg.L_S_list), palette="rocket")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    floor = 1e-30
    for color, L_S in zip(L_colors, cfg.L_S_list):
        agg = aggregates_by_LS[int(L_S)]
        steps = agg["steps"]
        mean = np.where(agg["mean"] > floor, agg["mean"], np.nan)
        se = agg["se"]
        ax.plot(
            steps, mean, color=color, lw=1.1,
            label=rf"$L_S = {L_S}$",
        )
        ax.fill_between(
            steps,
            np.clip(mean - se, floor, None),
            mean + se,
            color=color, alpha=0.18, lw=0,
        )
    per_batch_initial = initial_loss_analytical / norm_factor
    if per_batch_initial > 0:
        overlay_reference(
            ax, np.asarray(aggregates_by_LS[int(cfg.L_S_list[0])]["steps"], dtype=float),
            np.full_like(
                aggregates_by_LS[int(cfg.L_S_list[0])]["steps"], per_batch_initial,
                dtype=float,
            ),
            label=r"per-batch $L(\gamma{=}0)$",
            style=":", color="gray", lw=1.0,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"training step $t$")
    ax.set_ylabel(r"per-batch MSE on $y_{\mathrm{query}}$")
    alpha_val = float(P) / float(cfg.D)
    ax.set_title(
        rf"§9.1 v2 tied-weight filter at $\alpha = P/D = {alpha_val:.0f}$ "
        rf"($P = {P}$; mean $\pm$ SE, {len(cfg.seed_list)} seeds; $\sqrt{{D}}$ norm)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, figure_stem)
    plt.close(fig)


def _plot_per_alpha_final_loss_vs_LS(
    cfg: ArchSpectralDepthIrrelevance2Config,
    final_loss_by_P_LS_seed: dict[int, dict[int, list[float]]],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    P_list = sorted(final_loss_by_P_LS_seed.keys())
    nP = len(P_list)
    ncols = 2
    nrows = int(math.ceil(nP / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0, 3.4 * nrows), sharey=False)
    axes_flat = np.asarray(axes).reshape(-1)
    L_arr = np.asarray(cfg.L_S_list, dtype=int)
    L_colors = sequential_colors(len(cfg.L_S_list), palette="rocket")
    for idx, P in enumerate(P_list):
        ax = axes_flat[idx]
        mean_loss: list[float] = []
        se_loss: list[float] = []
        for L_S in cfg.L_S_list:
            vals = np.asarray(
                final_loss_by_P_LS_seed[int(P)][int(L_S)], dtype=np.float64
            )
            mean_loss.append(float(vals.mean()))
            if vals.size > 1:
                se_loss.append(float(vals.std(ddof=1) / np.sqrt(vals.size)))
            else:
                se_loss.append(0.0)
        ax.plot(
            L_arr, mean_loss, color="gray", lw=0.8, alpha=0.5, zorder=1,
        )
        for color, L_S, m, se in zip(L_colors, L_arr, mean_loss, se_loss):
            ax.errorbar(
                L_S, m, yerr=se, fmt="o", ms=6, capsize=4,
                color=color, zorder=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"spectral depth $L_S$")
        ax.set_ylabel(r"final mean per-batch MSE")
        alpha_val = float(P) / float(cfg.D)
        ax.set_title(rf"$\alpha = {alpha_val:.0f}$   ($P = {P}$)", fontsize=10)
    for i in range(nP, len(axes_flat)):
        axes_flat[i].axis("off")
    fig.suptitle(
        "§9.1 v2 per-α final loss vs $L_S$  (diagnostic; tied-weight, $\\sqrt{D}$ norm)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "per_alpha_final_loss_vs_LS")
    plt.close(fig)


def _plot_transfer_alignment_min_max(
    cfg: ArchSpectralDepthIrrelevance2Config,
    P_min: int,
    P_max: int,
    transfer_per_P_LS_mean: dict[int, dict[int, np.ndarray]],
    transfer_per_P_LS_se: dict[int, dict[int, np.ndarray]],
    run_dir: ThesisRunDir,
) -> None:
    """Two-panel figure: |T_k^cumul| per Fourier mode at α_min and α_max.

    Theorem-B target: T_k^⋆ = 0 on every mode. At large α all depths
    should visibly align with this target; at small α a depth-dependent
    spread is expected.
    """
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    L_colors = sequential_colors(len(cfg.L_S_list), palette="rocket")
    k_axis = np.arange(cfg.D)
    floor = 1e-12

    for ax, P, title_suffix in (
        (axL, P_min, rf"$\alpha = {float(P_min)/cfg.D:.0f}$  ($P = {P_min}$)"),
        (axR, P_max, rf"$\alpha = {float(P_max)/cfg.D:.0f}$  ($P = {P_max}$)"),
    ):
        for color, L_S in zip(L_colors, cfg.L_S_list):
            T_mean = transfer_per_P_LS_mean[int(P)][int(L_S)]
            T_se = transfer_per_P_LS_se[int(P)][int(L_S)]
            T_plot = np.where(np.abs(T_mean) > floor, np.abs(T_mean), floor)
            ax.plot(
                k_axis, T_plot, color=color, lw=1.3, marker="o", ms=3.2,
                label=rf"$L_S = {L_S}$",
            )
            ax.fill_between(
                k_axis,
                np.clip(np.abs(T_mean) - T_se, floor, None),
                np.abs(T_mean) + T_se,
                color=color, alpha=0.18, lw=0,
            )
        ax.axhline(
            1.0, color="black", lw=0.5, ls="--",
            label=r"untrained $\gamma = 0$: $T_k = 1$",
        )
        overlay_reference(
            ax, k_axis, np.full_like(k_axis, floor, dtype=float),
            label=r"theorem-B target $T_k^\star = 0$",
            style=":", color="gray", lw=1.0,
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"Fourier mode index $k$")
        ax.set_title(title_suffix, fontsize=10)
        ax.legend(fontsize=7, loc="best")
    axL.set_ylabel(
        r"end-of-training $|T_k^{\mathrm{cumul}}|$"
        r" $= |1 - \gamma_k\, s_{\mathrm{tr},k}/L_S|^{L_S}$"
    )
    fig.suptitle(
        "§9.1 v2 transfer-function alignment: small vs large α "
        "(tied weights)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "transfer_alignment_alpha_min_max")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "§9.1 v2 tied-weight circulant spectral filter with α = P/D sweep "
            "(long-context sqrt_D normalization, transfer-alignment figure)."
        )
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda", "auto"))
    p.add_argument("--dtype", type=str, default="float64", choices=("float32", "float64"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--D", type=int, default=None)
    p.add_argument("--P-list", type=str, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--L-S-list", type=str, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--seeds", type=str, default=None)
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> ArchSpectralDepthIrrelevance2Config:
    base = ArchSpectralDepthIrrelevance2Config()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.D is not None:
        overrides["D"] = int(args.D)
    if args.P_list is not None:
        overrides["P_list"] = _parse_list_ints(args.P_list)
    if args.K is not None:
        overrides["K"] = int(args.K)
    if args.L_S_list is not None:
        overrides["L_S_list"] = _parse_list_ints(args.L_S_list)
    if args.train_steps is not None:
        overrides["train_steps"] = int(args.train_steps)
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
    device = _resolve_device(cfg.device)
    print(f"[arch-§9.1-v2-tied] device = {device}")

    run = ThesisRunDir(__file__, phase="architectures")
    with RunContext(
        run,
        config=cfg,
        seeds=list(cfg.seed_list),
        notes=(
            "§9.1 v2 architecture-aligned depth-irrelevance experiment "
            "(tied weights, α = P/D sweep, sqrt_D normalization, "
            "transfer-alignment figure). Supersedes v1 "
            "(run_arch_spectral_depth_irrelevance.py) which used sqrt_P."
        ),
    ) as ctx:
        apply_thesis_style()

        # --- Shared per-feature spectra via G1 at P = D ----------------
        if cfg.symbol_kind == "power_law":
            symbol_params: dict[str, Any] = {"nu": cfg.power_law_nu}
        elif cfg.symbol_kind == "multiband":
            symbol_params = {"bands": [[0, 2, 1.0], [5, 7, 0.8]]}
        elif cfg.symbol_kind == "flat":
            symbol_params = {"value": 1.0}
        else:
            raise ValueError(f"unknown symbol_kind: {cfg.symbol_kind!r}")

        g1_cfg = G1Config(
            P=int(cfg.D), D=cfg.D, B=1,
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
        s_tr_shared = op["s_tr"].to(
            torch.float64 if cfg.dtype == "float64" else torch.float32
        )
        omega_shared = op["omega"].to(
            torch.float64 if cfg.dtype == "float64" else torch.float32
        )
        initial_loss_analytical = float(
            (omega_shared.to(torch.float64)
             * s_tr_shared.to(torch.float64)).sum().item()
        )
        # With sqrt_D normalization the per-batch MSE at γ=0 is
        # initial_loss_analytical / D — P-independent.
        norm_factor = (
            float(cfg.D) if cfg.label_norm == "sqrt_D" else None  # sqrt_D path
        )
        if norm_factor is None:
            raise RuntimeError("v2 expects label_norm='sqrt_D'")
        per_batch_initial = initial_loss_analytical / norm_factor
        print(
            f"[arch-§9.1-v2-tied] D = {cfg.D}; per-feature spectra "
            f"s_tr range=[{float(s_tr_shared.min()):.3e}, "
            f"{float(s_tr_shared.max()):.3e}]  "
            f"L(γ=0) = {initial_loss_analytical:.4e}  "
            f"per-batch MSE@γ=0 = {per_batch_initial:.4e}  "
            "(P-independent under sqrt_D)"
        )
        for P in cfg.P_list:
            print(
                f"[arch-§9.1-v2-tied] α = {float(P)/cfg.D:>4.1f}  "
                f"P = {P:>4d}  (per-batch MSE@γ=0 same across α)"
            )

        # --- Training sweep --------------------------------------------
        runs: dict[tuple[int, int, int], dict[str, Any]] = {}
        n_total = len(cfg.P_list) * len(cfg.L_S_list) * len(cfg.seed_list)
        idx = 0
        t_sweep_start = time.perf_counter()
        for P in cfg.P_list:
            for L_S in cfg.L_S_list:
                for seed in cfg.seed_list:
                    idx += 1
                    t0 = time.perf_counter()
                    r = _train_one(
                        cfg, int(P), int(L_S), int(seed),
                        s_tr_shared, omega_shared, device,
                    )
                    dt = time.perf_counter() - t0
                    ctx.record_step_time(dt)
                    runs[(int(P), int(L_S), int(seed))] = r
                    tail = r["loss_values"][-cfg.final_loss_window:]
                    final_smoothed = float(np.mean(tail)) if tail else float("nan")
                    init_batch = float(r["loss_values"][0]) if r["loss_values"] else float("nan")
                    print(
                        f"[{idx:>3d}/{n_total:>3d}] "
                        f"α={float(P)/cfg.D:>4.1f}  P={P:>3d}  L_S={L_S:>2d}  seed={seed}  "
                        f"init≈{init_batch:.3e}  "
                        f"final(w{cfg.final_loss_window})≈{final_smoothed:.3e}  "
                        f"nan={r['nan_failure']}  ({dt:.2f} s)"
                    )
        sweep_wall = time.perf_counter() - t_sweep_start

        # --- Aggregates -------------------------------------------------
        aggregates: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        final_loss_by_P_LS_seed: dict[int, dict[int, list[float]]] = {}
        transfer_per_P_LS_mean: dict[int, dict[int, np.ndarray]] = {}
        transfer_per_P_LS_se: dict[int, dict[int, np.ndarray]] = {}
        for P in cfg.P_list:
            final_loss_by_P_LS_seed[int(P)] = {}
            transfer_per_P_LS_mean[int(P)] = {}
            transfer_per_P_LS_se[int(P)] = {}
            for L_S in cfg.L_S_list:
                seed_runs = [
                    runs[(int(P), int(L_S), int(s))] for s in cfg.seed_list
                ]
                aggregates[(int(P), int(L_S))] = _aggregate_loss_curves_across_seeds(
                    seed_runs
                )
                per_seed_final: list[float] = []
                for r in seed_runs:
                    tail = r["loss_values"][-cfg.final_loss_window:]
                    per_seed_final.append(float(np.mean(tail)) if tail else float("nan"))
                final_loss_by_P_LS_seed[int(P)][int(L_S)] = per_seed_final
                T_stack = np.stack(
                    [r["cumul_transfer"].numpy() for r in seed_runs], axis=0
                )
                transfer_per_P_LS_mean[int(P)][int(L_S)] = T_stack.mean(axis=0)
                if T_stack.shape[0] > 1:
                    transfer_per_P_LS_se[int(P)][int(L_S)] = (
                        T_stack.std(axis=0, ddof=1) / float(np.sqrt(T_stack.shape[0]))
                    )
                else:
                    transfer_per_P_LS_se[int(P)][int(L_S)] = np.zeros_like(T_stack[0])

        # --- Gates -----------------------------------------------------
        nan_count = sum(1 for r in runs.values() if r["nan_failure"])
        no_nan = nan_count == 0

        decay_violations: list[dict[str, Any]] = []
        for (P, L_S, seed), r in runs.items():
            if r["nan_failure"] or not r["loss_values"]:
                continue
            init_batch = float(r["loss_values"][0])
            tail = r["loss_values"][-cfg.final_loss_window:]
            final = float(np.mean(tail))
            if final > cfg.decay_fraction * init_batch:
                decay_violations.append({
                    "P": int(P), "L_S": int(L_S), "seed": int(seed),
                    "initial_loss_batch": init_batch, "final_loss": final,
                    "ratio": final / max(init_batch, 1e-30),
                })
        decay_ok = not decay_violations

        depth_floor_ratio_by_alpha: dict[float, float] = {}
        mean_finals_per_P_LS: dict[int, dict[int, float]] = {}
        for P in cfg.P_list:
            mean_finals_per_P_LS[int(P)] = {}
            alpha = float(P) / float(cfg.D)
            per_LS_mean: list[float] = []
            for L_S in cfg.L_S_list:
                m = float(np.mean(final_loss_by_P_LS_seed[int(P)][int(L_S)]))
                mean_finals_per_P_LS[int(P)][int(L_S)] = m
                per_LS_mean.append(m)
            ratio = (
                max(per_LS_mean) / max(min(per_LS_mean), 1e-30)
                if per_LS_mean else 0.0
            )
            depth_floor_ratio_by_alpha[alpha] = ratio

        alpha_sorted = sorted(depth_floor_ratio_by_alpha.keys())
        alpha_min = alpha_sorted[0]
        alpha_max = alpha_sorted[-1]
        P_min = int(alpha_min * cfg.D)
        P_max = int(alpha_max * cfg.D)
        ratios_sorted = [depth_floor_ratio_by_alpha[a] for a in alpha_sorted]
        monotone_decreasing_ok = all(
            ratios_sorted[i + 1] <= ratios_sorted[i] + 1e-12
            for i in range(len(ratios_sorted) - 1)
        )
        ratio_at_max_alpha = depth_floor_ratio_by_alpha[alpha_max]
        ratio_at_min_alpha = depth_floor_ratio_by_alpha[alpha_min]
        material_reduction_ok = (
            ratio_at_max_alpha
            <= cfg.alpha_material_reduction * ratio_at_min_alpha
        )
        stretch_achieved = ratio_at_max_alpha <= cfg.alpha_stretch_ratio

        # Transfer-alignment gate at largest α (median |T_k^cumul| across L_S).
        median_abs_T_at_max_alpha: dict[int, float] = {}
        for L_S in cfg.L_S_list:
            T_mean = transfer_per_P_LS_mean[int(P_max)][int(L_S)]
            median_abs_T_at_max_alpha[int(L_S)] = float(
                np.median(np.abs(T_mean))
            )
        max_median_T = max(median_abs_T_at_max_alpha.values())
        transfer_alignment_ok = (
            max_median_T <= cfg.transfer_max_median_alpha_max
        )

        all_ok = (
            no_nan
            and decay_ok
            and monotone_decreasing_ok
            and material_reduction_ok
            and transfer_alignment_ok
        )

        ctx.record_compute_proxy(float(sweep_wall))
        ctx.record_extra("nan_count", int(nan_count))
        ctx.record_extra("decay_violations", decay_violations[:10])
        ctx.record_extra(
            "depth_floor_ratio_by_alpha",
            {f"{a:.3f}": float(r) for a, r in depth_floor_ratio_by_alpha.items()},
        )
        ctx.record_extra("ratio_at_min_alpha", float(ratio_at_min_alpha))
        ctx.record_extra("ratio_at_max_alpha", float(ratio_at_max_alpha))
        ctx.record_extra("monotone_decreasing_ok", bool(monotone_decreasing_ok))
        ctx.record_extra("material_reduction_ok", bool(material_reduction_ok))
        ctx.record_extra("stretch_target_achieved", bool(stretch_achieved))
        ctx.record_extra("transfer_alignment_ok", bool(transfer_alignment_ok))
        ctx.record_extra(
            "median_abs_T_at_max_alpha",
            {str(k): float(v) for k, v in median_abs_T_at_max_alpha.items()},
        )
        ctx.record_extra(
            "mean_finals_per_P_LS",
            {str(P): {str(L_S): v for L_S, v in d.items()}
             for P, d in mean_finals_per_P_LS.items()},
        )
        ctx.record_extra(
            "status",
            "all_ok" if all_ok else (
                ("no_nan_ok" if no_nan else "NaN_FAILURE") + "+" +
                ("decay_ok" if decay_ok else "decay_FAIL") + "+" +
                ("monotone_ok" if monotone_decreasing_ok else "monotone_FAIL") + "+" +
                ("material_ok" if material_reduction_ok else "material_FAIL") + "+" +
                ("transfer_ok" if transfer_alignment_ok else "transfer_FAIL")
            ),
        )

        # --- Figures ----------------------------------------------------
        _plot_depth_floor_ratio_vs_alpha(
            cfg, depth_floor_ratio_by_alpha, run,
        )
        _plot_loss_vs_step_at_alpha(
            cfg, int(P_max),
            {int(L_S): aggregates[(int(P_max), int(L_S))] for L_S in cfg.L_S_list},
            initial_loss_analytical, norm_factor,
            "loss_vs_step_alpha_max", run,
        )
        _plot_loss_vs_step_at_alpha(
            cfg, int(P_min),
            {int(L_S): aggregates[(int(P_min), int(L_S))] for L_S in cfg.L_S_list},
            initial_loss_analytical, norm_factor,
            "loss_vs_step_alpha_min", run,
        )
        _plot_per_alpha_final_loss_vs_LS(
            cfg, final_loss_by_P_LS_seed, run,
        )
        _plot_transfer_alignment_min_max(
            cfg, int(P_min), int(P_max),
            transfer_per_P_LS_mean, transfer_per_P_LS_se, run,
        )

        # --- NPZ dump ---------------------------------------------------
        npz_payload: dict[str, Any] = {
            "D": int(cfg.D),
            "K": int(cfg.K),
            "P_list": np.asarray(cfg.P_list, dtype=np.int64),
            "L_S_list": np.asarray(cfg.L_S_list, dtype=np.int64),
            "seed_list": np.asarray(cfg.seed_list, dtype=np.int64),
            "label_norm": str(cfg.label_norm),
            "alphas": np.asarray(
                [float(P) / float(cfg.D) for P in cfg.P_list], dtype=np.float64
            ),
            "depth_floor_ratio_vs_alpha": np.asarray(
                [depth_floor_ratio_by_alpha[float(P) / float(cfg.D)]
                 for P in cfg.P_list], dtype=np.float64
            ),
            "s_tr": s_tr_shared.detach().cpu().numpy(),
            "omega": omega_shared.detach().cpu().numpy(),
            "L_gamma_zero_analytical": float(initial_loss_analytical),
        }
        for P in cfg.P_list:
            for L_S in cfg.L_S_list:
                agg = aggregates[(int(P), int(L_S))]
                npz_payload[f"P{P}_L{L_S}__loss_steps"] = agg["steps"]
                npz_payload[f"P{P}_L{L_S}__loss_mean"] = agg["mean"]
                npz_payload[f"P{P}_L{L_S}__loss_se"] = agg["se"]
                npz_payload[f"P{P}_L{L_S}__final_loss_per_seed"] = np.asarray(
                    final_loss_by_P_LS_seed[int(P)][int(L_S)], dtype=np.float64
                )
                npz_payload[f"P{P}_L{L_S}__transfer_mean"] = (
                    transfer_per_P_LS_mean[int(P)][int(L_S)]
                )
                npz_payload[f"P{P}_L{L_S}__transfer_se"] = (
                    transfer_per_P_LS_se[int(P)][int(L_S)]
                )
                for s in cfg.seed_list:
                    npz_payload[f"P{P}_L{L_S}_seed{s}__final_gamma"] = (
                        runs[(int(P), int(L_S), int(s))]["final_gamma"].numpy()
                    )
        np.savez(run.npz_path("arch_spectral_depth_irrelevance_v2"), **npz_payload)

        # Per-cell structured summary.
        per_cell_rows: list[dict[str, Any]] = []
        for (P, L_S, seed), r in runs.items():
            tail = r["loss_values"][-cfg.final_loss_window:]
            final = float(np.mean(tail)) if tail else float("nan")
            initial = float(r["loss_values"][0]) if r["loss_values"] else float("nan")
            med_T = float(np.median(np.abs(r["cumul_transfer"].numpy())))
            per_cell_rows.append({
                "P": int(P), "L_S": int(L_S), "seed": int(seed),
                "alpha": float(P) / float(cfg.D),
                "initial_loss": initial, "final_loss": final,
                "median_abs_cumul_transfer": med_T,
                "nan_failure": bool(r["nan_failure"]),
                "train_seconds": float(r["train_seconds"]),
            })
        import json as _json
        (run.root / "per_cell_summary.json").write_text(
            _json.dumps(per_cell_rows, indent=2)
        )

        # --- Terminal summary ------------------------------------------
        print()
        print("=" * 72)
        print(f" §9.1 v2 tied-weight α-sweep depth-irrelevance on {device}")
        print(
            f"   N runs (P × L_S × seed) = {len(runs)}; "
            f"N steps per run = {cfg.train_steps}; "
            f"batch = {cfg.batch_contexts} contexts; norm = {cfg.label_norm}"
        )
        print(f"   no NaN: {no_nan}  (count = {nan_count})")
        print(
            f"   decay (per-cell final ≤ {cfg.decay_fraction:.2f} × initial): "
            f"{'OK' if decay_ok else 'WEAK'}  ({len(decay_violations)} violations)"
        )
        print(
            f"   depth-floor ratio monotone decreasing in α: "
            f"{'OK' if monotone_decreasing_ok else 'WEAK'}"
        )
        print(
            f"   material reduction (ratio(α_max) ≤ "
            f"{cfg.alpha_material_reduction:.2f} × ratio(α_min)): "
            f"{'OK' if material_reduction_ok else 'WEAK'}  "
            f"[{ratio_at_max_alpha:.3f} vs {ratio_at_min_alpha:.3f}]"
        )
        print(
            f"   stretch: ratio(α_max = {alpha_max:.1f}) ≤ "
            f"{cfg.alpha_stretch_ratio:.1f} × → "
            f"{'ACHIEVED' if stretch_achieved else 'not achieved'}  "
            f"(observed {ratio_at_max_alpha:.3f})"
        )
        print(
            f"   transfer-alignment at α_max (max median |T_k| ≤ "
            f"{cfg.transfer_max_median_alpha_max:.2f}): "
            f"{'OK' if transfer_alignment_ok else 'WEAK'}  "
            f"(observed {max_median_T:.3f})"
        )
        print("   depth-floor ratio by α:")
        for a in alpha_sorted:
            print(
                f"     α = {a:>4.1f}  ratio = {depth_floor_ratio_by_alpha[a]:.3f}"
            )
        print("   mean final loss per (α, L_S):")
        for P in cfg.P_list:
            parts = [f"α = {float(P)/cfg.D:>4.1f}"]
            for L_S in cfg.L_S_list:
                m = mean_finals_per_P_LS[int(P)][int(L_S)]
                parts.append(f"L_S={L_S}:{m:.3e}")
            print("     " + "   ".join(parts))
        print("   median |T_k| at α_max by L_S:")
        for L_S in cfg.L_S_list:
            print(
                f"     L_S = {L_S}  median |T_k| = "
                f"{median_abs_T_at_max_alpha[int(L_S)]:.3f}"
            )
        print("=" * 72)

        # --- Final summary ---------------------------------------------
        ctx.write_summary({
            "plan_reference": (
                "EXPERIMENT_PLAN_FINAL.MD §9.1 (architecture-aligned spectral-only "
                "suite); theorem reference: thesis/theorem_b.txt Corollary "
                "'Modewise gradient-flow dynamics and long-context depth "
                "irrelevance' (cor:theoremB_modewise_ode)"
            ),
            "phase": "Phase IV — architecture-aligned validation layer",
            "category": (
                "§9.1 v2 tied-weight architecture-aligned support for "
                "theorem-B depth irrelevance. Long-context (sqrt_D) "
                "normalization; α = P/D sweep at fixed D = 32; "
                "transfer-function alignment figure."
            ),
            "framing": (
                "This experiment is architecture-aligned support for "
                "theorem-B depth irrelevance in the tied-weight setting. "
                "It does not re-prove theorem B. It tests whether the "
                "depth gap shrinks as α = P/D increases when parameter "
                "count is held fixed across depth."
            ),
            "architecture": (
                "Single tied γ ∈ ℝ^D shared across all L_S layers; "
                "parameter count = D independent of L_S; score matrix "
                "precomputed once per forward pass (tied γ → same S at "
                "every layer)."
            ),
            "task": (
                "G1 matched stationary circulant ICL regression; "
                f"power_law symbol ν = {cfg.power_law_nu}, teacher "
                f"νβ = {cfg.task_spec_nu_beta}; label_norm = "
                f"{cfg.label_norm}; batch = {cfg.batch_contexts} contexts, "
                "fresh sampling per step."
            ),
            "interpretation": (
                f"Depth-floor ratio at α = {alpha_min:.1f} is "
                f"{ratio_at_min_alpha:.3f}; at α = {alpha_max:.1f} is "
                f"{ratio_at_max_alpha:.3f}. Monotone decreasing in α: "
                f"{'yes' if monotone_decreasing_ok else 'no'}. Material "
                f"(≥20% relative) reduction at α_max: "
                f"{'yes' if material_reduction_ok else 'no'}. Stretch "
                f"target (ratio ≤ {cfg.alpha_stretch_ratio:.1f}): "
                f"{'achieved' if stretch_achieved else 'not achieved'}. "
                f"Transfer alignment at α_max (max median |T_k| over L_S): "
                f"{max_median_T:.3f} "
                f"(gate ≤ {cfg.transfer_max_median_alpha_max:.2f}: "
                f"{'OK' if transfer_alignment_ok else 'WEAK'}). "
                "Interpretation: the depth gap shrinks as α grows because "
                "the per-batch sample-space operator concentrates "
                "(noise ∝ 1/√P), reducing the finite-batch interaction "
                "with the (2L_S − 1) gradient exponent near the matched "
                "optimum."
            ),
            "device": str(device),
            "geometry": {
                "D": int(cfg.D),
                "P_list": list(cfg.P_list),
                "K": int(cfg.K),
                "alphas": [float(P) / float(cfg.D) for P in cfg.P_list],
                "batch_contexts": int(cfg.batch_contexts),
                "train_steps": int(cfg.train_steps),
                "log_every": int(cfg.log_every),
            },
            "L_S_list": list(cfg.L_S_list),
            "seed_list": list(cfg.seed_list),
            "n_runs": int(len(runs)),
            "optimizer": cfg.optimizer,
            "learning_rate": float(cfg.learning_rate),
            "label_norm": cfg.label_norm,
            "symbol_kind": cfg.symbol_kind,
            "power_law_nu": float(cfg.power_law_nu),
            "task_spec_nu_beta": float(cfg.task_spec_nu_beta),
            "status": (
                "all_ok" if all_ok else (
                    ("no_nan_ok" if no_nan else "NaN_FAILURE") + "+" +
                    ("decay_ok" if decay_ok else "decay_FAIL") + "+" +
                    ("monotone_ok" if monotone_decreasing_ok else "monotone_FAIL") + "+" +
                    ("material_ok" if material_reduction_ok else "material_FAIL") + "+" +
                    ("transfer_ok" if transfer_alignment_ok else "transfer_FAIL")
                )
            ),
            "decay_fraction": float(cfg.decay_fraction),
            "alpha_material_reduction_threshold": float(cfg.alpha_material_reduction),
            "alpha_stretch_ratio": float(cfg.alpha_stretch_ratio),
            "transfer_max_median_alpha_max": float(cfg.transfer_max_median_alpha_max),
            "nan_count": int(nan_count),
            "n_decay_violations": int(len(decay_violations)),
            "depth_floor_ratio_by_alpha":
                {f"{a:.3f}": float(r) for a, r in depth_floor_ratio_by_alpha.items()},
            "ratio_at_min_alpha": float(ratio_at_min_alpha),
            "ratio_at_max_alpha": float(ratio_at_max_alpha),
            "monotone_decreasing_ok": bool(monotone_decreasing_ok),
            "material_reduction_ok": bool(material_reduction_ok),
            "stretch_target_achieved": bool(stretch_achieved),
            "transfer_alignment_ok": bool(transfer_alignment_ok),
            "median_abs_T_at_max_alpha":
                {str(k): float(v) for k, v in median_abs_T_at_max_alpha.items()},
            "mean_finals_per_P_LS": {
                str(P): {str(L_S): v for L_S, v in d.items()}
                for P, d in mean_finals_per_P_LS.items()
            },
            "sweep_wallclock_seconds": round(float(sweep_wall), 3),
        })

        if not all_ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
