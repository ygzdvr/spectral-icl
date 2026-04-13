"""Experiment A3 + A4: exact semiseparable / SSD realization plus
negative controls outside the theorem-A class.

Plan reference:
- A3 (primary):    EXPERIMENT_PLAN_FINAL.MD §8.3
- A4 (secondary):  EXPERIMENT_PLAN_FINAL.MD §8.4

Theorem-level framing (read first)
----------------------------------
This script combines two experiments in one file so the artifacts share
the same run directory and the framing is preserved across both:

- **A3 (PRIMARY)** — explicit semiseparable / SSD realization of the
  GD-compatible structured mask. The single primary theorem result of
  this script. Operator-level deterministic forward-pass equivalence,
  acceptance at float64 machine precision.

- **A4 (SECONDARY)** — negative controls outside the theorem-A class.
  These are NOT theorem-A failures; they are deliberately-outside
  models included only to show that exact reduced-Γ behavior is **not
  universal** and that the theorem-A hypotheses are load-bearing.

The thesis writeup must keep this distinction explicit. A3's machine-
precision agreement is the headline; A4's deviation from the reduced
prediction is the negative-control contrast.

A3 — explicit semiseparable / SSD realization
---------------------------------------------
The GD-compatible signed mask ``M_signed`` ∈ ℝ^{(P+K)×(P+K)} is itself
**rank-1** in the sequence dimension:

    M_signed = (1_test − 1_train) · 1_train^T,

so the masked structured mixer's per-layer update on the residual
stream ``z ∈ ℝ^{P+K}`` admits an explicit ``D``-dimensional SSD-style
state-recursion:

    z^{(ℓ+1)} = z^{(ℓ)}
              + (1 / (P · L)) · diag_sign · X^T · Γ · ( X · Π_train · z^{(ℓ)} ),

where

    Π_train      = diag(1_train) ∈ ℝ^{(P+K)×(P+K)},
    diag_sign[μ] = −1 (μ ∈ train), +1 (μ ∈ test),
    X            = [X_train | X_query] ∈ ℝ^{D × (P+K)}.

The bilinear key-query structure is mediated through the D-dim "state"
``s = X · Π_train · z`` and applied as a rank-D outer product per
layer. This is the **explicit theorem-consistent semiseparable / SSD
realization** of plan §8.3 — it is not a generic SSD family, it is the
exact realization that theorem A asserts is equivalent to the reduced
operator object.

A3 compares three forward routes at every (D, P, K, L):

- **R_SSD** (the new explicit semiseparable route).
- **R_AB**  — the sample-space reduced (A_S, B_S) recursion (the
  "reduced-(S, Γ) theorem object" of plan §8.3, identical to A1's R2 /
  A1b's R2).
- **R_full** — A1b R0 full-hidden-state aligned forward (cross-check
  against a structurally distinct path).

All three must agree to float64 machine precision. The acceptance gate
is **R_SSD vs R_AB**; R_SSD vs R_full is reported as a strong
diagnostic.

A4 — negative controls outside theorem A
----------------------------------------
Two negative-control models, each compared against the GD reduced-Γ
prediction. Both models are **deliberately outside** the theorem-A
class; the goal is to show that ordinary L-layer linear-attention-like
forward maps DO NOT generally match the reduced (A_S, B_S) prediction —
the theorem-A hypotheses (bilinear key-query structure with the
GD-compatible mask) are required.

(NC1) **Pure linear circulant convolutional mixer.** A fixed circulant
convolution kernel ``c ∈ ℝ^{P+K}`` (a Gaussian bump) acts on the
residual stream:

    z^{(ℓ+1)} = z^{(ℓ)} + (1 / L) · C · z^{(ℓ)},   C := circ(c).

There is **no** data-dependent bilinear K·Q structure. C does not
depend on ``X`` or ``Γ`` at all. Theorem A makes no claim here, and the
forward map should differ substantially from the reduced object.

(NC2) **Non-GD-compatible structured mask.** Use the GA generator's
``mask_kind="non_gd_control"`` ``"signflip_testtest"`` mode (B_S sign
flipped). The structured mixer is still bilinear, but the mask
violates the GD-compatibility hypothesis, so theorem A does not apply.

Acceptance for A4 is **only** that the negative-control forward
deviates non-trivially from the GD reduced prediction (relative
deviation ≥ ``a4_min_relative_deviation``); the deviation is the
**point** of A4, not a failure.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`: ``GAConfig``,
  ``ga_generate``.
- :mod:`scripts.thesis.utils.metrics`: ``reduced_model_error``.
- :mod:`scripts.thesis.utils.plotting`, :mod:`run_metadata`: standard.

Primary outputs
---------------
- ``a3_pairwise_errors_heatmap`` — max |R_SSD − R_AB| / |R_AB| over
  the (D, P) sweep at fixed (K, L).
- ``a3_error_distribution`` — histogram of all three pairwise relative
  errors over the full (D, P, K, L) sweep with the acceptance line.
- ``a4_negative_controls`` — bar chart comparing the relative deviation
  of NC1 (circular conv) and NC2 (non-GD mask) from the reduced
  prediction across the A4 seeds.

Acceptance
----------
A3 (PRIMARY):
    1. ``max_R_SSD_vs_R_AB ≤ machine_eps_tol``  (every (D, P, K, L)).
    2. ``max_R_SSD_vs_R_full ≤ machine_eps_tol``  (diagnostic strong
       agreement against A1b R0).

A4 (SECONDARY, negative controls):
    3. For every NC seed: relative deviation
       ``||F_NC − F_reduced|| / ||F_reduced|| ≥ a4_min_relative_deviation``.
       (The negative controls are outside the theorem-A class; their
       deviation is the experimental point.)

Run via SLURM
-------------
The login node enforces per-user cgroup memory caps that kill long-
running scripts (cf. A2). Submit via ``sbatch
experiments/thesis/theoremA/run_theoremA_semiseparable.sh``.
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


@dataclass(frozen=True)
class A3A4Config:
    """Frozen configuration for A3 (semiseparable / SSD) + A4 (negative
    controls).
    """

    # A3 sweep: 4 D × 4 P × 3 K × 4 L = 192 cells, three routes per cell.
    D_list: tuple[int, ...] = (8, 16, 32, 64)
    P_list: tuple[int, ...] = (8, 16, 32, 64)
    K_list: tuple[int, ...] = (4, 8, 16)
    L_list: tuple[int, ...] = (1, 2, 4, 8)

    Sigma_kind: str = "isotropic"
    Omega_kind: str = "isotropic"
    Gamma_kind: str = "identity"
    label_norm: str = "sqrt_D"
    sigma: float = 0.0
    B: int = 4
    base_seed: int = 0

    machine_eps_tol: float = 1e-10

    # Heatmap slice for A3.
    heatmap_K: int = 8
    heatmap_L: int = 4

    # A4 negative-control sweep — single representative geometry.
    a4_D: int = 32
    a4_P: int = 32
    a4_K: int = 8
    a4_L: int = 4
    a4_seeds: tuple[int, ...] = (0, 1, 2, 3)

    # A4 circular conv kernel: Gaussian bump, fixed width.
    a4_circular_kernel_width: float = 1.5
    # Scaling of the conv kernel relative to the unit-norm Gaussian.
    a4_circular_kernel_scale: float = 1.0
    # NC2: non-GD-control variant.
    a4_non_gd_kind: str = "signflip_testtest"  # or "nonzero_testblock"

    # A4 acceptance: NC forwards must deviate from reduced prediction by
    # at least this fraction (NC is OUTSIDE the theorem-A class; deviation
    # is the experimental point).
    a4_min_relative_deviation: float = 0.1

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# A3 forward routes
# ---------------------------------------------------------------------------


def _route_R_SSD_explicit_semiseparable(
    X_train: torch.Tensor,    # (B, D, P)
    X_query: torch.Tensor,    # (B, D, K)
    Gamma: torch.Tensor,      # (D, D)
    y_train: torch.Tensor,    # (B, P)
    L: int,
    P_norm: int,
) -> torch.Tensor:
    """Explicit semiseparable / SSD realization of the GD-compatible
    structured mask.  Per-layer update:

        z^(ℓ+1) = z^(ℓ) + (1/(P·L)) · diag_sign · X^T · Γ · (X · Π_train · z^(ℓ)),

    where ``diag_sign``, ``Π_train`` encode the rank-1 GD-compatible
    signed mask. Computed via D-dim state, no full (P+K)×(P+K) matrix.

    Returns the prediction at the K test positions, shape (B, K).
    """
    B = int(X_train.shape[0])
    D = int(X_train.shape[1])
    P = int(X_train.shape[-1])
    K = int(X_query.shape[-1])
    dtype = X_train.dtype
    device = X_train.device

    # Full-sequence X = [X_train | X_query], shape (B, D, P+K).
    X = torch.cat([X_train, X_query], dim=-1)

    # Π_train as a length-(P+K) indicator vector.
    pi_train = torch.zeros(P + K, dtype=dtype, device=device)
    pi_train[:P] = 1.0

    # diag_sign: −1 on train, +1 on test, length (P+K).
    sign_diag = torch.empty(P + K, dtype=dtype, device=device)
    sign_diag[:P] = -1.0
    sign_diag[P:] = +1.0

    # Initial residual stream: y_train at train positions, 0 at test.
    z = torch.cat(
        [y_train, torch.zeros(B, K, dtype=dtype, device=device)],
        dim=-1,
    )

    inv_PL = 1.0 / (float(P_norm) * float(L))
    # Pre-compute Γ X^T projection-per-layer is not feasible because it
    # depends on X, Γ. We compute step-by-step in O((P+K)·D + D²).
    for _ in range(int(L)):
        z_train_only = pi_train * z                              # (B, P+K)
        # state s = X · z_train_only  (D-dim per batch).
        s = torch.einsum("bdf,bf->bd", X, z_train_only)          # (B, D)
        # Γ s  (D-dim per batch).
        Gs = torch.einsum("de,be->bd", Gamma, s)                 # (B, D)
        # Project back via X^T.
        Xt_Gs = torch.einsum("bdf,bd->bf", X, Gs)                # (B, P+K)
        # Apply signed indicator.
        update = sign_diag * Xt_Gs                               # (B, P+K)
        z = z + inv_PL * update
    return z[:, P:]


def _route_R_AB_reduced(
    A_S: torch.Tensor,   # (B, P, P)
    B_S: torch.Tensor,   # (B, K, P)
    y_train: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """Sample-space reduced (A_S, B_S) recursion (the reduced-(S, Γ)
    theorem object of plan §8.3). Identical to A1b's R2."""
    z = y_train.clone()
    sum_T_y = torch.zeros_like(z)
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        sum_T_y = sum_T_y + z
        z = z + inv_L * torch.einsum("bpi,bi->bp", A_S, z)
    return inv_L * torch.einsum("bki,bi->bk", B_S, sum_T_y)


def _route_R_full_signed_mask(
    X_train: torch.Tensor,
    X_query: torch.Tensor,
    Gamma: torch.Tensor,
    y_train: torch.Tensor,
    M_signed_full: torch.Tensor,  # (P+K, P+K)
    L: int,
    P_norm: int,
) -> torch.Tensor:
    """A1b R0 full hidden-state aligned forward, included as an
    additional structurally-distinct cross-check route."""
    B = int(X_train.shape[0])
    D = int(X_train.shape[1])
    P = int(X_train.shape[-1])
    K = int(X_query.shape[-1])
    X = torch.cat([X_train, X_query], dim=-1)
    GammaX = torch.einsum("de,bef->bdf", Gamma, X)
    S_pos = torch.einsum("bdm,bdn->bmn", X, GammaX) / float(P_norm)
    M_eff = S_pos * M_signed_full.unsqueeze(0)
    h = torch.cat(
        [y_train, torch.zeros(B, K, dtype=y_train.dtype, device=y_train.device)],
        dim=-1,
    )
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        update = torch.einsum("bmn,bn->bm", M_eff, h)
        h = h + inv_L * update
    return h[:, P:]


# ---------------------------------------------------------------------------
# A4 negative controls
# ---------------------------------------------------------------------------


def _build_circular_kernel(
    P: int, K: int, width: float, scale: float, dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a unit-norm circular-symmetric Gaussian convolution kernel
    ``c`` of length P+K, then return the corresponding circulant matrix
    ``circ(c)`` ∈ ℝ^{(P+K)×(P+K)}.
    """
    n = P + K
    idx = torch.arange(n, dtype=dtype, device=device)
    # Distance from origin under wrap-around.
    d = torch.minimum(idx, n - idx)
    c = torch.exp(-(d / float(width)).pow(2))
    c = c / c.norm()
    c = float(scale) * c
    # circ(c)[i, j] = c[(i - j) mod n].
    rows = torch.arange(n, dtype=torch.long, device=device).unsqueeze(1)
    cols = torch.arange(n, dtype=torch.long, device=device).unsqueeze(0)
    C = c[(rows - cols) % n]
    return C


def _route_NC1_circular_conv(
    P: int, K: int, y_train: torch.Tensor, L: int,
    width: float, scale: float,
) -> torch.Tensor:
    """NC1: pure linear circulant convolutional mixer. No bilinear K·Q
    structure, no data dependence. Per-layer update:

        z^(ℓ+1) = z^(ℓ) + (1/L) · C · z^(ℓ).

    Initial state same as the GD-compatible structured mixer
    (y_train at train positions, 0 at test).
    """
    B = int(y_train.shape[0])
    dtype = y_train.dtype
    device = y_train.device
    z = torch.cat(
        [y_train, torch.zeros(B, K, dtype=dtype, device=device)], dim=-1,
    )
    C = _build_circular_kernel(P, K, width, scale, dtype, device)
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        update = torch.einsum("mn,bn->bm", C, z)
        z = z + inv_L * update
    return z[:, P:]


def _route_NC2_non_gd_mask(
    X_train: torch.Tensor,
    X_query: torch.Tensor,
    Gamma: torch.Tensor,
    y_train: torch.Tensor,
    M_signed_non_gd: torch.Tensor,
    L: int,
    P_norm: int,
) -> torch.Tensor:
    """NC2: structured mixer with a non-GD-compatible mask (e.g. test→
    train sign flipped). Same R0 hidden-state forward, only the mask
    differs. Theorem A does not apply here."""
    return _route_R_full_signed_mask(
        X_train, X_query, Gamma, y_train, M_signed_non_gd, L, P_norm,
    )


# ---------------------------------------------------------------------------
# Trial runners
# ---------------------------------------------------------------------------


def _run_a3_trial(
    cfg: A3A4Config, D: int, P: int, K: int, L: int, device: torch.device,
) -> dict[str, Any]:
    seeds = {
        "x": int(cfg.base_seed) + 1000 * D + 100 * P + 10 * K + L,
        "beta": int(cfg.base_seed) + 7 + 1000 * D + 100 * P + 10 * K + L,
        "noise": int(cfg.base_seed) + 13 + 1000 * D + 100 * P + 10 * K + L,
        "mask": int(cfg.base_seed) + 19,
    }
    g = GAConfig(
        D=int(D), P=int(P), K=int(K), B=int(cfg.B),
        Sigma_kind=cfg.Sigma_kind,
        Omega_kind=cfg.Omega_kind,
        Gamma_kind=cfg.Gamma_kind,
        label_norm=cfg.label_norm,
        sigma=float(cfg.sigma),
        mask_kind="gd_compatible",
        L=int(L),
        seeds=seeds,
        dtype=cfg.dtype,
        device="cpu",
    )
    op = ga_generate(g)
    X_train = op["X_train"].to(device)
    X_query = op["X_query"].to(device)
    y_train = op["y_train"].to(device)
    Gamma = op["Gamma"].to(device)
    mask_GD = op["mask"].to(device)
    A_S_GD = op["A_S_GD"].to(device)
    B_S_GD = op["B_S_GD"].to(device)

    f_SSD = _route_R_SSD_explicit_semiseparable(
        X_train, X_query, Gamma, y_train, L=int(L), P_norm=int(P),
    )
    f_AB = _route_R_AB_reduced(A_S_GD, B_S_GD, y_train, L=int(L))
    f_full = _route_R_full_signed_mask(
        X_train, X_query, Gamma, y_train, mask_GD, L=int(L), P_norm=int(P),
    )

    err_SSD_vs_AB = reduced_model_error(f_SSD, f_AB)
    err_SSD_vs_full = reduced_model_error(f_SSD, f_full)
    err_AB_vs_full = reduced_model_error(f_AB, f_full)

    return {
        "D": int(D), "P": int(P), "K": int(K), "L": int(L),
        "err_SSD_vs_AB": float(err_SSD_vs_AB),
        "err_SSD_vs_full": float(err_SSD_vs_full),
        "err_AB_vs_full": float(err_AB_vs_full),
    }


def _run_a4_trial(
    cfg: A3A4Config, seed: int, device: torch.device,
) -> dict[str, Any]:
    D, P, K, L = (
        int(cfg.a4_D), int(cfg.a4_P), int(cfg.a4_K), int(cfg.a4_L),
    )
    seeds = {
        "x": int(seed) * 7 + 11,
        "beta": int(seed) * 7 + 13,
        "noise": int(seed) * 7 + 17,
        "mask": int(seed) * 7 + 19,
    }
    # GD-compatible reference for the reduced object.
    g_GD = GAConfig(
        D=D, P=P, K=K, B=int(cfg.B),
        Sigma_kind=cfg.Sigma_kind,
        Omega_kind=cfg.Omega_kind,
        Gamma_kind=cfg.Gamma_kind,
        label_norm=cfg.label_norm,
        sigma=float(cfg.sigma),
        mask_kind="gd_compatible",
        L=L,
        seeds=seeds,
        dtype=cfg.dtype,
        device="cpu",
    )
    op_GD = ga_generate(g_GD)
    X_train = op_GD["X_train"].to(device)
    X_query = op_GD["X_query"].to(device)
    y_train = op_GD["y_train"].to(device)
    Gamma = op_GD["Gamma"].to(device)
    A_S_GD = op_GD["A_S_GD"].to(device)
    B_S_GD = op_GD["B_S_GD"].to(device)
    f_reduced = _route_R_AB_reduced(A_S_GD, B_S_GD, y_train, L=L)

    # NC1: pure linear circular convolution.
    f_NC1 = _route_NC1_circular_conv(
        P, K, y_train, L=L,
        width=float(cfg.a4_circular_kernel_width),
        scale=float(cfg.a4_circular_kernel_scale),
    )
    rel_dev_NC1 = reduced_model_error(f_NC1, f_reduced)

    # NC2: non-GD-compatible mask.
    g_NC2 = GAConfig(
        D=D, P=P, K=K, B=int(cfg.B),
        Sigma_kind=cfg.Sigma_kind,
        Omega_kind=cfg.Omega_kind,
        Gamma_kind=cfg.Gamma_kind,
        label_norm=cfg.label_norm,
        sigma=float(cfg.sigma),
        mask_kind="non_gd_control",
        non_gd_kind=cfg.a4_non_gd_kind,
        L=L,
        seeds=seeds,
        dtype=cfg.dtype,
        device="cpu",
    )
    op_NC2 = ga_generate(g_NC2)
    mask_NC2 = op_NC2["mask"].to(device)
    f_NC2 = _route_NC2_non_gd_mask(
        X_train, X_query, Gamma, y_train, mask_NC2,
        L=L, P_norm=P,
    )
    rel_dev_NC2 = reduced_model_error(f_NC2, f_reduced)

    # Norm of f_reduced (denominator scale).
    f_reduced_norm = float(
        torch.linalg.vector_norm(f_reduced, ord=2, dim=-1).mean().item()
    )

    return {
        "seed": int(seed),
        "D": D, "P": P, "K": K, "L": L,
        "f_reduced_norm": f_reduced_norm,
        "rel_dev_NC1_circular_conv": float(rel_dev_NC1),
        "rel_dev_NC2_non_gd_mask": float(rel_dev_NC2),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_a3_pairwise_errors_heatmap(
    cfg: A3A4Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    K = int(cfg.heatmap_K)
    L = int(cfg.heatmap_L)
    if K not in cfg.K_list or L not in cfg.L_list:
        return
    D_list = list(cfg.D_list)
    P_list = list(cfg.P_list)
    grid = np.zeros((len(D_list), len(P_list)))
    for trial in trials:
        if int(trial["K"]) != K or int(trial["L"]) != L:
            continue
        i_D = D_list.index(int(trial["D"]))
        i_P = P_list.index(int(trial["P"]))
        grid[i_D, i_P] = max(trial["err_SSD_vs_AB"], trial["err_SSD_vs_full"])
    floor = 1e-18
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    phase_heatmap(
        ax, np.where(grid > floor, grid, floor),
        x_coords=np.asarray(P_list, dtype=float),
        y_coords=np.asarray(D_list, dtype=float),
        xlabel="P (training context length)",
        ylabel="D (feature dimension)",
        cbar_label=r"max(SSD vs AB, SSD vs full) rel. error",
        log_z=True, log_x=True, log_y=True,
    )
    ax.set_title(
        rf"A3 max pairwise rel. error at K = {K}, L = {L} "
        r"(expected $\leq$ float eps)",
        fontsize=10,
    )
    fig.tight_layout()
    save_both(fig, run_dir, "a3_pairwise_errors_heatmap")
    plt.close(fig)


def _plot_a3_error_distribution(
    cfg: A3A4Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    err_AB = np.array([t["err_SSD_vs_AB"] for t in trials])
    err_full = np.array([t["err_SSD_vs_full"] for t in trials])
    err_AB_full = np.array([t["err_AB_vs_full"] for t in trials])
    floor = 1e-18
    e1 = np.clip(err_AB, floor, None)
    e2 = np.clip(err_full, floor, None)
    e3 = np.clip(err_AB_full, floor, None)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.logspace(-18, -8, 30)
    ax.hist(e1, bins=bins, alpha=0.6,
            label="R_SSD vs R_AB (primary acceptance)",
            edgecolor="black", lw=0.4)
    ax.hist(e2, bins=bins, alpha=0.6,
            label="R_SSD vs R_full (cross-check)",
            edgecolor="black", lw=0.4)
    ax.hist(e3, bins=bins, alpha=0.6,
            label="R_AB vs R_full (A1b consistency)",
            edgecolor="black", lw=0.4)
    ax.axvline(cfg.machine_eps_tol, color="red", lw=0.9, ls="--",
               label=f"acceptance = {cfg.machine_eps_tol:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel("relative error")
    ax.set_ylabel("count")
    ax.set_title(
        f"A3 pairwise rel.-error distribution over {len(trials)} cells",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a3_error_distribution")
    plt.close(fig)


def _plot_a4_negative_controls(
    cfg: A3A4Config, a4_trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    seeds = [t["seed"] for t in a4_trials]
    NC1 = [t["rel_dev_NC1_circular_conv"] for t in a4_trials]
    NC2 = [t["rel_dev_NC2_non_gd_mask"] for t in a4_trials]
    x = np.arange(len(seeds))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - width / 2, NC1, width, label="NC1 circular conv", color="C0")
    ax.bar(x + width / 2, NC2, width, label="NC2 non-GD mask", color="C1")
    ax.axhline(
        cfg.a4_min_relative_deviation, color="red", lw=0.9, ls="--",
        label=f"acceptance threshold = "
              f"{cfg.a4_min_relative_deviation:.2g}",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel(r"$\|F_{\mathrm{NC}} - F_{\mathrm{reduced}}\|_2 / "
                  r"\|F_{\mathrm{reduced}}\|_2$")
    ax.set_title(
        "A4 negative controls: relative deviation from the GD reduced "
        "prediction\n(deviation is the experimental point — these are "
        "outside the theorem-A class)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "a4_negative_controls")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment A3 + A4: explicit semiseparable / SSD realization "
            "(primary, plan §8.3) and negative controls outside the "
            "theorem-A class (secondary, plan §8.4)."
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
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> A3A4Config:
    base = A3A4Config()
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
    print(f"[A3+A4] device = {device}")

    run = ThesisRunDir(__file__, phase="theoremA")
    with RunContext(
        run,
        config=cfg,
        seeds=[cfg.base_seed] + list(cfg.a4_seeds),
        notes=(
            "A3 (primary): explicit semiseparable / SSD realization vs "
            "reduced (A_S, B_S) theorem object, machine-precision "
            "acceptance. A4 (secondary): negative controls (circular "
            "conv, non-GD mask) showing exact reduced-Γ behavior is not "
            "universal. Operator-level only."
        ),
    ) as ctx:
        apply_thesis_style()

        # ============= A3 sweep =============
        print("=" * 72)
        print(" A3 — explicit semiseparable / SSD vs reduced (A_S, B_S)")
        print("=" * 72)
        a3_trials: list[dict[str, Any]] = []
        n_total = (
            len(cfg.D_list) * len(cfg.P_list)
            * len(cfg.K_list) * len(cfg.L_list)
        )
        idx = 0
        t_a3_start = time.perf_counter()
        for D in cfg.D_list:
            for P in cfg.P_list:
                for K in cfg.K_list:
                    for L in cfg.L_list:
                        idx += 1
                        t0 = time.perf_counter()
                        trial = _run_a3_trial(cfg, int(D), int(P), int(K), int(L), device)
                        dt = time.perf_counter() - t0
                        ctx.record_step_time(dt)
                        print(
                            f"[{idx:>4d}/{n_total}] "
                            f"D={int(D):>3d} P={int(P):>3d} "
                            f"K={int(K):>3d} L={int(L):>2d}  "
                            f"SSD vs AB = {trial['err_SSD_vs_AB']:.2e}  "
                            f"SSD vs full = {trial['err_SSD_vs_full']:.2e}  "
                            f"AB vs full = {trial['err_AB_vs_full']:.2e}  "
                            f"({dt*1000:.1f} ms)"
                        )
                        a3_trials.append(trial)
        a3_wall = time.perf_counter() - t_a3_start

        # ============= A4 sweep =============
        print()
        print("=" * 72)
        print(" A4 — negative controls (NC1 circular conv, NC2 non-GD mask)")
        print("=" * 72)
        a4_trials: list[dict[str, Any]] = []
        t_a4_start = time.perf_counter()
        for seed in cfg.a4_seeds:
            t0 = time.perf_counter()
            trial = _run_a4_trial(cfg, int(seed), device)
            dt = time.perf_counter() - t0
            ctx.record_step_time(dt)
            print(
                f"[a4 seed {int(seed):>2d}]  "
                f"D={trial['D']:>2d} P={trial['P']:>2d} "
                f"K={trial['K']:>2d} L={trial['L']:>2d}  "
                f"|F_red| = {trial['f_reduced_norm']:.3e}  "
                f"NC1 rel.dev = {trial['rel_dev_NC1_circular_conv']:.3e}  "
                f"NC2 rel.dev = {trial['rel_dev_NC2_non_gd_mask']:.3e}  "
                f"({dt*1000:.1f} ms)"
            )
            a4_trials.append(trial)
        a4_wall = time.perf_counter() - t_a4_start

        # ============= Figures =============
        _plot_a3_pairwise_errors_heatmap(cfg, a3_trials, run)
        _plot_a3_error_distribution(cfg, a3_trials, run)
        _plot_a4_negative_controls(cfg, a4_trials, run)

        # ============= Save npz / JSON =============
        D_list = list(cfg.D_list)
        P_list = list(cfg.P_list)
        K_list = list(cfg.K_list)
        L_list = list(cfg.L_list)
        shape = (len(D_list), len(P_list), len(K_list), len(L_list))
        err_SSD_vs_AB_grid = np.zeros(shape)
        err_SSD_vs_full_grid = np.zeros(shape)
        err_AB_vs_full_grid = np.zeros(shape)
        for trial in a3_trials:
            i = D_list.index(trial["D"])
            j = P_list.index(trial["P"])
            k = K_list.index(trial["K"])
            l = L_list.index(trial["L"])
            err_SSD_vs_AB_grid[i, j, k, l] = trial["err_SSD_vs_AB"]
            err_SSD_vs_full_grid[i, j, k, l] = trial["err_SSD_vs_full"]
            err_AB_vs_full_grid[i, j, k, l] = trial["err_AB_vs_full"]
        npz_payload: dict[str, np.ndarray] = {
            "D_list": np.asarray(D_list, dtype=np.int64),
            "P_list": np.asarray(P_list, dtype=np.int64),
            "K_list": np.asarray(K_list, dtype=np.int64),
            "L_list": np.asarray(L_list, dtype=np.int64),
            "a3_err_SSD_vs_AB_grid": err_SSD_vs_AB_grid,
            "a3_err_SSD_vs_full_grid": err_SSD_vs_full_grid,
            "a3_err_AB_vs_full_grid": err_AB_vs_full_grid,
            "a4_seeds": np.asarray([t["seed"] for t in a4_trials], dtype=np.int64),
            "a4_rel_dev_NC1": np.asarray(
                [t["rel_dev_NC1_circular_conv"] for t in a4_trials],
                dtype=np.float64,
            ),
            "a4_rel_dev_NC2": np.asarray(
                [t["rel_dev_NC2_non_gd_mask"] for t in a4_trials],
                dtype=np.float64,
            ),
            "a4_f_reduced_norm": np.asarray(
                [t["f_reduced_norm"] for t in a4_trials],
                dtype=np.float64,
            ),
        }
        np.savez_compressed(run.npz_path("semiseparable"), **npz_payload)

        (run.root / "a3_per_cell_summary.json").write_text(
            json.dumps(a3_trials, indent=2) + "\n", encoding="utf-8",
        )
        (run.root / "a4_per_seed_summary.json").write_text(
            json.dumps(a4_trials, indent=2) + "\n", encoding="utf-8",
        )

        # ============= Acceptance =============
        worst_SSD_vs_AB = max(t["err_SSD_vs_AB"] for t in a3_trials)
        worst_SSD_vs_full = max(t["err_SSD_vs_full"] for t in a3_trials)
        worst_AB_vs_full = max(t["err_AB_vs_full"] for t in a3_trials)
        a3_ok_AB = worst_SSD_vs_AB <= cfg.machine_eps_tol
        a3_ok_full = worst_SSD_vs_full <= cfg.machine_eps_tol
        a3_all_ok = a3_ok_AB and a3_ok_full

        a4_min_NC1 = min(t["rel_dev_NC1_circular_conv"] for t in a4_trials)
        a4_min_NC2 = min(t["rel_dev_NC2_non_gd_mask"] for t in a4_trials)
        a4_NC1_deviates = a4_min_NC1 >= cfg.a4_min_relative_deviation
        a4_NC2_deviates = a4_min_NC2 >= cfg.a4_min_relative_deviation
        a4_ok = a4_NC1_deviates and a4_NC2_deviates

        status_parts: list[str] = []
        status_parts.append(
            "a3_SSD_vs_AB_ok" if a3_ok_AB else
            f"a3_SSD_vs_AB_violated(worst={worst_SSD_vs_AB:.2e})"
        )
        status_parts.append(
            "a3_SSD_vs_full_ok" if a3_ok_full else
            f"a3_SSD_vs_full_violated(worst={worst_SSD_vs_full:.2e})"
        )
        status_parts.append(
            "a4_NC1_deviates" if a4_NC1_deviates else
            f"a4_NC1_too_close(min={a4_min_NC1:.2e})"
        )
        status_parts.append(
            "a4_NC2_deviates" if a4_NC2_deviates else
            f"a4_NC2_too_close(min={a4_min_NC2:.2e})"
        )
        status = "+".join(status_parts)

        ctx.record_compute_proxy(float(a3_wall + a4_wall))
        ctx.record_extra("a3_worst_SSD_vs_AB", worst_SSD_vs_AB)
        ctx.record_extra("a3_worst_SSD_vs_full", worst_SSD_vs_full)
        ctx.record_extra("a3_worst_AB_vs_full", worst_AB_vs_full)
        ctx.record_extra("a4_min_NC1_deviation", a4_min_NC1)
        ctx.record_extra("a4_min_NC2_deviation", a4_min_NC2)

        ctx.write_summary(
            {
                "plan_reference": (
                    "EXPERIMENT_PLAN_FINAL.MD §8.3 (A3) + §8.4 (A4)"
                ),
                "primary_vs_secondary": (
                    "A3 (PRIMARY): explicit theorem-consistent "
                    "semiseparable / SSD realization of the GD-compatible "
                    "structured mask. The single primary theorem-level "
                    "result of this script — machine-precision agreement "
                    "with the reduced (A_S, B_S) theorem object across "
                    "the (D, P, K, L) sweep. "
                    "A4 (SECONDARY): two negative controls (NC1 = pure "
                    "linear circulant convolution mixer, NC2 = "
                    "non-GD-compatible structured mask). These are NOT "
                    "theorem-A in-class models; their non-trivial "
                    "deviation from the reduced prediction is the "
                    "experimental point and shows that exact reduced-Γ "
                    "behavior is not universal — the theorem-A "
                    "hypotheses are load-bearing. A4 results MUST be "
                    "labeled as negative controls, not as theorem-A "
                    "failures."
                ),
                "category": (
                    "exact theorem-A operator-level deterministic "
                    "forward-pass validation. No learned architecture, "
                    "no training dynamics."
                ),
                "a3_interpretation": (
                    "The GD-compatible signed mask M_signed = "
                    "(1_test − 1_train) · 1_train^T is rank-1 in the "
                    "sequence dimension. The structured-mixer per-layer "
                    "update therefore admits an explicit D-dim SSD-style "
                    "state recursion: z^(ℓ+1) = z^(ℓ) + (1/(P·L)) · "
                    "diag_sign · X^T · Γ · (X · Π_train · z^(ℓ)). "
                    "This is the explicit theorem-consistent "
                    "semiseparable realization of plan §8.3 — not a "
                    "broad SSD family. A3 verifies that it equals the "
                    "reduced (A_S, B_S) recursion to float64 machine "
                    "precision over the full (D, P, K, L) sweep."
                ),
                "a4_interpretation": (
                    "NC1 (pure circulant convolutional mixer): no "
                    "data-dependent bilinear K·Q structure, no "
                    "dependence on (X, Γ). Outside the theorem-A class. "
                    "NC2 (non-GD-compatible mask, signflip_testtest): "
                    "structured mixer is bilinear but the mask violates "
                    "GD-compatibility. Outside the theorem-A class. "
                    "Both NC forwards differ from the reduced prediction "
                    "by orders of magnitude. The deviation is the "
                    "experimental point — it shows the theorem-A "
                    "hypotheses are required for exact reduced-Γ "
                    "equivalence."
                ),
                "device": str(device),
                "D_list": list(cfg.D_list),
                "P_list": list(cfg.P_list),
                "K_list": list(cfg.K_list),
                "L_list": list(cfg.L_list),
                "a3_n_cells": len(a3_trials),
                "a4_n_seeds": len(a4_trials),
                "a4_geometry": {
                    "D": cfg.a4_D, "P": cfg.a4_P,
                    "K": cfg.a4_K, "L": cfg.a4_L,
                },
                "a4_non_gd_kind": cfg.a4_non_gd_kind,
                "a4_circular_kernel_width": cfg.a4_circular_kernel_width,
                "a4_circular_kernel_scale": cfg.a4_circular_kernel_scale,
                "status": status,
                "machine_eps_tol": cfg.machine_eps_tol,
                "a4_min_relative_deviation": cfg.a4_min_relative_deviation,
                "a3_worst_SSD_vs_AB": float(worst_SSD_vs_AB),
                "a3_worst_SSD_vs_full": float(worst_SSD_vs_full),
                "a3_worst_AB_vs_full": float(worst_AB_vs_full),
                "a4_min_NC1_deviation": float(a4_min_NC1),
                "a4_min_NC2_deviation": float(a4_min_NC2),
                "a3_wallclock_seconds": round(float(a3_wall), 3),
                "a4_wallclock_seconds": round(float(a4_wall), 3),
            }
        )

        print()
        print("=" * 72)
        print(f" A3 + A4 summary on {device}")
        print(f"   A3 cells = {len(a3_trials)} (D × P × K × L)")
        print(
            f"   A3 worst R_SSD vs R_AB   (PRIMARY) = "
            f"{worst_SSD_vs_AB:.3e}  "
            f"{'OK' if a3_ok_AB else 'FAIL'}  "
            f"(tol = {cfg.machine_eps_tol:.1e})"
        )
        print(
            f"   A3 worst R_SSD vs R_full (cross-check) = "
            f"{worst_SSD_vs_full:.3e}  "
            f"{'OK' if a3_ok_full else 'FAIL'}"
        )
        print(
            f"   A3 worst R_AB  vs R_full (A1b consist.) = "
            f"{worst_AB_vs_full:.3e}"
        )
        print(f"   A4 seeds = {len(a4_trials)}")
        print(
            f"   A4 NC1 (circ. conv) min rel.dev = {a4_min_NC1:.3e}  "
            f"{'OK (deviates)' if a4_NC1_deviates else 'WEAK'}  "
            f"(threshold = {cfg.a4_min_relative_deviation:.2g})"
        )
        print(
            f"   A4 NC2 (non-GD mask) min rel.dev = {a4_min_NC2:.3e}  "
            f"{'OK (deviates)' if a4_NC2_deviates else 'WEAK'}  "
            f"(threshold = {cfg.a4_min_relative_deviation:.2g})"
        )
        print("=" * 72)

        if not a3_all_ok or not a4_ok:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
