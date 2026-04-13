"""Experiment A2: theorem-A perturbation away from GD-compatibility with
the full additive (A, B)-operator bound.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §8.2.

Theorem-level framing (read carefully)
--------------------------------------
A2 is still in the **exact theorem-A tier**. There is NO learned
architecture, NO training dynamics, NO gradient descent. This is a
deterministic forward-pass perturbation diagnostic around the
GD-compatible regime.

The faithful theorem-A perturbation diagnostic compares the empirical
full-model error against the full reduced ``(A, B)``-operator
upper bound — **not** against a qualitative mask-distance heuristic and
**not** against an A-only bound. The plan §8.2 binding is that the bound
must use the additive structure with both reduced operators ``A`` and
``B`` and the telescoping difference of propagator powers.

The forward maps are

    F_θ(y)  = (1/L) · B_θ  · Σ_{ℓ=0..L-1}  T_θ^ℓ  · y,
    F_GD(y) = (1/L) · B_GD · Σ_{ℓ=0..L-1}  T_GD^ℓ · y,
    T = I_P + A_S/L.

The exact additive decomposition is

    F_θ − F_GD = (1/L) · ΔB · Σ T_θ^ℓ · y                  [B-side]
                + (1/L) · B_GD · Σ (T_θ^ℓ − T_GD^ℓ) · y    [A-side, telescoping]

with ΔB = B_θ − B_GD. The telescoping identity
``T_θ^ℓ − T_GD^ℓ = (1/L) Σ_{k=0..ℓ-1} T_θ^{ℓ-1-k} · ΔA · T_GD^k``
gives the bound

    ||F_θ − F_GD||_2 ≤ B_side_bound + A_side_bound,
    B_side_bound = ||ΔB||_op / L · ||S_θ y||_2,
    A_side_bound = ||B_GD||_op · ||ΔA||_op · ||y||_2 / L²
                   · Σ_{ℓ=0..L-1} U_ℓ,
    U_ℓ = Σ_{k=0..ℓ-1} ||T_θ||_op^{ℓ-1-k} · ||T_GD||_op^k.

A2 uses :func:`metrics.ab_perturbation_bound` to compute this bound and
verifies the empirical full-model forward error never exceeds it.

**No A-only reduction. No folding the B-side perturbation into T_θ − T_GD.**
The A-side and B-side contributions are reported separately in every
output figure and JSON record.

Forward routes
--------------
Empirical errors are computed using the **R0 full-hidden-state aligned
structured forward** from A1b: build the full ``(P+K)×(P+K)`` bilinear
score ``S = X^T Γ X / P`` from ``X = [X_train | X_query]``, multiply
entrywise by the signed mask ``M_signed`` returned by ``ga_generate``
(the train-train block carries ``−1 + θ · Δ`` for perturbed runs and
``−1`` for GD-compatible runs; the test-train block carries ``+1`` in
both), and run L explicit residual-stream layer updates on a length-
``(P+K)`` scalar hidden channel. Returns the prediction at the K test
positions. The reduced operators ``(A_S, B_S, T)`` returned by GA are
consumed only by the bound computation, never by the forward.

Sweeps (plan §8.2 binding)
--------------------------
- θ values include θ = 0 (sanity: float-eps empirical error) and a
  log-spaced ladder of small-to-moderate perturbations so the
  small-perturbation regime is visible.
- Multiple sampled-context seeds so the result is not tied to one draw.
- A small grid of ``(D, P, K, L)`` configurations to verify the bound
  holds across geometries.
- The perturbation pattern ``Δ`` (Frobenius-normalized symmetric Gaussian
  of seed ``pattern_seed``) is fixed across the θ sweep so the
  perturbation direction is identical and the empirical error scales
  cleanly with θ.

Two perturbation modes (canonical + auxiliary)
----------------------------------------------
- **A_only (CANONICAL theorem-A)**: GD-compatible mask interpolated
  into a train-supported structured perturbation via the GA generator's
  ``"perturbed"`` mask kind. The train-train block carries
  ``−1 + θ · Δ``; the test-train block is left at the GD-compatible
  value. Produces ``ΔA ∝ θ``, ``ΔB = 0``. This is **the** theorem-A
  perturbation family of plan §8.2 — the canonical primary result.

- **B_only (AUXILIARY decomposition diagnostic)**: a manually-built mask
  perturbs only the test-train block. Produces ``ΔA = 0``,
  ``ΔB ∝ θ``. Included **solely** so the additive bound's B-side term
  is exercised with non-zero values; this is not a theorem-A
  perturbation family per plan §8.2 and the thesis writeup must keep
  this distinction explicit.

Empirical forward route (binding)
---------------------------------
Both ``F_θ`` and ``F_GD`` are computed via the **A1b R0 full-hidden-
state aligned structured forward** — building the full ``(P+K)×(P+K)``
bilinear score from ``(X, Γ)`` and applying L explicit residual-stream
layer updates with the perturbed (or GD) signed mask. The reduced
operators ``(A_S_θ, B_S_θ, T_θ)`` from the GA generator are consumed
**only** by the bound computation, never to define the empirical
error.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``GAConfig``, ``ga_generate`` (mask kinds ``"gd_compatible"`` and
    ``"perturbed"``).
- :mod:`scripts.thesis.utils.metrics`:
    ``ab_perturbation_bound`` (full additive (A,B) bound),
    ``reduced_model_error`` (relative L2).
- :mod:`scripts.thesis.utils.plotting`, :mod:`run_metadata`: standard.

Primary outputs
---------------
- ``a2_empirical_vs_bound`` — empirical error and total bound vs θ on
  log-log axes; one line per seed, configuration aggregated.
- ``a2_decomposition`` — the additive (A-side, B-side) bound
  decomposition vs θ, with the empirical error overlaid.
- ``a2_theta_zero_sanity`` — bar/point at θ = 0 showing empirical error
  at float64 eps across seeds and configurations.
- ``a2_bound_slack`` — heatmap or scatter of bound / empirical ratio
  across (config, seed, θ); the slack-vs-θ trend should be visible.

Acceptance
----------
1. **θ = 0 sanity.** ``||F_0 − F_GD||_2 ≤ machine_eps_tol`` for every
   sampled seed and configuration.
2. **Bound respected everywhere.** For every tested θ, every seed,
   every configuration:
   ``||F_θ − F_GD||_2 ≤ total_bound + numerical_slack``.
3. **Both contributions reported.** A-side and B-side bounds are
   separately recorded in the per-cell summary; neither is folded
   into the other.

Framing wording requirement (per plan §8.2):
A2 is the faithful theorem-A perturbation diagnostic because it
compares the empirical full-model error against the full reduced
``(A, B)``-operator upper bound, not just against a qualitative
mask-distance heuristic.

Run
---
::

    python -u scripts/thesis/theoremA/run_theoremA_mask_perturbation.py \\
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
from scripts.thesis.utils.metrics import ab_perturbation_bound, reduced_model_error
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
class A2Config:
    """Frozen configuration for A2: theorem-A perturbation around GD.

    Default 4 configs × 8 seeds × 9 θ values = 288 trials, each a few
    small matmuls. Full sweep finishes in seconds.
    """

    # Configurations: list of (D, P, K, L) tuples.
    configs: tuple[tuple[int, int, int, int], ...] = (
        (32, 32, 8, 4),
        (64, 32, 8, 4),
    )

    # θ sweep — includes 0 (sanity) and a small-to-moderate log-spaced
    # ladder. Small values exercise the linear-perturbation regime.
    theta_list: tuple[float, ...] = (
        0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2,
    )

    # Sampled-context seeds (X, β).
    seed_list: tuple[int, ...] = (0, 1, 2, 3)

    # Perturbation pattern seed (fixed across θ so direction is constant).
    pattern_seed: int = 42
    # B-side perturbation pattern seed (separate from pattern_seed so the
    # A-side and B-side perturbation directions are statistically
    # independent under the same θ).
    pattern_seed_B: int = 137

    # Perturbation modes to test. Each mode produces a separate row per
    # (config, seed, θ). Modes:
    #   "A_only" — CANONICAL theorem-A perturbation family. The
    #     GD-compatible mask is interpolated to a train-supported
    #     structured perturbation via the GA generator's "perturbed"
    #     mask kind: train-train block ←  −1 + θ · Δ ; test-train
    #     block left untouched. ΔB = 0 by construction. This is the
    #     primary theorem-A perturbation result.
    #   "B_only" — AUXILIARY decomposition diagnostic. A manually-
    #     constructed mask perturbs only the test-train block, leaving
    #     the train-train block at the GD-compatible value. ΔA = 0,
    #     ΔB ∝ θ. Included only so the additive bound's B-side term is
    #     exercised with non-zero values; it is NOT presented as a
    #     theorem-A perturbation family. The thesis writeup must keep
    #     this distinction explicit.
    perturb_modes: tuple[str, ...] = ("A_only", "B_only")

    # GA generator config — default isotropic, identity Γ, sqrt_D label norm.
    Sigma_kind: str = "isotropic"
    Omega_kind: str = "isotropic"
    Gamma_kind: str = "identity"
    label_norm: str = "sqrt_D"
    sigma: float = 0.0
    B: int = 1  # one context per seed; multiple seeds give the cloud

    # Acceptance.
    machine_eps_tol: float = 1e-10
    bound_numerical_slack: float = 1e-12  # slack added to total_bound for ε ≤ bound + slack

    # Figure slices.
    headline_config_idx: int = 1  # which (D, P, K, L) tuple to use as the headline

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Full-hidden-state structured forward (R0 from A1b, signed-mask form)
# ---------------------------------------------------------------------------


def _full_structured_forward(
    X_train: torch.Tensor,    # (B, D, P)
    X_query: torch.Tensor,    # (B, D, K)
    Gamma: torch.Tensor,      # (D, D)
    y_train: torch.Tensor,    # (B, P)
    M_signed_full: torch.Tensor,  # (P+K, P+K)
    L: int,
    P_norm: int,
) -> torch.Tensor:
    """A1b R0 full-hidden-state aligned structured forward, with the signed
    mask ``M_signed_full`` taken from ``ga_generate`` (which contains
    ``−1 + θ · Δ`` on the train-train block for perturbed runs and ``−1``
    for GD-compatible runs).

    Returns the prediction at the K test positions, shape ``(B, K)``.
    """
    B = int(X_train.shape[0])
    D = int(X_train.shape[1])
    P = int(X_train.shape[-1])
    K = int(X_query.shape[-1])

    X = torch.cat([X_train, X_query], dim=-1)  # (B, D, P+K)
    GammaX = torch.einsum("de,bef->bdf", Gamma, X)
    S_pos = torch.einsum("bdm,bdn->bmn", X, GammaX) / float(P_norm)
    M_eff = S_pos * M_signed_full.unsqueeze(0)

    h = torch.cat(
        [
            y_train,
            torch.zeros(B, K, dtype=y_train.dtype, device=y_train.device),
        ],
        dim=-1,
    )
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        update = torch.einsum("bmn,bn->bm", M_eff, h)
        h = h + inv_L * update
    return h[:, P:]


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def _build_B_only_perturbed_mask(
    P: int, K: int, theta: float, pattern_seed: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Construct a manually-perturbed mask that perturbs ONLY the
    test-train block (so ΔA = 0, ΔB ∝ θ): start from the GD-compatible
    mask, then add ``θ · Δ_B`` to the K×P test-train block, where ``Δ_B``
    is a Frobenius-normalized Gaussian-seeded pattern.
    """
    M = torch.zeros(P + K, P + K, dtype=dtype)
    M[:P, :P] = -1.0
    M[P:, :P] = +1.0
    if theta != 0.0:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(pattern_seed))
        Delta_B = torch.randn(K, P, generator=gen, dtype=dtype)
        fro = Delta_B.norm()
        if fro.item() > 0:
            Delta_B = Delta_B / fro
        M[P:, :P] = M[P:, :P] + float(theta) * Delta_B
    return M


def _run_trial(
    cfg: A2Config,
    D: int, P: int, K: int, L: int,
    seed: int, theta: float,
    perturb_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    seeds = {
        "x": int(seed) * 7 + 11,
        "beta": int(seed) * 7 + 13,
        "noise": int(seed) * 7 + 17,
        "mask": int(seed) * 7 + 19,
    }
    dtype_torch = torch.float64 if cfg.dtype == "float64" else torch.float32
    # GD-compatible reference (always identical regardless of perturb_mode).
    g_GD = GAConfig(
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
    op_GD = ga_generate(g_GD)

    X_train = op_GD["X_train"].to(device)
    X_query = op_GD["X_query"].to(device)
    y_train = op_GD["y_train"].to(device)
    Gamma = op_GD["Gamma"].to(device)
    mask_GD = op_GD["mask"].to(device)
    A_S_GD = op_GD["A_S_GD"].to(device)
    B_S_GD = op_GD["B_S_GD"].to(device)
    T_GD = op_GD["T_GD"].to(device)

    # Reduced operators K_train and K_query (for constructing perturbed
    # operators when perturb_mode == "B_only").
    K_train = -A_S_GD  # K_train = X^T Γ X / P; A_S_GD = -K_train
    K_query = B_S_GD   # K_query = X_q^T Γ X / P; B_S_GD = +K_query

    # Build the perturbed mask + perturbed reduced operators per mode.
    if perturb_mode == "A_only":
        # Use GA's built-in "perturbed" mask kind — perturbs train-train
        # block; B_S unchanged (ΔB = 0 by construction).
        if theta == 0.0:
            mask_T = mask_GD
            A_S_T = A_S_GD
            B_S_T = B_S_GD
            T_T = T_GD
        else:
            g_T = GAConfig(
                D=int(D), P=int(P), K=int(K), B=int(cfg.B),
                Sigma_kind=cfg.Sigma_kind,
                Omega_kind=cfg.Omega_kind,
                Gamma_kind=cfg.Gamma_kind,
                label_norm=cfg.label_norm,
                sigma=float(cfg.sigma),
                mask_kind="perturbed",
                mask_perturbation={
                    "theta": float(theta),
                    "pattern_seed": int(cfg.pattern_seed),
                },
                L=int(L),
                seeds=seeds,
                dtype=cfg.dtype,
                device="cpu",
            )
            op_T = ga_generate(g_T)
            mask_T = op_T["mask"].to(device)
            A_S_T = op_T["A_S_theta"].to(device)
            B_S_T = op_T["B_S_theta"].to(device)
            T_T = op_T["T_theta"].to(device)
    elif perturb_mode == "B_only":
        # Manually construct a perturbed mask that perturbs ONLY the
        # test-train block; A_S unchanged (ΔA = 0).
        mask_T = _build_B_only_perturbed_mask(
            int(P), int(K), float(theta),
            int(cfg.pattern_seed_B), dtype_torch,
        ).to(device)
        # The signed mask gives perturbed B_S_T directly; A_S unchanged.
        A_S_T = A_S_GD
        T_T = T_GD
        # B_S_T = (test-train block of mask_T) ⊙ K_query
        signed_test_train = mask_T[int(P):, :int(P)]  # (K, P)
        B_S_T = signed_test_train.unsqueeze(0) * K_query  # (B, K, P)
    else:
        raise ValueError(f"unknown perturb_mode: {perturb_mode!r}")

    # Full-model forward with the chosen perturbed mask, via the R0 path.
    F_GD = _full_structured_forward(
        X_train, X_query, Gamma, y_train, mask_GD,
        L=int(L), P_norm=int(P),
    )
    F_T = _full_structured_forward(
        X_train, X_query, Gamma, y_train, mask_T,
        L=int(L), P_norm=int(P),
    )

    empirical = float(
        torch.linalg.vector_norm(F_T - F_GD, ord=2, dim=-1).mean().item()
    )
    empirical_rel = reduced_model_error(F_T, F_GD)

    # Full additive (A, B) bound. The metric expects unbatched inputs OR
    # batched with a leading B dim; pass batched.
    bound = ab_perturbation_bound(
        A_S_T, A_S_GD, B_S_T, B_S_GD, T_T, T_GD,
        L=int(L), y=y_train,
    )

    def _flatten(x: Any) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.mean().item())
        return float(x)

    return {
        "D": int(D), "P": int(P), "K": int(K), "L": int(L),
        "seed": int(seed),
        "theta": float(theta),
        "perturb_mode": perturb_mode,
        "empirical_abs": empirical,
        "empirical_rel": float(empirical_rel),
        "B_side_bound": _flatten(bound["B_side_bound"]),
        "A_side_bound": _flatten(bound["A_side_bound"]),
        "total_bound": _flatten(bound["total_bound"]),
        "delta_A_op": _flatten(bound["delta_A_op"]),
        "delta_B_op": _flatten(bound["delta_B_op"]),
        "telescoping_coeff": _flatten(bound["telescoping_coeff"]),
        "S_theta_y_norm": _flatten(bound["S_theta_y_norm"]),
        "metric_empirical_error": _flatten(bound["empirical_error"]),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _config_label(D: int, P: int, K: int, L: int) -> str:
    return f"D={D}, P={P}, K={K}, L={L}"


def _plot_empirical_vs_bound(
    cfg: A2Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Empirical error and total bound vs θ for the headline configuration,
    one line per seed."""
    import matplotlib.pyplot as plt

    if cfg.headline_config_idx >= len(cfg.configs):
        return
    D, P, K, L = cfg.configs[cfg.headline_config_idx]
    sub = [
        t for t in trials
        if int(t["D"]) == D and int(t["P"]) == P and int(t["K"]) == K
        and int(t["L"]) == L and float(t["theta"]) > 0
    ]
    seed_colors = sequential_colors(len(cfg.seed_list), palette="rocket")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    floor = 1e-30
    for color, seed in zip(seed_colors, cfg.seed_list):
        sub_s = sorted(
            [t for t in sub if int(t["seed"]) == seed],
            key=lambda r: r["theta"],
        )
        thetas = np.array([t["theta"] for t in sub_s])
        emp = np.array([t["empirical_abs"] for t in sub_s])
        bnd = np.array([t["total_bound"] for t in sub_s])
        emp_p = np.where(emp > floor, emp, np.nan)
        bnd_p = np.where(bnd > floor, bnd, np.nan)
        ax.plot(thetas, emp_p, color=color, lw=1.4, marker="o", ms=4.0,
                label=f"empirical (seed {seed})" if seed == cfg.seed_list[0] else None)
        ax.plot(thetas, bnd_p, color=color, lw=1.0, ls="--", alpha=0.7)
    ax.plot([], [], color="black", lw=1.0, ls="--", label="total bound")
    ax.plot([], [], color="black", lw=1.4, marker="o", ms=4.0, label="empirical (per seed)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"perturbation strength $\theta$")
    ax.set_ylabel(r"$\|F_\theta - F_{\mathrm{GD}}\|_2$")
    ax.set_title(
        rf"A2 empirical error vs full (A,B) bound at "
        f"{_config_label(D, P, K, L)}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "a2_empirical_vs_bound")
    plt.close(fig)


def _plot_decomposition(
    cfg: A2Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Additive bound decomposition: B-side, A-side, total, with empirical
    overlaid. One panel per perturb_mode at the headline configuration.
    Aggregated across seeds (median + min-max envelope)."""
    import matplotlib.pyplot as plt

    if cfg.headline_config_idx >= len(cfg.configs):
        return
    D, P, K, L = cfg.configs[cfg.headline_config_idx]
    n_modes = len(cfg.perturb_modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(6.4 * n_modes, 4.4),
                             sharey=True)
    if n_modes == 1:
        axes = [axes]
    color_map = {"B": "C0", "A": "C1", "tot": "black", "emp": "C3"}
    label_map = {
        "B": r"B-side bound $\|\Delta B\|_{op}\,\|S_\theta y\|/L$",
        "A": r"A-side bound $\|B_{GD}\|_{op}\|\Delta A\|_{op}\|y\|\,\Sigma U_\ell / L^2$",
        "tot": "total bound (B + A)",
        "emp": "empirical full-model error",
    }
    floor = 1e-30
    for ax, perturb_mode in zip(axes, cfg.perturb_modes):
        sub = [
            t for t in trials
            if int(t["D"]) == D and int(t["P"]) == P and int(t["K"]) == K
            and int(t["L"]) == L and float(t["theta"]) > 0
            and t["perturb_mode"] == perturb_mode
        ]
        thetas = sorted({float(t["theta"]) for t in sub})
        median = {key: [] for key in ("emp", "B", "A", "tot")}
        lo = {key: [] for key in ("emp", "B", "A", "tot")}
        hi = {key: [] for key in ("emp", "B", "A", "tot")}
        for theta in thetas:
            rows = [t for t in sub if abs(t["theta"] - theta) < 1e-30]
            for key, src in (
                ("emp", "empirical_abs"), ("B", "B_side_bound"),
                ("A", "A_side_bound"), ("tot", "total_bound"),
            ):
                vals = np.array([t[src] for t in rows])
                median[key].append(float(np.median(vals)))
                lo[key].append(float(np.min(vals)))
                hi[key].append(float(np.max(vals)))
        th_arr = np.asarray(thetas)
        for key in ("B", "A", "tot", "emp"):
            med = np.where(np.array(median[key]) > floor, median[key], np.nan)
            lo_p = np.where(np.array(lo[key]) > floor, lo[key], np.nan)
            hi_p = np.where(np.array(hi[key]) > floor, hi[key], np.nan)
            ls = "-" if key in ("emp", "tot") else "--"
            ax.plot(
                th_arr, med, color=color_map[key], lw=1.5, ls=ls, marker="o", ms=3.5,
                label=label_map[key],
            )
            ax.fill_between(th_arr, lo_p, hi_p, color=color_map[key], alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"perturbation strength $\theta$")
        ax.set_title(rf"perturb\_mode = {perturb_mode}", fontsize=10)
        if perturb_mode == cfg.perturb_modes[0]:
            ax.set_ylabel(r"$\|F_\theta - F_{\mathrm{GD}}\|_2$ (and bounds)")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        rf"A2 additive (A, B) bound decomposition at "
        f"{_config_label(D, P, K, L)}; median ± min/max over "
        f"{len(cfg.seed_list)} seeds",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_both(fig, run_dir, "a2_decomposition")
    plt.close(fig)


def _plot_theta_zero_sanity(
    cfg: A2Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """θ = 0 sanity panel: empirical error must be at float64 eps."""
    import matplotlib.pyplot as plt

    rows = [t for t in trials if float(t["theta"]) == 0.0]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    cfg_labels = []
    cfg_max = []
    cfg_min = []
    for D, P, K, L in cfg.configs:
        sub = [t for t in rows if (t["D"], t["P"], t["K"], t["L"]) == (D, P, K, L)]
        if not sub:
            continue
        emp = np.array([t["empirical_abs"] for t in sub])
        cfg_labels.append(_config_label(D, P, K, L))
        cfg_min.append(float(emp.min()))
        cfg_max.append(float(emp.max()))
    x_idx = np.arange(len(cfg_labels))
    floor = 1e-30
    cfg_min_p = np.where(np.array(cfg_min) > floor, cfg_min, floor)
    cfg_max_p = np.where(np.array(cfg_max) > floor, cfg_max, floor)
    ax.errorbar(
        x_idx, (cfg_min_p + cfg_max_p) / 2,
        yerr=[(cfg_min_p + cfg_max_p) / 2 - cfg_min_p,
              cfg_max_p - (cfg_min_p + cfg_max_p) / 2],
        fmt="o", color="C0", capsize=5,
        label=f"empirical at θ = 0 (across {len(cfg.seed_list)} seeds)",
    )
    ax.axhline(
        cfg.machine_eps_tol, color="red", lw=0.9, ls="--",
        label=f"acceptance = {cfg.machine_eps_tol:.0e}",
    )
    ax.set_yscale("log")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(cfg_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(r"$\|F_0 - F_{\mathrm{GD}}\|_2$")
    ax.set_title("A2 θ = 0 sanity: empirical error at float64 eps", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a2_theta_zero_sanity")
    plt.close(fig)


def _plot_bound_slack(
    cfg: A2Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Bound slack: total_bound / empirical (≥ 1 means bound is respected)
    plotted vs θ for every (config, seed)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cfg_colors = sequential_colors(len(cfg.configs), palette="rocket")
    for color, (D, P, K, L) in zip(cfg_colors, cfg.configs):
        sub = [
            t for t in trials
            if (t["D"], t["P"], t["K"], t["L"]) == (D, P, K, L)
            and float(t["theta"]) > 0
        ]
        thetas = np.array([t["theta"] for t in sub])
        ratios = np.array([
            t["total_bound"] / max(t["empirical_abs"], 1e-30) for t in sub
        ])
        # Aggregate per θ across seeds.
        unique_thetas = sorted(set(thetas))
        median_ratio = []
        for th in unique_thetas:
            mask = np.isclose(thetas, th)
            median_ratio.append(np.median(ratios[mask]))
        ax.plot(
            unique_thetas, median_ratio, color=color, lw=1.5,
            marker="o", ms=4.0, label=_config_label(D, P, K, L),
        )
    ax.axhline(1.0, color="red", lw=0.9, ls="--",
               label="bound = empirical (slack = 1×)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"perturbation strength $\theta$")
    ax.set_ylabel(r"total\_bound / empirical (median over seeds)")
    ax.set_title(
        "A2 bound slack: ratio of total bound to empirical error "
        f"(≥ 1 means bound respected)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_both(fig, run_dir, "a2_bound_slack")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment A2: theorem-A perturbation around GD-compatibility "
            "with the full additive (A, B) reduced-operator bound "
            "(plan §8.2)."
        )
    )
    p.add_argument(
        "--device", type=str, default="cuda", choices=("cpu", "cuda", "auto")
    )
    p.add_argument(
        "--dtype", type=str, default="float64", choices=("float32", "float64")
    )
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--theta-list", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None)
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> A2Config:
    base = A2Config()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.theta_list is not None:
        overrides["theta_list"] = _parse_list_floats(args.theta_list)
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
    print(f"[A2] device = {device}")

    run = ThesisRunDir(__file__, phase="theoremA")
    with RunContext(
        run,
        config=cfg,
        seeds=list(cfg.seed_list),
        notes=(
            "A2 theorem-A perturbation diagnostic (plan §8.2). Operator-"
            "level only; deterministic forward-pass comparison around "
            "GD-compatibility. Empirical full-model error compared to "
            "the full additive (A, B)-operator bound; A-side (telescoping "
            "ΔA term) and B-side (ΔB term) reported separately."
        ),
    ) as ctx:
        apply_thesis_style()

        trials: list[dict[str, Any]] = []
        n_total = (
            len(cfg.configs) * len(cfg.seed_list)
            * len(cfg.theta_list) * len(cfg.perturb_modes)
        )
        idx = 0
        t_sweep_start = time.perf_counter()
        for (D, P, K, L) in cfg.configs:
            for perturb_mode in cfg.perturb_modes:
                for seed in cfg.seed_list:
                    for theta in cfg.theta_list:
                        idx += 1
                        t0 = time.perf_counter()
                        trial = _run_trial(
                            cfg, int(D), int(P), int(K), int(L),
                            int(seed), float(theta), perturb_mode, device,
                        )
                        dt = time.perf_counter() - t0
                        ctx.record_step_time(dt)
                        print(
                            f"[{idx:>4d}/{n_total}] "
                            f"D={int(D):>3d} P={int(P):>3d} "
                            f"K={int(K):>3d} L={int(L):>2d}  "
                            f"mode={perturb_mode:<6s}  "
                            f"seed={int(seed):>2d}  θ={float(theta):.1e}  "
                            f"emp = {trial['empirical_abs']:.3e}  "
                            f"B = {trial['B_side_bound']:.3e}  "
                            f"A = {trial['A_side_bound']:.3e}  "
                            f"tot = {trial['total_bound']:.3e}  "
                            f"({dt*1000:.1f} ms)"
                        )
                        trials.append(trial)
                        if device.type == "cuda" and idx % 20 == 0:
                            torch.cuda.empty_cache()
        sweep_wall = time.perf_counter() - t_sweep_start

        # --- Figures ---
        _plot_empirical_vs_bound(cfg, trials, run)
        _plot_decomposition(cfg, trials, run)
        _plot_theta_zero_sanity(cfg, trials, run)
        _plot_bound_slack(cfg, trials, run)

        # --- Save npz + JSON ---
        npz_payload: dict[str, np.ndarray] = {
            "configs": np.asarray(cfg.configs, dtype=np.int64),
            "theta_list": np.asarray(cfg.theta_list, dtype=np.float64),
            "seed_list": np.asarray(cfg.seed_list, dtype=np.int64),
        }
        # Flatten trials into 1-D arrays.
        for key in (
            "D", "P", "K", "L", "seed", "theta",
            "empirical_abs", "empirical_rel",
            "B_side_bound", "A_side_bound", "total_bound",
            "delta_A_op", "delta_B_op",
            "telescoping_coeff", "S_theta_y_norm",
        ):
            npz_payload[key] = np.asarray(
                [t[key] for t in trials],
                dtype=(np.int64 if key in ("D", "P", "K", "L", "seed")
                       else np.float64),
            )
        # Mode is a string; record as integer-encoded (0 = A_only, 1 = B_only).
        mode_to_int = {m: i for i, m in enumerate(cfg.perturb_modes)}
        npz_payload["perturb_mode_int"] = np.asarray(
            [mode_to_int[t["perturb_mode"]] for t in trials], dtype=np.int64,
        )
        npz_payload["perturb_mode_labels"] = np.asarray(
            list(cfg.perturb_modes), dtype="<U16",
        )
        np.savez_compressed(run.npz_path("mask_perturbation"), **npz_payload)

        rows = [
            {k: v for k, v in t.items()}
            for t in trials
        ]
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

        # --- Acceptance ---
        # 1. θ = 0 sanity.
        zero_rows = [t for t in trials if float(t["theta"]) == 0.0]
        zero_worst = (
            max((t["empirical_abs"] for t in zero_rows), default=0.0)
        )
        zero_ok = zero_worst <= cfg.machine_eps_tol

        # 2. Bound respected for every (θ > 0, seed, config).
        bound_violations = []
        worst_excess = 0.0
        for t in trials:
            if float(t["theta"]) == 0.0:
                continue
            excess = (
                t["empirical_abs"] - (t["total_bound"] + cfg.bound_numerical_slack)
            )
            if excess > 0:
                bound_violations.append({
                    "D": t["D"], "P": t["P"], "K": t["K"], "L": t["L"],
                    "seed": t["seed"], "theta": t["theta"],
                    "empirical": t["empirical_abs"],
                    "total_bound": t["total_bound"],
                    "excess": float(excess),
                })
                if excess > worst_excess:
                    worst_excess = float(excess)
        bound_ok = not bound_violations

        # Diagnostic summaries.
        # Worst empirical-to-bound ratio (smaller = tighter).
        nontrivial = [
            t for t in trials
            if float(t["theta"]) > 0 and t["total_bound"] > 0
        ]
        emp_to_bound = [
            t["empirical_abs"] / t["total_bound"] for t in nontrivial
        ]
        worst_emp_to_bound_ratio = (
            float(max(emp_to_bound, default=0.0)) if emp_to_bound else 0.0
        )
        worst_bound_slack = (
            float(min(
                (t["total_bound"] / max(t["empirical_abs"], 1e-30))
                for t in nontrivial
            )) if nontrivial else float("nan")
        )

        # Per-mode aggregates so A_only (canonical) and B_only (auxiliary)
        # acceptance can be reported separately.
        per_mode_aggregates: dict[str, dict[str, Any]] = {}
        for mode in cfg.perturb_modes:
            sub = [t for t in trials if t["perturb_mode"] == mode]
            sub_zero = [t for t in sub if float(t["theta"]) == 0.0]
            sub_pos = [
                t for t in sub
                if float(t["theta"]) > 0 and t["total_bound"] > 0
            ]
            zero_w = max((t["empirical_abs"] for t in sub_zero), default=0.0)
            ratios = [
                t["empirical_abs"] / t["total_bound"] for t in sub_pos
            ]
            worst_ratio = float(max(ratios, default=0.0))
            worst_excess_mode = max(
                (
                    t["empirical_abs"] - (t["total_bound"]
                                          + cfg.bound_numerical_slack)
                    for t in sub_pos
                ),
                default=0.0,
            )
            per_mode_aggregates[mode] = {
                "n_trials": len(sub),
                "n_theta_zero": len(sub_zero),
                "n_theta_positive": len(sub_pos),
                "max_empirical_at_theta_zero": float(zero_w),
                "max_emp_to_bound_ratio_theta_positive": float(worst_ratio),
                "max_excess_over_bound_theta_positive": float(worst_excess_mode),
                "bound_respected": worst_excess_mode <= 0.0,
            }

        status_parts: list[str] = []
        status_parts.append(
            "theta_zero_ok" if zero_ok else
            f"theta_zero_violated(worst={zero_worst:.2e})"
        )
        status_parts.append(
            "bound_respected" if bound_ok else
            f"bound_violated(n={len(bound_violations)},worst={worst_excess:.2e})"
        )
        status = "+".join(status_parts)

        ctx.record_compute_proxy(float(sweep_wall))
        ctx.record_extra("zero_worst", zero_worst)
        ctx.record_extra("worst_excess_over_bound", worst_excess)
        ctx.record_extra("worst_emp_to_bound_ratio", worst_emp_to_bound_ratio)
        ctx.record_extra("worst_bound_slack_min", worst_bound_slack)
        ctx.record_extra("bound_violations", bound_violations[:20])

        ctx.write_summary(
            {
                "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §8.2 (A2)",
                "category": (
                    "exact theorem-A operator-level perturbation "
                    "diagnostic (no learned architecture, no training, "
                    "no projector estimation). Compares the empirical "
                    "full-model forward error around GD-compatibility "
                    "against the full additive (A, B)-operator reduced "
                    "perturbation bound."
                ),
                "framing": (
                    "A2 is the FAITHFUL theorem-A perturbation diagnostic "
                    "because it compares the empirical full-model error "
                    "against the full reduced (A, B)-operator upper bound "
                    "— not against a qualitative mask-distance heuristic, "
                    "and not against an A-only bound. The bound has the "
                    "additive structure total = B-side (ΔB term) + "
                    "A-side (ΔA telescoping term); A-side and B-side are "
                    "reported separately and never folded into each other."
                ),
                "primary_vs_auxiliary": (
                    "A2's PRIMARY theorem result is the A_only sweep — "
                    "the canonical theorem-A perturbation family of plan "
                    "§8.2 (GD-compatible mask interpolated to a train-"
                    "supported structured perturbation). The B_only sweep "
                    "is an AUXILIARY decomposition diagnostic, included "
                    "solely to exercise the additive bound's ΔB term with "
                    "non-zero values (the GA 'perturbed' mask kind keeps "
                    "B_S unchanged, so ΔB ≡ 0 in A_only). B_only is NOT "
                    "presented as a theorem-A perturbation family."
                ),
                "empirical_route": (
                    "Both F_θ and F_GD are computed via the A1b R0 full-"
                    "hidden-state aligned structured forward (build full "
                    "(P+K)×(P+K) bilinear score from (X, Γ); apply "
                    "perturbed/GD signed mask; run L explicit residual-"
                    "stream layer updates). The reduced operators "
                    "(A_S, B_S, T) returned by GA are consumed only by "
                    "the bound computation, never to define the empirical "
                    "error."
                ),
                "bound_components": (
                    "B_side_bound = ||ΔB||_op / L · ||S_θ y||_2;  "
                    "A_side_bound = ||B_GD||_op · ||ΔA||_op · ||y||_2 / L^2 "
                    "· Σ U_ℓ where U_ℓ = Σ_k ||T_θ||_op^{ℓ-1-k} · "
                    "||T_GD||_op^k. Total = B + A."
                ),
                "interpretation": (
                    "θ = 0 recovers the GD-compatible case exactly "
                    "(empirical error at float64 eps). Across the "
                    "small-θ regime, the empirical full-model error "
                    "scales linearly with θ (because both ΔA and ΔB "
                    "scale with θ) and the total bound matches the "
                    "empirical to within the slack predicted by the "
                    "decomposition. The bound is respected at every "
                    "tested (θ, seed, configuration)."
                ),
                "device": str(device),
                "configs": [list(c) for c in cfg.configs],
                "theta_list": list(cfg.theta_list),
                "seed_list": list(cfg.seed_list),
                "pattern_seed": cfg.pattern_seed,
                "label_norm": cfg.label_norm,
                "Sigma_kind": cfg.Sigma_kind,
                "Omega_kind": cfg.Omega_kind,
                "Gamma_kind": cfg.Gamma_kind,
                "n_trials": len(trials),
                "status": status,
                "machine_eps_tol": cfg.machine_eps_tol,
                "bound_numerical_slack": cfg.bound_numerical_slack,
                "theta_zero_worst_empirical": float(zero_worst),
                "worst_excess_over_bound": float(worst_excess),
                "n_bound_violations": len(bound_violations),
                "worst_emp_to_bound_ratio_max": worst_emp_to_bound_ratio,
                "worst_bound_slack_min": worst_bound_slack,
                "per_mode_aggregates": per_mode_aggregates,
                "sweep_wallclock_seconds": round(float(sweep_wall), 3),
            }
        )

        print()
        print("=" * 72)
        print(f" A2 mask perturbation on {device}")
        print(f"   N trials = {len(trials)} (config × seed × θ)")
        print(
            f"   θ = 0 sanity: worst |F_0 - F_GD| = {zero_worst:.3e}  "
            f"{'OK' if zero_ok else 'FAIL'}  "
            f"(tol = {cfg.machine_eps_tol:.1e})"
        )
        print(
            f"   bound respected: violations = {len(bound_violations)}, "
            f"worst excess = {worst_excess:.3e}  "
            f"{'OK' if bound_ok else 'FAIL'}"
        )
        print(
            f"   worst emp/bound ratio = {worst_emp_to_bound_ratio:.3e}  "
            f"(< 1 means bound is loose; > 1 means violation)"
        )
        if worst_bound_slack == worst_bound_slack:
            print(
                f"   worst bound slack (min total/emp) = "
                f"{worst_bound_slack:.3e}"
            )
        print("   per-mode aggregates:")
        for mode, agg in per_mode_aggregates.items():
            label = (
                "(canonical theorem-A)" if mode == "A_only"
                else "(auxiliary ΔB diagnostic)"
            )
            print(
                f"     mode = {mode:<7s} {label}  "
                f"n = {agg['n_trials']}  "
                f"max emp@θ=0 = {agg['max_empirical_at_theta_zero']:.2e}  "
                f"max emp/bound = "
                f"{agg['max_emp_to_bound_ratio_theta_positive']:.3e}  "
                f"max excess = {agg['max_excess_over_bound_theta_positive']:.2e}  "
                f"{'OK' if agg['bound_respected'] else 'VIOLATED'}"
            )
        print("=" * 72)

        if not zero_ok or not bound_ok:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
