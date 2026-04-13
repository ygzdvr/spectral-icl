"""Experiment A1b: full hidden-state structured forward vs reduced (A_S, B_S).

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §8.1 (extension of A1).

Why A1b exists (clarification of A1's R1 route)
-----------------------------------------------
A1's "R1" route iterates the reduced sample-space operators ``A_S, B_S, T``
from :func:`ga_generate` directly:

    z_test  ← z_test  + (1/L) · B_S_θ · z_train
    z_train ← z_train + (1/L) · A_S_θ · z_train.

That is an *iterative form* of the reduced recursion, not a full structured
forward pass — it consumes the already-reduced ``(A_S, B_S, T)`` as inputs.
A1b adds the missing route:

    R0 — **true full hidden-state aligned structured forward pass**:
    constructs the full ``(P + K)``-position residual stream and the full
    ``(P + K) × (P + K)`` bilinear score matrix from ``(X_train, X_query, Γ)``
    directly via the theorem-A channel/alignment construction, applies a
    GD-compatible signed mask, and runs L explicit residual-stream layer
    updates. R0 *does not* consume ``A_S, B_S, T`` as inputs.

A1b compares R0 against the same R2 / R3 closed forms used in A1:

    R2 — sample-space reduced ``(A_S, B_S)`` recursion (closed form).
    R3 — feature-space reduced-Γ closed form (GD-compatible special case).

Acceptance is the same: float64 machine precision pairwise across
``(R0, R2, R3)`` over a sweep of ``(D, P, K, L)``.

A1's canonical result is unchanged. A1b is a strictly additional check
that closes the gap between "full-model forward pass" and "reduced
recursion" identified in the user clarification.

Aligned-mixer construction (R0)
-------------------------------
For the GD-compatible setting with isotropic ``Σ, Ω`` and
``Γ = identity`` (or arbitrary ``Γ``), the theorem-A aligned linear
structured mixer at one layer applies, on the full ``(P + K)``-position
residual-stream scalar channel ``h``,

    h_μ^{(ℓ + 1)} = h_μ^{(ℓ)} + (1/L) · Σ_ν  M_signed[μ, ν] · S[μ, ν] · h_ν^{(ℓ)},

where

    S[μ, ν] = (x_μ)^T · Γ · (x_ν) / P                  (bilinear score)
    M_signed[μ, ν] = −1   if μ ∈ train and ν ∈ train     (residual descent)
    M_signed[μ, ν] = +1   if μ ∈ test  and ν ∈ train     (prediction ascent)
    M_signed[μ, ν] =  0   otherwise.

The signed mask encodes the GD direction: train positions descend the
residual via attention into other train positions; test positions
accumulate prediction by attending to train positions. The reduced
operators recovered from this construction are exactly

    A_S^GD = −(1/P) · X_train^T · Γ · X_train,
    B_S^GD = +(1/P) · X_query^T · Γ · X_train,

so the layered residual-stream forward of R0 produces

    h_train^{(L)} = T_GD^{L} · y_train,
    h_test^{(L)}  = (1/L) · B_S^GD · Σ_{ℓ = 0..L−1} T_GD^{ℓ} · y_train,

which is the theorem-A reduced prediction at the query positions.

R0 verifies that the GA generator's ``(A_S, B_S, T)`` outputs are
consistent with what the FULL aligned linear-attention structured mixer
produces from the same ``(X, Γ)``, as a forward map. That is the
end-to-end exactness statement A1 was meant to establish; A1b makes it
explicit.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``GAConfig``, ``ga_generate``.
- :mod:`scripts.thesis.utils.metrics`:
    ``reduced_model_error``.
- :mod:`scripts.thesis.utils.plotting` / :mod:`run_metadata`: standard.

Primary outputs
---------------
- ``a1b_pairwise_errors_heatmap`` — max pairwise relative error
  ``max(err(R0,R2), err(R2,R3))`` over the (D, P) sweep at fixed (K, L).
- ``a1b_error_distribution`` — histogram of the three pairwise relative
  errors over all (D, P, K, L) cells.
- ``a1b_error_vs_L`` — diagnostic on how the worst error grows with L.

Acceptance
----------
1. **R0 vs R2 (full model vs reduced AB)**: ``max_err ≤ machine_eps_tol``
   over every sweep cell. This is the missing check that A1's R1 did not
   exercise.
2. **R2 vs R3 (reduced AB vs reduced Γ)**: ``max_err ≤ machine_eps_tol``
   over every sweep cell.

Both strict.

Run
---
::

    python -u scripts/thesis/theoremA/run_theoremA_exact_equivalence_full_model.py \\
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
class A1bConfig:
    """Frozen configuration for A1b: full-hidden-state forward vs reduced.

    Default 4 D × 4 P × 3 K × 4 L = 192 cells × 3 routes = 576 forward
    evaluations; finishes in seconds. Mirrors A1's sweep so the two
    canonicals are directly comparable.
    """

    D_list: tuple[int, ...] = (8, 16, 32, 64)
    P_list: tuple[int, ...] = (8, 16, 32, 64)
    K_list: tuple[int, ...] = (4, 8, 16)
    L_list: tuple[int, ...] = (1, 2, 4, 8)

    Sigma_kind: str = "isotropic"
    Omega_kind: str = "isotropic"
    Gamma_kind: str = "identity"
    label_norm: str = "sqrt_D"   # plan §3 third
    sigma: float = 0.0
    B: int = 4
    base_seed: int = 0

    machine_eps_tol: float = 1e-10

    heatmap_K: int = 8
    heatmap_L: int = 4

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Three forward routes — R0 is the genuine full-hidden-state structured
# forward; R2 and R3 are the same closed forms used in A1.
# ---------------------------------------------------------------------------


def _route_R0_full_hidden_state_forward(
    X_train: torch.Tensor,    # (B, D, P)
    X_query: torch.Tensor,    # (B, D, K)
    Gamma: torch.Tensor,      # (D, D)
    y_train: torch.Tensor,    # (B, P)
    L: int,
    P_norm: int,
) -> torch.Tensor:
    """**True full-hidden-state aligned structured forward pass.**

    Builds the full ``(P + K, P + K)`` bilinear score matrix from
    ``X = [X_train | X_query]``, applies a GD-compatible signed mask,
    and runs an explicit L-layer residual-stream forward on a length-
    ``(P + K)`` scalar hidden channel. Does NOT consume ``A_S, B_S, T``
    as inputs — those are constructed only implicitly by the bilinear
    score on the full sequence. Returns the prediction at the K test
    positions, shape ``(B, K)``.
    """
    B = int(X_train.shape[0])
    D = int(X_train.shape[1])
    P = int(X_train.shape[-1])
    K = int(X_query.shape[-1])

    dtype = X_train.dtype
    device = X_train.device

    # Full-sequence feature tensor X ∈ R^{B × D × (P + K)}.
    X = torch.cat([X_train, X_query], dim=-1)

    # Bilinear "positive" score: S_pos[μ, ν] = x_μ^T Γ x_ν / P, shape (B, P+K, P+K).
    GammaX = torch.einsum("de,bef->bdf", Gamma, X)
    S_pos = torch.einsum("bdm,bdn->bmn", X, GammaX) / float(P_norm)

    # Signed GD-compatible mask:
    #   train → train: -1   (residual descent on training residuals)
    #   test  → train: +1   (prediction accumulation at queries)
    #   train → test:  0    (train does not see test labels)
    #   test  → test:  0    (queries do not attend to other queries)
    M_signed = torch.zeros(P + K, P + K, dtype=dtype, device=device)
    M_signed[:P, :P] = -1.0
    M_signed[P:, :P] = +1.0
    M_eff = S_pos * M_signed.unsqueeze(0)  # (B, P+K, P+K)

    # Hidden state initialization: y_train at train positions, 0 at test.
    h = torch.cat(
        [
            y_train,
            torch.zeros(B, K, dtype=dtype, device=device),
        ],
        dim=-1,
    )  # (B, P+K)

    # L explicit residual-stream layer updates.
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        update = torch.einsum("bmn,bn->bm", M_eff, h)
        h = h + inv_L * update

    # Prediction = test-position channel value after L layers.
    return h[:, P:]


def _route_R2_reduced_AB(
    A_S: torch.Tensor,
    B_S: torch.Tensor,
    y_train: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """R2: closed-form sample-space reduced (A_S, B_S) recursion (matches A1)."""
    z = y_train.clone()
    sum_T_y = torch.zeros_like(z)
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        sum_T_y = sum_T_y + z
        z = z + inv_L * torch.einsum("bpi,bi->bp", A_S, z)
    return inv_L * torch.einsum("bki,bi->bk", B_S, sum_T_y)


def _route_R3_reduced_Gamma_feature_space(
    X_train: torch.Tensor,
    X_query: torch.Tensor,
    Gamma: torch.Tensor,
    y_train: torch.Tensor,
    L: int,
    P_norm: int,
) -> torch.Tensor:
    """R3: feature-space preconditioned-GD reduced-Γ closed form (matches A1)."""
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
    cfg: A1bConfig, D: int, P: int, K: int, L: int, device: torch.device,
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
        return_feature_space=False,
        seeds=seeds,
        dtype=cfg.dtype,
        device="cpu",
    )
    op = ga_generate(g)
    X_train = op["X_train"].to(device)
    X_query = op["X_query"].to(device)
    y_train = op["y_train"].to(device)
    Gamma = op["Gamma"].to(device)
    A_S_GD = op["A_S_GD"].to(device)
    B_S_GD = op["B_S_GD"].to(device)

    t0 = time.perf_counter()
    f_R0 = _route_R0_full_hidden_state_forward(
        X_train, X_query, Gamma, y_train, L=int(L), P_norm=int(P),
    )
    t_R0 = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R2 = _route_R2_reduced_AB(A_S_GD, B_S_GD, y_train, L=int(L))
    t_R2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R3 = _route_R3_reduced_Gamma_feature_space(
        X_train, X_query, Gamma, y_train, L=int(L), P_norm=int(P),
    )
    t_R3 = time.perf_counter() - t0

    err_R0_R2 = reduced_model_error(f_R0, f_R2)
    err_R2_R3 = reduced_model_error(f_R2, f_R3)
    err_R0_R3 = reduced_model_error(f_R0, f_R3)

    return {
        "D": int(D), "P": int(P), "K": int(K), "L": int(L),
        "f_R0": f_R0.detach().cpu(),
        "f_R2": f_R2.detach().cpu(),
        "f_R3": f_R3.detach().cpu(),
        "err_R0_R2": float(err_R0_R2),
        "err_R2_R3": float(err_R2_R3),
        "err_R0_R3": float(err_R0_R3),
        "t_R0_seconds": float(t_R0),
        "t_R2_seconds": float(t_R2),
        "t_R3_seconds": float(t_R3),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_pairwise_errors_heatmap(
    cfg: A1bConfig, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
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
        grid[i_D, i_P] = max(trial["err_R0_R2"], trial["err_R2_R3"])
    floor = 1e-18
    grid_plot = np.where(grid > floor, grid, floor)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    _pc, _cb = phase_heatmap(
        ax, grid_plot,
        x_coords=np.asarray(P_list, dtype=float),
        y_coords=np.asarray(D_list, dtype=float),
        xlabel="P (training context length)",
        ylabel="D (feature dimension)",
        cbar_label=r"max pairwise rel. error",
        log_z=True, log_x=True, log_y=True,
    )
    ax.set_title(
        rf"A1b max pairwise relative error at K = {K}, L = {L} "
        r"(R0 full-model vs R2 reduced AB)",
        fontsize=10,
    )
    fig.tight_layout()
    save_both(fig, run_dir, "a1b_pairwise_errors_heatmap")
    plt.close(fig)


def _plot_error_distribution(
    cfg: A1bConfig, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    err_R0_R2 = np.array([t["err_R0_R2"] for t in trials])
    err_R2_R3 = np.array([t["err_R2_R3"] for t in trials])
    err_R0_R3 = np.array([t["err_R0_R3"] for t in trials])
    floor = 1e-18
    e0 = np.clip(err_R0_R2, floor, None)
    e2 = np.clip(err_R2_R3, floor, None)
    e3 = np.clip(err_R0_R3, floor, None)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.logspace(-18, -8, 30)
    ax.hist(e0, bins=bins, alpha=0.6,
            label="R0 vs R2 (full model vs reduced AB)",
            edgecolor="black", lw=0.4)
    ax.hist(e2, bins=bins, alpha=0.6,
            label="R2 vs R3 (reduced AB vs reduced Γ)",
            edgecolor="black", lw=0.4)
    ax.hist(e3, bins=bins, alpha=0.6,
            label="R0 vs R3 (full model vs reduced Γ)",
            edgecolor="black", lw=0.4)
    ax.axvline(cfg.machine_eps_tol, color="red", lw=0.9, ls="--",
               label=f"acceptance = {cfg.machine_eps_tol:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel("relative error")
    ax.set_ylabel("count")
    ax.set_title(
        f"A1b pairwise relative-error distribution over {len(trials)} cells",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a1b_error_distribution")
    plt.close(fig)


def _plot_error_vs_L(
    cfg: A1bConfig, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    L_list = list(cfg.L_list)
    L_max_err = []
    for L in L_list:
        sub = [t for t in trials if int(t["L"]) == int(L)]
        if not sub:
            L_max_err.append(np.nan)
            continue
        L_max_err.append(
            max(max(t["err_R0_R2"], t["err_R2_R3"]) for t in sub)
        )
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(
        L_list, L_max_err, color="C0", lw=1.5, marker="o", ms=5.0,
        label="max pairwise rel. error across (D, P, K)",
    )
    ax.axhline(
        cfg.machine_eps_tol, color="red", lw=0.9, ls="--",
        label=f"acceptance = {cfg.machine_eps_tol:.0e}",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("depth L")
    ax.set_ylabel("max pairwise rel. error")
    ax.set_title("A1b error growth with depth", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a1b_error_vs_L")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment A1b: full-hidden-state structured forward vs "
            "reduced (A_S, B_S) and reduced-Γ closed forms (plan §8.1, "
            "extension of A1)."
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


def _config_from_cli(args: argparse.Namespace) -> A1bConfig:
    base = A1bConfig()
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
    print(f"[A1b] device = {device}")

    run = ThesisRunDir(__file__, phase="theoremA")
    with RunContext(
        run,
        config=cfg,
        seeds=[cfg.base_seed, cfg.base_seed + 7,
               cfg.base_seed + 13, cfg.base_seed + 19],
        notes=(
            "A1b: true full-hidden-state aligned structured forward vs "
            "reduced (A_S, B_S) and reduced-Γ closed forms. R0 builds "
            "the full (P+K)-position residual stream and (P+K)×(P+K) "
            "bilinear score from (X, Γ) directly, applies a "
            "GD-compatible signed mask, and runs L explicit residual-"
            "stream layer updates — without consuming the GA-generator's "
            "(A_S, B_S, T) as inputs. Closes the gap left by A1's R1, "
            "which iterated the reduced operators directly."
        ),
    ) as ctx:
        apply_thesis_style()

        trials: list[dict[str, Any]] = []
        n_total = (
            len(cfg.D_list) * len(cfg.P_list)
            * len(cfg.K_list) * len(cfg.L_list)
        )
        idx = 0
        t_sweep_start = time.perf_counter()
        for D in cfg.D_list:
            for P in cfg.P_list:
                for K in cfg.K_list:
                    for L in cfg.L_list:
                        idx += 1
                        t0 = time.perf_counter()
                        trial = _run_trial(cfg, int(D), int(P), int(K), int(L), device)
                        dt = time.perf_counter() - t0
                        ctx.record_step_time(dt)
                        print(
                            f"[{idx:>4d}/{n_total}] "
                            f"D={int(D):>3d} P={int(P):>3d} "
                            f"K={int(K):>3d} L={int(L):>2d}  "
                            f"err_R0_R2 = {trial['err_R0_R2']:.2e}  "
                            f"err_R2_R3 = {trial['err_R2_R3']:.2e}  "
                            f"err_R0_R3 = {trial['err_R0_R3']:.2e}  "
                            f"({dt*1000:.1f} ms)"
                        )
                        trials.append(trial)
        sweep_wall = time.perf_counter() - t_sweep_start

        # --- Figures ---
        _plot_pairwise_errors_heatmap(cfg, trials, run)
        _plot_error_distribution(cfg, trials, run)
        _plot_error_vs_L(cfg, trials, run)

        # --- Save npz ---
        D_list = list(cfg.D_list)
        P_list = list(cfg.P_list)
        K_list = list(cfg.K_list)
        L_list = list(cfg.L_list)
        shape = (len(D_list), len(P_list), len(K_list), len(L_list))
        err_R0_R2_grid = np.zeros(shape)
        err_R2_R3_grid = np.zeros(shape)
        err_R0_R3_grid = np.zeros(shape)
        for trial in trials:
            i = D_list.index(trial["D"])
            j = P_list.index(trial["P"])
            k = K_list.index(trial["K"])
            l = L_list.index(trial["L"])
            err_R0_R2_grid[i, j, k, l] = trial["err_R0_R2"]
            err_R2_R3_grid[i, j, k, l] = trial["err_R2_R3"]
            err_R0_R3_grid[i, j, k, l] = trial["err_R0_R3"]
        npz_payload: dict[str, np.ndarray] = {
            "D_list": np.asarray(D_list, dtype=np.int64),
            "P_list": np.asarray(P_list, dtype=np.int64),
            "K_list": np.asarray(K_list, dtype=np.int64),
            "L_list": np.asarray(L_list, dtype=np.int64),
            "err_R0_R2_grid": err_R0_R2_grid,
            "err_R2_R3_grid": err_R2_R3_grid,
            "err_R0_R3_grid": err_R0_R3_grid,
        }
        np.savez_compressed(
            run.npz_path("exact_equivalence_full_model"), **npz_payload,
        )

        # --- Per-cell JSON ---
        rows = [
            {
                "D": t["D"], "P": t["P"], "K": t["K"], "L": t["L"],
                "err_R0_R2": float(t["err_R0_R2"]),
                "err_R2_R3": float(t["err_R2_R3"]),
                "err_R0_R3": float(t["err_R0_R3"]),
                "t_R0_seconds": float(t["t_R0_seconds"]),
                "t_R2_seconds": float(t["t_R2_seconds"]),
                "t_R3_seconds": float(t["t_R3_seconds"]),
            }
            for t in trials
        ]
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

        # --- Acceptance ---
        worst_R0_R2 = max(t["err_R0_R2"] for t in trials)
        worst_R2_R3 = max(t["err_R2_R3"] for t in trials)
        worst_R0_R3 = max(t["err_R0_R3"] for t in trials)
        ok_R0_R2 = worst_R0_R2 <= cfg.machine_eps_tol
        ok_R2_R3 = worst_R2_R3 <= cfg.machine_eps_tol
        all_ok = ok_R0_R2 and ok_R2_R3

        worst_cell: dict[str, Any] | None = None
        worst_err = 0.0
        for t in trials:
            err_max = max(t["err_R0_R2"], t["err_R2_R3"])
            if err_max > worst_err:
                worst_err = err_max
                worst_cell = {
                    "D": t["D"], "P": t["P"], "K": t["K"], "L": t["L"],
                    "err_R0_R2": float(t["err_R0_R2"]),
                    "err_R2_R3": float(t["err_R2_R3"]),
                }

        ctx.record_compute_proxy(float(sweep_wall))
        ctx.record_extra("worst_R0_R2", worst_R0_R2)
        ctx.record_extra("worst_R2_R3", worst_R2_R3)
        ctx.record_extra("worst_R0_R3", worst_R0_R3)
        ctx.record_extra("worst_cell", worst_cell)

        status_parts: list[str] = []
        status_parts.append(
            "R0_R2_ok" if ok_R0_R2 else
            f"R0_R2_violated(worst={worst_R0_R2:.2e})"
        )
        status_parts.append(
            "R2_R3_ok" if ok_R2_R3 else
            f"R2_R3_violated(worst={worst_R2_R3:.2e})"
        )
        status = "+".join(status_parts)

        ctx.write_summary(
            {
                "plan_reference": (
                    "EXPERIMENT_PLAN_FINAL.MD §8.1 (A1, A1b extension)"
                ),
                "category": (
                    "exact theorem-A operator-level forward-pass "
                    "equivalence test — A1b adds the TRUE full-hidden-"
                    "state aligned structured forward pass (R0) that "
                    "A1's R1 (iterative reduced recursion) did not "
                    "exercise. R0 builds the full (P+K)×(P+K) bilinear "
                    "score from (X, Γ) and applies L explicit residual-"
                    "stream layer updates with a GD-compatible signed "
                    "mask, without consuming the GA-generator's "
                    "(A_S, B_S, T) as inputs. Compared against the same "
                    "R2 (reduced AB closed form) and R3 (reduced-Γ "
                    "closed form) used in A1."
                ),
                "interpretation": (
                    "If R0 (full model) agrees with R2 (reduced AB) "
                    "to float64 machine precision, the GA generator's "
                    "(A_S, B_S, T) outputs are exactly consistent with "
                    "the forward map of the aligned linear-attention "
                    "structured mixer in the GD-compatible regime — "
                    "this is the end-to-end exactness statement "
                    "theorem A asserts. A1's canonical result remains "
                    "valid and unchanged; A1b is a strictly additional "
                    "check closing the full-vs-reduced gap."
                ),
                "device": str(device),
                "D_list": list(cfg.D_list),
                "P_list": list(cfg.P_list),
                "K_list": list(cfg.K_list),
                "L_list": list(cfg.L_list),
                "n_cells": len(trials),
                "B": cfg.B,
                "label_norm": cfg.label_norm,
                "Sigma_kind": cfg.Sigma_kind,
                "Omega_kind": cfg.Omega_kind,
                "Gamma_kind": cfg.Gamma_kind,
                "mask_kind": "gd_compatible",
                "status": status,
                "machine_eps_tol": cfg.machine_eps_tol,
                "worst_R0_R2": float(worst_R0_R2),
                "worst_R2_R3": float(worst_R2_R3),
                "worst_R0_R3": float(worst_R0_R3),
                "worst_cell": worst_cell,
                "sweep_wallclock_seconds": round(float(sweep_wall), 3),
            }
        )

        print()
        print("=" * 72)
        print(f" A1b full-hidden-state vs reduced equivalence on {device}")
        print(f"   N cells = {len(trials)} (D × P × K × L)")
        print(
            f"   worst R0 vs R2 (full model vs reduced AB) = "
            f"{worst_R0_R2:.3e}  "
            f"{'OK' if ok_R0_R2 else 'FAIL'}  "
            f"(tol = {cfg.machine_eps_tol:.1e})"
        )
        print(
            f"   worst R2 vs R3 (reduced AB vs reduced Γ) = "
            f"{worst_R2_R3:.3e}  "
            f"{'OK' if ok_R2_R3 else 'FAIL'}  "
            f"(tol = {cfg.machine_eps_tol:.1e})"
        )
        print(
            f"   worst R0 vs R3 (full model vs reduced Γ) = "
            f"{worst_R0_R3:.3e}  (diagnostic)"
        )
        if worst_cell is not None and worst_err > 0:
            print(
                f"   worst cell: D={worst_cell['D']}, "
                f"P={worst_cell['P']}, K={worst_cell['K']}, "
                f"L={worst_cell['L']}"
            )
        print("=" * 72)

        if not all_ok:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
