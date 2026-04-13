"""Experiment A1: exact full structured model versus reduced (A_S, B_S) recursion.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §8.1.

Theorem-level framing (read carefully)
--------------------------------------
A1 is the first exact theorem-A experiment. It is a **deterministic
forward-pass equivalence test** at the operator level — there is NO
training, NO architecture grid search, NO learned model. The single
question A1 answers is:

    Does the L-layer fully-aligned linear-structured-mixer forward pass
    equal the closed-form sample-space reduced (A_S, B_S) recursion to
    float64 machine precision, for every (D, P, K, L) in the sweep?

Theorem A asserts this equivalence as an algebraic identity in the
GD-compatible case. A1 implements it three different ways and verifies
they all agree at float eps:

(R1) **Full layer simulation.** Initialize ``z_train = y_train``,
    ``z_test = 0``. At each of L layers apply the residual updates

        z_test  ← z_test  + (1/L) · B_S_θ · z_train
        z_train ← z_train + (1/L) · A_S_θ · z_train

    This is the "fully aligned linear structured mixer" forward pass —
    a discrete L-layer transformer-like residual stream restricted to the
    GD-compatible attention pattern.

(R2) **Sample-space reduced (A_S, B_S) closed form.** Compute

        f_red = (1/L) · B_S · Σ_{ℓ=0..L-1} T^ℓ · y_train,
        T = I_P + A_S/L.

    This is the theorem-A reduced-operator recursion of plan §8.1.

(R3) **Feature-space reduced-Γ closed form (GD-compatible only).** Run the
    L-step preconditioned GD iterate in feature space:

        w_0 = 0
        w_{ℓ+1} = w_ℓ + (1/L) · Γ · X_train · (y_train − X_train^T w_ℓ) / P
        f_Γ = (1/L) · Σ_{ℓ=0..L-1} X_query^T · (the running prediction sum)

    Equivalently, ``f_Γ = X_query^T · w_running`` where ``w_running``
    accumulates the L-step iterate. This is the **reduced-Γ special
    case** required by plan §8.1 in the GD-compatible setting.

All three quantities are EQUAL by theorem A in the GD-compatible case;
they are computed via three structurally different code paths so the
test is a strong implementation-correctness check. Acceptance is

    max( |R1 − R2| / |R1|, |R2 − R3| / |R2| ) ≤ 1e-10  (float64).

If any of the three pairwise errors exceeds float eps in float64, the
implementation of one path is wrong.

Sweep (plan §8.1 binding)
-------------------------
The script sweeps ``D, P, K, L`` per the plan. Default grid: 4×4×3×4 =
192 cells. Each cell is a few small matrix multiplications, so the full
sweep finishes in seconds.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``GAConfig``, ``ga_generate``.
- :mod:`scripts.thesis.utils.metrics`:
    ``reduced_model_error``.
- :mod:`scripts.thesis.utils.plotting` / :mod:`run_metadata`: standard.

Primary outputs
---------------
- ``a1_pairwise_errors`` — heatmap of max pairwise relative error
  (R1 vs R2, R2 vs R3) over the (D, P) sweep at fixed (K, L). Expected:
  every cell at float eps.
- ``a1_error_distribution`` — histogram of all pairwise errors across
  the full sweep, with the acceptance tolerance overlaid.
- ``a1_sweep_table`` — a per-cell dump in JSON with
  (D, P, K, L, err_R1_R2, err_R2_R3) for every trial.

Acceptance
----------
1. **Full-vs-reduced (R1 vs R2)**: ``max_err ≤ machine_eps_tol`` over
   every sweep cell.
2. **Reduced-AB-vs-Γ (R2 vs R3)** in the GD-compatible case:
   ``max_err ≤ machine_eps_tol`` over every sweep cell.

Both are strict — no tolerance for "qualitative agreement". A1 is an
exactness test.

Run
---
::

    python -u scripts/thesis/theoremA/run_theoremA_exact_equivalence.py \\
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
class A1Config:
    """Frozen configuration for the A1 exact-equivalence sweep.

    Default 4 D × 4 P × 3 K × 4 L = 192 cells × 3 forward routes
    = 576 evaluations. Each evaluation is a few small matmuls, so the
    full sweep completes in seconds.
    """

    D_list: tuple[int, ...] = (8, 16, 32, 64)
    P_list: tuple[int, ...] = (8, 16, 32, 64)
    K_list: tuple[int, ...] = (4, 8, 16)
    L_list: tuple[int, ...] = (1, 2, 4, 8)

    # GA generator config — GD-compatible mask, isotropic Σ, Ω, Γ. The
    # exact equivalence holds for ANY Σ, Ω, Γ choice; isotropic is the
    # canonical theorem-A reference.
    Sigma_kind: str = "isotropic"
    Omega_kind: str = "isotropic"
    Gamma_kind: str = "identity"
    label_norm: str = "sqrt_D"   # plan §3 third: theorem-A defaults to sqrt_D
    sigma: float = 0.0           # noise-free for exactness
    B: int = 4                   # batch ≥ 1 stresses the einsum paths
    base_seed: int = 0

    # Acceptance.
    machine_eps_tol: float = 1e-10

    # Figure slices.
    heatmap_K: int = 8
    heatmap_L: int = 4

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Three forward routes
# ---------------------------------------------------------------------------


def _route_R1_full_layer_simulation(
    A_S: torch.Tensor,   # (B, P, P)
    B_S: torch.Tensor,   # (B, K, P)
    y_train: torch.Tensor,  # (B, P)
    L: int,
) -> torch.Tensor:
    """R1: explicit L-layer residual stream. The fully aligned linear
    structured mixer at each layer applies

        z_test  ← z_test  + (1/L) · B_S_θ · z_train
        z_train ← z_train + (1/L) · A_S_θ · z_train

    Returns ``f_full = z_test^(L)`` of shape ``(B, K)``.
    """
    z_train = y_train.clone()
    K = int(B_S.shape[-2])
    z_test = torch.zeros(
        *y_train.shape[:-1], K,
        dtype=y_train.dtype, device=y_train.device,
    )
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        z_test = z_test + inv_L * torch.einsum("bki,bi->bk", B_S, z_train)
        z_train = z_train + inv_L * torch.einsum("bpi,bi->bp", A_S, z_train)
    return z_test


def _route_R2_reduced_AB(
    A_S: torch.Tensor,
    B_S: torch.Tensor,
    y_train: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """R2: closed-form sample-space reduced (A_S, B_S) recursion:

        f_red = (1/L) · B_S · Σ_{ℓ=0..L-1} T^ℓ · y_train,   T = I + A_S/L.

    Returns ``f_red`` of shape ``(B, K)``. Built by accumulating the
    Krylov-style sum ``Σ T^ℓ y`` instead of materializing T^ℓ.
    """
    z = y_train.clone()
    sum_T_y = torch.zeros_like(z)
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        sum_T_y = sum_T_y + z
        # z := T z = z + A_S z / L
        z = z + inv_L * torch.einsum("bpi,bi->bp", A_S, z)
    return inv_L * torch.einsum("bki,bi->bk", B_S, sum_T_y)


def _route_R3_reduced_Gamma_feature_space(
    X_train: torch.Tensor,   # (B, D, P)
    X_query: torch.Tensor,   # (B, D, K)
    Gamma: torch.Tensor,     # (D, D)
    y_train: torch.Tensor,   # (B, P)
    L: int,
    P_norm: int,
) -> torch.Tensor:
    """R3: feature-space preconditioned GD reduced-Γ closed form
    (GD-compatible setting only). Iterates

        w_0 = 0
        r_ℓ = y_train − X_train^T w_ℓ
        w_{ℓ+1} = w_ℓ + (1/L) · Γ · X_train · r_ℓ / P

    For L iterations. Returns ``f_Γ = X_query^T w_L`` of shape (B, K).

    Mathematically equivalent to R2 because (i) the residual r_ℓ
    satisfies r_ℓ = T_GD^ℓ · y_train with T_GD = I + A_S_GD/L and
    A_S_GD = -X_train^T Γ X_train / P, and (ii) the cumulative w iterate
    satisfies w_L = (1/L) · Γ · X_train · Σ T_GD^ℓ · y_train / P.
    """
    B = int(X_train.shape[0])
    D = int(X_train.shape[1])
    inv_L = 1.0 / float(L)
    inv_P = 1.0 / float(P_norm)
    w = torch.zeros(B, D, dtype=y_train.dtype, device=y_train.device)
    for _ in range(int(L)):
        # r = y_train - X_train^T w   (per batch).
        Xt_w = torch.einsum("bdp,bd->bp", X_train, w)
        r = y_train - Xt_w
        # w := w + (1/L) · Γ · X · r / P
        Xr = torch.einsum("bdp,bp->bd", X_train, r)        # (B, D)
        GXr = torch.einsum("de,be->bd", Gamma, Xr)          # (B, D)
        w = w + inv_L * inv_P * GXr
    # f_Γ = X_query^T w
    return torch.einsum("bdk,bd->bk", X_query, w)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def _run_trial(
    cfg: A1Config, D: int, P: int, K: int, L: int, device: torch.device,
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
        device="cpu",  # GA generator builds tensors on CPU; we move below.
    )
    op = ga_generate(g)
    X_train = op["X_train"].to(device)
    X_query = op["X_query"].to(device)
    y_train = op["y_train"].to(device)
    Gamma = op["Gamma"].to(device)
    A_S = op["A_S_theta"].to(device)
    B_S = op["B_S_theta"].to(device)
    A_S_GD = op["A_S_GD"].to(device)
    # In the GD-compatible mask case A_S_θ ≡ A_S_GD; assert it for safety.
    if not torch.equal(A_S, A_S_GD):
        raise AssertionError(
            f"A_S_theta != A_S_GD under mask_kind='gd_compatible' "
            f"(D={D}, P={P}, K={K}, L={L})"
        )

    t0 = time.perf_counter()
    f_R1 = _route_R1_full_layer_simulation(A_S, B_S, y_train, L=int(L))
    t_R1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R2 = _route_R2_reduced_AB(A_S, B_S, y_train, L=int(L))
    t_R2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R3 = _route_R3_reduced_Gamma_feature_space(
        X_train, X_query, Gamma, y_train, L=int(L), P_norm=int(P),
    )
    t_R3 = time.perf_counter() - t0

    err_R1_R2 = reduced_model_error(f_R1, f_R2)
    err_R2_R3 = reduced_model_error(f_R2, f_R3)
    err_R1_R3 = reduced_model_error(f_R1, f_R3)

    return {
        "D": int(D), "P": int(P), "K": int(K), "L": int(L),
        "f_R1": f_R1.detach().cpu(),
        "f_R2": f_R2.detach().cpu(),
        "f_R3": f_R3.detach().cpu(),
        "err_R1_R2": float(err_R1_R2),
        "err_R2_R3": float(err_R2_R3),
        "err_R1_R3": float(err_R1_R3),
        "t_R1_seconds": float(t_R1),
        "t_R2_seconds": float(t_R2),
        "t_R3_seconds": float(t_R3),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_pairwise_errors_heatmap(
    cfg: A1Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Heatmap of max(err_R1_R2, err_R2_R3) over (D, P) at fixed (K, L)."""
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
        grid[i_D, i_P] = max(trial["err_R1_R2"], trial["err_R2_R3"])
    floor = 1e-18
    grid_plot = np.where(grid > floor, grid, floor)
    D_arr = np.asarray(D_list, dtype=float)
    P_arr = np.asarray(P_list, dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    _pc, _cb = phase_heatmap(
        ax, grid_plot,
        x_coords=P_arr, y_coords=D_arr,
        xlabel="P (training context length)",
        ylabel="D (feature dimension)",
        cbar_label=r"max pairwise rel. error",
        log_z=True, log_x=True, log_y=True,
    )
    ax.set_title(
        rf"A1 max pairwise relative error at K = {K}, L = {L} "
        r"(expected $\le$ float eps)",
        fontsize=10,
    )
    fig.tight_layout()
    save_both(fig, run_dir, "a1_pairwise_errors_heatmap")
    plt.close(fig)


def _plot_error_distribution(
    cfg: A1Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Histogram of all pairwise errors across the full sweep."""
    import matplotlib.pyplot as plt

    err_R1_R2 = np.array([t["err_R1_R2"] for t in trials])
    err_R2_R3 = np.array([t["err_R2_R3"] for t in trials])
    err_R1_R3 = np.array([t["err_R1_R3"] for t in trials])
    floor = 1e-18
    e1 = np.clip(err_R1_R2, floor, None)
    e2 = np.clip(err_R2_R3, floor, None)
    e3 = np.clip(err_R1_R3, floor, None)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.logspace(-18, -8, 30)
    ax.hist(
        e1, bins=bins, alpha=0.6, label="R1 vs R2 (full vs reduced AB)",
        edgecolor="black", lw=0.4,
    )
    ax.hist(
        e2, bins=bins, alpha=0.6,
        label="R2 vs R3 (reduced AB vs reduced Γ)",
        edgecolor="black", lw=0.4,
    )
    ax.hist(
        e3, bins=bins, alpha=0.6, label="R1 vs R3 (full vs reduced Γ)",
        edgecolor="black", lw=0.4,
    )
    ax.axvline(
        cfg.machine_eps_tol, color="red", lw=0.9, ls="--",
        label=f"acceptance = {cfg.machine_eps_tol:.0e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("relative error")
    ax.set_ylabel("count")
    ax.set_title(
        f"A1 pairwise relative-error distribution over {len(trials)} cells "
        f"(D × P × K × L)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a1_error_distribution")
    plt.close(fig)


def _plot_error_vs_L(
    cfg: A1Config, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Diagnostic: how does the worst pairwise error depend on L?"""
    import matplotlib.pyplot as plt

    L_list = list(cfg.L_list)
    L_max_err = []
    for L in L_list:
        sub = [t for t in trials if int(t["L"]) == int(L)]
        if not sub:
            L_max_err.append(np.nan)
            continue
        L_max_err.append(
            max(max(t["err_R1_R2"], t["err_R2_R3"]) for t in sub)
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
    ax.set_title("A1 error growth with depth (expected ~ L-step roundoff)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a1_error_vs_L")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment A1: exact full structured model vs reduced "
            "(A_S, B_S) recursion (plan §8.1)."
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


def _config_from_cli(args: argparse.Namespace) -> A1Config:
    base = A1Config()
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
    print(f"[A1] device = {device}")

    run = ThesisRunDir(__file__, phase="theoremA")
    with RunContext(
        run,
        config=cfg,
        seeds=[cfg.base_seed, cfg.base_seed + 7,
               cfg.base_seed + 13, cfg.base_seed + 19],
        notes=(
            "A1 exact theorem-A equivalence test. Operator-level "
            "deterministic forward-pass comparison; no training, no "
            "architecture search. Three forward routes (R1: full layer "
            "simulation; R2: reduced (A_S, B_S) closed form; R3: "
            "feature-space reduced-Γ closed form) must agree to float64 "
            "machine precision in the GD-compatible setting."
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
                            f"err_R1_R2 = {trial['err_R1_R2']:.2e}  "
                            f"err_R2_R3 = {trial['err_R2_R3']:.2e}  "
                            f"err_R1_R3 = {trial['err_R1_R3']:.2e}  "
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
        err_R1_R2_grid = np.zeros(shape)
        err_R2_R3_grid = np.zeros(shape)
        err_R1_R3_grid = np.zeros(shape)
        for trial in trials:
            i = D_list.index(trial["D"])
            j = P_list.index(trial["P"])
            k = K_list.index(trial["K"])
            l = L_list.index(trial["L"])
            err_R1_R2_grid[i, j, k, l] = trial["err_R1_R2"]
            err_R2_R3_grid[i, j, k, l] = trial["err_R2_R3"]
            err_R1_R3_grid[i, j, k, l] = trial["err_R1_R3"]
        npz_payload: dict[str, np.ndarray] = {
            "D_list": np.asarray(D_list, dtype=np.int64),
            "P_list": np.asarray(P_list, dtype=np.int64),
            "K_list": np.asarray(K_list, dtype=np.int64),
            "L_list": np.asarray(L_list, dtype=np.int64),
            "err_R1_R2_grid": err_R1_R2_grid,
            "err_R2_R3_grid": err_R2_R3_grid,
            "err_R1_R3_grid": err_R1_R3_grid,
        }
        np.savez_compressed(run.npz_path("exact_equivalence"), **npz_payload)

        # --- Per-cell JSON ---
        rows = [
            {
                "D": t["D"], "P": t["P"], "K": t["K"], "L": t["L"],
                "err_R1_R2": float(t["err_R1_R2"]),
                "err_R2_R3": float(t["err_R2_R3"]),
                "err_R1_R3": float(t["err_R1_R3"]),
                "t_R1_seconds": float(t["t_R1_seconds"]),
                "t_R2_seconds": float(t["t_R2_seconds"]),
                "t_R3_seconds": float(t["t_R3_seconds"]),
            }
            for t in trials
        ]
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

        # --- Acceptance ---
        worst_R1_R2 = max(t["err_R1_R2"] for t in trials)
        worst_R2_R3 = max(t["err_R2_R3"] for t in trials)
        worst_R1_R3 = max(t["err_R1_R3"] for t in trials)
        ok_R1_R2 = worst_R1_R2 <= cfg.machine_eps_tol
        ok_R2_R3 = worst_R2_R3 <= cfg.machine_eps_tol
        all_ok = ok_R1_R2 and ok_R2_R3

        # Identify the cell with the worst overall error.
        worst_cell: dict[str, Any] | None = None
        worst_err = 0.0
        for t in trials:
            err_max = max(t["err_R1_R2"], t["err_R2_R3"])
            if err_max > worst_err:
                worst_err = err_max
                worst_cell = {
                    "D": t["D"], "P": t["P"],
                    "K": t["K"], "L": t["L"],
                    "err_R1_R2": float(t["err_R1_R2"]),
                    "err_R2_R3": float(t["err_R2_R3"]),
                }

        ctx.record_compute_proxy(float(sweep_wall))
        ctx.record_extra("worst_R1_R2", worst_R1_R2)
        ctx.record_extra("worst_R2_R3", worst_R2_R3)
        ctx.record_extra("worst_R1_R3", worst_R1_R3)
        ctx.record_extra("worst_cell", worst_cell)

        status_parts: list[str] = []
        status_parts.append(
            "R1_R2_ok" if ok_R1_R2 else
            f"R1_R2_violated(worst={worst_R1_R2:.2e})"
        )
        status_parts.append(
            "R2_R3_ok" if ok_R2_R3 else
            f"R2_R3_violated(worst={worst_R2_R3:.2e})"
        )
        status = "+".join(status_parts)

        ctx.write_summary(
            {
                "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §8.1 (A1)",
                "category": (
                    "exact theorem-A operator-level forward-pass "
                    "equivalence test. Three structurally distinct "
                    "forward routes (R1 = full L-layer linear-attention "
                    "structured-mixer simulation; R2 = sample-space "
                    "reduced (A_S, B_S) recursion; R3 = feature-space "
                    "reduced-Γ closed form) must agree to float64 "
                    "machine precision in the GD-compatible setting. "
                    "No training, no architecture search."
                ),
                "interpretation": (
                    "Theorem A asserts that, in the GD-compatible mask "
                    "regime, the L-layer fully-aligned linear-structured "
                    "forward map is an exact algebraic identity equal to "
                    "the closed-form reduced (A_S, B_S) recursion and "
                    "to its feature-space reduced-Γ special case. A1 "
                    "verifies all three routes agree to float eps over "
                    "a (D, P, K, L) sweep. Any cell exceeding the "
                    "tolerance indicates an implementation bug, not a "
                    "theorem failure (theorem A is exact in this "
                    "regime). Architecture-aligned theorem-A bridge "
                    "checks belong to §9."
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
                "worst_R1_R2": float(worst_R1_R2),
                "worst_R2_R3": float(worst_R2_R3),
                "worst_R1_R3": float(worst_R1_R3),
                "worst_cell": worst_cell,
                "sweep_wallclock_seconds": round(float(sweep_wall), 3),
            }
        )

        print()
        print("=" * 72)
        print(f" A1 exact theorem-A equivalence on {device}")
        print(f"   N cells = {len(trials)} (D × P × K × L)")
        print(
            f"   worst R1 vs R2 (full vs reduced AB)  = "
            f"{worst_R1_R2:.3e}  "
            f"{'OK' if ok_R1_R2 else 'FAIL'}  "
            f"(tol = {cfg.machine_eps_tol:.1e})"
        )
        print(
            f"   worst R2 vs R3 (reduced AB vs reduced Γ) = "
            f"{worst_R2_R3:.3e}  "
            f"{'OK' if ok_R2_R3 else 'FAIL'}  "
            f"(tol = {cfg.machine_eps_tol:.1e})"
        )
        print(
            f"   worst R1 vs R3 (full vs reduced Γ)   = "
            f"{worst_R1_R3:.3e}  (diagnostic)"
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
