"""Experiment B1: exact finite-P circulant mode closure.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §6.2.

Purpose
-------
Test the theorem-B prediction that when the training covariance is circulant,
the reduced-Γ recursion decouples *exactly* into per-Fourier-mode scalar ODEs.
Concretely, we run two recursions and require them to agree to machine
precision:

1. **Matrix recursion** in ``R^{P×P}``::

        Γ(t+1) = Γ(t) + η · Ω · Σ² · (I − L⁻¹ · Σ · Γ(t))^(2L-1)

   with Σ, Ω both real-symmetric circulant. The per-mode values are extracted
   at every step by the (unnormalized) DFT of the first column of Γ (which is
   the circulant symbol of Γ).

2. **Per-mode recursion** from :mod:`metrics`::

        γ_k(t+1) = γ_k(t) + η · ω_k · s_k² · (1 − L⁻¹ · s_k · γ_k(t))^(2L-1)

   i.e., :func:`metrics.gamma_star_trajectory_circulant`.

If the circulant diagonalization is exact, the matrix recursion's per-mode
extract must match the per-mode recursion to float eps.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``G1Config``, ``g1_generate`` (exact mode, ``query_mode='full_window'``,
    ``matched_query_realization='independent'`` by default per v4 §10.2.2).
- :mod:`scripts.thesis.utils.metrics`:
    ``gamma_star_trajectory_circulant``, ``mode_trajectory_error``,
    ``transfer_function_error``.
- :mod:`scripts.thesis.utils.fourier_ops`:
    ``off_diagonal_fourier_energy``, ``circulant_from_symbol``.
- :mod:`scripts.thesis.utils.plotting`:
    ``apply_thesis_style``, ``save_both``, ``overlay_reference``,
    ``sequential_colors``.
- :mod:`scripts.thesis.utils.run_metadata`:
    ``ThesisRunDir``, ``RunContext``.

Primary outputs (§6.2)
----------------------
- Per-trial ``mode_trajectory_error`` (matrix vs per-mode recursion).
- Per-trial ``transfer_function_error`` (empirical vs theoretical terminal
  transfer function).
- Per-trial ``off_diagonal_fourier_energy`` of the final Γ (must be ~0 since
  Γ is circulant by construction).
- Primary figure: Fourier-mode trajectories with exact theory overlays
  (Bordelon Fig 3a analogue in spectral language).
- Secondary figure: terminal transfer-function spectrum (empirical vs theory).
- Diagnostic figure: max error heat-bars across ``(P, L, symbol_kind)``.

Acceptance
----------
All three error metrics must be below ``cfg.tol_err`` (default ``1e-10``)
across every swept ``(P, L, symbol_kind)`` trial. A nonzero exit code
indicates a failure of the circulant mode-closure claim (hence either the
generator or the recursion is wrong).

Run
---
::

    python -u scripts/thesis/theoremB/run_theoremB_circulant_modes.py \\
           --device cpu --dtype float64 --no-show
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root on sys.path so ``scripts.thesis.utils.*`` resolves.
_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import matplotlib
import numpy as np
import torch

from scripts.thesis.utils.data_generators import G1Config, g1_generate
from scripts.thesis.utils.fourier_ops import (
    circulant_from_symbol,
    off_diagonal_fourier_energy,
)
from scripts.thesis.utils.metrics import (
    gamma_star_trajectory_circulant,
    mode_trajectory_error,
    transfer_function_error,
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
class B1Config:
    """Frozen configuration for the B1 circulant-modes experiment.

    Defaults produce a 3 × 4 × 3 sweep (P × L × symbol_kind) that runs to
    completion in well under a minute on CPU and exercises every theorem-B
    closure path (flat, power-law, multiband spectra).
    """

    P_list: tuple[int, ...] = (16, 32, 64)
    L_list: tuple[int, ...] = (1, 2, 4, 8)
    symbol_kinds: tuple[str, ...] = ("flat", "power_law", "multiband")
    T: int = 100
    # eta tuned to be stable across every (P, L, symbol_kind) combination in the
    # default sweep. The stability condition near the per-mode fixed point is
    # eta · omega_k · s_k^3 · (2L-1) / L < 2, and the B1 exactness test runs
    # far from the fixed point where transient overshoot is the real failure
    # mode. 1e-4 leaves ~2 orders of margin for the largest-s power-law mode.
    eta: float = 1e-4
    # Per-symbol parameters (chosen for stable B1 closure under the shared eta).
    power_law_nu: float = 0.5
    task_spec_nu_beta: float = 1.0
    multiband: tuple[tuple[int, int, float], ...] = (
        (0, 2, 1.0),
        (5, 7, 0.8),
    )
    # Query regime (v4 §10.2.2).
    query_mode: str = "full_window"
    matched_query_realization: str = "independent"
    # Acceptance tolerance for the max-scaled relative error (§6.2
    # "machine precision up to numerical integration and floating-point
    # tolerances"). 1e-10 is float64 machine-precision with a 3-4 order margin
    # for the O(P^3) matrix-multiply error accumulation over T steps.
    tol_err: float = 1e-10
    # Figure slices for the primary mode-trajectory figure.
    figure_P: int = 32
    figure_symbol: str = "power_law"
    figure_mode_indices: tuple[int, ...] = (0, 1, 2, 4, 8)
    figure_L_list: tuple[int, ...] = (1, 2, 4, 8)
    # Transfer-function figure sweep.
    transfer_P: int = 32
    transfer_L_list: tuple[int, ...] = (1, 2, 4, 8)
    transfer_symbol: str = "power_law"
    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Matrix recursion (empirical side)
# ---------------------------------------------------------------------------


def _extract_symbol_from_circulant(
    C: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Return the real symbol of a real-symmetric-circulant matrix C via the
    unnormalized DFT of its first column. O(P log P); does not assert
    off-diagonal structure (callers use :func:`off_diagonal_fourier_energy`
    separately when needed). Preserves the input device.
    """
    c = C[:, 0].to(torch.complex128)
    s = torch.fft.fft(c).real.to(dtype)
    return s


def _matrix_circulant_recursion(
    Sigma: torch.Tensor,
    Omega: torch.Tensor,
    eta: float,
    L: int,
    T: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the reduced-Γ recursion at the full matrix level:

        Γ(t+1) = Γ(t) + η · Ω · Σ² · (I − L⁻¹ · Σ · Γ(t))^(2L-1)

    from ``Γ(0) = 0``. Returns::

        gamma_traj : (T+1, P) real tensor   # diag(F Γ(t) F*) per step, via FFT
        Gamma_final: (P, P)   real tensor   # the final Γ(T)

    Both returns are on the input ``device``. Callers should ``.cpu()`` before
    saving. Here ``Σ = Sigma`` and ``Ω = Omega`` must be real-symmetric
    circulant. Since ``Γ(0) = 0`` commutes with Σ and Ω, and Σ, Ω commute
    between themselves, Γ remains circulant for all t; the off-diagonal
    Fourier energy of ``Γ_final`` is returned as a separate diagnostic.
    """
    Sigma = Sigma.to(device=device, dtype=dtype)
    Omega = Omega.to(device=device, dtype=dtype)
    P = Sigma.shape[0]
    Gamma = torch.zeros(P, P, dtype=dtype, device=device)
    I_P = torch.eye(P, dtype=dtype, device=device)
    gamma_traj = torch.zeros(T + 1, P, dtype=dtype, device=device)
    exponent = 2 * L - 1
    Sigma_sq = Sigma @ Sigma
    Omega_Sigma_sq = Omega @ Sigma_sq
    for t in range(T):
        M = I_P - (Sigma @ Gamma) / L
        M_pow = M.matrix_power(exponent)
        Gamma = Gamma + eta * (Omega_Sigma_sq @ M_pow)
        gamma_traj[t + 1] = _extract_symbol_from_circulant(Gamma, dtype)
    return gamma_traj, Gamma


def _transfer_function_matrix_and_theory(
    Sigma: torch.Tensor,
    Gamma_final: torch.Tensor,
    s_tr: torch.Tensor,
    gamma_final_theory: torch.Tensor,
    L: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Terminal transfer function, empirical vs theoretical. Inputs are moved
    to ``device`` before matrix operations; returns are on ``device``.

    Empirical::
        T_emp(k) = diag(F (I − L⁻¹ Σ Γ_final)^L F^*)
    extracted via the unnormalized-DFT-of-first-column trick.

    Theoretical::
        T_star(k) = (1 − L⁻¹ · s_tr_k · γ★_k(t=T))^L
    """
    Sigma = Sigma.to(device=device, dtype=dtype)
    Gamma_final = Gamma_final.to(device=device, dtype=dtype)
    s_tr = s_tr.to(device=device, dtype=dtype)
    gamma_final_theory = gamma_final_theory.to(device=device, dtype=dtype)
    P = Sigma.shape[0]
    I_P = torch.eye(P, dtype=dtype, device=device)
    M_final = I_P - (Sigma @ Gamma_final) / L
    T_mat = M_final.matrix_power(L)
    T_emp = _extract_symbol_from_circulant(T_mat, dtype)
    T_star = (1.0 - s_tr * gamma_final_theory / L).pow(L)
    return T_emp, T_star


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def _build_g1_config(cfg: B1Config, P: int, symbol_kind: str) -> G1Config:
    """Construct the G1Config for a given (P, symbol_kind) trial.

    - ``exact_mode=True`` (v4 §10.2 full circulant exact path).
    - ``query_mode='full_window'`` with K = P (v4 §10.2.2).
    - ``matched_query_realization='independent'`` by default (v4 §10.2.2 binding).
    """
    if symbol_kind == "power_law":
        symbol_params: dict[str, Any] = {"nu": cfg.power_law_nu}
    elif symbol_kind == "multiband":
        symbol_params = {"bands": list(cfg.multiband)}
    elif symbol_kind == "flat":
        symbol_params = {"value": 1.0}
    else:
        raise ValueError(f"unknown symbol_kind: {symbol_kind!r}")

    return G1Config(
        P=P,
        B=1,
        query_mode=cfg.query_mode,
        matched_query_realization=cfg.matched_query_realization,
        symbol_kind_tr=symbol_kind,
        symbol_params_tr=symbol_params,
        symbol_kind_te="matched",
        symbol_params_te={},
        task_spec_kind="power_law",
        task_spec_params={"nu_beta": cfg.task_spec_nu_beta},
        sigma=0.0,
        label_norm="sqrt_P",
        exact_mode=True,
        sample_data=False,  # operator-level only; no X / y needed
        population_mode=False,
        dtype=cfg.dtype,
    )


def _run_trial(
    cfg: B1Config, P: int, L: int, symbol_kind: str, device: torch.device
) -> dict[str, Any]:
    """Run one ``(P, L, symbol_kind)`` closure trial. Returns a result dict
    with raw trajectories and the three primary metrics.
    """
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    g1_cfg = _build_g1_config(cfg, P, symbol_kind)
    op = g1_generate(g1_cfg)
    s_tr = op["s_tr"]
    omega = op["omega"]
    Sigma_tr = op["Sigma_tr"]
    # Ω is circulant with symbol ω (same construction).
    Omega_phys = circulant_from_symbol(omega)

    # Empirical matrix recursion (on device).
    t0 = time.perf_counter()
    gamma_emp_dev, Gamma_final_dev = _matrix_circulant_recursion(
        Sigma_tr, Omega_phys, cfg.eta, L, cfg.T, dtype, device
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_matrix = time.perf_counter() - t0

    # Theoretical per-mode recursion (cheap, keep on CPU float64).
    t0 = time.perf_counter()
    gamma_star = gamma_star_trajectory_circulant(
        s_tr, omega, L=L, eta=cfg.eta, T=cfg.T
    )
    t_theory = time.perf_counter() - t0

    # Bring matrix-side trajectories back to CPU for comparison / saving.
    gamma_emp = gamma_emp_dev.detach().cpu()

    # --- Primary metrics ---
    # Absolute max error between matrix recursion and per-mode recursion.
    abs_err = (gamma_emp - gamma_star).abs()
    mode_abs_err_max = float(abs_err.max().item())
    # Max-magnitude-scaled relative error. This avoids the pathology of
    # mode_trajectory_error's per-element |emp-star| / (|star| + eps) blowing
    # up where |γ★_k| is near zero but |emp_k| is also tiny (float noise).
    # The max-magnitude scale is the correct notion for a "machine-precision
    # trajectory closure" test (v4 spec / plan §6.2).
    gs_mag = float(gamma_star.abs().max().item())
    mode_rel_err_max = mode_abs_err_max / (gs_mag + 1e-30)
    # Also report the unscaled pointwise metric from metrics.mode_trajectory_error
    # as a diagnostic (large when γ★ is near-zero; not used for acceptance).
    mode_traj_err = mode_trajectory_error(gamma_emp, gamma_star)
    mode_traj_err_max = float(mode_traj_err.max().item())
    mode_traj_err_median = float(mode_traj_err.median().item())

    T_emp_dev, T_star_dev = _transfer_function_matrix_and_theory(
        Sigma_tr, Gamma_final_dev, s_tr, gamma_star[-1], L, dtype, device
    )
    T_emp = T_emp_dev.detach().cpu()
    T_star = T_star_dev.detach().cpu()
    T_err_l2 = transfer_function_error(T_emp, T_star)
    T_err_abs_max = float((T_emp - T_star).abs().max().item())
    T_mag = float(T_star.abs().max().item())
    T_err_rel = T_err_abs_max / (T_mag + 1e-30)

    ode = off_diagonal_fourier_energy(Gamma_final_dev.detach().cpu())

    return {
        "P": int(P),
        "L": int(L),
        "symbol_kind": symbol_kind,
        "T": cfg.T,
        "eta": cfg.eta,
        "device": str(device),
        "s_tr": s_tr.detach().cpu(),
        "omega": omega.detach().cpu(),
        "gamma_emp": gamma_emp,
        "gamma_star": gamma_star,
        "T_emp": T_emp,
        "T_star": T_star,
        # Primary acceptance metrics (max-magnitude-scaled).
        "mode_abs_err_max": mode_abs_err_max,
        "mode_rel_err_max": mode_rel_err_max,
        "transfer_abs_err_max": T_err_abs_max,
        "transfer_rel_err_max": T_err_rel,
        "off_diagonal_fourier_energy": float(ode),
        # Diagnostic metrics.
        "mode_traj_err_max": mode_traj_err_max,
        "mode_traj_err_median": mode_traj_err_median,
        "transfer_err_l2": float(T_err_l2),
        "gamma_star_magnitude_max": gs_mag,
        "transfer_star_magnitude_max": T_mag,
        "matrix_recursion_seconds": float(t_matrix),
        "theory_recursion_seconds": float(t_theory),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_mode_trajectories(
    trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Primary figure (Bordelon Fig 3a spectral analogue): per-Fourier-mode
    trajectories γ_k(t) over several depths L, with exact theory overlays.
    """
    import matplotlib.pyplot as plt

    slice_trials = [
        t for t in trials
        if t["P"] == cfg.figure_P and t["symbol_kind"] == cfg.figure_symbol
        and t["L"] in cfg.figure_L_list
    ]
    slice_trials.sort(key=lambda t: t["L"])
    if not slice_trials:
        return

    n_L = len(slice_trials)
    fig, axes = plt.subplots(1, n_L, figsize=(4.0 * n_L, 3.8), sharey=True)
    if n_L == 1:
        axes = [axes]
    mode_colors = sequential_colors(len(cfg.figure_mode_indices), palette="rocket")
    t_axis = np.arange(cfg.T + 1)
    for ax, trial in zip(axes, slice_trials):
        for color, k in zip(mode_colors, cfg.figure_mode_indices):
            emp = trial["gamma_emp"][:, k].numpy()
            thy = trial["gamma_star"][:, k].numpy()
            ax.plot(t_axis, emp, color=color, lw=1.4,
                    label=f"empirical k={k}", alpha=0.9)
            overlay_reference(
                ax, t_axis, thy, label=None, style="--", color="black", lw=1.0
            )
        ax.set_title(f"L = {trial['L']}", fontsize=11)
        ax.set_xlabel("step t")
        ax.set_xscale("log")
    axes[0].set_ylabel(r"$\gamma_k(t)$")
    # Legend only on the last axis.
    axes[-1].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        f"B1 mode trajectories ({cfg.figure_symbol}, P={cfg.figure_P}); "
        "dashed black = exact theory",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "mode_trajectories")
    plt.close(fig)


def _plot_mode_trajectories_loglog(
    trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Thesis-facing log-log variant of the mode-trajectory figure.

    We drop the ``t=0`` point because ``gamma_k(0)=0`` exactly, and a log y-axis
    requires strictly positive values. Any nonpositive values beyond ``t=0`` are
    masked so the figure remains well-defined if a future sweep explores an
    unstable parameter regime.
    """
    import matplotlib.pyplot as plt

    slice_trials = [
        t for t in trials
        if t["P"] == cfg.figure_P and t["symbol_kind"] == cfg.figure_symbol
        and t["L"] in cfg.figure_L_list
    ]
    slice_trials.sort(key=lambda t: t["L"])
    if not slice_trials:
        return

    n_L = len(slice_trials)
    fig, axes = plt.subplots(1, n_L, figsize=(4.0 * n_L, 3.8), sharey=True)
    if n_L == 1:
        axes = [axes]
    mode_colors = sequential_colors(len(cfg.figure_mode_indices), palette="rocket")
    t_axis = np.arange(1, cfg.T + 1, dtype=float)
    for ax, trial in zip(axes, slice_trials):
        for color, k in zip(mode_colors, cfg.figure_mode_indices):
            emp = trial["gamma_emp"][1:, k].numpy()
            thy = trial["gamma_star"][1:, k].numpy()
            emp = np.where(emp > 0.0, emp, np.nan)
            thy = np.where(thy > 0.0, thy, np.nan)
            ax.plot(t_axis, emp, color=color, lw=1.4,
                    label=f"empirical k={k}", alpha=0.9)
            overlay_reference(
                ax, t_axis, thy, label=None, style="--", color="black", lw=1.0
            )
        ax.set_title(f"L = {trial['L']}", fontsize=11)
        ax.set_xlabel("step t")
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].set_ylabel(r"$\gamma_k(t)$")
    axes[-1].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        f"B1 mode trajectories, log-log ({cfg.figure_symbol}, P={cfg.figure_P}); "
        "dashed black = exact theory",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "mode_trajectories_loglog")
    plt.close(fig)


def _plot_transfer_function(
    trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Secondary figure: overlaid terminal transfer-function depth sweep."""
    import matplotlib.pyplot as plt

    slice_trials = [
        t for t in trials
        if t["P"] == cfg.transfer_P
        and t["symbol_kind"] == cfg.transfer_symbol
        and t["L"] in cfg.transfer_L_list
    ]
    slice_trials.sort(key=lambda t: t["L"])
    if not slice_trials:
        return

    P = slice_trials[0]["P"]
    k_axis = np.arange(P)
    depth_colors = sequential_colors(len(slice_trials), palette="rocket")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for color, trial in zip(depth_colors, slice_trials):
        ax.plot(
            k_axis,
            trial["T_emp"].numpy(),
            color=color,
            lw=1.5,
            label=f"empirical L={trial['L']}",
        )
        overlay_reference(
            ax,
            k_axis,
            trial["T_star"].numpy(),
            label=f"theory L={trial['L']}",
            style="--",
            color=color,
            lw=1.0,
        )
    ax.set_xlabel("mode index k")
    ax.set_ylabel(r"$T(k) = (1 - L^{-1} s_k \gamma_k)^L$")
    fig.suptitle(
        f"B1 terminal transfer function depth sweep, overlaid "
        f"({cfg.transfer_symbol}, P={P})",
        fontsize=11,
    )
    ax.legend(fontsize=8, frameon=True, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_both(fig, run_dir, "transfer_function")
    plt.close(fig)


def _plot_error_summary(
    trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Diagnostic figure: max mode / transfer / off-diagonal error per trial.

    Uses the max-magnitude-scaled relative errors (the acceptance metrics) so
    the y-axis is comparable to the v4 "machine precision" tolerance.
    """
    import matplotlib.pyplot as plt

    labels = [
        f"P={t['P']} L={t['L']} {t['symbol_kind'][:4]}" for t in trials
    ]
    mode_errs = np.array([t["mode_rel_err_max"] for t in trials])
    transfer_errs = np.array([t["transfer_rel_err_max"] for t in trials])
    ode_errs = np.array([t["off_diagonal_fourier_energy"] for t in trials])
    floor = 1e-18  # for log-scale plotting of zeros

    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(trials)), 4.2))
    x = np.arange(len(trials))
    ax.scatter(x, np.maximum(mode_errs, floor), s=24,
               label="mode rel err (max-magnitude-scaled)", color="C0")
    ax.scatter(x, np.maximum(transfer_errs, floor), s=24, marker="^",
               label="transfer rel err (max-magnitude-scaled)", color="C1")
    ax.scatter(x, np.maximum(ode_errs, floor), s=24, marker="s",
               label="off-diag Fourier energy", color="C2")
    ax.axhline(cfg.tol_err, color="red", lw=0.8, ls="--",
               label=f"tolerance ({cfg.tol_err:.0e})")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, fontsize=7)
    ax.set_ylabel("error (log scale)")
    ax.set_title("B1 exact-closure errors across (P, L, symbol) trials", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "error_summary")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_list_strs(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment B1: exact finite-P circulant mode closure "
            "(v4 spec / plan §6.2)."
        )
    )
    p.add_argument("--device", type=str, default="cpu",
                   choices=("cpu", "cuda", "auto"))
    p.add_argument("--dtype", type=str, default="float64",
                   choices=("float32", "float64"))
    p.add_argument("--no-show", action="store_true",
                   help="suppress matplotlib display (headless use)")
    p.add_argument("--P-list", type=str, default=None,
                   help="comma-separated P values; default 16,32,64")
    p.add_argument("--L-list", type=str, default=None,
                   help="comma-separated L values; default 1,2,4,8")
    p.add_argument("--symbol-kinds", type=str, default=None,
                   help="comma-separated symbol kinds from {flat,power_law,multiband}")
    p.add_argument("--T", type=int, default=None,
                   help="trajectory length (steps of the recursion)")
    p.add_argument("--eta", type=float, default=None, help="learning rate")
    p.add_argument("--tol-err", type=float, default=None,
                   help="acceptance tolerance for every error metric")
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> B1Config:
    base = B1Config()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.P_list is not None:
        overrides["P_list"] = _parse_list_ints(args.P_list)
    if args.L_list is not None:
        overrides["L_list"] = _parse_list_ints(args.L_list)
    if args.symbol_kinds is not None:
        overrides["symbol_kinds"] = _parse_list_strs(args.symbol_kinds)
    if args.T is not None:
        overrides["T"] = int(args.T)
    if args.eta is not None:
        overrides["eta"] = float(args.eta)
    if args.tol_err is not None:
        overrides["tol_err"] = float(args.tol_err)
    from dataclasses import replace

    return replace(base, **overrides) if overrides else base


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Source starter.sh in an environment with CUDA, or pass --device cpu."
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
    print(f"[B1] device = {device}")
    run = ThesisRunDir(__file__, phase="theoremB")
    with RunContext(run, config=cfg, seeds=[0, 1, 2, 3]) as ctx:
        apply_thesis_style()

        # Main sweep.
        trials: list[dict[str, Any]] = []
        n_total = len(cfg.P_list) * len(cfg.L_list) * len(cfg.symbol_kinds)
        t_sweep_start = time.perf_counter()
        idx = 0
        for P in cfg.P_list:
            for symbol_kind in cfg.symbol_kinds:
                for L in cfg.L_list:
                    idx += 1
                    t0 = time.perf_counter()
                    trial = _run_trial(cfg, P, L, symbol_kind, device)
                    dt = time.perf_counter() - t0
                    ctx.record_step_time(dt)
                    print(
                        f"[{idx:>3d}/{n_total}] "
                        f"P={P:>3d} L={L:>2d} {symbol_kind:<10s} "
                        f"rel_err={trial['mode_rel_err_max']:.2e} "
                        f"T_rel={trial['transfer_rel_err_max']:.2e} "
                        f"off_diag={trial['off_diagonal_fourier_energy']:.2e} "
                        f"({dt*1000:6.1f} ms)"
                    )
                    trials.append(trial)
        t_sweep = time.perf_counter() - t_sweep_start

        # Acceptance check: each max-magnitude-scaled relative error must be
        # below tol_err. These are the primary B1 metrics per §6.2.
        tol = cfg.tol_err
        failures: list[dict[str, Any]] = []
        for trial in trials:
            failed = []
            if trial["mode_rel_err_max"] > tol:
                failed.append(("mode_rel", trial["mode_rel_err_max"]))
            if trial["transfer_rel_err_max"] > tol:
                failed.append(("transfer_rel", trial["transfer_rel_err_max"]))
            if trial["off_diagonal_fourier_energy"] > tol:
                failed.append(("off_diag", trial["off_diagonal_fourier_energy"]))
            if failed:
                failures.append({
                    "P": trial["P"], "L": trial["L"],
                    "symbol_kind": trial["symbol_kind"],
                    "metrics_failed": failed,
                })

        # Save raw per-trial data to npz + pt.
        npz_path = run.npz_path("circulant_modes")
        np.savez_compressed(
            npz_path,
            **{
                f"P{t['P']}_L{t['L']}_{t['symbol_kind']}__gamma_emp":
                    t["gamma_emp"].numpy()
                for t in trials
            },
            **{
                f"P{t['P']}_L{t['L']}_{t['symbol_kind']}__gamma_star":
                    t["gamma_star"].numpy()
                for t in trials
            },
            **{
                f"P{t['P']}_L{t['L']}_{t['symbol_kind']}__T_emp":
                    t["T_emp"].numpy()
                for t in trials
            },
            **{
                f"P{t['P']}_L{t['L']}_{t['symbol_kind']}__T_star":
                    t["T_star"].numpy()
                for t in trials
            },
        )

        summary_rows = [
            {
                "P": t["P"],
                "L": t["L"],
                "symbol_kind": t["symbol_kind"],
                "mode_abs_err_max": t["mode_abs_err_max"],
                "mode_rel_err_max": t["mode_rel_err_max"],
                "transfer_abs_err_max": t["transfer_abs_err_max"],
                "transfer_rel_err_max": t["transfer_rel_err_max"],
                "transfer_err_l2": t["transfer_err_l2"],
                "off_diagonal_fourier_energy": t["off_diagonal_fourier_energy"],
                "mode_traj_err_max_unscaled": t["mode_traj_err_max"],
                "mode_traj_err_median_unscaled": t["mode_traj_err_median"],
                "gamma_star_magnitude_max": t["gamma_star_magnitude_max"],
                "transfer_star_magnitude_max": t["transfer_star_magnitude_max"],
                "matrix_recursion_seconds": t["matrix_recursion_seconds"],
                "theory_recursion_seconds": t["theory_recursion_seconds"],
            }
            for t in trials
        ]
        summary_json_path = run.root / "per_trial_summary.json"
        summary_json_path.write_text(
            json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8"
        )

        # Figures.
        _plot_mode_trajectories(trials, cfg, run)
        _plot_mode_trajectories_loglog(trials, cfg, run)
        _plot_transfer_function(trials, cfg, run)
        _plot_error_summary(trials, cfg, run)

        # Aggregate metrics for summary.txt and metadata.
        max_mode_rel = max(t["mode_rel_err_max"] for t in trials)
        max_mode_abs = max(t["mode_abs_err_max"] for t in trials)
        max_transfer_rel = max(t["transfer_rel_err_max"] for t in trials)
        max_transfer_abs = max(t["transfer_abs_err_max"] for t in trials)
        max_ode = max(t["off_diagonal_fourier_energy"] for t in trials)
        median_mode_rel = float(
            np.median([t["mode_rel_err_max"] for t in trials])
        )
        compute_proxy_scalar = float(t_sweep)  # wall-clock proxy for this op-level run
        ctx.record_compute_proxy(compute_proxy_scalar)
        ctx.record_extra("n_trials", len(trials))
        ctx.record_extra("n_failures", len(failures))
        ctx.record_extra("failures", failures[:20])
        ctx.record_extra("device", str(device))
        ctx.record_extra(
            "max_errors",
            {
                "mode_rel_err_max": max_mode_rel,
                "mode_abs_err_max": max_mode_abs,
                "transfer_rel_err_max": max_transfer_rel,
                "transfer_abs_err_max": max_transfer_abs,
                "off_diagonal_fourier_energy": max_ode,
            },
        )

        ctx.write_summary(
            {
                "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §6.2 (B1)",
                "device": str(device),
                "acceptance_tolerance": tol,
                "n_trials": len(trials),
                "n_failures": len(failures),
                "status": "closure_ok" if not failures else "closure_failed",
                "max_mode_rel_err": max_mode_rel,
                "max_mode_abs_err": max_mode_abs,
                "median_mode_rel_err": median_mode_rel,
                "max_transfer_rel_err": max_transfer_rel,
                "max_transfer_abs_err": max_transfer_abs,
                "max_off_diagonal_fourier_energy": max_ode,
                "sweep_wallclock_seconds": round(t_sweep, 3),
            }
        )

        print()
        print("=" * 72)
        print(f" B1 closure test: {len(trials)} trials on {device}")
        print(f"   max mode_rel_err            = {max_mode_rel:.3e}")
        print(f"   max mode_abs_err            = {max_mode_abs:.3e}")
        print(f"   max transfer_rel_err        = {max_transfer_rel:.3e}")
        print(f"   max transfer_abs_err        = {max_transfer_abs:.3e}")
        print(f"   max off-diagonal Fourier    = {max_ode:.3e}")
        print(f"   tolerance                   = {tol:.1e}")
        if failures:
            print(f"   FAILED trials: {len(failures)}")
            for f in failures[:5]:
                print(f"     P={f['P']} L={f['L']} {f['symbol_kind']}: "
                      f"{f['metrics_failed']}")
        else:
            print(f"   all trials within tolerance")
        print("=" * 72)

        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
