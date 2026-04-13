"""Experiment B2: long-context depth irrelevance in the matched stationary regime.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §6.3.

Purpose
-------
Test the theorem-B prediction that, in the matched-symbol stationary regime
(``s_tr == s_te``; G1 population mode), **increasing spectral depth beyond a
shallow baseline does not improve the asymptotic reduced-Γ loss**. Depth may
still affect *transient* optimization rates (the approach to the stationary
optimum), but the t→∞ floor collapses onto the matched stationary asymptote
for every ``L``.

Mathematically, at the matched fixed point ``γ_k = L / s_k`` we have
``(1 − L⁻¹ s_k γ_k) = 0``, so the per-mode transfer vanishes and the
stationary loss is zero for every ``L``. The figure therefore shows a family
of loss-vs-time curves at several depths, all decaying toward the common zero
floor but at L-dependent rates (L=1 exponential; L>1 polynomial of order
``−1/(2L−2)``).

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``G1Config``, ``g1_generate`` (exact + population mode, matched symbols).
- :mod:`scripts.thesis.utils.metrics`:
    ``gamma_star_trajectory_circulant`` — single source of the per-mode
    recursion. B1 validated at machine precision that this equals the matrix
    recursion, so B2 uses the per-mode recursion directly.
- :mod:`scripts.thesis.utils.plotting`:
    ``apply_thesis_style``, ``save_both``, ``sequential_colors``,
    ``overlay_reference``.
- :mod:`scripts.thesis.utils.run_metadata`:
    ``ThesisRunDir``, ``RunContext``.

Primary outputs (§6.3)
----------------------
- Primary figure (Bordelon Fig 3b spectral analogue): ``loss_vs_time`` —
  reduced-Γ loss L(t) at several depths, log-log, at the large-context figure
  slice (``figure_P``, ``figure_symbol``).
- Secondary figure: ``final_loss_vs_depth`` — loss at several snapshots
  ``t ∈ {T_final, T_final//4, T_final//16}`` as a function of L. Early
  snapshots are L-dependent (transient); late snapshots collapse onto the
  matched stationary asymptote.
- Diagnostic figure: ``long_context_collapse`` — loss L(t) at the shallow
  baseline L=1 across ``P_list``, showing long-context convergence.
- Diagnostic figure: ``per_mode_residuals`` — terminal residual transfer
  spectrum ``(1 − L⁻¹ s_k γ_k(T))^(2L)`` per mode at each L.

Interpretation (finite-time framing)
------------------------------------
**B2 is a finite-time matched-stationary depth-irrelevance experiment, not a
claim that every depth reaches numerical zero within the compute budget.**
What the figures are designed to show:

- No evidence of a depth-dependent *asymptotic* floor. Every depth's loss
  trajectory continues to decay toward zero throughout the observed horizon;
  no plateau emerges at a finite nonzero value for any ``L``.
- Finite-T cross-L differences are *transient-rate* effects. ``L = 1`` decays
  exponentially near the fixed point, while ``L > 1`` decays polynomially at
  rate ``t^{-1/(2L-2)}``. Within any finite budget ``T`` the deeper model's
  loss is necessarily larger than ``L = 1``'s — a difference that vanishes
  as ``t → ∞``, not a depth-dependent floor.

The acceptance tests below encode the *operational* form of this framing;
neither is a tolerance against a machine-precision target.

Acceptance
----------
This is not a closure test; unlike B1 there is no machine-precision tolerance.
Instead we check two qualitative properties:

1. **Monotonicity**: every depth's loss trajectory is monotonically
   nonincreasing (the matched recursion is contractive toward the fixed
   point). No spurious oscillation or overshoot.
2. **Per-trial decay**: each trial's terminal loss must fall below a fixed
   fraction of its initial (``γ = 0``) loss. This is the operational form
   of "every depth can reach the matched asymptote": shallow and deep
   alike show substantial decay. We deliberately avoid a cross-L ratio
   threshold because ``L = 1`` often reaches float-eps while deeper ``L``
   is still visibly positive — a finite-T transient difference, not a
   depth-dependent floor.

Both acceptance checks run automatically and are surfaced in ``summary.txt``.

Run
---
::

    python -u scripts/thesis/theoremB/run_theoremB_depth_stationary.py \\
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

# Project root on sys.path so ``scripts.thesis.utils.*`` resolves.
_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import matplotlib
import numpy as np
import torch

from scripts.thesis.utils.data_generators import G1Config, g1_generate
from scripts.thesis.utils.metrics import gamma_star_trajectory_circulant
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
class B2Config:
    """Frozen configuration for the B2 depth-irrelevance experiment.

    Default sweep is split into two parts for symbol-stability reasons (both
    power_law and multiband generators normalize their symbol to ``mean = 1``,
    so the peak amplitude ``max(s_k)`` grows with ``P``; multiband scales
    especially unfavorably since only a few modes are active):

    - **Main sweep** (both symbols): ``P_list × symbol_kinds × L_list``. Uses
      ``P`` small enough that the shared ``eta`` is safely below the stability
      boundary ``η · ω_k · s_k^3 · (2L−1)/L < 2`` at every L, for *both*
      symbols.
    - **Long-context sweep** (``long_context_symbol`` only, power_law by
      default which scales more benignly with P):
      ``long_context_P_list × L_list``. Demonstrates the long-context
      convergence promised by §6.3.

    Total default trials: 2 P × 2 symbols × 5 L + 2 P × 1 symbol × 5 L = 30.
    """

    # Main sweep (both symbols at moderate context so the shared eta is safe
    # for the ill-conditioned multiband generator).
    P_list: tuple[int, ...] = (32, 64)
    L_list: tuple[int, ...] = (1, 2, 4, 8, 16)
    symbol_kinds: tuple[str, ...] = ("power_law", "multiband")

    # Long-context sub-sweep (power_law only; multiband peak amplitude grows
    # like P because it concentrates mass in a fixed number of modes).
    long_context_P_list: tuple[int, ...] = (128, 256)
    long_context_symbol: str = "power_law"

    # Recursion horizon and step size, shared across all trials so the time
    # axis is directly comparable. eta = 5e-5 keeps max(η · ω · s^3 · (2L−1)/L)
    # ≲ 0.6 at the worst case (P=256 power_law, L=16; stability bound is < 2)
    # — about 3× safety margin. T = 100000 ⇒ η·T = 5, enough for L=1 modes
    # with ω_k · s_k^3 ≳ 1 to decay to O(e^{-5}) while the slower, high-k
    # tail is dominated by the (L-independent) ω_k · s_k initial contribution.
    T: int = 100000
    eta: float = 5e-5

    # Symbol parameters (matched for B2: s_te = s_tr by ``symbol_kind_te``).
    power_law_nu: float = 0.5
    task_spec_nu_beta: float = 1.0
    multiband: tuple[tuple[int, int, float], ...] = (
        (0, 2, 1.0),
        (5, 7, 0.8),
    )

    # Query regime (v4 §10.2.2).
    query_mode: str = "full_window"
    matched_query_realization: str = "independent"

    # Figure slices.
    # The primary loss-vs-time figure uses the largest main-sweep P so both
    # symbols are available.
    figure_P: int = 64
    figure_L_list: tuple[int, ...] = (1, 2, 4, 8, 16)
    snapshot_fractions: tuple[float, ...] = (1 / 64, 1 / 16, 1 / 4, 1.0)
    # long_context_L_list may be a subset of L_list (the long-context sub-
    # sweep runs all L_list but the figure can truncate for legibility).
    long_context_L: int = 1

    # Acceptance thresholds (qualitative; see module docstring).
    monotonicity_slack: float = 1e-9
    # Every trial's terminal loss must drop below this fraction of its
    # initial (γ=0) loss. This is the operational form of "every depth can
    # reach the matched stationary asymptote": shallow and deep alike show
    # substantial decay toward the common zero floor. The metric is
    # per-trial (not a cross-L ratio) because the L=1 exponential decay
    # often reaches float-eps while L=16 polynomial decay is still visibly
    # above, which would produce an uninformatively-large cross-L ratio
    # without indicating any depth-dependent floor.
    depth_decay_fraction: float = 0.2

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Loss helper — matched stationary, circulant
# ---------------------------------------------------------------------------


def _matched_stationary_loss(
    s: torch.Tensor, omega: torch.Tensor, gamma_traj: torch.Tensor, L: int
) -> torch.Tensor:
    """Per-step matched stationary loss

        L(t) = Σ_k  ω_k · s_k · (1 − L⁻¹ · s_k · γ_k(t))^(2L)

    Matches the diagonal evaluation of ``Tr[Ω Σ (I − L⁻¹ Σ Γ)^(2L)]`` under
    the circulant diagonalization with identical train/test symbols. Inputs
    are real ``float64`` CPU tensors; ``gamma_traj`` has shape ``(T+1, P)``.
    Returns a ``(T+1,)`` tensor of per-step scalar losses.
    """
    if gamma_traj.ndim != 2 or gamma_traj.shape[1] != s.shape[0]:
        raise ValueError(
            f"gamma_traj must be (T+1, P) with P={s.shape[0]}; "
            f"got shape {tuple(gamma_traj.shape)}"
        )
    s64 = s.to(torch.float64)
    w64 = omega.to(torch.float64)
    residual = 1.0 - (s64.unsqueeze(0) * gamma_traj) / int(L)
    transfer_sq = residual.pow(2 * int(L))
    per_mode = w64.unsqueeze(0) * s64.unsqueeze(0) * transfer_sq
    return per_mode.sum(dim=1)


def _matched_stationary_loss_initial(
    s: torch.Tensor, omega: torch.Tensor
) -> float:
    """Closed form at γ=0: Σ_k ω_k · s_k · 1 = <ω, s>. Used as a reference
    level and for monotonicity sanity checks.
    """
    return float((omega.to(torch.float64) * s.to(torch.float64)).sum().item())


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def _build_g1_config(cfg: B2Config, P: int, symbol_kind: str) -> G1Config:
    """Build the G1Config for a B2 trial.

    - ``exact_mode=True`` (full circulant exact path).
    - ``population_mode=True`` (no sample-complexity bottleneck; semantically
      the large-context matched stationary regime).
    - ``sample_data=False`` (B2 is operator-level).
    - ``symbol_kind_te='matched'`` (B2 is the matched stationary regime).
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
        sample_data=False,
        population_mode=True,
        dtype=cfg.dtype,
    )


def _run_trial(
    cfg: B2Config, P: int, L: int, symbol_kind: str
) -> dict[str, Any]:
    """Run one (P, L, symbol_kind) trial. Returns a dict with the per-mode
    γ trajectory, the induced scalar loss trajectory, and summary metrics.
    """
    g1_cfg = _build_g1_config(cfg, P, symbol_kind)
    op = g1_generate(g1_cfg)
    s_tr = op["s_tr"]
    omega = op["omega"]

    t0 = time.perf_counter()
    gamma_traj = gamma_star_trajectory_circulant(
        s_tr, omega, L=L, eta=cfg.eta, T=cfg.T
    )
    t_rec = time.perf_counter() - t0

    loss_traj = _matched_stationary_loss(s_tr, omega, gamma_traj, L)
    loss_init = _matched_stationary_loss_initial(s_tr, omega)
    loss0 = float(loss_traj[0].item())
    loss_final = float(loss_traj[-1].item())

    # Monotonicity diagnostic: loss should be non-increasing; we allow a
    # small absolute slack because the per-step update has L-dependent stencil
    # and numerical roundoff can create tiny positive differences.
    diffs = loss_traj[1:] - loss_traj[:-1]
    max_monotonicity_violation = float(diffs.max().clamp_min(0.0).item())

    # Per-mode terminal residual transfer spectrum (diagnostic).
    final_residual = 1.0 - s_tr.to(torch.float64) * gamma_traj[-1] / int(L)
    final_transfer_sq = final_residual.pow(2 * int(L))

    return {
        "P": int(P),
        "L": int(L),
        "symbol_kind": symbol_kind,
        "T": int(cfg.T),
        "eta": float(cfg.eta),
        "s_tr": s_tr.detach().cpu(),
        "omega": omega.detach().cpu(),
        "gamma_traj": gamma_traj.detach().cpu(),
        "loss_traj": loss_traj.detach().cpu(),
        "final_residual": final_residual.detach().cpu(),
        "final_transfer_sq": final_transfer_sq.detach().cpu(),
        "loss_analytic_initial": loss_init,
        "loss_initial": loss0,
        "loss_final": loss_final,
        "monotonicity_violation_max": max_monotonicity_violation,
        "recursion_seconds": float(t_rec),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _select(
    trials: list[dict[str, Any]],
    *,
    P: int | None = None,
    symbol_kind: str | None = None,
    L: int | None = None,
    L_in: tuple[int, ...] | None = None,
    P_in: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for t in trials:
        if P is not None and t["P"] != P:
            continue
        if symbol_kind is not None and t["symbol_kind"] != symbol_kind:
            continue
        if L is not None and t["L"] != L:
            continue
        if L_in is not None and t["L"] not in L_in:
            continue
        if P_in is not None and t["P"] not in P_in:
            continue
        out.append(t)
    return out


def _plot_loss_vs_time(
    trials: list[dict[str, Any]], cfg: B2Config, run_dir: ThesisRunDir
) -> None:
    """Primary B2 figure: loss-vs-time curves at several depths, large context.

    One subplot per symbol in ``cfg.symbol_kinds``, fixed at ``cfg.figure_P``.
    """
    import matplotlib.pyplot as plt

    valid_subplots = [
        (sym, cfg.figure_P)
        for sym in cfg.symbol_kinds
        if cfg.figure_P in cfg.P_list
    ]
    if not valid_subplots:
        return

    n_sub = len(valid_subplots)
    fig, axes = plt.subplots(
        1, n_sub, figsize=(4.8 * n_sub, 3.8), sharey=True
    )
    if n_sub == 1:
        axes = [axes]
    L_colors = sequential_colors(len(cfg.figure_L_list), palette="rocket")
    t_axis = np.arange(1, cfg.T + 1, dtype=float)
    for ax, (sym, P) in zip(axes, valid_subplots):
        slice_trials = _select(
            trials, P=P, symbol_kind=sym, L_in=cfg.figure_L_list
        )
        slice_trials.sort(key=lambda t: t["L"])
        for color, trial in zip(L_colors, slice_trials):
            loss = trial["loss_traj"][1:].numpy()
            loss = np.where(loss > 0.0, loss, np.nan)
            ax.plot(
                t_axis, loss, color=color, lw=1.4,
                label=f"L = {trial['L']}", alpha=0.95,
            )
        if slice_trials:
            loss0 = slice_trials[0]["loss_analytic_initial"]
            overlay_reference(
                ax, t_axis, np.full_like(t_axis, loss0),
                label=r"$\mathcal{L}(\gamma=0)$",
                style=":", color="gray", lw=1.0,
            )
        ax.set_title(f"{sym}, P = {P}", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("step t")
    axes[0].set_ylabel(r"matched stationary loss $\mathcal{L}(t)$")
    axes[-1].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        "B2 matched stationary loss, loss-vs-time across depths (Bordelon "
        "Fig 3b analogue)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_both(fig, run_dir, "loss_vs_time")
    plt.close(fig)


def _plot_final_loss_vs_depth(
    trials: list[dict[str, Any]], cfg: B2Config, run_dir: ThesisRunDir
) -> None:
    """Secondary B2 figure: loss at several snapshots vs depth. Early
    snapshots are L-dependent (transient); late snapshots collapse onto the
    matched stationary asymptote."""
    import matplotlib.pyplot as plt

    T = cfg.T
    snapshot_fracs = cfg.snapshot_fractions
    # Snapshot indices within [1, T].
    snapshot_idx = [max(1, int(round(f * T))) for f in snapshot_fracs]

    subplots = [
        (sym, cfg.figure_P)
        for sym in cfg.symbol_kinds
    ]
    valid_subplots = [
        (sym, P) for sym, P in subplots if P in cfg.P_list
    ]
    if not valid_subplots:
        return

    n_sub = len(valid_subplots)
    fig, axes = plt.subplots(
        1, n_sub, figsize=(4.8 * n_sub, 3.8), sharey=True
    )
    if n_sub == 1:
        axes = [axes]
    snap_colors = sequential_colors(len(snapshot_idx), palette="rocket")
    for ax, (sym, P) in zip(axes, valid_subplots):
        slice_trials = _select(
            trials, P=P, symbol_kind=sym, L_in=cfg.L_list
        )
        slice_trials.sort(key=lambda t: t["L"])
        Ls = np.array([t["L"] for t in slice_trials], dtype=float)
        for color, frac, t_idx in zip(snap_colors, snapshot_fracs, snapshot_idx):
            losses = np.array(
                [float(t["loss_traj"][t_idx].item()) for t in slice_trials]
            )
            ax.plot(
                Ls, np.where(losses > 0, losses, np.nan),
                marker="o", lw=1.2, color=color,
                label=f"t = {t_idx} ({frac:.3g}·T)",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{sym}, P = {P}", fontsize=11)
        ax.set_xlabel(r"depth $L$")
    axes[0].set_ylabel(r"matched stationary loss $\mathcal{L}(t)$")
    axes[-1].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        "B2 final-loss-vs-depth: early snapshots are L-dependent, late "
        "snapshots collapse onto the stationary asymptote",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_both(fig, run_dir, "final_loss_vs_depth")
    plt.close(fig)


def _plot_long_context_collapse(
    trials: list[dict[str, Any]], cfg: B2Config, run_dir: ThesisRunDir
) -> None:
    """Diagnostic: loss L(t) at a fixed depth across P values for the
    long-context symbol, showing long-context convergence of the matched
    stationary trajectory.
    """
    import matplotlib.pyplot as plt

    L_plot = cfg.long_context_L
    sym = cfg.long_context_symbol
    # All P values that ran this symbol: main sweep + long-context sweep.
    all_P = tuple(sorted(set(cfg.P_list) | set(cfg.long_context_P_list)))
    P_colors = sequential_colors(max(1, len(all_P)), palette="rocket")

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    t_axis = np.arange(1, cfg.T + 1, dtype=float)
    slice_trials = _select(trials, L=L_plot, symbol_kind=sym)
    slice_trials.sort(key=lambda t: t["P"])
    if not slice_trials:
        plt.close(fig)
        return
    color_map = {P: P_colors[i] for i, P in enumerate(all_P)}
    for trial in slice_trials:
        color = color_map.get(trial["P"], "C0")
        loss = trial["loss_traj"][1:].numpy()
        loss = np.where(loss > 0.0, loss, np.nan)
        ax.plot(
            t_axis, loss, color=color, lw=1.4,
            label=f"P = {trial['P']}",
        )
    ax.set_title(f"{sym}, L = {L_plot}", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel(r"matched stationary loss $\mathcal{L}(t)$")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        f"B2 long-context collapse: matched stationary loss at L = {L_plot} "
        "as P increases",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_both(fig, run_dir, "long_context_collapse")
    plt.close(fig)


def _plot_per_mode_residuals(
    trials: list[dict[str, Any]], cfg: B2Config, run_dir: ThesisRunDir
) -> None:
    """Diagnostic: terminal residual transfer spectrum per mode at each L,
    for the figure slice."""
    import matplotlib.pyplot as plt

    subplots = [
        (sym, cfg.figure_P) for sym in cfg.symbol_kinds
    ]
    valid_subplots = [(s, P) for s, P in subplots if P in cfg.P_list]
    n_sub = len(valid_subplots)
    if n_sub == 0:
        return
    fig, axes = plt.subplots(
        1, n_sub, figsize=(4.8 * n_sub, 3.8), sharey=True
    )
    if n_sub == 1:
        axes = [axes]
    L_colors = sequential_colors(len(cfg.figure_L_list), palette="rocket")
    for ax, (sym, P) in zip(axes, valid_subplots):
        slice_trials = _select(
            trials, P=P, symbol_kind=sym, L_in=cfg.figure_L_list
        )
        slice_trials.sort(key=lambda t: t["L"])
        k_axis = np.arange(P)
        for color, trial in zip(L_colors, slice_trials):
            trans = trial["final_transfer_sq"].numpy()
            trans = np.where(trans > 0.0, trans, np.nan)
            ax.plot(
                k_axis, trans, color=color, lw=1.2,
                label=f"L = {trial['L']}", alpha=0.95,
            )
        ax.set_title(f"{sym}, P = {P}", fontsize=11)
        ax.set_yscale("log")
        ax.set_xlabel("mode index k")
    axes[0].set_ylabel(
        r"terminal residual transfer $(1 - L^{-1} s_k \gamma_k(T))^{2L}$"
    )
    axes[-1].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        f"B2 terminal residual transfer spectrum at t = T = {cfg.T}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_both(fig, run_dir, "per_mode_residuals")
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
            "Experiment B2: matched stationary depth-irrelevance "
            "(plan §6.3)."
        )
    )
    p.add_argument(
        "--device", type=str, default="cuda", choices=("cpu", "cuda", "auto")
    )
    p.add_argument(
        "--dtype", type=str, default="float64", choices=("float32", "float64")
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="suppress matplotlib display (headless use)"
    )
    p.add_argument(
        "--P-list", type=str, default=None,
        help="comma-separated P values; default 64,256,1024"
    )
    p.add_argument(
        "--L-list", type=str, default=None,
        help="comma-separated L values; default 1,2,4,8,16"
    )
    p.add_argument(
        "--symbol-kinds", type=str, default=None,
        help="comma-separated symbol kinds; default power_law,multiband"
    )
    p.add_argument("--T", type=int, default=None, help="trajectory length")
    p.add_argument("--eta", type=float, default=None, help="learning rate")
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> B2Config:
    base = B2Config()
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
    return replace(base, **overrides) if overrides else base


def _resolve_device(requested: str) -> torch.device:
    """Resolve and validate the device. B2 itself runs the per-mode recursion
    on CPU float64 (sequential Python loop; GPU offers no speedup). The
    ``device`` flag still gates environment sanity: if the user requested CUDA
    we require it to be available, matching the launcher contract.
    """
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Source starter.sh in an environment with CUDA, or pass "
                "--device cpu for a local dry run."
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
    print(f"[B2] device = {device}")
    run = ThesisRunDir(__file__, phase="theoremB")
    with RunContext(run, config=cfg, seeds=[0, 1, 2, 3]) as ctx:
        apply_thesis_style()

        # Build the ordered list of (P, L, symbol_kind) trials. Main sweep
        # first, then the long-context sub-sweep.
        trial_specs: list[tuple[int, int, str]] = []
        for P in cfg.P_list:
            for symbol_kind in cfg.symbol_kinds:
                for L in cfg.L_list:
                    trial_specs.append((P, L, symbol_kind))
        for P in cfg.long_context_P_list:
            for L in cfg.L_list:
                trial_specs.append((P, L, cfg.long_context_symbol))

        trials: list[dict[str, Any]] = []
        n_total = len(trial_specs)
        t_sweep_start = time.perf_counter()
        for idx, (P, L, symbol_kind) in enumerate(trial_specs, start=1):
            t0 = time.perf_counter()
            trial = _run_trial(cfg, P, L, symbol_kind)
            dt = time.perf_counter() - t0
            ctx.record_step_time(dt)
            print(
                f"[{idx:>3d}/{n_total}] "
                f"P={P:>4d} L={L:>2d} {symbol_kind:<10s} "
                f"L(0)={trial['loss_initial']:.3e} "
                f"L(T)={trial['loss_final']:.3e} "
                f"mono_viol={trial['monotonicity_violation_max']:.1e} "
                f"({dt*1000:6.1f} ms)"
            )
            trials.append(trial)
        t_sweep = time.perf_counter() - t_sweep_start

        # Save raw trajectories (npz). We deliberately exclude the full
        # (T+1, P) γ trajectories — they reproduce exactly via
        # ``gamma_star_trajectory_circulant`` from (s_tr, omega, L, eta, T),
        # and storing them bloats the archive to O(GB). We keep the scalar
        # loss trajectories (primary observable), the terminal residual
        # transfer spectrum (per-mode diagnostic), and the (P, symbol)-level
        # symbols (stored once per distinct pair).
        npz_payload: dict[str, np.ndarray] = {}
        for t in trials:
            key = f"P{t['P']}_L{t['L']}_{t['symbol_kind']}"
            npz_payload[f"{key}__loss"] = t["loss_traj"].numpy()
            npz_payload[f"{key}__final_transfer_sq"] = (
                t["final_transfer_sq"].numpy()
            )
        seen_pairs: set[tuple[int, str]] = set()
        for t in trials:
            pair = (t["P"], t["symbol_kind"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            npz_payload[f"P{t['P']}_{t['symbol_kind']}__s_tr"] = (
                t["s_tr"].numpy()
            )
            npz_payload[f"P{t['P']}_{t['symbol_kind']}__omega"] = (
                t["omega"].numpy()
            )
        npz_path = run.npz_path("depth_stationary")
        np.savez_compressed(npz_path, **npz_payload)

        per_trial_rows = [
            {
                "P": t["P"],
                "L": t["L"],
                "symbol_kind": t["symbol_kind"],
                "T": t["T"],
                "eta": t["eta"],
                "loss_analytic_initial": t["loss_analytic_initial"],
                "loss_initial": t["loss_initial"],
                "loss_final": t["loss_final"],
                "monotonicity_violation_max": t["monotonicity_violation_max"],
                "recursion_seconds": t["recursion_seconds"],
            }
            for t in trials
        ]
        (run.root / "per_trial_summary.json").write_text(
            json.dumps(per_trial_rows, indent=2) + "\n", encoding="utf-8"
        )

        # Figures.
        _plot_loss_vs_time(trials, cfg, run)
        _plot_final_loss_vs_depth(trials, cfg, run)
        _plot_long_context_collapse(trials, cfg, run)
        _plot_per_mode_residuals(trials, cfg, run)

        # --- Acceptance checks (qualitative) ---
        # 1. Monotonicity: no trial should increase its loss step-to-step
        #    beyond `monotonicity_slack`.
        max_mono_viol = max(t["monotonicity_violation_max"] for t in trials)
        mono_ok = max_mono_viol <= cfg.monotonicity_slack

        # 2. Decay: for every trial, loss_final / loss_initial must fall below
        #    ``depth_decay_fraction``. A shallow baseline and a deep model
        #    both reaching a common floor (0 in the matched-regime theorem)
        #    implies each has decayed substantially from its initial value;
        #    no L should stagnate near the initial loss.
        decay_rows: list[dict[str, Any]] = []
        decay_ok = True
        for trial in trials:
            decay_frac = (
                trial["loss_final"] / (trial["loss_initial"] + 1e-30)
            )
            if decay_frac > cfg.depth_decay_fraction:
                decay_ok = False
            decay_rows.append(
                {
                    "P": trial["P"],
                    "L": trial["L"],
                    "symbol_kind": trial["symbol_kind"],
                    "loss_initial": trial["loss_initial"],
                    "loss_final": trial["loss_final"],
                    "decay_fraction": float(decay_frac),
                }
            )

        # Diagnostic (not acceptance): cross-L terminal ratio per (P, symbol).
        depth_ratios: list[dict[str, Any]] = []
        unique_keys = sorted({(t["P"], t["symbol_kind"]) for t in trials})
        for P, sym in unique_keys:
            slice_trials = _select(trials, P=P, symbol_kind=sym)
            if not slice_trials:
                continue
            slice_trials.sort(key=lambda t: t["L"])
            l_min = slice_trials[0]["L"]
            l_max = slice_trials[-1]["L"]
            loss_min = slice_trials[0]["loss_final"]
            loss_max = slice_trials[-1]["loss_final"]
            ratio = (loss_max + 1e-30) / (loss_min + 1e-30)
            depth_ratios.append(
                {
                    "P": int(P),
                    "symbol_kind": sym,
                    "L_min": int(l_min),
                    "L_max": int(l_max),
                    "loss_final_L_min": loss_min,
                    "loss_final_L_max": loss_max,
                    "ratio": float(ratio),
                }
            )

        # Aggregate summary.
        final_losses = {
            f"P{t['P']}_L{t['L']}_{t['symbol_kind']}": t["loss_final"]
            for t in trials
        }
        initial_losses = {
            f"P{t['P']}_L{t['L']}_{t['symbol_kind']}": t["loss_initial"]
            for t in trials
        }

        ctx.record_compute_proxy(float(t_sweep))
        ctx.record_extra("n_trials", len(trials))
        ctx.record_extra("device", str(device))
        ctx.record_extra("max_monotonicity_violation", max_mono_viol)
        ctx.record_extra("decay_rows", decay_rows)
        ctx.record_extra("depth_ratios", depth_ratios)
        ctx.record_extra("initial_losses", initial_losses)
        ctx.record_extra("final_losses", final_losses)

        max_decay_frac = max(row["decay_fraction"] for row in decay_rows)
        worst_decay = max(decay_rows, key=lambda r: r["decay_fraction"])

        status_parts: list[str] = []
        if mono_ok:
            status_parts.append("monotonicity_ok")
        else:
            status_parts.append(
                f"monotonicity_violated(max={max_mono_viol:.2e})"
            )
        if decay_ok:
            status_parts.append("depth_decay_ok")
        else:
            status_parts.append("depth_decay_violated")
        status = "+".join(status_parts)

        ctx.write_summary(
            {
                "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §6.3 (B2)",
                "interpretation": (
                    "Finite-time matched-stationary depth-irrelevance "
                    "experiment: no evidence of a depth-dependent asymptotic "
                    "floor; finite-T cross-L differences are transient-rate "
                    "effects (L=1 exponential, L>1 polynomial of order "
                    "-1/(2L-2))."
                ),
                "device": str(device),
                "n_trials": len(trials),
                "status": status,
                "monotonicity_slack": cfg.monotonicity_slack,
                "max_monotonicity_violation": max_mono_viol,
                "depth_decay_fraction_max": cfg.depth_decay_fraction,
                "max_decay_fraction_observed": max_decay_frac,
                "worst_decay_trial": worst_decay,
                "depth_ratios": depth_ratios,
                "sweep_wallclock_seconds": round(t_sweep, 3),
            }
        )

        print()
        print("=" * 72)
        print(f" B2 depth-irrelevance: {len(trials)} trials on {device}")
        print(f"   max monotonicity violation  = {max_mono_viol:.3e}")
        print(
            f"   monotonicity ok             = {mono_ok} "
            f"(slack = {cfg.monotonicity_slack:.1e})"
        )
        print(
            f"   decay_ok                    = {decay_ok} "
            f"(max frac = {max_decay_frac:.3e} vs threshold "
            f"{cfg.depth_decay_fraction:.2f})"
        )
        print(f"   cross-L terminal diagnostics:")
        for row in depth_ratios:
            print(
                f"   P={row['P']:<5d} {row['symbol_kind']:<10s} "
                f"L{row['L_min']}→L{row['L_max']}: "
                f"ratio = {row['ratio']:.3e}  "
                f"(L_final: {row['loss_final_L_min']:.3e} → "
                f"{row['loss_final_L_max']:.3e})"
            )
        print("=" * 72)

        if not mono_ok or not decay_ok:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
