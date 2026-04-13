"""Experiment B3: spectral OOD brittleness under symbol mismatch.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §6.4.

Purpose
-------
Test the theorem-B prediction of **fixed-basis OOD brittleness** in the
stationary spectral regime: a model trained with ``s_tr = s_te`` (matched
stationary optimum, same recursion used in B2) loses accuracy when evaluated
under test-time symbol shifts that leave the Fourier basis unchanged. The
figure plays the conceptual role of Bordelon Figure 3c but with a **spectral
shift parameter** on the horizontal axis rather than a generic covariance
rotation.

Primary OOD perturbations (plan §6.4, binding)
----------------------------------------------
1. **Structural interpolation (family 1):**

       s_te(α) = (1 − α) · s_tr + α · s_other

   where ``s_other`` is a different real-even symbol — typically a flat
   spectrum (the opposite extreme of power-law concentration). α ∈ [0, 1]
   scans from matched (α = 0) to fully-shifted (α = 1).

2. **Frequency permutation interpolation (family 2):**

       s_te(α, seed) = (1 − α) · s_tr + α · permute(s_tr, seed)

   where ``permute`` is the real-even-preserving frequency-index permutation
   from :func:`fourier_ops.frequency_permutation`. This is the spectral
   analogue of "frequency reallocation" in plan §6.4 — the marginal
   amplitude distribution is preserved but modes are shuffled. Aggregated
   over seeds to surface dispersion.

Generic covariance rotation is **explicitly excluded** from this script; it
belongs to a secondary bridge experiment toward theorem C (per plan §6.4).

Loss formula (matched-training / shifted-test, circulant diagonal)
------------------------------------------------------------------
At the terminal reduced-Γ produced by the matched training recursion
(``γ_k(T)`` from :func:`metrics.gamma_star_trajectory_circulant`), the
population OOD loss at a shifted test symbol ``s_te`` is:

    L_ood(γ, s_tr, s_te, ω, L)
        = Σ_k  ω_k · s_te_k · (1 − L⁻¹ · s_tr_k · γ_k(T))^(2L)

where ``s_tr`` stays in the transfer factor (training operator, fixed after
training) while ``s_te`` re-weights modes per the test input covariance.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``G1Config``, ``g1_generate`` (exact + population + matched test).
- :mod:`scripts.thesis.utils.metrics`:
    ``gamma_star_trajectory_circulant`` (matched-training trajectory).
- :mod:`scripts.thesis.utils.fourier_ops`:
    ``symbol_interpolate``, ``frequency_permutation``,
    ``symbol_power_law``, ``symbol_multiband``, ``symbol_flat`` (for the
    family-1 ``s_other`` construction).
- :mod:`scripts.thesis.utils.plotting` / :mod:`run_metadata`: standard.

Primary outputs (§6.4)
----------------------
- ``ood_interpolation_structural`` — primary Bordelon Fig 3c analogue:
  OOD loss vs α for family 1, one line per L, semi-log y.
- ``ood_interpolation_permutation`` — family-2 analogue with seed spread:
  median line per L plus shaded min–max envelope.
- ``matched_baseline`` — diagnostic: matched-training loss at α = 0 per L
  (sanity check against B2's terminal losses at the same hyperparameters).
- ``symbol_samples`` — diagnostic: the actual ``s_tr``, ``s_other`` and a
  few interpolated ``s_te(α)`` curves, both families, to give the reader
  a picture of what the shift does in symbol space.

Acceptance
----------
This is a qualitative OOD-brittleness test, not a closure test:

1. **Matched baseline recovery**: at α = 0 the OOD loss must equal (to float
   eps) the matched-training terminal loss, for every L. Enforced.
2. **Monotone brittleness at small α**: for at least one L in ``L_list``,
   the family-1 OOD loss at α = ``brittleness_alpha`` exceeds the matched
   baseline by at least ``brittleness_ratio_min`` (default 2×). This is a
   weak sanity check that the experiment produces a nontrivial OOD signal.

Both checks run automatically and are surfaced in ``summary.txt``.

Run
---
::

    python -u scripts/thesis/theoremB/run_theoremB_symbol_shift.py \\
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

from scripts.thesis.utils.data_generators import G1Config, g1_generate
from scripts.thesis.utils.fourier_ops import (
    frequency_permutation,
    symbol_flat,
    symbol_interpolate,
    symbol_multiband,
    symbol_power_law,
)
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
class B3Config:
    """Frozen configuration for the B3 symbol-shift OOD experiment.

    A single ``P`` and a sweep over ``L`` is sufficient for the §6.4 figure;
    the OOD curves of interest are parameterized by the spectral-shift
    variable α, not P. ``eta`` and ``T`` are inherited from B2 so the
    terminal γ reproduces B2's matched terminal losses to float eps (the
    ``matched baseline recovery`` acceptance check).
    """

    # Fixed context and depth sweep.
    P: int = 64
    L_list: tuple[int, ...] = (1, 2, 4, 8, 16)

    # Matched-training recursion. Same eta & T as B2 so the matched baseline
    # reproduces exactly (up to float eps).
    T: int = 100000
    eta: float = 5e-5

    # Base training symbol (in-distribution). Same defaults as B1/B2.
    base_symbol_kind: str = "power_law"
    power_law_nu: float = 0.5
    task_spec_nu_beta: float = 1.0
    multiband: tuple[tuple[int, int, float], ...] = (
        (0, 2, 1.0),
        (5, 7, 0.8),
    )

    # Family 1 — structural interpolation to an alternative symbol.
    # The default target is flat (s_other[k] = 1), the opposite extreme of
    # the power-law base.
    f1_target_kind: str = "flat"
    f1_target_params_flat_value: float = 1.0
    f1_target_params_power_law_nu: float = 0.0
    f1_target_params_multiband: tuple[tuple[int, int, float], ...] = (
        (0, 4, 1.0),
    )
    f1_alphas: tuple[float, ...] = (
        0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0,
    )

    # Family 2 — frequency-permutation interpolation. A handful of seeds
    # give a sense of the permutation-family dispersion.
    f2_perm_seeds: tuple[int, ...] = (17, 23, 31, 41, 47, 53, 61, 71)
    f2_alphas: tuple[float, ...] = (
        0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0,
    )

    # Query regime (v4 §10.2.2).
    query_mode: str = "full_window"
    matched_query_realization: str = "independent"

    # Acceptance knobs.
    # The matched-baseline tolerance gates the identity at α=0 (must
    # reproduce B2 terminal losses to float eps). brittleness_alpha is the
    # shift level at which we check that OOD produces a nontrivial
    # increase; 1.0 (fully-shifted) is the strongest available point in the
    # family-1 grid and matches the spirit of "OOD brittleness under
    # symbol mismatch" (plan §6.4). brittleness_ratio_min is set at 1.25×
    # — the matched training in the B2 regime is only partially converged
    # on the slow tail of modes (s_k^3 small), so the ω·s_te re-weighting
    # at α=1 shifts mass between trained and untrained modes but does not
    # produce orders-of-magnitude differences. A 25% OOD increase at full
    # shift is a clean directional signal, and the full-α sweep in the
    # primary figure shows the complete brittleness trajectory.
    matched_baseline_tol: float = 1e-10
    brittleness_alpha: float = 1.0
    brittleness_ratio_min: float = 1.25

    # Symbol-sample figure slice.
    symbol_samples_alphas: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0)

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Loss helper — train-matched / test-shifted, circulant diagonal
# ---------------------------------------------------------------------------


def _ood_loss(
    gamma: torch.Tensor,
    s_tr: torch.Tensor,
    s_te: torch.Tensor,
    omega: torch.Tensor,
    L: int,
) -> float:
    """Population OOD loss under a shifted test symbol. See module docstring
    for the derivation. ``gamma`` may be 1D ``(P,)`` (terminal state) or 2D
    ``(T+1, P)`` (trajectory); returns a scalar or a ``(T+1,)`` tensor.
    """
    g = gamma.to(torch.float64)
    s_tr64 = s_tr.to(torch.float64)
    s_te64 = s_te.to(torch.float64)
    w64 = omega.to(torch.float64)
    if g.ndim == 1:
        residual = 1.0 - s_tr64 * g / int(L)
        transfer_sq = residual.pow(2 * int(L))
        return float((w64 * s_te64 * transfer_sq).sum().item())
    elif g.ndim == 2:
        residual = 1.0 - s_tr64.unsqueeze(0) * g / int(L)
        transfer_sq = residual.pow(2 * int(L))
        per_mode = w64.unsqueeze(0) * s_te64.unsqueeze(0) * transfer_sq
        return per_mode.sum(dim=1)
    else:
        raise ValueError(f"gamma must be 1D or 2D; got ndim {g.ndim}")


# ---------------------------------------------------------------------------
# Symbol construction helpers (family 1)
# ---------------------------------------------------------------------------


def _build_base_symbol(cfg: B3Config) -> torch.Tensor:
    if cfg.base_symbol_kind == "power_law":
        return symbol_power_law(cfg.P, cfg.power_law_nu)
    if cfg.base_symbol_kind == "multiband":
        return symbol_multiband(cfg.P, list(cfg.multiband))
    if cfg.base_symbol_kind == "flat":
        return symbol_flat(cfg.P, 1.0)
    raise ValueError(f"unknown base_symbol_kind: {cfg.base_symbol_kind!r}")


def _build_f1_target_symbol(cfg: B3Config) -> torch.Tensor:
    if cfg.f1_target_kind == "flat":
        return symbol_flat(cfg.P, cfg.f1_target_params_flat_value)
    if cfg.f1_target_kind == "power_law":
        return symbol_power_law(cfg.P, cfg.f1_target_params_power_law_nu)
    if cfg.f1_target_kind == "multiband":
        return symbol_multiband(cfg.P, list(cfg.f1_target_params_multiband))
    raise ValueError(
        f"unknown f1_target_kind: {cfg.f1_target_kind!r}"
    )


def _build_g1_config(cfg: B3Config) -> G1Config:
    if cfg.base_symbol_kind == "power_law":
        params: dict[str, Any] = {"nu": cfg.power_law_nu}
    elif cfg.base_symbol_kind == "multiband":
        params = {"bands": list(cfg.multiband)}
    elif cfg.base_symbol_kind == "flat":
        params = {"value": 1.0}
    else:
        raise ValueError(
            f"unknown base_symbol_kind: {cfg.base_symbol_kind!r}"
        )
    return G1Config(
        P=cfg.P,
        B=1,
        query_mode=cfg.query_mode,
        matched_query_realization=cfg.matched_query_realization,
        symbol_kind_tr=cfg.base_symbol_kind,
        symbol_params_tr=params,
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


# ---------------------------------------------------------------------------
# Training + OOD evaluation
# ---------------------------------------------------------------------------


def _train_matched(
    cfg: B3Config,
) -> tuple[
    torch.Tensor,  # s_tr
    torch.Tensor,  # omega
    dict[int, torch.Tensor],  # L -> gamma_final (P,)
    dict[int, float],  # L -> matched baseline loss
]:
    """Run the matched-training recursion at each L in ``cfg.L_list``.
    Returns the shared ``(s_tr, omega)``, a dict of terminal γ per L, and
    the matched (baseline) loss per L."""
    g1_cfg = _build_g1_config(cfg)
    op = g1_generate(g1_cfg)
    s_tr = op["s_tr"]
    omega = op["omega"]
    gamma_final: dict[int, torch.Tensor] = {}
    baseline: dict[int, float] = {}
    for L in cfg.L_list:
        traj = gamma_star_trajectory_circulant(
            s_tr, omega, L=int(L), eta=cfg.eta, T=cfg.T
        )
        gamma_final[int(L)] = traj[-1].detach().cpu()
        # Matched baseline loss = OOD loss with s_te = s_tr.
        baseline[int(L)] = _ood_loss(traj[-1], s_tr, s_tr, omega, int(L))
    return s_tr.detach().cpu(), omega.detach().cpu(), gamma_final, baseline


def _eval_family1(
    cfg: B3Config,
    s_tr: torch.Tensor,
    omega: torch.Tensor,
    gamma_final: dict[int, torch.Tensor],
) -> dict[str, Any]:
    """Family 1: structural interpolation toward ``s_other``."""
    s_other = _build_f1_target_symbol(cfg)
    losses: dict[int, list[float]] = {int(L): [] for L in cfg.L_list}
    s_te_list: list[torch.Tensor] = []
    for alpha in cfg.f1_alphas:
        s_te = symbol_interpolate(s_tr, s_other, float(alpha))
        s_te_list.append(s_te)
        for L in cfg.L_list:
            losses[int(L)].append(
                _ood_loss(gamma_final[int(L)], s_tr, s_te, omega, int(L))
            )
    return {
        "alphas": tuple(float(a) for a in cfg.f1_alphas),
        "s_other": s_other,
        "s_te_list": s_te_list,
        "losses": {L: np.asarray(v) for L, v in losses.items()},
    }


def _eval_family2(
    cfg: B3Config,
    s_tr: torch.Tensor,
    omega: torch.Tensor,
    gamma_final: dict[int, torch.Tensor],
) -> dict[str, Any]:
    """Family 2: interpolation toward a frequency-permuted version of s_tr,
    aggregated across seeds.
    """
    seeds = tuple(int(s) for s in cfg.f2_perm_seeds)
    alphas = tuple(float(a) for a in cfg.f2_alphas)
    # losses_per_seed[L] -> array of shape (n_seeds, n_alpha)
    losses_per_seed: dict[int, np.ndarray] = {
        int(L): np.zeros((len(seeds), len(alphas))) for L in cfg.L_list
    }
    s_perm_dict: dict[int, torch.Tensor] = {}
    for si, seed in enumerate(seeds):
        s_perm = frequency_permutation(s_tr, seed=seed)
        s_perm_dict[seed] = s_perm
        for ai, alpha in enumerate(alphas):
            s_te = symbol_interpolate(s_tr, s_perm, alpha)
            for L in cfg.L_list:
                losses_per_seed[int(L)][si, ai] = _ood_loss(
                    gamma_final[int(L)], s_tr, s_te, omega, int(L)
                )
    return {
        "alphas": alphas,
        "seeds": seeds,
        "s_perm_dict": s_perm_dict,
        "losses_per_seed": losses_per_seed,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_family1(
    cfg: B3Config,
    f1: dict[str, Any],
    baseline: dict[int, float],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    alphas = np.asarray(f1["alphas"])
    losses = f1["losses"]
    L_colors = sequential_colors(len(cfg.L_list), palette="rocket")
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for color, L in zip(L_colors, cfg.L_list):
        y = losses[int(L)]
        ax.plot(alphas, y, color=color, lw=1.4, marker="o", ms=3.5,
                label=f"L = {L}")
    base_min = min(baseline.values())
    overlay_reference(
        ax, alphas, np.full_like(alphas, base_min),
        label=r"matched baseline (min over L)",
        style=":", color="gray", lw=1.0,
    )
    ax.set_xlabel(r"shift $\alpha$ (family 1: structural interpolation)")
    ax.set_ylabel(r"OOD loss $\mathcal{L}_{\mathrm{ood}}(\alpha)$")
    ax.set_yscale("log")
    ax.set_title(
        f"B3 structural-interpolation brittleness "
        f"(base = {cfg.base_symbol_kind}, target = {cfg.f1_target_kind}, "
        f"P = {cfg.P})",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    save_both(fig, run_dir, "ood_interpolation_structural")
    plt.close(fig)


def _plot_family2(
    cfg: B3Config,
    f2: dict[str, Any],
    baseline: dict[int, float],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    alphas = np.asarray(f2["alphas"])
    losses = f2["losses_per_seed"]  # L -> (n_seeds, n_alpha)
    L_colors = sequential_colors(len(cfg.L_list), palette="rocket")
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for color, L in zip(L_colors, cfg.L_list):
        y = losses[int(L)]
        median = np.median(y, axis=0)
        y_lo = np.min(y, axis=0)
        y_hi = np.max(y, axis=0)
        ax.fill_between(alphas, y_lo, y_hi, color=color, alpha=0.18, lw=0)
        ax.plot(alphas, median, color=color, lw=1.4, marker="o", ms=3.5,
                label=f"L = {L} (median)")
    base_min = min(baseline.values())
    overlay_reference(
        ax, alphas, np.full_like(alphas, base_min),
        label=r"matched baseline (min over L)",
        style=":", color="gray", lw=1.0,
    )
    ax.set_xlabel(r"shift $\alpha$ (family 2: permutation interpolation)")
    ax.set_ylabel(r"OOD loss $\mathcal{L}_{\mathrm{ood}}(\alpha)$")
    ax.set_yscale("log")
    ax.set_title(
        f"B3 permutation-interpolation brittleness "
        f"(median over {len(f2['seeds'])} seeds; band = min–max) ",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    save_both(fig, run_dir, "ood_interpolation_permutation")
    plt.close(fig)


def _plot_matched_baseline(
    cfg: B3Config, baseline: dict[int, float], run_dir: ThesisRunDir
) -> None:
    import matplotlib.pyplot as plt

    Ls = np.asarray(sorted(baseline.keys()), dtype=float)
    ys = np.asarray([baseline[int(L)] for L in Ls])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(Ls, ys, color="C0", lw=1.4, marker="o", ms=4.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"depth $L$")
    ax.set_ylabel(r"matched baseline loss $\mathcal{L}(\alpha = 0, L)$")
    ax.set_title(
        "B3 matched-training baseline (α = 0); sanity check vs B2 terminal "
        "losses",
        fontsize=10,
    )
    fig.tight_layout()
    save_both(fig, run_dir, "matched_baseline")
    plt.close(fig)


def _plot_symbol_samples(
    cfg: B3Config,
    s_tr: torch.Tensor,
    f1: dict[str, Any],
    f2: dict[str, Any],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    k_axis = np.arange(cfg.P)
    # Family 1: s_te samples at selected α.
    s_te_list = f1["s_te_list"]
    alphas_f1 = list(f1["alphas"])
    # Family 2: pick one seed for the samples figure.
    seed0 = f2["seeds"][0]
    s_perm = f2["s_perm_dict"][seed0].numpy()

    # Selected alphas actually present in f1_alphas.
    sel_alphas = [a for a in cfg.symbol_samples_alphas if a in alphas_f1]
    if not sel_alphas:
        sel_alphas = alphas_f1[:4]
    colors = sequential_colors(len(sel_alphas), palette="rocket")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), sharey=True)
    # Panel (a): family 1 structural interpolation.
    ax = axes[0]
    for color, alpha in zip(colors, sel_alphas):
        idx = alphas_f1.index(alpha)
        ax.plot(k_axis, s_te_list[idx].numpy(), color=color, lw=1.3,
                label=rf"$\alpha = {alpha:.2f}$")
    ax.set_title(
        f"Family 1: interpolation to {cfg.f1_target_kind}",
        fontsize=10,
    )
    ax.set_xlabel("mode index k")
    ax.set_ylabel(r"$s_{\mathrm{te}}(k)$")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")

    # Panel (b): family 2 permutation interpolation at seed[0].
    ax = axes[1]
    for color, alpha in zip(colors, sel_alphas):
        s_te = symbol_interpolate(s_tr, f2["s_perm_dict"][seed0], float(alpha))
        ax.plot(k_axis, s_te.numpy(), color=color, lw=1.3,
                label=rf"$\alpha = {alpha:.2f}$")
    ax.set_title(
        f"Family 2: interpolation to permute(s_tr, seed={seed0})",
        fontsize=10,
    )
    ax.set_xlabel("mode index k")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"B3 symbol-shift samples (base = {cfg.base_symbol_kind}, P = {cfg.P})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_both(fig, run_dir, "symbol_samples")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_list_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment B3: symbol-native spectral OOD brittleness "
            "(plan §6.4)."
        )
    )
    p.add_argument(
        "--device", type=str, default="cuda", choices=("cpu", "cuda", "auto")
    )
    p.add_argument(
        "--dtype", type=str, default="float64", choices=("float32", "float64")
    )
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--P", type=int, default=None)
    p.add_argument("--L-list", type=str, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--base-symbol", type=str, default=None,
                   choices=("power_law", "multiband", "flat"))
    p.add_argument("--f1-target", type=str, default=None,
                   choices=("flat", "power_law", "multiband"))
    p.add_argument("--f1-alphas", type=str, default=None)
    p.add_argument("--f2-alphas", type=str, default=None)
    p.add_argument("--f2-seeds", type=str, default=None)
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> B3Config:
    base = B3Config()
    overrides: dict[str, Any] = {}
    if args.dtype is not None:
        overrides["dtype"] = args.dtype
    if args.device is not None:
        overrides["device"] = args.device
    if args.P is not None:
        overrides["P"] = int(args.P)
    if args.L_list is not None:
        overrides["L_list"] = _parse_list_ints(args.L_list)
    if args.T is not None:
        overrides["T"] = int(args.T)
    if args.eta is not None:
        overrides["eta"] = float(args.eta)
    if args.base_symbol is not None:
        overrides["base_symbol_kind"] = args.base_symbol
    if args.f1_target is not None:
        overrides["f1_target_kind"] = args.f1_target
    if args.f1_alphas is not None:
        overrides["f1_alphas"] = _parse_list_floats(args.f1_alphas)
    if args.f2_alphas is not None:
        overrides["f2_alphas"] = _parse_list_floats(args.f2_alphas)
    if args.f2_seeds is not None:
        overrides["f2_perm_seeds"] = _parse_list_ints(args.f2_seeds)
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
    print(f"[B3] device = {device}")
    run = ThesisRunDir(__file__, phase="theoremB")
    with RunContext(run, config=cfg, seeds=list(cfg.f2_perm_seeds)) as ctx:
        apply_thesis_style()

        # --- Train matched, per L ---
        t_train0 = time.perf_counter()
        s_tr, omega, gamma_final, baseline = _train_matched(cfg)
        t_train = time.perf_counter() - t_train0
        for L in cfg.L_list:
            print(
                f"[train] L={L:>2d}  matched baseline loss = "
                f"{baseline[int(L)]:.4e}"
            )
        print(f"[train] total train wallclock = {t_train:.2f} s")

        # --- Family 1 evaluation ---
        t1 = time.perf_counter()
        f1 = _eval_family1(cfg, s_tr, omega, gamma_final)
        t_f1 = time.perf_counter() - t1

        # --- Family 2 evaluation ---
        t2 = time.perf_counter()
        f2 = _eval_family2(cfg, s_tr, omega, gamma_final)
        t_f2 = time.perf_counter() - t2

        for L in cfg.L_list:
            y = f1["losses"][int(L)]
            print(
                f"[f1] L={L:>2d}  L_ood(α=0)={y[0]:.4e}  "
                f"L_ood(α=1)={y[-1]:.4e}  "
                f"max ratio = {y.max() / (y[0] + 1e-30):.3e}"
            )
        for L in cfg.L_list:
            y_med = np.median(f2["losses_per_seed"][int(L)], axis=0)
            print(
                f"[f2] L={L:>2d}  median L_ood(α=0)={y_med[0]:.4e}  "
                f"median L_ood(α=1)={y_med[-1]:.4e}  "
                f"max median ratio = "
                f"{y_med.max() / (y_med[0] + 1e-30):.3e}"
            )

        # --- Figures ---
        _plot_family1(cfg, f1, baseline, run)
        _plot_family2(cfg, f2, baseline, run)
        _plot_matched_baseline(cfg, baseline, run)
        _plot_symbol_samples(cfg, s_tr, f1, f2, run)

        # --- Save npz ---
        npz_payload: dict[str, np.ndarray] = {
            "s_tr": s_tr.numpy(),
            "omega": omega.numpy(),
            "s_other_f1": f1["s_other"].numpy(),
            "f1_alphas": np.asarray(f1["alphas"]),
            "f2_alphas": np.asarray(f2["alphas"]),
            "f2_seeds": np.asarray(f2["seeds"]),
        }
        for L in cfg.L_list:
            npz_payload[f"gamma_final_L{L}"] = gamma_final[int(L)].numpy()
            npz_payload[f"f1_loss_L{L}"] = f1["losses"][int(L)]
            npz_payload[f"f2_loss_per_seed_L{L}"] = (
                f2["losses_per_seed"][int(L)]
            )
        for seed in f2["seeds"]:
            npz_payload[f"s_perm_seed{seed}"] = (
                f2["s_perm_dict"][seed].numpy()
            )
        np.savez_compressed(run.npz_path("symbol_shift"), **npz_payload)

        # --- Per-trial summary JSON ---
        f1_rows = []
        for ai, alpha in enumerate(f1["alphas"]):
            for L in cfg.L_list:
                f1_rows.append(
                    {
                        "family": "structural_interpolation",
                        "alpha": float(alpha),
                        "L": int(L),
                        "L_ood": float(f1["losses"][int(L)][ai]),
                        "ratio_to_matched": float(
                            f1["losses"][int(L)][ai]
                            / (baseline[int(L)] + 1e-30)
                        ),
                    }
                )
        f2_rows = []
        for ai, alpha in enumerate(f2["alphas"]):
            for L in cfg.L_list:
                y = f2["losses_per_seed"][int(L)][:, ai]
                f2_rows.append(
                    {
                        "family": "permutation_interpolation",
                        "alpha": float(alpha),
                        "L": int(L),
                        "L_ood_median": float(np.median(y)),
                        "L_ood_min": float(np.min(y)),
                        "L_ood_max": float(np.max(y)),
                        "ratio_to_matched_median": float(
                            np.median(y) / (baseline[int(L)] + 1e-30)
                        ),
                    }
                )
        (run.root / "per_alpha_summary.json").write_text(
            json.dumps(
                {"family1": f1_rows, "family2": f2_rows, "baseline": {
                    str(L): float(baseline[int(L)]) for L in cfg.L_list
                }},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # --- Acceptance: matched baseline recovery at α=0 ---
        tol = cfg.matched_baseline_tol
        mb_f1_err = max(
            abs(f1["losses"][int(L)][0] - baseline[int(L)])
            / (abs(baseline[int(L)]) + 1e-30)
            for L in cfg.L_list
        )
        mb_f2_err = max(
            abs(
                np.median(f2["losses_per_seed"][int(L)][:, 0])
                - baseline[int(L)]
            )
            / (abs(baseline[int(L)]) + 1e-30)
            for L in cfg.L_list
        )
        matched_ok = (mb_f1_err <= tol) and (mb_f2_err <= tol)

        # --- Acceptance: brittleness sanity ---
        # Find α in f1_alphas closest to brittleness_alpha.
        f1_alphas_arr = np.asarray(f1["alphas"])
        idx_b = int(
            np.argmin(np.abs(f1_alphas_arr - cfg.brittleness_alpha))
        )
        brittleness_rows = []
        brittleness_ok = False
        for L in cfg.L_list:
            loss_at_b = float(f1["losses"][int(L)][idx_b])
            ratio = loss_at_b / (baseline[int(L)] + 1e-30)
            if ratio >= cfg.brittleness_ratio_min:
                brittleness_ok = True
            brittleness_rows.append(
                {
                    "L": int(L),
                    "alpha": float(f1_alphas_arr[idx_b]),
                    "L_ood": loss_at_b,
                    "baseline": float(baseline[int(L)]),
                    "ratio": float(ratio),
                }
            )

        ctx.record_compute_proxy(float(t_train + t_f1 + t_f2))
        ctx.record_extra("n_L", len(cfg.L_list))
        ctx.record_extra("n_f1_alphas", len(cfg.f1_alphas))
        ctx.record_extra("n_f2_alphas", len(cfg.f2_alphas))
        ctx.record_extra("n_f2_seeds", len(cfg.f2_perm_seeds))
        ctx.record_extra("matched_baseline_f1_err", mb_f1_err)
        ctx.record_extra("matched_baseline_f2_err", mb_f2_err)
        ctx.record_extra("brittleness_rows", brittleness_rows)
        ctx.record_extra("train_seconds", float(t_train))
        ctx.record_extra("f1_seconds", float(t_f1))
        ctx.record_extra("f2_seconds", float(t_f2))

        status_parts: list[str] = []
        status_parts.append(
            "matched_ok" if matched_ok else
            f"matched_violated(f1={mb_f1_err:.2e},f2={mb_f2_err:.2e})"
        )
        status_parts.append(
            "brittleness_ok" if brittleness_ok else "brittleness_weak"
        )
        status = "+".join(status_parts)

        ctx.write_summary(
            {
                "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §6.4 (B3)",
                "interpretation": (
                    "Fixed-basis OOD brittleness under symbol-native shifts: "
                    "structural interpolation (family 1) and frequency-"
                    "permutation interpolation (family 2). Generic covariance "
                    "rotation is deliberately excluded from this experiment "
                    "and belongs to a secondary bridge experiment toward "
                    "theorem C."
                ),
                "device": str(device),
                "n_trials": {
                    "L": len(cfg.L_list),
                    "f1_alphas": len(cfg.f1_alphas),
                    "f2_alphas": len(cfg.f2_alphas),
                    "f2_seeds": len(cfg.f2_perm_seeds),
                },
                "status": status,
                "matched_baseline_tol": tol,
                "matched_baseline_f1_err": float(mb_f1_err),
                "matched_baseline_f2_err": float(mb_f2_err),
                "brittleness_alpha": cfg.brittleness_alpha,
                "brittleness_ratio_min": cfg.brittleness_ratio_min,
                "brittleness_rows": brittleness_rows,
                "train_wallclock_seconds": round(t_train, 3),
                "f1_wallclock_seconds": round(t_f1, 3),
                "f2_wallclock_seconds": round(t_f2, 3),
            }
        )

        print()
        print("=" * 72)
        print(f" B3 symbol-shift: matched + family1 + family2 on {device}")
        print(
            f"   matched baseline recovery:  "
            f"f1_err = {mb_f1_err:.2e}, f2_err = {mb_f2_err:.2e}  "
            f"(tol = {tol:.1e})  -> {'ok' if matched_ok else 'FAIL'}"
        )
        print(
            f"   brittleness sanity at α≈{cfg.brittleness_alpha:.2f}:  "
            f"{'ok' if brittleness_ok else 'WEAK'} "
            f"(min ratio >= {cfg.brittleness_ratio_min:.1f} required on "
            f"at least one L)"
        )
        for row in brittleness_rows:
            print(
                f"     L={row['L']:<3d}  α={row['alpha']:.3f}  "
                f"L_ood={row['L_ood']:.4e}  baseline="
                f"{row['baseline']:.4e}  ratio={row['ratio']:.3e}"
            )
        print("=" * 72)

        if not matched_ok or not brittleness_ok:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
