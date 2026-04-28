"""Spectral rank bottleneck experiment (Theorem B in a trained architecture).

Trains :class:`SpectralAttention` at varying spectral bottleneck ``r`` on
stationary circulant data and shows that the terminal loss floor scales as a
power law in ``r``.

The stationary sampler's population kernel is circulant with symbol
``s_data = (1 + k_star)^{-nu}`` (mean-1 normalized, ``k_star = min(k, P - k)``).
The ideal filter inverts ``s_data`` mode-by-mode, so a rank-``r`` bottleneck
on the spectral filter leaves an irreducible tail-sum contribution from the
uncontrolled modes. Theorem B predicts this tail decays as
``tail(r) ~ r^{-(nu - 1)}`` for power-law ``nu > 1``. We pick ``nu = 1.5`` for
a clean reference exponent ``-0.5``: ``nu = 1.0`` would be logarithmic and
``nu >> 1`` would saturate too quickly.

All SpectralAttention models use ``s_init="gd"`` regardless of ``r``. The
GD-compatible mask has spectral support only at DC, which is representable at
any ``r >= 1``, so every model starts from an identical effective mask and
differences in terminal loss come purely from learnable spectral capacity.

Sweep: 1 :class:`LinearAttention` + 7 :class:`SpectralAttention` r-values, each
at 4 seeds, 8000 Adam steps per run = 32 total runs. Quick mode drops to 2
seeds, 4 r-values, 2000 steps (10 runs, ~3-5 min on H200) and is the
default login-node smoke-test.

Outputs under ``outputs/thesis/architectures/<script_stem>/<run_id>/``:

- ``losses.npz``, ``terminal_losses.npz``, ``symbols.npz`` -- per-run data.
- ``spectra.npz`` -- data symbol, sqrt_symbol, analytical tail sums per r,
                     fitted empirical / tail / theory exponents.
- ``figures/`` + ``pdfs/`` -- four figures:
    * ``rank_floor`` (log-log terminal loss vs r with tail + linear refs + fit)
    * ``rank_loss_vs_steps`` (loss curves at selected r values)
    * ``rank_learned_symbol`` (terminal S_TT symbols at selected r values)
    * ``rank_exponent_comparison`` (empirical vs tail vs theory slopes).
- ``summary.txt`` -- acceptance criteria + fitted exponents.

CUDA only (per durable thesis preference); float32 for training. Supports
``--quick`` for login-node sanity runs and ``--reprocess <run_dir>`` to
regenerate figures + summary from an existing run's NPZ without retraining.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.thesis.architectures.models import (  # noqa: E402
    LinearAttention,
    SpectralAttention,
)
from scripts.thesis.architectures.samplers import make_stationary_sampler  # noqa: E402
from scripts.thesis.architectures.training import train_icl_online  # noqa: E402
from scripts.thesis.utils.fit_powerlaws import fit_loglog  # noqa: E402
from scripts.thesis.utils.fourier_ops import off_diagonal_fourier_energy  # noqa: E402
from scripts.thesis.utils.plotting import (  # noqa: E402
    apply_thesis_style,
    save_both,
    sequential_colors,
)
from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


D: int = 64
N: int = 128
P: int = 64
K_Q: int = 64                         # full-window query (K == P)
B_FULL: int = 32
B_QUICK: int = 32
NU: float = 1.5                        # power-law exponent of the data symbol
SIGMA: float = 0.0
L_S: int = 8                           # fixed depth, chosen large enough that
                                        # depth is not the bottleneck

R_VALUES_FULL: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
R_VALUES_QUICK: tuple[int, ...] = (1, 4, 16, 64)

SEEDS_FULL: tuple[int, ...] = (0, 1, 2, 3)
SEEDS_QUICK: tuple[int, ...] = (0, 1)

N_STEPS_FULL: int = 8000
N_STEPS_QUICK: int = 2000
LR: float = 1e-3
TAIL_WINDOW: int = 1000                # terminal = mean of last TAIL_WINDOW steps

FIT_WINDOW: tuple[float, float] = (4.0, 32.0)
THEORY_EXPONENT: float = -(NU - 1.0)   # = -0.5 for NU=1.5

FAMILIES: tuple[str, ...] = ("linear", "spectral_gd")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_key(family: str, r_val: int | None, seed: int) -> str:
    if family == "linear":
        return f"linear_seed{seed}"
    return f"spectral_r{r_val}_seed{seed}"


def _build_model(
    family: str, r_val: int | None,
) -> LinearAttention | SpectralAttention:
    if family == "linear":
        return LinearAttention(D, N, P, K_Q, L_S)
    if family == "spectral_gd":
        assert r_val is not None, "spectral_gd needs r_val"
        return SpectralAttention(D, N, P, K_Q, L_S, r=r_val, s_init="gd")
    raise ValueError(f"unknown family {family!r}")


def _align_errors(model: LinearAttention | SpectralAttention) -> dict[str, float]:
    with torch.no_grad():
        wy = model.w_y.data
        return {
            "Wx_T_wy": float(torch.linalg.vector_norm(model.W_x.data.T @ wy)),
            "wy_norm_minus_1": abs(float(wy.norm()) - 1.0),
            "Wq_wy": float(torch.linalg.vector_norm(model.W_q.data @ wy)),
            "Wk_wy": float(torch.linalg.vector_norm(model.W_k.data @ wy)),
        }


def _fft_symbol(S_TT: torch.Tensor) -> np.ndarray:
    """Length-P real symbol via FFT of the first column of ``S_TT``."""
    first_col = S_TT[:, 0].detach().cpu().to(torch.float64)
    return torch.fft.fft(first_col).real.numpy()


def _tail_sum(symbol_cpu_f64: torch.Tensor, r: int, P_: int) -> float:
    """Analytical tail contribution of modes NOT in the first ``r`` half-spectrum
    entries, normalized by ``P`` to match the operator-level scale.

        tail(r) = (1 / P) * sum_{k: k_star(k) >= r_half} symbol[k]

    where ``k_star(k) = min(k, P - k)`` is the centered-circular index and
    ``r_half = min(r, P // 2 + 1)``. Mode multiplicities (1 for DC and Nyquist,
    2 elsewhere) are handled implicitly by summing over the full P-length
    symbol.
    """
    r_half = min(int(r), P_ // 2 + 1)
    k = torch.arange(P_, dtype=torch.long)
    k_star = torch.minimum(k, P_ - k)
    mask = k_star >= r_half
    return float(symbol_cpu_f64[mask].sum().item()) / float(P_)


def _run_one(
    family: str,
    r_val: int | None,
    seed: int,
    n_steps: int,
    lr: float,
    sampler_fn,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = _build_model(family, r_val).to(device=device, dtype=dtype)
    model.enforce_alignment()

    s_half_init: np.ndarray | None = None
    if isinstance(model, SpectralAttention):
        s_half_init = model.s_half.data.detach().cpu().numpy().copy()

    result = train_icl_online(model, sampler_fn, n_steps, lr=lr)
    losses = np.asarray(result["losses"], dtype=np.float32)

    tail = min(TAIL_WINDOW, max(1, n_steps // 2))
    terminal = float(losses[-tail:].mean())
    initial = float(losses[0])
    errs = _align_errors(model)

    off_energy: float | None = None
    symbol: np.ndarray | None = None
    s_half_final: np.ndarray | None = None
    s_half_change: float | None = None
    if isinstance(model, SpectralAttention):
        S_TT = model._build_S_TT().detach().cpu().to(torch.float64)
        off_energy = float(off_diagonal_fourier_energy(S_TT))
        symbol = _fft_symbol(S_TT)
        s_half_final = model.s_half.data.detach().cpu().numpy().copy()
        s_half_change = float(np.max(np.abs(s_half_final - s_half_init)))

    return {
        "family": family,
        "r": int(r_val) if r_val is not None else None,
        "seed": seed,
        "losses": losses,
        "terminal": terminal,
        "initial": initial,
        "errs": errs,
        "off_energy": off_energy,
        "symbol": symbol,
        "s_half_init": s_half_init,
        "s_half_final": s_half_final,
        "s_half_change": s_half_change,
    }


# ---------------------------------------------------------------------------
# Per-r aggregation
# ---------------------------------------------------------------------------


def _spectral_terminal_by_r(
    results: list[dict[str, Any]], r_values: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, min, max) of spectral_gd terminal losses per r value."""
    means, los, his = [], [], []
    for r in r_values:
        vals = [
            res["terminal"] for res in results
            if res["family"] == "spectral_gd" and res["r"] == r
        ]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            means.append(float(arr.mean()))
            los.append(float(arr.min()))
            his.append(float(arr.max()))
        else:
            means.append(float("nan"))
            los.append(float("nan"))
            his.append(float("nan"))
    return np.asarray(means), np.asarray(los), np.asarray(his)


def _linear_terminal(results: list[dict[str, Any]]) -> tuple[float, float, float]:
    vals = [res["terminal"] for res in results if res["family"] == "linear"]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.min()), float(arr.max())


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------


def _fit_power_law(
    r_values: tuple[int, ...], y_values: np.ndarray,
    fit_window: tuple[float, float],
) -> dict[str, float]:
    """Wrap fit_loglog with the tensor conversions. Returns a pure-float dict."""
    x = torch.tensor(list(r_values), dtype=torch.float64)
    y = torch.tensor(y_values, dtype=torch.float64)
    out = fit_loglog(x, y, fit_window=fit_window)
    return {
        "slope": float(out["slope"]),
        "intercept": float(out["intercept"]),
        "r2": float(out["r2"]),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _anchor_r(r_values: tuple[int, ...]) -> int:
    """Middle-ish r for rescaling the tail-sum reference onto the empirical
    curve. Prefer r=8 when present; otherwise the middle element."""
    if 8 in r_values:
        return 8
    return r_values[len(r_values) // 2]


def _plot_rank_floor(
    results: list[dict[str, Any]],
    r_values: tuple[int, ...],
    tail_sums: np.ndarray,
    emp_fit: dict[str, float],
    tail_fit: dict[str, float],
    run_dir,
) -> None:
    emp_mean, emp_lo, emp_hi = _spectral_terminal_by_r(results, r_values)
    lin_mean, lin_lo, lin_hi = _linear_terminal(results)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    r_arr = np.asarray(r_values, dtype=np.float64)
    yerr = np.vstack([emp_mean - emp_lo, emp_hi - emp_mean])
    ax.errorbar(
        r_arr, emp_mean, yerr=yerr, color="C0", marker="o", ms=6, lw=1.4,
        capsize=3, label="SpectralAttention(gd, r) (mean +/- seed range)",
    )

    # Rescaled tail reference at the anchor r (shape comparison).
    anchor = _anchor_r(r_values)
    anchor_idx = r_values.index(anchor)
    if tail_sums[anchor_idx] > 0 and np.isfinite(emp_mean[anchor_idx]):
        scale = emp_mean[anchor_idx] / tail_sums[anchor_idx]
    else:
        scale = 1.0
    tail_rescaled = tail_sums * scale
    ax.plot(
        r_arr, tail_rescaled, color="C3", lw=1.2, ls="--",
        label=f"analytical tail (rescaled at r={anchor})",
    )

    # Linear reference as a horizontal band.
    if np.isfinite(lin_mean):
        ax.axhline(lin_mean, color="C2", lw=1.2, ls=":", label=f"LinearAttention mean")
        if np.isfinite(lin_lo) and np.isfinite(lin_hi):
            ax.axhspan(lin_lo, lin_hi, color="C2", alpha=0.12, linewidth=0)

    # Fitted empirical power law line.
    lo, hi = FIT_WINDOW
    r_grid = np.logspace(math.log10(lo), math.log10(hi), 40)
    emp_line = math.exp(emp_fit["intercept"]) * r_grid ** emp_fit["slope"]
    ax.plot(
        r_grid, emp_line, color="black", lw=1.2, ls="-.",
        label=(
            f"empirical power-law fit (slope = {emp_fit['slope']:.3f}, "
            f"R^2 = {emp_fit['r2']:.3f})"
        ),
    )
    ax.axvspan(lo, hi, color="0.85", alpha=0.4, linewidth=0, zorder=0)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("spectral bottleneck r")
    ax.set_ylabel("terminal loss (mean of last 1000 steps)")
    ax.set_title(
        f"Rank-floor scaling (nu = {NU:.1f}; theory slope = {THEORY_EXPONENT:.2f})",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save_both(fig, run_dir, "rank_floor")
    plt.close(fig)


def _plot_rank_loss_vs_steps(
    results: list[dict[str, Any]],
    r_values: tuple[int, ...],
    run_dir,
    r_subset: tuple[int, ...] = (1, 4, 16, 64),
) -> None:
    r_plot = tuple(r for r in r_subset if r in r_values)
    if not r_plot:
        r_plot = r_values
    colors = sequential_colors(len(r_plot), palette="rocket")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for ci, r in enumerate(r_plot):
        subset = [
            res for res in results
            if res["family"] == "spectral_gd" and res["r"] == r
        ]
        if not subset:
            continue
        stacked = np.stack([res["losses"] for res in subset])
        mean = stacked.mean(axis=0)
        lo, hi = stacked.min(axis=0), stacked.max(axis=0)
        steps = np.arange(mean.shape[0])
        ax.plot(steps, mean, color=colors[ci], lw=1.2, label=f"r = {r}")
        ax.fill_between(steps, lo, hi, color=colors[ci], alpha=0.2, linewidth=0)

    # Linear reference.
    lin_runs = [res for res in results if res["family"] == "linear"]
    if lin_runs:
        stacked_lin = np.stack([res["losses"] for res in lin_runs])
        mean_lin = stacked_lin.mean(axis=0)
        ax.plot(
            np.arange(mean_lin.shape[0]), mean_lin,
            color="C2", lw=1.2, ls=":", label="LinearAttention",
        )

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (mean over seeds)")
    ax.set_title("Training loss at selected r values", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save_both(fig, run_dir, "rank_loss_vs_steps")
    plt.close(fig)


def _plot_rank_learned_symbol(
    results: list[dict[str, Any]],
    r_values: tuple[int, ...],
    run_dir,
    data_symbol: np.ndarray,
    r_subset: tuple[int, ...] = (4, 16, 64),
) -> None:
    r_plot = tuple(r for r in r_subset if r in r_values)
    if not r_plot:
        r_plot = (r_values[-1],)

    fig, axes = plt.subplots(1, len(r_plot), figsize=(4.2 * len(r_plot), 3.8), sharey=True)
    if len(r_plot) == 1:
        axes = [axes]  # normalize to iterable
    P_x = np.arange(P)
    for ax, r, color in zip(axes, r_plot, sequential_colors(len(r_plot), palette="rocket")):
        subset = [
            res for res in results
            if res["family"] == "spectral_gd" and res["r"] == r and res["symbol"] is not None
        ]
        if not subset:
            ax.set_title(f"r = {r}: no data", fontsize=9)
            continue
        syms = np.stack([res["symbol"] for res in subset])
        mean_sym = syms.mean(axis=0)
        lo, hi = syms.min(axis=0), syms.max(axis=0)
        ax.plot(P_x, mean_sym, color=color, lw=1.4, label=f"learned S_TT symbol (mean)")
        ax.fill_between(P_x, lo, hi, color=color, alpha=0.25, linewidth=0)
        # Reference GD init symbol: [-P, 0, ..., 0].
        ref = np.zeros(P)
        ref[0] = -float(P)
        ax.plot(P_x, ref, color="black", lw=0.8, ls=":", label="GD init [-P, 0, ...]")
        ax.axhline(0.0, color="0.5", lw=0.4)
        # Shaded region = modes outside the bottleneck (k_star >= r_half).
        r_half = min(r, P // 2 + 1)
        # For visual clarity we shade full-spectrum indices with k_star >= r_half.
        k = np.arange(P)
        k_star = np.minimum(k, P - k)
        outside = k_star >= r_half
        if outside.any():
            for k_i in np.where(outside)[0]:
                ax.axvspan(k_i - 0.5, k_i + 0.5, color="gray", alpha=0.08, linewidth=0, zorder=0)
        ax.set_xlabel("Fourier mode k")
        ax.set_title(f"r = {r}  (r_half = {r_half})", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("symbol s[k]")
        ax.legend(fontsize=7)
    fig.suptitle("Learned S_TT symbols at convergence (gray bands = modes outside bottleneck)", fontsize=10)
    fig.tight_layout()
    save_both(fig, run_dir, "rank_learned_symbol")
    plt.close(fig)


def _plot_rank_exponent_comparison(
    emp_fit: dict[str, float],
    tail_fit: dict[str, float],
    theory_slope: float,
    run_dir,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    labels = ["empirical\nfit", "analytical\ntail fit", f"theory\n-(nu-1)"]
    values = [emp_fit["slope"], tail_fit["slope"], theory_slope]
    colors = ["C0", "C3", "black"]
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + (0.04 if v < 0 else -0.04),
            f"{v:.3f}",
            ha="center", va=("bottom" if v < 0 else "top"), fontsize=9,
        )
    ax.axhline(0.0, color="0.4", lw=0.5)
    ax.set_ylabel("fitted slope (log-log loss vs r)")
    ax.set_title(
        f"Slope comparison  (empirical R^2 = {emp_fit['r2']:.3f}, "
        f"tail R^2 = {tail_fit['r2']:.3f})",
        fontsize=9,
    )
    fig.tight_layout()
    save_both(fig, run_dir, "rank_exponent_comparison")
    plt.close(fig)


def _make_all_figures(
    results: list[dict[str, Any]],
    r_values: tuple[int, ...],
    tail_sums: np.ndarray,
    data_symbol: np.ndarray,
    emp_fit: dict[str, float],
    tail_fit: dict[str, float],
    run_dir,
) -> None:
    apply_thesis_style()
    _plot_rank_floor(results, r_values, tail_sums, emp_fit, tail_fit, run_dir)
    _plot_rank_loss_vs_steps(results, r_values, run_dir)
    _plot_rank_learned_symbol(results, r_values, run_dir, data_symbol=data_symbol)
    _plot_rank_exponent_comparison(emp_fit, tail_fit, THEORY_EXPONENT, run_dir)


# ---------------------------------------------------------------------------
# Acceptance + summary
# ---------------------------------------------------------------------------


def _compute_acceptance(
    results: list[dict[str, Any]],
    r_values: tuple[int, ...],
    tail_sums: np.ndarray,
    emp_fit: dict[str, float],
    tail_fit: dict[str, float],
) -> dict[str, Any]:
    emp_mean, _, _ = _spectral_terminal_by_r(results, r_values)
    lin_mean, _, _ = _linear_terminal(results)

    # 1. Monotone non-increasing in r.
    monotone_ok = all(
        emp_mean[i] >= emp_mean[i + 1] - 1e-12
        for i in range(len(emp_mean) - 1)
    )

    # Convenience lookups.
    r_to_mean = {r: float(m) for r, m in zip(r_values, emp_mean)}

    # 2. Bottleneck exists: terminal(r=4) / terminal(r=r_max) > 1.5.
    r_max = max(r_values)
    r_bot = 4 if 4 in r_to_mean else r_values[1] if len(r_values) > 1 else r_values[0]
    bottleneck_ratio = (
        r_to_mean[r_bot] / max(r_to_mean[r_max], 1e-30)
        if r_bot in r_to_mean and r_max in r_to_mean else float("nan")
    )
    bottleneck_ok = np.isfinite(bottleneck_ratio) and bottleneck_ratio > 1.5

    # 3. Power-law fit quality.
    fit_quality_ok = emp_fit["r2"] > 0.90

    # 4. Exponent reasonable: slope in [-1.5, -0.1].
    exp_ok = -1.5 <= emp_fit["slope"] <= -0.1

    # 5. LinearAttention terminal is between terminal(r_max) and terminal(r=1).
    r_low = 1 if 1 in r_to_mean else r_values[0]
    linear_in_range_ok = (
        np.isfinite(lin_mean)
        and r_to_mean.get(r_max, float("nan")) - 1e-9 <= lin_mean <= r_to_mean.get(r_low, float("inf")) + 1e-9
    )

    # 6. Circulant preservation.
    circ_vals = [
        res["off_energy"] for res in results
        if res["family"] == "spectral_gd" and res["off_energy"] is not None
    ]
    circ_ok = all(v < 0.01 for v in circ_vals) if circ_vals else False
    max_off = max(circ_vals) if circ_vals else 0.0

    # 7. Alignment.
    align_maxes = [max(res["errs"].values()) for res in results]
    max_align = max(align_maxes) if align_maxes else 0.0
    align_ok = max_align < 1e-5

    all_ok = (
        monotone_ok and bottleneck_ok and fit_quality_ok and exp_ok
        and linear_in_range_ok and circ_ok and align_ok
    )

    return {
        "all_pass": all_ok,
        "monotone": {"ok": monotone_ok, "means": emp_mean.tolist()},
        "bottleneck": {
            "ok": bottleneck_ok, "ratio": float(bottleneck_ratio),
            "r_low": int(r_bot), "r_high": int(r_max),
            "terminal_low": float(r_to_mean.get(r_bot, float("nan"))),
            "terminal_high": float(r_to_mean.get(r_max, float("nan"))),
        },
        "fit_quality": {"ok": fit_quality_ok, "r2": float(emp_fit["r2"])},
        "exponent": {
            "ok": exp_ok, "slope": float(emp_fit["slope"]),
            "bounds": (-1.5, -0.1), "theory": float(THEORY_EXPONENT),
            "tail_slope": float(tail_fit["slope"]),
            "tail_r2": float(tail_fit["r2"]),
        },
        "linear_in_range": {
            "ok": linear_in_range_ok, "linear_mean": float(lin_mean),
            "bracket_low": float(r_to_mean.get(r_max, float("nan"))),
            "bracket_high": float(r_to_mean.get(r_low, float("nan"))),
            "r_low": int(r_low), "r_high": int(r_max),
        },
        "circ": {"ok": circ_ok, "max_off_energy": float(max_off)},
        "align": {"ok": align_ok, "max_align_err": float(max_align)},
    }


def _format_summary(
    acceptance: dict[str, Any],
    results: list[dict[str, Any]],
    r_values: tuple[int, ...],
    tail_sums: np.ndarray,
    emp_fit: dict[str, float],
    tail_fit: dict[str, float],
    n_steps: int,
    seeds: tuple[int, ...],
) -> str:
    gate = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    lines: list[str] = []
    lines.append("Architecture-aligned experiment: spectral rank bottleneck on stationary data.")
    lines.append("")
    lines.append(
        f"Total runs: {len(results)}; "
        f"D = {D}, N = {N}, P = {P}, K = {K_Q}, B = {B_FULL if n_steps == N_STEPS_FULL else B_QUICK}, "
        f"L_S = {L_S}, nu = {NU}, sigma = {SIGMA}"
    )
    lines.append(f"r_values = {list(r_values)}, seeds = {list(seeds)}, n_steps = {n_steps}, lr = {LR}")
    lines.append(f"fit_window (in r) = {FIT_WINDOW}, theory exponent -(nu-1) = {THEORY_EXPONENT:.3f}")
    lines.append("")

    lines.append(f"[{gate(acceptance['all_pass'])}] overall")
    lines.append("")

    m = acceptance["monotone"]
    lines.append(
        f"[{gate(m['ok'])}] Monotone non-increasing terminal loss in r: "
        f"means = [" + ", ".join(f"{v:.3e}" for v in m["means"]) + "]"
    )

    b = acceptance["bottleneck"]
    lines.append(
        f"[{gate(b['ok'])}] Bottleneck exists (terminal(r={b['r_low']}) / terminal(r={b['r_high']}) > 1.5): "
        f"{b['terminal_low']:.3e} / {b['terminal_high']:.3e} = {b['ratio']:.3f}"
    )

    f = acceptance["fit_quality"]
    lines.append(
        f"[{gate(f['ok'])}] Power-law fit R^2 > 0.90: R^2 = {f['r2']:.4f}"
    )

    e = acceptance["exponent"]
    lines.append(
        f"[{gate(e['ok'])}] Fitted exponent in {e['bounds']}: "
        f"empirical slope = {e['slope']:.3f}, "
        f"tail-sum slope = {e['tail_slope']:.3f}, "
        f"theory = {e['theory']:.3f}"
    )

    lr = acceptance["linear_in_range"]
    lines.append(
        f"[{gate(lr['ok'])}] LinearAttention terminal in "
        f"[terminal(r={lr['r_high']}), terminal(r={lr['r_low']})]: "
        f"linear = {lr['linear_mean']:.3e}, bracket = "
        f"[{lr['bracket_low']:.3e}, {lr['bracket_high']:.3e}]"
    )

    ci = acceptance["circ"]
    lines.append(
        f"[{gate(ci['ok'])}] Circulant preservation (off_energy < 0.01): "
        f"max = {ci['max_off_energy']:.2e}"
    )

    al = acceptance["align"]
    lines.append(
        f"[{gate(al['ok'])}] Assumption 1 alignment (< 1e-5): "
        f"max = {al['max_align_err']:.2e}"
    )

    lines.append("")
    lines.append("Per-r summary (spectral_gd, mean over seeds):")
    lines.append("  r  | terminal   | tail_sum   | tail*scale_at_anchor")
    lines.append("  ---+------------+------------+---------------------")
    emp_mean, _, _ = _spectral_terminal_by_r(results, r_values)
    anchor = _anchor_r(r_values)
    anchor_idx = r_values.index(anchor)
    scale = (
        emp_mean[anchor_idx] / tail_sums[anchor_idx]
        if tail_sums[anchor_idx] > 0 and np.isfinite(emp_mean[anchor_idx]) else 1.0
    )
    for i, r in enumerate(r_values):
        lines.append(
            f"  {r:>2d} | {emp_mean[i]:10.3e} | {tail_sums[i]:10.3e} | "
            f"{tail_sums[i] * scale:10.3e}"
        )
    lin_mean, _, _ = _linear_terminal(results)
    lines.append(f"  LinearAttention mean terminal: {lin_mean:.3e}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Data IO
# ---------------------------------------------------------------------------


def _save_losses(results: list[dict[str, Any]], run_dir: ThesisRunDir) -> None:
    d: dict[str, np.ndarray] = {}
    for res in results:
        d[_run_key(res["family"], res["r"], res["seed"])] = res["losses"]
    np.savez_compressed(run_dir.npz_path("losses"), **d)


def _save_terminal(results: list[dict[str, Any]], run_dir: ThesisRunDir) -> None:
    keys = [_run_key(res["family"], res["r"], res["seed"]) for res in results]
    np.savez_compressed(
        run_dir.npz_path("terminal_losses"),
        keys=np.array(keys),
        family=np.array([res["family"] for res in results]),
        r=np.array([res["r"] if res["r"] is not None else -1 for res in results], dtype=np.int64),
        seed=np.array([res["seed"] for res in results], dtype=np.int64),
        terminal=np.array([res["terminal"] for res in results], dtype=np.float64),
        initial=np.array([res["initial"] for res in results], dtype=np.float64),
        max_align_err=np.array(
            [max(res["errs"].values()) for res in results], dtype=np.float64,
        ),
        off_energy=np.array(
            [res["off_energy"] if res["off_energy"] is not None else np.nan for res in results],
            dtype=np.float64,
        ),
        s_half_change=np.array(
            [res["s_half_change"] if res["s_half_change"] is not None else np.nan for res in results],
            dtype=np.float64,
        ),
    )


def _save_symbols(results: list[dict[str, Any]], run_dir: ThesisRunDir) -> None:
    d: dict[str, np.ndarray] = {}
    for res in results:
        if res["symbol"] is None:
            continue
        key = _run_key(res["family"], res["r"], res["seed"])
        d[f"{key}__symbol"] = res["symbol"]
        if res["s_half_init"] is not None:
            d[f"{key}__s_half_init"] = res["s_half_init"]
        if res["s_half_final"] is not None:
            d[f"{key}__s_half_final"] = res["s_half_final"]
    np.savez_compressed(run_dir.npz_path("symbols"), **d)


def _save_spectra(
    sampler_meta: dict[str, Any],
    r_values: tuple[int, ...],
    tail_sums: np.ndarray,
    emp_fit: dict[str, float],
    tail_fit: dict[str, float],
    run_dir: ThesisRunDir,
) -> None:
    np.savez_compressed(
        run_dir.npz_path("spectra"),
        data_symbol=sampler_meta["symbol"].cpu().numpy(),
        data_sqrt_symbol=sampler_meta["sqrt_symbol"].cpu().numpy(),
        r_values=np.asarray(r_values, dtype=np.int64),
        tail_sums=tail_sums,
        nu=np.float64(NU),
        theory_exponent=np.float64(THEORY_EXPONENT),
        emp_slope=np.float64(emp_fit["slope"]),
        emp_intercept=np.float64(emp_fit["intercept"]),
        emp_r2=np.float64(emp_fit["r2"]),
        tail_slope=np.float64(tail_fit["slope"]),
        tail_intercept=np.float64(tail_fit["intercept"]),
        tail_r2=np.float64(tail_fit["r2"]),
        fit_window=np.asarray(FIT_WINDOW, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Reprocess
# ---------------------------------------------------------------------------


class _ExistingRunDir:
    """Minimal :class:`ThesisRunDir` stand-in for reprocess mode."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {self.root}")
        self.figures = self.root / "figures"
        self.pdfs = self.root / "pdfs"
        self.npz = self.root / "npz"
        self.pt = self.root / "pt"
        for d in (self.figures, self.pdfs, self.npz, self.pt):
            d.mkdir(parents=True, exist_ok=True)
        self.script_stem = self.root.parent.name
        self.run_id = self.root.name
        self.phase = self.root.parent.parent.name

    def png(self, name: str) -> Path:
        return self.figures / f"{name}.png"

    def pdf(self, name: str) -> Path:
        return self.pdfs / f"{name}.pdf"

    def npz_path(self, name: str) -> Path:
        return self.npz / f"{name}.npz"

    def pt_path(self, name: str) -> Path:
        return self.pt / f"{name}.pt"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.txt"


def _load_from_run_dir(run_dir_path: Path) -> tuple[
    list[dict[str, Any]], tuple[int, ...], np.ndarray, np.ndarray,
]:
    """Load per-run results, r_values, tail_sums, and data_symbol from NPZ."""
    losses_path = run_dir_path / "npz" / "losses.npz"
    terminal_path = run_dir_path / "npz" / "terminal_losses.npz"
    symbols_path = run_dir_path / "npz" / "symbols.npz"
    spectra_path = run_dir_path / "npz" / "spectra.npz"
    for p in (losses_path, terminal_path, symbols_path, spectra_path):
        if not p.exists():
            raise FileNotFoundError(f"missing expected npz file: {p}")

    losses_npz = np.load(losses_path)
    terminal_npz = np.load(terminal_path)
    symbols_npz = np.load(symbols_path)
    spectra_npz = np.load(spectra_path)

    keys = [str(k) for k in terminal_npz["keys"]]
    families = [str(f) for f in terminal_npz["family"]]
    rs = terminal_npz["r"].tolist()
    seeds = terminal_npz["seed"].tolist()
    terminals = terminal_npz["terminal"].tolist()
    initials = terminal_npz["initial"].tolist()
    max_align_errs = terminal_npz["max_align_err"].tolist()
    off_energies = terminal_npz["off_energy"].tolist()
    s_half_changes = terminal_npz["s_half_change"].tolist()

    results: list[dict[str, Any]] = []
    for i, key in enumerate(keys):
        family = families[i]
        losses_arr = np.asarray(losses_npz[key]).astype(np.float32)
        symbol: np.ndarray | None = None
        s_half_init: np.ndarray | None = None
        s_half_final: np.ndarray | None = None
        if family == "spectral_gd":
            if f"{key}__symbol" in symbols_npz.files:
                symbol = np.asarray(symbols_npz[f"{key}__symbol"])
            if f"{key}__s_half_init" in symbols_npz.files:
                s_half_init = np.asarray(symbols_npz[f"{key}__s_half_init"])
            if f"{key}__s_half_final" in symbols_npz.files:
                s_half_final = np.asarray(symbols_npz[f"{key}__s_half_final"])
        off_val = float(off_energies[i])
        off_energy = None if math.isnan(off_val) else off_val
        sh_val = float(s_half_changes[i])
        s_half_change = None if math.isnan(sh_val) else sh_val
        r_val: int | None = None if int(rs[i]) == -1 else int(rs[i])

        results.append({
            "family": family,
            "r": r_val,
            "seed": int(seeds[i]),
            "losses": losses_arr,
            "terminal": float(terminals[i]),
            "initial": float(initials[i]),
            "errs": {"max_align_err": float(max_align_errs[i])},
            "off_energy": off_energy,
            "symbol": symbol,
            "s_half_init": s_half_init,
            "s_half_final": s_half_final,
            "s_half_change": s_half_change,
        })

    r_values = tuple(int(r) for r in spectra_npz["r_values"].tolist())
    tail_sums = np.asarray(spectra_npz["tail_sums"], dtype=np.float64)
    data_symbol = np.asarray(spectra_npz["data_symbol"], dtype=np.float64)
    return results, r_values, tail_sums, data_symbol


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_seeds(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    return tuple(int(s.strip()) for s in raw.split(",") if s.strip())


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Train SpectralAttention at varying spectral bottleneck r on "
            "stationary circulant data and fit the rank-floor power law. "
            "Supports --quick for login-node sanity runs and --reprocess "
            "for regenerating figures + summary from an existing run."
        )
    )
    ap.add_argument("--device", type=str, default="cuda", choices=("cuda",))
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seed override")
    ap.add_argument("--quick", action="store_true",
                    help="quick mode: smaller r grid, fewer seeds, fewer steps")
    ap.add_argument("--reprocess", type=str, default=None,
                    help="path to existing run dir to regenerate figures + summary from NPZ")
    return ap.parse_args()


def _main_train(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    dtype = torch.float32

    if args.quick:
        r_values = R_VALUES_QUICK
        seeds = _parse_seeds(args.seeds, SEEDS_QUICK)
        n_steps = N_STEPS_QUICK
        B = B_QUICK
    else:
        r_values = R_VALUES_FULL
        seeds = _parse_seeds(args.seeds, SEEDS_FULL)
        n_steps = N_STEPS_FULL
        B = B_FULL

    total_runs = len(seeds) * (1 + len(r_values))  # linear + spectral per seed

    apply_thesis_style()
    run_dir = ThesisRunDir(__file__, phase="architectures")

    print(f"device:  {device} ({torch.cuda.get_device_name(0)})")
    print(f"dtype:   {dtype}   quick={args.quick}")
    print(f"r_values = {list(r_values)}   seeds = {list(seeds)}   n_steps = {n_steps}")
    print(f"output: {run_dir.root}")

    # Build the stationary sampler once -- all models share the same data.
    sampler_fn, sampler_meta = make_stationary_sampler(
        P=P, D=D, K=K_Q, B=B, symbol_kind="power_law", nu=NU, sigma=SIGMA,
        device=device, dtype=dtype,
    )
    data_symbol_cpu_f64 = sampler_meta["symbol"]

    # Precompute analytical tail sums per r (operator-level reference).
    tail_sums = np.asarray(
        [_tail_sum(data_symbol_cpu_f64, r, P) for r in r_values],
        dtype=np.float64,
    )

    cfg_json = {
        "D": D, "N": N, "P": P, "K": K_Q, "B": B, "L_S": L_S, "nu": NU,
        "sigma": SIGMA, "r_values": list(r_values), "seeds": list(seeds),
        "n_steps": n_steps, "lr": LR, "tail_window": TAIL_WINDOW,
        "fit_window": list(FIT_WINDOW), "theory_exponent": THEORY_EXPONENT,
        "families": list(FAMILIES), "dtype": "float32", "device": "cuda",
        "quick": bool(args.quick),
    }

    with RunContext(run_dir, config=cfg_json, seeds=list(seeds)) as ctx:
        results: list[dict[str, Any]] = []
        run_idx = 0
        t_start = time.time()

        # Order: for each seed, run linear first then all spectral r values.
        # This way the data RNG progression is deterministic per seed.
        for seed in seeds:
            # Linear
            run_idx += 1
            t0 = time.time()
            res = _run_one(
                family="linear", r_val=None, seed=seed,
                n_steps=n_steps, lr=LR, sampler_fn=sampler_fn,
                device=device, dtype=dtype,
            )
            dt = time.time() - t0
            ctx.record_step_time(dt)
            results.append(res)
            eta = dt * (total_runs - run_idx)
            print(
                f"[{run_idx:>3}/{total_runs}] linear           "
                f"seed={seed}: init={res['initial']:.3e} term={res['terminal']:.3e} "
                f"align={max(res['errs'].values()):.1e} wall={dt:.2f}s  "
                f"eta={eta/60:.1f}min",
                flush=True,
            )
            # Spectral r sweep
            for r in r_values:
                run_idx += 1
                t0 = time.time()
                res = _run_one(
                    family="spectral_gd", r_val=r, seed=seed,
                    n_steps=n_steps, lr=LR, sampler_fn=sampler_fn,
                    device=device, dtype=dtype,
                )
                dt = time.time() - t0
                ctx.record_step_time(dt)
                results.append(res)
                eta = dt * (total_runs - run_idx)
                print(
                    f"[{run_idx:>3}/{total_runs}] spectral_r{r:<3d}     "
                    f"seed={seed}: init={res['initial']:.3e} term={res['terminal']:.3e} "
                    f"align={max(res['errs'].values()):.1e} "
                    f"off_energy={res['off_energy']:.1e} wall={dt:.2f}s  "
                    f"eta={eta/60:.1f}min",
                    flush=True,
                )

        total_wall = time.time() - t_start
        print(f"all runs complete in {total_wall / 60:.2f} min", flush=True)

        # Fits (empirical and tail).
        emp_mean, _, _ = _spectral_terminal_by_r(results, r_values)
        emp_fit = _fit_power_law(r_values, emp_mean, FIT_WINDOW)
        tail_fit = _fit_power_law(r_values, tail_sums, FIT_WINDOW)

        # Persist data.
        _save_losses(results, run_dir)
        _save_terminal(results, run_dir)
        _save_symbols(results, run_dir)
        _save_spectra(sampler_meta, r_values, tail_sums, emp_fit, tail_fit, run_dir)

        # Figures.
        _make_all_figures(
            results, r_values, tail_sums, data_symbol_cpu_f64.cpu().numpy(),
            emp_fit, tail_fit, run_dir,
        )

        # Acceptance + summary.
        acceptance = _compute_acceptance(results, r_values, tail_sums, emp_fit, tail_fit)
        summary_text = _format_summary(
            acceptance, results, r_values, tail_sums, emp_fit, tail_fit,
            n_steps=n_steps, seeds=seeds,
        )
        ctx.write_summary(summary_text)
        ctx.record_compute_proxy(float(total_runs) * float(n_steps))

        print("\n" + summary_text, flush=True)

    return 0 if acceptance["all_pass"] else 1


def _main_reprocess(args: argparse.Namespace) -> int:
    run_dir_path = Path(args.reprocess).resolve()
    print(f"reprocessing existing run directory: {run_dir_path}")

    results, r_values, tail_sums, data_symbol = _load_from_run_dir(run_dir_path)
    print(f"loaded {len(results)} runs from NPZ; r_values={list(r_values)}")

    run_dir = _ExistingRunDir(run_dir_path)

    emp_mean, _, _ = _spectral_terminal_by_r(results, r_values)
    emp_fit = _fit_power_law(r_values, emp_mean, FIT_WINDOW)
    tail_fit = _fit_power_law(r_values, tail_sums, FIT_WINDOW)

    apply_thesis_style()
    _make_all_figures(results, r_values, tail_sums, data_symbol, emp_fit, tail_fit, run_dir)

    acceptance = _compute_acceptance(results, r_values, tail_sums, emp_fit, tail_fit)
    seeds = tuple(sorted({res["seed"] for res in results}))
    # Infer n_steps from the loss array length of the first run.
    n_steps_inferred = int(results[0]["losses"].shape[0]) if results else 0
    summary_text = _format_summary(
        acceptance, results, r_values, tail_sums, emp_fit, tail_fit,
        n_steps=n_steps_inferred, seeds=seeds,
    )

    summary_path = run_dir.summary_path
    original_header_lines: list[str] = []
    if summary_path.exists():
        original = summary_path.read_text(encoding="utf-8")
        if original:
            for line in original.splitlines():
                if not line.strip():
                    break
                original_header_lines.append(line)
    reprocess_stamp = (
        "reprocessed_utc: "
        f"{__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}"
    )
    header_block = original_header_lines + [
        reprocess_stamp,
        "reprocess_note: acceptance + fits recomputed; no retraining.",
    ]
    text = "\n".join(header_block) + "\n\n" + summary_text
    summary_path.write_text(text, encoding="utf-8")

    print("\n" + summary_text, flush=True)
    return 0 if acceptance["all_pass"] else 1


def main() -> int:
    args = _cli()
    if args.reprocess is not None:
        return _main_reprocess(args)
    return _main_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
