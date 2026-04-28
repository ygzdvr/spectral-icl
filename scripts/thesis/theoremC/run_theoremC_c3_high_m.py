"""High-m companion to the C3 patch: showcases tightening of the Cor. 3.13
prediction as block size ``m`` grows.

Runs the same closed-form vs. Chebyshev-at-q♯ comparison as
``run_theoremC_c3_patch.py``, but at ``D = 512`` with ``m ∈
{2, 4, 8, 16, 32, 64, 128, 256, 512}`` so that the per-block condition
number κ controls the asymptotics while the within-block mass count grows.

Output
------
``outputs/thesis/theoremC/c3_high_m/<run_id>/figures/``
    c3_obstruction_vs_kappa_high_m.(png|pdf)
    c3_chebyshev_ratio_high_m.(png|pdf)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np
import torch

from scripts.thesis.theoremC.run_theoremC_c3_patch import (
    C3PatchConfig,
    _build_grid,
    _run_trial,
)
from scripts.thesis.utils.plotting import (
    apply_thesis_style,
    save_both,
    sequential_colors,
)
from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir


def _plot_obstruction_vs_kappa_high_m(
    cfg: C3PatchConfig,
    trials: list[dict[str, Any]],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    L_grid = _build_grid(trials, cfg, "L_cf")
    C_grid = _build_grid(trials, cfg, "chebyshev_total")
    m_list = list(cfg.partition_m_list)
    k_arr = np.asarray(cfg.kappa_list, dtype=float)

    plotted_m_rows = [
        (i, int(m)) for i, m in enumerate(m_list) if int(m) > 1
    ]
    colors = sequential_colors(len(plotted_m_rows), palette="mako")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    floor = 1e-18
    for color, (i, m) in zip(colors, plotted_m_rows):
        y_exact = L_grid[i, :]
        y_exact = np.where(y_exact > floor, y_exact, np.nan)
        y_bound = C_grid[i, :]
        y_bound = np.where(y_bound > floor, y_bound, np.nan)
        ax.plot(
            k_arr, y_exact, color=color, lw=1.5, marker="o", ms=4.0,
            label=f"m = {m} (exact)",
        )
        ax.plot(
            k_arr, y_bound, color=color, lw=1.0, ls="--",
            marker=None,
        )
    ax.plot(
        [], [], color="gray", lw=1.0, ls="--",
        label=r"Cor. prediction $\Sigma\,\omega\lambda(1-\eta^{\sharp}\lambda)^{2}$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"within-block heterogeneity $\kappa$")
    ax.set_ylabel(
        r"theorem-C obstruction $L^\star_{L=1}(m, \kappa)$"
    )
    ax.legend(fontsize=7.5, loc="upper left", ncol=1)
    ax.text(
        0.02, 0.02,
        "solid: exact closed-form optimum   dashed: Cor. prediction at $q^{\\sharp}$",
        transform=ax.transAxes, fontsize=7.5, color="black",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                  edgecolor="gray", linewidth=0.5),
    )
    fig.tight_layout()
    save_both(fig, run_dir, "c3_obstruction_vs_kappa_high_m")
    plt.close(fig)


def _plot_chebyshev_ratio_high_m(
    cfg: C3PatchConfig,
    trials: list[dict[str, Any]],
    run_dir: ThesisRunDir,
) -> None:
    import matplotlib.pyplot as plt

    L_grid = _build_grid(trials, cfg, "L_cf")
    C_grid = _build_grid(trials, cfg, "chebyshev_total")
    m_list = list(cfg.partition_m_list)
    k_arr = np.asarray(cfg.kappa_list, dtype=float)

    plotted_m_rows = [
        (i, int(m)) for i, m in enumerate(m_list) if int(m) > 1
    ]
    colors = sequential_colors(len(plotted_m_rows), palette="mako")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for color, (i, m) in zip(colors, plotted_m_rows):
        ratio = np.where(
            C_grid[i, :] > 1e-18, L_grid[i, :] / C_grid[i, :], np.nan
        )
        ax.plot(
            k_arr, ratio, color=color, lw=1.5, marker="o", ms=4.0,
            label=f"m = {m}",
        )
    ax.axhline(1.0, color="red", lw=1.0, ls="--", label="bound = exact (tight)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"within-block heterogeneity $\kappa$")
    ax.set_ylabel(r"$L^\star_\mathrm{cf}\,/\,L^\mathrm{Cheb}$")
    ax.legend(fontsize=7.5, loc="lower left", ncol=1)
    fig.tight_layout()
    save_both(fig, run_dir, "c3_chebyshev_ratio_high_m")
    plt.close(fig)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "C3 high-m companion: obstruction-vs-κ with large block sizes "
            "to illustrate tightening of the Cor. prediction."
        )
    )
    p.add_argument("--device", type=str, default="cuda",
                   choices=("cpu", "cuda", "auto"))
    p.add_argument("--dtype", type=str, default="float64",
                   choices=("float32", "float64"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--D", type=int, default=512)
    p.add_argument("--m-list", type=str, default="64,128,256,512")
    p.add_argument("--kappa-list", type=str,
                   default="1.0,1.1,1.2,1.5,2.0,3.0,5.0,10.0")
    return p.parse_args()


def _parse_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def main() -> int:
    args = _cli()
    cfg = replace(
        C3PatchConfig(),
        D=int(args.D),
        partition_m_list=_parse_ints(args.m_list),
        kappa_list=_parse_floats(args.kappa_list),
        dtype=args.dtype,
        device=args.device,
    )
    print(f"[C3-HIGH-M] D = {cfg.D}  m_list = {list(cfg.partition_m_list)}  "
          f"kappa_list = {list(cfg.kappa_list)}")

    synthetic_script = Path(__file__).parent / "c3_high_m.py"
    run = ThesisRunDir(synthetic_script, phase="theoremC")
    print(f"[C3-HIGH-M] output: {run.root}")

    with RunContext(
        run,
        config=cfg,
        seeds=[0, 1, 2, 3],
        notes=(
            "C3 high-m companion: obstruction-vs-κ with m up to D, "
            "illustrating that the Cor. prediction tightens as m grows."
        ),
    ) as ctx:
        apply_thesis_style()

        trials: list[dict[str, Any]] = []
        n_total = len(cfg.partition_m_list) * len(cfg.kappa_list)
        idx = 0
        t0 = time.perf_counter()
        for m in cfg.partition_m_list:
            for kappa in cfg.kappa_list:
                trial = _run_trial(cfg, int(m), float(kappa))
                trials.append(trial)
                idx += 1
                print(f"[{idx:3d}/{n_total}] m = {m:4d}  κ = {kappa:5.2f}  "
                      f"L_cf = {trial['L_cf']:.4e}  "
                      f"L_Cheb = {trial['chebyshev_total']:.4e}")
        print(f"[C3-HIGH-M] sweep done in {time.perf_counter() - t0:.1f} s")

        _plot_obstruction_vs_kappa_high_m(cfg, trials, run)
        _plot_chebyshev_ratio_high_m(cfg, trials, run)

        ctx.record_extra("n_trials", len(trials))
        print(f"[C3-HIGH-M] figures: {run.figures}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
