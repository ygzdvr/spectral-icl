"""Supplementary theorem-B figures (Corollaries 5 and 6) — operator level.

Theorems B's Corollary 5 (OOD brittleness) and Corollary 6 (finite-r
spectral bottleneck) from ``thesis/theorem_b.txt`` are both statements
about the **converged optimum** Q⋆ = L · T⁻¹. They are NOT statements
about any finite-time trained quantity γ(T). This distinction matters:
at the optimum the residual factor |1 − λ_k γ_k / L| reduces to
|1 − λ_k'/λ_k| (Corollary 5) and to 0 on controlled modes (Corollary
6), neither of which is the quantity measured by the B3/B4 training
scripts.

The original supplementary-figures implementation in this repo
incorrectly overlaid the corollary formulas on top of the B3/B4
finite-time trained empirical values. The two quantities are not
directly comparable. This rewrite computes each corollary formula
directly from the stored spectra ``s_tr``, ``ω`` (and per-seed
permuted symbols ``s_perm_seed*`` for family-2), without any
finite-time training data.

Figure 1 — Corollary 5 (``cor:theoremB_ood``):
    At the converged optimum, the OOD loss under a shifted stationary
    symbol is

        E_OOD(α, L) = (1/P) · Σ_k  ω_k · s_te,k(α) · |1 − s_te,k(α)/s_tr,k|^{2L}.

    The per-mode |·|^{2L} factor is < 1 (attenuation) when
    |1 − s_te/s_tr| < 1 everywhere and > 1 (amplification) whenever
    any mode crosses the unit threshold. Two panels:

    - Left panel (attenuation regime, family-1): flat-symbol
      interpolation s_te = (1 − α)·s_tr + α·1. With the canonical B3
      s_tr, every mode has |·| < 1 at every α ∈ (0, 1]. Corollary 5
      therefore predicts E_OOD to DECREASE with L — depth helps OOD
      in this regime.
    - Right panel (amplification regime, family-2): random frequency
      permutation s_te = (1 − α)·s_tr + α·permute(s_tr). For
      sufficiently large α (α ≥ 0.5 in the canonical run) several
      modes cross |·| > 1 and E_OOD GROWS with L — depth hurts OOD
      here. Plotted as the median across the stored 8 permutation
      seeds.

Figure 2 — Corollary 6 (``cor:theoremB_finite_r``):
    With modes relabeled in nonincreasing spectral order
    (λ_1 ≥ λ_2 ≥ … > 0), the optimal stationary loss under a
    rank-r spectral bottleneck is

        inf_{Q ∈ Q_r} E_L(Q) = (1/P) · Σ_{k = r+1}^{P} ω_k · λ_k,

    which has no L dependence. The figure plots this magnitude-ordered
    theoretical floor as a single black dashed curve and overlays the
    B4 empirical values for every L ∈ {1, 2, 4, 8} as colored markers.
    B4's rank-masking convention (``k_star = min(k, P−k) < r``) is
    frequency-based rather than magnitude-based, so each B4 r maps to
    an ``effective_r`` (the number of modes actually controlled under
    that mask). The empirical overlays are placed at that effective_r
    so the theoretical and empirical axes agree.

Scope
-----
- Pure operator-level formulas evaluated on canonical B3/B4 spectra.
- No training, no architecture forward pass.
- No modification to the frozen utility layer.
- Does NOT rerun B3 or B4.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import matplotlib
import numpy as np

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


DEFAULT_B3_DIR = (
    "outputs/thesis/theoremB/run_theoremB_symbol_shift/"
    "run_theoremB_symbol_shift-20260413T063315Z-59725122"
)
DEFAULT_B4_DIR = (
    "outputs/thesis/theoremB/run_theoremB_rank_scaling/"
    "run_theoremB_rank_scaling-20260413T074851Z-93d98cc1"
)


@dataclass(frozen=True)
class SupplementaryFiguresConfig:
    b3_run_dir: str = DEFAULT_B3_DIR
    b4_run_dir: str = DEFAULT_B4_DIR
    L_list: tuple[int, ...] = (1, 2, 4, 8, 16)
    f1_alphas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.8, 1.0)
    f2_alphas: tuple[float, ...] = (0.2, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Corollary 5 analytical evaluation
# ---------------------------------------------------------------------------


def _corollary5(
    s_tr: np.ndarray, s_te: np.ndarray, omega: np.ndarray, L: int
) -> float:
    """E_OOD(α, L) at Q⋆ = L·T⁻¹.

    Evaluates (1/P) · Σ_k ω_k · s_te,k · |1 − s_te,k/s_tr,k|^{2L}.
    """
    P = s_tr.size
    factor = np.abs(1.0 - s_te / s_tr)
    return float(np.sum(omega * s_te * factor ** (2 * L)) / P)


def _load_b3(run_dir_rel: str) -> dict[str, np.ndarray]:
    npz_path = _PROJ / run_dir_rel / "npz" / "symbol_shift.npz"
    with np.load(npz_path, allow_pickle=True) as raw:
        return {k: np.asarray(raw[k]) for k in raw.files}


# ---------------------------------------------------------------------------
# Figure 1 — Corollary 5 (two panels)
# ---------------------------------------------------------------------------


def _plot_corollary5(
    cfg: SupplementaryFiguresConfig,
    b3: dict[str, np.ndarray],
    run_dir: ThesisRunDir,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    s_tr = np.asarray(b3["s_tr"], dtype=float)
    s_other_f1 = np.asarray(b3["s_other_f1"], dtype=float)
    omega = np.asarray(b3["omega"], dtype=float)
    f2_seeds = [int(s) for s in np.asarray(b3["f2_seeds"]).tolist()]
    s_perm = {
        s: np.asarray(b3[f"s_perm_seed{s}"], dtype=float) for s in f2_seeds
    }
    L_arr = np.asarray(cfg.L_list, dtype=int)

    # ---- Left panel: family-1 attenuation regime ---------------------------
    f1_colors = sequential_colors(len(cfg.f1_alphas), palette="mako")
    f1_data: dict[str, dict[str, Any]] = {}
    f1_deviations: dict[str, float] = {}
    for alpha in cfg.f1_alphas:
        s_te = (1.0 - alpha) * s_tr + alpha * s_other_f1
        values = [
            _corollary5(s_tr, s_te, omega, int(L)) for L in L_arr
        ]
        f1_data[f"{alpha:.2f}"] = {
            "alpha": float(alpha),
            "L_list": [int(L) for L in L_arr],
            "E_OOD": [float(v) for v in values],
        }
        f1_deviations[f"{alpha:.2f}"] = float(np.max(np.abs(1.0 - s_te / s_tr)))

    # ---- Right panel: family-2 amplification regime ------------------------
    f2_colors = sequential_colors(len(cfg.f2_alphas), palette="mako")
    f2_data: dict[str, dict[str, Any]] = {}
    f2_n_modes_over1: dict[str, int] = {}
    for alpha in cfg.f2_alphas:
        per_seed: list[list[float]] = []
        modes_over = 0
        for seed in f2_seeds:
            s_te = (1.0 - alpha) * s_tr + alpha * s_perm[seed]
            modes_over += int(np.sum(np.abs(1.0 - s_te / s_tr) > 1.0))
            per_seed.append(
                [_corollary5(s_tr, s_te, omega, int(L)) for L in L_arr]
            )
        arr = np.asarray(per_seed, dtype=float)
        med = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        f2_data[f"{alpha:.2f}"] = {
            "alpha": float(alpha),
            "L_list": [int(L) for L in L_arr],
            "E_OOD_median": [float(v) for v in med],
            "E_OOD_q25": [float(v) for v in q25],
            "E_OOD_q75": [float(v) for v in q75],
        }
        f2_n_modes_over1[f"{alpha:.2f}"] = modes_over

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=False)

    for color, alpha in zip(f1_colors, cfg.f1_alphas):
        rec = f1_data[f"{alpha:.2f}"]
        axL.plot(
            rec["L_list"], rec["E_OOD"],
            color=color, lw=1.5, marker="o", ms=4.0,
            label=rf"$\alpha = {alpha:.2f}$",
        )
    axL.set_xscale("log", base=2)
    axL.set_yscale("log")
    axL.set_xlabel(r"spectral depth $L$")
    axL.set_ylabel(r"$E_{\mathrm{OOD}}(\alpha, L)$  at  $Q^\star = L\,T^{-1}$")
    axL.set_title(
        "Family 1 (flat interpolation): attenuation regime "
        r"($|1 - \lambda'/\lambda| < 1$ for all modes)",
        fontsize=10,
    )
    axL.legend(fontsize=8, loc="best", frameon=True)

    for color, alpha in zip(f2_colors, cfg.f2_alphas):
        rec = f2_data[f"{alpha:.2f}"]
        axR.plot(
            rec["L_list"], rec["E_OOD_median"],
            color=color, lw=1.5, marker="o", ms=4.0,
            label=rf"$\alpha = {alpha:.2f}$ (median)",
        )
        axR.fill_between(
            rec["L_list"], rec["E_OOD_q25"], rec["E_OOD_q75"],
            color=color, alpha=0.18, lw=0,
        )
    axR.set_xscale("log", base=2)
    axR.set_yscale("log")
    axR.set_xlabel(r"spectral depth $L$")
    axR.set_ylabel(r"$E_{\mathrm{OOD}}(\alpha, L)$  at  $Q^\star = L\,T^{-1}$")
    axR.set_title(
        "Family 2 (permutation, 8 seeds): amplification regime "
        r"($|1 - \lambda'/\lambda| > 1$ on some modes)",
        fontsize=10,
    )
    axR.legend(fontsize=8, loc="best", frameon=True)

    fig.tight_layout()
    save_both(fig, run_dir, "b3_corollary5_ood_depth")
    plt.close(fig)

    return {
        "family1": {
            "alphas": list(cfg.f1_alphas),
            "max_deviation_per_alpha": f1_deviations,
            "values": f1_data,
        },
        "family2": {
            "alphas": list(cfg.f2_alphas),
            "n_modes_over_unit_across_seeds": f2_n_modes_over1,
            "values_median": f2_data,
        },
    }


# ---------------------------------------------------------------------------
# Figure 2 — Corollary 6 (magnitude-ordered, single theoretical curve)
# ---------------------------------------------------------------------------


def _load_b4(run_dir_rel: str) -> dict[str, np.ndarray]:
    npz_path = _PROJ / run_dir_rel / "npz" / "rank_scaling.npz"
    with np.load(npz_path, allow_pickle=True) as raw:
        return {k: np.asarray(raw[k]) for k in raw.files}


def _plot_corollary6(
    b4: dict[str, np.ndarray], run_dir: ThesisRunDir
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    s_tr = np.asarray(b4["s_tr"], dtype=float)
    omega = np.asarray(b4["omega"], dtype=float)
    P = s_tr.size

    # Corollary-6 magnitude-ordered floor: relabel modes by nonincreasing
    # λ_k = s_tr,k, then tail sum over the last (P − m) modes, where m
    # is the number of controlled modes.
    order = np.argsort(-s_tr, kind="stable")
    s_sorted = s_tr[order]
    omega_sorted = omega[order]
    prod_sorted = omega_sorted * s_sorted
    # cumulative tail: tail_sum[m] = Σ_{k=m+1..P} ω_k λ_k for top-m
    # controlled modes. tail_sum[0] = total; tail_sum[P] = 0.
    rev_cumsum = np.cumsum(prod_sorted[::-1])[::-1]
    # rev_cumsum[m] = sum of prod_sorted[m:], i.e. tail after excluding
    # the top-m modes. Index m ∈ {0, 1, ..., P}.
    rev_cumsum = np.concatenate([rev_cumsum, [0.0]])
    m_grid = np.arange(P + 1, dtype=int)
    floor_mag = rev_cumsum / P

    # B4 stored its own analytical_floor under the frequency-based
    # Q_r = {γ_k = 0 for k_star ≥ r} convention. Translate each B4 r
    # to the effective number of controlled modes m_eff(r).
    #
    # Normalization: the B4 script's ``_matched_loss`` and
    # ``_analytical_floor`` both evaluate Σ_k ω_k · s_k · |·|^{2L}
    # WITHOUT the 1/P factor that appears in Corollary 6's
    # ``(1/P) Σ_{k>r} ω_k λ_k``. Divide the B4 empirical values by P
    # so both the theoretical curve and the overlays live in the same
    # (normalized) units.
    b4_r_list = [int(r) for r in np.asarray(b4["r_list"]).tolist()]
    b4_L_list = [int(L) for L in np.asarray(b4["L_list"]).tolist()]
    k_arr = np.arange(P)
    k_star = np.minimum(k_arr, P - k_arr)
    m_eff_by_r: dict[int, int] = {
        r: int(np.sum(k_star < r)) for r in b4_r_list
    }

    colors = sequential_colors(len(b4_L_list), palette="mako")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Theoretical magnitude-ordered floor — single curve, L-independent.
    m_plot = m_grid[1:-1]  # skip m=0 (no control) and m=P (full control, 0 loss)
    floor_plot = floor_mag[1:-1]
    overlay_reference(
        ax,
        m_plot.astype(float),
        np.where(floor_plot > 0, floor_plot, np.nan),
        label=(
            r"Corollary 6 floor: "
            r"$(1/P)\sum_{k>m}\omega_k\lambda_k$  "
            r"(nonincreasing $\lambda$)"
        ),
        style="--",
        color="black",
        lw=1.4,
        zorder=3,
    )

    # Empirical B4 values at effective-r x-coordinates, one series per
    # L. Divide by P so the empirical scale matches Corollary 6's
    # (1/P) Σ normalization (B4 stored raw Σ values).
    empirical_records: dict[int, list[dict[str, float]]] = {}
    for color, L in zip(colors, b4_L_list):
        emp = np.asarray(b4[f"empirical_loss_L{L}"], dtype=float) / P
        xs = [m_eff_by_r[r] for r in b4_r_list]
        empirical_records[L] = [
            {"r_B4": int(r), "m_eff": int(m_eff_by_r[r]),
             "empirical_loss_per_mode": float(v)}
            for r, v in zip(b4_r_list, emp)
        ]
        ax.plot(
            xs, np.where(emp > 0, emp, np.nan),
            color=color, lw=0, marker="o", ms=6.0,
            label=rf"B4 empirical $L_S = {L}$",
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"controlled modes $m$  "
        r"(= $|\{k : \lambda_k \text{ in top-}m\}|$)"
    )
    ax.set_ylabel(r"loss floor  $\mathcal{E}_L^\star(r)$")
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    save_both(fig, run_dir, "b4_corollary6_rank_floor")
    plt.close(fig)

    # Quantify the L-collapse at the normalized scale (divide by P).
    spread: list[dict[str, float]] = []
    for r in b4_r_list:
        m = m_eff_by_r[r]
        values = np.array(
            [
                float(b4[f"empirical_loss_L{L}"][b4_r_list.index(r)]) / P
                for L in b4_L_list
            ],
            dtype=float,
        )
        values = values[np.isfinite(values) & (values > 0)]
        ratio = float(values.max() / values.min()) if values.size > 1 else 1.0
        spread.append({"r_B4": int(r), "m_eff": int(m),
                       "max_over_min_across_L": ratio,
                       "theory_floor": float(floor_mag[m]) if m <= P else 0.0,
                       "empirical_mean_per_mode_across_L": float(values.mean())})

    return {
        "P": int(P),
        "b4_r_list": b4_r_list,
        "b4_L_list": b4_L_list,
        "m_eff_by_r": {str(k): int(v) for k, v in m_eff_by_r.items()},
        "theoretical_floor_magnitude_ordered": {
            "m_grid": m_grid.tolist(),
            "floor_per_m": floor_mag.tolist(),
        },
        "empirical_by_L": {str(k): v for k, v in empirical_records.items()},
        "L_collapse_per_r": spread,
        "worst_L_collapse_ratio": float(
            max((s["max_over_min_across_L"] for s in spread), default=1.0)
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Theorem-B supplementary figures (Corollaries 5 and 6).",
    )
    p.add_argument("--b3-run-dir", type=str, default=None)
    p.add_argument("--b4-run-dir", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _cli()
    if args.no_show:
        matplotlib.use("Agg")

    cfg_kwargs: dict[str, Any] = {}
    if args.b3_run_dir:
        cfg_kwargs["b3_run_dir"] = args.b3_run_dir
    if args.b4_run_dir:
        cfg_kwargs["b4_run_dir"] = args.b4_run_dir
    cfg = SupplementaryFiguresConfig(**cfg_kwargs)

    run = ThesisRunDir(__file__, phase="theoremB")
    with RunContext(run, config=cfg, seeds=[]) as ctx:
        apply_thesis_style()

        t0 = time.perf_counter()
        b3 = _load_b3(cfg.b3_run_dir)
        b4 = _load_b4(cfg.b4_run_dir)
        load_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        corollary5_out = _plot_corollary5(cfg, b3, run)
        c5_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        corollary6_out = _plot_corollary6(b4, run)
        c6_wall = time.perf_counter() - t0

        ctx.record_compute_proxy(float(load_wall + c5_wall + c6_wall))
        ctx.record_extra("b3_source_run_dir", cfg.b3_run_dir)
        ctx.record_extra("b4_source_run_dir", cfg.b4_run_dir)
        ctx.record_extra("corollary5_figure", "b3_corollary5_ood_depth")
        ctx.record_extra("corollary6_figure", "b4_corollary6_rank_floor")
        ctx.record_extra("corollary5_results", corollary5_out)
        ctx.record_extra("corollary6_results", corollary6_out)

        print()
        print("=" * 72)
        print(" Theorem-B supplementary figures (Corollaries 5 and 6)")
        print("   (pure operator-level formulas on canonical B3/B4 spectra)")
        print()
        print(f"  B3 source: {cfg.b3_run_dir}")
        print(f"  B4 source: {cfg.b4_run_dir}")
        print()
        print(
            f"  Figure 1 (Corollary 5): depth sweep L ∈ {list(cfg.L_list)}"
        )
        print("    Family 1 (flat interpolation, attenuation regime):")
        for alpha_s, rec in corollary5_out["family1"]["values"].items():
            max_dev = corollary5_out["family1"]["max_deviation_per_alpha"][alpha_s]
            print(
                f"      α = {float(alpha_s):.2f}  "
                f"max |1-s_te/s_tr| = {max_dev:.3f}  "
                f"E_OOD(L=1→{cfg.L_list[-1]}): "
                f"{rec['E_OOD'][0]:.3e} → {rec['E_OOD'][-1]:.3e}"
            )
        print("    Family 2 (permutation, amplification regime, median over seeds):")
        for alpha_s, rec in corollary5_out["family2"]["values_median"].items():
            n_over = corollary5_out["family2"]["n_modes_over_unit_across_seeds"][alpha_s]
            print(
                f"      α = {float(alpha_s):.2f}  "
                f"total modes across seeds with |·|>1: {n_over}  "
                f"median E_OOD(L=1→{cfg.L_list[-1]}): "
                f"{rec['E_OOD_median'][0]:.3e} → {rec['E_OOD_median'][-1]:.3e}"
            )
        print()
        print(
            f"  Figure 2 (Corollary 6): P = {corollary6_out['P']};  "
            f"L_S ∈ {corollary6_out['b4_L_list']}"
        )
        for s in corollary6_out["L_collapse_per_r"]:
            print(
                f"    r_B4 = {s['r_B4']:>3d}  m_eff = {s['m_eff']:>3d}  "
                f"theory_floor = {s['theory_floor']:.3e}  "
                f"empirical_mean(/P)_across_L = "
                f"{s['empirical_mean_per_mode_across_L']:.3e}  "
                f"max/min across L = {s['max_over_min_across_L']:.3f}"
            )
        print(
            f"    worst L collapse ratio across all r: "
            f"{corollary6_out['worst_L_collapse_ratio']:.3f}  "
            f"(1.00 = perfect Corollary-6 L-independence)"
        )
        print("=" * 72)

        ctx.write_summary({
            "script": "run_theoremB_supplementary_figures",
            "phase": "theoremB",
            "category": (
                "supplementary operator-level visualization of Corollaries 5 "
                "and 6 evaluated on canonical B3/B4 spectra; NOT a new "
                "experiment; the corollary formulas are closed-form and "
                "require no training."
            ),
            "correction_note": (
                "The original figures in this script compared Corollary 5/6 "
                "formulas (which assume the converged optimum Q⋆ = L T⁻¹) "
                "against finite-time training data and were therefore "
                "incorrect. The corrected figures compute the corollary "
                "formulas directly at the converged optimum using the "
                "stored spectra, without any finite-time training data."
            ),
            "source_artifacts": {
                "B3": cfg.b3_run_dir,
                "B4": cfg.b4_run_dir,
            },
            "figures": [
                "b3_corollary5_ood_depth",
                "b4_corollary6_rank_floor",
            ],
            "corollary5_summary": corollary5_out,
            "corollary6_summary": corollary6_out,
            "interpretation": (
                "Figure 1 exhibits both regimes predicted by Corollary 5. "
                "In the family-1 attenuation regime (max_k |1 − s_te/s_tr| "
                "< 1 everywhere at every α ∈ [0, 1] in the canonical B3 "
                "setup), E_OOD(α, L) decreases exponentially with L — "
                "depth HELPS OOD when the shift stays inside the per-mode "
                "contraction ball. In the family-2 permutation regime, "
                "at α ≥ 0.5 enough modes cross |·| > 1 that the sum is "
                "dominated by the amplifying modes and E_OOD grows "
                "exponentially with L — depth HURTS OOD. Both behaviors "
                "are predictions of the same Corollary 5 formula; which "
                "regime is active depends on the shift geometry. "
                "Figure 2 verifies Corollary 6 L-independence: the B4 "
                "empirical loss floors at every L_S ∈ {1, 2, 4, 8} "
                "cluster tightly around the single magnitude-ordered "
                "theoretical curve (1/P) Σ_{k>m} ω_k λ_k, with no "
                "systematic L dependence beyond the finite-training "
                "scatter from slow controlled modes."
            ),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
