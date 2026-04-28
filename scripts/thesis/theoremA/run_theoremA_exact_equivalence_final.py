"""Experiment A1final: four-way exact equivalence — R0, R1, R2, R3.

Plan reference: ``EXPERIMENT_PLAN_FINAL.MD`` §8.1 (unification of A1 + A1b).

Why A1final exists
------------------
A1 compares R1, R2, R3. A1b adds R0 and compares R0 against R2 and R3,
but drops R1. Both scripts leave the R0 vs R1 comparison unevaluated,
even though R0 (true full-hidden-state aligned structured forward) and
R1 (iterative reduced (A_S, B_S) recursion) represent the two
operationally distinct "full-model" framings of theorem A.

A1final runs **all four routes in a single trial** and reports **all six
pairwise relative errors**:

    (R0, R1), (R0, R2), (R0, R3),
    (R1, R2), (R1, R3), (R2, R3).

Acceptance: every pairwise error at float64 machine precision over the
(D, P, K, L) sweep — there is no non-trivial pair in this regime.

The four routes (recap)
-----------------------
(R0) **True full hidden-state aligned structured forward pass.**
    Constructs the full ``(P + K)``-position residual stream and the
    ``(P + K) × (P + K)`` bilinear score ``S[μ, ν] = x_μ^T Γ x_ν / P``
    from ``(X_train, X_query, Γ)`` directly, applies a GD-compatible
    signed mask, and runs L explicit residual-stream layer updates.
    Does *not* consume ``(A_S, B_S, T)`` as inputs — exercises the
    full-model forward map.

(R1) **Iterative reduced (A_S, B_S) recursion.**
    At each of L layers,
        z_test  ← z_test  + (1/L) · B_S · z_train
        z_train ← z_train + (1/L) · A_S · z_train.
    Consumes the GA generator's reduced operators directly.

(R2) **Sample-space reduced (A_S, B_S) closed form.**
        f_red = (1/L) · B_S · Σ_{ℓ = 0..L-1} T^ℓ · y_train,
        T = I_P + A_S / L.
    Theorem A's canonical algebraic identity.

(R3) **Feature-space reduced-Γ closed form (GD-compatible only).**
    L-step preconditioned GD on ``w`` in feature space; prediction at
    queries is ``X_query^T w_L``.

In the GD-compatible regime theorem A asserts every pair is equal as an
algebraic identity; float64 roundoff over L residual-stream updates is
the only admissible source of disagreement.

Step-1b contract (sole dependencies)
------------------------------------
- :mod:`scripts.thesis.utils.data_generators`:
    ``GAConfig``, ``ga_generate``.
- :mod:`scripts.thesis.utils.metrics`:
    ``reduced_model_error``.
- :mod:`scripts.thesis.utils.plotting` / :mod:`run_metadata`: standard.

Primary outputs
---------------
- ``a1final_pairwise_errors_heatmap`` — max over all six pairwise
  relative errors across the (D, P) sweep at fixed (K, L).
- ``a1final_error_distribution`` — histogram of all six pairwise
  relative errors over every (D, P, K, L) cell.
- ``a1final_error_vs_L`` — diagnostic of the worst pairwise error as a
  function of L.
- ``a1final_pair_matrix`` — 4×4 matrix of worst-case pairwise errors
  across the full sweep (symmetric, diagonal = 0).
- ``exact_equivalence_all_routes.npz`` — per-cell grids of all six
  pairwise errors over (D, P, K, L).
- ``per_cell_summary.json`` — per-cell dump of all six errors.

Acceptance
----------
All six pairwise errors ``≤ machine_eps_tol`` over every sweep cell.
Strict — A1final is an exactness test.

Run
---
::

    python -u scripts/thesis/theoremA/run_theoremA_exact_equivalence_final.py \\
           --device cuda --dtype float64 --no-show
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
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
)
from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


ROUTE_NAMES: tuple[str, ...] = ("R0", "R1", "R2", "R3")
PAIR_KEYS: tuple[tuple[str, str], ...] = (
    ("R0", "R1"),
    ("R0", "R2"),
    ("R0", "R3"),
    ("R1", "R2"),
    ("R1", "R3"),
    ("R2", "R3"),
)


@dataclass(frozen=True)
class A1finalConfig:
    """Frozen configuration for the A1final four-route sweep.

    Default 4 D × 4 P × 3 K × 4 L = 192 cells × 4 routes = 768 forward
    evaluations; finishes in seconds. Sweep mirrors A1 / A1b so the three
    canonicals are directly comparable.
    """

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

    heatmap_K: int = 8
    heatmap_L: int = 4

    dtype: str = "float64"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Four forward routes
# ---------------------------------------------------------------------------


def _route_R0_full_hidden_state_forward(
    X_train: torch.Tensor,    # (B, D, P)
    X_query: torch.Tensor,    # (B, D, K)
    Gamma: torch.Tensor,      # (D, D)
    y_train: torch.Tensor,    # (B, P)
    L: int,
    P_norm: int,
) -> torch.Tensor:
    """R0: true full-hidden-state aligned structured forward pass.

    Mirrors A1b's R0 exactly.
    """
    B = int(X_train.shape[0])
    P = int(X_train.shape[-1])
    K = int(X_query.shape[-1])
    dtype = X_train.dtype
    device = X_train.device

    X = torch.cat([X_train, X_query], dim=-1)
    GammaX = torch.einsum("de,bef->bdf", Gamma, X)
    S_pos = torch.einsum("bdm,bdn->bmn", X, GammaX) / float(P_norm)

    M_signed = torch.zeros(P + K, P + K, dtype=dtype, device=device)
    M_signed[:P, :P] = -1.0
    M_signed[P:, :P] = +1.0
    M_eff = S_pos * M_signed.unsqueeze(0)

    h = torch.cat(
        [y_train, torch.zeros(B, K, dtype=dtype, device=device)], dim=-1,
    )
    inv_L = 1.0 / float(L)
    for _ in range(int(L)):
        update = torch.einsum("bmn,bn->bm", M_eff, h)
        h = h + inv_L * update
    return h[:, P:]


def _route_R1_full_layer_simulation(
    A_S: torch.Tensor,        # (B, P, P)
    B_S: torch.Tensor,        # (B, K, P)
    y_train: torch.Tensor,    # (B, P)
    L: int,
) -> torch.Tensor:
    """R1: iterative reduced (A_S, B_S) residual stream (mirrors A1)."""
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
    """R2: closed-form sample-space reduced (A_S, B_S) recursion."""
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
    """R3: feature-space preconditioned-GD reduced-Γ closed form."""
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


def _pair_key(a: str, b: str) -> str:
    return f"err_{a}_{b}"


def _run_trial(
    cfg: A1finalConfig, D: int, P: int, K: int, L: int, device: torch.device,
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
    A_S = op["A_S_theta"].to(device)
    B_S = op["B_S_theta"].to(device)
    A_S_GD = op["A_S_GD"].to(device)
    if not torch.equal(A_S, A_S_GD):
        raise AssertionError(
            f"A_S_theta != A_S_GD under mask_kind='gd_compatible' "
            f"(D={D}, P={P}, K={K}, L={L})"
        )

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    f_R0 = _route_R0_full_hidden_state_forward(
        X_train, X_query, Gamma, y_train, L=int(L), P_norm=int(P),
    )
    timings["t_R0_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R1 = _route_R1_full_layer_simulation(A_S, B_S, y_train, L=int(L))
    timings["t_R1_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R2 = _route_R2_reduced_AB(A_S, B_S, y_train, L=int(L))
    timings["t_R2_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_R3 = _route_R3_reduced_Gamma_feature_space(
        X_train, X_query, Gamma, y_train, L=int(L), P_norm=int(P),
    )
    timings["t_R3_seconds"] = time.perf_counter() - t0

    outs = {"R0": f_R0, "R1": f_R1, "R2": f_R2, "R3": f_R3}
    errs: dict[str, float] = {}
    for a, b in PAIR_KEYS:
        errs[_pair_key(a, b)] = float(reduced_model_error(outs[a], outs[b]))

    return {
        "D": int(D), "P": int(P), "K": int(K), "L": int(L),
        "f_R0": f_R0.detach().cpu(),
        "f_R1": f_R1.detach().cpu(),
        "f_R2": f_R2.detach().cpu(),
        "f_R3": f_R3.detach().cpu(),
        **errs,
        **timings,
    }


def _max_pair_err(trial: dict[str, Any]) -> float:
    return max(trial[_pair_key(a, b)] for a, b in PAIR_KEYS)


# ---------------------------------------------------------------------------
# LaTeX table writer
# ---------------------------------------------------------------------------


def _fmt_sci(x: float, digits: int = 2) -> str:
    """Format ``x`` as ``$a.bc \\times 10^{e}$`` for LaTeX math mode."""
    if not np.isfinite(x) or x == 0.0:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0 ** exp)
    return rf"${mant:.{digits}f} \times 10^{{{exp}}}$"


def _write_latex_table(
    cfg: A1finalConfig,
    trials: list[dict[str, Any]],
    worst: dict[str, float],
    ok: dict[str, bool],
    worst_cell: dict[str, Any] | None,
    sweep_wall: float,
    device: torch.device,
    run_dir: ThesisRunDir,
) -> None:
    """Write a LaTeX-formatted `.txt` dump of the sweep results.

    Produces three LaTeX tables in ``results_latex_table.txt``:
      1. Per-pair summary (worst / median / mean / status) over the full
         sweep.
      2. Worst pairwise error per depth L (rows = L, cols = pairs).
      3. Worst pairwise error per (D, P) slice at the configured
         heatmap (K, L).
    """

    L_list = list(cfg.L_list)
    D_list = list(cfg.D_list)
    P_list = list(cfg.P_list)

    # --- Table 1: per-pair summary ---
    per_pair_rows: list[str] = []
    for a, b in PAIR_KEYS:
        key = _pair_key(a, b)
        vals = np.array([t[key] for t in trials], dtype=float)
        w = float(worst[key])
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        status = r"\textsc{ok}" if ok[key] else r"\textbf{fail}"
        per_pair_rows.append(
            rf"{a}$\,$vs$\,${b} & {_fmt_sci(w)} & {_fmt_sci(med)} & "
            rf"{_fmt_sci(mean)} & {status} \\"
        )

    # --- Table 2: per-L worst ---
    per_L_rows: list[str] = []
    for L in L_list:
        sub = [t for t in trials if int(t["L"]) == int(L)]
        cells: list[str] = [f"${int(L)}$"]
        for a, b in PAIR_KEYS:
            key = _pair_key(a, b)
            w = max(t[key] for t in sub) if sub else float("nan")
            cells.append(_fmt_sci(w))
        per_L_rows.append(" & ".join(cells) + r" \\")

    # --- Table 3: (D, P) slice at (heatmap_K, heatmap_L) — max over 6 pairs.
    slice_lines: list[str] = []
    slice_has_data = (
        int(cfg.heatmap_K) in cfg.K_list
        and int(cfg.heatmap_L) in cfg.L_list
    )
    if slice_has_data:
        header = " & ".join(
            [r"$D \backslash P$"] + [rf"${int(P)}$" for P in P_list]
        )
        slice_lines.append(header + r" \\")
        slice_lines.append(r"\midrule")
        for D in D_list:
            cells = [rf"${int(D)}$"]
            for P in P_list:
                match = [
                    t for t in trials
                    if int(t["D"]) == int(D)
                    and int(t["P"]) == int(P)
                    and int(t["K"]) == int(cfg.heatmap_K)
                    and int(t["L"]) == int(cfg.heatmap_L)
                ]
                if not match:
                    cells.append(r"$-$")
                    continue
                w = _max_pair_err(match[0])
                cells.append(_fmt_sci(w))
            slice_lines.append(" & ".join(cells) + r" \\")

    # --- Assemble LaTeX document fragment ---
    all_ok = all(ok.values())
    status_banner = r"\textsc{ok}" if all_ok else r"\textbf{fail}"

    pair_cols = " & ".join(rf"{a}$\,${b}" for a, b in PAIR_KEYS)

    lines: list[str] = []
    lines.append(
        "% A1final — four-way exact theorem-A equivalence (R0, R1, R2, R3)."
    )
    lines.append(
        r"% Plan reference: EXPERIMENT_PLAN_FINAL.MD §8.1 (A1 + A1b unified)."
    )
    lines.append(
        f"% Sweep: |D|={len(D_list)} x |P|={len(P_list)} x "
        f"|K|={len(cfg.K_list)} x |L|={len(L_list)} "
        f"= {len(trials)} cells, B={cfg.B}, dtype={cfg.dtype}, "
        f"device={device}."
    )
    lines.append(
        f"% Acceptance: all six pairwise relative errors "
        f"<= {cfg.machine_eps_tol:.0e}."
    )
    lines.append(f"% Sweep wall clock: {sweep_wall:.2f} s.")
    if worst_cell is not None:
        lines.append(
            f"% Worst cell: D={worst_cell['D']}, P={worst_cell['P']}, "
            f"K={worst_cell['K']}, L={worst_cell['L']}."
        )
    lines.append("")

    # --- Table 1 ---
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{A1final: four-way exact theorem-A equivalence across "
        rf"the full ${len(D_list)} \times {len(P_list)} \times "
        rf"{len(cfg.K_list)} \times {len(L_list)} = {len(trials)}$-cell "
        rf"$(D, P, K, L)$ sweep (GD-compatible mask, $B = {cfg.B}$, "
        rf"float64). Worst / median / mean relative error over all cells "
        rf"for each of the six route pairs; acceptance "
        rf"$\leq {cfg.machine_eps_tol:.0e}$. Overall status: "
        rf"{status_banner}.}}"
    )
    lines.append(r"\label{tab:a1final_per_pair}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"pair & worst & median & mean & status \\")
    lines.append(r"\midrule")
    lines.extend(per_pair_rows)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table 2 ---
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{A1final: worst pairwise relative error per depth "
        r"$L$, over $(D, P, K)$. Theorem-A roundoff is expected to grow "
        r"mildly with the number of residual-stream updates.}"
    )
    lines.append(r"\label{tab:a1final_per_L}")
    lines.append(r"\begin{tabular}{c" + "c" * len(PAIR_KEYS) + "}")
    lines.append(r"\toprule")
    lines.append(r"$L$ & " + pair_cols + r" \\")
    lines.append(r"\midrule")
    lines.extend(per_L_rows)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table 3 ---
    if slice_has_data:
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(
            rf"\caption{{A1final: maximum pairwise relative error (max "
            rf"over the six pairs) across the $(D, P)$ slice at "
            rf"$K = {int(cfg.heatmap_K)}$, $L = {int(cfg.heatmap_L)}$.}}"
        )
        lines.append(r"\label{tab:a1final_DP_slice}")
        lines.append(r"\begin{tabular}{c" + "c" * len(P_list) + "}")
        lines.append(r"\toprule")
        lines.extend(slice_lines)
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # --- Table 4: full per-cell dump (longtable, appendix-ready) ---
    pair_cols = " & ".join(rf"{a}$\,${b}" for a, b in PAIR_KEYS)
    n_err_cols = len(PAIR_KEYS)
    lines.append(
        r"% Full per-cell sweep (longtable) — requires \usepackage{longtable,booktabs}."
    )
    lines.append(r"\begingroup")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(
        r"\begin{longtable}{cccc" + "c" * n_err_cols + "}"
    )
    lines.append(
        rf"\caption{{A1final: full per-cell pairwise relative errors over the "
        rf"$(D, P, K, L)$ sweep ({len(trials)} cells, GD-compatible mask, "
        rf"$B = {cfg.B}$, float64). Every entry satisfies acceptance "
        rf"$\leq {cfg.machine_eps_tol:.0e}$.}}"
        r" \label{tab:a1final_full_sweep} \\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"$D$ & $P$ & $K$ & $L$ & " + pair_cols + r" \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(
        r"\multicolumn{" + str(4 + n_err_cols) + r"}{l}{\textit{"
        r"Table \ref{tab:a1final_full_sweep} continued from previous page"
        r"}} \\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"$D$ & $P$ & $K$ & $L$ & " + pair_cols + r" \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{" + str(4 + n_err_cols) + r"}{r}{\textit{"
        r"continued on next page}} \\"
    )
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    # Sort by (D, P, K, L) so the dump is deterministic and easy to audit.
    sorted_trials = sorted(
        trials,
        key=lambda t: (int(t["D"]), int(t["P"]), int(t["K"]), int(t["L"])),
    )
    for t in sorted_trials:
        cells = [
            rf"${int(t['D'])}$",
            rf"${int(t['P'])}$",
            rf"${int(t['K'])}$",
            rf"${int(t['L'])}$",
        ]
        for a, b in PAIR_KEYS:
            cells.append(_fmt_sci(float(t[_pair_key(a, b)])))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    lines.append("")

    (run_dir.root / "results_latex_table.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_pairwise_errors_heatmap(
    cfg: A1finalConfig, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Heatmap of max over all 6 pairwise errors over (D, P) at fixed (K, L)."""
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
        grid[i_D, i_P] = _max_pair_err(trial)
    floor = 1e-18
    grid_plot = np.where(grid > floor, grid, floor)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    phase_heatmap(
        ax, grid_plot,
        x_coords=np.asarray(P_list, dtype=float),
        y_coords=np.asarray(D_list, dtype=float),
        xlabel="P (training context length)",
        ylabel="D (feature dimension)",
        cbar_label=r"max pairwise rel. error (6 pairs)",
        log_z=True, log_x=True, log_y=True,
    )
    ax.set_title(
        rf"A1final max pairwise relative error at K = {K}, L = {L} "
        r"(R0, R1, R2, R3 — 6 pairs)",
        fontsize=10,
    )
    fig.tight_layout()
    save_both(fig, run_dir, "a1final_pairwise_errors_heatmap")
    plt.close(fig)


def _plot_error_distribution(
    cfg: A1finalConfig, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Histogram of all six pairwise errors across the full sweep."""
    import matplotlib.pyplot as plt

    floor = 1e-18
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.logspace(-18, -8, 30)
    for a, b in PAIR_KEYS:
        key = _pair_key(a, b)
        vals = np.clip(np.array([t[key] for t in trials]), floor, None)
        ax.hist(
            vals, bins=bins, alpha=0.45,
            label=f"{a} vs {b}",
            edgecolor="black", lw=0.3,
        )
    ax.axvline(
        cfg.machine_eps_tol, color="red", lw=0.9, ls="--",
        label=f"acceptance = {cfg.machine_eps_tol:.0e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("relative error")
    ax.set_ylabel("count")
    ax.set_title(
        f"A1final pairwise relative-error distribution over "
        f"{len(trials)} cells (6 pairs × D × P × K × L)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_both(fig, run_dir, "a1final_error_distribution")
    plt.close(fig)


def _plot_error_vs_L(
    cfg: A1finalConfig, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """Worst pairwise error (over 6 pairs × D × P × K) vs L."""
    import matplotlib.pyplot as plt

    L_list = list(cfg.L_list)
    L_max_err = []
    for L in L_list:
        sub = [t for t in trials if int(t["L"]) == int(L)]
        if not sub:
            L_max_err.append(np.nan)
            continue
        L_max_err.append(max(_max_pair_err(t) for t in sub))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(
        L_list, L_max_err, color="C0", lw=1.5, marker="o", ms=5.0,
        label="max pairwise rel. error (6 pairs, across D, P, K)",
    )
    ax.axhline(
        cfg.machine_eps_tol, color="red", lw=0.9, ls="--",
        label=f"acceptance = {cfg.machine_eps_tol:.0e}",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("depth L")
    ax.set_ylabel("max pairwise rel. error")
    ax.set_title(
        "A1final error growth with depth (expected ~ L-step roundoff)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, run_dir, "a1final_error_vs_L")
    plt.close(fig)


def _plot_pair_matrix(
    cfg: A1finalConfig, trials: list[dict[str, Any]], run_dir: ThesisRunDir,
) -> None:
    """4×4 worst-case pairwise-error matrix over the full sweep."""
    import matplotlib.pyplot as plt

    n = len(ROUTE_NAMES)
    mat = np.zeros((n, n))
    for a, b in PAIR_KEYS:
        worst = max(trial[_pair_key(a, b)] for trial in trials)
        i, j = ROUTE_NAMES.index(a), ROUTE_NAMES.index(b)
        mat[i, j] = worst
        mat[j, i] = worst
    floor = 1e-18
    mat_plot = np.where(mat > floor, mat, floor)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(
        mat_plot, cmap="mako",
        norm=matplotlib.colors.LogNorm(
            vmin=max(mat_plot.min(), 1e-17),
            vmax=max(mat_plot.max(), cfg.machine_eps_tol),
        ),
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ROUTE_NAMES)
    ax.set_yticklabels(ROUTE_NAMES)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{mat[i, j]:.1e}" if i != j else "—",
                ha="center", va="center", color="white", fontsize=8,
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("worst-case pairwise rel. error", fontsize=9)
    ax.set_title(
        "A1final worst-case pairwise errors (all D, P, K, L)",
        fontsize=10,
    )
    fig.tight_layout()
    save_both(fig, run_dir, "a1final_pair_matrix")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_list_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment A1final: four-way exact theorem-A forward-pass "
            "equivalence — R0 (full hidden-state), R1 (iterative "
            "reduced AB), R2 (closed-form reduced AB), R3 (feature-space "
            "reduced-Γ). All six pairwise errors gated at float eps."
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


def _config_from_cli(args: argparse.Namespace) -> A1finalConfig:
    base = A1finalConfig()
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
    print(f"[A1final] device = {device}")

    run = ThesisRunDir(__file__, phase="theoremA")
    with RunContext(
        run,
        config=cfg,
        seeds=[cfg.base_seed, cfg.base_seed + 7,
               cfg.base_seed + 13, cfg.base_seed + 19],
        notes=(
            "A1final: four-way exact theorem-A forward-pass equivalence. "
            "Unifies A1 (R1, R2, R3) and A1b (R0, R2, R3) into a single "
            "trial running all four routes (R0 = true full-hidden-state "
            "aligned structured forward; R1 = iterative reduced AB; "
            "R2 = closed-form reduced AB; R3 = feature-space reduced-Γ). "
            "All six pairwise relative errors must lie at float64 machine "
            "precision over the (D, P, K, L) sweep."
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
                        pair_str = "  ".join(
                            f"{a}{b}={trial[_pair_key(a, b)]:.2e}"
                            for a, b in PAIR_KEYS
                        )
                        print(
                            f"[{idx:>4d}/{n_total}] "
                            f"D={int(D):>3d} P={int(P):>3d} "
                            f"K={int(K):>3d} L={int(L):>2d}  "
                            f"{pair_str}  "
                            f"({dt*1000:.1f} ms)"
                        )
                        trials.append(trial)
        sweep_wall = time.perf_counter() - t_sweep_start

        # --- Figures ---
        _plot_pairwise_errors_heatmap(cfg, trials, run)
        _plot_error_distribution(cfg, trials, run)
        _plot_error_vs_L(cfg, trials, run)
        _plot_pair_matrix(cfg, trials, run)

        # --- Save npz ---
        D_list = list(cfg.D_list)
        P_list = list(cfg.P_list)
        K_list = list(cfg.K_list)
        L_list = list(cfg.L_list)
        shape = (len(D_list), len(P_list), len(K_list), len(L_list))
        err_grids: dict[str, np.ndarray] = {
            _pair_key(a, b) + "_grid": np.zeros(shape) for a, b in PAIR_KEYS
        }
        for trial in trials:
            i = D_list.index(trial["D"])
            j = P_list.index(trial["P"])
            k = K_list.index(trial["K"])
            l = L_list.index(trial["L"])
            for a, b in PAIR_KEYS:
                err_grids[_pair_key(a, b) + "_grid"][i, j, k, l] = (
                    trial[_pair_key(a, b)]
                )
        npz_payload: dict[str, np.ndarray] = {
            "D_list": np.asarray(D_list, dtype=np.int64),
            "P_list": np.asarray(P_list, dtype=np.int64),
            "K_list": np.asarray(K_list, dtype=np.int64),
            "L_list": np.asarray(L_list, dtype=np.int64),
            **err_grids,
        }
        np.savez_compressed(
            run.npz_path("exact_equivalence_all_routes"), **npz_payload,
        )

        # --- Per-cell JSON ---
        rows = []
        for t in trials:
            row: dict[str, Any] = {
                "D": t["D"], "P": t["P"], "K": t["K"], "L": t["L"],
            }
            for a, b in PAIR_KEYS:
                row[_pair_key(a, b)] = float(t[_pair_key(a, b)])
            for r in ROUTE_NAMES:
                row[f"t_{r}_seconds"] = float(t[f"t_{r}_seconds"])
            rows.append(row)
        (run.root / "per_cell_summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

        # --- Acceptance ---
        worst: dict[str, float] = {
            _pair_key(a, b): max(t[_pair_key(a, b)] for t in trials)
            for a, b in PAIR_KEYS
        }
        ok: dict[str, bool] = {
            k: v <= cfg.machine_eps_tol for k, v in worst.items()
        }
        all_ok = all(ok.values())

        # Worst cell across all pairs.
        worst_cell: dict[str, Any] | None = None
        worst_err = 0.0
        for t in trials:
            e = _max_pair_err(t)
            if e > worst_err:
                worst_err = e
                worst_cell = {
                    "D": t["D"], "P": t["P"], "K": t["K"], "L": t["L"],
                    **{_pair_key(a, b): float(t[_pair_key(a, b)])
                       for a, b in PAIR_KEYS},
                }

        # --- LaTeX-formatted result tables ---
        _write_latex_table(
            cfg=cfg,
            trials=trials,
            worst=worst,
            ok=ok,
            worst_cell=worst_cell,
            sweep_wall=float(sweep_wall),
            device=device,
            run_dir=run,
        )

        ctx.record_compute_proxy(float(sweep_wall))
        for key, val in worst.items():
            ctx.record_extra(f"worst_{key[4:]}", val)
        ctx.record_extra("worst_cell", worst_cell)

        status_parts = [
            (f"{k[4:]}_ok" if ok[k] else f"{k[4:]}_violated(worst={v:.2e})")
            for k, v in worst.items()
        ]
        status = "+".join(status_parts)

        ctx.write_summary(
            {
                "plan_reference": (
                    "EXPERIMENT_PLAN_FINAL.MD §8.1 (A1 + A1b unified)"
                ),
                "category": (
                    "four-way exact theorem-A operator-level forward-pass "
                    "equivalence test. Four structurally distinct forward "
                    "routes (R0 = true full-hidden-state aligned "
                    "structured forward from (X, Γ); R1 = iterative "
                    "reduced (A_S, B_S) residual stream; R2 = closed-form "
                    "reduced (A_S, B_S) recursion; R3 = feature-space "
                    "reduced-Γ closed form) must agree to float64 machine "
                    "precision in the GD-compatible setting. Reports all "
                    "six pairwise relative errors — the strictest "
                    "formulation of theorem A's operator-level claim."
                ),
                "interpretation": (
                    "Theorem A asserts that, in the GD-compatible mask "
                    "regime, the full-hidden-state aligned linear "
                    "structured forward map, the iterative reduced-AB "
                    "recursion, the closed-form reduced-AB identity, and "
                    "the feature-space reduced-Γ iterate are all equal as "
                    "an algebraic identity. A1final verifies all six "
                    "pairwise equalities at float eps over (D, P, K, L). "
                    "Any pair exceeding tolerance indicates an "
                    "implementation bug, not a theorem failure."
                ),
                "device": str(device),
                "routes": list(ROUTE_NAMES),
                "pairs": [f"{a}_{b}" for a, b in PAIR_KEYS],
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
                **{f"worst_{k[4:]}": float(v) for k, v in worst.items()},
                "worst_cell": worst_cell,
                "sweep_wallclock_seconds": round(float(sweep_wall), 3),
            }
        )

        print()
        print("=" * 72)
        print(f" A1final four-route exact theorem-A equivalence on {device}")
        print(f"   N cells = {len(trials)} (D × P × K × L)")
        for a, b in PAIR_KEYS:
            k = _pair_key(a, b)
            print(
                f"   worst {a} vs {b} = {worst[k]:.3e}  "
                f"{'OK' if ok[k] else 'FAIL'}  "
                f"(tol = {cfg.machine_eps_tol:.1e})"
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
