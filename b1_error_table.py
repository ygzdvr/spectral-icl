#!/usr/bin/env python3
"""
b1_error_table.py
-----------------
Read the B1 exact-closure per_trial_summary.json and print a LaTeX booktabs
table containing the three acceptance metrics for every (P, L, symbol) trial.

Usage:
    python b1_error_table.py            # auto-detects most recent run dir
    python b1_error_table.py <run_dir>  # explicit path to the run directory
"""

import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the run directory
# ---------------------------------------------------------------------------
ARTIFACT_ROOT = (
    Path(__file__).resolve().parent
    / "outputs" / "thesis" / "theoremB" / "run_theoremB_circulant_modes"
)


def _find_run_dir() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.is_dir():
            sys.exit(f"ERROR: {p} is not a directory")
        return p
    dirs = sorted(ARTIFACT_ROOT.glob("run_theoremB_circulant_modes-*/"), reverse=True)
    if not dirs:
        sys.exit(f"ERROR: no run directories found under {ARTIFACT_ROOT}")
    return dirs[0]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sci(x: float, digits: int = 2) -> str:
    """LaTeX scientific notation: $1.07\\times10^{-14}$ or $0$ for exact zero."""
    if x == 0.0:
        return r"$0$"
    exp = math.floor(math.log10(abs(x)))
    mantissa = x / 10 ** exp
    return rf"${mantissa:.{digits}f}\times10^{{{exp}}}$"


SYMBOL_DISPLAY = {
    "flat":       "flat",
    "power_law":  "power-law",
    "multiband":  "multiband",
}
SYMBOL_ORDER = ["flat", "power_law", "multiband"]
TOL = 1e-10


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_dir = _find_run_dir()
    summary_path = run_dir / "per_trial_summary.json"
    if not summary_path.exists():
        sys.exit(f"ERROR: {summary_path} not found")

    trials: list[dict] = json.loads(summary_path.read_text())

    # Sort: symbol group first (flat → power-law → multiband), then P, then L
    trials.sort(key=lambda t: (SYMBOL_ORDER.index(t["symbol_kind"]), t["P"], t["L"]))

    # Check whether off-diagonal energy is uniformly zero (it is, by design)
    all_ode_zero = all(t["off_diagonal_fourier_energy"] == 0.0 for t in trials)

    # -----------------------------------------------------------------------
    # Build the table
    # -----------------------------------------------------------------------
    out = []

    out.append(r"\begin{table}[ht]")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{5pt}")
    out.append(r"\caption{%")
    out.append(r"  Exact circulant-closure errors for all 36 $(P, L, \text{symbol})$")
    out.append(r"  trials in experiment B1.")
    out.append(r"  $\varepsilon_{\gamma}$: max-magnitude-scaled relative error between")
    out.append(r"  the matrix-recursion mode trajectory and the per-mode theory")
    out.append(r"  recursion.")
    out.append(r"  $\varepsilon_{T}$: same metric on the transfer function $T(\omega)$.")
    out.append(r"  $E_{\perp}$: off-diagonal Fourier energy of the reduced operator")
    out.append(r"  (measures how far $\Gamma$ stays circulant).")
    if all_ode_zero:
        out.append(r"  $E_{\perp}=0$ exactly for every trial (bit-level zero).")
    out.append(r"  Acceptance threshold: $10^{-10}$;")
    out.append(r"  \checkmark\ denotes all three metrics below threshold.%")
    out.append(r"}")
    out.append(r"\label{tab:b1_error_summary}")
    out.append(r"\begin{tabular}{@{}cc l ccc c@{}}")
    out.append(r"\toprule")
    out.append(
        r"$P$ & $L$ & Symbol & "
        r"$\varepsilon_{\gamma}$ & $\varepsilon_{T}$ & $E_{\perp}$ & Pass \\"
    )
    out.append(r"\midrule")

    prev_symbol = None
    for t in trials:
        sym = t["symbol_kind"]
        if prev_symbol is not None and sym != prev_symbol:
            out.append(r"\midrule")
        prev_symbol = sym

        mode_err  = t["mode_rel_err_max"]
        trans_err = t["transfer_rel_err_max"]
        ode_err   = t["off_diagonal_fourier_energy"]
        passed    = mode_err <= TOL and trans_err <= TOL and ode_err <= TOL
        mark      = r"\checkmark" if passed else r"$\boldsymbol{\times}$"

        out.append(
            rf"  {t['P']} & {t['L']} & {SYMBOL_DISPLAY[sym]} &"
            rf" {_sci(mode_err)} & {_sci(trans_err)} & {_sci(ode_err)} & {mark} \\"
        )

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")

    print("\n".join(out))


if __name__ == "__main__":
    main()
