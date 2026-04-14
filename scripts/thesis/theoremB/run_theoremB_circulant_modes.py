"""Experiment B1: exact finite-P circulant mode closure.

B1 exact = finite-P discrete residual/query recursion in LAYER INDEX ell
=========================================================================
Plan reference: EXPERIMENT_PLAN_FINAL.MD §6.2.

Mathematical content
--------------------
Validates the exact finite-P proposition of Theorem B
(Proposition theoremB_finiteP_circulant).

Let
    G      := circulant training operator, eigenvalues lambda_k  = s_tr[k]
    G_star := circulant query operator,   eigenvalues lambda*_k = s_te[k]

At the operator-level validation tier G = Sigma_tr and G_star = Sigma_te
from the G1 generator.  Both are diagonalized by the *sample-space* unitary
DFT  F_P  (NOT the feature-space eigenbasis).

The exact L-layer residual recursion is:

    r^{ell+1} = (I - G/L) r^ell,    r^0 = y      (ell = 0, ..., L-1)

In the sample-space Fourier basis:

    rhat_k^ell = (1 - lambda_k / L)^ell * yhat_k

The full-window query prediction is:

    f = (1/L) G_star * sum_{ell=0}^{L-1} r^ell

Modewise:

    fhat_k = h_{L,k} * yhat_k

where the transfer function is:

    h_{L,k} = lambda*_k * phi_L(lambda_k)
    phi_L(z) = (1 - (1 - z/L)^L) / z    for z != 0
    phi_L(0) = 1

Matched special case (G_star = G):

    h_{L,k} = 1 - (1 - lambda_k / L)^L

THIS is the primary B1 theorem target. x-axis of all primary figures is
LAYER INDEX ell, NOT optimization step t.

Deterministic label
-------------------
y = e_0 (first standard basis vector in sample space), so that
yhat_k = 1/sqrt(P) for ALL k.  This makes the empirical transfer
h_emp[k] = fhat_k / yhat_k numerically stable for every k.

B2/stationary bridge (OFF by default)
--------------------------------------
The stationary gamma_k(t) gradient-flow dynamics belong to the B2/stationary
corollary and are NOT the primary B1 theorem object.  They are available only
under --bridge-to-b2.  Default figures do not include any optimization-time
quantities.

Terminology
-----------
  residual contraction factor:   c_{L,k} = (1 - lambda_k/L)^L   (NOT transfer)
  transfer function:             h_{L,k} = lambda*_k phi_L(lambda_k)

DO NOT conflate residual contraction with transfer.

Acceptance
----------
Across the full matched+mismatched sweep
(P in {16,32,64}, L in {1,2,4,8}, symbol in {flat,power_law,multiband}):

    residual_mode_rel_err    <= tol_err = 1e-10
    transfer_rel_err         <= tol_err
    train_offdiag_fourier    <= tol_err
    query_offdiag_fourier    <= tol_err

Run
---
::

    python -u scripts/thesis/theoremB/run_theoremB_circulant_modes.py \\
           --device cpu --dtype float64 --no-show

    # Include optional B2 bridge figures:
    python -u scripts/thesis/theoremB/run_theoremB_circulant_modes.py \\
           --device cpu --dtype float64 --no-show --bridge-to-b2
"""

from __future__ import annotations

import argparse
import json
import math
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
    circulant_from_symbol,
    off_diagonal_fourier_energy,
    unitary_dft,
)
from scripts.thesis.utils.metrics import (
    gamma_star_trajectory_circulant,  # bridge mode only
    mode_trajectory_error,            # diagnostic only
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
    """Frozen configuration for B1 exact finite-P circulant-modes experiment.

    Primary sweep: P_list × L_list × symbol_kinds × {matched, mismatched}.

    Each trial runs the L-layer residual recursion with y = e_0 (deterministic)
    and compares to the theorem predictions in the *sample-space* Fourier basis.

    B2/stationary bridge fields (T, eta) are stored here but only consumed
    when bridge_to_b2=True.  Default B1 figures do NOT depend on T or eta.
    """

    # ----- Primary sweep -------------------------------------------------------
    P_list: tuple[int, ...] = (16, 32, 64)
    L_list: tuple[int, ...] = (1, 2, 4, 8)
    symbol_kinds: tuple[str, ...] = ("flat", "power_law", "multiband")

    # ----- Symbol parameters ---------------------------------------------------
    power_law_nu: float = 0.5
    task_spec_nu_beta: float = 1.0
    multiband: tuple[tuple[int, int, float], ...] = (
        (0, 2, 1.0),
        (5, 7, 0.8),
    )

    # ----- Mismatched query symbol (Figure 3) ----------------------------------
    # "auto" chooses based on training kind: power_law → flat; else → power_law.
    mismatch_symbol_kind: str = "auto"

    # ----- Query regime --------------------------------------------------------
    query_mode: str = "full_window"

    # ----- Acceptance ----------------------------------------------------------
    tol_err: float = 1e-10

    # ----- Figure parameters ---------------------------------------------------
    figure_P: int = 32
    figure_symbol: str = "power_law"
    figure_mode_indices: tuple[int, ...] = (0, 1, 2, 4, 8)
    figure_L_list: tuple[int, ...] = (1, 2, 4, 8)
    transfer_P: int = 32
    transfer_L_list: tuple[int, ...] = (1, 2, 4, 8)
    transfer_symbol: str = "power_law"

    # ----- B2/stationary bridge (OFF by default) ------------------------------
    # B1 exact  = finite-P residual/query recursion in layer index ell.
    # B2 bridge = optimization-time gamma_k(t) stationary gradient-flow.
    bridge_to_b2: bool = False
    T: int = 100       # steps for stationary gradient flow (bridge only)
    eta: float = 1e-4  # learning rate for stationary gradient flow (bridge only)

    # ----- Runtime -------------------------------------------------------------
    dtype: str = "float64"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------


def _phi_L_vec(lam: torch.Tensor, L: int) -> torch.Tensor:
    """Vectorized phi_L(lambda_k) for the theorem-B transfer formula.

    phi_L(z) = (1 - (1 - z/L)^L) / z   for z != 0
    phi_L(0) = 1                         (limit)

    Numerically stable implementation via log1p / expm1.
    -------------------------------------------------------
    The naive formula ``(1 - (1-z/L)^L) / z`` loses O(log10(L/z)) significant
    digits when z << L via catastrophic cancellation in the numerator
    (1 - (1-z/L)^L ≈ z when z → 0, but computed as 1 - (nearly 1)).
    A first-order Taylor patch still has O(z²) residual that is amplified when
    lambda*_k is large (mismatched multiband+power_law trials).

    The correct fix uses the identities
        (1-z/L)^L = exp(L * log1p(-z/L))
        1 - exp(x) = -expm1(x)
    Both log1p and expm1 are accurate to 1 ULP even for tiny arguments, so
        phi_L(z) = -expm1(L * log1p(-z/L)) / z
    is accurate for all z in (0, L) without any Taylor branch.

    For z >= L (oscillating / contracting modes where |1-z/L| > 1): the naive
    direct formula is fine (no cancellation since (1-z/L)^L is not near 1).
    For z = 0: return 1 (limit).

    Input/output shape: (P,) real.
    """
    L_f = float(L)
    finfo = torch.finfo(lam.dtype)
    eps = finfo.eps * 10
    tiny = finfo.tiny

    # Stable branch for |z| < L: -expm1(L * log1p(-z/L)) / z
    # log1p(-z/L) = log(1 - z/L), accurate to 1 ULP for all z/L in (-1, 0].
    z_over_L = lam / L_f
    # Clamp argument of log1p to keep it in domain (-1, 0] for positive lam.
    # For our positive-semidefinite circulant eigenvalues this is automatic.
    log1p_arg = (-z_over_L).clamp(min=-1.0 + tiny)
    log1p_val = torch.log1p(log1p_arg)            # log(1 - z/L), accurate
    expm1_val = torch.expm1(L_f * log1p_val)      # (1-z/L)^L - 1, accurate
    num_stable = -expm1_val                        # 1 - (1-z/L)^L
    stable = num_stable / lam.clamp(min=tiny)      # phi_L(z)

    # Direct branch for |z| >= L (no cancellation).
    contraction = (1.0 - z_over_L).pow(L)
    direct = (1.0 - contraction) / lam.clamp(min=tiny)

    # Compose: stable for 0 < |z| < L; direct for |z| >= L; limit 1 at z=0.
    result = torch.where(lam.abs() < L_f, stable, direct)
    result = torch.where(lam.abs() < eps, torch.ones_like(lam), result)
    return result


def _deterministic_y(P: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return y = e_0 (first standard basis vector in sample space).

    unitary_dft(e_0)[k] = P^{-1/2} for ALL k, giving stable nonzero
    denominators when computing empirical transfer h_emp[k] = fhat_k / yhat_k.
    """
    y = torch.zeros(P, dtype=dtype, device=device)
    y[0] = 1.0
    return y


def _uft(x: torch.Tensor) -> torch.Tensor:
    """Unitary DFT of a real or complex tensor (last dimension).

    Wraps fourier_ops.unitary_dft; returns complex128.
    """
    return unitary_dft(x.to(torch.float64))


# ---------------------------------------------------------------------------
# G1 config builders
# ---------------------------------------------------------------------------


def _mismatch_symbol(symbol_kind: str, cfg: B1Config) -> tuple[str, dict[str, Any]]:
    """Return (kind, params) for the mismatched query symbol G_star != G."""
    if cfg.mismatch_symbol_kind != "auto":
        kind = cfg.mismatch_symbol_kind
    elif symbol_kind == "power_law":
        kind = "flat"
    else:
        kind = "power_law"
    if kind == "flat":
        params: dict[str, Any] = {"value": 1.0}
    elif kind == "power_law":
        # Use nu=1.5 if training nu < 1; nu=0.3 otherwise — genuinely distinct.
        params = {"nu": 1.5 if cfg.power_law_nu < 1.0 else 0.3}
    elif kind == "multiband":
        params = {"bands": list(cfg.multiband)}
    else:
        raise ValueError(f"unknown mismatch symbol kind: {kind!r}")
    return kind, params


def _build_g1_config(
    cfg: B1Config,
    P: int,
    symbol_kind: str,
    match_mode: str,  # "matched" or "mismatched"
) -> G1Config:
    """Build G1Config for one B1 trial.

    - exact_mode=True, sample_data=False (operator-level only).
    - query_mode='full_window' with K = P.
    - Matched: symbol_kind_te='matched' (G_star = G).
    - Mismatched: symbol_kind_te from _mismatch_symbol (G_star != G).
    """
    if symbol_kind == "power_law":
        tr_params: dict[str, Any] = {"nu": cfg.power_law_nu}
    elif symbol_kind == "multiband":
        tr_params = {"bands": list(cfg.multiband)}
    elif symbol_kind == "flat":
        tr_params = {"value": 1.0}
    else:
        raise ValueError(f"unknown symbol_kind: {symbol_kind!r}")

    if match_mode == "matched":
        te_kind = "matched"
        te_params: dict[str, Any] = {}
    else:
        te_kind, te_params = _mismatch_symbol(symbol_kind, cfg)

    return G1Config(
        P=P,
        B=1,
        query_mode=cfg.query_mode,
        matched_query_realization="independent",
        symbol_kind_tr=symbol_kind,
        symbol_params_tr=tr_params,
        symbol_kind_te=te_kind,
        symbol_params_te=te_params,
        task_spec_kind="power_law",
        task_spec_params={"nu_beta": cfg.task_spec_nu_beta},
        sigma=0.0,
        label_norm="sqrt_P",
        exact_mode=True,
        sample_data=False,
        population_mode=False,
        dtype=cfg.dtype,
    )


# ---------------------------------------------------------------------------
# Core B1 exact finite-P trial
#
# B1 exact = L-layer discrete residual/query recursion in layer index ell.
# DO NOT confuse with the stationary gamma_k(t) gradient flow (B2/bridge).
# ---------------------------------------------------------------------------


def _run_b1_exact_trial(
    cfg: B1Config,
    P: int,
    L: int,
    symbol_kind: str,
    match_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run one B1 exact finite-P trial.

    Constructs G = Sigma_tr (circulant, eigenvalues s_tr) and
    G_star = Sigma_te (circulant, eigenvalues s_te; equals G for matched).
    Uses y = e_0 (all-mode-support deterministic label).

    Runs the L-layer residual recursion in LAYER INDEX ell:
        r^0 = y
        r^{ell+1} = (I - G/L) r^ell,   ell = 0, ..., L-1

    Computes the query prediction:
        f = (1/L) G_star * sum_{ell=0}^{L-1} r^ell

    Transforms to the sample-space Fourier basis F_P (unitary DFT) and
    compares to theorem predictions:
        rhat_th[ell, k] = (1 - lambda_k/L)^ell / sqrt(P)
        h_th[k] = 1 - (1 - lambda_k/L)^L           [matched]
                  lambda*_k * phi_L(lambda_k)        [general]

    Returns a dict with raw arrays and four primary acceptance metrics.
    """
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32

    # ---- Operator-level objects from G1 ---
    g1_cfg = _build_g1_config(cfg, P, symbol_kind, match_mode)
    op = g1_generate(g1_cfg)

    s_tr = op["s_tr"].to(dtype=dtype, device=device)      # (P,) eigenvalues of G
    s_te = op["s_te"].to(dtype=dtype, device=device)      # (P,) eigenvalues of G_star
    G = op["Sigma_tr"].to(dtype=dtype, device=device)     # (P, P) circulant
    G_star = op["Sigma_te"].to(dtype=dtype, device=device)  # (P, P) circulant

    # Enforce exact equality in matched mode (no floating-point drift from
    # independent generator calls that differ by reference).
    if match_mode == "matched":
        G_star = G.clone()
        s_te = s_tr.clone()

    # ---- Deterministic label: y = e_0 ---
    y = _deterministic_y(P, dtype, device)
    sqrt_P = math.sqrt(P)
    # yhat_k = 1/sqrt(P) for all k (unitary DFT of e_0 is flat)

    # ---- L-layer residual recursion in layer index ell ---
    # B1 exact: runs L steps (NOT T gradient-flow steps).
    I_P = torch.eye(P, dtype=dtype, device=device)
    M = I_P - G / float(L)   # contraction operator (P, P)

    residuals: list[torch.Tensor] = []
    r = y.clone()
    residuals.append(r.clone())      # r^0 = y
    for _ in range(L):
        r = M @ r
        residuals.append(r.clone()) # r^{ell}, ell = 1, ..., L
    # residuals[ell] = r^ell, shape (P,), for ell = 0 ... L

    # ---- Query prediction ---
    sum_r = torch.stack(residuals[:L], dim=0).sum(dim=0)  # sum_{ell=0}^{L-1} r^ell
    f = G_star @ sum_r / float(L)

    # ---- Fourier transform to sample-space basis F_P ---
    # unitary_dft returns complex128; imaginary parts are ~0 for real y, G, G_star.
    yhat = _uft(y.cpu())                                          # (P,) complex
    rhat_emp_c = torch.stack([_uft(residuals[el].cpu())
                               for el in range(L + 1)], dim=0)   # (L+1, P) complex
    fhat_emp_c = _uft(f.cpu())                                    # (P,) complex

    # Real parts (imag parts are float-eps noise for real-valued inputs).
    rhat_emp = rhat_emp_c.real.to(dtype)    # (L+1, P)
    fhat_emp = fhat_emp_c.real.to(dtype)    # (P,)
    yhat_re  = yhat.real.to(dtype)          # (P,); all ≈ 1/sqrt(P)

    # ---- Theorem predictions ---
    # Residual: rhat_th[ell, k] = (1 - lambda_k/L)^ell * yhat_k
    # = (1 - s_tr_k/L)^ell / sqrt(P)  [since yhat_k = 1/sqrt(P) for e_0]
    s_tr_cpu = s_tr.cpu().to(dtype)
    s_te_cpu = s_te.cpu().to(dtype)
    ell_idx = torch.arange(L + 1, dtype=dtype)          # (L+1,)
    cf = (1.0 - s_tr_cpu.unsqueeze(0) / float(L))       # (1, P) contraction factor
    rhat_th = (cf ** ell_idx.unsqueeze(1)) / sqrt_P     # (L+1, P)

    # Transfer functions.
    phi_L_lam = _phi_L_vec(s_tr_cpu, L)                           # (P,)
    h_th_matched = 1.0 - (1.0 - s_tr_cpu / float(L)).pow(L)      # (P,)
    h_th_general = s_te_cpu * phi_L_lam                           # (P,)
    h_th = h_th_matched if match_mode == "matched" else h_th_general

    # Empirical transfer: h_emp[k] = fhat_k / yhat_k = fhat_k * sqrt(P)
    h_emp = fhat_emp * sqrt_P   # (P,)

    # Terminal residual contraction (NOT transfer function; for optional Fig 5).
    c_L = (1.0 - s_tr_cpu / float(L)).pow(L)   # (P,), labeled clearly

    # ---- Off-diagonal Fourier energies ---
    # Both G and G_star must be circulant; energy should be ~0.
    train_offdiag = float(off_diagonal_fourier_energy(G.cpu()))
    query_offdiag = float(off_diagonal_fourier_energy(G_star.cpu()))

    # ---- Primary acceptance metrics ---
    resid_abs = (rhat_emp - rhat_th).abs()
    resid_abs_max = float(resid_abs.max().item())
    resid_th_mag = float(rhat_th.abs().max().item())
    residual_mode_rel_err = resid_abs_max / (resid_th_mag + 1e-30)

    h_abs = (h_emp - h_th).abs()
    h_abs_max = float(h_abs.max().item())
    h_th_mag = float(h_th.abs().max().item())
    transfer_rel_err = h_abs_max / (h_th_mag + 1e-30)

    # Diagnostic L2 transfer error.
    h_err_l2 = float(transfer_function_error(h_emp, h_th))

    # Imaginary-part check (should be float-eps noise).
    rhat_imag_max = float(rhat_emp_c.imag.abs().max().item())
    fhat_imag_max = float(fhat_emp_c.imag.abs().max().item())

    return {
        # Identity
        "P": int(P),
        "L": int(L),
        "symbol_kind": symbol_kind,
        "match_mode": match_mode,
        # Spectra (eigenvalue arrays)
        "s_tr": s_tr_cpu,                # (P,) training eigenvalues lambda_k
        "s_te": s_te_cpu,                # (P,) query eigenvalues lambda*_k
        # Residual recursion in Fourier basis (x-axis = ell)
        "rhat_emp": rhat_emp,            # (L+1, P) empirical, real
        "rhat_th": rhat_th,              # (L+1, P) theorem prediction, real
        # Transfer (query prediction)
        "h_emp": h_emp,                  # (P,) empirical transfer h_{L,k}
        "h_th": h_th,                    # (P,) theorem transfer (matched or general)
        "h_th_matched": h_th_matched,    # (P,) matched formula (always stored)
        "h_th_general": h_th_general,    # (P,) general formula (always stored)
        # Terminal residual contraction (NOT transfer function)
        "c_L": c_L,                      # (P,) = (1 - lambda_k/L)^L
        # Primary acceptance metrics
        "residual_mode_rel_err": float(residual_mode_rel_err),
        "residual_mode_abs_err": float(resid_abs_max),
        "transfer_rel_err": float(transfer_rel_err),
        "transfer_abs_err": float(h_abs_max),
        "train_offdiag_fourier_energy": float(train_offdiag),
        "query_offdiag_fourier_energy": float(query_offdiag),
        # Diagnostic
        "transfer_err_l2": float(h_err_l2),
        "rhat_imag_max": float(rhat_imag_max),
        "fhat_imag_max": float(fhat_imag_max),
        "resid_th_magnitude_max": float(resid_th_mag),
        "h_th_magnitude_max": float(h_th_mag),
    }


# ---------------------------------------------------------------------------
# Bridge: stationary gamma_k(t) dynamics
#
# B2/stationary bridge = optimization-time gamma_k(t) gradient flow.
# This is NOT the primary B1 theorem object.
# Only executed when cfg.bridge_to_b2 = True.
# ---------------------------------------------------------------------------


def _extract_symbol_from_circulant(
    C: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Real symbol of a real-symmetric-circulant matrix C via unnormalized DFT
    of its first column.  Used by the bridge stationary recursion."""
    c = C[:, 0].to(torch.complex128)
    s = torch.fft.fft(c).real.to(dtype)
    return s


def _bridge_matrix_circulant_recursion(
    Sigma: torch.Tensor,
    Omega: torch.Tensor,
    eta: float,
    L: int,
    T: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bridge mode only (B2/stationary).

    Runs the stationary reduced-Gamma gradient flow:
        Gamma(t+1) = Gamma(t) + eta * Omega * Sigma^2 * (I - L^{-1} Sigma Gamma(t))^{2L-1}
    from Gamma(0) = 0.

    Returns gamma_traj (T+1, P) and Gamma_final (P, P).
    These are OPTIMIZATION-TIME quantities, not layer-index quantities.
    """
    Sigma = Sigma.to(device=device, dtype=dtype)
    Omega = Omega.to(device=device, dtype=dtype)
    P = Sigma.shape[0]
    Gamma = torch.zeros(P, P, dtype=dtype, device=device)
    I_P = torch.eye(P, dtype=dtype, device=device)
    gamma_traj = torch.zeros(T + 1, P, dtype=dtype, device=device)
    exponent = 2 * L - 1
    Omega_Sigma_sq = Omega @ (Sigma @ Sigma)
    for t in range(T):
        M = I_P - (Sigma @ Gamma) / L
        M_pow = M.matrix_power(exponent)
        Gamma = Gamma + eta * (Omega_Sigma_sq @ M_pow)
        gamma_traj[t + 1] = _extract_symbol_from_circulant(Gamma, dtype)
    return gamma_traj, Gamma


def _bridge_run_stationary_trial(
    cfg: B1Config,
    P: int,
    L: int,
    symbol_kind: str,
    device: torch.device,
) -> dict[str, Any]:
    """Bridge mode only.  Runs the stationary gamma_k(t) gradient flow.

    Returns a dict compatible with the bridge plotting functions.
    gamma_emp and gamma_star are (T+1, P) optimization-time trajectories.
    These are labeled clearly as bridge quantities, NOT B1 exact quantities.
    """
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    g1_cfg = _build_g1_config(cfg, P, symbol_kind, "matched")
    op = g1_generate(g1_cfg)
    s_tr = op["s_tr"]
    omega = op["omega"]
    Sigma_tr = op["Sigma_tr"]
    Omega_phys = circulant_from_symbol(omega)

    t0 = time.perf_counter()
    gamma_emp_dev, _ = _bridge_matrix_circulant_recursion(
        Sigma_tr, Omega_phys, cfg.eta, L, cfg.T, dtype, device
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_matrix = time.perf_counter() - t0

    t0 = time.perf_counter()
    gamma_star = gamma_star_trajectory_circulant(
        s_tr, omega, L=L, eta=cfg.eta, T=cfg.T
    )
    t_theory = time.perf_counter() - t0

    gamma_emp = gamma_emp_dev.detach().cpu()

    abs_err = (gamma_emp - gamma_star).abs()
    mode_abs_err_max = float(abs_err.max().item())
    gs_mag = float(gamma_star.abs().max().item())
    mode_rel_err_max = mode_abs_err_max / (gs_mag + 1e-30)

    return {
        "P": int(P),
        "L": int(L),
        "symbol_kind": symbol_kind,
        "match_mode": "matched",
        "s_tr": s_tr.detach().cpu(),
        "omega": omega.detach().cpu(),
        # B2/bridge: optimization-time gamma_k(t) trajectories
        "gamma_emp": gamma_emp,       # (T+1, P) — step t, NOT layer ell
        "gamma_star": gamma_star,     # (T+1, P) — step t, NOT layer ell
        "bridge_mode_rel_err_max": mode_rel_err_max,
        "bridge_mode_abs_err_max": mode_abs_err_max,
        "bridge_matrix_seconds": float(t_matrix),
        "bridge_theory_seconds": float(t_theory),
    }


# ---------------------------------------------------------------------------
# Primary B1 figures (default, no bridge)
# ---------------------------------------------------------------------------


def _plot_residual_mode_trajectories(
    trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Figure 1 (primary B1 figure): Fourier residual mode trajectories vs LAYER INDEX ell.

    x-axis: layer index ell = 0, 1, ..., L
    y-axis: Re(rhat_k^ell) = (1 - lambda_k/L)^ell / sqrt(P)
    Shows: empirical recursion (colored) and exact theorem prediction (dashed black).
    Should overlap exactly to machine precision.

    Uses only matched trials (G_star = G) since the residual recursion does not
    depend on G_star.
    """
    import matplotlib.pyplot as plt

    slice_trials = [
        t for t in trials
        if t["P"] == cfg.figure_P
        and t["symbol_kind"] == cfg.figure_symbol
        and t["L"] in cfg.figure_L_list
        and t["match_mode"] == "matched"
    ]
    slice_trials.sort(key=lambda t: t["L"])
    if not slice_trials:
        return

    n_L = len(slice_trials)
    fig, axes = plt.subplots(1, n_L, figsize=(4.0 * n_L, 4.0), sharey=False)
    if n_L == 1:
        axes = [axes]

    valid_modes = [k for k in cfg.figure_mode_indices if k < cfg.figure_P]
    mode_colors = sequential_colors(len(valid_modes), palette="rocket")

    for ax, trial in zip(axes, slice_trials):
        L = trial["L"]
        ell_axis = np.arange(L + 1)
        rhat_emp = trial["rhat_emp"].numpy()   # (L+1, P)
        rhat_th  = trial["rhat_th"].numpy()    # (L+1, P)
        for color, k in zip(mode_colors, valid_modes):
            ax.plot(ell_axis, rhat_emp[:, k], color=color, lw=1.6,
                    label=f"k={k}", alpha=0.9)
            overlay_reference(
                ax, ell_axis, rhat_th[:, k],
                label="exact theory" if k == valid_modes[0] else None,
                style="--", color="black", lw=1.0
            )
        ax.set_title(f"$L = {L}$", fontsize=11)
        ax.set_xlabel(r"layer index $\ell$", fontsize=10)
        ax.set_xticks(ell_axis)
    axes[0].set_ylabel(
        r"$\hat{r}_k^{(\ell)}$ in sample-space Fourier basis", fontsize=10
    )
    axes[-1].legend(loc="best", fontsize=8, frameon=True, ncol=2)
    fig.suptitle(
        f"B1 Fig 1: residual mode trajectories vs layer index"
        f" ({cfg.figure_symbol}, P={cfg.figure_P})\n"
        "dashed black = exact theorem prediction",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_both(fig, run_dir, "b1_exact_residual_mode_trajectories")
    plt.close(fig)


def _plot_transfer_matched(
    trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Figure 2: matched transfer spectrum h_{L,k} = 1 - (1 - lambda_k/L)^L.

    x-axis: mode index k = 0, ..., P-1 (sample-space Fourier modes)
    y-axis: h_{L,k}
    Shows empirical h_emp[k] = fhat_k * sqrt(P) vs theorem h_th_matched[k].
    """
    import matplotlib.pyplot as plt

    slice_trials = [
        t for t in trials
        if t["P"] == cfg.transfer_P
        and t["symbol_kind"] == cfg.transfer_symbol
        and t["L"] in cfg.transfer_L_list
        and t["match_mode"] == "matched"
    ]
    slice_trials.sort(key=lambda t: t["L"])
    if not slice_trials:
        return

    P = slice_trials[0]["P"]
    k_axis = np.arange(P)
    depth_colors = sequential_colors(len(slice_trials), palette="rocket")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for color, trial in zip(depth_colors, slice_trials):
        ax.plot(k_axis, trial["h_emp"].numpy(), color=color, lw=1.6,
                label=f"empirical L={trial['L']}", alpha=0.9, zorder=2)
    for i, trial in enumerate(slice_trials):
        overlay_reference(
            ax, k_axis, trial["h_th_matched"].numpy(),
            label="theorem $h_{L,k}$" if i == 0 else None,
            style="--", color="black", lw=1.1, zorder=3,
        )
    ax.set_xlabel("sample-space Fourier mode index $k$", fontsize=10)
    ax.set_ylabel(
        r"$h_{L,k} = 1 - (1 - \lambda_k/L)^L$", fontsize=10
    )
    fig.suptitle(
        f"B1 Fig 2: matched transfer spectrum $h_{{L,k}}$ depth sweep\n"
        f"({cfg.transfer_symbol}, P={P}); dashed black = theorem",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=True, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_both(fig, run_dir, "b1_exact_transfer_spectrum_matched")
    plt.close(fig)


def _plot_transfer_mismatched(
    trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Figure 3: mismatched transfer spectrum h_{L,k} = lambda*_k * phi_L(lambda_k).

    Validates the GENERAL theorem formula when G_star != G (lambda*_k != lambda_k).
    x-axis: mode index k = 0, ..., P-1
    y-axis: h_{L,k}
    Shows empirical vs theorem h_th_general for selected depths.
    """
    import matplotlib.pyplot as plt

    slice_trials = [
        t for t in trials
        if t["P"] == cfg.transfer_P
        and t["symbol_kind"] == cfg.transfer_symbol
        and t["L"] in cfg.transfer_L_list
        and t["match_mode"] == "mismatched"
    ]
    slice_trials.sort(key=lambda t: t["L"])
    if not slice_trials:
        return

    P = slice_trials[0]["P"]
    k_axis = np.arange(P)
    depth_colors = sequential_colors(len(slice_trials), palette="mako")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for color, trial in zip(depth_colors, slice_trials):
        ax.plot(k_axis, trial["h_emp"].numpy(), color=color, lw=1.6,
                label=f"empirical L={trial['L']}", alpha=0.9, zorder=2)
    for i, trial in enumerate(slice_trials):
        overlay_reference(
            ax, k_axis, trial["h_th_general"].numpy(),
            label=r"theorem $\lambda^*_k \phi_L(\lambda_k)$" if i == 0 else None,
            style="--", color="black", lw=1.1, zorder=3,
        )
    ax.set_xlabel("sample-space Fourier mode index $k$", fontsize=10)
    ax.set_ylabel(
        r"$h_{L,k} = \lambda^*_k \, \phi_L(\lambda_k)$", fontsize=10
    )
    fig.suptitle(
        f"B1 Fig 3: mismatched transfer spectrum (G_star != G)\n"
        f"({cfg.transfer_symbol} train; mismatched query, P={P});"
        " dashed black = theorem",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=True, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_both(fig, run_dir, "b1_exact_transfer_spectrum_mismatched")
    plt.close(fig)


def _plot_error_summary(
    trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Figure 4: four acceptance metrics across all (P, L, symbol, match_mode) trials.

    Metrics shown (log scale):
      - residual_mode_rel_err       (max-magnitude-scaled)
      - transfer_rel_err            (max-magnitude-scaled)
      - train_offdiag_fourier_energy (should be 0 for circulant G)
      - query_offdiag_fourier_energy (should be 0 for circulant G_star)

    Tolerance line at cfg.tol_err.
    """
    import matplotlib.pyplot as plt

    labels = [
        f"P={t['P']} L={t['L']} {t['symbol_kind'][:4]} {t['match_mode'][:3]}"
        for t in trials
    ]
    resid_errs = np.array([t["residual_mode_rel_err"] for t in trials])
    trans_errs  = np.array([t["transfer_rel_err"] for t in trials])
    train_ode   = np.array([t["train_offdiag_fourier_energy"] for t in trials])
    query_ode   = np.array([t["query_offdiag_fourier_energy"] for t in trials])
    floor = 1e-18

    fig, ax = plt.subplots(figsize=(max(7.0, 0.35 * len(trials)), 4.5))
    x = np.arange(len(trials))
    ax.scatter(x, np.maximum(resid_errs, floor), s=20,
               label="residual mode rel err", color="C0", zorder=3)
    ax.scatter(x, np.maximum(trans_errs, floor), s=20, marker="^",
               label="transfer rel err", color="C1", zorder=3)
    ax.scatter(x, np.maximum(train_ode, floor), s=20, marker="s",
               label="train offdiag Fourier energy", color="C2", zorder=3)
    ax.scatter(x, np.maximum(query_ode, floor), s=20, marker="D",
               label="query offdiag Fourier energy", color="C3", zorder=3)
    ax.axhline(cfg.tol_err, color="red", lw=0.9, ls="--",
               label=f"tolerance ({cfg.tol_err:.0e})", zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, fontsize=6)
    ax.set_ylabel("error (log scale)", fontsize=10)
    ax.set_title(
        "B1 Fig 4: exact-closure errors across all (P, L, symbol, match_mode) trials",
        fontsize=10,
    )
    ax.legend(fontsize=7, frameon=True, ncol=2)
    fig.tight_layout()
    save_both(fig, run_dir, "b1_exact_error_summary")
    plt.close(fig)


def _plot_residual_contraction_spectrum(
    trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Figure 5 (optional): terminal residual contraction spectrum.

    c_{L,k} = (1 - lambda_k/L)^L   for each mode k.

    This is the terminal residual factor, NOT the transfer function h_{L,k}.
    Titled clearly to avoid confusion with the transfer function.
    """
    import matplotlib.pyplot as plt

    slice_trials = [
        t for t in trials
        if t["P"] == cfg.transfer_P
        and t["symbol_kind"] == cfg.transfer_symbol
        and t["L"] in cfg.transfer_L_list
        and t["match_mode"] == "matched"
    ]
    slice_trials.sort(key=lambda t: t["L"])
    if not slice_trials:
        return

    P = slice_trials[0]["P"]
    k_axis = np.arange(P)
    depth_colors = sequential_colors(len(slice_trials), palette="rocket")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for color, trial in zip(depth_colors, slice_trials):
        ax.plot(k_axis, trial["c_L"].numpy(), color=color, lw=1.5,
                label=f"L={trial['L']}", alpha=0.9)
    ax.set_xlabel("sample-space Fourier mode index $k$", fontsize=10)
    ax.set_ylabel(
        r"terminal residual contraction $c_{L,k} = (1 - \lambda_k/L)^L$",
        fontsize=10,
    )
    ax.set_title(
        f"B1 Fig 5: terminal residual contraction spectrum\n"
        f"(NOT the transfer function h_{{L,k}}; {cfg.transfer_symbol}, P={P})",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    save_both(fig, run_dir, "b1_exact_residual_contraction_spectrum")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bridge figures (B2/stationary; only produced when bridge_to_b2=True)
# ---------------------------------------------------------------------------


def _bridge_plot_gamma_trajectories(
    bridge_trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Bridge Fig: gamma_k(t) optimization-time trajectories (NOT B1 exact).

    x-axis: step t (optimization time, NOT layer index ell).
    This is the B2/stationary gradient-flow content.
    Saved as b1_bridge_stationary_gamma_trajectories.{png,pdf}.
    """
    import matplotlib.pyplot as plt

    slice_trials = [
        t for t in bridge_trials
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
    valid_modes = [k for k in cfg.figure_mode_indices if k < cfg.figure_P]
    mode_colors = sequential_colors(len(valid_modes), palette="rocket")
    t_axis = np.arange(cfg.T + 1)
    for ax, trial in zip(axes, slice_trials):
        for color, k in zip(mode_colors, valid_modes):
            emp = trial["gamma_emp"][:, k].numpy()
            thy = trial["gamma_star"][:, k].numpy()
            ax.plot(t_axis, emp, color=color, lw=1.4, label=f"k={k}", alpha=0.9)
            overlay_reference(ax, t_axis, thy, label=None,
                              style="--", color="black", lw=1.0)
        ax.set_title(f"L = {trial['L']}", fontsize=11)
        ax.set_xlabel("optimization step $t$")
        ax.set_xscale("log")
    axes[0].set_ylabel(r"$\gamma_k(t)$ [bridge: stationary GF, NOT B1 exact]")
    axes[-1].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        f"B1→B2 BRIDGE: gamma_k(t) stationary gradient-flow"
        f" ({cfg.figure_symbol}, P={cfg.figure_P})\n"
        "x-axis = optimization step t (NOT layer ell). dashed = theory.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_both(fig, run_dir, "b1_bridge_stationary_gamma_trajectories")
    plt.close(fig)


def _bridge_plot_gamma_loglog(
    bridge_trials: list[dict[str, Any]], cfg: B1Config, run_dir: ThesisRunDir
) -> None:
    """Bridge Fig: log-log gamma_k(t) trajectories (B2/stationary, NOT B1 exact)."""
    import matplotlib.pyplot as plt

    slice_trials = [
        t for t in bridge_trials
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
    valid_modes = [k for k in cfg.figure_mode_indices if k < cfg.figure_P]
    mode_colors = sequential_colors(len(valid_modes), palette="rocket")
    t_axis = np.arange(1, cfg.T + 1, dtype=float)
    for ax, trial in zip(axes, slice_trials):
        for color, k in zip(mode_colors, valid_modes):
            if k >= trial["P"]:
                continue
            emp = trial["gamma_emp"][1:, k].numpy()
            thy = trial["gamma_star"][1:, k].numpy()
            emp = np.where(emp > 0.0, emp, np.nan)
            thy = np.where(thy > 0.0, thy, np.nan)
            ax.plot(t_axis, emp, color=color, lw=1.4, label=f"k={k}", alpha=0.9)
            overlay_reference(ax, t_axis, thy, label=None,
                              style="--", color="black", lw=1.0)
        ax.set_title(f"L = {trial['L']}", fontsize=11)
        ax.set_xlabel("step $t$")
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].set_ylabel(r"$\gamma_k(t)$ [bridge, log-log]")
    axes[-1].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        f"B1→B2 BRIDGE: gamma_k(t) log-log ({cfg.figure_symbol}, P={cfg.figure_P})\n"
        "x-axis = optimization step t (NOT layer ell).",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_both(fig, run_dir, "b1_bridge_stationary_gamma_loglog")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI and config
# ---------------------------------------------------------------------------


def _parse_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_strs(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "B1: exact finite-P circulant mode closure "
            "(EXPERIMENT_PLAN_FINAL.MD §6.2). "
            "Primary theorem object: L-layer residual recursion in LAYER INDEX ell."
        )
    )
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    p.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--P-list", default=None,
                   help="comma-separated P values; default 16,32,64")
    p.add_argument("--L-list", default=None,
                   help="comma-separated L values; default 1,2,4,8")
    p.add_argument("--symbol-kinds", default=None,
                   help="comma-separated; default flat,power_law,multiband")
    p.add_argument("--tol-err", type=float, default=None)
    p.add_argument("--mismatch-symbol-kind", default=None,
                   help="override auto-selection of mismatched query symbol")
    # B2/bridge flags (off by default)
    p.add_argument("--bridge-to-b2", action="store_true",
                   help="also run and save B2/stationary bridge figures "
                        "(gamma_k(t) optimization-time plots; NOT B1 exact)")
    p.add_argument("--T", type=int, default=None,
                   help="bridge: stationary gradient-flow steps")
    p.add_argument("--eta", type=float, default=None,
                   help="bridge: stationary gradient-flow learning rate")
    return p.parse_args()


def _config_from_cli(args: argparse.Namespace) -> B1Config:
    overrides: dict[str, Any] = {}
    if args.dtype:
        overrides["dtype"] = args.dtype
    if args.device:
        overrides["device"] = args.device
    if args.P_list:
        overrides["P_list"] = _parse_ints(args.P_list)
    if args.L_list:
        overrides["L_list"] = _parse_ints(args.L_list)
    if args.symbol_kinds:
        overrides["symbol_kinds"] = _parse_strs(args.symbol_kinds)
    if args.tol_err is not None:
        overrides["tol_err"] = args.tol_err
    if args.mismatch_symbol_kind:
        overrides["mismatch_symbol_kind"] = args.mismatch_symbol_kind
    if args.bridge_to_b2:
        overrides["bridge_to_b2"] = True
    if args.T is not None:
        overrides["T"] = args.T
    if args.eta is not None:
        overrides["eta"] = args.eta
    return replace(B1Config(), **overrides) if overrides else B1Config()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "Source starter.sh or pass --device cpu."
            )
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _cli()
    if args.no_show:
        matplotlib.use("Agg")

    cfg = _config_from_cli(args)
    device = _resolve_device(cfg.device)
    print(f"[B1] device = {device}  bridge_to_b2 = {cfg.bridge_to_b2}")

    run = ThesisRunDir(__file__, phase="theoremB")
    with RunContext(run, config=cfg, seeds=[0, 1, 2, 3]) as ctx:
        apply_thesis_style()

        # ---- Primary B1 exact sweep ----
        # x-axis = layer index ell.  No optimization-time content.
        print("[B1] Running exact finite-P trials (matched + mismatched) ...")
        trials: list[dict[str, Any]] = []
        n_total = len(cfg.P_list) * len(cfg.L_list) * len(cfg.symbol_kinds) * 2
        t_sweep_start = time.perf_counter()
        idx = 0
        for P in cfg.P_list:
            for symbol_kind in cfg.symbol_kinds:
                for L in cfg.L_list:
                    for match_mode in ("matched", "mismatched"):
                        idx += 1
                        t0 = time.perf_counter()
                        trial = _run_b1_exact_trial(
                            cfg, P, L, symbol_kind, match_mode, device
                        )
                        dt = time.perf_counter() - t0
                        ctx.record_step_time(dt)
                        print(
                            f"  [{idx:>3d}/{n_total}] "
                            f"P={P:>3d} L={L:>2d} {symbol_kind:<10s} {match_mode:<10s} "
                            f"resid_rel={trial['residual_mode_rel_err']:.2e} "
                            f"h_rel={trial['transfer_rel_err']:.2e} "
                            f"tr_ode={trial['train_offdiag_fourier_energy']:.2e} "
                            f"qr_ode={trial['query_offdiag_fourier_energy']:.2e} "
                            f"({dt * 1000:.1f} ms)"
                        )
                        trials.append(trial)
        t_sweep = time.perf_counter() - t_sweep_start

        # ---- Acceptance check ----
        tol = cfg.tol_err
        failures: list[dict[str, Any]] = []
        for trial in trials:
            failed = []
            if trial["residual_mode_rel_err"] > tol:
                failed.append(("resid_rel", trial["residual_mode_rel_err"]))
            if trial["transfer_rel_err"] > tol:
                failed.append(("transfer_rel", trial["transfer_rel_err"]))
            if trial["train_offdiag_fourier_energy"] > tol:
                failed.append(("train_offdiag", trial["train_offdiag_fourier_energy"]))
            if trial["query_offdiag_fourier_energy"] > tol:
                failed.append(("query_offdiag", trial["query_offdiag_fourier_energy"]))
            if failed:
                failures.append({
                    "P": trial["P"], "L": trial["L"],
                    "symbol_kind": trial["symbol_kind"],
                    "match_mode": trial["match_mode"],
                    "metrics_failed": failed,
                })

        # ---- Save raw arrays ----
        npz_path = run.npz_path("b1_exact_circulant_modes")
        npz_dict: dict[str, np.ndarray] = {}
        for t in trials:
            key = f"P{t['P']}_L{t['L']}_{t['symbol_kind']}_{t['match_mode']}"
            npz_dict[key + "__rhat_emp"] = t["rhat_emp"].numpy()
            npz_dict[key + "__rhat_th"]  = t["rhat_th"].numpy()
            npz_dict[key + "__h_emp"]    = t["h_emp"].numpy()
            npz_dict[key + "__h_th"]     = t["h_th"].numpy()
            npz_dict[key + "__c_L"]      = t["c_L"].numpy()
            npz_dict[key + "__s_tr"]     = t["s_tr"].numpy()
            npz_dict[key + "__s_te"]     = t["s_te"].numpy()
        np.savez_compressed(npz_path, **npz_dict)

        # ---- Per-trial summary JSON ----
        summary_rows = [
            {
                "P": t["P"],
                "L": t["L"],
                "symbol_kind": t["symbol_kind"],
                "match_mode": t["match_mode"],
                "residual_mode_rel_err": t["residual_mode_rel_err"],
                "residual_mode_abs_err": t["residual_mode_abs_err"],
                "transfer_rel_err": t["transfer_rel_err"],
                "transfer_abs_err": t["transfer_abs_err"],
                "train_offdiag_fourier_energy": t["train_offdiag_fourier_energy"],
                "query_offdiag_fourier_energy": t["query_offdiag_fourier_energy"],
                "transfer_err_l2": t["transfer_err_l2"],
                "rhat_imag_max": t["rhat_imag_max"],
                "fhat_imag_max": t["fhat_imag_max"],
            }
            for t in trials
        ]
        (run.root / "per_trial_summary.json").write_text(
            json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8"
        )

        # ---- Primary B1 figures ----
        _plot_residual_mode_trajectories(trials, cfg, run)
        _plot_transfer_matched(trials, cfg, run)
        _plot_transfer_mismatched(trials, cfg, run)
        _plot_error_summary(trials, cfg, run)
        _plot_residual_contraction_spectrum(trials, cfg, run)

        # ---- Optional B2/stationary bridge ----
        bridge_trials: list[dict[str, Any]] = []
        if cfg.bridge_to_b2:
            print(
                "[B1→B2 bridge] Running stationary gamma_k(t) gradient flow "
                "(x-axis = optimization step t, NOT layer ell) ..."
            )
            for P in cfg.P_list:
                for symbol_kind in cfg.symbol_kinds:
                    for L in cfg.L_list:
                        bt = _bridge_run_stationary_trial(cfg, P, L, symbol_kind, device)
                        bridge_trials.append(bt)
                        print(
                            f"  bridge P={P} L={L} {symbol_kind:<10s} "
                            f"mode_rel={bt['bridge_mode_rel_err_max']:.2e}"
                        )
            _bridge_plot_gamma_trajectories(bridge_trials, cfg, run)
            _bridge_plot_gamma_loglog(bridge_trials, cfg, run)
            # Save bridge arrays.
            bridge_npz: dict[str, np.ndarray] = {}
            for bt in bridge_trials:
                key = f"bridge_P{bt['P']}_L{bt['L']}_{bt['symbol_kind']}"
                bridge_npz[key + "__gamma_emp"]  = bt["gamma_emp"].numpy()
                bridge_npz[key + "__gamma_star"] = bt["gamma_star"].numpy()
            np.savez_compressed(run.npz_path("b1_bridge_stationary_gamma"), **bridge_npz)

        # ---- Aggregate metrics ----
        max_resid_rel  = max(t["residual_mode_rel_err"] for t in trials)
        max_trans_rel  = max(t["transfer_rel_err"] for t in trials)
        max_train_ode  = max(t["train_offdiag_fourier_energy"] for t in trials)
        max_query_ode  = max(t["query_offdiag_fourier_energy"] for t in trials)
        max_resid_abs  = max(t["residual_mode_abs_err"] for t in trials)
        max_trans_abs  = max(t["transfer_abs_err"] for t in trials)
        max_imag_rhat  = max(t["rhat_imag_max"] for t in trials)
        max_imag_fhat  = max(t["fhat_imag_max"] for t in trials)

        ctx.record_compute_proxy(float(t_sweep))
        ctx.record_extra("n_trials", len(trials))
        ctx.record_extra("n_failures", len(failures))
        ctx.record_extra("failures", failures[:20])
        ctx.record_extra("bridge_to_b2", cfg.bridge_to_b2)
        ctx.record_extra("max_errors", {
            "residual_mode_rel_err": max_resid_rel,
            "transfer_rel_err": max_trans_rel,
            "train_offdiag_fourier_energy": max_train_ode,
            "query_offdiag_fourier_energy": max_query_ode,
        })

        ctx.write_summary({
            # --- Theorem provenance ---
            "plan_reference": "EXPERIMENT_PLAN_FINAL.MD §6.2 (B1)",
            "theorem_object": (
                "exact finite-P residual/query recursion in layer index ell; "
                "NOT optimization-time gamma_k(t) dynamics"
            ),
            "primary_theorem": (
                "rhat_k^ell = (1 - lambda_k/L)^ell * yhat_k; "
                "h_{L,k} = lambda*_k * phi_L(lambda_k) [general], "
                "= 1 - (1 - lambda_k/L)^L [matched]"
            ),
            "sample_space_fourier_basis": (
                "F_P = unitary DFT (P-dimensional sample-space); "
                "NOT feature-space eigenbasis"
            ),
            "deterministic_label": "y = e_0; yhat_k = 1/sqrt(P) for all k",
            # --- Sweep ---
            "category": "theorem-B exact finite-P circulant mode closure",
            "device": str(device),
            "n_trials": len(trials),
            "n_match_modes": 2,
            "bridge_to_b2": cfg.bridge_to_b2,
            # --- Acceptance ---
            "acceptance_tolerance": tol,
            "n_failures": len(failures),
            "status": "closure_ok" if not failures else "closure_failed",
            # --- Primary metrics ---
            "max_residual_mode_rel_err": max_resid_rel,
            "max_residual_mode_abs_err": max_resid_abs,
            "max_transfer_rel_err": max_trans_rel,
            "max_transfer_abs_err": max_trans_abs,
            "max_train_offdiag_fourier_energy": max_train_ode,
            "max_query_offdiag_fourier_energy": max_query_ode,
            "max_rhat_imag_component": max_imag_rhat,
            "max_fhat_imag_component": max_imag_fhat,
            # --- Timing ---
            "sweep_wallclock_seconds": round(t_sweep, 3),
            # --- Terminology note (binding) ---
            "terminology_note": (
                "residual_contraction c_{L,k} = (1-lambda_k/L)^L is stored in "
                "c_L and plotted in Fig 5 as 'terminal residual contraction'. "
                "It is NOT the transfer function h_{L,k}."
            ),
        })

        # ---- Console summary ----
        print()
        print("=" * 76)
        print(f" B1 exact finite-P circulant mode closure: {len(trials)} trials on {device}")
        print(f"   Theorem object: L-layer residual recursion in layer index ell")
        print(f"   (NOT optimization-time gamma_k(t) dynamics)")
        print()
        print(f"   max residual_mode_rel_err        = {max_resid_rel:.3e}")
        print(f"   max residual_mode_abs_err        = {max_resid_abs:.3e}")
        print(f"   max transfer_rel_err             = {max_trans_rel:.3e}")
        print(f"   max transfer_abs_err             = {max_trans_abs:.3e}")
        print(f"   max train offdiag Fourier energy = {max_train_ode:.3e}")
        print(f"   max query offdiag Fourier energy = {max_query_ode:.3e}")
        print(f"   max rhat imaginary component     = {max_imag_rhat:.3e}")
        print(f"   tolerance                        = {tol:.1e}")
        if failures:
            print(f"   FAILED trials: {len(failures)}")
            for f in failures[:5]:
                print(f"     P={f['P']} L={f['L']} {f['symbol_kind']} "
                      f"{f['match_mode']}: {f['metrics_failed']}")
        else:
            print("   ALL trials within tolerance  [PASS]")
        print("=" * 76)

        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
