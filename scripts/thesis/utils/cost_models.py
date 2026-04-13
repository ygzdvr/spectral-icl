"""Canonical compute proxy and wall-clock calibration.

Plan correspondence: EXPERIMENT_PLAN_FINAL.MD §4.4 (cost models). Step-1
Generator / Utility Specification v4 §§3, 8.

This module encodes **one** architecture: the canonical adaptive-first then
spectral hybrid (v4 §3.1),

    adaptive module  = L_A dense linear-attention layers  (per-layer cost P^2)
    spectral backbone = L_S FFT layers with bottleneck r  (per-layer cost P log P + P r,
                                                            natural log)

combined via

    C_proxy(t, P, L_A, L_S, r)
        = t * ( c_A * P^2 * L_A  +  c_S * (P log P + P r) * L_S ).

``c_A`` and ``c_S`` are O(1) hardware-dependent constants fit from measured
wall-clock via :func:`calibrate`. A secondary truncated-Fourier variant
``phi_spectral_trunc_linear = P * r`` is available for callers whose spectral
layer uses that cost model; it is **not** the default.

This module does NOT support configurable architecture composition. The
canonical thesis architecture is hardcoded by :func:`compute_proxy`;
robustness-tier variants (interleaved, parallel, local-attention hybrids) must
be modeled in a different utility.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import torch


# ---------------------------------------------------------------------------
# Per-layer cost primitives
# ---------------------------------------------------------------------------


def phi_adaptive(
    P: int | float | torch.Tensor,
) -> float | torch.Tensor:
    """Per-layer cost of the canonical dense linear-attention adaptive module:

        phi_adaptive(P) = P^2.

    Returns a Python float when ``P`` is a Python scalar and a ``float64``
    tensor when ``P`` is a tensor (useful for vectorized sweeps).
    """
    if isinstance(P, torch.Tensor):
        return P.to(torch.float64).pow(2)
    return float(P) ** 2


def phi_spectral_fft(
    P: int | float | torch.Tensor,
    r: int | float | torch.Tensor,
) -> float | torch.Tensor:
    """Per-layer cost of the canonical FFT-based spectral backbone:

        phi_spectral_fft(P, r) = P * log(P) + P * r,

    with the natural logarithm. Primary spectral cost model for
    :func:`compute_proxy`. Returns a Python float when both inputs are scalar,
    otherwise a ``float64`` tensor with broadcasting over tensor inputs.
    """
    if isinstance(P, torch.Tensor) or isinstance(r, torch.Tensor):
        Pt = torch.as_tensor(P, dtype=torch.float64)
        rt = torch.as_tensor(r, dtype=torch.float64)
        return Pt * torch.log(Pt) + Pt * rt
    P_f = float(P)
    return P_f * math.log(P_f) + P_f * float(r)


def phi_spectral_trunc_linear(
    P: int | float | torch.Tensor,
    r: int | float | torch.Tensor,
) -> float | torch.Tensor:
    """Secondary (truncated-Fourier) spectral-backbone cost model:

        phi_spectral_trunc_linear(P, r) = P * r.

    **Not** the default for :func:`compute_proxy`; provided for callers who
    know their spectral layer uses this cost model. The primary default is
    :func:`phi_spectral_fft`.
    """
    if isinstance(P, torch.Tensor) or isinstance(r, torch.Tensor):
        Pt = torch.as_tensor(P, dtype=torch.float64)
        rt = torch.as_tensor(r, dtype=torch.float64)
        return Pt * rt
    return float(P) * float(r)


# ---------------------------------------------------------------------------
# Canonical compute proxy (adaptive-first then spectral; hardcoded)
# ---------------------------------------------------------------------------


def compute_proxy(
    t: int | float | torch.Tensor,
    P: int | float | torch.Tensor,
    L_A: int | float | torch.Tensor,
    L_S: int | float | torch.Tensor,
    r: int | float | torch.Tensor,
    *,
    c_A: float = 1.0,
    c_S: float = 1.0,
    phi_S: Callable[..., Any] = phi_spectral_fft,
) -> float | torch.Tensor:
    """Canonical adaptive-first then spectral compute proxy (v4 §§3.2, 8):

        C_proxy = t * ( c_A * P^2 * L_A  +  c_S * phi_S(P, r) * L_S ).

    The default ``phi_S`` is :func:`phi_spectral_fft` (primary,
    ``P log P + P r``). Callers aware of a truncated-linear spectral layer may
    pass :func:`phi_spectral_trunc_linear`. ``c_A`` and ``c_S`` are O(1)
    hardware-dependent constants -- call :func:`calibrate` to fit them from
    measured wall-clock.

    This function encodes exactly one architecture (adaptive-first then
    spectral). Composition is NOT configurable; robustness-tier variants use
    a different utility.

    Returns a Python float when all inputs are scalar; if any input is a
    tensor, the result is a tensor (arithmetic broadcasts).
    """
    adaptive_term = float(c_A) * phi_adaptive(P) * L_A
    spectral_term = float(c_S) * phi_S(P, r) * L_S
    return t * (adaptive_term + spectral_term)


# ---------------------------------------------------------------------------
# Wall-clock timing
# ---------------------------------------------------------------------------


class WallClockMeter:
    """Context manager that records per-step wall-clock and aggregates to a total.

    Enter the manager to start a :func:`time.perf_counter` timer; call
    :meth:`step` after each iteration to record a per-step duration; on exit
    the total duration is frozen. :attr:`total_seconds` returns the elapsed
    time (live while the meter is open, frozen after exit) and
    :attr:`per_step_seconds` returns the list of step durations.

    Usage::

        with WallClockMeter() as meter:
            for i in range(N):
                ...       # work
                meter.step()
        total = meter.total_seconds
        steps = meter.per_step_seconds
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._end: float | None = None
        self._last: float | None = None
        self._per_step: list[float] = []

    def __enter__(self) -> "WallClockMeter":
        self._start = time.perf_counter()
        self._last = self._start
        self._end = None
        self._per_step = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._end = time.perf_counter()
        return False  # do not suppress exceptions

    def step(self) -> None:
        """Record the duration since the previous :meth:`step` (or :meth:`__enter__`)."""
        if self._last is None:
            raise RuntimeError(
                "WallClockMeter.step() called outside a 'with WallClockMeter()' block"
            )
        now = time.perf_counter()
        self._per_step.append(float(now - self._last))
        self._last = now

    @property
    def total_seconds(self) -> float:
        """Total elapsed seconds. Live while the meter is open; frozen after exit."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return float(end - self._start)

    @property
    def per_step_seconds(self) -> list[float]:
        """List of per-step durations (one entry per :meth:`step` call)."""
        return list(self._per_step)


# ---------------------------------------------------------------------------
# Calibration: fit (c_A, c_S) to measured wall-clock
# ---------------------------------------------------------------------------


def calibrate(
    runs: list[dict[str, Any]],
    *,
    phi_S: Callable[..., Any] = phi_spectral_fft,
) -> dict[str, Any]:
    """Least-squares fit of ``(c_A, c_S)`` to measured wall-clock over a grid
    of runs (v4 §8).

    Each run must be a dict with keys

        't', 'P', 'L_A', 'L_S', 'r', 'wall_clock_seconds'.

    The fit is linear in the two unknowns::

        wall_clock = c_A * (t * P^2 * L_A) + c_S * (t * phi_S(P, r) * L_S)
                     + residual.

    The default ``phi_S`` is :func:`phi_spectral_fft` (primary). The same
    ``phi_S`` must be used by downstream :func:`compute_proxy` calls for the
    calibrated constants to apply.

    Requires at least 2 runs with a non-degenerate design matrix. Raises
    :class:`ValueError` on insufficient or degenerate runs.

    Returns
    -------
    dict
        Keys ``'c_A'`` (float), ``'c_S'`` (float), ``'r2'`` (float),
        ``'residuals'`` (float64 tensor of shape ``(n_runs,)``).
    """
    n = len(runs)
    if n < 2:
        raise ValueError(f"calibrate requires at least 2 runs; got {n}")

    X = torch.zeros(n, 2, dtype=torch.float64)
    y = torch.zeros(n, dtype=torch.float64)
    required = ("t", "P", "L_A", "L_S", "r", "wall_clock_seconds")
    for i, run in enumerate(runs):
        for key in required:
            if key not in run:
                raise ValueError(
                    f"calibrate: run #{i} missing required key {key!r}"
                )
        t_val = float(run["t"])
        P_val = int(run["P"])
        L_A_val = float(run["L_A"])
        L_S_val = float(run["L_S"])
        r_val = float(run["r"])
        wc = float(run["wall_clock_seconds"])
        X[i, 0] = t_val * phi_adaptive(P_val) * L_A_val
        X[i, 1] = t_val * phi_S(P_val, r_val) * L_S_val
        y[i] = wc

    # Normal equations: X^T X c = X^T y.  LinAlgError on singular X^T X.
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        c = torch.linalg.solve(XtX, Xty)
    except RuntimeError as exc:
        raise ValueError(
            "calibrate: design matrix is singular; provide runs that vary "
            "both the adaptive term (t * P^2 * L_A) and the spectral term "
            "(t * phi_S(P, r) * L_S) independently"
        ) from exc

    residuals = y - X @ c
    ss_res = residuals.pow(2).sum().item()
    ss_tot = (y - y.mean()).pow(2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "c_A": float(c[0].item()),
        "c_S": float(c[1].item()),
        "r2": float(r2),
        "residuals": residuals,
    }


__all__ = [
    "phi_adaptive",
    "phi_spectral_fft",
    "phi_spectral_trunc_linear",
    "compute_proxy",
    "WallClockMeter",
    "calibrate",
]
