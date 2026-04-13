"""Log-log power-law fits for the thesis.

Plan correspondence: EXPERIMENT_PLAN_FINAL.MD §4.3 (fit utilities). Step-1
Generator / Utility Specification v4 §9.

This module is the **single entry point** for every thesis exponent fit. All
log-log fits, bootstrap confidence intervals, and held-out evaluations must
go through these three functions; no inline log-log fits are permitted
anywhere else in the codebase.

Binding (v4 §9.1): :func:`fit_loglog` REQUIRES an explicit ``fit_window`` and
it is never auto-selected. Fit windows must be fixed a priori in configs so
that the thesis does not pick windows after looking at plots.

Return values are purely numerical / serializable: Python floats, real
``torch.Tensor`` objects, and plain dicts. No callables.
"""

from __future__ import annotations

import math
from typing import Any

import torch


_BOOTSTRAP_GENERATOR_SEED: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_fit_window(fit_window: Any) -> tuple[float, float]:
    if not isinstance(fit_window, (tuple, list)) or len(fit_window) != 2:
        raise TypeError(
            f"fit_window must be a 2-tuple of floats; got {type(fit_window).__name__}"
        )
    lo, hi = float(fit_window[0]), float(fit_window[1])
    if not (lo < hi):
        raise ValueError(f"fit_window must satisfy lo < hi; got ({lo}, {hi})")
    if lo <= 0:
        raise ValueError(
            f"fit_window lower bound must be strictly positive for log-log "
            f"fitting; got lo = {lo}"
        )
    return lo, hi


def _check_1d_real(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    name_x: str = "x",
    name_y: str = "y",
) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name_x} must be 1D; got shape {tuple(x.shape)}")
    if y.ndim != 1:
        raise ValueError(f"{name_y} must be 1D; got shape {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"{name_x} and {name_y} must have equal length; "
            f"got {x.shape[0]} vs {y.shape[0]}"
        )
    if not x.is_floating_point():
        raise TypeError(f"{name_x} must be real floating-point; got dtype {x.dtype}")
    if not y.is_floating_point():
        raise TypeError(f"{name_y} must be real floating-point; got dtype {y.dtype}")


# ---------------------------------------------------------------------------
# §9.1 fit_loglog
# ---------------------------------------------------------------------------


def fit_loglog(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    fit_window: tuple[float, float],
    heteroskedastic_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Weighted log-log least-squares fit of the power law

        log y = slope * log x + intercept

    restricted to points with ``x`` inside the inclusive ``fit_window``.

    ``fit_window`` is **required**; this function never auto-selects it
    (v4 §9.1 binding). ``heteroskedastic_weights`` (when provided) must have
    length equal to ``x``; it is filtered by the same mask applied to
    ``(x, y)``.

    Returns a dict with keys:

    - ``'slope'`` (float): fitted exponent.
    - ``'intercept'`` (float): fitted intercept of ``log y`` at ``log x = 0``.
    - ``'r2'`` (float): coefficient of determination on the log-transformed
      fit points (weighted when ``heteroskedastic_weights`` is provided).
    - ``'residuals'`` (float64 Tensor): log-space residuals
      ``log(y_fit) - (slope * log(x_fit) + intercept)`` for the fit points.
    - ``'fit_x'`` (float64 Tensor): the ``x`` values actually used (subset of
      the input inside ``fit_window``).
    - ``'fit_y'`` (float64 Tensor): the corresponding ``y`` values.

    Raises
    ------
    ValueError
        If fewer than 2 points fall inside ``fit_window``, if any ``x`` or
        ``y`` in the window is non-positive, or if weights have invalid
        shape / sign.
    """
    _check_1d_real(x, y)
    lo, hi = _validate_fit_window(fit_window)

    x64 = x.to(torch.float64)
    y64 = y.to(torch.float64)
    mask = (x64 >= lo) & (x64 <= hi)
    n = int(mask.sum().item())
    if n < 2:
        raise ValueError(
            f"fit_loglog: at least 2 points required inside fit_window {fit_window}; "
            f"got {n}"
        )

    x_fit_pts = x64[mask]
    y_fit_pts = y64[mask]
    if (x_fit_pts <= 0).any():
        raise ValueError("fit_loglog: all x values in fit_window must be > 0")
    if (y_fit_pts <= 0).any():
        raise ValueError("fit_loglog: all y values in fit_window must be > 0")

    lx = x_fit_pts.log()
    ly = y_fit_pts.log()

    # Design matrix [log x, 1]. Order of columns (slope, intercept) matches params.
    design = torch.stack([lx, torch.ones_like(lx)], dim=1)  # (n, 2)

    if heteroskedastic_weights is None:
        XtX = design.T @ design
        Xty = design.T @ ly
        params = torch.linalg.solve(XtX, Xty)  # (slope, intercept)
        ly_hat = design @ params
        resid = ly - ly_hat
        ss_res = resid.pow(2).sum().item()
        ss_tot = (ly - ly.mean()).pow(2).sum().item()
    else:
        w_full = heteroskedastic_weights
        if w_full.ndim != 1 or w_full.shape[0] != x.shape[0]:
            raise ValueError(
                f"heteroskedastic_weights must have shape ({x.shape[0]},); "
                f"got {tuple(w_full.shape)}"
            )
        if not w_full.is_floating_point():
            raise TypeError(
                f"heteroskedastic_weights must be real floating-point; "
                f"got dtype {w_full.dtype}"
            )
        w = w_full.to(torch.float64)[mask]
        if (w < 0).any():
            raise ValueError("heteroskedastic_weights must be non-negative")
        XtWX = design.T @ (design * w.unsqueeze(-1))
        XtWy = design.T @ (ly * w)
        params = torch.linalg.solve(XtWX, XtWy)
        ly_hat = design @ params
        resid = ly - ly_hat
        ss_res = (w * resid.pow(2)).sum().item()
        w_sum = w.sum().item()
        if w_sum > 0:
            ly_w_mean = (w * ly).sum().item() / w_sum
            ss_tot = (w * (ly - ly_w_mean).pow(2)).sum().item()
        else:
            ss_tot = 0.0

    slope = float(params[0].item())
    intercept = float(params[1].item())
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": float(r2),
        "residuals": resid.contiguous(),
        "fit_x": x_fit_pts.contiguous(),
        "fit_y": y_fit_pts.contiguous(),
    }


# ---------------------------------------------------------------------------
# §9.2 bootstrap_exponent
# ---------------------------------------------------------------------------


def bootstrap_exponent(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    fit_window: tuple[float, float],
    seed_axis: int,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap the log-log slope and intercept along the declared
    ``seed_axis`` of ``y``.

    For each of ``n_bootstrap`` iterations, ``n_seeds = y.shape[seed_axis]``
    indices are drawn with replacement along ``seed_axis``; ``y`` is
    averaged over those resampled indices to produce a 1-D curve aligned with
    ``x``; then :func:`fit_loglog` is applied with the shared ``fit_window``.
    The bootstrap collects slopes and intercepts and returns their mean and
    two-sided ``alpha``-level quantile envelopes.

    The internal :class:`torch.Generator` is seeded deterministically so that
    repeated calls with identical inputs produce identical outputs.

    Returns a dict with keys

        ``'slope_mean'``, ``'slope_lo'``, ``'slope_hi'``,
        ``'intercept_mean'``, ``'intercept_lo'``, ``'intercept_hi'``

    (all Python floats).
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D; got shape {tuple(x.shape)}")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("x and y must be real floating-point")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1; got {n_bootstrap}")

    seed_axis_norm = seed_axis if seed_axis >= 0 else y.ndim + seed_axis
    if not (0 <= seed_axis_norm < y.ndim):
        raise ValueError(
            f"seed_axis = {seed_axis} out of range for y with ndim = {y.ndim}"
        )
    n_seeds = y.shape[seed_axis_norm]
    if n_seeds < 1:
        raise ValueError("seed_axis dimension has zero samples; cannot bootstrap")

    # Validate fit_window once (fit_loglog re-validates per call; cheap).
    _validate_fit_window(fit_window)

    slopes = torch.zeros(n_bootstrap, dtype=torch.float64)
    intercepts = torch.zeros(n_bootstrap, dtype=torch.float64)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(_BOOTSTRAP_GENERATOR_SEED)

    for b in range(n_bootstrap):
        idx = torch.randint(0, n_seeds, (n_seeds,), generator=gen)
        y_resampled = torch.index_select(y, seed_axis_norm, idx)
        y_avg = y_resampled.mean(dim=seed_axis_norm)
        if y_avg.ndim != 1 or y_avg.shape[0] != x.shape[0]:
            raise ValueError(
                f"after averaging along seed_axis, y_avg must have shape "
                f"({x.shape[0]},); got {tuple(y_avg.shape)}"
            )
        result = fit_loglog(x, y_avg, fit_window=fit_window)
        slopes[b] = float(result["slope"])
        intercepts[b] = float(result["intercept"])

    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    return {
        "slope_mean": float(slopes.mean().item()),
        "slope_lo": float(torch.quantile(slopes, q_lo).item()),
        "slope_hi": float(torch.quantile(slopes, q_hi).item()),
        "intercept_mean": float(intercepts.mean().item()),
        "intercept_lo": float(torch.quantile(intercepts, q_lo).item()),
        "intercept_hi": float(torch.quantile(intercepts, q_hi).item()),
    }


# ---------------------------------------------------------------------------
# §9.3 holdout_evaluate
# ---------------------------------------------------------------------------


def holdout_evaluate(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    fit_window: tuple[float, float],
) -> dict[str, float]:
    """Fit the log-log model on ``(x_fit, y_fit)`` and evaluate predictive
    error on the held-out ``(x_val, y_val)``. Both sets are filtered by the
    same ``fit_window``.

    The prediction on the validation set is

        y_pred = exp(intercept_fit) * x_val^slope_fit

    and

        rel_err = | y_pred - y_val | / ( | y_val | + 1e-30 ).

    ``slope_val`` is an independent fit on the validation subset alone,
    reported so the caller can compare exponent stability across the
    train/validation split (v4 §10.2 metric).

    Returns ``{'median_rel_err', 'max_rel_err', 'slope_fit', 'slope_val'}``
    (all Python floats).
    """
    lo, hi = _validate_fit_window(fit_window)

    fit_result = fit_loglog(x_fit, y_fit, fit_window=fit_window)
    val_result = fit_loglog(x_val, y_val, fit_window=fit_window)
    slope = float(fit_result["slope"])
    intercept = float(fit_result["intercept"])

    # Filter validation points to fit_window for predictive-error evaluation.
    _check_1d_real(x_val, y_val, name_x="x_val", name_y="y_val")
    x_v_f64 = x_val.to(torch.float64)
    y_v_f64 = y_val.to(torch.float64)
    mask_val = (x_v_f64 >= lo) & (x_v_f64 <= hi)
    n_val = int(mask_val.sum().item())
    if n_val < 1:
        raise ValueError(
            f"no validation points in fit_window {fit_window}; cannot evaluate"
        )
    x_v = x_v_f64[mask_val]
    y_v = y_v_f64[mask_val]
    if (y_v <= 0).any():
        raise ValueError("holdout_evaluate: all y_val values in fit_window must be > 0")

    y_pred = math.exp(intercept) * x_v.pow(slope)
    rel = (y_pred - y_v).abs() / (y_v.abs() + 1e-30)

    return {
        "median_rel_err": float(rel.median().item()),
        "max_rel_err": float(rel.max().item()),
        "slope_fit": slope,
        "slope_val": float(val_result["slope"]),
    }


__all__ = [
    "fit_loglog",
    "bootstrap_exponent",
    "holdout_evaluate",
]
