"""Samplers for the architecture-aligned thesis tier.

Each sampler is a zero-argument callable returning a fresh ICL batch
``(X_train, y_train, X_query, y_query)`` on CUDA, in the column-sample
convention ``X in R^{B x D x P}`` that the thesis models consume. Samplers
are consumed by :func:`scripts.thesis.architectures.training.train_icl_online`.

This module currently provides one sampler beyond the smoke-test isotropic
sampler already living in :mod:`scripts.thesis.architectures.training`:

- :func:`make_stationary_sampler` -- stationary-Gaussian data with a
  user-specified circulant token-position covariance. The population kernel
  of this data,
      E[(1/P) x_mu^T Gamma x_nu] = (tr(Gamma) / P) * [Sigma_token]_{mu - nu mod P},
  is circulant with symbol proportional to ``s_data`` regardless of ``Gamma``.
  This is what lets Theorem B-style spectral bottleneck phenomena appear
  once a :class:`SpectralAttention` trains on these contexts.

The construction avoids materializing the ``P x P`` circulant via the
Fourier-domain square-root filter: ``X = irfft(rfft(Z) * sqrt_symbol_half)``.
See the docstring of :func:`make_stationary_sampler` for details.

Conventions (matching :func:`make_isotropic_sampler`):

- CUDA-only (no CPU fallback).
- Sampler keeps the symbol / square-root symbol as closure variables; each
  call only draws fresh white noise and runs the FFT filter.
- No fixed seed inside the sampler -- draws come from torch's global CUDA
  RNG state so consecutive calls produce independent batches.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from scripts.thesis.utils.fourier_ops import (
    symbol_flat,
    symbol_power_law,
)


IclBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
SampleFn = Callable[[], IclBatch]


def make_stationary_sampler(
    P: int,
    D: int,
    K: int,
    B: int,
    symbol_kind: str = "power_law",
    nu: float = 1.0,
    sigma: float = 0.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
) -> tuple[SampleFn, dict[str, Any]]:
    """Stationary-Gaussian ICL sampler with a prescribed circulant token-
    position covariance.

    Construction. A length-``P`` real-even symbol ``s_data`` defines the
    circulant ``Sigma_token = F_P^H diag(s_data) F_P``. For each feature
    dimension independently, the length-``P`` time series follows
    ``N(0, Sigma_token)``. Implemented efficiently in the Fourier domain::

        Z         ~ N(0, 1)                     # shape (B, D, P)
        Z_rfft    = torch.fft.rfft(Z)           # shape (B, D, P // 2 + 1)
        X         = torch.fft.irfft(Z_rfft * sqrt_symbol_half, n=P)

    For isotropic features, the population kernel
    ``E[(1/P) x_mu^T Gamma x_nu] = (tr(Gamma) / P) * [Sigma_token]_{mu, nu}``
    is circulant with symbol proportional to ``s_data`` for **any** ``Gamma``.
    The trained :class:`SpectralAttention`'s learned ``Gamma`` therefore
    cannot break the circulant structure of the expected kernel.

    Query tokens ``X_query`` are drawn as the first ``K`` positions of an
    independent length-``P`` stationary sample -- i.e., from the same
    underlying stationary process, so the query marginal matches the training
    marginal. Requires ``K <= P``.

    Parameters
    ----------
    P
        Training context length (and the length of the circulant symbol).
    D
        Feature dimension.
    K
        Number of query positions per context. Must satisfy ``K <= P``.
    B
        Batch size.
    symbol_kind
        One of ``"power_law"`` (uses :func:`symbol_power_law`) or ``"flat"``
        (uses :func:`symbol_flat` with value ``1.0``; gives isotropic iid data).
    nu
        Power-law exponent. Only used when ``symbol_kind="power_law"``.
    sigma
        Label noise standard deviation. When positive, adds
        ``sigma * N(0, 1)`` to both ``y_train`` and ``y_query``.
    device, dtype
        Target device / dtype for the returned tensors. ``dtype`` is typically
        ``float32`` for training.

    Returns
    -------
    (sampler_fn, metadata)
        ``sampler_fn()`` returns a fresh ``(X_train, y_train, X_query,
        y_query)``. ``metadata`` is a dict with keys ``"symbol"``,
        ``"sqrt_symbol"`` (both length-``P`` CPU float64 tensors),
        ``"symbol_kind"``, ``"nu"``, ``"P"``, ``"D"`` -- used by theory
        overlays and post-training diagnostics.
    """
    if P < 1 or D < 1 or K < 1 or B < 1:
        raise ValueError(f"P, D, K, B must be >= 1; got ({P}, {D}, {K}, {B})")
    if K > P:
        raise ValueError(
            f"K ({K}) must be <= P ({P}); X_query is drawn as the first K "
            "positions of a length-P stationary sample."
        )
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0; got {sigma}")

    if symbol_kind == "power_law":
        symbol_cpu = symbol_power_law(P, float(nu))
    elif symbol_kind == "flat":
        symbol_cpu = symbol_flat(P, 1.0)
    else:
        raise ValueError(
            f"symbol_kind must be 'power_law' or 'flat'; got {symbol_kind!r}"
        )

    sqrt_symbol_cpu = torch.sqrt(torch.clamp(symbol_cpu, min=1e-30))
    # Device-resident half-spectrum of the square-root symbol -- the filter
    # applied in the Fourier domain. Keep as closure var (one-time cost).
    sqrt_half = sqrt_symbol_cpu[: P // 2 + 1].to(device=device, dtype=dtype)

    label_norm_factor = math.sqrt(D)

    def sample() -> IclBatch:
        # Train: length-P stationary Gaussian via circulant square-root filter.
        Z_train = torch.randn(B, D, P, device=device, dtype=dtype)
        Z_train_rfft = torch.fft.rfft(Z_train, dim=-1)
        X_train = torch.fft.irfft(Z_train_rfft * sqrt_half, n=P, dim=-1)
        # Query: independent length-P draw; first K positions used (same
        # stationary process, so the query marginal matches training).
        Z_query = torch.randn(B, D, P, device=device, dtype=dtype)
        Z_query_rfft = torch.fft.rfft(Z_query, dim=-1)
        X_query_full = torch.fft.irfft(Z_query_rfft * sqrt_half, n=P, dim=-1)
        X_query = X_query_full[:, :, :K].contiguous()
        # Labels.
        beta = torch.randn(B, D, device=device, dtype=dtype)
        y_train = torch.einsum("bd,bdp->bp", beta, X_train) / label_norm_factor
        y_query = torch.einsum("bd,bdk->bk", beta, X_query) / label_norm_factor
        if sigma > 0.0:
            y_train = y_train + sigma * torch.randn(B, P, device=device, dtype=dtype)
            y_query = y_query + sigma * torch.randn(B, K, device=device, dtype=dtype)
        return X_train, y_train, X_query, y_query

    metadata: dict[str, Any] = {
        "symbol": symbol_cpu,
        "sqrt_symbol": sqrt_symbol_cpu,
        "symbol_kind": symbol_kind,
        "nu": float(nu),
        "P": int(P),
        "D": int(D),
        "K": int(K),
        "B": int(B),
        "sigma": float(sigma),
    }
    return sample, metadata


__all__ = ["make_stationary_sampler", "IclBatch", "SampleFn"]
