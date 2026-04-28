"""Shared online training loop and isotropic ICL sampler.

:func:`train_icl_online` is a single-source online SGD loop that works with
any :class:`ICLRegressionModel` subclass: it samples a fresh batch from a
caller-supplied ``sample_fn`` per step, runs one forward / backward / optimizer
step, and re-applies :meth:`ICLRegressionModel.enforce_alignment` so
Assumption 1 stays at machine precision throughout training.

:func:`make_isotropic_sampler` returns a zero-argument sampler for i.i.d.
Gaussian ICL contexts - used by the smoke tests. Real experiment scripts wrap
the G1 / G2 thesis generators the same way and pass them to
:func:`train_icl_online`.

Convention: CUDA is required (no CPU fallback); column-sample convention
``X in R^{B x D x P}`` matches the thesis generators.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from scripts.thesis.architectures.models.base import ICLRegressionModel


IclBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
SampleFn = Callable[[], IclBatch]
DiagnosticFn = Callable[[ICLRegressionModel, int, float], dict[str, Any]]


def make_isotropic_sampler(
    D: int,
    P: int,
    K: int,
    B: int,
    sigma: float = 0.0,
    label_norm: str = "sqrt_D",
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
) -> SampleFn:
    """Return a zero-argument callable yielding fresh isotropic ICL batches.

    Each call draws ``beta, X_train, X_query`` i.i.d. from ``N(0, I)`` and
    computes
        ``y = beta^T X / sqrt(D)``        (or ``/ sqrt(P)`` when
                                           ``label_norm="sqrt_P"``),
    optionally adding ``sigma * N(0, 1)`` label noise. The sampler intentionally
    does NOT hold a seeded :class:`torch.Generator`: every call pulls fresh
    samples from torch's global (CUDA) RNG state, matching the online-training
    semantics of the Bordelon baseline.
    """
    if label_norm not in ("sqrt_D", "sqrt_P"):
        raise ValueError(f"label_norm must be 'sqrt_D' or 'sqrt_P'; got {label_norm!r}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0; got {sigma}")
    norm_factor = math.sqrt(D) if label_norm == "sqrt_D" else math.sqrt(P)

    def sample() -> IclBatch:
        beta = torch.randn(B, D, device=device, dtype=dtype)
        X_train = torch.randn(B, D, P, device=device, dtype=dtype)
        X_query = torch.randn(B, D, K, device=device, dtype=dtype)
        y_train = torch.einsum("bd,bdp->bp", beta, X_train) / norm_factor
        y_query = torch.einsum("bd,bdk->bk", beta, X_query) / norm_factor
        if sigma > 0.0:
            y_train = y_train + sigma * torch.randn(B, P, device=device, dtype=dtype)
            y_query = y_query + sigma * torch.randn(B, K, device=device, dtype=dtype)
        return X_train, y_train, X_query, y_query

    return sample


def train_icl_online(
    model: ICLRegressionModel,
    sample_fn: SampleFn,
    n_steps: int,
    lr: float = 1e-3,
    optimizer_cls: type = torch.optim.Adam,
    diagnostic_fn: DiagnosticFn | None = None,
    diagnostic_every: int = 100,
) -> dict[str, Any]:
    """Online training loop for any :class:`ICLRegressionModel`.

    Per step:

    1. ``X_train, y_train, X_query, y_query = sample_fn()``
    2. ``f_pred = model(X_train, y_train, X_query)``
    3. ``loss = model.loss(f_pred, y_query)``
    4. ``optimizer.zero_grad(); loss.backward(); optimizer.step()``
    5. ``model.enforce_alignment()`` to restore Assumption 1 invariants
    6. append ``loss.item()`` to ``losses``
    7. if ``diagnostic_fn`` is provided and ``step % diagnostic_every == 0``,
       append ``diagnostic_fn(model, step, loss)`` to ``diagnostics``.

    The optimizer is constructed inside this function so it sees the current
    parameter objects after any prior ``.to(device, dtype)`` moves.

    Returns ``{"losses": list[float], "diagnostics": list[dict]}``.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    if diagnostic_every < 1:
        raise ValueError(f"diagnostic_every must be >= 1; got {diagnostic_every}")

    opt = optimizer_cls(model.parameters(), lr=lr)

    losses: list[float] = []
    diagnostics: list[dict[str, Any]] = []

    for step in range(n_steps):
        X_train, y_train, X_query, y_query = sample_fn()
        opt.zero_grad()
        f_pred = model(X_train, y_train, X_query)
        loss = model.loss(f_pred, y_query)
        loss.backward()
        opt.step()
        model.enforce_alignment()

        loss_val = float(loss.detach().item())
        losses.append(loss_val)

        if diagnostic_fn is not None and (step % diagnostic_every == 0):
            diagnostics.append(dict(diagnostic_fn(model, step, loss_val)))

    return {"losses": losses, "diagnostics": diagnostics}


__all__ = ["make_isotropic_sampler", "train_icl_online", "IclBatch", "SampleFn", "DiagnosticFn"]
