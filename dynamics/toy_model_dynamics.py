"""Toy scalar and multi-variable gradient descent models for studying ICL dynamics.

This module provides simplified dynamical models that capture the essential
nonlinear structure of transformer in-context learning. Instead of training
full transformer weights, these models optimize a small number of scalar
parameters whose product mimics the effective "gamma" parameter that controls
ICL performance.

The key insight is that the ICL loss decomposes spectrally as:

    Loss = sum_m (1 - gamma * spec_m / L)^{2L} * spec_m * w_star_m^2

where gamma is the product of several weight parameters. This module studies
the gradient descent dynamics of learning gamma through its factored
representation.

Two models are provided:
  - ``pretrain_dynamics``: A single scalar w with gamma = w^7.
  - ``pretrain_dynamics_two_var``: Six parameters (wx, wy, wk, wq, wv, wo)
    mimicking transformer weight structure, with gamma = wx^2 * wk * wq * wv.
"""
from __future__ import annotations

import math

import torch


Tensor = torch.Tensor


def pretrain_dynamics(
    spec: Tensor,
    w_star: Tensor,
    beta0: float = 1.0,
    L: int = 4,
    T: int = 100,
    eta: float = 0.05,
    w0: float = 0.5,
) -> Tensor:
    """Simulate gradient descent dynamics of a single-scalar toy model.

    This toy model reduces the ICL pretraining problem to optimizing a single
    scalar ``w``. The effective attention strength is gamma = w^7, chosen so
    that the degree matches the product of transformer weight matrices
    (wx^2 * wk * wq * wv * wo^2, where wo is frozen). The loss landscape is:

        Loss(w) = sum_m (1 - beta0/L * w^7 * spec_m)^{2L} * spec_m * w_star_m^2

    which captures the spectral decomposition of the ICL regression loss for
    a depth-L linear attention model.

    Args:
        spec: Eigenvalues of the input covariance matrix. Shape ``(d,)``.
        w_star: Squared norms of the target projected onto each eigendirection.
            Shape ``(d,)``.
        beta0: Residual connection scaling factor (beta/depth in the full model).
        L: Number of attention layers (depth). Controls the exponent 2L in the
            polynomial decay and the 1/L scaling of the per-layer update.
        T: Number of gradient descent steps to simulate.
        eta: Learning rate for gradient descent.
        w0: Initial value of the scalar parameter w.

    Returns:
        Tensor of shape ``(T,)`` containing the loss at each GD step.
    """
    device = spec.device
    dtype = spec.dtype

    w = torch.tensor(w0, device=device, dtype=dtype, requires_grad=True)
    losses = torch.zeros(T, device=device, dtype=dtype)

    for t in range(T):
        decay = (1.0 - beta0 / L * w.pow(7) * spec).pow(2 * L)
        loss = torch.sum(decay * spec * w_star**2)
        losses[t] = loss.detach()

        (g,) = torch.autograd.grad(loss, w)
        with torch.no_grad():
            w = (w - eta * g).detach().requires_grad_(True)

    return losses


def pretrain_dynamics_two_var(
    spec: Tensor,
    w_star: Tensor,
    beta0: float = 1.0,
    L: int = 4,
    T: int = 100,
    eta: float = 0.05,
    w0: float = 0.25,
) -> tuple[Tensor, list[Tensor]]:
    """Simulate gradient descent dynamics of a six-variable toy model.

    This model factorizes the effective attention parameter gamma into six
    scalar variables (wx, wy, wk, wq, wv, wo) that mimic the structure of
    transformer weight matrices. The effective gamma is wx^2 * wk * wq * wv;
    notably wy and wo do not appear in the loss (they receive zero gradients),
    demonstrating that only certain weight combinations matter for ICL.

    The loss function is:

        Loss = sum_m (1 - beta0/L * wx^2 * wk * wq * wv * spec_m)^{2L}
                     * spec_m * w_star_m^2

    Initial values: wx = sqrt(2) * w0, all others = w0.

    Args:
        spec: Eigenvalues of the input covariance matrix. Shape ``(d,)``.
        w_star: Squared norms of the target projected onto each eigendirection.
            Shape ``(d,)``.
        beta0: Residual connection scaling factor.
        L: Number of attention layers (depth).
        T: Number of gradient descent steps to simulate.
        eta: Learning rate for gradient descent.
        w0: Base initial value for all six parameters (wx is scaled by sqrt(2)).

    Returns:
        A tuple ``(losses, param_histories)`` where:
            - ``losses``: Tensor of shape ``(T,)`` with the loss at each step.
            - ``param_histories``: List of 6 tensors, each of shape ``(T,)``,
              recording the trajectory of [wx, wy, wk, wq, wv, wo] over time.
    """
    device = spec.device
    dtype = spec.dtype

    param_vals = [math.sqrt(2.0) * w0, w0, w0, w0, w0, w0]
    params = [
        torch.tensor(v, device=device, dtype=dtype, requires_grad=True)
        for v in param_vals
    ]

    losses = torch.zeros(T, device=device, dtype=dtype)
    param_history = [torch.zeros(T, device=device, dtype=dtype) for _ in range(6)]

    for t in range(T):
        wx, wy, wk, wq, wv, wo = params
        decay = (1.0 - beta0 / L * wx**2 * wk * wq * wv * spec).pow(2 * L)
        loss = torch.sum(decay * spec * w_star**2)

        losses[t] = loss.detach()
        for k in range(6):
            param_history[k][t] = params[k].detach()

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        with torch.no_grad():
            params = [
                (p - eta * (g if g is not None else zero)).detach().requires_grad_(True)
                for p, g in zip(params, grads)
            ]

    return losses, param_history
