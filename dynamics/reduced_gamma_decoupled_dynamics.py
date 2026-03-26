"""Decoupled-layer variant of the reduced Gamma model.

This module extends the reduced Gamma model (``reduced_gamma_dynamics``) by
giving each attention layer its own independent Gamma matrix. Instead of
sharing a single N x N matrix Gamma across all L layers, we maintain L
separate matrices Gamma[0], ..., Gamma[L-1].

The forward pass composes all layers:

    v_0 = beta
    v_{l+1} = (I - A^T Gamma[l] A / N @ Sigma_c / L) @ v_l
    loss = ||O^T v_L||_Sigma^2 / B

Gradients are computed through the full L-layer composition, and each
Gamma[l] is updated independently with a learning rate scaled by L:

    Gamma[l] <- Gamma[l] - eta * L * grad_l

The L-scaling ensures that each layer's per-step update has the same
magnitude regardless of depth (since each layer's contribution to the
overall residual is scaled by 1/L).

Eigenvalue tracking is done on layer 0's Gamma_eff = A^T Gamma[0] A / N.
"""
from __future__ import annotations

import math
import sys

import torch


Tensor = torch.Tensor


def _randn(shape: tuple[int, ...], seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Generate a standard normal tensor with a deterministic device-local seed.

    Args:
        shape: Shape of the output tensor.
        seed: Random seed for reproducibility.
        device: Target device for the output tensor and generator.
        dtype: Data type for the output tensor.

    Returns:
        Tensor of the given shape filled with i.i.d. N(0,1) values.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


def _bernoulli(shape: tuple[int, ...], seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Generate a Rademacher (+/-1) random tensor with a deterministic seed.

    Samples Bernoulli(0.5) and maps {0, 1} to {-1, +1}.

    Args:
        shape: Shape of the output tensor.
        seed: Random seed for reproducibility.
        device: Target device for the output tensor and generator.
        dtype: Data type for the output tensor.

    Returns:
        Tensor of the given shape with i.i.d. values in {-1, +1}.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    t = torch.empty(shape, device=device, dtype=dtype)
    t.bernoulli_(generator=gen)
    return 2.0 * t - 1.0


def reduced_gamma_decoupled_depth_structured_sgd_dynamics(
    spec: Tensor,
    w_star: Tensor,
    N: int,
    L: int,
    B: int,
    K: int,
    P: int,
    sigma: float = 0.0,
    eta: float = 0.01,
    T: int = 100,
    lamb: float = 1e-3,
    ctx_sample: bool = True,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[list[float], list[float], list[float]]:
    """Simulate SGD dynamics of the decoupled-layer reduced Gamma model.

    Each of the L attention layers has its own independent N x N matrix
    Gamma[l], updated via gradient descent. The forward pass composes all
    layers sequentially:

        v_0 = betas
        v_{l+1} = (I - (A^T Gamma[l] A / N) @ Sigma_c / L) @ v_l

    The loss is the expected squared prediction error:

        loss = sum_k spec_k * (O^T v_L)_k^2 / B

    Gradients flow through the full composition, and each Gamma[l] is
    updated with learning rate eta * L to compensate for the 1/L scaling
    per layer.

    Data generation follows the same protocol as the shared-Gamma variant:
    QR-rotated covariance, Bernoulli-signed targets.

    Args:
        spec: Eigenvalues of the input covariance. Shape ``(d,)``.
        w_star: Target weight norms per eigendirection. Shape ``(d,)``.
        N: Dimension of each per-layer Gamma matrix.
        L: Number of attention layers (depth). Also the number of Gamma matrices.
        B: Mini-batch size (number of tasks per SGD step).
        K: Number of training context tokens (unused, kept for API consistency).
        P: Number of context tokens for empirical covariance.
        sigma: Noise parameter (unused, reserved for future extension).
        eta: Base learning rate. Actual per-layer rate is eta * L.
        T: Number of SGD steps to simulate.
        lamb: L2 regularization coefficient (unused in current implementation
            but kept for API consistency).
        ctx_sample: If True, form empirical covariance from sampled context.
            If False, use the population covariance.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        A tuple ``(losses, mean_eigs, var_eigs)`` where:
            - ``losses``: List of T floats, the loss at each SGD step.
            - ``mean_eigs``: List of T floats, mean eigenvalue of layer 0's
              Gamma_eff = A^T Gamma[0] A / N.
            - ``var_eigs``: List of T floats, normalized eigenvalue variance
              of layer 0's Gamma_eff at each step.
    """
    device = torch.device(device)
    spec = spec.to(device=device, dtype=dtype)
    w_star = w_star.to(device=device, dtype=dtype)

    d = len(spec)
    Gamma = [torch.zeros(N, N, device=device, dtype=dtype) for _ in range(L)]

    m = min(N, d)
    A = torch.zeros(N, d, device=device, dtype=dtype)
    A[:m, :m] = torch.eye(m, device=device, dtype=dtype) * math.sqrt(N)

    eye_d = torch.eye(d, device=device, dtype=dtype)
    ones_B = torch.ones(B, device=device, dtype=dtype)

    losses: list[float] = []
    mean_eigs: list[float] = []
    var_eigs: list[float] = []

    for t in range(T):
        # --- Sample data ---
        O = torch.linalg.qr(
            _randn((B, d, d), 3 * t + 1, device=device, dtype=dtype)
        ).Q

        if ctx_sample:
            X = _randn((B, P, d), 3 * t, device=device, dtype=dtype)
            X = X * spec.sqrt().unsqueeze(0).unsqueeze(0)
            OX = torch.einsum("bdc,bpc->bpd", O, X)
            Sigma_c = torch.einsum("bpd,bpe->bde", OX, OX) / P
        else:
            Sigma_c = torch.einsum("bdc,bec,c->bde", O, O, spec)

        bernoulli = _bernoulli((B, d), 3 * t + 2, device=device, dtype=dtype)
        w_sign = w_star.unsqueeze(0) * bernoulli
        betas = torch.einsum("bdc,bc->bd", O, w_sign)

        # --- Forward pass & loss (with autograd) ---
        Gamma_params = [G.detach().requires_grad_(True) for G in Gamma]

        vs = betas.detach().clone()
        identity = torch.einsum("b,de->bde", ones_B, eye_d)

        for l_idx in range(L):
            A_G_A = (A.T @ Gamma_params[l_idx] @ A) / N
            M = identity - (1.0 / L) * torch.einsum("de,bef->bdf", A_G_A, Sigma_c.detach())
            vs = torch.einsum("bde,be->bd", M, vs)

        Ov = torch.einsum("bdc,bd->bc", O.detach(), vs)
        loss_t = torch.einsum("bc,bc,c->", Ov, Ov, spec) / B

        losses.append(float(loss_t.detach().cpu()))

        # --- Gradient step on each Gamma[l] ---
        grads = torch.autograd.grad(loss_t, Gamma_params)
        Gamma = [(G - eta * L * g).detach() for G, g in zip(Gamma, grads)]

        # --- Track eigenvalues of layer 0 ---
        eigs = torch.linalg.eigvalsh(A.T @ Gamma[0] @ A / N)
        mean_eig = float(eigs.mean().cpu())
        var_eig_t = float(
            ((eigs**2).mean() - eigs.mean() ** 2) / (eigs.mean() ** 2 + 1e-30)
        )
        mean_eigs.append(mean_eig)
        var_eigs.append(var_eig_t)

        sys.stdout.write(
            f"\r step = {t}  |  loss = {losses[-1]:.6f} | var_eigs = {var_eig_t:.6f}"
        )

    print()
    return losses, mean_eigs, var_eigs
