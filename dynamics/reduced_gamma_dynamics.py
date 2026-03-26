"""Reduced Gamma model: learns an N x N matrix Gamma via SGD with spectral data.

This module implements a reduced-dimension model where the effective attention
operator is parameterized as an N x N matrix Gamma, projected into the full
d-dimensional space via a fixed projection matrix A of shape (N, d).

The effective d x d attention operator is:

    Gamma_eff = A^T @ Gamma @ A / N

At each SGD step:
  1. A random orthogonal matrix O is sampled via QR decomposition to rotate
     the covariance structure (simulating diverse task orientations).
  2. Context features X are drawn with power-law spectrum and rotated by O.
  3. Regression targets use Bernoulli-signed w_star rotated by O.
  4. The depth-L linear attention residual is computed:
     v <- (I - Gamma_eff @ Sigma_c / L)^L @ beta.
  5. The loss is the expected squared prediction error plus L2 regularization.
  6. Gamma is updated by gradient descent.

The module also provides ``powerlaw_loss_landscape`` for computing the
analytical loss landscape as a function of a scalar gamma.
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
        seed: Random seed for reproducibility (seeded on the target device).
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


def reduced_gamma_structured_sgd_rmt_isotropic_dynamics(
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
    rotate: bool = True,
    ctx_sample: bool = True,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[list[float], list[float], list[float]]:
    """Simulate SGD dynamics of the reduced Gamma model with QR-rotated spectral data.

    Learns an N x N matrix Gamma whose effective d x d operator is
    A^T @ Gamma @ A / N, where A is a fixed (N, d) projection matrix with
    sqrt(N) * I_{min(N,d)} in the top-left block.

    At each SGD step t:
      1. Sample a batch of B random orthogonal matrices O via QR decomposition
         of Gaussian matrices (seed 3t+1).
      2. If ctx_sample: draw X ~ N(0, diag(spec)), rotate to OX = O^T @ X,
         and form empirical covariance Sigma_c = OX^T OX / P.
         Otherwise: use the population covariance O^T diag(spec) O.
      3. Draw Bernoulli signs (seed 3t+2), set betas = O @ (w_star * signs).
      4. Compute the depth-L residual:
         v = (I - Gamma_eff @ Sigma_c / L)^L @ betas.
      5. Project back: Ov = O^T @ v, and compute spectral loss
         = sum(Ov^2 * spec) / B + lamb * mean(Gamma^2).
      6. Update Gamma via gradient descent: Gamma <- Gamma - eta * grad.
      7. Track eigenvalues of Gamma_eff for monitoring convergence.

    Args:
        spec: Eigenvalues of the input covariance. Shape ``(d,)``.
        w_star: Target weight norms per eigendirection. Shape ``(d,)``.
        N: Dimension of the reduced Gamma matrix.
        L: Number of attention layers (depth).
        B: Mini-batch size (number of tasks per SGD step).
        K: Number of training context tokens (currently unused in covariance
            construction but kept for API consistency).
        P: Number of context tokens used for empirical covariance.
        sigma: Noise parameter (currently unused, reserved for future extension).
        eta: Learning rate for gradient descent on Gamma.
        T: Number of SGD steps to simulate.
        lamb: L2 regularization coefficient on Gamma.
        rotate: Whether to use random rotations (currently always True by
            construction via QR).
        ctx_sample: If True, form empirical covariance from sampled context.
            If False, use the population covariance (no sampling noise).
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        A tuple ``(losses, mean_eigs, var_eigs)`` where:
            - ``losses``: List of T floats, the loss at each SGD step.
            - ``mean_eigs``: List of T floats, mean eigenvalue of Gamma_eff.
            - ``var_eigs``: List of T floats, normalized eigenvalue variance
              (Var(eig) / E[eig]^2) of Gamma_eff at each step.
    """
    device = torch.device(device)
    spec = spec.to(device=device, dtype=dtype)
    w_star = w_star.to(device=device, dtype=dtype)

    d = len(spec)
    Gamma = torch.zeros(N, N, device=device, dtype=dtype)

    # A: (N, d) with identity block scaled by sqrt(N)
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
            OX = torch.einsum("bdc,bpc->bpd", O, X)  # O^T X
            Sigma_c = torch.einsum("bpd,bpe->bde", OX, OX) / P
        else:
            Sigma_c = torch.einsum("bdc,bec,c->bde", O, O, spec)

        bernoulli = _bernoulli((B, d), 3 * t + 2, device=device, dtype=dtype)
        w_sign = w_star.unsqueeze(0) * bernoulli
        betas = torch.einsum("bdc,bc->bd", O, w_sign)

        # --- Compute loss ---
        A_G_A = (A.T @ Gamma @ A) / N  # (d, d)

        vs = betas.clone()  # (B, d)
        M = torch.einsum("b,de->bde", ones_B, eye_d) - (1.0 / L) * torch.einsum(
            "de,bef->bdf", A_G_A, Sigma_c
        )
        for _ in range(L):
            vs = torch.einsum("bde,be->bd", M, vs)

        Ov = torch.einsum("bdc,bd->bc", O, vs)  # O^T v → project back
        loss_t = torch.einsum("bc,bc,c->", Ov, Ov, spec) / B + lamb * torch.mean(Gamma**2)

        losses.append(float(loss_t.detach().cpu()))

        # --- Gradient step on Gamma ---
        Gamma_param = Gamma.detach().requires_grad_(True)
        A_G_A_g = (A.T @ Gamma_param @ A) / N
        vs_g = betas.detach().clone()
        M_g = torch.einsum("b,de->bde", ones_B, eye_d) - (1.0 / L) * torch.einsum(
            "de,bef->bdf", A_G_A_g, Sigma_c.detach()
        )
        for _ in range(L):
            vs_g = torch.einsum("bde,be->bd", M_g, vs_g)
        Ov_g = torch.einsum("bdc,bd->bc", O.detach(), vs_g)
        loss_g = torch.einsum("bc,bc,c->", Ov_g, Ov_g, spec) / B + lamb * torch.mean(Gamma_param**2)

        (grad_Gamma,) = torch.autograd.grad(loss_g, Gamma_param)
        Gamma = (Gamma - eta * grad_Gamma).detach()

        # --- Track eigenvalues ---
        eigs = torch.linalg.eigvalsh(A.T @ Gamma @ A / N)
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


def powerlaw_loss_landscape(
    gammas: Tensor,
    spec: Tensor,
    w_star: Tensor,
    L: int = 4,
    lamb: float = 0.0,
) -> Tensor:
    """Compute the scalar-gamma loss landscape for a power-law spectrum.

    For each candidate gamma value, computes the depth-L linear attention loss
    assuming a diagonal (isotropic per eigendirection) attention operator:

        loss(gamma) = sum_j (1 - gamma * spec_j / L)^{2L} * spec_j * w_star_j^2

    This is the loss achieved by a depth-L model where each layer applies
    the same scalar attention strength gamma, with no per-direction adaptation.

    Note: the ``lamb`` parameter is accepted for API consistency but is NOT
    added to the returned loss (regularization is not included).

    Args:
        gammas: 1D tensor of candidate scalar gamma values. Shape ``(G,)``.
        spec: Eigenvalues of the input covariance. Shape ``(d,)``.
        w_star: Target weight norms per eigendirection. Shape ``(d,)``.
        L: Number of attention layers (depth).
        lamb: Regularization coefficient (unused in computation, kept for
            API consistency).

    Returns:
        Tensor of shape ``(G,)`` with the loss at each gamma value.
    """
    decay = (1.0 - (1.0 / L) * torch.outer(gammas, spec)).pow(2 * L)
    return torch.einsum("gm,m->g", decay, spec * w_star**2)
