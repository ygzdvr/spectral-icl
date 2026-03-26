"""Fixed-covariance variant of the reduced Gamma model and OOD rotation evaluation.

This module provides two key functions:

1. ``reduced_gamma_structured_fixed_sgd_rmt_isotropic_dynamics``:
   A simplified variant of the reduced Gamma model where the rotation matrix O
   is fixed to the identity. This preserves the diagonal covariance structure,
   making Gamma_eff learn independently along each eigendirection. The diagonal
   entries of A^T @ Gamma @ A / N directly correspond to per-eigenvalue
   attention strengths.

2. ``ood_loss_fixed_covariance``:
   Evaluates how the optimal fixed-covariance solution (Gamma = L * diag(spec^{-1}))
   degrades when the test covariance is rotated by an angle theta. The rotation
   is generated via matrix exponential of a random antisymmetric matrix:
   O_theta = exp(theta * A_rand), where A_rand = (R - R^T) / (2*sqrt(d)).
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


def reduced_gamma_structured_fixed_sgd_rmt_isotropic_dynamics(
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
    lamb: float = 1e-6,
    ctx_sample: bool = True,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[list[float], list[Tensor]]:
    """Simulate SGD dynamics of the fixed-covariance reduced Gamma model.

    Unlike ``reduced_gamma_structured_sgd_rmt_isotropic_dynamics``, this variant
    sets O = I (identity) for all tasks and all steps. This means:
      - The covariance structure is preserved (no rotation mixing eigendirections).
      - Gamma_eff = A^T @ Gamma @ A / N remains approximately diagonal.
      - Each eigendirection can be learned independently.

    This is useful for studying whether the model can discover the optimal
    per-eigenvalue attention strengths gamma_k = L / spec_k.

    At each SGD step t:
      1. If ctx_sample: draw X ~ N(0, diag(spec)) and form empirical covariance
         Sigma_c = X^T X / P. Otherwise: use population covariance diag(spec).
      2. Draw Bernoulli signs (seed 3t+2), set betas = w_star * signs.
      3. Compute depth-L residual: v = (I - Gamma_eff @ Sigma_c / L)^L @ betas.
      4. Compute loss = sum(v^2 * spec) / B + lamb * mean(Gamma^2).
      5. Update Gamma via gradient descent.
      6. Track diagonal of Gamma_eff (since O=I, these are the per-eigenvalue
         attention strengths).

    Args:
        spec: Eigenvalues of the input covariance. Shape ``(d,)``.
        w_star: Target weight norms per eigendirection. Shape ``(d,)``.
        N: Dimension of the reduced Gamma matrix.
        L: Number of attention layers (depth).
        B: Mini-batch size.
        K: Number of training context tokens (unused, kept for API consistency).
        P: Number of context tokens for empirical covariance.
        sigma: Noise parameter (unused, reserved for future extension).
        eta: Learning rate for gradient descent on Gamma.
        T: Number of SGD steps.
        lamb: L2 regularization coefficient on Gamma.
        ctx_sample: If True, form empirical covariance from sampled context.
            If False, use population covariance.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        A tuple ``(losses, all_eigs)`` where:
            - ``losses``: List of T floats, the loss at each SGD step.
            - ``all_eigs``: List of T tensors, each of shape ``(d,)``,
              containing the diagonal of A^T @ Gamma @ A / N (the per-eigenvalue
              attention strengths) at each step.
    """
    device = torch.device(device)
    spec = spec.to(device=device, dtype=dtype)
    w_star = w_star.to(device=device, dtype=dtype)

    d = len(spec)
    Gamma = torch.zeros(N, N, device=device, dtype=dtype)

    m = min(N, d)
    A = torch.zeros(N, d, device=device, dtype=dtype)
    A[:m, :m] = torch.eye(m, device=device, dtype=dtype) * math.sqrt(N)

    eye_d = torch.eye(d, device=device, dtype=dtype)
    ones_B = torch.ones(B, device=device, dtype=dtype)

    # O is batch identity (no rotation)
    O = torch.einsum("b,de->bde", ones_B, eye_d)  # (B, d, d)

    losses: list[float] = []
    all_eigs: list[Tensor] = []

    for t in range(T):
        # --- Sample data ---
        if ctx_sample:
            X = _randn((B, P, d), 3 * t, device=device, dtype=dtype)
            X = X * spec.sqrt().unsqueeze(0).unsqueeze(0)
            Sigma_c = torch.einsum("bpd,bpe->bde", X, X) / P
        else:
            Sigma_c = torch.einsum("b,de->bde", ones_B, torch.diag(spec))

        bernoulli = _bernoulli((B, d), 3 * t + 2, device=device, dtype=dtype)
        w_sign = w_star.unsqueeze(0) * bernoulli
        betas = torch.einsum("bde,be->bd", O, w_sign)  # O @ w_sign = w_sign

        # --- Compute loss ---
        A_G_A = (A.T @ Gamma @ A) / N

        vs = betas.clone()
        M = torch.einsum("b,de->bde", ones_B, eye_d) - (1.0 / L) * torch.einsum(
            "de,bef->bdf", A_G_A, Sigma_c
        )
        for _ in range(L):
            vs = torch.einsum("bde,be->bd", M, vs)

        Ov = torch.einsum("bde,bd->be", O, vs)
        loss_t = torch.einsum("be,be,e->", Ov, Ov, spec) / B + lamb * torch.mean(Gamma**2)
        losses.append(float(loss_t.detach().cpu()))

        # --- Gradient step ---
        Gamma_param = Gamma.detach().requires_grad_(True)
        A_G_A_g = (A.T @ Gamma_param @ A) / N
        vs_g = betas.detach().clone()
        M_g = torch.einsum("b,de->bde", ones_B, eye_d) - (1.0 / L) * torch.einsum(
            "de,bef->bdf", A_G_A_g, Sigma_c.detach()
        )
        for _ in range(L):
            vs_g = torch.einsum("bde,be->bd", M_g, vs_g)
        Ov_g = torch.einsum("bde,bd->be", O, vs_g)
        loss_g = torch.einsum("be,be,e->", Ov_g, Ov_g, spec) / B + lamb * torch.mean(Gamma_param**2)

        (grad_Gamma,) = torch.autograd.grad(loss_g, Gamma_param)
        Gamma = (Gamma - eta * grad_Gamma).detach()

        # --- Track diagonal eigenvalues ---
        eigs = torch.diag(A.T @ Gamma @ A / N)
        all_eigs.append(eigs.detach().cpu())

        sys.stdout.write(
            f"\r step = {t}  |  loss = {losses[-1]:.6f} | var_eigs = {float(eigs.var().cpu()):.6f}"
        )

    print()
    return losses, all_eigs


def ood_loss_fixed_covariance(
    spec: Tensor,
    w_star: Tensor,
    Lvals: list[int],
    thetas: Tensor,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> list[list[float]]:
    """Evaluate OOD loss of the optimal diagonal solution under covariance rotation.

    Sets Gamma to the analytically optimal diagonal solution for the
    fixed-covariance case:

        Gamma_opt = L * diag(1/spec_1, ..., 1/spec_d)

    Then evaluates the depth-L linear attention loss under a rotated covariance:

        Sigma_theta = O_theta @ diag(spec) @ O_theta^T

    where O_theta = exp(theta * A_rand) is a rotation parameterized by angle
    theta, with A_rand being a fixed random antisymmetric matrix scaled by
    1/(2*sqrt(d)).

    The residual for each (L, theta) pair is:

        v = (I - Gamma_opt @ Sigma_theta / L)^L @ (O_theta @ w_star)

    and the loss is v^T @ Sigma_theta @ v.

    At theta=0, the loss should be near zero (optimal in-distribution).
    As theta increases, the loss grows because the diagonal Gamma cannot
    compensate for the rotated covariance.

    Args:
        spec: Eigenvalues of the in-distribution covariance. Shape ``(d,)``.
        w_star: Target weight vector. Shape ``(d,)``.
        Lvals: List of depth values to evaluate.
        thetas: 1D tensor of rotation angles to sweep.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        Nested list ``all_losses[L_idx][theta_idx]`` of floats, where
        all_losses[i][j] is the loss for depth Lvals[i] at angle thetas[j].
    """
    device = torch.device(device)
    spec = spec.to(device=device, dtype=dtype)
    w_star = w_star.to(device=device, dtype=dtype)
    thetas = thetas.to(device=device, dtype=dtype)

    d = len(spec)
    eye_d = torch.eye(d, device=device, dtype=dtype)
    Sigma = torch.diag(spec)

    # Random antisymmetric matrix for rotation generator
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    A_rand = torch.randn(d, d, generator=gen, device=device, dtype=dtype)
    A_rand = 0.5 * (A_rand - A_rand.T) / math.sqrt(d)

    all_losses: list[list[float]] = []
    for L in Lvals:
        Gamma_opt = L * torch.diag(spec.pow(-1.0))
        losses_i: list[float] = []
        for theta in thetas:
            theta_val = float(theta.cpu())
            if theta_val == 0.0:
                O_theta = eye_d
            else:
                O_theta = torch.matrix_exp(theta * A_rand)

            Sigma_2 = O_theta @ Sigma @ O_theta.T
            Mat = eye_d - Gamma_opt @ Sigma_2 / L
            v = O_theta @ w_star

            for _ in range(L):
                v = Mat @ v

            loss_val = float(torch.sum(v * (Sigma_2 @ v)).cpu())
            losses_i.append(loss_val)
        all_losses.append(losses_i)

    return all_losses
