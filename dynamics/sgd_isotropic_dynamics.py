"""SGD dynamics simulations for isotropic linear regression in-context learning.

This module simulates the stochastic gradient descent (SGD) dynamics of a
matrix M (or Gamma) that represents the effective attention operator in a
linear attention model performing in-context linear regression.

The key abstraction is that a depth-L linear attention transformer's ICL
behavior can be captured by a single d x d matrix M (or Gamma) that evolves
under SGD with mini-batch noise. The loss measures how far M (or I - Gamma)
is from the identity, which corresponds to perfect in-context regression.

Three simulation variants are provided:
  - ``simple_sgd_isotropic_dynamics``: Basic SGD with a single context sequence.
  - ``simple_sgd_rmt_isotropic_dynamics``: Random matrix theory (RMT) variant
    with separate train (K tokens) and test (P tokens) sequences.
  - ``simple_sgd_noisy_rmt_isotropic_dynamics``: RMT variant with additive
    label noise of magnitude sigma.

Two theoretical prediction functions match the simulations:
  - ``simple_sgd_isotropic_theory``: Closed-form loss trajectory for the basic
    SGD case.
  - ``simple_sgd_rmt_isotropic_theory``: Closed-form loss trajectory for the
    RMT variant.

A loss landscape visualization is also provided:
  - ``visualize_loss_landscape``: Computes the ICL loss as a function of a
    scalar gamma under the Marchenko-Pastur spectral distribution.
"""
from __future__ import annotations

import torch


Tensor = torch.Tensor


def simple_sgd_isotropic_dynamics(
    d: int,
    B: int,
    P: int,
    eta: float = 0.01,
    T: int = 100,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Simulate SGD dynamics of an error matrix M on isotropic linear regression.

    M starts as the identity I_d (representing maximum prediction error) and is
    updated at each step via a mini-batch SGD rule:

        M <- M - (eta / B) * sum_b (beta_b beta_b^T) @ M @ (X_b^T X_b / P)

    where beta_b are i.i.d. N(0, I_d) task vectors and X_b are i.i.d.
    N(0, I_d) context matrices of shape (P, d). The loss at step t is the
    normalized Frobenius norm:

        loss_t = (1/d) * ||M_t||_F^2

    Random seeds are deterministic: step t uses seed 2t for betas, 2t+1 for X.

    Args:
        d: Ambient dimension of the linear regression problem.
        B: Mini-batch size (number of tasks per SGD step).
        P: Context sequence length (number of in-context examples per task).
        eta: Learning rate.
        T: Number of SGD steps to simulate.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        Tensor of shape ``(T,)`` containing the normalized loss at each step.
    """
    device = torch.device(device)
    M = torch.eye(d, device=device, dtype=dtype)
    losses = torch.zeros(T, device=device, dtype=dtype)

    for t in range(T):
        gen_beta = torch.Generator(device=device)
        gen_beta.manual_seed(2 * t)
        gen_x = torch.Generator(device=device)
        gen_x.manual_seed(2 * t + 1)

        Betas = torch.randn(B, d, generator=gen_beta, device=device, dtype=dtype)
        Xs = torch.randn(B, P, d, generator=gen_x, device=device, dtype=dtype)

        losses[t] = (1.0 / d) * torch.sum(M**2)

        # Omegas: (B, d, d) = X^T X / P
        Omegas = torch.einsum("bpd,bpe->bde", Xs, Xs) / P

        # Beta_outer: (B, d, d)
        Beta_outer = torch.einsum("bi,bj->bij", Betas, Betas)

        # Update: M += -eta/B * einsum('bij,jk,bkl->il', Beta_outer, M, Omegas)
        M = M - (eta / B) * torch.einsum("bij,jk,bkl->il", Beta_outer, M, Omegas)

    return losses


def simple_sgd_rmt_isotropic_dynamics(
    d: int,
    B: int,
    K: int,
    P: int,
    eta: float = 0.01,
    T: int = 100,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Simulate SGD dynamics with separate train and test context sequences.

    This is the random matrix theory (RMT) variant that separates the context
    into K training tokens and P test tokens, producing distinct empirical
    covariance matrices. Gamma starts at zero and evolves toward the identity
    (the optimal solution for isotropic regression).

    The update rule is:

        M_eff[b] = I - Gamma @ Sigma_test[b]
        Gamma <- Gamma + (eta / B) * sum_b Omega_train[b] @ M_eff[b]
                                            @ (beta_b beta_b^T) @ Sigma_test[b]

    where:
      - Omega_train[b] = X_train[b]^T X_train[b] / K  (train empirical cov)
      - Sigma_test[b]  = X_test[b]^T X_test[b] / P    (test empirical cov)

    The loss at each step is:

        loss_t = (1/d) * ||I - Gamma_t||_F^2

    Random seeds: step t uses 3t for betas, 3t+1 for X_train, 3t+2 for X_test.

    Args:
        d: Ambient dimension of the linear regression problem.
        B: Mini-batch size (number of tasks per SGD step).
        K: Number of training context tokens (used to form Omega).
        P: Number of test context tokens (used to form Sigma).
        eta: Learning rate.
        T: Number of SGD steps to simulate.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        Tensor of shape ``(T,)`` containing the normalized loss at each step.
    """
    device = torch.device(device)
    Gamma = torch.zeros(d, d, device=device, dtype=dtype)
    eye = torch.eye(d, device=device, dtype=dtype)
    ones_B = torch.ones(B, device=device, dtype=dtype)
    losses = torch.zeros(T, device=device, dtype=dtype)

    for t in range(T):
        gen_beta = torch.Generator(device=device)
        gen_beta.manual_seed(3 * t)
        gen_x = torch.Generator(device=device)
        gen_x.manual_seed(3 * t + 1)
        gen_xte = torch.Generator(device=device)
        gen_xte.manual_seed(3 * t + 2)

        Betas = torch.randn(B, d, generator=gen_beta, device=device, dtype=dtype)
        Xs = torch.randn(B, K, d, generator=gen_x, device=device, dtype=dtype)
        Xtes = torch.randn(B, P, d, generator=gen_xte, device=device, dtype=dtype)

        losses[t] = (1.0 / d) * torch.sum((eye - Gamma) ** 2)

        Omegas = torch.einsum("bpd,bpe->bde", Xs, Xs) / K
        Sigmas = torch.einsum("bpd,bpe->bde", Xtes, Xtes) / P
        Beta_outer = torch.einsum("bi,bj->bij", Betas, Betas)

        # M_eff[b] = I - Gamma @ Sigmas[b]
        M_eff = torch.einsum("b,jk->bjk", ones_B, eye) - torch.einsum(
            "jk,bkl->bjl", Gamma, Sigmas
        )

        Gamma = Gamma + (eta / B) * torch.einsum(
            "bde,bef,bfg,bgh->dh", Omegas, M_eff, Beta_outer, Sigmas
        )

    return losses


def simple_sgd_rmt_isotropic_theory(
    tau: float,
    alpha: float,
    kappa: float,
    eta: float = 0.01,
    T: int = 100,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Closed-form theoretical prediction for RMT isotropic SGD dynamics.

    Computes the analytically-derived loss trajectory for the RMT variant of
    isotropic SGD dynamics. The theory tracks two quantities (a deterministic
    "signal" term and a stochastic "noise" term) that together yield the loss.

    The key parameters are:
      - tau = B (batch size, controls SGD noise magnitude)
      - alpha = P/d (test tokens-to-dimension ratio)
      - kappa = K/d (train tokens-to-dimension ratio)

    The per-step contraction rate is:

        a = 1 - 2*eta*(1+1/alpha) + eta^2*(1 + 1/tau*(1+1/kappa)*(1+1/alpha))*(1+1/alpha)

    And the noise accumulation rate is:

        b = eta^2 * (1 + 1/tau*(1+1/kappa)*(1+1/alpha)) / (1+alpha)

    The loss combines an exponentially decaying signal term and a noise floor:

        loss_t = C_t + 2/(1+1/alpha) * 1/(1+alpha) * (1-eta*(1+1/alpha))^t + 1/(1+alpha)^2

    where C_t = alpha^2/(1+alpha)^2 * a^t + (1-a^t)/(1-a) * b.

    Args:
        tau: Effective batch size parameter (= B in simulations, controls
            variance of SGD noise).
        alpha: Ratio P/d (test sequence length to dimension).
        kappa: Ratio K/d (train sequence length to dimension).
        eta: Learning rate.
        T: Number of SGD steps.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        Tensor of shape ``(T,)`` with the predicted loss at each step.
    """
    device = torch.device(device)
    tvals = torch.linspace(1, T, T, device=device, dtype=dtype)

    inv_alpha = 1.0 / alpha
    ratio = 1.0 + inv_alpha

    a = (
        1.0
        - 2.0 * eta * ratio
        + eta**2 * (1.0 + 1.0 / tau * (1.0 + 1.0 / kappa) * ratio) * ratio
    )
    b = eta**2 * (1.0 + 1.0 / tau * (1.0 + 1.0 / kappa) * ratio) / (1.0 + alpha)

    a_t = torch.exp(tvals * torch.log(torch.tensor(a, device=device, dtype=dtype)))

    C = alpha**2 / (1.0 + alpha) ** 2 * a_t + (1.0 - a_t) / (1.0 - a) * b

    losses = (
        C
        + 2.0 / ratio * 1.0 / (1.0 + alpha)
        * torch.exp(
            tvals
            * torch.log(torch.tensor(1.0 - eta * ratio, device=device, dtype=dtype))
        )
        + 1.0 / (1.0 + alpha) ** 2
    )

    return losses


def simple_sgd_noisy_rmt_isotropic_dynamics(
    d: int,
    B: int,
    K: int,
    P: int,
    sigma: float = 0.0,
    eta: float = 0.01,
    T: int = 100,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Simulate noisy RMT SGD dynamics with label noise.

    Identical to ``simple_sgd_rmt_isotropic_dynamics`` but includes additive
    Gaussian label noise of standard deviation ``sigma``. The noise term
    b = sigma * X_test^T eps / P * sqrt(d) is computed but currently does NOT
    affect the Gamma update (the update rule is the same as the noiseless
    version). This function is structured to allow easy incorporation of the
    noise term in future experiments.

    The loss at each step is:

        loss_t = (1/d) * ||I - Gamma_t||_F^2

    Random seeds: step t uses 4t for betas, 4t+1 for X_train, 4t+2 for X_test,
    4t+3 for noise eps.

    Args:
        d: Ambient dimension of the linear regression problem.
        B: Mini-batch size (number of tasks per SGD step).
        K: Number of training context tokens.
        P: Number of test context tokens.
        sigma: Standard deviation of additive label noise. The noise vector
            is computed but does not currently enter the Gamma update.
        eta: Learning rate.
        T: Number of SGD steps to simulate.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        Tensor of shape ``(T,)`` containing the normalized loss at each step.
    """
    device = torch.device(device)
    import math

    Gamma = torch.zeros(d, d, device=device, dtype=dtype)
    eye = torch.eye(d, device=device, dtype=dtype)
    ones_B = torch.ones(B, device=device, dtype=dtype)
    losses = torch.zeros(T, device=device, dtype=dtype)

    for t in range(T):
        gen_beta = torch.Generator(device=device)
        gen_beta.manual_seed(4 * t)
        gen_x = torch.Generator(device=device)
        gen_x.manual_seed(4 * t + 1)
        gen_xte = torch.Generator(device=device)
        gen_xte.manual_seed(4 * t + 2)
        gen_eps = torch.Generator(device=device)
        gen_eps.manual_seed(4 * t + 3)

        Betas = torch.randn(B, d, generator=gen_beta, device=device, dtype=dtype)
        Xs = torch.randn(B, K, d, generator=gen_x, device=device, dtype=dtype)
        Xtes = torch.randn(B, P, d, generator=gen_xte, device=device, dtype=dtype)
        eps = torch.randn(B, P, generator=gen_eps, device=device, dtype=dtype)

        losses[t] = (1.0 / d) * torch.sum((eye - Gamma) ** 2)

        # noise term (not used in Gamma update in the notebook, but computed)
        # b = sigma * einsum('bpd,bp->bd', Xtes, eps) / P * sqrt(d)
        _ = sigma * torch.einsum("bpd,bp->bd", Xtes, eps) / P * math.sqrt(d)

        Omegas = torch.einsum("bpd,bpe->bde", Xs, Xs) / K
        Sigmas = torch.einsum("bpd,bpe->bde", Xtes, Xtes) / P
        Beta_outer = torch.einsum("bi,bj->bij", Betas, Betas)

        M_eff = torch.einsum("b,jk->bjk", ones_B, eye) - torch.einsum(
            "jk,bkl->bjl", Gamma, Sigmas
        )

        Gamma = Gamma + (eta / B) * torch.einsum(
            "bde,bef,bfg,bgh->dh", Omegas, M_eff, Beta_outer, Sigmas
        )

    return losses


def visualize_loss_landscape(
    gamma: float | Tensor,
    lamb_grid: Tensor,
    alpha: float,
    L: int,
    sigma: float = 0.0,
) -> Tensor:
    """Compute the ICL loss landscape as a function of scalar gamma.

    Evaluates the depth-L linear attention ICL loss at a given gamma value,
    integrating over eigenvalues distributed according to the Marchenko-Pastur
    (MP) law with aspect ratio alpha = P/d.

    The Marchenko-Pastur density is:

        rho(lambda) = alpha / (2 * pi * lambda) * sqrt((lambda+ - lambda)(lambda - lambda-))

    for lambda in [lambda-, lambda+] where lambda+/- = (1 +/- 1/sqrt(alpha))^2.

    The loss integrand is:

        loss = integral rho(lambda) * |1 - gamma*lambda/L|^{2L} d(lambda)

    When sigma > 0, a noise term is added:

        + sigma^2/alpha * integral rho(lambda) * (1 - (1-gamma*lambda/L)^L)^2 / lambda d(lambda)

    When alpha < 1, a point mass (1 - alpha) at lambda=0 contributes a constant
    offset of (1 - alpha) to the loss.

    Args:
        gamma: Scalar attention strength parameter (or a tensor for batched
            evaluation).
        lamb_grid: 1D tensor of eigenvalue grid points for numerical integration.
        alpha: Aspect ratio P/d controlling the Marchenko-Pastur distribution.
        L: Number of attention layers (depth).
        sigma: Label noise standard deviation (default 0).

    Returns:
        Scalar tensor with the numerically integrated loss value.
    """
    import math

    sqrt_alpha = math.sqrt(alpha)
    mp_min = (1.0 - 1.0 / sqrt_alpha) ** 2
    mp_max = (1.0 + 1.0 / sqrt_alpha) ** 2

    diff = (mp_max - lamb_grid) * (lamb_grid - mp_min)
    diff_pos = diff * (diff > 0.0).to(lamb_grid.dtype)
    rho = alpha / (2.0 * math.pi * lamb_grid) * torch.sqrt(diff_pos)

    dlamb = (lamb_grid.max() - lamb_grid.min()) / len(lamb_grid)

    loss_lamb = torch.exp(2 * L * torch.log(torch.abs(1.0 - gamma * lamb_grid / L)))

    loss = torch.sum(rho * loss_lamb * dlamb)

    loss = loss + sigma**2 / alpha * torch.sum(
        dlamb * rho * (1.0 - (1.0 - gamma * lamb_grid / L) ** L) ** 2 / lamb_grid
    )

    if alpha < 1.0:
        loss = loss + (1.0 - alpha)

    return loss


def simple_sgd_isotropic_theory(
    tau: float,
    kappa: float,
    eta: float = 0.01,
    T: int = 100,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Closed-form theoretical prediction for basic isotropic SGD dynamics.

    Computes the analytically-derived loss trajectory for the simple (non-RMT)
    isotropic SGD dynamics where a single context sequence of length P is used.

    The per-step contraction factor combines the deterministic shrinkage
    (1 - eta)^2 with an additive SGD noise term:

        rate = (1 - eta)^2 + eta^2 / tau * (1 + 1/kappa)

    so that:

        loss(t) = rate^t

    This is the geometric decay expected when M starts at I_d and each step
    multiplies by a random matrix whose expected squared norm is ``rate``.

    Args:
        tau: Effective batch size parameter (= B in simulations).
        kappa: Ratio P/d (context length to dimension).
        eta: Learning rate.
        T: Number of SGD steps.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        Tensor of shape ``(T,)`` with the predicted loss at each step.
    """
    device = torch.device(device)
    tvals = torch.linspace(1, T, T, device=device, dtype=dtype)
    log_rate = torch.log(
        torch.tensor((1 - eta) ** 2 + eta**2 / tau * (1 + 1 / kappa), device=device, dtype=dtype)
    )
    losses = torch.exp(tvals * log_rate)
    return losses
