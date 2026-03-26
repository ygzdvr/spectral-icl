"""Newton's method solver for the N-dependent ICL loss floor.

This module computes the minimum achievable loss for an in-context learner
with N effective parameters (context tokens) given a spectral decomposition
of the input covariance.

The key equation is the Stieltjes-transform constraint:

    sum_m  spec_m / (spec_m + z)  =  N

which implicitly defines the regularization parameter z as a function of N.
The loss floor is then:

    L(z) = z^2 * sum_m  spec_m * w_star_m^2 / (z + spec_m)^2

This is the ridge regression Bayes-optimal loss for a problem with spectrum
``spec`` and N training samples, expressed through the resolvent.

This appears in the theory of ICL scaling laws: as the context length N
increases, the achievable loss decreases following a power law determined
by the spectral decay of the input covariance.
"""
from __future__ import annotations

import torch


Tensor = torch.Tensor


def solve_n_final(
    spec: Tensor,
    w_star: Tensor,
    N: float,
) -> float:
    """Find the N-dependent loss floor via Newton's method on the resolvent equation.

    Solves for z > 0 satisfying the Stieltjes-transform constraint:

        f(z) = sum_m spec_m / (spec_m + z) - N = 0

    using damped Newton iterations (step size 0.1, 100 iterations). The
    derivative f'(z) = -sum_m spec_m / (spec_m + z)^2 is computed via autograd.

    Once z is found, the loss floor is computed as:

        loss = z^2 * sum_m spec_m * w_star_m^2 / (z + spec_m)^2

    This corresponds to the Bayes-optimal ridge regression risk for a linear
    model with spectrum ``spec``, target weights ``w_star``, and N effective
    training samples.

    Args:
        spec: Eigenvalues of the input covariance matrix. Shape ``(d,)``.
            Should be positive.
        w_star: Squared norms of the target projected onto each eigendirection.
            Shape ``(d,)``.
        N: Effective number of context tokens (need not be integer). Controls
            the resolution of the in-context learner.

    Returns:
        The scalar loss floor as a Python float.
    """
    eps = 1e-8
    z = torch.tensor(1e-2, device=spec.device, dtype=spec.dtype, requires_grad=True)

    for _ in range(100):
        eq_val = torch.sum(spec / (spec + z + eps)) - N
        (g,) = torch.autograd.grad(eq_val, z)
        with torch.no_grad():
            z = (z - 0.1 * eq_val / g).detach().requires_grad_(True)

    with torch.no_grad():
        z_val = z.detach()
        loss = (z_val + eps) ** 2 * torch.sum(spec * w_star**2 / (z_val + eps + spec) ** 2)

    return float(loss.cpu())
