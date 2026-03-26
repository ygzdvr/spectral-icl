"""Analysis helpers reused by scaling/landscape scripts."""

from __future__ import annotations

import numpy as np
import torch


def compute_loss_inf_depth(
    spec: torch.Tensor,
    w_star: torch.Tensor,
    n: int,
    n_samples: int = 100,
) -> float:
    """Estimate the ``L -> infinity`` loss floor via random QR projections.

    Args:
        spec: Spectrum vector (shape ``[d]``).
        w_star: Teacher vector (shape ``[d]``).
        n: Effective retained rank in the projection mask.
        n_samples: Number of random orthogonal draws to average.

    Returns:
        Monte Carlo estimate of the asymptotic loss floor.
    """
    device = spec.device
    dtype = spec.dtype
    d = len(spec)

    a_vec = torch.zeros(d, device=device, dtype=dtype)
    a_vec[: min(n, d)] = 1.0

    z_small = 1e-6
    eye_d = torch.eye(d, device=device, dtype=dtype)
    all_loss: list[float] = []

    for i in range(n_samples):
        gen = torch.Generator(device=device)
        gen.manual_seed(i)
        r = torch.randn(d, d, generator=gen, device=device, dtype=dtype)
        q = torch.linalg.qr(r).Q

        mat = q @ torch.diag(a_vec) @ q.T @ torch.diag(spec)
        v_final = z_small * torch.linalg.solve(mat + z_small * eye_d, w_star)
        all_loss.append(float(torch.sum(v_final**2 * spec).cpu()))

    return float(np.mean(all_loss))


def loss_landscape(
    w: torch.Tensor,
    spec: torch.Tensor,
    w_star: torch.Tensor,
    l: int = 4,
    beta0: float = 0.5,
    lamb: float = 0.0,
) -> torch.Tensor:
    """Evaluate closed-form loss values over a 1D weight grid.

    Args:
        w: Candidate scalar weight grid (shape ``[W]``).
        spec: Power-law spectrum vector (shape ``[M]``).
        w_star: Teacher weights (shape ``[M]``).
        l: Effective depth parameter in the formula.
        beta0: Base update scale factor.
        lamb: Optional quadratic regularization coefficient.

    Returns:
        Per-grid loss values (shape ``[W]``).
    """
    decay = (1.0 - beta0 / l * torch.outer(w.pow(7), spec)) ** (2 * l)
    return torch.einsum("wm,m->w", decay, spec * w_star**2) + lamb * w**2
