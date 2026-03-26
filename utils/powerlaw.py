"""Power-law problem builders shared across experiment scripts.

The research scripts repeatedly construct two objects:
1. ``spec``   : a power-law spectrum controlling input covariance eigenvalues.
2. ``w_star`` : a teacher vector aligned with that spectrum through ``beta``.

This module keeps that construction consistent so all scripts use the same
conventions for normalization and exponent parameterization.
"""

from __future__ import annotations

import torch


def make_powerlaw_spec_and_wstar(
    m: int,
    alpha: float,
    beta: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    normalize_w_star: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a 1D power-law spectrum and matching teacher vector.

    Mathematical form used in this codebase:
        - coordinates: ``k = 1, 2, ..., m``
        - spectrum: ``spec_k = k^{-alpha}``
        - teacher:  ``w*_k = k^{-(alpha * beta + 1 - alpha)/2}``

    Optional normalization:
        If ``normalize_w_star=True``, we rescale ``w_star`` so that
        ``sum_k w*_k^2 * spec_k = 1``. This is the common normalization used in
        most training/evaluation scripts, while a few analytical scripts keep
        the unnormalized form for fidelity to notebook derivations.

    Args:
        m: Ambient dimension / number of coordinates.
        alpha: Spectrum decay exponent.
        beta: Teacher-shape exponent parameter used by the notebooks.
        device: Torch device for output tensors.
        dtype: Torch floating dtype for output tensors.
        normalize_w_star: Whether to rescale ``w_star`` by ``spec``-weighted norm.

    Returns:
        A tuple ``(spec, w_star)`` where each tensor has shape ``[m]``.
    """
    # Coordinates are 1-indexed to match the notebook formulas directly.
    coords = torch.linspace(1, m, m, device=device, dtype=dtype)

    # Power-law covariance spectrum.
    spec = coords.pow(-alpha)

    # Teacher weights before optional normalization.
    w_star = coords.pow(-(alpha * beta + 1 - alpha) * 0.5)

    if normalize_w_star:
        # Normalize under the spectrum-weighted quadratic form:
        #   ||w_star||_spec^2 = sum_k w_star_k^2 * spec_k
        w_star = w_star / torch.sqrt(torch.sum(w_star**2 * spec))

    return spec, w_star
