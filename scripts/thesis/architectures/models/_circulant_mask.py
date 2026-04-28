"""Shared differentiable circulant mask construction.

Identical to the inline construction used by
:class:`scripts.thesis.architectures.models.SpectralAttention`; see that file
for the design rationale (``torch.fft.irfft`` + pre-registered ``circ_idx``
gather buffer + ``torch.cat`` assembly, kept autograd-safe so gradients flow
from loss back through ``s_half``). Factored out here so
:class:`SpectralFilter` and :class:`AdaptiveFirstHybrid` can share one
implementation without touching the already-tested :class:`SpectralAttention`.
"""

from __future__ import annotations

import torch


def build_S_TT(
    s_half: torch.Tensor,
    r_half: int,
    P: int,
    circ_idx: torch.Tensor,
) -> torch.Tensor:
    """Differentiable ``P x P`` real symmetric circulant from the learnable
    half-spectrum ``s_half`` of length ``r_half``.

    Pads ``s_half`` to length ``P // 2 + 1`` with zeros (via ``torch.cat``),
    inverts to a real first-column signal via ``torch.fft.irfft``, then builds
    the full circulant by gathering ``first_col[(i - j) mod P]`` from the
    pre-registered ``circ_idx`` index buffer. Every operation is autograd-
    traceable end-to-end, so backward populates ``s_half.grad``.
    """
    full_half_len = P // 2 + 1
    if r_half < full_half_len:
        pad = s_half.new_zeros(full_half_len - r_half)
        padded = torch.cat([s_half, pad])
    else:
        padded = s_half
    first_col = torch.fft.irfft(padded, n=P)
    return first_col[circ_idx]


def build_S_full(
    S_TT: torch.Tensor, P: int, K: int, negate_qt: bool = False,
) -> torch.Tensor:
    """Assemble the full ``(P + K, P + K)`` mask ``[[S_TT, 0], [S_QT, 0]]``.

    Uses ``torch.cat`` (not in-place assignment) so the autograd path from
    ``s_half`` through ``S_TT`` (and through ``S_QT`` when ``negate_qt=True``)
    survives into every downstream use of the assembled mask.

    Parameters
    ----------
    negate_qt
        If ``False`` (default), the query-train block is the constant
        all-ones matrix ``ones(K, P)``. This is the
        :class:`SpectralAttention` convention.

        If ``True``, the query-train block is ``-S_TT[:K, :]`` -- i.e. the
        negated first ``K`` rows of ``S_TT``. At the GD-compatible init
        ``s_half = [-P, 0, ...]`` we have ``S_TT = -ones``, so ``S_QT =
        +ones`` and the assembled mask reproduces the GD-compatible
        ``M_GD`` exactly. Away from that init, higher-frequency modes of
        ``s_half`` now shape ``S_QT`` too, giving different query
        positions different linear combinations of ``V_train`` -- this is
        what lets the spectral bottleneck ``r`` actually bind on the query
        readout (without it, ``ones_QT`` is a DC projector and
        ``s_half[k >= 1]`` gets zero gradient from the loss). Required
        ``K <= P``.
    """
    if negate_qt:
        if K > P:
            raise ValueError(
                f"negate_qt=True requires K <= P (S_QT = -S_TT[:K, :]); "
                f"got K={K}, P={P}"
            )
        S_QT = -S_TT[:K, :]
    else:
        S_QT = S_TT.new_ones(K, P)
    zeros_TQ = S_TT.new_zeros(P, K)
    zeros_QQ = S_TT.new_zeros(K, K)
    top = torch.cat([S_TT, zeros_TQ], dim=1)
    bottom = torch.cat([S_QT, zeros_QQ], dim=1)
    return torch.cat([top, bottom], dim=0)


__all__ = ["build_S_TT", "build_S_full"]
