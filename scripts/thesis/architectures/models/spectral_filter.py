"""``SpectralFilter`` - pure spectral filter on the value-projected state.

Applies a learnable circulant ``S_TT`` directly to the rank-1 value
projection ``V = alpha_v (H w_y) w_y^T``, **without** the QK bilinear form
that :class:`SpectralAttention` uses. The effective sample-space operator at
train positions is therefore ``(alpha_v / P) S_TT`` -- the spectral
bottleneck ``r`` on ``S_TT`` binds **directly** on this operator. There is
no Hadamard-with-QK-kernel mixing that would let Gamma act as a bypass.

Why SpectralFilter alone cannot do meaningful ICL: the query-train block of
the mask is the all-ones block (copied from :class:`SpectralAttention` for
consistency). So every query position accumulates the same scaled sum of
``V_train``, independent of its own feature content. Without a
content-dependent query-to-train routing mechanism (which attention provides
via ``Q K^T``), queries cannot distinguish themselves. SpectralFilter is
therefore the spectral half of the :class:`AdaptiveFirstHybrid` design,
meant to be **stacked after** an attention stage that resolves the
content-dependent routing.

The differentiable circulant construction (``torch.fft.irfft`` + gather
buffer + ``torch.cat``) is factored into
:mod:`scripts.thesis.architectures.models._circulant_mask` and shared with
:class:`AdaptiveFirstHybrid`; :class:`SpectralAttention` keeps its own
inline copy to avoid touching its already-tested implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.thesis.architectures.models._circulant_mask import (
    build_S_full,
    build_S_TT,
)
from scripts.thesis.architectures.models.base import ICLRegressionModel


class SpectralFilter(ICLRegressionModel):
    """Pure spectral filter applied directly to ``V`` (no QK gating)."""

    def __init__(
        self, D: int, N: int, P: int, K: int, L: int, r: int,
        s_init: str = "zero",
    ) -> None:
        super().__init__(D, N, P, K, L)

        if not (1 <= int(r) <= P):
            raise ValueError(f"r must satisfy 1 <= r <= P; got r={r}, P={P}")
        self.r = int(r)
        full_half = P // 2 + 1
        self.r_half = min(self.r, full_half)

        self.s_half = nn.Parameter(torch.empty(self.r_half))
        if s_init == "zero":
            nn.init.zeros_(self.s_half)
        elif s_init == "gd":
            with torch.no_grad():
                self.s_half.zero_()
                self.s_half[0] = -float(P)
        else:
            raise ValueError(f"s_init must be 'zero' or 'gd'; got {s_init!r}")

        row_idx = torch.arange(P).unsqueeze(1).expand(P, P)
        col_idx = torch.arange(P).unsqueeze(0).expand(P, P)
        self.register_buffer("_circ_idx", ((row_idx - col_idx) % P).contiguous())

        self.enforce_alignment()

    def enforce_alignment(self) -> None:
        # No ``W_q`` / ``W_k`` in SpectralFilter; base class handles w_y / W_x.
        super().enforce_alignment()

    def _build_S_TT(self) -> torch.Tensor:
        return build_S_TT(self.s_half, self.r_half, self.P, self._circ_idx)

    def _build_S_full(self) -> torch.Tensor:
        # ``negate_qt=True``: the query-train block is ``-S_TT[:K, :]`` rather
        # than the content-independent ``ones_QT``. This gives each query
        # position a distinct linear combination of ``V_train`` (different
        # rows of ``-S_TT``), so higher-frequency ``s_half[k >= 1]`` modes
        # actually propagate to the query readout and the spectral bottleneck
        # ``r`` binds directly on the effective operator. At the GD init
        # ``s_half = [-P, 0, ...]`` the assembled mask still reproduces the
        # GD-compatible ``M_GD`` exactly.
        return build_S_full(self._build_S_TT(), self.P, self.K, negate_qt=True)

    def mixer_forward(self, H: torch.Tensor, ell: int) -> torch.Tensor:
        V = self._value_projection(H)                      # (B, T, N)
        S = self._build_S_full()                            # (T, T)
        inv_LP = 1.0 / (float(self.L) * float(self.P))
        update = inv_LP * torch.einsum("ij,bjn->bin", S, V)
        return H + update


__all__ = ["SpectralFilter"]
