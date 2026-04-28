"""``SpectralAttention`` - linear attention with a learnable circulant
train-train mask parameterized in the Fourier half-spectrum.

Identical to :class:`LinearAttention` except that the fixed GD-compatible
train-train mask ``M_GD[:P, :P] = -1`` is replaced by a learnable real
symmetric circulant ``S_TT`` with a spectral bottleneck ``r``:

    s_half in R^{r_half}      where r_half = min(r, P // 2 + 1)
    padded = [s_half, 0, ..., 0]               in R^{P // 2 + 1}
    first_col = irfft(padded, n=P)             in R^{P}
    S_TT[i, j] = first_col[(i - j) mod P].

The test-train block remains ``+1`` and the other two blocks remain ``0``, so
Assumption 1 of Theorem A (train-supported mask) holds unconditionally.

The entire forward pass is built from :func:`torch.fft.irfft` and advanced
indexing on a precomputed buffer. It deliberately does NOT call
``scripts.thesis.utils.fourier_ops`` (those routines are for the operator-level
tier and have complex-leakage assertions that aren't autograd-safe). The
operator-level fourier_ops are used only in the tests for cross-verification.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from scripts.thesis.architectures.models.base import ICLRegressionModel


class SpectralAttention(ICLRegressionModel):
    """Circulant-gated linear attention with a learnable ``r``-mode spectral
    filter in the training block of the mask.
    """

    def __init__(
        self,
        D: int,
        N: int,
        P: int,
        K: int,
        L: int,
        r: int,
        s_init: str = "zero",
    ) -> None:
        super().__init__(D, N, P, K, L)

        if not (1 <= int(r) <= P):
            raise ValueError(f"r must satisfy 1 <= r <= P; got r={r}, P={P}")
        self.r = int(r)
        full_half = P // 2 + 1
        self.r_half = min(self.r, full_half)

        self.W_q = nn.Parameter(torch.empty(N, N))
        self.W_k = nn.Parameter(torch.empty(N, N))
        nn.init.normal_(self.W_q, mean=0.0, std=1.0 / math.sqrt(N))
        nn.init.normal_(self.W_k, mean=0.0, std=1.0 / math.sqrt(N))

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

    # -- alignment -----------------------------------------------------------

    def enforce_alignment(self) -> None:
        super().enforce_alignment()
        self.project_orthogonal_to_wy(self.W_q)
        self.project_orthogonal_to_wy(self.W_k)

    # -- differentiable circulant ------------------------------------------

    def _build_S_TT(self) -> torch.Tensor:
        """Real symmetric circulant ``(P, P)`` from the learnable half-spectrum.

        Pure differentiable PyTorch: zero-pad via ``torch.cat``, invert via
        ``torch.fft.irfft``, build the full matrix by indexing
        ``first_col[(i - j) mod P]`` against the precomputed ``_circ_idx``
        buffer.
        """
        full_half_len = self.P // 2 + 1
        if self.r_half < full_half_len:
            pad = self.s_half.new_zeros(full_half_len - self.r_half)
            padded = torch.cat([self.s_half, pad])
        else:
            padded = self.s_half
        first_col = torch.fft.irfft(padded, n=self.P)
        return first_col[self._circ_idx]

    def _build_S_full(self) -> torch.Tensor:
        """Full ``(P + K, P + K)`` mask: ``[[S_TT, 0], [1, 0]]``.

        Built with ``torch.cat`` (not in-place assignment) so the autograd
        graph from ``s_half`` through ``S_TT`` survives into every downstream
        use of the mask.
        """
        S_TT = self._build_S_TT()
        zeros_TQ = S_TT.new_zeros(self.P, self.K)
        ones_QT = S_TT.new_ones(self.K, self.P)
        zeros_QQ = S_TT.new_zeros(self.K, self.K)
        top = torch.cat([S_TT, zeros_TQ], dim=1)
        bottom = torch.cat([ones_QT, zeros_QQ], dim=1)
        return torch.cat([top, bottom], dim=0)

    # -- forward -------------------------------------------------------------

    def mixer_forward(self, H: torch.Tensor, ell: int) -> torch.Tensor:
        P, L = self.P, self.L
        S = self._build_S_full()
        Q = H @ self.W_q.T
        K_proj = H @ self.W_k.T
        scores = Q @ K_proj.transpose(-1, -2) / P
        gated = S.unsqueeze(0) * scores
        V = self._value_projection(H)
        update = gated @ V / L
        return H + update


__all__ = ["SpectralAttention"]
