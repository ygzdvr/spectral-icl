"""``AdaptiveFirstHybrid`` - Section 9.3 thesis architecture.

Sequential composition of ``L_A`` attention layers followed by ``L_S``
spectral-filter layers. The attention stage provides the content-dependent
routing that neither :class:`SpectralFilter` nor :class:`STUNative` can do
on their own; the spectral stage provides the bottlenecked, theorem-B-style
filter whose spectral rank ``r`` **directly** controls the effective
operator (unlike :class:`SpectralAttention` where ``S_TT`` is Hadamard-
multiplied into a fully-learnable kernel and the bottleneck is bypassed).

Both stages share ``W_x``, ``w_y``, ``alpha_v`` from the base class (one
embedding, one readout, one value-channel gain). Stage-specific behavior
comes from the stage-specific operators: ``(W_q_A, W_k_A)`` for attention
and the learnable half-spectrum ``s_half`` for the spectral circulant.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from scripts.thesis.architectures.models._circulant_mask import (
    build_S_full,
    build_S_TT,
)
from scripts.thesis.architectures.models.base import ICLRegressionModel


class AdaptiveFirstHybrid(ICLRegressionModel):
    """``L_A`` attention layers, then ``L_S`` spectral-filter layers."""

    def __init__(
        self,
        D: int,
        N: int,
        P: int,
        K: int,
        L_A: int,
        L_S: int,
        r: int,
        s_init: str = "zero",
    ) -> None:
        if L_A < 0 or L_S < 0:
            raise ValueError(f"L_A, L_S must be >= 0; got ({L_A}, {L_S})")
        if L_A + L_S < 1:
            raise ValueError(f"L_A + L_S must be >= 1; got {L_A + L_S}")

        # Base class needs L >= 1; store the total depth there for bookkeeping.
        super().__init__(D, N, P, K, L_A + L_S)
        self.L_A = int(L_A)
        self.L_S = int(L_S)

        if not (1 <= int(r) <= P):
            raise ValueError(f"r must satisfy 1 <= r <= P; got r={r}, P={P}")
        self.r = int(r)
        full_half = P // 2 + 1
        self.r_half = min(self.r, full_half)

        # Attention-stage parameters.
        self.W_q_A = nn.Parameter(torch.empty(N, N))
        self.W_k_A = nn.Parameter(torch.empty(N, N))
        nn.init.normal_(self.W_q_A, mean=0.0, std=1.0 / math.sqrt(N))
        nn.init.normal_(self.W_k_A, mean=0.0, std=1.0 / math.sqrt(N))

        # Spectral-stage parameters.
        self.s_half = nn.Parameter(torch.empty(self.r_half))
        if s_init == "zero":
            nn.init.zeros_(self.s_half)
        elif s_init == "gd":
            with torch.no_grad():
                self.s_half.zero_()
                self.s_half[0] = -float(P)
        else:
            raise ValueError(f"s_init must be 'zero' or 'gd'; got {s_init!r}")

        # Fixed GD-compatible attention mask (same as LinearAttention).
        M = torch.zeros(P + K, P + K)
        M[:P, :P] = -1.0
        M[P:, :P] = +1.0
        self.register_buffer("M_GD", M)

        # Circulant gather index buffer (same as SpectralAttention).
        row_idx = torch.arange(P).unsqueeze(1).expand(P, P)
        col_idx = torch.arange(P).unsqueeze(0).expand(P, P)
        self.register_buffer("_circ_idx", ((row_idx - col_idx) % P).contiguous())

        self.enforce_alignment()

    # -- alignment -----------------------------------------------------------

    def enforce_alignment(self) -> None:
        super().enforce_alignment()
        self.project_orthogonal_to_wy(self.W_q_A)
        self.project_orthogonal_to_wy(self.W_k_A)

    # -- differentiable S_TT / S_full (shared helper) ------------------------

    def _build_S_TT(self) -> torch.Tensor:
        return build_S_TT(self.s_half, self.r_half, self.P, self._circ_idx)

    def _build_S_full(self) -> torch.Tensor:
        # ``negate_qt=True`` for the same reason as :class:`SpectralFilter`:
        # the query-train block is ``-S_TT[:K, :]`` so non-DC modes of
        # ``s_half`` propagate to the query readout and the spectral
        # bottleneck ``r`` binds. GD init still reproduces the
        # GD-compatible ``M_GD`` exactly.
        return build_S_full(self._build_S_TT(), self.P, self.K, negate_qt=True)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_query: torch.Tensor,
    ) -> torch.Tensor:
        H = self.embed(X_train, y_train, X_query)

        # Stage 1: attention (content-dependent routing via GD-compatible mask).
        if self.L_A > 0:
            inv_LA = 1.0 / float(self.L_A)
            for _ in range(self.L_A):
                Q = H @ self.W_q_A.T
                K_proj = H @ self.W_k_A.T
                scores = Q @ K_proj.transpose(-1, -2) / self.P
                gated = self.M_GD * scores
                V = self._value_projection(H)
                H = H + inv_LA * (gated @ V)

        # Stage 2: spectral filter (S applied directly to V, no QK gating).
        if self.L_S > 0:
            S = self._build_S_full()                       # (T, T)
            inv_LSP = 1.0 / (float(self.L_S) * float(self.P))
            for _ in range(self.L_S):
                V = self._value_projection(H)
                H = H + inv_LSP * torch.einsum("ij,bjn->bin", S, V)

        return self.readout(H)

    def mixer_forward(self, H: torch.Tensor, ell: int) -> torch.Tensor:
        raise NotImplementedError(
            "AdaptiveFirstHybrid overrides forward directly; mixer_forward "
            "is not used."
        )


__all__ = ["AdaptiveFirstHybrid"]
