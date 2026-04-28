"""``LinearAttention`` - Bordelon-style baseline wrapped in ``ICLRegressionModel``.

One residual-stream layer runs the canonical masked linear-attention update:

    Q       = H @ W_q^T
    K_proj  = H @ W_k^T
    scores  = Q @ K_proj^T / P
    gated   = M_GD * scores
    V       = alpha_v * (H @ w_y).unsqueeze(-1) * w_y
    H_next  = H + (1 / L) * gated @ V

with ``M_GD`` the fixed GD-compatible signed mask

    M_GD[mu in train, nu in train] = -1   (residual descent)
    M_GD[mu in test,  nu in train] = +1   (query accumulation)
    M_GD[*, *]                     =  0   elsewhere.

The 1/(LP) scale is split as 1/P in the scores and 1/L in the residual update,
matching the Bordelon convention. The rank-1 value projection + Assumption 1
alignment make the full forward pass collapse exactly to the Theorem A
reduced-Gamma recursion with

    Gamma = alpha_v * W_x^T W_q^T W_k W_x.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from scripts.thesis.architectures.models.base import ICLRegressionModel


class LinearAttention(ICLRegressionModel):
    """Linear attention with GD-compatible mask, rank-1 value, tied readout."""

    def __init__(self, D: int, N: int, P: int, K: int, L: int) -> None:
        super().__init__(D, N, P, K, L)

        self.W_q = nn.Parameter(torch.empty(N, N))
        self.W_k = nn.Parameter(torch.empty(N, N))
        nn.init.normal_(self.W_q, mean=0.0, std=1.0 / math.sqrt(N))
        nn.init.normal_(self.W_k, mean=0.0, std=1.0 / math.sqrt(N))

        M = torch.zeros(P + K, P + K)
        M[:P, :P] = -1.0
        M[P:, :P] = +1.0
        self.register_buffer("M_GD", M)

        self.enforce_alignment()

    def enforce_alignment(self) -> None:
        super().enforce_alignment()
        self.project_orthogonal_to_wy(self.W_q)
        self.project_orthogonal_to_wy(self.W_k)

    def mixer_forward(self, H: torch.Tensor, ell: int) -> torch.Tensor:
        P, L = self.P, self.L
        Q = H @ self.W_q.T
        K_proj = H @ self.W_k.T
        scores = Q @ K_proj.transpose(-1, -2) / P
        gated = self.M_GD * scores
        V = self._value_projection(H)
        update = gated @ V / L
        return H + update


__all__ = ["LinearAttention"]
