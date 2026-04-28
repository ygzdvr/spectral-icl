"""``ICLRegressionModel`` - abstract base for trainable ICL linear-regression models.

The base class owns the Theorem A aligned-channel construction shared by every
architecture in the thesis' §9 tier:

- Embedding: ``h_mu = W_x x_mu + y_mu * w_y`` for train tokens,
  ``h_mu = W_x x_mu`` for query tokens, with ``w_y`` tied to the readout.
- Rank-1 value projection: ``W_v = alpha_v * w_y w_y^T``, never materialized
  as a full matrix. In any layer, the value projection ``H W_v^T`` is computed
  as ``alpha_v * (H @ w_y).unsqueeze(-1) * w_y``.
- Assumption 1 alignment: ``W_x^T w_y = 0``, ``||w_y|| = 1``, plus (per
  subclass) ``W_q w_y = W_k w_y = 0``. :meth:`enforce_alignment` restores the
  invariants in place; subclasses extend it via ``super().enforce_alignment()``
  followed by :meth:`project_orthogonal_to_wy` on their own parameters.
- Depth loop: :meth:`forward` calls :meth:`mixer_forward` ``L`` times.
- Readout: ``f_kappa = w_y . h_{P + kappa}``.

Convention: inputs use the column-sample convention ``X in R^{B x D x P}``,
matching the thesis data generators.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ICLRegressionModel(nn.Module):
    """Abstract base for all trainable ICL regression models in the thesis.

    Subclasses must implement :meth:`mixer_forward` and typically override
    :meth:`enforce_alignment` to extend the base constraints with their own
    (``W_q w_y = W_k w_y = 0``).
    """

    def __init__(self, D: int, N: int, P: int, K: int, L: int) -> None:
        super().__init__()
        if D < 1 or N < 1 or P < 1 or K < 1 or L < 1:
            raise ValueError(
                f"D, N, P, K, L must be >= 1; got ({D}, {N}, {P}, {K}, {L})"
            )
        self.D = int(D)
        self.N = int(N)
        self.P = int(P)
        self.K = int(K)
        self.L = int(L)

        self.W_x = nn.Parameter(torch.empty(N, D))
        self.w_y = nn.Parameter(torch.empty(N))
        self.alpha_v = nn.Parameter(torch.tensor(1.0))

        self._init_base_parameters()

    # -- initialization ------------------------------------------------------

    def _init_base_parameters(self) -> None:
        nn.init.normal_(self.W_x, mean=0.0, std=1.0 / math.sqrt(self.D))
        nn.init.normal_(self.w_y, mean=0.0, std=1.0)
        with torch.no_grad():
            norm = self.w_y.data.norm()
            if float(norm) < 1e-12:
                raise RuntimeError("w_y initialized with near-zero norm")
            self.w_y.data.div_(norm)
            self.W_x.data.sub_(
                torch.outer(self.w_y.data, self.W_x.data.T @ self.w_y.data)
            )

    # -- alignment (Assumption 1) -------------------------------------------

    @torch.no_grad()
    def enforce_alignment(self) -> None:
        """Restore the base alignment invariants in place:

            ||w_y|| = 1     and     W_x^T w_y = 0.

        Subclasses override by calling ``super().enforce_alignment()`` first
        and then projecting their own (N, N) parameters via
        :meth:`project_orthogonal_to_wy`.
        """
        norm = self.w_y.data.norm()
        if float(norm) < 1e-12:
            raise RuntimeError("w_y has near-zero norm; cannot enforce alignment")
        self.w_y.data.div_(norm)
        self.W_x.data.sub_(
            torch.outer(self.w_y.data, self.W_x.data.T @ self.w_y.data)
        )

    @torch.no_grad()
    def project_orthogonal_to_wy(self, param: nn.Parameter) -> None:
        """In place, replace ``param.data`` with its projection satisfying
        ``param.data @ w_y = 0``.

        Expects ``param`` of shape ``(N, N)`` and assumes ``||w_y|| = 1``
        (callers invoke :meth:`enforce_alignment` first, which normalizes).
        """
        if tuple(param.shape) != (self.N, self.N):
            raise ValueError(
                f"project_orthogonal_to_wy expects (N, N) = ({self.N}, {self.N}); "
                f"got {tuple(param.shape)}"
            )
        wy = self.w_y.data
        param.data.sub_(torch.outer(param.data @ wy, wy))

    # -- forward stack -------------------------------------------------------

    def embed(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_query: torch.Tensor,
    ) -> torch.Tensor:
        """Build the initial residual stream ``H0`` of shape ``(B, P + K, N)``.

        ``X_train`` is column-sample ``(B, D, P)`` and ``X_query`` is
        ``(B, D, K)``. Train tokens carry ``W_x x_mu + y_mu * w_y``; query
        tokens carry ``W_x x_mu`` only.
        """
        if X_train.ndim != 3 or X_train.shape[-2] != self.D or X_train.shape[-1] != self.P:
            raise ValueError(
                f"X_train must have shape (B, D={self.D}, P={self.P}); "
                f"got {tuple(X_train.shape)}"
            )
        if X_query.ndim != 3 or X_query.shape[-2] != self.D or X_query.shape[-1] != self.K:
            raise ValueError(
                f"X_query must have shape (B, D={self.D}, K={self.K}); "
                f"got {tuple(X_query.shape)}"
            )
        if y_train.ndim != 2 or y_train.shape[-1] != self.P:
            raise ValueError(
                f"y_train must have shape (B, P={self.P}); got {tuple(y_train.shape)}"
            )

        H_tr = X_train.transpose(-1, -2) @ self.W_x.T + y_train.unsqueeze(-1) * self.w_y
        H_q = X_query.transpose(-1, -2) @ self.W_x.T
        return torch.cat([H_tr, H_q], dim=-2)

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        """Query readout ``f_kappa = w_y . h_{P + kappa}``; returns ``(B, K)``."""
        return (H[:, self.P :, :] * self.w_y).sum(dim=-1)

    def _value_projection(self, H: torch.Tensor) -> torch.Tensor:
        """Rank-1 value projection ``V = alpha_v * (H w_y) w_y^T`` of shape
        ``(B, T, N)``, computed without materializing ``W_v = alpha_v w_y w_y^T``.
        """
        proj = H @ self.w_y
        return self.alpha_v * proj.unsqueeze(-1) * self.w_y

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_query: torch.Tensor,
    ) -> torch.Tensor:
        H = self.embed(X_train, y_train, X_query)
        for ell in range(self.L):
            H = self.mixer_forward(H, ell)
        return self.readout(H)

    def mixer_forward(self, H: torch.Tensor, ell: int) -> torch.Tensor:
        """One residual-stream mixer layer; implemented by subclasses."""
        raise NotImplementedError

    def loss(self, f_pred: torch.Tensor, y_query: torch.Tensor) -> torch.Tensor:
        """Mean squared error ICL loss over ``(B, K)`` predictions."""
        if f_pred.shape != y_query.shape:
            raise ValueError(
                f"f_pred {tuple(f_pred.shape)} vs y_query {tuple(y_query.shape)} "
                "shape mismatch"
            )
        return ((f_pred - y_query) ** 2).mean()


__all__ = ["ICLRegressionModel"]
