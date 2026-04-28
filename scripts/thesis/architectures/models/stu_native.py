"""``STUNative`` - Hazan-style Spectral Transform Unit for ICL regression.

STUNative is a standalone :class:`torch.nn.Module`, **not** a subclass of
:class:`ICLRegressionModel`: Theorem A's alignment assumptions do not apply to
this architecture. Training through :func:`train_icl_online` is supported via
three contract methods (plus the inherited ``parameters``):

- ``forward(X_train, y_train, X_query) -> (B, K)`` query predictions.
- ``loss(f_pred, y_query)`` scalar MSE.
- ``enforce_alignment()`` no-op -- the model has no alignment invariants, but
  the method must exist because the training loop calls it after every step.

Architecture (one sandwich block = one depth layer L):

    h -> RMSNorm -> MLP_in  -> + residual
      -> RMSNorm -> STU_fil -> + residual
      -> RMSNorm -> MLP_out -> + residual

where ``STU_fil`` is a spectral filter on the token (time) axis parameterized
in the top-r Hankel eigenbasis (``basis="hankel"`` default) or in the real
part of the first r DFT modes (``basis="dft"``). Two filter modes:

- ``use_approx=True`` (default): ``filters = phi @ filter_proj``, then FFT
  convolution. Small parameter count.
- ``use_approx=False``: apply each of the r basis filters to a branch
  projection and mix via a learnable ``(r, branch_dim, branch_dim)`` tensor.

The **spectral bottleneck r binds directly on the token-mixing operator**,
unlike :class:`SpectralAttention` where the spectral filter is
Hadamard-multiplied against a fully-learnable QK kernel (the Hadamard
becomes circular convolution in Fourier, softening the bottleneck).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral basis helpers
# ---------------------------------------------------------------------------


def nearest_power_of_two(n: int) -> int:
    """Smallest power of two ``>= n``. Returns 1 for ``n <= 1``."""
    if n <= 1:
        return 1
    return 1 << (math.ceil(math.log2(n)))


def get_hankel(T: int) -> torch.Tensor:
    """Return the ``(T, T)`` float64 Hankel matrix

        H[i, j] = 2 / ((i + j)^3 - (i + j)),     i, j in {1, 2, ..., T}.

    Note indices start at 1. The matrix is symmetric positive-definite.
    """
    if T < 1:
        raise ValueError(f"T must be positive; got {T}")
    entries = torch.arange(1, T + 1, dtype=torch.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    return 2.0 / (i_plus_j ** 3 - i_plus_j)


def get_spectral_filters(T: int, r: int) -> torch.Tensor:
    """Top-``r`` Hankel eigenbasis with fourth-root eigenvalue weighting.

    Returns a ``(T, r)`` float64 tensor

        phi = Phi_r * clamp(sigma_r, min=1e-8) ** 0.25,

    where ``Phi_r, sigma_r`` are the top-``r`` eigenvectors/eigenvalues of
    :func:`get_hankel` (ascending from :func:`torch.linalg.eigh`, sliced with
    ``[-r:]``). The fourth-root weighting follows the STU convention.
    """
    if r < 1 or r > T:
        raise ValueError(f"r must satisfy 1 <= r <= T; got r={r}, T={T}")
    H = get_hankel(T)
    sigma, Phi = torch.linalg.eigh(H)
    sigma_r = sigma[-r:]
    Phi_r = Phi[:, -r:]
    return (Phi_r * torch.clamp(sigma_r, min=1e-8).pow(0.25)).contiguous()


def get_dft_filters(T: int, r: int) -> torch.Tensor:
    """Real part of the first ``r`` columns of the unitary DFT matrix,
    returned as a ``(T, r)`` float64 tensor.
    """
    if r < 1 or r > T:
        raise ValueError(f"r must satisfy 1 <= r <= T; got r={r}, T={T}")
    k = torch.arange(T, dtype=torch.float64).view(-1, 1)
    j = torch.arange(T, dtype=torch.float64).view(1, -1)
    phase = -2.0 * math.pi * k * j / T
    F_re = torch.cos(phase) / math.sqrt(T)
    return F_re[:, :r].contiguous()


def fft_convolve_1d(z: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """Batched FFT convolution along the time axis.

    ``z`` has shape ``(B, T, C)``, ``filters`` has shape ``(T, C)``. Returns a
    ``(B, T, C)`` tensor where the ``c``-th output channel is the causal
    convolution of ``z[..., c]`` against ``filters[:, c]`` (truncated to the
    first ``T`` output taps).
    """
    T = z.shape[1]
    fft_size = nearest_power_of_two(2 * T - 1)
    z_f = torch.fft.rfft(z.float(), n=fft_size, dim=1)
    f_f = torch.fft.rfft(filters.float(), n=fft_size, dim=0)
    y_f = z_f * f_f.unsqueeze(0)
    return torch.fft.irfft(y_f, n=fft_size, dim=1)[:, :T, :]


def fft_convolve_basis(z: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Apply each of the ``r`` basis filters to ``z``.

    ``z`` has shape ``(B, T, C)``, ``phi`` has shape ``(T, r)``. Returns a
    ``(B, T, r, C)`` tensor whose ``(b, t, k, c)`` entry is the ``t``-th sample
    of the convolution of ``z[b, :, c]`` with ``phi[:, k]``.
    """
    T = z.shape[1]
    fft_size = nearest_power_of_two(2 * T - 1)
    z_f = torch.fft.rfft(z.float(), n=fft_size, dim=1)
    phi_f = torch.fft.rfft(phi.float(), n=fft_size, dim=0)
    y_f = z_f[:, :, None, :] * phi_f[None, :, :, None]
    return torch.fft.irfft(y_f, n=fft_size, dim=1)[:, :T, :, :]


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square normalization without learnable affine parameters."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms


class MLP(nn.Module):
    """Two-layer bias-free MLP: ``Linear(d, hidden) -> GELU -> Linear(hidden, d)``."""

    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)
        nn.init.normal_(self.fc1.weight, std=1.0 / math.sqrt(d))
        nn.init.normal_(self.fc2.weight, std=1.0 / math.sqrt(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class STUFilterApprox(nn.Module):
    """Approximate STU filter: collapse the ``r``-dim basis to
    ``branch_dim`` learnable filters, then FFT-convolve each channel.
    """

    def __init__(self, N: int, r: int, branch_dim: int) -> None:
        super().__init__()
        self.N = N
        self.r = r
        self.branch_dim = branch_dim
        self.W_branch = nn.Parameter(torch.empty(branch_dim, N))
        self.filter_proj = nn.Parameter(torch.empty(r, branch_dim))
        self.W_out = nn.Parameter(torch.empty(N, branch_dim))
        nn.init.normal_(self.W_branch, std=1.0 / math.sqrt(N))
        nn.init.normal_(self.filter_proj, std=1.0 / math.sqrt(branch_dim))
        nn.init.normal_(self.W_out, std=1.0 / math.sqrt(branch_dim))

    def forward(self, h: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        z = h @ self.W_branch.T                # (B, T, branch_dim)
        filters = phi @ self.filter_proj       # (T, branch_dim)
        y = fft_convolve_1d(z, filters)         # (B, T, branch_dim)
        return y @ self.W_out.T                 # (B, T, N)


class STUFilterFull(nn.Module):
    """Full STU filter: apply all ``r`` basis filters to ``z``, mix via a
    learnable ``(r, branch_dim, branch_dim)`` tensor.
    """

    def __init__(self, N: int, r: int, branch_dim: int) -> None:
        super().__init__()
        self.N = N
        self.r = r
        self.branch_dim = branch_dim
        self.W_branch = nn.Parameter(torch.empty(branch_dim, N))
        self.full_mix = nn.Parameter(torch.empty(r, branch_dim, branch_dim))
        self.W_out = nn.Parameter(torch.empty(N, branch_dim))
        nn.init.normal_(self.W_branch, std=1.0 / math.sqrt(N))
        nn.init.normal_(self.full_mix, std=1.0 / math.sqrt(branch_dim))
        nn.init.normal_(self.W_out, std=1.0 / math.sqrt(branch_dim))

    def forward(self, h: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        z = h @ self.W_branch.T                # (B, T, branch_dim)
        basis = fft_convolve_basis(z, phi)     # (B, T, r, branch_dim)
        y = torch.einsum("btkc,kco->bto", basis, self.full_mix)  # (B, T, branch_dim)
        return y @ self.W_out.T                # (B, T, N)


class STUBlock(nn.Module):
    """One sandwich block."""

    def __init__(
        self, N: int, r: int, mlp_ratio: float, use_approx: bool, branch_dim: int,
    ) -> None:
        super().__init__()
        hidden = int(N * mlp_ratio)
        self.norm1 = RMSNorm()
        self.mlp_in = MLP(N, hidden)
        self.norm2 = RMSNorm()
        if use_approx:
            self.stu: nn.Module = STUFilterApprox(N, r, branch_dim)
        else:
            self.stu = STUFilterFull(N, r, branch_dim)
        self.norm3 = RMSNorm()
        self.mlp_out = MLP(N, hidden)

    def forward(self, h: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        h = h + self.mlp_in(self.norm1(h))
        h = h + self.stu(self.norm2(h), phi)
        h = h + self.mlp_out(self.norm3(h))
        return h


# ---------------------------------------------------------------------------
# STUNative
# ---------------------------------------------------------------------------


class STUNative(nn.Module):
    """Hazan-style Spectral Transform Unit sequence model for ICL regression.

    Parameters
    ----------
    D
        Input feature dimension.
    N
        Residual stream width.
    P
        Training context length.
    K
        Number of query positions. The full sequence length is ``T = P + K``.
    L
        Number of sandwich blocks.
    r
        Spectral rank (number of basis filters kept). Must satisfy ``r <= T``.
    mlp_ratio
        Hidden dimension of each MLP, as ``int(N * mlp_ratio)``. Default ``4.0``.
    use_approx
        If ``True`` (default), use the approximate STU filter. If ``False``,
        use the full r-basis filter with a ``(r, branch_dim, branch_dim)`` mix.
    basis
        ``"hankel"`` (default, STU-canonical) or ``"dft"``.
    share_layers
        If ``True``, all ``L`` blocks share the same parameters.
    """

    def __init__(
        self,
        D: int,
        N: int,
        P: int,
        K: int,
        L: int,
        r: int,
        mlp_ratio: float = 4.0,
        use_approx: bool = True,
        basis: str = "hankel",
        share_layers: bool = False,
    ) -> None:
        super().__init__()
        if D < 1 or N < 1 or P < 1 or K < 1 or L < 1 or r < 1:
            raise ValueError(
                f"D, N, P, K, L, r must be >= 1; got "
                f"({D}, {N}, {P}, {K}, {L}, {r})"
            )
        T = P + K
        if r > T:
            raise ValueError(f"r must be <= P + K = {T}; got r={r}")

        self.D = int(D)
        self.N = int(N)
        self.P = int(P)
        self.K = int(K)
        self.L = int(L)
        self.r = int(r)
        self.T = int(T)
        self.basis_kind = basis
        self.share_layers = bool(share_layers)

        self.W_in = nn.Parameter(torch.empty(N, D + 1))
        self.W_readout = nn.Parameter(torch.empty(1, N))
        nn.init.normal_(self.W_in, std=1.0 / math.sqrt(D + 1))
        nn.init.normal_(self.W_readout, std=1.0 / math.sqrt(N))

        if basis == "hankel":
            phi = get_spectral_filters(T, r)
        elif basis == "dft":
            phi = get_dft_filters(T, r)
        else:
            raise ValueError(f"basis must be 'hankel' or 'dft'; got {basis!r}")
        self.register_buffer("phi", phi.to(torch.float32))

        branch_dim = N
        if self.share_layers:
            self.block: nn.Module | None = STUBlock(
                N, r, mlp_ratio, use_approx, branch_dim,
            )
            self.blocks: nn.ModuleList | None = None
        else:
            self.block = None
            self.blocks = nn.ModuleList([
                STUBlock(N, r, mlp_ratio, use_approx, branch_dim)
                for _ in range(L)
            ])

    # -- embedding + forward ------------------------------------------------

    def embed(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_query: torch.Tensor,
    ) -> torch.Tensor:
        """Pack ``[x; y]`` training tokens and ``[x; 0]`` query tokens, then
        project to width ``N`` via ``W_in``. Returns ``(B, T, N)``.
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

        B = X_train.shape[0]
        device = X_train.device
        dtype = X_train.dtype

        train_tokens = torch.cat(
            [X_train.transpose(-1, -2), y_train.unsqueeze(-1)], dim=-1,
        )                                                  # (B, P, D+1)
        query_tokens = torch.cat(
            [
                X_query.transpose(-1, -2),
                torch.zeros(B, self.K, 1, dtype=dtype, device=device),
            ],
            dim=-1,
        )                                                  # (B, K, D+1)
        tokens = torch.cat([train_tokens, query_tokens], dim=1)  # (B, T, D+1)
        return tokens @ self.W_in.T                         # (B, T, N)

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_query: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed(X_train, y_train, X_query)
        if self.share_layers:
            assert self.block is not None
            for _ in range(self.L):
                h = self.block(h, self.phi)
        else:
            assert self.blocks is not None
            for block in self.blocks:
                h = block(h, self.phi)
        query_h = h[:, self.P :, :]                        # (B, K, N)
        f = query_h @ self.W_readout.T                      # (B, K, 1)
        return f.squeeze(-1)                                # (B, K)

    # -- train_icl_online contract ------------------------------------------

    def loss(self, f_pred: torch.Tensor, y_query: torch.Tensor) -> torch.Tensor:
        if f_pred.shape != y_query.shape:
            raise ValueError(
                f"f_pred {tuple(f_pred.shape)} vs y_query {tuple(y_query.shape)} "
                "shape mismatch"
            )
        return ((f_pred - y_query) ** 2).mean()

    def enforce_alignment(self) -> None:
        """No alignment invariants on STUNative; kept as a no-op so that
        :func:`train_icl_online` can call it uniformly across model types.
        """
        return None


__all__ = [
    "STUNative",
    "STUBlock",
    "STUFilterApprox",
    "STUFilterFull",
    "RMSNorm",
    "MLP",
    "get_hankel",
    "get_spectral_filters",
    "get_dft_filters",
    "fft_convolve_1d",
    "fft_convolve_basis",
    "nearest_power_of_two",
]
