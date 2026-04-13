"""Fourier primitives for the thesis generator layer.

Plan correspondence: EXPERIMENT_PLAN_FINAL.MD Section 4.7 (``fourier_ops.py``),
Step-1 Generator / Utility Specification v4 §4.

This module is the sole place in the thesis codebase where complex arithmetic
is allowed. The four complex-returning endpoints
    dft_matrix, idft_matrix, unitary_dft, unitary_idft
are labeled "# complex - isolated" and are NOT consumed directly by
``data_generators.py`` or any architecture-facing code - consumers use the
real-valued wrapper functions in this module. Every real-valued wrapper that
builds its output through a complex intermediate asserts that ``max(|imag|)``
is below ``1e-10`` before casting to ``float64``, so a non-real-even symbol
(or a non-real-symmetric-circulant input) raises cleanly instead of propagating
complex noise downstream.

Conventions (used throughout):

- Unitary DFT matrix ``F`` with entries
      F[k, j] = exp(-2*pi*i*j*k / P) / sqrt(P).
- Circulant diagonalization
      C = F^H diag(s) F,
  where ``s`` is the real-even Fourier symbol.
- Real-even symbol: s[P - k] = s[k] for k = 1, ..., (P - 1)//2; with k = 0 and
  (for even P) k = P/2 being self-mirrors.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


_COMPLEX_LEAK_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# Frequency grid
# ---------------------------------------------------------------------------


def freq_grid(P: int) -> torch.Tensor:
    """Length-P real frequency grid ``2*pi*k / P`` for ``k = 0, 1, ..., P - 1``.

    Returns a real ``float64`` tensor.
    """
    if P < 1:
        raise ValueError(f"P must be positive, got {P}")
    return (2.0 * math.pi / P) * torch.arange(P, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Isolated complex Fourier helpers
# ---------------------------------------------------------------------------


def dft_matrix(P: int) -> torch.Tensor:
    """# complex - isolated

    Return the P x P UNITARY DFT matrix F, with
        F[k, j] = exp(-2*pi*i*j*k / P) / sqrt(P).

    Output dtype is ``complex128``.
    """
    if P < 1:
        raise ValueError(f"P must be positive, got {P}")
    k = torch.arange(P, dtype=torch.float64).view(-1, 1)
    j = torch.arange(P, dtype=torch.float64).view(1, -1)
    phase = -2.0 * math.pi * k * j / P
    real = torch.cos(phase) / math.sqrt(P)
    imag = torch.sin(phase) / math.sqrt(P)
    return torch.complex(real, imag)


def idft_matrix(P: int) -> torch.Tensor:
    """# complex - isolated

    Return the inverse unitary DFT matrix ``F^H = F*``. Output dtype is
    ``complex128``.
    """
    return dft_matrix(P).conj().transpose(-2, -1).contiguous()


def unitary_dft(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """# complex - isolated

    Unitary DFT along ``dim``. Accepts real or complex input; output is
    ``complex128`` (divides by ``sqrt(P)``).
    """
    return torch.fft.fft(x, dim=dim, norm="ortho").to(torch.complex128)


def unitary_idft(X: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """# complex - isolated

    Unitary inverse DFT along ``dim``. Output dtype is ``complex128``.
    """
    return torch.fft.ifft(X, dim=dim, norm="ortho").to(torch.complex128)


# ---------------------------------------------------------------------------
# Real spectral basis (fixed F for the canonical FFT spectral backbone)
# ---------------------------------------------------------------------------


def real_spectral_basis(D: int, kind: str = "dct2") -> torch.Tensor:
    """Return a D x D real orthogonal matrix F used as the fixed spectral basis
    for the canonical FFT spectral backbone (v4 spec §3.1).

    Supported kinds:
        "dct2"     : orthonormal DCT-II basis (default). Rows k = 0, ..., D - 1
                     are given by
                         F[k, j] = c_k * cos(pi * k * (2*j + 1) / (2 * D)),
                     with c_0 = sqrt(1 / D) and c_k = sqrt(2 / D) for k >= 1.
        "identity" : identity matrix (spectral basis = physical basis).

    Output is a real ``float64`` tensor.
    """
    if D < 1:
        raise ValueError(f"D must be positive, got {D}")
    if kind == "dct2":
        k = torch.arange(D, dtype=torch.float64).view(-1, 1)
        j = torch.arange(D, dtype=torch.float64).view(1, -1)
        C = torch.cos(math.pi * k * (2 * j + 1) / (2 * D))
        c = torch.full((D,), math.sqrt(2.0 / D), dtype=torch.float64)
        c[0] = math.sqrt(1.0 / D)
        return (c.view(-1, 1) * C).contiguous()
    if kind == "identity":
        return torch.eye(D, dtype=torch.float64)
    raise ValueError(
        f"unknown real_spectral_basis kind: {kind!r}; expected 'dct2' or 'identity'"
    )


# ---------------------------------------------------------------------------
# Circulant diagonalization (real boundary wrappers)
# ---------------------------------------------------------------------------


def _check_real_symbol(s: torch.Tensor, name: str = "s") -> None:
    if s.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {tuple(s.shape)}")
    if not s.is_floating_point():
        raise TypeError(f"{name} must be real floating-point; got dtype {s.dtype}")


def circulant_from_symbol(s: torch.Tensor) -> torch.Tensor:
    """Real (P, P) circulant matrix with Fourier symbol ``s``:

        C = F^H diag(s) F,

    where F is the unitary DFT. For C to be real, s must be real-even. This
    function computes the complex intermediate, asserts
    ``max|imag(F^H diag(s) F)| < 1e-10``, and returns the real part cast to
    ``float64``. A non-real-even symbol raises ``ValueError``.
    """
    _check_real_symbol(s, "symbol s")
    P = s.shape[0]
    F = dft_matrix(P)
    s_c = s.to(torch.complex128)
    C = F.conj().transpose(-2, -1) @ torch.diag(s_c) @ F
    leak = C.imag.abs().max().item()
    if leak > _COMPLEX_LEAK_TOL:
        raise ValueError(
            f"circulant_from_symbol: max|imag(F^H diag(s) F)| = {leak:.3e} "
            f"exceeds tolerance {_COMPLEX_LEAK_TOL:.0e}; "
            "symbol is not real-even enough"
        )
    return C.real.to(torch.float64).contiguous()


def symbol_of_circulant(C: torch.Tensor) -> torch.Tensor:
    """Extract the real symbol of a real-symmetric-circulant matrix C:

        s = diag(F C F^H).

    Asserts C is square real floating-point, that ``F C F^H`` is diagonal
    (off-diagonal leakage below ``1e-10``), and that the diagonal is real
    (imaginary leakage below ``1e-10``); otherwise raises ``ValueError``.
    Output dtype is ``float64``.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square 2D; got shape {tuple(C.shape)}")
    if not C.is_floating_point():
        raise TypeError(f"C must be real floating-point; got dtype {C.dtype}")
    P = C.shape[0]
    F = dft_matrix(P)
    D = F @ C.to(torch.complex128) @ F.conj().transpose(-2, -1)
    s = torch.diagonal(D)
    off = D - torch.diag(s)
    off_leak = off.abs().max().item()
    if off_leak > _COMPLEX_LEAK_TOL:
        raise ValueError(
            f"symbol_of_circulant: C is not circulant; "
            f"max|off-diagonal of F C F^H| = {off_leak:.3e} "
            f"exceeds tolerance {_COMPLEX_LEAK_TOL:.0e}"
        )
    imag_leak = s.imag.abs().max().item()
    if imag_leak > _COMPLEX_LEAK_TOL:
        raise ValueError(
            f"symbol_of_circulant: diagonal has imaginary leakage {imag_leak:.3e} "
            f"exceeds tolerance {_COMPLEX_LEAK_TOL:.0e}; "
            "C is not real-symmetric-circulant"
        )
    return s.real.to(torch.float64).contiguous()


# ---------------------------------------------------------------------------
# Real-even symbol constructors
# ---------------------------------------------------------------------------


def real_even_symbol_from_half(half: torch.Tensor, P: int) -> torch.Tensor:
    """Extend a half-spectrum to a real-even length-P symbol.

    ``half`` has length ``P // 2 + 1`` with entries ``s[0], s[1], ..., s[P//2]``.
    The returned length-P symbol satisfies s[P - k] = s[k] for
    k = 1, ..., (P - 1)//2; for even P, s[P/2] is taken directly from ``half``.
    Output dtype is ``float64``.
    """
    _check_real_symbol(half, "half")
    if P < 1:
        raise ValueError(f"P must be positive, got {P}")
    n_half = P // 2 + 1
    if half.shape[0] != n_half:
        raise ValueError(
            f"expected half length P//2 + 1 = {n_half} for P = {P}; "
            f"got {half.shape[0]}"
        )
    s = torch.zeros(P, dtype=torch.float64)
    h = half.to(torch.float64)
    s[:n_half] = h
    for k in range(1, (P + 1) // 2):
        s[P - k] = h[k]
    return s


def symbol_power_law(P: int, nu: float, *, eps: float = 1e-6) -> torch.Tensor:
    """Real-even power-law symbol

        s[k] = (1 + k_star)^(-nu),

    where ``k_star = min(k, P - k)`` is the centered circular mode index.
    Floored at ``eps`` and normalized so that ``mean(s) == 1``. Output dtype is
    ``float64``.
    """
    if P < 1:
        raise ValueError(f"P must be positive, got {P}")
    k = torch.arange(P, dtype=torch.float64)
    k_star = torch.minimum(k, P - k)
    s = (1.0 + k_star).pow(-float(nu))
    s = torch.clamp(s, min=float(eps))
    s = s / s.mean()
    return s


def symbol_multiband(
    P: int, bands: Sequence[tuple[int, int, float]]
) -> torch.Tensor:
    """Piecewise-constant real-even symbol.

    Each band ``(k_lo, k_hi, value)`` sets s[k_lo..k_hi] = value on the
    half-spectrum (indices 0, ..., P // 2); the full-length symbol is produced
    via :func:`real_even_symbol_from_half`. Entries outside all bands default to
    ``1e-6``. Output is normalized so that ``mean(s) == 1`` and has dtype
    ``float64``.
    """
    if P < 1:
        raise ValueError(f"P must be positive, got {P}")
    _default_eps = 1e-6
    n_half = P // 2 + 1
    half = torch.full((n_half,), _default_eps, dtype=torch.float64)
    for i, band in enumerate(bands):
        k_lo, k_hi, value = band
        if not (0 <= k_lo <= k_hi < n_half):
            raise ValueError(
                f"band #{i} = ({k_lo}, {k_hi}, {value}) out of half-spectrum "
                f"range [0, {n_half - 1}] or has k_hi < k_lo"
            )
        if value <= 0:
            raise ValueError(
                f"band #{i} = ({k_lo}, {k_hi}, {value}) has non-positive value"
            )
        half[k_lo : k_hi + 1] = float(value)
    s = real_even_symbol_from_half(half, P)
    s = s / s.mean()
    return s


def symbol_flat(P: int, value: float = 1.0) -> torch.Tensor:
    """Flat real-even symbol with ``s[k] == value`` for all k. Output dtype is
    ``float64``.
    """
    if P < 1:
        raise ValueError(f"P must be positive, got {P}")
    return torch.full((P,), float(value), dtype=torch.float64)


def symbol_interpolate(
    s0: torch.Tensor, s1: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Convex interpolation ``(1 - alpha) * s0 + alpha * s1``. Real-even
    structure is preserved when both endpoints are real-even. Output dtype is
    ``float64``.

    Used for the B3 OOD spectral shift family.
    """
    _check_real_symbol(s0, "s0")
    _check_real_symbol(s1, "s1")
    if s0.shape != s1.shape:
        raise ValueError(
            f"s0 and s1 shapes must match; got {tuple(s0.shape)} vs {tuple(s1.shape)}"
        )
    a = float(alpha)
    return (
        (1.0 - a) * s0.to(torch.float64) + a * s1.to(torch.float64)
    ).contiguous()


def frequency_permutation(s: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Permute frequency indices of a real-even symbol preserving real-evenness.

    The positive-frequency indices ``{1, 2, ..., (P - 1) // 2}`` are permuted
    uniformly at random (seeded by ``seed``); each k is mirrored to P - k so
    that ``s'[P - k'] = s'[k']`` holds for every permuted index k'. DC (k = 0)
    and (for even P) Nyquist (k = P/2) are fixed. Output dtype is ``float64``.

    Used for the B3 OOD spectral reallocation family.
    """
    _check_real_symbol(s, "s")
    P = s.shape[0]
    if P < 3:
        return s.to(torch.float64).clone()
    n_pos = (P - 1) // 2
    if n_pos == 0:
        return s.to(torch.float64).clone()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    pos_perm = torch.randperm(n_pos, generator=gen)
    full_perm = torch.arange(P, dtype=torch.long)
    for i in range(n_pos):
        src = i + 1
        dst = int(pos_perm[i].item()) + 1
        full_perm[src] = dst
        full_perm[P - src] = (P - dst) % P
    return s[full_perm].to(torch.float64).contiguous()


# ---------------------------------------------------------------------------
# Closure diagnostic
# ---------------------------------------------------------------------------


def off_diagonal_fourier_energy(M: torch.Tensor) -> float:
    """Circulant-closure diagnostic:

        ||offdiag(F M F^H)||_F^2 / ||F M F^H||_F^2.

    Equals 0 (up to float eps) for a circulant ``M`` and is close to 1 for a
    generic non-circulant matrix. Returns a float in [0, 1].
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square 2D; got shape {tuple(M.shape)}")
    if not M.is_floating_point():
        raise TypeError(f"M must be real floating-point; got dtype {M.dtype}")
    P = M.shape[0]
    F = dft_matrix(P)
    FMFH = F @ M.to(torch.complex128) @ F.conj().transpose(-2, -1)
    total = FMFH.abs().pow(2).sum().item()
    if total == 0.0:
        return 0.0
    diag = torch.diagonal(FMFH)
    diag_energy = diag.abs().pow(2).sum().item()
    off = total - diag_energy
    return max(0.0, off / total)


__all__ = [
    "freq_grid",
    "dft_matrix",
    "idft_matrix",
    "unitary_dft",
    "unitary_idft",
    "real_spectral_basis",
    "circulant_from_symbol",
    "symbol_of_circulant",
    "real_even_symbol_from_half",
    "symbol_power_law",
    "symbol_multiband",
    "symbol_flat",
    "symbol_interpolate",
    "frequency_permutation",
    "off_diagonal_fourier_energy",
]
