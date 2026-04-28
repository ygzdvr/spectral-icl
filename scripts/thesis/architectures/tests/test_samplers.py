"""Unit tests for :func:`make_stationary_sampler`.

Run as either::

    python -m scripts.thesis.architectures.tests.test_samplers
    python scripts/thesis/architectures/tests/test_samplers.py

All tests on CUDA, float32 (training precision). Four checks:

- **Test 1** - shape + device + fresh draws + metadata sanity.
- **Test 2** - empirical ``(1/P) X^T X`` averaged over contexts is
  approximately circulant (off-diagonal Fourier energy below ``0.01``).
- **Test 3** - the symbol extracted from the empirical kernel matches
  ``(D / P) * s_data`` both in shape (Pearson r > 0.99) and scale
  (mean within 5% of ``D / P``). Symbol extraction uses ``fft`` of the
  first column directly; :func:`symbol_of_circulant`'s strict 1e-10 tolerance
  is too tight for a Monte Carlo empirical kernel built in float32.
- **Test 4** - flat symbol collapses to isotropic iid: the empirical kernel
  is approximately ``(D / P) I_P``, i.e. off-diagonal Frobenius norm is
  small compared to diagonal Frobenius norm.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[4]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np
import torch

from scripts.thesis.architectures.samplers import make_stationary_sampler  # noqa: E402
from scripts.thesis.utils.fourier_ops import off_diagonal_fourier_energy  # noqa: E402


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this test - no CPU fallback. "
        "Run on a GPU node (A100 login node on the thesis cluster)."
    )

DEVICE = torch.device("cuda")
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empirical_kernel(
    sampler_fn, n_batches: int, P: int, device: torch.device,
) -> torch.Tensor:
    """Accumulate ``(1/P) X^T X`` across ``n_batches`` sampler calls and return
    the per-context mean as a CPU float64 ``(P, P)`` tensor.
    """
    acc = torch.zeros(P, P, device=device, dtype=torch.float64)
    n_contexts = 0
    for _ in range(n_batches):
        X_train, _, _, _ = sampler_fn()
        X64 = X_train.to(torch.float64)
        k_per_b = torch.einsum("bdp,bdq->bpq", X64, X64) / P
        acc = acc + k_per_b.sum(dim=0)
        n_contexts += X_train.shape[0]
    return (acc / n_contexts).cpu()


# ---------------------------------------------------------------------------
# Test 1 - shape + device + freshness + metadata
# ---------------------------------------------------------------------------


def test_1_shape_and_device() -> tuple[bool, str]:
    torch.manual_seed(0)
    P, D, K, B = 32, 64, 16, 8
    sampler_fn, meta = make_stationary_sampler(
        P=P, D=D, K=K, B=B, symbol_kind="power_law", nu=1.0,
        device=DEVICE, dtype=DTYPE,
    )

    b1 = sampler_fn()
    b2 = sampler_fn()
    X_tr, y_tr, X_q, y_q = b1

    shape_ok = (
        tuple(X_tr.shape) == (B, D, P)
        and tuple(y_tr.shape) == (B, P)
        and tuple(X_q.shape) == (B, D, K)
        and tuple(y_q.shape) == (B, K)
    )
    device_ok = all(t.device.type == "cuda" for t in b1)
    dtype_ok = all(t.dtype == DTYPE for t in b1)
    diff_ok = not torch.allclose(X_tr, b2[0])
    meta_keys_ok = {
        "symbol", "sqrt_symbol", "symbol_kind", "nu", "P", "D"
    }.issubset(meta.keys())
    meta_values_ok = (
        meta["P"] == P
        and meta["D"] == D
        and meta["symbol_kind"] == "power_law"
        and tuple(meta["symbol"].shape) == (P,)
        and meta["symbol"].dtype == torch.float64
    )

    ok = shape_ok and device_ok and dtype_ok and diff_ok and meta_keys_ok and meta_values_ok
    detail = (
        f"shapes=({tuple(X_tr.shape)},{tuple(y_tr.shape)},"
        f"{tuple(X_q.shape)},{tuple(y_q.shape)}); "
        f"device={X_tr.device.type}; dtype={X_tr.dtype}; "
        f"batches_differ={diff_ok}; metadata_keys_ok={meta_keys_ok}; "
        f"metadata_values_ok={meta_values_ok}"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Test 2 - empirical kernel is approximately circulant
# ---------------------------------------------------------------------------


def test_2_empirical_kernel_circulant() -> tuple[bool, str]:
    torch.manual_seed(1)
    P, D, K, B = 32, 64, 16, 64
    n_batches = 500
    sampler_fn, _ = make_stationary_sampler(
        P=P, D=D, K=K, B=B, symbol_kind="power_law", nu=1.0,
        device=DEVICE, dtype=DTYPE,
    )

    K_emp = _empirical_kernel(sampler_fn, n_batches, P, DEVICE)
    # Symmetrize before the circulant check so the diagnostic is on the
    # symmetric-circulant part only (the kernel is symmetric in population).
    K_emp_sym = 0.5 * (K_emp + K_emp.T)
    off_energy = off_diagonal_fourier_energy(K_emp_sym)
    ok = off_energy < 0.01
    detail = (
        f"n_contexts = {n_batches * B}, "
        f"off_diagonal_fourier_energy = {off_energy:.4e} (threshold < 0.01)"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Test 3 - symbol of empirical kernel matches prescribed data symbol
# ---------------------------------------------------------------------------


def test_3_symbol_match() -> tuple[bool, str]:
    torch.manual_seed(2)
    P, D, K, B = 32, 64, 16, 64
    n_batches = 500
    sampler_fn, meta = make_stationary_sampler(
        P=P, D=D, K=K, B=B, symbol_kind="power_law", nu=1.0,
        device=DEVICE, dtype=DTYPE,
    )

    K_emp = _empirical_kernel(sampler_fn, n_batches, P, DEVICE)
    K_emp_sym = 0.5 * (K_emp + K_emp.T)
    # FFT of the first column yields the symbol of the (symmetric-circulant)
    # projection of K_emp. We use this direct route rather than
    # symbol_of_circulant because the Monte Carlo kernel has float32 noise
    # that trips the latter's 1e-10 tolerance checks.
    first_col = K_emp_sym[:, 0]
    extracted = torch.fft.fft(first_col).real.numpy()

    # Expected population symbol: (D / P) * s_data. symbol_power_law is
    # mean-1 normalized, so the extracted symbol should have mean D/P.
    expected = (D / P) * meta["symbol"].numpy()

    pearson = float(np.corrcoef(extracted, expected)[0, 1])
    mean_ext = float(extracted.mean())
    expected_mean = float(D / P)
    scale_rel_err = abs(mean_ext - expected_mean) / expected_mean

    corr_ok = pearson > 0.99
    scale_ok = scale_rel_err < 0.05
    ok = corr_ok and scale_ok
    detail = (
        f"pearson_r = {pearson:.4f} (> 0.99 required), "
        f"mean_extracted = {mean_ext:.4f}, expected_mean = D/P = {expected_mean:.4f}, "
        f"scale_rel_err = {scale_rel_err:.4e} (< 0.05 required)"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Test 4 - flat symbol reduces to isotropic iid
# ---------------------------------------------------------------------------


def test_4_flat_symbol_isotropic() -> tuple[bool, str]:
    torch.manual_seed(3)
    P, D, K, B = 32, 64, 16, 64
    n_batches = 500
    sampler_fn, _ = make_stationary_sampler(
        P=P, D=D, K=K, B=B, symbol_kind="flat",
        device=DEVICE, dtype=DTYPE,
    )

    K_emp = _empirical_kernel(sampler_fn, n_batches, P, DEVICE)
    diag = torch.diagonal(K_emp)
    off_diag = K_emp - torch.diag(diag)
    off_diag_fro = float(off_diag.norm().item())
    diag_fro = float(diag.norm().item())
    ratio = off_diag_fro / max(diag_fro, 1e-30)
    mean_diag = float(diag.mean().item())
    expected_diag = float(D / P)
    ok = ratio < 0.1
    detail = (
        f"off_diag_fro / diag_fro = {ratio:.4e} (< 0.1 required), "
        f"mean_diag = {mean_diag:.4f}, expected = D/P = {expected_diag:.4f}"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"dtype:  {DTYPE}")
    tests = [
        ("Test 1 (shape + device + freshness + metadata)", test_1_shape_and_device),
        ("Test 2 (empirical kernel is approximately circulant)", test_2_empirical_kernel_circulant),
        ("Test 3 (extracted symbol matches prescribed s_data)", test_3_symbol_match),
        ("Test 4 (flat symbol -> isotropic iid data)", test_4_flat_symbol_isotropic),
    ]
    all_pass = True
    for name, fn in tests:
        ok, detail = fn()
        tag = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok
        print(f"[{tag}] {name}")
        print(f"      {detail}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
