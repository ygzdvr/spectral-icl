"""Unit tests for :class:`SpectralAttention`.

Run as either::

    python -m scripts.thesis.architectures.tests.test_spectral_attention
    python scripts/thesis/architectures/tests/test_spectral_attention.py

All tests run on CUDA in float64 (no CPU fallback). Four checks:

- **Test 1** - shape + alignment + circulant structure: after construction and
  ``enforce_alignment``, the four alignment invariants hold, the forward pass
  returns ``(B, K)``, and ``_build_S_TT()`` produces a real symmetric
  circulant (verified via :func:`off_diagonal_fourier_energy`).
- **Test 2** - bridge to :class:`LinearAttention`: with ``r = P`` and
  ``s_init="gd"`` and shared ``(W_x, w_y, alpha_v, W_q, W_k)``,
  ``SpectralAttention`` is bitwise-identical to ``LinearAttention`` across
  four geometries. Cross-checks the differentiable ``irfft``-based ``S_TT``
  against the operator-level
  :func:`circulant_from_symbol` / :func:`real_even_symbol_from_half` path.
- **Test 3** - spectral bottleneck: with ``r < P // 2 + 1``, every
  half-spectrum mode beyond ``r_half`` is zero; the first ``r_half`` modes
  equal ``s_half`` exactly.
- **Test 4** - gradient flow: ``loss.backward()`` populates a nonzero
  ``s_half.grad`` together with nonzero ``W_q.grad`` and ``W_k.grad``,
  confirming the differentiable DFT parameterization propagates gradients.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[4]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import torch

from scripts.thesis.architectures.models import (  # noqa: E402
    LinearAttention,
    SpectralAttention,
)
from scripts.thesis.utils.fourier_ops import (  # noqa: E402
    circulant_from_symbol,
    off_diagonal_fourier_energy,
    real_even_symbol_from_half,
    symbol_of_circulant,
)


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this test - no CPU fallback. "
        "Run on a GPU node (A100 login node on the thesis cluster)."
    )

DEVICE = torch.device("cuda")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_errors(model: SpectralAttention | LinearAttention) -> dict[str, float]:
    with torch.no_grad():
        wy = model.w_y.data
        return {
            "Wx_T_wy": float(torch.linalg.vector_norm(model.W_x.data.T @ wy)),
            "wy_norm_minus_1": abs(float(wy.norm()) - 1.0),
            "Wq_wy": float(torch.linalg.vector_norm(model.W_q.data @ wy)),
            "Wk_wy": float(torch.linalg.vector_norm(model.W_k.data @ wy)),
        }


def _sample_icl_batch(
    B: int, D: int, P: int, K: int, *, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    X_train = torch.randn(B, D, P, generator=gen, dtype=DTYPE, device=DEVICE)
    X_query = torch.randn(B, D, K, generator=gen, dtype=DTYPE, device=DEVICE)
    beta = torch.randn(B, D, generator=gen, dtype=DTYPE, device=DEVICE)
    y_train = torch.einsum("bd,bdp->bp", beta, X_train) / math.sqrt(D)
    y_query = torch.einsum("bd,bdk->bk", beta, X_query) / math.sqrt(D)
    return X_train, y_train, X_query, y_query


def _build_spectral(
    D: int, N: int, P: int, K: int, L: int, r: int, s_init: str,
) -> SpectralAttention:
    model = SpectralAttention(D, N, P, K, L, r, s_init=s_init).to(
        device=DEVICE, dtype=DTYPE
    )
    model.enforce_alignment()
    return model


def _build_linear(D: int, N: int, P: int, K: int, L: int) -> LinearAttention:
    model = LinearAttention(D, N, P, K, L).to(device=DEVICE, dtype=DTYPE)
    model.enforce_alignment()
    return model


# ---------------------------------------------------------------------------
# Test 1 - shape + alignment + circulant structure
# ---------------------------------------------------------------------------


def test_1_shape_alignment_circulant() -> tuple[bool, str]:
    torch.manual_seed(0)
    D, N, P, K, L, r = 16, 32, 16, 8, 4, 8
    B = 4

    model = _build_spectral(D, N, P, K, L, r=r, s_init="zero")

    errs = _align_errors(model)
    align_ok = all(v < 1e-12 for v in errs.values())

    gen = torch.Generator(device=DEVICE).manual_seed(0)
    X_train = torch.randn(B, D, P, generator=gen, dtype=DTYPE, device=DEVICE)
    X_query = torch.randn(B, D, K, generator=gen, dtype=DTYPE, device=DEVICE)
    y_train = torch.randn(B, P, generator=gen, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        f = model(X_train, y_train, X_query)
    shape_ok = tuple(f.shape) == (B, K)

    S_TT = model._build_S_TT().detach()
    S_TT_shape_ok = tuple(S_TT.shape) == (P, P)
    # `off_diagonal_fourier_energy` lives in the operator-level fourier_ops
    # (CPU float64); move S_TT to CPU for the check.
    off_energy = off_diagonal_fourier_energy(S_TT.cpu().to(torch.float64))
    circ_ok = off_energy < 1e-10

    ok = align_ok and shape_ok and S_TT_shape_ok and circ_ok
    parts = [f"{k}={v:.2e}" for k, v in errs.items()]
    parts.append(f"forward_shape={tuple(f.shape)} expected=({B},{K})")
    parts.append(f"S_TT_shape={tuple(S_TT.shape)}")
    parts.append(f"off_diag_fourier={off_energy:.2e}")
    return ok, ", ".join(parts)


# ---------------------------------------------------------------------------
# Test 2 - bridge to LinearAttention
# ---------------------------------------------------------------------------


def test_2_bridge_linear_attention() -> tuple[bool, str]:
    torch.manual_seed(1)
    geometries = [(8, 8, 4, 1), (16, 16, 8, 2), (32, 32, 16, 4), (64, 64, 32, 8)]
    B = 4
    all_ok = True
    lines: list[str] = []

    for (D, P, K, L) in geometries:
        N = 2 * D

        torch.manual_seed(100 + 37 * D + 11 * P)
        linear = _build_linear(D, N, P, K, L)
        spectral = _build_spectral(D, N, P, K, L, r=P, s_init="gd")

        with torch.no_grad():
            spectral.W_x.data.copy_(linear.W_x.data)
            spectral.w_y.data.copy_(linear.w_y.data)
            spectral.alpha_v.data.copy_(linear.alpha_v.data)
            spectral.W_q.data.copy_(linear.W_q.data)
            spectral.W_k.data.copy_(linear.W_k.data)
        spectral.enforce_alignment()

        s_err_dc = abs(float(spectral.s_half.data[0].item()) - (-float(P)))
        if spectral.r_half > 1:
            s_err_rest = float(spectral.s_half.data[1:].abs().max().item())
        else:
            s_err_rest = 0.0
        s_ok = (s_err_dc < 1e-14) and (s_err_rest < 1e-14)

        s_half_cpu = spectral.s_half.data.cpu().to(torch.float64)
        full_symbol = real_even_symbol_from_half(s_half_cpu, P)
        S_TT_fourier = circulant_from_symbol(full_symbol)
        S_TT_model = spectral._build_S_TT().detach().cpu().to(torch.float64)
        st_err = float((S_TT_model - S_TT_fourier).abs().max().item())
        st_ok = st_err < 1e-10

        S_TT_neg_err = float((S_TT_model + 1.0).abs().max().item())
        neg_ok = S_TT_neg_err < 1e-10

        X_train, y_train, X_query, _ = _sample_icl_batch(
            B, D, P, K, seed=2000 + 37 * D + 11 * P,
        )
        with torch.no_grad():
            f_lin = linear(X_train, y_train, X_query)
            f_spec = spectral(X_train, y_train, X_query)
        diff = float(torch.linalg.vector_norm((f_lin - f_spec).flatten()).item())
        denom = float(torch.linalg.vector_norm(f_lin.flatten()).item()) + 1e-30
        rel = diff / denom
        rel_ok = rel < 1e-10

        ok = s_ok and st_ok and neg_ok and rel_ok
        all_ok = all_ok and ok
        tag = "" if ok else " <-- FAIL"
        lines.append(
            f"(D={D:>2},P={P:>2},K={K:>2},L={L}): "
            f"s_dc={s_err_dc:.1e}, s_rest={s_err_rest:.1e}, "
            f"S_fourier_err={st_err:.1e}, S_neg_err={S_TT_neg_err:.1e}, "
            f"forward_rel={rel:.2e}{tag}"
        )
    return all_ok, "\n      ".join(lines)


# ---------------------------------------------------------------------------
# Test 3 - spectral bottleneck enforcement
# ---------------------------------------------------------------------------


def test_3_bottleneck() -> tuple[bool, str]:
    torch.manual_seed(2)
    D, N, P, K, L, r = 16, 32, 16, 8, 4, 4
    model = _build_spectral(D, N, P, K, L, r=r, s_init="zero")
    full_half = P // 2 + 1
    assert model.r_half == min(r, full_half) == 4

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    random_sh = torch.randn(model.r_half, generator=gen, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        model.s_half.data.copy_(random_sh)

    S_TT = model._build_S_TT().detach()
    S_TT_cpu = S_TT.cpu().to(torch.float64)

    off_energy = off_diagonal_fourier_energy(S_TT_cpu)
    circ_ok = off_energy < 1e-10

    symbol = symbol_of_circulant(S_TT_cpu)
    beyond_r = float(symbol[model.r_half : full_half].abs().max().item())
    beyond_ok = beyond_r < 1e-10

    sh_cpu = model.s_half.data.cpu().to(torch.float64)
    within_err = float((symbol[: model.r_half] - sh_cpu).abs().max().item())
    within_ok = within_err < 1e-10

    ok = circ_ok and beyond_ok and within_ok
    detail = (
        f"r={r}, r_half={model.r_half}, P//2+1={full_half}, "
        f"off_diag_fourier={off_energy:.2e}, "
        f"beyond_r_max={beyond_r:.2e}, within_r_err={within_err:.2e}"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Test 4 - gradient flow through s_half
# ---------------------------------------------------------------------------


def test_4_gradient_flow() -> tuple[bool, str]:
    torch.manual_seed(3)
    D, N, P, K, L, r = 16, 32, 16, 8, 4, 8
    B = 8

    model = _build_spectral(D, N, P, K, L, r=r, s_init="gd")

    X_train, y_train, X_query, y_query = _sample_icl_batch(
        B, D, P, K, seed=999,
    )
    f = model(X_train, y_train, X_query)
    loss = model.loss(f, y_query)
    loss.backward()

    s_grad = model.s_half.grad
    wq_grad = model.W_q.grad
    wk_grad = model.W_k.grad

    s_max = float(s_grad.abs().max().item()) if s_grad is not None else 0.0
    wq_max = float(wq_grad.abs().max().item()) if wq_grad is not None else 0.0
    wk_max = float(wk_grad.abs().max().item()) if wk_grad is not None else 0.0

    s_ok = (s_grad is not None) and (s_max > 1e-15)
    wq_ok = (wq_grad is not None) and (wq_max > 0.0)
    wk_ok = (wk_grad is not None) and (wk_max > 0.0)

    ok = s_ok and wq_ok and wk_ok
    detail = (
        f"s_half.grad_max={s_max:.2e} (>1e-15 required), "
        f"W_q.grad_max={wq_max:.2e}, W_k.grad_max={wk_max:.2e}, "
        f"loss={float(loss.detach().item()):.4e}"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"dtype:  {DTYPE}")
    tests = [
        ("Test 1 (shape + alignment + circulant)", test_1_shape_alignment_circulant),
        ("Test 2 (bridge SpectralAttention(r=P, gd) == LinearAttention)",
         test_2_bridge_linear_attention),
        ("Test 3 (spectral bottleneck r=4, P=16)", test_3_bottleneck),
        ("Test 4 (gradient flow through s_half)", test_4_gradient_flow),
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
