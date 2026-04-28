"""Unit tests for :class:`SpectralFilter` and :class:`AdaptiveFirstHybrid`.

Run as either::

    python -m scripts.thesis.architectures.tests.test_hybrid
    python scripts/thesis/architectures/tests/test_hybrid.py

All tests on CUDA. Five groups:

- **Test 1** - shapes + alignment for ``AdaptiveFirstHybrid`` (float32).
- **Test 2** - diagnostic ``FINDING``: ``SpectralFilter`` alone cannot do
  meaningful ICL even on isotropic data because the query-train block is
  content-independent (all-ones). Parallels the STU finding.
- **Test 3** - training: ``AdaptiveFirstHybrid`` reaches terminal loss
  below ``0.30 x`` initial in 2000 Adam steps on isotropic data. A
  ``LinearAttention(L=L_A)`` reference is printed alongside.
- **Test 4** - **the critical test**: on stationary power-law data
  (``nu = 1.5``), training two hybrids at ``r=2`` and ``r=16`` (same
  seed) yields ``terminal(r=2) / terminal(r=16) > 1.10`` (PASS) / ``>
  1.05`` (WARN) / else FAIL. This is the test that :class:`SpectralAttention`
  failed (Hadamard-with-QK-kernel bypassed the rank bottleneck) and that
  :class:`STUNative` could not attempt (no content-dependent routing).
- **Test 5** (float64) - edge-case equivalences:
  - **5a** ``AdaptiveFirstHybrid(L_A=0, L_S=L, r)`` with shared weights
    matches ``SpectralFilter(L, r)`` at machine precision.
  - **5b** ``AdaptiveFirstHybrid(L_A=L, L_S=0, r)`` with shared weights
    matches ``LinearAttention(L)`` at machine precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[4]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np
import torch

from scripts.thesis.architectures.models import (  # noqa: E402
    AdaptiveFirstHybrid,
    LinearAttention,
    SpectralFilter,
)
from scripts.thesis.architectures.samplers import make_stationary_sampler  # noqa: E402
from scripts.thesis.architectures.training import (  # noqa: E402
    make_isotropic_sampler,
    train_icl_online,
)


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this test - no CPU fallback. "
        "Run on a GPU node (A100 login node on the thesis cluster)."
    )

DEVICE = torch.device("cuda")
DTYPE = torch.float32
DTYPE_F64 = torch.float64


def _hybrid_align_errors(model: AdaptiveFirstHybrid) -> dict[str, float]:
    with torch.no_grad():
        wy = model.w_y.data
        return {
            "Wx_T_wy": float(torch.linalg.vector_norm(model.W_x.data.T @ wy)),
            "wy_norm_minus_1": abs(float(wy.norm()) - 1.0),
            "Wq_A_wy": float(torch.linalg.vector_norm(model.W_q_A.data @ wy)),
            "Wk_A_wy": float(torch.linalg.vector_norm(model.W_k_A.data @ wy)),
        }


# ---------------------------------------------------------------------------
# Test 1 - shape + alignment
# ---------------------------------------------------------------------------


def test_1_shape_alignment() -> tuple[str, str]:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    D, N, P, K, L_A, L_S, r = 16, 32, 16, 8, 2, 2, 8
    B = 4

    model = AdaptiveFirstHybrid(D, N, P, K, L_A, L_S, r).to(device=DEVICE, dtype=DTYPE)
    model.enforce_alignment()

    errs = _hybrid_align_errors(model)
    align_ok = all(v < 1e-5 for v in errs.values())

    X_train = torch.randn(B, D, P, device=DEVICE, dtype=DTYPE)
    y_train = torch.randn(B, P, device=DEVICE, dtype=DTYPE)
    X_query = torch.randn(B, D, K, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        f = model(X_train, y_train, X_query)
    shape_ok = tuple(f.shape) == (B, K)

    S_TT = model._build_S_TT().detach()
    S_TT_shape_ok = tuple(S_TT.shape) == (P, P)

    ok = align_ok and shape_ok and S_TT_shape_ok
    status = "PASS" if ok else "FAIL"
    detail = (
        ", ".join(f"{k}={v:.2e}" for k, v in errs.items())
        + f"; forward_shape={tuple(f.shape)} expected=({B}, {K})"
        + f"; S_TT_shape={tuple(S_TT.shape)} expected=({P}, {P})"
    )
    return status, detail


# ---------------------------------------------------------------------------
# Test 2 - SpectralFilter alone is not an ICL learner (FINDING)
# ---------------------------------------------------------------------------


def test_2_spectral_filter_alone_is_marginal() -> tuple[str, str]:
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    D, N, P, K, L, r = 32, 64, 32, 16, 4, 16
    B = 32
    n_steps = 1000
    lr = 1e-3

    model = SpectralFilter(D, N, P, K, L, r, s_init="zero").to(
        device=DEVICE, dtype=DTYPE
    )
    sampler = make_isotropic_sampler(
        D, P, K, B, sigma=0.0, device=DEVICE, dtype=DTYPE,
    )
    result = train_icl_online(model, sampler, n_steps, lr=lr)
    losses = np.asarray(result["losses"], dtype=np.float64)
    initial = float(losses[0])
    final = float(losses[-1])
    var_y = 1.0
    final_over_var = final / var_y

    # FINDING expected: final should saturate near Var(y). If it doesn't,
    # something surprising happened (flag as PASS with a note).
    if final > 0.8 * var_y:
        status = "FINDING"
        note = (
            "SpectralFilter alone does not achieve meaningful ICL -- no "
            "content-dependent cross-token comparison (query-train block is "
            "all-ones; every query sees the same V_train average)."
        )
    else:
        status = "PASS"
        note = (
            "SpectralFilter learned faster than expected; worth investigating "
            "whether a different mechanism kicked in."
        )

    detail = (
        f"n_steps={n_steps}; initial={initial:.4e}, final={final:.4e}, "
        f"final / Var(y) = {final_over_var:.3f}.  {note}"
    )
    return status, detail


# ---------------------------------------------------------------------------
# Test 3 - Hybrid trains; LinearAttention reference printed
# ---------------------------------------------------------------------------


def test_3_hybrid_trains() -> tuple[str, str]:
    D, N, P, K, L_A, L_S, r = 32, 64, 32, 16, 4, 2, 16
    B = 32
    n_steps = 2000
    lr = 1e-3

    # Hybrid.
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    hybrid = AdaptiveFirstHybrid(D, N, P, K, L_A, L_S, r).to(
        device=DEVICE, dtype=DTYPE
    )
    sampler_h = make_isotropic_sampler(
        D, P, K, B, sigma=0.0, device=DEVICE, dtype=DTYPE,
    )
    result_h = train_icl_online(hybrid, sampler_h, n_steps, lr=lr)
    losses_h = np.asarray(result_h["losses"], dtype=np.float64)
    init_h = float(losses_h[0])
    final_h = float(losses_h[-1])
    ratio_h = final_h / max(init_h, 1e-30)

    # LinearAttention reference at the same seed / sampler.
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    linear = LinearAttention(D, N, P, K, L_A).to(device=DEVICE, dtype=DTYPE)
    sampler_l = make_isotropic_sampler(
        D, P, K, B, sigma=0.0, device=DEVICE, dtype=DTYPE,
    )
    result_l = train_icl_online(linear, sampler_l, n_steps, lr=lr)
    losses_l = np.asarray(result_l["losses"], dtype=np.float64)
    init_l = float(losses_l[0])
    final_l = float(losses_l[-1])
    ratio_l = final_l / max(init_l, 1e-30)

    ok = ratio_h < 0.30
    status = "PASS" if ok else "FAIL"
    detail = (
        f"Hybrid(L_A={L_A}, L_S={L_S}, r={r}): "
        f"init={init_h:.4e}, final={final_h:.4e}, ratio={ratio_h:.3f} "
        f"(< 0.30 required)\n"
        f"      Linear(L={L_A}) reference: "
        f"init={init_l:.4e}, final={final_l:.4e}, ratio={ratio_l:.3f}"
    )
    return status, detail


# ---------------------------------------------------------------------------
# Test 4 - THE CRITICAL TEST: spectral bottleneck binds in the hybrid
# ---------------------------------------------------------------------------


def _train_hybrid_at_r(r: int, n_steps: int, s_init: str = "gd") -> dict:
    D, N, P, K, L_A, L_S = 32, 64, 32, 16, 4, 4
    B = 32
    nu = 1.5
    lr = 1e-3

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = AdaptiveFirstHybrid(D, N, P, K, L_A, L_S, r, s_init=s_init).to(
        device=DEVICE, dtype=DTYPE
    )
    sampler, _ = make_stationary_sampler(
        P=P, D=D, K=K, B=B, symbol_kind="power_law", nu=nu, sigma=0.0,
        device=DEVICE, dtype=DTYPE,
    )
    s_half_init = model.s_half.data.clone().detach().cpu()

    result = train_icl_online(model, sampler, n_steps, lr=lr)
    losses = np.asarray(result["losses"], dtype=np.float64)

    s_half_final = model.s_half.data.clone().detach().cpu()
    s_half_delta = (s_half_final - s_half_init)
    s_half_change_max = float(s_half_delta.abs().max().item())

    # Per-component magnitude of the final s_half, plus a "beyond-DC" norm
    # that equals 0 exactly when only s_half[0] moved.
    s_half_dc_abs = float(s_half_final[0].abs().item()) if s_half_final.numel() > 0 else 0.0
    if s_half_final.numel() > 1:
        beyond_dc_norm = float(s_half_final[1:].norm().item())
    else:
        beyond_dc_norm = 0.0

    return {
        "initial": float(losses[0]),
        "terminal": float(losses[-500:].mean()),
        "s_half_change_max": s_half_change_max,
        "s_half_dc_abs": s_half_dc_abs,
        "s_half_beyond_dc_norm": beyond_dc_norm,
        "r_half": int(model.r_half),
    }


def test_4_bottleneck_binds_in_hybrid() -> tuple[str, str]:
    """Critical test: with the ``S_QT = -S_TT[:K, :]`` design (negate_qt=True)
    and ``s_init="gd"``, higher-frequency ``s_half[k >= 1]`` now has a real
    path to the query readout (different query positions see different rows
    of ``-S_TT``, sampling non-DC modes). The test trains two hybrids at
    ``r = 2`` and ``r = 16`` on stationary power-law data (nu = 1.5) for
    5000 Adam steps with seed 42 and compares terminal losses.

    Gate:
      terminal(r=2) / terminal(r=16) > 1.10   -> PASS (bottleneck binds)
      terminal(r=2) / terminal(r=16) > 1.05   -> WARN (weak effect)
      else                                      -> FAIL

    The diagnostic also prints ``|s_half[1:]|`` to confirm non-DC modes
    actually move (the original ``ones_QT`` design froze them at zero).
    """
    n_steps = 5000
    d2 = _train_hybrid_at_r(r=2, n_steps=n_steps, s_init="gd")
    d16 = _train_hybrid_at_r(r=16, n_steps=n_steps, s_init="gd")

    ratio = d2["terminal"] / max(d16["terminal"], 1e-30)
    if ratio > 1.10:
        status = "PASS"
    elif ratio > 1.05:
        status = "WARN"
    else:
        status = "FAIL"

    detail = (
        f"Hybrid on stationary data (nu=1.5), L_A=4, L_S=4, s_init='gd', "
        f"n_steps={n_steps}, seed=42 for both.\n"
        f"      r=2 (r_half={d2['r_half']}): "
        f"init={d2['initial']:.6e}, terminal={d2['terminal']:.6e}, "
        f"|s_half_delta|_max={d2['s_half_change_max']:.3e}, "
        f"|s_half[0]|={d2['s_half_dc_abs']:.3e}, "
        f"|s_half[1:]|={d2['s_half_beyond_dc_norm']:.3e}\n"
        f"      r=16 (r_half={d16['r_half']}): "
        f"init={d16['initial']:.6e}, terminal={d16['terminal']:.6e}, "
        f"|s_half_delta|_max={d16['s_half_change_max']:.3e}, "
        f"|s_half[0]|={d16['s_half_dc_abs']:.3e}, "
        f"|s_half[1:]|={d16['s_half_beyond_dc_norm']:.3e}\n"
        f"      ratio terminal(r=2) / terminal(r=16) = {ratio:.6f} "
        f"(>1.10 PASS; >1.05 WARN; else FAIL)"
    )
    return status, detail


# ---------------------------------------------------------------------------
# Test 5 - edge-case equivalences (float64)
# ---------------------------------------------------------------------------


def test_5a_LA_zero_matches_spectral_filter() -> tuple[str, str]:
    D, N, P, K, L, r = 16, 32, 16, 8, 4, 8
    B = 4

    # Must use s_init="gd" here: with negate_qt=True and s_init="zero",
    # both models output exactly zero by alignment, and rel_err = 0/0
    # noise is numerically unstable. GD init gives a non-trivial forward
    # output so the equivalence check is meaningful.
    torch.manual_seed(50)
    torch.cuda.manual_seed(50)
    sf = SpectralFilter(D, N, P, K, L, r, s_init="gd").to(
        device=DEVICE, dtype=DTYPE_F64
    )
    sf.enforce_alignment()

    torch.manual_seed(51)
    torch.cuda.manual_seed(51)
    hy = AdaptiveFirstHybrid(
        D, N, P, K, L_A=0, L_S=L, r=r, s_init="gd",
    ).to(device=DEVICE, dtype=DTYPE_F64)

    # Copy shared weights; leave hy.W_q_A, hy.W_k_A as-is (unused at L_A=0).
    with torch.no_grad():
        hy.W_x.data.copy_(sf.W_x.data)
        hy.w_y.data.copy_(sf.w_y.data)
        hy.alpha_v.data.copy_(sf.alpha_v.data)
        hy.s_half.data.copy_(sf.s_half.data)
    hy.enforce_alignment()

    gen = torch.Generator(device=DEVICE).manual_seed(777)
    X_train = torch.randn(B, D, P, generator=gen, dtype=DTYPE_F64, device=DEVICE)
    y_train = torch.randn(B, P, generator=gen, dtype=DTYPE_F64, device=DEVICE)
    X_query = torch.randn(B, D, K, generator=gen, dtype=DTYPE_F64, device=DEVICE)

    with torch.no_grad():
        f_sf = sf(X_train, y_train, X_query)
        f_hy = hy(X_train, y_train, X_query)

    num = float(torch.linalg.vector_norm((f_sf - f_hy).flatten()).item())
    den = float(torch.linalg.vector_norm(f_sf.flatten()).item()) + 1e-30
    rel = num / den
    ok = rel < 1e-10
    status = "PASS" if ok else "FAIL"
    detail = (
        f"SpectralFilter(L={L}, r={r}, s_init='gd') vs "
        f"AdaptiveFirstHybrid(L_A=0, L_S={L}, r={r}, s_init='gd') "
        f"with shared (W_x, w_y, alpha_v, s_half): |f_sf|={den:.3e}, "
        f"rel_err = {rel:.3e} (< 1e-10 required)"
    )
    return status, detail


def test_5b_LS_zero_matches_linear() -> tuple[str, str]:
    D, N, P, K, L, r = 16, 32, 16, 8, 4, 8
    B = 4

    torch.manual_seed(60)
    torch.cuda.manual_seed(60)
    la = LinearAttention(D, N, P, K, L).to(device=DEVICE, dtype=DTYPE_F64)
    la.enforce_alignment()

    torch.manual_seed(61)
    torch.cuda.manual_seed(61)
    hy = AdaptiveFirstHybrid(
        D, N, P, K, L_A=L, L_S=0, r=r, s_init="zero",
    ).to(device=DEVICE, dtype=DTYPE_F64)

    with torch.no_grad():
        hy.W_x.data.copy_(la.W_x.data)
        hy.w_y.data.copy_(la.w_y.data)
        hy.alpha_v.data.copy_(la.alpha_v.data)
        hy.W_q_A.data.copy_(la.W_q.data)
        hy.W_k_A.data.copy_(la.W_k.data)
    hy.enforce_alignment()

    gen = torch.Generator(device=DEVICE).manual_seed(888)
    X_train = torch.randn(B, D, P, generator=gen, dtype=DTYPE_F64, device=DEVICE)
    y_train = torch.randn(B, P, generator=gen, dtype=DTYPE_F64, device=DEVICE)
    X_query = torch.randn(B, D, K, generator=gen, dtype=DTYPE_F64, device=DEVICE)

    with torch.no_grad():
        f_la = la(X_train, y_train, X_query)
        f_hy = hy(X_train, y_train, X_query)

    num = float(torch.linalg.vector_norm((f_la - f_hy).flatten()).item())
    den = float(torch.linalg.vector_norm(f_la.flatten()).item()) + 1e-30
    rel = num / den
    ok = rel < 1e-10
    status = "PASS" if ok else "FAIL"
    detail = (
        f"LinearAttention(L={L}) vs AdaptiveFirstHybrid(L_A={L}, L_S=0, r={r}) "
        f"with shared (W_x, w_y, alpha_v, W_q=W_q_A, W_k=W_k_A): "
        f"rel_err = {rel:.3e} (< 1e-10 required)"
    )
    return status, detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"dtype:  {DTYPE} training / {DTYPE_F64} Test 5")
    tests = [
        ("Test 1 (Hybrid shape + alignment invariants)", test_1_shape_alignment),
        ("Test 2 (SpectralFilter alone is marginal; FINDING)",
         test_2_spectral_filter_alone_is_marginal),
        ("Test 3 (Hybrid trains; Linear reference printed)", test_3_hybrid_trains),
        ("Test 4 (CRITICAL: bottleneck binds in hybrid)",
         test_4_bottleneck_binds_in_hybrid),
        ("Test 5a (L_A=0 hybrid == SpectralFilter, float64)",
         test_5a_LA_zero_matches_spectral_filter),
        ("Test 5b (L_S=0 hybrid == LinearAttention, float64)",
         test_5b_LS_zero_matches_linear),
    ]
    any_fail = False
    for name, fn in tests:
        status, detail = fn()
        if status == "FAIL":
            any_fail = True
        print(f"[{status}] {name}")
        print(f"      {detail}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
