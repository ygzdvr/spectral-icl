"""Smoke tests for the shared online training loop.

Run as either::

    python -m scripts.thesis.architectures.tests.test_training_smoke
    python scripts/thesis/architectures/tests/test_training_smoke.py

All tests on CUDA, float32 (training precision, not algebraic float64). Three
checks:

- **Test 1** - :class:`LinearAttention` trains: 500 Adam steps on the
  isotropic sampler drive loss below 0.5 x initial; Assumption 1 alignment
  stays below 1e-5.
- **Test 2** - :class:`SpectralAttention` (``s_init="gd"``, ``r = P``) trains:
  same cadence, loss below 0.5 x initial, alignment below 1e-5; initial loss
  is within a factor of 2 of :class:`LinearAttention`'s initial loss (GD init
  makes the circulant mask identical to ``M_GD``).
- **Test 3** - :class:`SpectralAttention` (``s_init="zero"``, ``r = 8 < P``)
  trains: loss below 0.8 x initial, alignment below 1e-5, and
  ``max |s_half_after - s_half_before|`` exceeds 1e-3 (spectral parameters
  learn through the differentiable DFT path).

CUDA is required; there is no CPU fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[4]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import torch

from scripts.thesis.architectures.models import LinearAttention, SpectralAttention  # noqa: E402
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

# Shared smoke-test geometry. These values are deliberately small so the test
# finishes in a few seconds on an A100 while still giving Adam room to move
# the loss by the required factors.
D, N, P, K, L = 32, 64, 32, 16, 4
B = 32
N_STEPS = 500
LR = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_errors(model: LinearAttention | SpectralAttention) -> dict[str, float]:
    with torch.no_grad():
        wy = model.w_y.data
        return {
            "Wx_T_wy": float(torch.linalg.vector_norm(model.W_x.data.T @ wy)),
            "wy_norm_minus_1": abs(float(wy.norm()) - 1.0),
            "Wq_wy": float(torch.linalg.vector_norm(model.W_q.data @ wy)),
            "Wk_wy": float(torch.linalg.vector_norm(model.W_k.data @ wy)),
        }


def _new_sampler():
    return make_isotropic_sampler(D, P, K, B, sigma=0.0, device=DEVICE, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Test 1 - LinearAttention trains, loss decreases, alignment holds
# ---------------------------------------------------------------------------


def test_1_linear_trains() -> tuple[bool, str, float]:
    torch.manual_seed(0)

    model = LinearAttention(D, N, P, K, L).to(device=DEVICE, dtype=DTYPE)
    model.enforce_alignment()

    result = train_icl_online(model, _new_sampler(), N_STEPS, lr=LR)
    losses = result["losses"]
    initial, final = losses[0], losses[-1]
    ratio = final / initial if initial > 0 else float("inf")
    loss_ok = final < 0.5 * initial

    errs = _align_errors(model)
    max_err = max(errs.values())
    align_ok = max_err < 1e-5

    ok = loss_ok and align_ok
    detail = (
        f"initial={initial:.4e}, final={final:.4e}, ratio={ratio:.3f} (<0.5 required), "
        f"max_align_err={max_err:.2e} (<1e-5 required)"
    )
    return ok, detail, initial


# ---------------------------------------------------------------------------
# Test 2 - SpectralAttention (s_init="gd", r=P) trains
# ---------------------------------------------------------------------------


def test_2_spectral_gd_trains(linear_initial_loss: float) -> tuple[bool, str]:
    torch.manual_seed(1)

    model = SpectralAttention(D, N, P, K, L, r=P, s_init="gd").to(
        device=DEVICE, dtype=DTYPE
    )
    model.enforce_alignment()

    result = train_icl_online(model, _new_sampler(), N_STEPS, lr=LR)
    losses = result["losses"]
    initial, final = losses[0], losses[-1]
    ratio = final / initial if initial > 0 else float("inf")
    loss_ok = final < 0.5 * initial

    errs = _align_errors(model)
    max_err = max(errs.values())
    align_ok = max_err < 1e-5

    lo, hi = 0.5 * linear_initial_loss, 2.0 * linear_initial_loss
    within_2x = lo <= initial <= hi

    ok = loss_ok and align_ok and within_2x
    detail = (
        f"initial={initial:.4e} (linear={linear_initial_loss:.4e}, "
        f"within 2x: {within_2x}), final={final:.4e}, ratio={ratio:.3f} "
        f"(<0.5 required), max_align_err={max_err:.2e} (<1e-5 required)"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Test 3 - SpectralAttention (s_init="zero", r=8 < P) trains + s_half changes
# ---------------------------------------------------------------------------


def test_3_spectral_zero_bottleneck_trains() -> tuple[bool, str]:
    torch.manual_seed(2)
    r = 8

    model = SpectralAttention(D, N, P, K, L, r=r, s_init="zero").to(
        device=DEVICE, dtype=DTYPE
    )
    model.enforce_alignment()

    s_before = model.s_half.data.clone()

    result = train_icl_online(model, _new_sampler(), N_STEPS, lr=LR)
    losses = result["losses"]
    initial, final = losses[0], losses[-1]
    ratio = final / initial if initial > 0 else float("inf")
    loss_ok = final < 0.8 * initial

    errs = _align_errors(model)
    max_err = max(errs.values())
    align_ok = max_err < 1e-5

    s_change = float((model.s_half.data - s_before).abs().max().item())
    s_ok = s_change > 1e-3

    ok = loss_ok and align_ok and s_ok
    detail = (
        f"r={r}, initial={initial:.4e}, final={final:.4e}, ratio={ratio:.3f} "
        f"(<0.8 required), max|s_half_delta|={s_change:.2e} (>1e-3 required), "
        f"max_align_err={max_err:.2e} (<1e-5 required)"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"dtype:  {DTYPE}")
    print(f"config: D={D}, N={N}, P={P}, K={K}, L={L}, B={B}, n_steps={N_STEPS}, lr={LR}")

    all_pass = True

    ok1, detail1, linear_initial = test_1_linear_trains()
    tag = "PASS" if ok1 else "FAIL"
    all_pass = all_pass and ok1
    print(f"[{tag}] Test 1 (LinearAttention trains, loss decreases)")
    print(f"      {detail1}")

    ok2, detail2 = test_2_spectral_gd_trains(linear_initial)
    tag = "PASS" if ok2 else "FAIL"
    all_pass = all_pass and ok2
    print(f"[{tag}] Test 2 (SpectralAttention s_init='gd', r=P trains)")
    print(f"      {detail2}")

    ok3, detail3 = test_3_spectral_zero_bottleneck_trains()
    tag = "PASS" if ok3 else "FAIL"
    all_pass = all_pass and ok3
    print(f"[{tag}] Test 3 (SpectralAttention s_init='zero', r=8<P trains + s_half changes)")
    print(f"      {detail3}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
