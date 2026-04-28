"""Unit tests for :class:`ICLRegressionModel` and :class:`LinearAttention`.

Run as either::

    python -m scripts.thesis.architectures.tests.test_base_and_linear_attention
    python scripts/thesis/architectures/tests/test_base_and_linear_attention.py

Three checks are performed, all in float64 on CUDA (A100) so that Assumption 1
alignment and the Theorem A exact reduction can be asserted at machine
precision:

- **Test 1** - shape + alignment: after construction and
  :meth:`enforce_alignment`, all four alignment invariants hold to < 1e-12,
  and the forward pass returns ``(B, K)``.
- **Test 2** - alignment preservation: run 100 SGD steps on random ICL data,
  call :meth:`enforce_alignment` after each step, and verify the alignment
  invariants stay below 1e-10 across all steps.
- **Test 3** - exact match to Theorem A reduced-Gamma closed form: for four
  geometries ``(D, P, K, L) in {(8,8,4,1), (16,16,8,2), (32,32,16,4),
  (64,64,32,8)}``, ``model(X_train, y_train, X_query)`` equals the reduced
  prediction with ``Gamma = alpha_v * W_x^T W_q^T W_k W_x`` to relative
  error < 1e-10.

CUDA is required; there is no CPU fallback.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[4]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import torch

from scripts.thesis.architectures.models import ICLRegressionModel, LinearAttention  # noqa: E402


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this test — no CPU fallback. "
        "Run on a GPU node (A100 login node on the thesis cluster)."
    )

DEVICE = torch.device("cuda")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_errors(model: LinearAttention) -> dict[str, float]:
    """Return the four Assumption 1 violation magnitudes."""
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
    """Random column-sample ICL batch with sqrt(D) label normalization (CUDA, float64)."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    X_train = torch.randn(B, D, P, generator=gen, dtype=DTYPE, device=DEVICE)
    X_query = torch.randn(B, D, K, generator=gen, dtype=DTYPE, device=DEVICE)
    beta = torch.randn(B, D, generator=gen, dtype=DTYPE, device=DEVICE)
    y_train = torch.einsum("bd,bdp->bp", beta, X_train) / math.sqrt(D)
    y_query = torch.einsum("bd,bdk->bk", beta, X_query) / math.sqrt(D)
    return X_train, y_train, X_query, y_query


def _build_model(D: int, N: int, P: int, K: int, L: int) -> LinearAttention:
    """Construct on CPU then move to CUDA/float64 and re-enforce alignment."""
    model = LinearAttention(D, N, P, K, L).to(device=DEVICE, dtype=DTYPE)
    model.enforce_alignment()
    return model


def _reduced_gamma_forward(
    model: LinearAttention,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_query: torch.Tensor,
) -> torch.Tensor:
    """Theorem A reduced-Gamma closed-form query prediction for the current
    model weights:

        Gamma = alpha_v * W_x^T W_q^T W_k W_x                    in R^{D x D}
        K_train[b] = (1/LP) X_train[b]^T Gamma X_train[b]        in R^{P x P}
        T[b]       = I_P - K_train[b]                            in R^{P x P}
        acc[b]     = sum_{l=0..L-1} T[b]^l @ y_train[b]          in R^{P}
        f_red[b]   = (1/LP) (X_query[b]^T Gamma X_train[b]) @ acc[b]
    """
    P, L = model.P, model.L
    with torch.no_grad():
        wx = model.W_x.data
        wq = model.W_q.data
        wk = model.W_k.data
        Gamma = model.alpha_v.data * (wx.T @ wq.T @ wk @ wx)

        K_train = torch.einsum("bdp,de,beq->bpq", X_train, Gamma, X_train) / (L * P)
        K_xq = torch.einsum("bdk,de,bep->bkp", X_query, Gamma, X_train)
        B = X_train.shape[0]
        eye = torch.eye(P, dtype=X_train.dtype, device=X_train.device).expand(B, P, P)
        T = eye - K_train

        r = y_train.clone()
        acc = torch.zeros_like(r)
        for _ in range(L):
            acc = acc + r
            r = torch.einsum("bpq,bq->bp", T, r)
        return torch.einsum("bkp,bp->bk", K_xq, acc) / (L * P)


# ---------------------------------------------------------------------------
# Test 1 - shape + alignment invariants
# ---------------------------------------------------------------------------


def test_1_shape_and_alignment() -> tuple[bool, str]:
    torch.manual_seed(0)
    D, N, P, K, L = 16, 32, 16, 8, 4
    B = 4

    model = _build_model(D, N, P, K, L)

    errs = _align_errors(model)
    tol_align = 1e-12
    align_ok = all(v < tol_align for v in errs.values())

    gen = torch.Generator(device=DEVICE).manual_seed(123)
    X_train = torch.randn(B, D, P, generator=gen, dtype=DTYPE, device=DEVICE)
    X_query = torch.randn(B, D, K, generator=gen, dtype=DTYPE, device=DEVICE)
    y_train = torch.randn(B, P, generator=gen, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        f = model(X_train, y_train, X_query)
    shape_ok = tuple(f.shape) == (B, K)

    ok = align_ok and shape_ok
    parts = [f"{k}={v:.2e}" for k, v in errs.items()]
    parts.append(f"forward_shape={tuple(f.shape)} expected=({B},{K})")
    return ok, ", ".join(parts)


# ---------------------------------------------------------------------------
# Test 2 - alignment preservation under SGD
# ---------------------------------------------------------------------------


def test_2_alignment_preservation() -> tuple[bool, str]:
    torch.manual_seed(1)
    D, N, P, K, L = 16, 32, 16, 8, 4
    B = 32
    n_steps = 100

    model = _build_model(D, N, P, K, L)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    max_errs = {"Wx_T_wy": 0.0, "wy_norm_minus_1": 0.0, "Wq_wy": 0.0, "Wk_wy": 0.0}
    for step in range(n_steps):
        X_train, y_train, X_query, y_query = _sample_icl_batch(
            B, D, P, K, seed=1000 + step,
        )
        opt.zero_grad()
        f = model(X_train, y_train, X_query)
        loss = model.loss(f, y_query)
        loss.backward()
        opt.step()
        model.enforce_alignment()
        e = _align_errors(model)
        for key, val in e.items():
            if val > max_errs[key]:
                max_errs[key] = val

    tol = 1e-10
    ok = all(v < tol for v in max_errs.values())
    return ok, ", ".join(f"{k}_max={v:.2e}" for k, v in max_errs.items())


# ---------------------------------------------------------------------------
# Test 3 - exact match to Theorem A reduced-Gamma closed form
# ---------------------------------------------------------------------------


def test_3_exact_match_reduced() -> tuple[bool, str]:
    torch.manual_seed(2)
    geometries = [(8, 8, 4, 1), (16, 16, 8, 2), (32, 32, 16, 4), (64, 64, 32, 8)]
    B = 4
    tol = 1e-10

    all_ok = True
    lines: list[str] = []
    for (D, P, K, L) in geometries:
        N = 2 * D
        model = _build_model(D, N, P, K, L)

        X_train, y_train, X_query, _ = _sample_icl_batch(
            B, D, P, K, seed=2000 + 37 * D + 11 * P,
        )

        with torch.no_grad():
            f_full = model(X_train, y_train, X_query)
            f_red = _reduced_gamma_forward(model, X_train, y_train, X_query)

        diff = torch.linalg.vector_norm((f_full - f_red).flatten()).item()
        denom = torch.linalg.vector_norm(f_full.flatten()).item() + 1e-30
        rel = diff / denom
        ok = rel < tol
        all_ok = all_ok and ok
        tag = "" if ok else " <-- FAIL"
        lines.append(f"(D={D:>2},P={P:>2},K={K:>2},L={L}): rel_err={rel:.3e}{tag}")
    return all_ok, "\n      ".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"dtype:  {DTYPE}")
    tests = [
        ("Test 1 (shape + alignment after enforce_alignment)", test_1_shape_and_alignment),
        ("Test 2 (alignment preserved under 100 SGD steps)", test_2_alignment_preservation),
        ("Test 3 (full model == Theorem A reduced-Gamma)", test_3_exact_match_reduced),
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
