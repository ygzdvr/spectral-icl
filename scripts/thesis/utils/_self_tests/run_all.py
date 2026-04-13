"""Self-test harness for ``scripts/thesis/utils/``.

Plan correspondence: Step-1 Generator / Utility Specification v4 §12.

Runs the full v4 §12 test suite, separated into two categories:

- **Exact algebraic tests** — must fail hard. A single failure anywhere in
  this category causes the driver to exit nonzero.
- **Monte Carlo tolerance tests** — report the observed metric, the tolerance
  used, and the sample size; pass / fail is reported but a failure here does
  NOT fail the gate (statistical fluctuations are expected).

Usage (from project root, with the repo venv active)::

    python -u scripts/thesis/utils/_self_tests/run_all.py

Exit code: ``0`` if every exact test passed; ``1`` if any exact test failed
(MC failures do not affect exit status, but are reported in the summary).

Phase-by-phase output shows a check mark / cross, the category, the test
name, and the elapsed time. MC tests additionally print the observed metric,
tolerance, and sample size.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# --- Path setup: make the project root importable. ---
_HERE = Path(__file__).resolve()
_PROJ = _HERE.parents[4]  # scripts/thesis/utils/_self_tests/ -> repo root
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import torch

# Module-level imports (fail fast if any Step-1b file is missing).
from scripts.thesis.utils import (  # noqa: F401  (imported for the audit test)
    commutants,
    cost_models,
    data_generators,
    fit_powerlaws,
    fourier_ops,
    metrics,
    partitions,
)
from scripts.thesis.utils.commutants import (
    commutant_projection,
    commutant_violation,
    extract_block_scalars,
    reconstruct_from_block_scalars,
    refines,
)
from scripts.thesis.utils.cost_models import (
    calibrate,
    compute_proxy,
    phi_adaptive,
    phi_spectral_fft,
)
from scripts.thesis.utils.data_generators import (
    G1Config,
    G2Config,
    G3Config,
    GAConfig,
    g1_generate,
    g2_generate_operator,
    g2_generate_sampled,
    g2_to_spectral_basis,
    g3_generate,
    ga_generate,
)
from scripts.thesis.utils.fit_powerlaws import (
    bootstrap_exponent,
    fit_loglog,
    holdout_evaluate,
)
from scripts.thesis.utils.fourier_ops import (
    circulant_from_symbol,
    dft_matrix,
    frequency_permutation,
    off_diagonal_fourier_energy,
    real_spectral_basis,
    symbol_flat,
    symbol_of_circulant,
    symbol_power_law,
)
from scripts.thesis.utils.metrics import (
    ab_perturbation_bound,
    contraction_depth_overlay,
    gamma_star_trajectory_circulant,
    oracle_commutant_loss,
    reduced_model_error,
)
from scripts.thesis.utils.partitions import (
    BlockPartition,
    dyadic_ladder,
    equal_blocks,
    mass_preserving_block_spectrum,
)


# ---------------------------------------------------------------------------
# Result container + test drivers
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    name: str
    phase: str
    category: str  # "exact" | "monte_carlo"
    passed: bool
    detail: str
    elapsed: float = 0.0
    metric: float | None = None
    tolerance: float | None = None
    sample_size: int | None = None


def run_exact(phase: str, name: str, fn: Callable[[], str]) -> TestResult:
    """Run an exact algebraic test.

    ``fn()`` should return a one-line detail string on success (for logging),
    or raise ``AssertionError`` on failure.
    """
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        passed = True
    except AssertionError as e:
        detail = f"FAIL: {e}"
        passed = False
    except Exception as e:
        detail = f"ERROR: {type(e).__name__}: {e}"
        passed = False
    return TestResult(
        name=name,
        phase=phase,
        category="exact",
        passed=passed,
        detail=detail,
        elapsed=time.perf_counter() - t0,
    )


def run_mc(
    phase: str, name: str, fn: Callable[[], tuple[bool, str, float, float, int]]
) -> TestResult:
    """Run a Monte Carlo tolerance test.

    ``fn()`` must return ``(passed, detail, metric, tolerance, sample_size)``.
    A False ``passed`` is reported but does NOT fail the gate.
    """
    t0 = time.perf_counter()
    try:
        passed, detail, metric, tol, n = fn()
    except Exception as e:
        return TestResult(
            name=name,
            phase=phase,
            category="monte_carlo",
            passed=False,
            detail=f"ERROR: {type(e).__name__}: {e}",
            elapsed=time.perf_counter() - t0,
        )
    return TestResult(
        name=name,
        phase=phase,
        category="monte_carlo",
        passed=passed,
        detail=detail,
        elapsed=time.perf_counter() - t0,
        metric=metric,
        tolerance=tol,
        sample_size=n,
    )


def print_phase_results(phase: str, results: list[TestResult]) -> None:
    print(f"\n--- {phase} ---")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        cat = "exact" if r.category == "exact" else "  MC "
        mc_info = ""
        if r.category == "monte_carlo" and r.metric is not None:
            mc_info = (
                f"  [metric={r.metric:.3e}, tol={r.tolerance:.3e}, n={r.sample_size}]"
            )
        print(
            f"  [{mark}] [{cat}] {r.name:58s}  ({r.elapsed * 1000:7.1f} ms){mc_info}"
        )
        if r.detail:
            print(f"         {r.detail}")


def print_summary(all_results: list[tuple[str, list[TestResult]]]) -> None:
    total_exact = 0
    pass_exact = 0
    total_mc = 0
    pass_mc = 0
    total_time = 0.0
    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    for phase, results in all_results:
        e = [r for r in results if r.category == "exact"]
        m = [r for r in results if r.category == "monte_carlo"]
        ep = sum(1 for r in e if r.passed)
        mp = sum(1 for r in m if r.passed)
        t = sum(r.elapsed for r in results)
        total_exact += len(e)
        pass_exact += ep
        total_mc += len(m)
        pass_mc += mp
        total_time += t
        print(
            f"  {phase:50s}  exact: {ep}/{len(e):2d}  MC: {mp}/{len(m):2d}"
            f"  ({t:6.2f} s)"
        )
    print("-" * 78)
    print(
        f"  TOTAL{'':45s}  exact: {pass_exact}/{total_exact:2d}"
        f"  MC: {pass_mc}/{total_mc:2d}  ({total_time:6.2f} s)"
    )

    exact_failures = [
        r
        for _, results in all_results
        for r in results
        if r.category == "exact" and not r.passed
    ]
    mc_failures = [
        r
        for _, results in all_results
        for r in results
        if r.category == "monte_carlo" and not r.passed
    ]
    if exact_failures:
        print(f"\n  EXACT FAILURES ({len(exact_failures)}):")
        for r in exact_failures:
            print(f"    [{r.phase}] {r.name}")
            print(f"        {r.detail}")
    if mc_failures:
        print(f"\n  MC FAILURES ({len(mc_failures)}, non-blocking):")
        for r in mc_failures:
            print(f"    [{r.phase}] {r.name}")
            print(f"        {r.detail}")
    if not exact_failures and not mc_failures:
        print("\n  All checks passed.")


# ---------------------------------------------------------------------------
# Phase 1 — fourier_ops (v4 §12.2)
# ---------------------------------------------------------------------------


def phase_fourier_ops() -> list[TestResult]:
    phase = "§12.2"
    results: list[TestResult] = []

    def t1() -> str:
        F = dft_matrix(8)
        err = (
            F @ F.conj().T - torch.eye(8, dtype=torch.complex128)
        ).abs().max().item()
        assert err < 1e-12, f"||F F^H - I||_inf = {err:.2e}"
        return f"||F F^H - I||_inf = {err:.2e}"

    results.append(run_exact(phase, "unitary_dft is unitary", t1))

    def t2() -> str:
        s = symbol_power_law(32, nu=1.5)
        C = circulant_from_symbol(s)
        assert C.dtype == torch.float64 and C.shape == (32, 32)
        return f"circulant_from_symbol dtype={C.dtype}, shape={tuple(C.shape)}"

    results.append(
        run_exact(phase, "circulant_from_symbol returns real (imag-leak asserted)", t2)
    )

    def t3() -> str:
        s = symbol_power_law(32, nu=1.5)
        C = circulant_from_symbol(s)
        s_back = symbol_of_circulant(C)
        err = (s - s_back).abs().max().item()
        assert err < 1e-12, f"err = {err:.2e}"
        return f"max|s - s_back|_inf = {err:.2e}"

    results.append(
        run_exact(phase, "symbol_of_circulant ∘ circulant_from_symbol == id", t3)
    )

    def t4() -> str:
        s = symbol_power_law(32, nu=1.5)
        sp = frequency_permutation(s, seed=7)
        P = sp.shape[0]
        err = 0.0
        for k in range(1, (P + 1) // 2):
            err = max(err, abs(sp[k].item() - sp[P - k].item()))
        assert err < 1e-12, f"err = {err:.2e}"
        return f"max|s'[k] - s'[P-k]| = {err:.2e}"

    results.append(
        run_exact(phase, "frequency_permutation preserves real-even", t4)
    )

    def t5a() -> str:
        s = symbol_power_law(32, nu=1.5)
        C = circulant_from_symbol(s)
        e = off_diagonal_fourier_energy(C)
        assert e < 1e-12, f"e = {e:.2e}"
        return f"off-diagonal fourier energy = {e:.2e}"

    results.append(
        run_exact(phase, "off_diagonal_fourier_energy(circulant) < 1e-12", t5a)
    )

    def t5b() -> str:
        torch.manual_seed(0)
        M = torch.randn(32, 32, dtype=torch.float64)
        e = off_diagonal_fourier_energy(M)
        assert e > 0.1, f"e = {e:.2e}"
        return f"off-diagonal fourier energy (random) = {e:.3f}"

    results.append(
        run_exact(phase, "off_diagonal_fourier_energy(random) > 0.1", t5b)
    )

    def t6() -> str:
        R = real_spectral_basis(16, kind="dct2")
        err = (R @ R.T - torch.eye(16, dtype=torch.float64)).abs().max().item()
        assert err < 1e-12, f"err = {err:.2e}"
        return f"||R R^T - I||_inf = {err:.2e}"

    results.append(
        run_exact(phase, "real_spectral_basis(D, 'dct2') orthogonal", t6)
    )

    return results


# ---------------------------------------------------------------------------
# Phase 2 — partitions / commutants (v4 §12.3)
# ---------------------------------------------------------------------------


def phase_partitions_commutants() -> list[TestResult]:
    phase = "§12.3"
    results: list[TestResult] = []

    def t1() -> str:
        ladder = dyadic_ladder(16)
        for j in range(len(ladder) - 1):
            assert refines(
                ladder[j + 1], ladder[j]
            ), f"level {j+1} fails to refine level {j}"
        return f"dyadic_ladder(16) = {len(ladder)} levels, all refine"

    results.append(
        run_exact(phase, "dyadic_ladder(16) adjacent refinements valid", t1)
    )

    def t2() -> str:
        p = equal_blocks(8, 4)
        means = torch.tensor([3.0, 5.0], dtype=torch.float64)
        worst = 0.0
        for kappa in [1.0, 2.0, 5.0, 10.0, 50.0]:
            kappas = torch.tensor([kappa, kappa], dtype=torch.float64)
            lam = mass_preserving_block_spectrum(p, means, kappas)
            worst = max(worst, abs(lam[:4].mean().item() - 3.0))
            worst = max(worst, abs(lam[4:].mean().item() - 5.0))
        assert worst < 1e-12, f"worst = {worst:.2e}"
        return f"mass preserved at κ ∈ {{1, 2, 5, 10, 50}}; max err = {worst:.2e}"

    results.append(run_exact(phase, "mass-preserving κ sweep", t2))

    def t3() -> str:
        torch.manual_seed(0)
        Q = torch.randn(12, 12, dtype=torch.float64)
        p = equal_blocks(12, 3)
        proj1 = commutant_projection(Q, p)
        proj2 = commutant_projection(proj1, p)
        err = (proj1 - proj2).norm().item()
        assert err < 1e-12, f"err = {err:.2e}"
        return f"||Π(Π(Q)) - Π(Q)||_F = {err:.2e}"

    results.append(run_exact(phase, "commutant_projection idempotent", t3))

    def t4() -> str:
        p = equal_blocks(12, 3)
        torch.manual_seed(1)
        q = torch.randn(p.n_blocks, dtype=torch.float64)
        Q = reconstruct_from_block_scalars(q, p)
        viol = commutant_violation(Q, p)
        assert viol < 1e-12, f"viol = {viol:.2e}"
        return f"violation(Σ q_b Π_b) = {viol:.2e}"

    results.append(
        run_exact(phase, "commutant_violation(Σ q_b Π_b) < 1e-12", t4)
    )

    def t5() -> str:
        p = equal_blocks(12, 3)
        q = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float64)
        Q = reconstruct_from_block_scalars(q, p)
        q_back = extract_block_scalars(Q, p)
        err = (q_back - q).abs().max().item()
        assert err < 1e-12, f"err = {err:.2e}"
        return f"max|err| = {err:.2e}"

    results.append(
        run_exact(phase, "extract ∘ reconstruct = id on block-scalar vectors", t5)
    )

    return results


# ---------------------------------------------------------------------------
# Phase 3 — GA (v4 §12.4)
# ---------------------------------------------------------------------------


def phase_ga() -> list[TestResult]:
    phase = "§12.4"
    results: list[TestResult] = []

    def t1() -> str:
        cfg = GAConfig(D=4, P=6, K=2)
        r = ga_generate(cfg)
        M = r["mask"]
        assert (M[:6, :6] == -1.0).all()
        assert (M[6:, :6] == 1.0).all()
        assert (M[:, 6:] == 0.0).all()
        return "top-left = -1, bottom-left = +1, right block = 0 (bitwise)"

    results.append(run_exact(phase, "default mask has exact GD structure", t1))

    def t2() -> str:
        seeds = {"x": 1, "beta": 2, "noise": 3, "mask": 4}
        cfg_p = GAConfig(
            D=4,
            P=6,
            K=2,
            B=1,
            mask_kind="perturbed",
            mask_perturbation={"theta": 0.0, "pattern_seed": 42},
            seeds=seeds,
        )
        cfg_g = GAConfig(D=4, P=6, K=2, B=1, seeds=seeds)
        r_p = ga_generate(cfg_p)
        r_g = ga_generate(cfg_g)
        m_err = (r_p["mask"] - r_g["mask"]).abs().max().item()
        a_err = (r_p["A_S_theta"] - r_g["A_S_GD"]).abs().max().item()
        assert m_err < 1e-12 and a_err < 1e-12
        return f"mask err = {m_err:.2e}, A_S err = {a_err:.2e}"

    results.append(run_exact(phase, "perturbed(θ=0) equals GD-compatible", t2))

    def t3() -> str:
        cfg1 = GAConfig(
            D=4, P=6, K=2, mask_kind="non_gd_control", non_gd_kind="signflip_testtest"
        )
        r1 = ga_generate(cfg1)
        assert (r1["mask"][6:, :6] == -1.0).all(), "signflip: bottom-left not -1"
        cfg2 = GAConfig(
            D=4, P=6, K=2, mask_kind="non_gd_control", non_gd_kind="nonzero_testblock"
        )
        r2 = ga_generate(cfg2)
        assert (r2["mask"][:6, 6:] != 0.0).any(), "nonzero_testblock: right zero"
        return "signflip + nonzero-right-block both violate GD structure"

    results.append(run_exact(phase, "non-GD control violates GD structure", t3))

    def t4() -> str:
        cfg = GAConfig(D=8, P=12, K=4, B=3)
        r = ga_generate(cfg)
        assert r["X_train"].shape == (3, 8, 12)
        assert r["X_query"].shape == (3, 8, 4)
        assert r["A_S_GD"].shape == (3, 12, 12)
        assert r["B_S_GD"].shape == (3, 4, 12)
        assert r["T_GD"].shape == (3, 12, 12)
        return "(B,D,P), (B,D,K), (B,P,P), (B,K,P), (B,P,P) — column-sample"

    results.append(run_exact(phase, "shapes column-sample convention", t4))

    def t5() -> str:
        # v4 §12.4(5) reference case: D=2, P=3, K=1, non-symmetric Γ = [[1,2],[3,4]]
        D, P, K = 2, 3, 1
        Gamma = [[1.0, 2.0], [3.0, 4.0]]
        cfg = GAConfig(
            D=D,
            P=P,
            K=K,
            B=1,
            Gamma_kind="full_matrix",
            Gamma_params={"matrix": Gamma},
            seeds={"x": 11, "beta": 12, "noise": 13, "mask": 14},
        )
        r = ga_generate(cfg)
        X_tr = r["X_train"][0]
        X_q = r["X_query"][0]
        G = torch.tensor(Gamma, dtype=torch.float64)
        A_mat = -(1.0 / P) * X_tr.T @ G @ X_tr
        A_elem = torch.zeros(P, P, dtype=torch.float64)
        for mu in range(P):
            for nu in range(P):
                A_elem[mu, nu] = -(1.0 / P) * X_tr[:, mu] @ G @ X_tr[:, nu]
        e1 = (A_mat - A_elem).abs().max().item()
        e2 = (r["A_S_GD"][0] - A_mat).abs().max().item()
        B_mat = (1.0 / P) * X_q.T @ G @ X_tr
        e3 = (r["B_S_GD"][0] - B_mat).abs().max().item()
        assert max(e1, e2, e3) < 1e-12
        return f"max(matrix/elementwise/A_S/B_S err) = {max(e1, e2, e3):.2e}"

    results.append(
        run_exact(phase, "A_S/B_S matrix vs elementwise agree bitwise", t5)
    )

    def t6() -> str:
        cfg = GAConfig(
            D=8,
            P=16,
            K=4,
            B=1,
            L=2,
            sigma=0.0,
            seeds={"x": 100, "beta": 200, "noise": 300, "mask": 400},
        )
        r = ga_generate(cfg)
        L_int = cfg.L
        B_S = r["B_S_GD"][0]
        T = r["T_GD"][0]
        y = r["y_train"][0]
        S = sum(T.matrix_power(ell) for ell in range(L_int))
        f_red = (1.0 / L_int) * B_S @ S @ y
        assert f_red.shape == (cfg.K,)
        err = reduced_model_error(f_red, f_red)
        assert err == 0.0
        return f"f_red shape = (K,) = ({cfg.K},); self-error = 0"

    results.append(
        run_exact(phase, "reduced forward shape + self-consistency", t6)
    )

    def t7() -> str:
        cfg = GAConfig(D=4, P=6, K=2, return_feature_space=True)
        r = ga_generate(cfg)
        for k in ["A_feat_GD", "B_feat_GD", "A_feat_theta", "B_feat_theta"]:
            assert k in r and r[k].shape == (4, 4)
        cfg2 = GAConfig(D=4, P=6, K=2, return_feature_space=False)
        r2 = ga_generate(cfg2)
        assert "A_feat_GD" not in r2
        return "feature-space keys present iff return_feature_space=True"

    results.append(
        run_exact(phase, "return_feature_space gates (D,D) helpers", t7)
    )

    return results


# ---------------------------------------------------------------------------
# Phase 4 — G1 (v4 §12.5)
# ---------------------------------------------------------------------------


def phase_g1() -> list[TestResult]:
    phase = "§12.5"
    results: list[TestResult] = []

    def t1() -> str:
        cfg = G1Config(P=16, B=1, sample_data=True)
        r = g1_generate(cfg)
        forbidden = ("t_star", "trajectory", "gamma_star", "transfer")
        for k, v in r.items():
            assert not callable(v), f"key {k!r} is callable"
            if isinstance(v, torch.Tensor):
                assert v.dtype == torch.float64, f"{k!r}: dtype {v.dtype}"
            for tok in forbidden:
                assert tok not in k.lower(), f"forbidden key substring {tok!r} in {k!r}"
        return "no callables, no trajectory / transfer keys, float64 only"

    results.append(
        run_exact(phase, "no callables, no trajectory outputs", t1)
    )

    def t2() -> str:
        cfg = G1Config(P=32, symbol_kind_te="matched")
        r = g1_generate(cfg)
        assert torch.equal(r["s_tr"], r["s_te"])
        return "s_te == s_tr bitwise (matched)"

    results.append(run_exact(phase, "matched → s_te bitwise == s_tr", t2))

    def t3() -> str:
        cfg_fw = G1Config(P=16, B=2, query_mode="full_window", sample_data=True)
        r_fw = g1_generate(cfg_fw)
        assert r_fw["X_query"].shape == (2, 16, 16)
        assert r_fw["y_query"].shape == (2, 16)
        cfg_sq = G1Config(P=16, B=2, query_mode="single_query", sample_data=True)
        r_sq = g1_generate(cfg_sq)
        assert r_sq["X_query"].shape == (2, 16, 1)
        assert r_sq["y_query"].shape == (2, 1)
        return "full_window: K=P=16; single_query: K=1"

    results.append(run_exact(phase, "query-mode shapes", t3))

    def t4() -> str:
        cfg = G1Config(P=16, B=1, sample_data=True)  # default = independent
        r = g1_generate(cfg)
        diff = (r["X_train"] - r["X_query"]).abs().max().item()
        assert diff > 0.01, f"matched-independent but X_query == X_train"
        return f"max|X_query - X_train| = {diff:.3f} (>> 0)"

    results.append(
        run_exact(phase, "matched-independent default: X_query independent", t4)
    )

    def t5() -> str:
        cfg = G1Config(
            P=16, B=1, matched_query_realization="shared", sample_data=True
        )
        r = g1_generate(cfg)
        assert torch.equal(r["X_train"], r["X_query"])
        return "X_query == X_train bitwise under shared"

    results.append(run_exact(phase, "shared sanity control: bitwise equality", t5))

    def t6() -> str:
        cfg = G1Config(P=16)
        r = g1_generate(cfg)
        s_back = symbol_of_circulant(r["Sigma_tr"])
        err = (s_back - r["s_tr"]).abs().max().item()
        assert err < 1e-12, f"err = {err:.2e}"
        return f"max|symbol_of_circulant(Σ_tr) - s_tr| = {err:.2e}"

    results.append(run_exact(phase, "circulant identity small P", t6))

    def t7() -> str:
        # OOD mode: s_te differs from s_tr
        cfg = G1Config(
            P=32,
            symbol_kind_te="interpolate",
            symbol_params_te={"other_kind": "flat", "alpha": 0.3},
        )
        r = g1_generate(cfg)
        assert not torch.equal(r["s_tr"], r["s_te"])
        return "interpolate mode: s_te differs from s_tr"

    results.append(run_exact(phase, "three distinct spectra (OOD mode)", t7))

    # MC: covariance identity. Reduced footprint (B*P = 65536) for harness
    # execution on the login node; the full §12.5(6) spec calls for P=64, B=4096.
    def tmc() -> tuple[bool, str, float, float, int]:
        P = 32
        B = 2048
        cfg = G1Config(
            P=P,
            B=B,
            sample_data=True,
            seeds={"x_tr": 123, "x_te": 456, "beta": 789, "noise": 111},
        )
        r = g1_generate(cfg)
        Sigma_emp = (
            torch.einsum("bdp,bep->de", r["X_train"], r["X_train"]) / (B * P)
        )
        err = (Sigma_emp - r["Sigma_tr"]).abs().max().item()
        sigma_scale = r["Sigma_tr"].abs().max().item()
        tol = 5.0 * sigma_scale / math.sqrt(B * P)
        passed = err < tol
        n_samples = B * P
        return (
            passed,
            f"max|Σ_emp - Σ_tr| = {err:.4e}; 5σ tol = {tol:.4e}",
            err,
            tol,
            n_samples,
        )

    results.append(run_mc(phase, "covariance identity via Monte Carlo", tmc))

    return results


# ---------------------------------------------------------------------------
# Phase 5 — G2 operator-only (v4 §12.6)
# ---------------------------------------------------------------------------


def phase_g2_operator() -> list[TestResult]:
    phase = "§12.6"
    results: list[TestResult] = []

    def t1() -> str:
        cfg_a = G2Config(
            D=8, partition_params={"m": 4},
            seeds={"R": 0, "x": 0, "beta": 0, "noise": 0},
        )
        cfg_b = G2Config(
            D=8, partition_params={"m": 4},
            seeds={"R": 999, "x": 999, "beta": 999, "noise": 999},
        )
        r_a = g2_generate_operator(cfg_a)
        r_b = g2_generate_operator(cfg_b)
        assert torch.equal(r_a["Lambda"], r_b["Lambda"])
        assert torch.equal(r_a["Omega"], r_b["Omega"])
        assert torch.equal(r_a["F"], r_b["F"])
        return "different seeds → identical Λ, Ω, F (bitwise)"

    results.append(
        run_exact(phase, "operator-only: no random draws", t1)
    )

    def t2() -> str:
        cfg = G2Config(D=8, partition_params={"m": 4})
        r = g2_generate_operator(cfg)
        p = r["partition"]
        torch.manual_seed(0)
        q = torch.randn(p.n_blocks, dtype=torch.float64)
        Q = reconstruct_from_block_scalars(q, p)
        viol = commutant_violation(Q, p)
        assert viol < 1e-12, f"viol = {viol:.2e}"
        return f"violation(Σ q_b Π_b) = {viol:.2e}"

    results.append(
        run_exact(phase, "commutant_violation(Σ q_b Π_b) < 1e-12", t2)
    )

    def t3() -> str:
        kappas = (2.0, 4.0)
        cfg = G2Config(D=8, partition_params={"m": 4}, block_kappas_lam=kappas)
        r = g2_generate_operator(cfg)
        expected = torch.tensor([(k - 1) / (k + 1) for k in kappas], dtype=torch.float64)
        err = (r["rho_star"] - expected).abs().max().item()
        assert err < 1e-12, f"err = {err:.2e}"
        return f"ρ* = {r['rho_star'].tolist()}"

    results.append(run_exact(phase, "rho_star = (κ-1)/(κ+1)", t3))

    def t4() -> str:
        worst = 0.0
        for kappa in [1.0, 2.0, 5.0, 10.0]:
            cfg = G2Config(
                D=8, partition_params={"m": 4},
                block_means_lam=(3.0, 5.0),
                block_kappas_lam=(kappa, kappa),
            )
            r = g2_generate_operator(cfg)
            worst = max(worst, abs(r["Lambda"][:4].mean().item() - 3.0))
            worst = max(worst, abs(r["Lambda"][4:].mean().item() - 5.0))
        assert worst < 1e-12, f"worst = {worst:.2e}"
        return f"mass-preserving sweep: max err = {worst:.2e}"

    results.append(run_exact(phase, "mass-preserving κ sweep (operator)", t4))

    def t5() -> str:
        cfg = G2Config(D=16)
        r = g2_generate_operator(cfg)
        F = r["F"]
        err = (F @ F.T - torch.eye(16, dtype=torch.float64)).abs().max().item()
        assert err < 1e-12, f"err = {err:.2e}"
        return f"||F F^T - I|| = {err:.2e}"

    results.append(run_exact(phase, "F is real orthogonal", t5))

    return results


# ---------------------------------------------------------------------------
# Phase 6 — G2 sampled-context (v4 §12.7; mixed exact + MC)
# ---------------------------------------------------------------------------


def phase_g2_sampled() -> list[TestResult]:
    phase = "§12.7"
    results: list[TestResult] = []

    def t1() -> str:
        cfg = G2Config(D=8, partition_params={"m": 4})
        r = g2_generate_sampled(cfg, n_contexts=10, P=32, K=8)
        R = r["R"]
        for c in range(R.shape[0]):
            Rc = R[c]
            err = (Rc @ Rc.T - torch.eye(8, dtype=torch.float64)).abs().max().item()
            assert err < 1e-10, f"c={c} orthogonality: {err:.2e}"
            # Block-diagonal: off-block zeros
            Rc_off = Rc.clone()
            Rc_off[:4, :4] = 0.0
            Rc_off[4:, 4:] = 0.0
            assert Rc_off.abs().max().item() == 0.0, f"c={c}: off-block not zero"
        return f"{R.shape[0]} contexts: block-diagonal orthogonal, max ortho err < 1e-10"

    results.append(run_exact(phase, "R_c block-Haar (orthogonal + block-diagonal)", t1))

    def tmc_phys_cov() -> tuple[bool, str, float, float, int]:
        P_samples = 4096
        cfg = G2Config(
            D=8, partition_params={"m": 4},
            block_means_lam=(1.0, 1.0), block_kappas_lam=(2.0, 2.0),
        )
        r = g2_generate_sampled(cfg, n_contexts=5, P=P_samples, K=1)
        F, R, Lambda = r["F"], r["R"], r["Lambda"]
        max_err = 0.0
        for c in range(5):
            Sigma_emp = r["X_train"][c] @ r["X_train"][c].T / P_samples
            Sigma_true = F.T @ R[c] @ torch.diag(Lambda) @ R[c].T @ F
            max_err = max(max_err, (Sigma_emp - Sigma_true).abs().max().item())
        lam_max = Lambda.max().item()
        tol = 5.0 * lam_max / math.sqrt(P_samples)
        passed = max_err < tol
        return (
            passed,
            f"max over 5 contexts = {max_err:.4f}; 5σ tol = {tol:.4f}",
            max_err,
            tol,
            P_samples,
        )

    results.append(
        run_mc(phase, "physical-basis covariance identity per context", tmc_phys_cov)
    )

    def tmc_haar_avg() -> tuple[bool, str, float, float, int]:
        n_ctx = 2048
        cfg = G2Config(
            D=8, partition_params={"m": 4},
            block_means_lam=(1.0, 1.0), block_kappas_lam=(3.0, 3.0),
        )
        r = g2_generate_sampled(cfg, n_contexts=n_ctx, P=64, K=1)
        F, R, Lambda = r["F"], r["R"], r["Lambda"]
        Sigma_sum = torch.zeros(8, 8, dtype=torch.float64)
        for c in range(n_ctx):
            Sigma_sum = Sigma_sum + F.T @ R[c] @ torch.diag(Lambda) @ R[c].T @ F
        mean_Sigma = Sigma_sum / n_ctx
        partition = r["partition"]
        mean_expanded = torch.zeros(8, dtype=torch.float64)
        for b_idx, block in enumerate(partition.blocks):
            for k in block:
                mean_expanded[k] = r["block_means_lam"][b_idx]
        expected = F.T @ torch.diag(mean_expanded) @ F
        err = (mean_Sigma - expected).abs().max().item()
        tol = 5.0 * Lambda.max().item() / math.sqrt(n_ctx)
        passed = err < tol
        return (
            passed,
            f"||Haar-avg Σ - F^T diag(means) F||_inf = {err:.4f}; 5σ tol = {tol:.4f}",
            err,
            tol,
            n_ctx,
        )

    results.append(
        run_mc(phase, "across-context Haar average → commutant-class matrix", tmc_haar_avg)
    )

    def tmc_spectral() -> tuple[bool, str, float, float, int]:
        # g2_to_spectral_basis(X, F) returns R_c z (NOT canonical z).
        P_samples = 4096
        cfg = G2Config(
            D=8, partition_params={"m": 4},
            block_means_lam=(1.0, 1.0), block_kappas_lam=(2.0, 2.0),
        )
        r = g2_generate_sampled(cfg, n_contexts=3, P=P_samples, K=1)
        F, R, Lambda = r["F"], r["R"], r["Lambda"]
        tilde_X = g2_to_spectral_basis(r["X_train"][0], F)
        Sigma_spec_emp = tilde_X @ tilde_X.T / P_samples
        Sigma_spec_true = R[0] @ torch.diag(Lambda) @ R[0].T
        err_spec = (Sigma_spec_emp - Sigma_spec_true).abs().max().item()
        z = R[0].T @ tilde_X  # canonical coordinates
        Sigma_z_emp = z @ z.T / P_samples
        err_z = (Sigma_z_emp - torch.diag(Lambda)).abs().max().item()
        metric = max(err_spec, err_z)
        tol = 5.0 * Lambda.max().item() / math.sqrt(P_samples)
        passed = err_spec < tol and err_z < tol
        return (
            passed,
            f"cov(F X) vs R Λ R^T: {err_spec:.3f}; cov(R^T F X) vs diag(Λ): {err_z:.3f}; 5σ tol: {tol:.3f}",
            metric,
            tol,
            P_samples,
        )

    results.append(
        run_mc(phase, "g2_to_spectral_basis returns R z (not canonical z)", tmc_spectral)
    )

    return results


# ---------------------------------------------------------------------------
# Phase 7 — G3 (v4 §12.8)
# ---------------------------------------------------------------------------


def phase_g3() -> list[TestResult]:
    phase = "§12.8"
    results: list[TestResult] = []

    def t1() -> str:
        ladder = dyadic_ladder(16)
        lam = torch.linspace(0.5, 2.0, 16, dtype=torch.float64)
        omega = torch.ones(16, dtype=torch.float64)
        levels = g3_generate(lam, omega, ladder)
        for j in range(len(levels) - 1):
            assert refines(
                levels[j + 1]["partition"], levels[j]["partition"]
            ), f"level {j+1} fails to refine {j}"
        return f"{len(levels)} levels, all adjacent refinements valid"

    results.append(
        run_exact(phase, "refines(ladder[j+1], ladder[j]) for all j", t1)
    )

    def t2() -> str:
        ladder = dyadic_ladder(16)
        lam = torch.linspace(0.5, 2.0, 16, dtype=torch.float64)
        omega = torch.ones(16, dtype=torch.float64)
        levels = g3_generate(lam, omega, ladder)
        for j in range(1, len(levels)):
            assert torch.equal(levels[j]["Lambda"], levels[0]["Lambda"])
            assert torch.equal(levels[j]["Omega"], levels[0]["Omega"])
            assert torch.equal(levels[j]["F"], levels[0]["F"])
        return f"Λ, Ω, F bitwise identical across {len(levels)} levels"

    results.append(
        run_exact(phase, "Λ, Ω, F bitwise-identical across ladder", t2)
    )

    def t3() -> str:
        # D=8 (4 levels) keeps oracle_commutant_loss LBFGS work bounded.
        ladder = dyadic_ladder(8)
        lam = torch.linspace(0.5, 2.0, 8, dtype=torch.float64)
        omega = torch.ones(8, dtype=torch.float64)
        levels = g3_generate(lam, omega, ladder)
        losses = []
        for level in levels:
            res = oracle_commutant_loss(
                level["Lambda"], level["Omega"], level["partition"], L=1, max_iter=100
            )
            losses.append(res["loss_star"])
        for j in range(len(losses) - 1):
            assert (
                losses[j + 1] <= losses[j] + 1e-10
            ), f"non-monotone at j={j}: {losses[j]} -> {losses[j+1]}"
        return f"oracle losses coarse→fine: {[round(x, 4) for x in losses]}"

    results.append(
        run_exact(phase, "oracle-loss monotonicity coarse → fine", t3)
    )

    return results


# ---------------------------------------------------------------------------
# Phase 8 — cost_models (v4 §12.9)
# ---------------------------------------------------------------------------


def phase_cost_models() -> list[TestResult]:
    phase = "§12.9"
    results: list[TestResult] = []

    def t1() -> str:
        for P in [4, 16, 128]:
            assert phi_adaptive(P) == P * P
        P_t = torch.tensor([4.0, 16.0], dtype=torch.float64)
        expected = P_t.pow(2)
        err = (phi_adaptive(P_t) - expected).abs().max().item()
        assert err < 1e-12
        return "scalar & tensor ok"

    results.append(run_exact(phase, "phi_adaptive(P) == P²", t1))

    def t2() -> str:
        worst = 0.0
        for P, r in [(16, 4), (64, 8), (128, 16)]:
            expected = P * math.log(P) + P * r
            worst = max(worst, abs(phi_spectral_fft(P, r) - expected))
        assert worst < 1e-10, f"worst = {worst:.2e}"
        return f"worst = {worst:.2e}"

    results.append(
        run_exact(phase, "phi_spectral_fft = P log P + P r", t2)
    )

    def t3() -> str:
        base = compute_proxy(t=10, P=8, L_A=2, L_S=3, r=4)
        e1 = abs(compute_proxy(t=20, P=8, L_A=2, L_S=3, r=4) - 2 * base)
        only_A = compute_proxy(t=10, P=8, L_A=2, L_S=0, r=4)
        e2 = abs(compute_proxy(t=10, P=8, L_A=4, L_S=0, r=4) - 2 * only_A)
        only_S = compute_proxy(t=10, P=8, L_A=0, L_S=3, r=4)
        e3 = abs(compute_proxy(t=10, P=8, L_A=0, L_S=6, r=4) - 2 * only_S)
        pq1 = compute_proxy(t=10, P=4, L_A=2, L_S=0, r=0)
        pq2 = compute_proxy(t=10, P=8, L_A=2, L_S=0, r=0)
        e4 = abs(pq2 / pq1 - 4.0)
        assert max(e1, e2, e3, e4) < 1e-10
        return "linear in t, L_A, L_S; quadratic in P"

    results.append(run_exact(phase, "compute_proxy linearity + P²", t3))

    def tmc() -> tuple[bool, str, float, float, int]:
        torch.manual_seed(0)
        true_c_A, true_c_S = 2.5, 0.3
        runs = []
        for t_v in [10, 50, 100]:
            for P_v in [16, 32, 64]:
                for L_A_v in [1, 2, 4]:
                    for L_S_v in [1, 2, 4]:
                        r_v = 8
                        wc_true = (
                            true_c_A * t_v * P_v ** 2 * L_A_v
                            + true_c_S
                            * t_v
                            * (P_v * math.log(P_v) + P_v * r_v)
                            * L_S_v
                        )
                        noise = 0.001 * wc_true * torch.randn(1).item()
                        runs.append(
                            {
                                "t": t_v,
                                "P": P_v,
                                "L_A": L_A_v,
                                "L_S": L_S_v,
                                "r": r_v,
                                "wall_clock_seconds": wc_true + noise,
                            }
                        )
        res = calibrate(runs)
        err_A = abs(res["c_A"] - true_c_A) / true_c_A
        err_S = abs(res["c_S"] - true_c_S) / true_c_S
        metric = max(err_A, err_S)
        tol = 0.10
        passed = err_A < tol and err_S < tol
        return (
            passed,
            f"c_A err = {err_A*100:.3f}%, c_S err = {err_S*100:.3f}%, R² = {res['r2']:.6f}",
            metric,
            tol,
            len(runs),
        )

    results.append(
        run_mc(phase, "calibrate recovers (c_A, c_S) under 0.1% noise", tmc)
    )

    return results


# ---------------------------------------------------------------------------
# Phase 9 — fit_powerlaws (v4 §12.10)
# ---------------------------------------------------------------------------


def phase_fit_powerlaws() -> list[TestResult]:
    phase = "§12.10"
    results: list[TestResult] = []

    def t1() -> str:
        true_alpha = -1.5
        true_c = 2.0
        x = torch.logspace(0, 2, 30, dtype=torch.float64)
        y = true_c * x.pow(true_alpha)
        r = fit_loglog(x, y, fit_window=(1.0, 100.0))
        err_s = abs(r["slope"] - true_alpha)
        err_i = abs(r["intercept"] - math.log(true_c))
        assert err_s < 1e-6 and err_i < 1e-6
        return f"slope err = {err_s:.2e}, intercept err = {err_i:.2e}, R² = {r['r2']:.6f}"

    results.append(
        run_exact(phase, "fit_loglog noise-free recovery < 1e-6", t1)
    )

    def tmc() -> tuple[bool, str, float, float, int]:
        # Smoke-level footprint (5 repeats × 50 bootstraps) to fit within the
        # login-node CPU-time budget. The full §12.10(2) spec calls for 200
        # repeats × 1000 bootstraps; run that separately on a compute node.
        true_alpha = -1.5
        true_c = 2.0
        x = torch.logspace(0, 2, 20, dtype=torch.float64)
        n_seeds = 20
        n_repeats = 5
        n_boot = 50
        covered = 0
        for rep in range(n_repeats):
            g = torch.Generator().manual_seed(rep)
            noise = 1.0 + 0.05 * torch.randn(
                n_seeds, x.shape[0], generator=g, dtype=torch.float64
            )
            y = (true_c * x.pow(true_alpha)).unsqueeze(0) * noise
            res = bootstrap_exponent(
                x, y, fit_window=(1.0, 100.0), seed_axis=0, n_bootstrap=n_boot
            )
            if res["slope_lo"] <= true_alpha <= res["slope_hi"]:
                covered += 1
        coverage = covered / n_repeats
        tol = 0.60  # smoke-level: 5 repeats cannot sharply estimate 95% coverage
        passed = coverage >= tol
        return (
            passed,
            f"CI coverage = {coverage:.2%} over {n_repeats} repeats × {n_boot} boot "
            f"(smoke-level; full 200×1000 run is out-of-band)",
            coverage,
            tol,
            n_repeats * n_boot,
        )

    results.append(run_mc(phase, "bootstrap 95% CI covers true slope", tmc))

    def t3() -> str:
        torch.manual_seed(0)
        true_alpha = -1.5
        true_c = 2.0
        x_all = torch.logspace(0, 2, 40, dtype=torch.float64)
        y_all = true_c * x_all.pow(true_alpha)
        y_noisy = y_all * (1.0 + 0.01 * torch.randn_like(y_all))
        x_fit, y_fit = x_all[::2], y_noisy[::2]
        x_val, y_val = x_all[1::2], y_noisy[1::2]
        res = holdout_evaluate(x_fit, y_fit, x_val, y_val, fit_window=(1.0, 100.0))
        assert res["median_rel_err"] < 0.10, f"median = {res['median_rel_err']}"
        return f"median = {res['median_rel_err']:.4f}, max = {res['max_rel_err']:.4f}"

    results.append(
        run_exact(phase, "holdout median rel err < 10% on noisy split", t3)
    )

    return results


# ---------------------------------------------------------------------------
# Phase 10 — metrics (v4 §12.11)
# ---------------------------------------------------------------------------


def phase_metrics() -> list[TestResult]:
    phase = "§12.11"
    results: list[TestResult] = []

    def t1() -> str:
        y = torch.randn(10, dtype=torch.float64)
        err = reduced_model_error(y, y)
        assert err < 1e-12
        return f"reduced_model_error(y, y) = {err:.2e}"

    results.append(
        run_exact(phase, "reduced_model_error(y, y) = 0", t1)
    )

    def _ab_setup():
        P, K, L = 8, 3, 4
        torch.manual_seed(0)
        A_GD = torch.randn(P, P, dtype=torch.float64) * 0.05
        B_GD = torch.randn(K, P, dtype=torch.float64)
        T_GD = torch.eye(P, dtype=torch.float64) + A_GD / L
        y = torch.randn(P, dtype=torch.float64)
        return P, K, L, A_GD, B_GD, T_GD, y

    def t2() -> str:
        P, K, L, A_GD, B_GD, T_GD, y = _ab_setup()
        r = ab_perturbation_bound(A_GD, A_GD, B_GD, B_GD, T_GD, T_GD, L, y)
        for k in [
            "delta_A_op",
            "delta_B_op",
            "A_side_bound",
            "B_side_bound",
            "total_bound",
            "empirical_error",
        ]:
            assert abs(r[k]) < 1e-12, f"{k} = {r[k]}"
        return "all bounds and empirical error = 0 at θ=0"

    results.append(run_exact(phase, "A2: θ=0 case", t2))

    def t3() -> str:
        P, K, L, A_GD, B_GD, T_GD, y = _ab_setup()
        torch.manual_seed(1)
        B_theta = B_GD + 1e-2 * torch.randn(K, P, dtype=torch.float64)
        r = ab_perturbation_bound(
            A_GD, A_GD, B_theta, B_GD, T_GD, T_GD, L, y
        )
        assert abs(r["A_side_bound"]) < 1e-12
        assert abs(r["total_bound"] - r["B_side_bound"]) < 1e-12
        return f"Δ_A=0 → A_side=0, total=B_side={r['total_bound']:.3e}"

    results.append(run_exact(phase, "A2: Δ_A=0 ⇒ A-side bound = 0", t3))

    def t4() -> str:
        P, K, L, A_GD, B_GD, T_GD, y = _ab_setup()
        torch.manual_seed(2)
        A_theta = A_GD + 1e-2 * torch.randn(P, P, dtype=torch.float64)
        T_theta = torch.eye(P, dtype=torch.float64) + A_theta / L
        r = ab_perturbation_bound(
            A_theta, A_GD, B_GD, B_GD, T_theta, T_GD, L, y
        )
        assert abs(r["B_side_bound"]) < 1e-12
        assert abs(r["total_bound"] - r["A_side_bound"]) < 1e-12
        return f"Δ_B=0 → B_side=0, total=A_side={r['total_bound']:.3e}"

    results.append(run_exact(phase, "A2: Δ_B=0 ⇒ B-side bound = 0", t4))

    def t5() -> str:
        P, K, L, A_GD, B_GD, T_GD, y = _ab_setup()
        torch.manual_seed(3)
        A_theta = A_GD + 1e-3 * torch.randn(P, P, dtype=torch.float64)
        B_theta = B_GD + 1e-3 * torch.randn(K, P, dtype=torch.float64)
        T_theta = torch.eye(P, dtype=torch.float64) + A_theta / L
        r = ab_perturbation_bound(
            A_theta, A_GD, B_theta, B_GD, T_theta, T_GD, L, y
        )
        assert r["empirical_error"] <= r["total_bound"] + 1e-12, (
            f"bound violated: emp={r['empirical_error']}, total={r['total_bound']}"
        )
        return f"emp = {r['empirical_error']:.3e} ≤ total = {r['total_bound']:.3e}"

    results.append(run_exact(phase, "A2: empirical ≤ total at small θ", t5))

    def t6() -> str:
        partition = BlockPartition(D=2, blocks=((0, 1),))
        lam = torch.tensor([1.0, 4.0], dtype=torch.float64)
        omega = torch.tensor([1.0, 1.0], dtype=torch.float64)
        res = oracle_commutant_loss(lam, omega, partition, L=1, max_iter=200)
        q_exp = 17.0 / 65.0
        l_exp = 468.0 / 845.0
        err_q = abs(res["q_star"].item() - q_exp)
        err_l = abs(res["loss_star"] - l_exp)
        assert err_q < 1e-8 and err_l < 1e-10
        return f"q* err = {err_q:.2e}, loss* err = {err_l:.2e}, converged={res['converged']}"

    results.append(
        run_exact(phase, "oracle closed form (1 block, λ=(1,4), ω=(1,1), L=1)", t6)
    )

    def t7() -> str:
        # κ=1, L>0 → zeros
        v = contraction_depth_overlay(1.0, torch.tensor([1.0, 2.0, 4.0, 8.0]))
        assert v.abs().max().item() < 1e-12, f"κ=1 L>0: max = {v.abs().max()}"
        # κ=1, L=0 → 1
        v0 = contraction_depth_overlay(1.0, torch.tensor([0.0]))
        assert abs(v0.item() - 1.0) < 1e-12
        # κ=9, L=5 → 0.8^10
        v_ref = contraction_depth_overlay(9.0, torch.tensor([5.0]))
        ref = 0.8 ** 10
        assert abs(v_ref.item() - ref) < 1e-10, f"κ=9 L=5 err: {abs(v_ref.item() - ref)}"
        # κ→∞, L=10 → close to 1
        v_inf = contraction_depth_overlay(1e6, torch.tensor([1.0, 10.0, 100.0]))
        assert (v_inf - 1.0).abs().max().item() < 1e-3, (
            f"κ→∞ failed: {v_inf.tolist()}"
        )
        # Strict decay
        v_decay = contraction_depth_overlay(2.0, torch.tensor([1.0, 5.0, 10.0]))
        assert all(
            v_decay[i] > v_decay[i + 1] for i in range(len(v_decay) - 1)
        ), "decay not strict"
        return f"κ=1→0 ✓; κ=9 L=5 = {v_ref.item():.6f} ≈ 0.8¹⁰; κ→∞ → 1; strict decay ✓"

    results.append(
        run_exact(phase, "contraction_depth_overlay all edge cases", t7)
    )

    def t8() -> str:
        # Use a flat symbol so the discrete recursion is unconditionally stable
        # under modest eta (power-law symbols put some modes well above the
        # 2L/(2L-1) / s^3 stability threshold at eta=0.05).
        P = 16
        s_tr = symbol_flat(P, value=1.0)
        omega = torch.ones(P, dtype=torch.float64)
        traj = gamma_star_trajectory_circulant(s_tr, omega, L=4, eta=0.05, T=1000)
        assert traj.shape == (1001, P), f"shape = {tuple(traj.shape)}"
        assert traj.dtype == torch.float64, f"dtype = {traj.dtype}"
        assert not torch.isnan(traj).any(), "NaN in trajectory"
        assert not torch.isinf(traj).any(), "Inf in trajectory"
        diffs = traj[1:] - traj[:-1]
        assert diffs.min().item() >= -1e-12, (
            f"non-monotone step: min = {diffs.min().item():.2e}"
        )
        return (
            f"shape={tuple(traj.shape)}, dtype=float64, "
            f"min step = {diffs.min().item():.2e}"
        )

    results.append(
        run_exact(phase, "gamma_star_trajectory shape + dtype + monotonicity", t8)
    )

    return results


# ---------------------------------------------------------------------------
# Phase 11 — real-valued + no-callable audit (v4 §12.1)
# ---------------------------------------------------------------------------


def phase_real_valued_audit() -> list[TestResult]:
    phase = "§12.1"
    results: list[TestResult] = []

    def t1() -> str:
        # The 4 labeled complex endpoints return complex.
        F = fourier_ops.dft_matrix(8)
        assert F.is_complex(), "dft_matrix should be complex"
        Fi = fourier_ops.idft_matrix(8)
        assert Fi.is_complex(), "idft_matrix should be complex"
        x = torch.randn(8, dtype=torch.float64)
        Xd = fourier_ops.unitary_dft(x)
        assert Xd.is_complex(), "unitary_dft should be complex"
        Xi = fourier_ops.unitary_idft(Xd)
        assert Xi.is_complex(), "unitary_idft should be complex"
        # Every other fourier_ops public return is real.
        for name, args in [
            ("freq_grid", (8,)),
            ("real_spectral_basis", (8,)),
            ("symbol_flat", (8,)),
            ("symbol_power_law", (8, 1.5)),
            ("circulant_from_symbol", (symbol_power_law(8, 1.5),)),
        ]:
            func = getattr(fourier_ops, name)
            out = func(*args)
            assert isinstance(out, torch.Tensor) and not out.is_complex(), (
                f"{name} returned complex"
            )
        return "4 complex endpoints + all real-wrapper endpoints verified"

    results.append(run_exact(phase, "fourier_ops: complex confined to 4 endpoints", t1))

    def t2() -> str:
        outputs = [
            ("GA", ga_generate(GAConfig(D=4, P=6, K=2))),
            ("G1", g1_generate(G1Config(P=16, sample_data=True))),
            ("G2op", g2_generate_operator(G2Config(D=8, partition_params={"m": 4}))),
            (
                "G2samp",
                g2_generate_sampled(
                    G2Config(D=8, partition_params={"m": 4}),
                    n_contexts=2,
                    P=8,
                    K=2,
                ),
            ),
        ]
        for name, obj in outputs:
            for k, v in obj.items():
                assert not callable(v), f"{name}: key {k!r} is callable"
                if isinstance(v, torch.Tensor):
                    assert not v.is_complex(), (
                        f"{name}: key {k!r} returned complex tensor"
                    )
        return "no callables, no complex in GA / G1 / G2-op / G2-sampled outputs"

    results.append(run_exact(phase, "no callables / no complex in generator returns", t2))

    def t3() -> str:
        # Spot-check a few publics from commutants / partitions / metrics.
        Q = torch.randn(8, 8, dtype=torch.float64)
        p = equal_blocks(8, 4)
        assert not commutant_projection(Q, p).is_complex()
        assert not mass_preserving_block_spectrum(
            p,
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            torch.tensor([2.0, 2.0], dtype=torch.float64),
        ).is_complex()
        s = symbol_power_law(16, 1.5)
        traj = gamma_star_trajectory_circulant(
            s, torch.ones(16, dtype=torch.float64), L=2, eta=0.05, T=100
        )
        assert not traj.is_complex()
        return "commutants / partitions / metrics publics return real"

    results.append(
        run_exact(phase, "non-fourier publics return real tensors", t3)
    )

    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    torch.manual_seed(0)

    print("=" * 78)
    print(" Thesis utils self-test harness  (v4 §12 reference)")
    print("=" * 78)

    phases: list[tuple[str, Callable[[], list[TestResult]]]] = [
        ("Phase 1  fourier_ops (§12.2)", phase_fourier_ops),
        ("Phase 2  partitions / commutants (§12.3)", phase_partitions_commutants),
        ("Phase 3  GA (§12.4)", phase_ga),
        ("Phase 4  G1 (§12.5)", phase_g1),
        ("Phase 5  G2 operator-only (§12.6)", phase_g2_operator),
        ("Phase 6  G2 sampled-context (§12.7)", phase_g2_sampled),
        ("Phase 7  G3 (§12.8)", phase_g3),
        ("Phase 8  cost_models (§12.9)", phase_cost_models),
        ("Phase 9  fit_powerlaws (§12.10)", phase_fit_powerlaws),
        ("Phase 10 metrics (§12.11)", phase_metrics),
        ("Phase 11 real-valued / no-callable audit (§12.1)", phase_real_valued_audit),
    ]

    all_results: list[tuple[str, list[TestResult]]] = []
    for phase_name, phase_fn in phases:
        results = phase_fn()
        print_phase_results(phase_name, results)
        all_results.append((phase_name, results))

    print_summary(all_results)

    exact_failed = any(
        not r.passed
        for _, results in all_results
        for r in results
        if r.category == "exact"
    )
    return 1 if exact_failed else 0


if __name__ == "__main__":
    sys.exit(main())
