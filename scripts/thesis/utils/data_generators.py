"""Data generators: GA, G1, G2 (operator + sampled), G3.

Plan correspondence: EXPERIMENT_PLAN_FINAL.MD §4.8 (data generators). Step-1
Generator / Utility Specification v4 §10.

Four generators exactly per the v4 spec:

- **GA** (theorem-A masked context): sample-space reduced operators
  ``A_S, B_S, T`` as the PRIMARY reduced objects, computed per-batch from the
  actual sampled ``(X_train, X_query)``; optional secondary ``(D, D)``
  feature-space helpers under ``return_feature_space=True``.

- **G1** (theorem-B stationary circulant): three distinct spectra
  ``s_tr`` (training), ``s_te`` (test / query), ``omega`` (teacher); real
  circulant covariances via :mod:`fourier_ops`; matched-mode default is
  INDEPENDENT ``X_query`` realizations, with ``matched_query_realization="shared"``
  as an opt-in sanity control; single-query and full-window query regimes;
  NO callables, NO trajectory outputs (use
  :func:`scripts.thesis.utils.metrics.gamma_star_trajectory_circulant`).

- **G2** (theorem-C band-RRS): operator-only and sampled-context modes are
  separate APIs; sampled-context mode generates PHYSICAL-BASIS data via

        Sigma_c = F^T @ R_c @ diag(Lambda) @ R_c^T @ F

  where F is the fixed spectral basis (DCT-II by default; v4 §3.1, §10.3.1)
  and R_c is a per-context block-Haar rotation in the spectral basis. The
  :func:`g2_to_spectral_basis` helper returns ``F @ X = R_c z`` (rotated
  spectral coordinates), NOT canonical diagonal coordinates ``z``.

- **G3** (refinement ladder): Lambda and Omega are bitwise-identical across
  every ladder level; only the partition changes. Both a direct-tensors API
  and a constructive config API are provided.

Every public return is purely numerical / serializable: real tensors, plain
dicts, floats, and frozen dataclass configs. No callables appear in any
generator output dictionary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from scripts.thesis.utils.commutants import refines
from scripts.thesis.utils.fourier_ops import (
    circulant_from_symbol,
    frequency_permutation as _fourier_frequency_permutation,
    real_spectral_basis,
    symbol_flat,
    symbol_interpolate,
    symbol_multiband,
    symbol_power_law,
)
from scripts.thesis.utils.partitions import (
    BlockPartition,
    custom_ladder,
    dyadic_ladder,
    equal_blocks,
    mass_preserving_block_spectrum,
    mass_preserving_block_task,
)


# ---------------------------------------------------------------------------
# Basis-conversion helpers (the only approved deviation from the column-sample
# convention; explicitly named so any use is visible in diff review).
# ---------------------------------------------------------------------------


def cols_to_rows(X: torch.Tensor) -> torch.Tensor:
    """Convert a column-sample tensor ``(..., D, P)`` to row-sample
    ``(..., P, D)`` by transposing the last two dims.

    The thesis canonical convention is column-sample (v4 §1.2); this helper
    exists only for internal use when an external routine requires row-sample
    inputs, and every call site must document the deviation.
    """
    return X.transpose(-2, -1)


def rows_to_cols(X: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`cols_to_rows`."""
    return X.transpose(-2, -1)


# ---------------------------------------------------------------------------
# Internal helpers: dtype parsing, covariance / gamma / partition / symbol
# ---------------------------------------------------------------------------


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _parse_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype {name!r}; expected one of {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def _build_covariance(
    D: int, kind: str, params: dict[str, Any], dtype: torch.dtype
) -> torch.Tensor:
    """Return a ``(D, D)`` symmetric PSD covariance matrix."""
    if kind == "isotropic":
        return torch.eye(D, dtype=dtype)
    if kind == "diag_spectrum":
        spec = params.get("spec")
        if spec is None:
            raise ValueError("kind='diag_spectrum' requires params['spec']")
        spec_t = torch.as_tensor(spec, dtype=dtype)
        if spec_t.shape != (D,):
            raise ValueError(
                f"spec must have shape ({D},); got {tuple(spec_t.shape)}"
            )
        if (spec_t < 0).any():
            raise ValueError("spec entries must be non-negative")
        return torch.diag(spec_t)
    if kind == "full_matrix":
        M = params.get("matrix")
        if M is None:
            raise ValueError("kind='full_matrix' requires params['matrix']")
        M_t = torch.as_tensor(M, dtype=dtype)
        if M_t.shape != (D, D):
            raise ValueError(
                f"matrix must have shape ({D}, {D}); got {tuple(M_t.shape)}"
            )
        return M_t.contiguous()
    raise ValueError(
        f"unknown covariance kind: {kind!r}; expected 'isotropic', "
        "'diag_spectrum', or 'full_matrix'"
    )


def _build_gamma(
    D: int, kind: str, params: dict[str, Any], dtype: torch.dtype
) -> torch.Tensor:
    """Return a ``(D, D)`` tensor used as ``Gamma`` in the A_S / B_S formulas."""
    if kind == "identity":
        return torch.eye(D, dtype=dtype)
    if kind == "diag_spectrum":
        spec = params.get("spec")
        if spec is None:
            raise ValueError("Gamma_kind='diag_spectrum' requires params['spec']")
        spec_t = torch.as_tensor(spec, dtype=dtype)
        if spec_t.shape != (D,):
            raise ValueError(
                f"Gamma spec must have shape ({D},); got {tuple(spec_t.shape)}"
            )
        return torch.diag(spec_t)
    if kind == "full_matrix":
        M = params.get("matrix")
        if M is None:
            raise ValueError("Gamma_kind='full_matrix' requires params['matrix']")
        M_t = torch.as_tensor(M, dtype=dtype)
        if M_t.shape != (D, D):
            raise ValueError(
                f"Gamma matrix must have shape ({D}, {D}); got {tuple(M_t.shape)}"
            )
        return M_t.contiguous()
    raise ValueError(f"unknown Gamma_kind: {kind!r}")


def _build_partition(
    D: int, kind: str, params: dict[str, Any]
) -> BlockPartition:
    if kind == "equal":
        m = int(params.get("m", 1))
        return equal_blocks(D, m)
    if kind == "dyadic":
        J = params.get("J")
        ladder = dyadic_ladder(D, J)
        return ladder[-1]  # finest level by default
    if kind == "custom":
        blocks = params.get("blocks")
        if blocks is None:
            raise ValueError("partition_kind='custom' requires params['blocks']")
        blocks_tuple = tuple(tuple(int(k) for k in b) for b in blocks)
        return BlockPartition(D=D, blocks=blocks_tuple)
    raise ValueError(f"unknown partition_kind: {kind!r}")


def _build_spectral_basis(
    D: int,
    kind: str,
    custom: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kind == "custom":
        if custom is None:
            raise ValueError("spectral_basis_kind='custom' requires spectral_basis_custom")
        F = torch.as_tensor(custom, dtype=dtype)
        if F.shape != (D, D):
            raise ValueError(
                f"spectral_basis_custom must have shape ({D}, {D}); got {tuple(F.shape)}"
            )
        # Sanity: F must be (close to) real orthogonal
        ortho_err = (F @ F.T - torch.eye(D, dtype=dtype)).abs().max().item()
        if ortho_err > 1e-6:
            raise ValueError(
                f"spectral_basis_custom must be real orthogonal; got max|F F^T - I| = {ortho_err:.2e}"
            )
        return F.contiguous()
    return real_spectral_basis(D, kind).to(dtype)


def _resolve_per_block(
    values: Sequence[float] | None,
    n_blocks: int,
    default: float,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if values is None:
        return torch.full((n_blocks,), float(default), dtype=dtype)
    if len(values) != n_blocks:
        raise ValueError(
            f"{name} must have length n_blocks = {n_blocks}; got {len(values)}"
        )
    return torch.tensor(tuple(float(v) for v in values), dtype=dtype)


def _build_symbol(
    P: int,
    kind: str,
    params: dict[str, Any],
    dtype: torch.dtype,
    *,
    reference_symbol: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct a real-even length-P symbol of dtype ``float64``.

    For kinds ``'interpolate'`` / ``'permute'`` the caller must supply a
    reference_symbol (the training symbol in the G1 OOD use case).
    """
    if kind == "power_law":
        nu = float(params.get("nu", params.get("nu_beta", 1.5)))
        eps_val = float(params.get("eps", 1e-6))
        return symbol_power_law(P, nu, eps=eps_val).to(dtype)
    if kind == "multiband":
        bands = params.get("bands")
        if bands is None:
            raise ValueError("symbol_kind='multiband' requires params['bands']")
        return symbol_multiband(P, bands).to(dtype)
    if kind == "flat":
        value = float(params.get("value", 1.0))
        return symbol_flat(P, value).to(dtype)
    if kind == "interpolate":
        if reference_symbol is None:
            raise ValueError(
                "symbol_kind='interpolate' requires a reference_symbol"
            )
        alpha = float(params["alpha"])
        other_kind = params["other_kind"]
        other_params = params.get("other_params", {})
        other = _build_symbol(P, other_kind, other_params, dtype)
        return symbol_interpolate(reference_symbol, other, alpha).to(dtype)
    if kind == "permute":
        if reference_symbol is None:
            raise ValueError(
                "symbol_kind='permute' requires a reference_symbol"
            )
        seed = int(params.get("seed", 0))
        return _fourier_frequency_permutation(reference_symbol, seed=seed).to(dtype)
    if kind == "custom":
        s = params.get("symbol")
        if s is None:
            raise ValueError("symbol_kind='custom' requires params['symbol']")
        s_t = torch.as_tensor(s, dtype=dtype)
        if s_t.shape != (P,):
            raise ValueError(f"custom symbol must have shape ({P},); got {tuple(s_t.shape)}")
        return s_t.contiguous()
    raise ValueError(f"unknown symbol kind: {kind!r}")


# ---------------------------------------------------------------------------
# Internal helpers: Gaussian sampling + block-Haar rotations
# ---------------------------------------------------------------------------


def _sqrt_psd(Sigma: torch.Tensor) -> torch.Tensor:
    """Return a real matrix ``L`` such that ``L L^T == Sigma`` (PSD).

    Uses Cholesky when possible; falls back to eigendecomposition if the
    matrix is rank-deficient.
    """
    try:
        return torch.linalg.cholesky(Sigma)
    except RuntimeError:
        evals, evecs = torch.linalg.eigh(Sigma)
        evals = evals.clamp(min=0.0)
        return evecs @ torch.diag(evals.sqrt())


def _sample_gaussian_columns(
    B: int, D: int, n: int, Sigma: torch.Tensor, gen: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    """Sample ``(B, D, n)`` where each column ~ N(0, Sigma) (i.i.d. across columns)."""
    L = _sqrt_psd(Sigma)
    Z = torch.randn(B, D, n, generator=gen, dtype=dtype)
    return torch.einsum("ij,bjn->bin", L, Z).contiguous()


def _sample_gaussian_vectors(
    B: int, D: int, Sigma: torch.Tensor, gen: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    """Sample ``(B, D)`` where each row ~ N(0, Sigma)."""
    L = _sqrt_psd(Sigma)
    Z = torch.randn(B, D, generator=gen, dtype=dtype)
    return torch.einsum("ij,bj->bi", L, Z).contiguous()


def _sample_block_haar(
    partition: BlockPartition,
    n_contexts: int,
    gen: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample ``(n_contexts, D, D)`` block-diagonal orthogonal matrices ``R_c``.

    Each block of size ``m_b`` is Haar-uniform on ``O(m_b)``, sampled via QR
    of a Gaussian with a sign correction on the diagonal of R (standard
    construction).
    """
    D = partition.D
    R = torch.eye(D, dtype=dtype).unsqueeze(0).expand(n_contexts, D, D).contiguous()
    for block in partition.blocks:
        m = len(block)
        A = torch.randn(n_contexts, m, m, generator=gen, dtype=dtype)
        Q, R_qr = torch.linalg.qr(A)
        diag_R = torch.diagonal(R_qr, dim1=-2, dim2=-1)
        # sign() returns 0 at exactly zero; replace with 1 for robustness.
        sign = torch.where(
            diag_R == 0, torch.ones_like(diag_R), diag_R.sign()
        )
        Q = Q * sign.unsqueeze(-2)
        idx = torch.tensor(list(block), dtype=torch.long)
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        R[:, ii, jj] = Q
    return R


# ---------------------------------------------------------------------------
# Mask construction (GA)
# ---------------------------------------------------------------------------


def _build_mask(
    P: int,
    K: int,
    mask_kind: str,
    mask_perturbation: dict[str, Any] | None,
    non_gd_kind: str | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the ``(P + K, P + K)`` mask matrix plus (for mask_kind='perturbed')
    the ``(P, P)`` train-train perturbation pattern ``Delta`` with unit Frobenius
    norm. For non-perturbed masks, the second return is ``None``.
    """
    M = torch.zeros(P + K, P + K, dtype=dtype)
    # GD-compatible baseline: -1_P 1_P^T on the train-train block, +1_K 1_P^T on
    # test-train, zeros elsewhere.
    M[:P, :P] = -1.0
    M[P:, :P] = 1.0

    if mask_kind == "gd_compatible":
        return M, None

    if mask_kind == "perturbed":
        if mask_perturbation is None:
            raise ValueError(
                "mask_kind='perturbed' requires mask_perturbation dict with "
                "'theta' and 'pattern_seed'"
            )
        theta = float(mask_perturbation["theta"])
        pattern_seed = int(mask_perturbation["pattern_seed"])
        gen = torch.Generator(device="cpu")
        gen.manual_seed(pattern_seed)
        Delta_raw = torch.randn(P, P, generator=gen, dtype=dtype)
        Delta = (Delta_raw + Delta_raw.T) / 2.0
        fro = Delta.norm()
        if fro.item() <= 0:
            raise RuntimeError("perturbation pattern has zero Frobenius norm")
        Delta = Delta / fro
        M[:P, :P] = M[:P, :P] + theta * Delta
        return M, Delta

    if mask_kind == "non_gd_control":
        if non_gd_kind == "signflip_testtest":
            # Flip sign in the test-train block: now train-train and test-train
            # have the same sign, structurally violating M_GD.
            M[P:, :P] = -1.0
        elif non_gd_kind == "nonzero_testblock":
            # Non-zero right-block: train tokens see query positions.
            M[:P, P:] = 1.0
        else:
            raise ValueError(
                f"non_gd_kind must be 'signflip_testtest' or 'nonzero_testblock'; "
                f"got {non_gd_kind!r}"
            )
        return M, None

    raise ValueError(
        f"unknown mask_kind: {mask_kind!r}; expected 'gd_compatible', "
        "'perturbed', or 'non_gd_control'"
    )


# ---------------------------------------------------------------------------
# GA: theorem-A masked-context generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GAConfig:
    D: int
    P: int
    K: int
    B: int = 1
    Sigma_kind: str = "isotropic"
    Sigma_params: dict[str, Any] = field(default_factory=dict)
    Omega_kind: str = "isotropic"
    Omega_params: dict[str, Any] = field(default_factory=dict)
    sigma: float = 0.0
    label_norm: str = "sqrt_D"
    mask_kind: str = "gd_compatible"
    mask_perturbation: dict[str, Any] | None = None
    non_gd_kind: str | None = None
    Gamma_kind: str = "identity"
    Gamma_params: dict[str, Any] = field(default_factory=dict)
    L: int = 1
    return_feature_space: bool = False
    seeds: dict[str, int] = field(
        default_factory=lambda: {"x": 0, "beta": 1, "noise": 2, "mask": 3}
    )
    dtype: str = "float64"
    device: str = "cpu"


def ga_generate(cfg: GAConfig) -> dict[str, Any]:
    """Theorem-A masked-context generator (v4 §10.1).

    Returns a dict with, among other keys, the PRIMARY sample-space reduced
    operators

        ``A_S_GD``, ``B_S_GD``, ``T_GD``, ``A_S_theta``, ``B_S_theta``, ``T_theta``

    computed per-batch from the actual sampled ``(X_train, X_query)`` via the
    thesis-canonical convention (v4 §10.1.2)

        A_S^GD    =  - (1/P) * X_train^T @ Gamma @ X_train        in R^{P×P}
        B_S^GD    =  + (1/P) * X_query^T @ Gamma @ X_train        in R^{K×P}
        T_GD      =    I_P + A_S^GD / L.

    Full return contract and validation per v4 §10.1.5.
    """
    D, P, K, B = cfg.D, cfg.P, cfg.K, cfg.B
    if D < 1 or P < 1 or K < 1 or B < 1:
        raise ValueError(f"D, P, K, B must be >= 1; got ({D}, {P}, {K}, {B})")
    if cfg.L < 1:
        raise ValueError(f"L must be >= 1; got {cfg.L}")
    if cfg.sigma < 0:
        raise ValueError(f"sigma must be >= 0; got {cfg.sigma}")
    if cfg.label_norm not in ("sqrt_D", "sqrt_P"):
        raise ValueError(
            f"label_norm must be 'sqrt_D' or 'sqrt_P'; got {cfg.label_norm!r}"
        )

    dtype = _parse_dtype(cfg.dtype)

    Sigma = _build_covariance(D, cfg.Sigma_kind, cfg.Sigma_params, dtype)
    Omega = _build_covariance(D, cfg.Omega_kind, cfg.Omega_params, dtype)
    Gamma = _build_gamma(D, cfg.Gamma_kind, cfg.Gamma_params, dtype)

    # Sample X = [X_train | X_query] with columns ~ N(0, Sigma).
    gen_x = torch.Generator(device="cpu")
    gen_x.manual_seed(int(cfg.seeds["x"]))
    X_full = _sample_gaussian_columns(B, D, P + K, Sigma, gen_x, dtype)
    X_train = X_full[:, :, :P].contiguous()
    X_query = X_full[:, :, P : P + K].contiguous()

    # Sample beta ~ N(0, Omega), one per batch.
    gen_b = torch.Generator(device="cpu")
    gen_b.manual_seed(int(cfg.seeds["beta"]))
    beta = _sample_gaussian_vectors(B, D, Omega, gen_b, dtype)

    # Labels.
    norm_factor = math.sqrt(D) if cfg.label_norm == "sqrt_D" else math.sqrt(P)
    y_train = torch.einsum("bd,bdp->bp", beta, X_train) / norm_factor
    y_query = torch.einsum("bd,bdk->bk", beta, X_query) / norm_factor
    if cfg.sigma > 0:
        gen_n = torch.Generator(device="cpu")
        gen_n.manual_seed(int(cfg.seeds["noise"]))
        y_train = y_train + cfg.sigma * torch.randn(
            B, P, generator=gen_n, dtype=dtype
        )
        y_query = y_query + cfg.sigma * torch.randn(
            B, K, generator=gen_n, dtype=dtype
        )

    # Mask.
    M, _Delta = _build_mask(
        P, K, cfg.mask_kind, cfg.mask_perturbation, cfg.non_gd_kind, dtype
    )

    # Sample-space reduced operators (thesis canonical: v4 §10.1.2).
    # K_train[b, mu, nu] = (1/P) x_mu^T Gamma x_nu, i.e., (X^T Gamma X)[mu, nu] / P.
    K_train = torch.einsum(
        "bim,ij,bjn->bmn", X_train, Gamma, X_train
    ) / P
    A_S_GD = -K_train
    B_S_GD = torch.einsum(
        "bim,ij,bjn->bmn", X_query, Gamma, X_train
    ) / P
    I_P = torch.eye(P, dtype=dtype).unsqueeze(0)
    T_GD = I_P + A_S_GD / cfg.L

    # Perturbed / non-GD variants.
    if cfg.mask_kind == "gd_compatible":
        A_S_theta = A_S_GD.clone()
        B_S_theta = B_S_GD.clone()
    elif cfg.mask_kind == "perturbed":
        # A_S_theta = (M_tr ⊙ K_train) where M_tr = -1 + theta·Delta
        # so A_S_theta = -K_train + theta · (Delta ⊙ K_train).
        if _Delta is None:  # safety; _build_mask returns Delta for 'perturbed'
            raise RuntimeError("internal: Delta missing for perturbed mask")
        # Broadcast Delta (P, P) across the batch dim.
        A_S_theta = A_S_GD + float(cfg.mask_perturbation["theta"]) * (
            _Delta.unsqueeze(0) * K_train
        )
        B_S_theta = B_S_GD.clone()
    elif cfg.mask_kind == "non_gd_control":
        if cfg.non_gd_kind == "signflip_testtest":
            A_S_theta = A_S_GD.clone()
            B_S_theta = -B_S_GD
        elif cfg.non_gd_kind == "nonzero_testblock":
            # Positive-sign A_S: residuals no longer contract.
            A_S_theta = K_train.clone()
            B_S_theta = B_S_GD.clone()
        else:
            raise ValueError(
                f"non_gd_kind must be 'signflip_testtest' or 'nonzero_testblock'; "
                f"got {cfg.non_gd_kind!r}"
            )
    else:
        raise ValueError(f"unknown mask_kind: {cfg.mask_kind!r}")
    T_theta = I_P + A_S_theta / cfg.L

    out: dict[str, Any] = {
        "X_train": X_train,
        "X_query": X_query,
        "y_train": y_train,
        "y_query": y_query,
        "beta": beta,
        "Sigma": Sigma,
        "Omega": Omega,
        "mask": M,
        "mask_kind": cfg.mask_kind,
        "Gamma": Gamma,
        "A_S_GD": A_S_GD,
        "B_S_GD": B_S_GD,
        "T_GD": T_GD,
        "A_S_theta": A_S_theta,
        "B_S_theta": B_S_theta,
        "T_theta": T_theta,
        "label_norm": cfg.label_norm,
        "config": cfg,
        "seeds": dict(cfg.seeds),
    }

    if cfg.return_feature_space:
        # Secondary (D, D) helpers. These are the asymptotic / analytic
        # counterparts used by some A1 / A3 derivations; A1-A4 acceptance is
        # evaluated against the sample-space matrices above.
        A_feat_GD = -Sigma @ Gamma
        B_feat_GD = Sigma @ Gamma
        if cfg.mask_kind == "non_gd_control" and cfg.non_gd_kind == "signflip_testtest":
            A_feat_theta = A_feat_GD
            B_feat_theta = -B_feat_GD
        elif cfg.mask_kind == "non_gd_control" and cfg.non_gd_kind == "nonzero_testblock":
            A_feat_theta = -A_feat_GD
            B_feat_theta = B_feat_GD
        else:
            A_feat_theta = A_feat_GD
            B_feat_theta = B_feat_GD
        out["A_feat_GD"] = A_feat_GD
        out["B_feat_GD"] = B_feat_GD
        out["A_feat_theta"] = A_feat_theta
        out["B_feat_theta"] = B_feat_theta

    return out


# ---------------------------------------------------------------------------
# G1: theorem-B stationary circulant generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class G1Config:
    P: int
    D: int | None = None
    B: int = 1
    query_mode: str = "full_window"
    query_position: int | None = None
    matched_query_realization: str = "independent"
    symbol_kind_tr: str = "power_law"
    symbol_params_tr: dict[str, Any] = field(default_factory=lambda: {"nu": 1.5})
    symbol_kind_te: str = "matched"
    symbol_params_te: dict[str, Any] = field(default_factory=dict)
    task_spec_kind: str = "power_law"
    task_spec_params: dict[str, Any] = field(
        default_factory=lambda: {"nu_beta": 1.75}
    )
    sigma: float = 0.0
    label_norm: str = "sqrt_P"
    exact_mode: bool = True
    sample_data: bool = False
    population_mode: bool = False
    seeds: dict[str, int] = field(
        default_factory=lambda: {"x_tr": 0, "x_te": 1, "beta": 2, "noise": 3}
    )
    dtype: str = "float64"


def g1_generate(cfg: G1Config) -> dict[str, Any]:
    """Theorem-B stationary circulant generator (v4 §10.2).

    Pure data / operator generator. No depth / step-size / trajectory-horizon
    parameters enter here. Exact theorem-B trajectories are obtained from
    :func:`scripts.thesis.utils.metrics.gamma_star_trajectory_circulant`;
    transfer functions are derived there by theorem-B scripts. **No callables**
    are returned.
    """
    P = cfg.P
    D = cfg.D if cfg.D is not None else P
    if P < 1:
        raise ValueError(f"P must be >= 1; got {P}")
    if cfg.exact_mode and D != P:
        raise ValueError(
            f"exact_mode=True requires D == P; got D={D}, P={P}"
        )
    if cfg.population_mode and cfg.sample_data:
        raise ValueError(
            "population_mode=True is incompatible with sample_data=True"
        )
    if (
        cfg.matched_query_realization == "shared"
        and cfg.symbol_kind_te != "matched"
    ):
        raise ValueError(
            "matched_query_realization='shared' requires symbol_kind_te='matched'"
        )
    if cfg.matched_query_realization not in ("independent", "shared"):
        raise ValueError(
            f"matched_query_realization must be 'independent' or 'shared'; "
            f"got {cfg.matched_query_realization!r}"
        )
    if cfg.label_norm not in ("sqrt_D", "sqrt_P"):
        raise ValueError(
            f"label_norm must be 'sqrt_D' or 'sqrt_P'; got {cfg.label_norm!r}"
        )
    if cfg.query_mode not in ("full_window", "single_query"):
        raise ValueError(
            f"query_mode must be 'full_window' or 'single_query'; "
            f"got {cfg.query_mode!r}"
        )

    dtype = _parse_dtype(cfg.dtype)

    s_tr = _build_symbol(P, cfg.symbol_kind_tr, cfg.symbol_params_tr, dtype)
    if cfg.symbol_kind_te == "matched":
        s_te = s_tr.clone()
    else:
        s_te = _build_symbol(
            P,
            cfg.symbol_kind_te,
            cfg.symbol_params_te,
            dtype,
            reference_symbol=s_tr,
        )
    omega = _build_symbol(P, cfg.task_spec_kind, cfg.task_spec_params, dtype)

    Sigma_tr = circulant_from_symbol(s_tr).to(dtype)
    Sigma_te = circulant_from_symbol(s_te).to(dtype)

    out: dict[str, Any] = {
        "s_tr": s_tr,
        "s_te": s_te,
        "omega": omega,
        "Sigma_tr": Sigma_tr,
        "Sigma_te": Sigma_te,
        "label_norm": cfg.label_norm,
        "query_mode": cfg.query_mode,
        "config": cfg,
        "seeds": dict(cfg.seeds),
    }

    if not cfg.sample_data:
        return out

    # Sampled-data path. Draw X_train and then X_query per the query_mode and
    # matched_query_realization rules.
    B = cfg.B
    gen_xtr = torch.Generator(device="cpu")
    gen_xtr.manual_seed(int(cfg.seeds["x_tr"]))
    X_train = _sample_gaussian_columns(B, P, P, Sigma_tr, gen_xtr, dtype)

    matched = cfg.symbol_kind_te == "matched"
    if cfg.query_mode == "full_window":
        K = P
        if matched and cfg.matched_query_realization == "shared":
            X_query = X_train.clone()
        else:
            gen_xte = torch.Generator(device="cpu")
            gen_xte.manual_seed(int(cfg.seeds["x_te"]))
            Sigma_query = Sigma_tr if matched else Sigma_te
            X_query = _sample_gaussian_columns(B, P, K, Sigma_query, gen_xte, dtype)
    else:  # single_query
        K = 1
        gen_xte = torch.Generator(device="cpu")
        gen_xte.manual_seed(int(cfg.seeds["x_te"]))
        Sigma_query = Sigma_tr if matched else Sigma_te
        X_query = _sample_gaussian_columns(B, P, K, Sigma_query, gen_xte, dtype)

    # Teacher beta ~ N(0, F^H diag(omega) F), a real circulant covariance.
    Sigma_beta = circulant_from_symbol(omega).to(dtype)
    gen_b = torch.Generator(device="cpu")
    gen_b.manual_seed(int(cfg.seeds["beta"]))
    beta = _sample_gaussian_vectors(B, P, Sigma_beta, gen_b, dtype)

    # Labels.
    norm_factor = math.sqrt(P) if cfg.label_norm == "sqrt_P" else math.sqrt(D)
    y_train = torch.einsum("bd,bdp->bp", beta, X_train) / norm_factor
    y_query = torch.einsum("bd,bdk->bk", beta, X_query) / norm_factor
    if cfg.sigma > 0:
        gen_n = torch.Generator(device="cpu")
        gen_n.manual_seed(int(cfg.seeds["noise"]))
        y_train = y_train + cfg.sigma * torch.randn(
            B, P, generator=gen_n, dtype=dtype
        )
        y_query = y_query + cfg.sigma * torch.randn(
            B, K, generator=gen_n, dtype=dtype
        )

    out.update(
        {
            "X_train": X_train,
            "X_query": X_query,
            "y_train": y_train,
            "y_query": y_query,
            "beta": beta,
        }
    )
    return out


# ---------------------------------------------------------------------------
# G2: theorem-C band-RRS generator (operator-only + sampled-context)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class G2Config:
    D: int
    partition_kind: str = "equal"
    partition_params: dict[str, Any] = field(
        default_factory=lambda: {"m": 4}
    )
    block_means_lam: tuple[float, ...] | None = None
    block_kappas_lam: tuple[float, ...] | None = None
    block_means_omega: tuple[float, ...] | None = None
    block_kappas_omega: tuple[float, ...] | None = None
    xi_shape: str = "linear"
    spectral_basis_kind: str = "dct2"
    spectral_basis_custom: torch.Tensor | None = None
    label_norm: str = "sqrt_D"
    sigma: float = 0.0
    seeds: dict[str, int] = field(
        default_factory=lambda: {"R": 0, "x": 1, "beta": 2, "noise": 3}
    )
    dtype: str = "float64"


def _g2_operator_core(cfg: G2Config) -> dict[str, Any]:
    """Shared operator-level construction used by both operator-only and
    sampled-context G2 entry points.
    """
    if cfg.D < 1:
        raise ValueError(f"D must be >= 1; got {cfg.D}")
    dtype = _parse_dtype(cfg.dtype)

    partition = _build_partition(cfg.D, cfg.partition_kind, cfg.partition_params)
    n_blocks = partition.n_blocks

    block_means_lam = _resolve_per_block(
        cfg.block_means_lam, n_blocks, 1.0, dtype, "block_means_lam"
    )
    block_kappas_lam = _resolve_per_block(
        cfg.block_kappas_lam, n_blocks, 1.0, dtype, "block_kappas_lam"
    )
    block_means_omega = _resolve_per_block(
        cfg.block_means_omega, n_blocks, 1.0, dtype, "block_means_omega"
    )
    block_kappas_omega = _resolve_per_block(
        cfg.block_kappas_omega, n_blocks, 1.0, dtype, "block_kappas_omega"
    )

    Lambda = mass_preserving_block_spectrum(
        partition,
        block_means_lam,
        block_kappas_lam,
        xi_shape=cfg.xi_shape,
        dtype=dtype,
    )
    Omega = mass_preserving_block_task(
        partition,
        block_means_omega,
        block_kappas_omega,
        xi_shape=cfg.xi_shape,
        dtype=dtype,
    )

    rho_star = (block_kappas_lam - 1.0) / (block_kappas_lam + 1.0)

    F = _build_spectral_basis(
        cfg.D, cfg.spectral_basis_kind, cfg.spectral_basis_custom, dtype
    )

    return {
        "partition": partition,
        "F": F,
        "Lambda": Lambda,
        "Omega": Omega,
        "block_means_lam": block_means_lam,
        "block_kappas_lam": block_kappas_lam,
        "block_means_omega": block_means_omega,
        "block_kappas_omega": block_kappas_omega,
        "rho_star": rho_star,
        "config": cfg,
    }


def g2_generate_operator(cfg: G2Config) -> dict[str, Any]:
    """Theorem-C operator-only generator (v4 §10.3.5). No random draws;
    returns spectra, partition, spectral basis F, per-block kappas/means, and
    rho_star. Consumed by C1-C7 operator-level scripts.
    """
    return _g2_operator_core(cfg)


def g2_generate_sampled(
    cfg: G2Config, n_contexts: int, P: int, K: int
) -> dict[str, Any]:
    """Theorem-C sampled-context generator (v4 §10.3.6). Generates
    **physical-basis** data via

        x_{c, mu} ~ N(0, F^T @ R_c @ diag(Lambda) @ R_c^T @ F),

    where F is the fixed spectral basis and ``R_c`` is per-context block-Haar
    in the spectral basis. Returns the sampled X / y / beta plus every
    operator-level key from :func:`g2_generate_operator`, plus the raw
    per-context rotation ``R`` for diagnostics.
    """
    if n_contexts < 1:
        raise ValueError(f"n_contexts must be >= 1; got {n_contexts}")
    if P < 1 or K < 1:
        raise ValueError(f"P and K must be >= 1; got P={P}, K={K}")
    if cfg.label_norm not in ("sqrt_D", "sqrt_P"):
        raise ValueError(
            f"label_norm must be 'sqrt_D' or 'sqrt_P'; got {cfg.label_norm!r}"
        )

    op = _g2_operator_core(cfg)
    partition = op["partition"]
    F = op["F"]
    Lambda = op["Lambda"]
    Omega = op["Omega"]
    D = partition.D
    dtype = F.dtype

    # Per-context block-Haar rotation in the spectral basis.
    gen_R = torch.Generator(device="cpu")
    gen_R.manual_seed(int(cfg.seeds["R"]))
    R = _sample_block_haar(partition, n_contexts, gen_R, dtype)

    # Sample z in canonical diagonal coordinates: z ~ N(0, diag(Lambda)).
    gen_x = torch.Generator(device="cpu")
    gen_x.manual_seed(int(cfg.seeds["x"]))
    lam_sqrt = Lambda.sqrt()  # (D,)
    z = torch.randn(n_contexts, D, P + K, generator=gen_x, dtype=dtype)
    z = z * lam_sqrt.view(1, D, 1)
    # Apply R_c: tilde_x = R_c @ z.
    tilde_x = torch.einsum("cij,cjn->cin", R, z)
    # Map to physical basis: x = F^T @ tilde_x.
    x = torch.einsum("ij,cjn->cin", F.T, tilde_x)

    X_train = x[:, :, :P].contiguous()
    X_query = x[:, :, P : P + K].contiguous()

    # Teacher beta with covariance F^T @ R @ diag(Omega) @ R^T @ F.
    gen_b = torch.Generator(device="cpu")
    gen_b.manual_seed(int(cfg.seeds["beta"]))
    om_sqrt = Omega.sqrt()
    z_beta = torch.randn(n_contexts, D, generator=gen_b, dtype=dtype)
    z_beta = z_beta * om_sqrt.view(1, D)
    tilde_beta = torch.einsum("cij,cj->ci", R, z_beta)
    beta = torch.einsum("ij,cj->ci", F.T, tilde_beta)

    norm_factor = math.sqrt(D) if cfg.label_norm == "sqrt_D" else math.sqrt(P)
    y_train = torch.einsum("cd,cdp->cp", beta, X_train) / norm_factor
    y_query = torch.einsum("cd,cdk->ck", beta, X_query) / norm_factor
    if cfg.sigma > 0:
        gen_n = torch.Generator(device="cpu")
        gen_n.manual_seed(int(cfg.seeds["noise"]))
        y_train = y_train + cfg.sigma * torch.randn(
            n_contexts, P, generator=gen_n, dtype=dtype
        )
        y_query = y_query + cfg.sigma * torch.randn(
            n_contexts, K, generator=gen_n, dtype=dtype
        )

    out = dict(op)
    out.update(
        {
            "X_train": X_train,
            "X_query": X_query,
            "y_train": y_train,
            "y_query": y_query,
            "beta": beta,
            "R": R,
            "seeds": dict(cfg.seeds),
        }
    )
    return out


def g2_to_spectral_basis(
    X: torch.Tensor, F: torch.Tensor
) -> torch.Tensor:
    """Map physical-basis sampled data to spectral coordinates via ``F @ X``.

    Shape: ``(D, n)`` in -> ``(D, n)`` out, or ``(n_contexts, D, n)`` in ->
    ``(n_contexts, D, n)`` out.

    **WHAT THIS RETURNS.** The spectral-basis coordinates of the sampled data,
    which equal ``R_c @ z`` for this context. That is, the per-context
    block-Haar rotation ``R_c`` in the spectral basis is **still applied** --
    this function does not undo it. The covariance of the output is
    ``R_c @ Lambda @ R_c^T``, NOT the canonical diagonal ``Lambda``.

    **To obtain the canonical diagonal coordinates ``z`` (with covariance
    ``Lambda``),** apply ``R_c^T`` to the output::

        z = R_c.T @ g2_to_spectral_basis(X, F)

    using the per-context ``R_c`` returned as ``'R'`` by
    :func:`g2_generate_sampled`.

    For diagnostics only. Canonical architecture-aligned pipelines consume
    physical-basis data directly and should not call this helper.
    """
    if X.ndim == 2:
        return F @ X
    if X.ndim == 3:
        return torch.einsum("ij,cjn->cin", F, X)
    raise ValueError(
        f"X must be (D, n) or (n_contexts, D, n); got shape {tuple(X.shape)}"
    )


# ---------------------------------------------------------------------------
# G3: refinement-ladder generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class G3Config:
    D: int
    ladder_kind: str = "dyadic"
    ladder_params: dict[str, Any] = field(default_factory=dict)
    reference_partition_index: int = 0
    base_block_means_lam: tuple[float, ...] = ()
    base_block_kappas_lam: tuple[float, ...] = ()
    base_block_means_omega: tuple[float, ...] = ()
    base_block_kappas_omega: tuple[float, ...] = ()
    xi_shape: str = "linear"
    spectral_basis_kind: str = "dct2"
    dtype: str = "float64"


def _compute_level_stats(
    lam: torch.Tensor, omega: torch.Tensor, level: BlockPartition, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Compute per-level block means / kappas / rho_star given the (shared)
    spectrum and a specific partition.
    """
    n = level.n_blocks
    means_lam = torch.zeros(n, dtype=dtype)
    kappas_lam = torch.zeros(n, dtype=dtype)
    means_omega = torch.zeros(n, dtype=dtype)
    kappas_omega = torch.zeros(n, dtype=dtype)
    for b_idx, block in enumerate(level.blocks):
        idx = list(block)
        lb = lam[idx]
        ob = omega[idx]
        means_lam[b_idx] = lb.mean()
        kappas_lam[b_idx] = lb.max() / lb.min()
        means_omega[b_idx] = ob.mean()
        kappas_omega[b_idx] = ob.max() / ob.min()
    rho_star = (kappas_lam - 1.0) / (kappas_lam + 1.0)
    return {
        "block_means_lam": means_lam,
        "block_kappas_lam": kappas_lam,
        "block_means_omega": means_omega,
        "block_kappas_omega": kappas_omega,
        "rho_star": rho_star,
    }


def g3_generate(
    lam: torch.Tensor,
    omega: torch.Tensor,
    ladder: list[BlockPartition],
    *,
    F: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, Any]]:
    """Direct-tensors G3 API (v4 §10.4.2). Given a fixed spectrum
    ``(lam, omega)`` and a refinement chain ``ladder``, return a list of
    per-level dicts. ``Lambda`` and ``Omega`` are BITWISE-identical across
    every level; only the partition changes. A single shared ``F`` is
    attached to every level (DCT-II by default when ``F is None``).

    Asserts at construction time that:

    1. every consecutive ``ladder[j+1]`` refines ``ladder[j]``;
    2. ``Lambda``, ``Omega``, ``F`` are bitwise identical across levels.
    """
    if len(ladder) < 1:
        raise ValueError("ladder must have at least one level")
    D = ladder[0].D
    if lam.ndim != 1 or lam.shape[0] != D:
        raise ValueError(f"lam must have shape ({D},); got {tuple(lam.shape)}")
    if omega.ndim != 1 or omega.shape[0] != D:
        raise ValueError(f"omega must have shape ({D},); got {tuple(omega.shape)}")
    for j, level in enumerate(ladder):
        if level.D != D:
            raise ValueError(
                f"ladder[{j}].D = {level.D} does not match ladder[0].D = {D}"
            )
    for j in range(len(ladder) - 1):
        if not refines(ladder[j + 1], ladder[j]):
            raise ValueError(
                f"ladder[{j + 1}] does not refine ladder[{j}]"
            )

    lam64 = lam.to(dtype).contiguous()
    omega64 = omega.to(dtype).contiguous()

    if F is None:
        F = real_spectral_basis(D, "dct2").to(dtype)
    else:
        F = F.to(dtype).contiguous()

    result: list[dict[str, Any]] = []
    for level in ladder:
        stats = _compute_level_stats(lam64, omega64, level, dtype)
        result.append(
            {
                "partition": level,
                "F": F,
                "Lambda": lam64,
                "Omega": omega64,
                **stats,
            }
        )

    # Post-check: Lambda, Omega, F bitwise identical across levels.
    for j, entry in enumerate(result):
        if not torch.equal(entry["Lambda"], result[0]["Lambda"]):
            raise AssertionError(
                f"internal: Lambda drifted at level {j}; invariant violated"
            )
        if not torch.equal(entry["Omega"], result[0]["Omega"]):
            raise AssertionError(
                f"internal: Omega drifted at level {j}; invariant violated"
            )
        if not torch.equal(entry["F"], result[0]["F"]):
            raise AssertionError(
                f"internal: F drifted at level {j}; invariant violated"
            )

    return result


def g3_generate_from_config(cfg: G3Config) -> list[dict[str, Any]]:
    """Constructive G3 API (v4 §10.4.2). Builds a ladder from
    ``cfg.ladder_kind`` / ``cfg.ladder_params``, constructs ``(lam, omega)``
    via :func:`scripts.thesis.utils.partitions.mass_preserving_block_spectrum`
    at ``ladder[cfg.reference_partition_index]``, and delegates to
    :func:`g3_generate`.
    """
    if cfg.D < 1:
        raise ValueError(f"D must be >= 1; got {cfg.D}")
    dtype = _parse_dtype(cfg.dtype)

    if cfg.ladder_kind == "dyadic":
        J = cfg.ladder_params.get("J")
        ladder = dyadic_ladder(cfg.D, J)
    elif cfg.ladder_kind == "equal_divisors":
        divisors = cfg.ladder_params.get("divisors")
        if divisors is None:
            raise ValueError(
                "ladder_kind='equal_divisors' requires ladder_params['divisors']"
            )
        ladder_candidate = [equal_blocks(cfg.D, int(m)) for m in divisors]
        ladder = custom_ladder(ladder_candidate)
    elif cfg.ladder_kind == "custom":
        ladder_list = cfg.ladder_params.get("ladder")
        if ladder_list is None:
            raise ValueError(
                "ladder_kind='custom' requires ladder_params['ladder']"
            )
        ladder = custom_ladder(list(ladder_list))
    else:
        raise ValueError(
            f"unknown ladder_kind: {cfg.ladder_kind!r}; expected 'dyadic', "
            "'equal_divisors', or 'custom'"
        )

    ref_idx = cfg.reference_partition_index
    if not (0 <= ref_idx < len(ladder)):
        raise ValueError(
            f"reference_partition_index = {ref_idx} out of range [0, {len(ladder)})"
        )
    ref_partition = ladder[ref_idx]
    n_ref = ref_partition.n_blocks

    for name, values in (
        ("base_block_means_lam", cfg.base_block_means_lam),
        ("base_block_kappas_lam", cfg.base_block_kappas_lam),
        ("base_block_means_omega", cfg.base_block_means_omega),
        ("base_block_kappas_omega", cfg.base_block_kappas_omega),
    ):
        if len(values) != n_ref:
            raise ValueError(
                f"{name} must have length {n_ref} (n_blocks at reference level); "
                f"got {len(values)}"
            )

    lam = mass_preserving_block_spectrum(
        ref_partition,
        torch.tensor(cfg.base_block_means_lam, dtype=dtype),
        torch.tensor(cfg.base_block_kappas_lam, dtype=dtype),
        xi_shape=cfg.xi_shape,
        dtype=dtype,
    )
    omega = mass_preserving_block_task(
        ref_partition,
        torch.tensor(cfg.base_block_means_omega, dtype=dtype),
        torch.tensor(cfg.base_block_kappas_omega, dtype=dtype),
        xi_shape=cfg.xi_shape,
        dtype=dtype,
    )

    F = _build_spectral_basis(cfg.D, cfg.spectral_basis_kind, None, dtype)

    return g3_generate(lam, omega, ladder, F=F, dtype=dtype)


__all__ = [
    "cols_to_rows",
    "rows_to_cols",
    "GAConfig",
    "ga_generate",
    "G1Config",
    "g1_generate",
    "G2Config",
    "g2_generate_operator",
    "g2_generate_sampled",
    "g2_to_spectral_basis",
    "G3Config",
    "g3_generate",
    "g3_generate_from_config",
]
