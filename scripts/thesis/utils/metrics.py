"""Metrics for theorem-A / B / C experiments and the conditional scaling-law tier.

Plan correspondence: EXPERIMENT_PLAN_FINAL.MD §4.5 (metrics). Step-1 Generator /
Utility Specification v4 §7.

Contents:

- :func:`reduced_model_error` (v4 §7.1, theorem-A A1)
- :func:`ab_perturbation_bound` (v4 §7.2, theorem-A A2 full ``(A, B)`` bound)
- :func:`gamma_star_trajectory_circulant` (v4 §7.3, theorem-B B1 - the single
  source of truth for the exact per-mode trajectory)
- :func:`mode_trajectory_error`, :func:`transfer_function_error` (v4 §7.3)
- :func:`commutant_violation` (re-exported from :mod:`commutants`)
- :func:`grouped_trajectory_error` (v4 §7.4)
- :func:`oracle_commutant_loss` (v4 §7.5, theorem-C C3 / C6 reference, minimized
  over the refined block-scalar commutant coordinates q_b only)
- :func:`contraction_depth_overlay` (v4 §7.6, theorem-C C7 binding overlay)
- :func:`ood_slope` (v4 §7.7, theorem-B B3)
- :func:`holdout_prediction_error`, :func:`frontier_regret` (v4 §7.8, Phase VI)

The A2 bound is the full additive ``(A, B)`` bound from v4 §7.2: the
B-side term depends only on ``Delta_B``, the A-side term uses the telescoping
identity for ``T^ell`` and depends only on ``Delta_A``. There is no cross term.
The C6 oracle is the minimizer of the population loss over the refined
block-scalar commutant class only (scalar per block), not over generic
block-diagonal matrices.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from scripts.thesis.utils.commutants import commutant_violation
from scripts.thesis.utils.partitions import BlockPartition


# ---------------------------------------------------------------------------
# §7.1 Theorem-A reduced-model error (A1)
# ---------------------------------------------------------------------------


def reduced_model_error(
    f_full: torch.Tensor, f_red: torch.Tensor, eps: float = 1e-30
) -> float:
    """Relative L2 error between the full forward map and the reduced forward map:

        err = ||f_full - f_red||_2 / (||f_full||_2 + eps).

    Acceptance for A1 on a GD-compatible mask: ``err < 1e-10`` in float64.
    """
    if f_full.shape != f_red.shape:
        raise ValueError(
            f"shape mismatch: f_full {tuple(f_full.shape)} vs f_red {tuple(f_red.shape)}"
        )
    num = torch.linalg.vector_norm((f_full - f_red).flatten(), ord=2).item()
    den = torch.linalg.vector_norm(f_full.flatten(), ord=2).item() + float(eps)
    return float(num / den)


# ---------------------------------------------------------------------------
# §7.2 Theorem-A (A, B)-perturbation bound (A2)
# ---------------------------------------------------------------------------


def _scalar_or_tensor(t: torch.Tensor) -> Any:
    """Return a Python float when ``t`` is 0-d; otherwise return ``t`` unchanged."""
    if t.ndim == 0:
        return float(t.item())
    return t


def _partial_sum_powers(M: torch.Tensor, L: int) -> torch.Tensor:
    """Compute ``S = I + M + M^2 + ... + M^(L - 1)`` for ``L >= 1``.

    Accepts ``(P, P)`` or ``(B, P, P)``; output has the same shape.
    """
    if L < 1:
        raise ValueError(f"L must be >= 1; got {L}")
    P = M.shape[-1]
    if M.ndim == 2:
        eye = torch.eye(P, dtype=M.dtype, device=M.device)
        S = eye.clone()
        power = eye.clone()
    elif M.ndim == 3:
        B = M.shape[0]
        eye = torch.eye(P, dtype=M.dtype, device=M.device).expand(B, P, P).contiguous()
        S = eye.clone()
        power = eye.clone()
    else:
        raise ValueError(
            f"expected 2D or 3D tensor; got shape {tuple(M.shape)}"
        )
    for _ in range(L - 1):
        power = power @ M
        S = S + power
    return S


def ab_perturbation_bound(
    A_theta: torch.Tensor,
    A_GD: torch.Tensor,
    B_theta: torch.Tensor,
    B_GD: torch.Tensor,
    T_theta: torch.Tensor,
    T_GD: torch.Tensor,
    L: int,
    y: torch.Tensor,
) -> dict[str, Any]:
    """Full ``(A, B)``-operator perturbation bound from v4 §7.2.

    Forward maps::

        F_theta(y) = (1 / L) * B_theta * sum_{ell=0..L-1} T_theta^ell * y
        F_GD(y)    = (1 / L) * B_GD    * sum_{ell=0..L-1} T_GD^ell    * y

    Exact additive decomposition::

        F_theta(y) - F_GD(y)
            = (1 / L) * (B_theta - B_GD) * sum_ell T_theta^ell * y       [B-side]
              + (1 / L) * B_GD * sum_ell (T_theta^ell - T_GD^ell) * y    [A-side tele.]

    Telescoping (A-side only; ``T_theta - T_GD = Delta_A / L``)::

        T_theta^ell - T_GD^ell
            = (1 / L) * sum_{k=0..ell-1} T_theta^(ell-1-k) * Delta_A * T_GD^k

    Bound::

        ||F_theta - F_GD||_2 <= B_side_bound + A_side_bound

        B_side_bound = ||Delta_B||_op / L * ||S_theta y||_2
        A_side_bound = ||B_GD||_op * ||Delta_A||_op * ||y||_2 / L^2
                       * sum_{ell=0..L-1} U_ell

    with ``S_theta = sum_ell T_theta^ell``,
    ``U_ell = sum_{k=0..ell-1} ||T_theta||_op^(ell-1-k) * ||T_GD||_op^k``,
    ``Delta_A = A_theta - A_GD``, ``Delta_B = B_theta - B_GD``.

    Structural property: the B-side bound depends only on ``Delta_B``, the
    A-side bound depends only on ``Delta_A``. No cross term.

    Accepts unbatched inputs (``A_theta, A_GD, T_theta, T_GD`` of shape
    ``(P, P)``, ``B_theta, B_GD`` of shape ``(K, P)``, ``y`` of shape ``(P,)``)
    or batched inputs (prepending a common batch dim ``B`` to each tensor).
    Returns a dict with keys

        delta_A_op, delta_B_op, S_theta_y_norm, B_side_bound,
        telescoping_coeff, A_side_bound, total_bound, empirical_error

    whose values are Python floats in the unbatched case and 1-D tensors of
    length ``B`` in the batched case.
    """
    L_int = int(L)
    if L_int != L or L_int < 1:
        raise ValueError(f"L must be a positive int; got {L}")
    if y.ndim not in (1, 2):
        raise ValueError(f"y must be (P,) or (B, P); got shape {tuple(y.shape)}")

    # Delta operators and spectral norms
    Delta_A = A_theta - A_GD
    Delta_B = B_theta - B_GD
    delta_A_op = torch.linalg.matrix_norm(Delta_A, ord=2)
    delta_B_op = torch.linalg.matrix_norm(Delta_B, ord=2)
    B_GD_op = torch.linalg.matrix_norm(B_GD, ord=2)
    T_theta_op = torch.linalg.matrix_norm(T_theta, ord=2)
    T_GD_op = torch.linalg.matrix_norm(T_GD, ord=2)

    # Propagator partial sums
    S_theta = _partial_sum_powers(T_theta, L_int)
    S_GD = _partial_sum_powers(T_GD, L_int)

    # Apply to y
    if y.ndim == 1:
        S_theta_y = S_theta @ y
        F_theta = (B_theta @ (S_theta @ y)) / L_int
        F_GD = (B_GD @ (S_GD @ y)) / L_int
    else:
        S_theta_y = torch.einsum("bij,bj->bi", S_theta, y)
        F_theta = torch.einsum("bij,bjk,bk->bi", B_theta, S_theta, y) / L_int
        F_GD = torch.einsum("bij,bjk,bk->bi", B_GD, S_GD, y) / L_int

    S_theta_y_norm = torch.linalg.vector_norm(S_theta_y, ord=2, dim=-1)
    y_norm = torch.linalg.vector_norm(y, ord=2, dim=-1)

    # Telescoping coefficient: sum_{ell=0..L-1} U_ell,
    # U_ell = sum_{k=0..ell-1} ||T_theta||^(ell-1-k) * ||T_GD||^k.  (U_0 = 0.)
    tele = torch.zeros_like(T_theta_op)
    for ell in range(1, L_int):
        U_ell = torch.zeros_like(T_theta_op)
        for k in range(ell):
            U_ell = U_ell + T_theta_op.pow(ell - 1 - k) * T_GD_op.pow(k)
        tele = tele + U_ell

    B_side_bound = delta_B_op * S_theta_y_norm / L_int
    A_side_bound = B_GD_op * delta_A_op * y_norm * tele / (L_int * L_int)
    total_bound = B_side_bound + A_side_bound

    empirical_error = torch.linalg.vector_norm(F_theta - F_GD, ord=2, dim=-1)

    return {
        "delta_A_op": _scalar_or_tensor(delta_A_op),
        "delta_B_op": _scalar_or_tensor(delta_B_op),
        "S_theta_y_norm": _scalar_or_tensor(S_theta_y_norm),
        "B_side_bound": _scalar_or_tensor(B_side_bound),
        "telescoping_coeff": _scalar_or_tensor(tele),
        "A_side_bound": _scalar_or_tensor(A_side_bound),
        "total_bound": _scalar_or_tensor(total_bound),
        "empirical_error": _scalar_or_tensor(empirical_error),
    }


# ---------------------------------------------------------------------------
# §7.3 Theorem-B exact gamma-star trajectory + mode / transfer errors
# ---------------------------------------------------------------------------


def gamma_star_trajectory_circulant(
    s_tr: torch.Tensor,
    omega: torch.Tensor,
    *,
    L: int,
    eta: float,
    T: int,
    gamma0: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Exact per-mode reduced-Gamma trajectory under a circulant training
    covariance (v4 §7.3). The single source of truth for the theorem-B B1
    closure experiments; ``g1_generate`` deliberately does not return this.

    Integrates::

        gamma_k(t + 1)
            = gamma_k(t)
              + eta * omega_k * s_tr_k^2
                * (1 - L^{-1} * s_tr_k * gamma_k(t))^(2*L - 1)

    starting from ``gamma0`` (``0`` by default). Returns a real ``float64``
    tensor of shape ``(T + 1, P)`` with row ``t`` holding ``gamma_k(t)``.
    """
    if s_tr.ndim != 1:
        raise ValueError(f"s_tr must be 1D; got shape {tuple(s_tr.shape)}")
    if omega.shape != s_tr.shape:
        raise ValueError(
            f"omega must match s_tr shape; got {tuple(omega.shape)} vs {tuple(s_tr.shape)}"
        )
    if not s_tr.is_floating_point() or not omega.is_floating_point():
        raise TypeError("s_tr and omega must be real floating-point")
    L_int = int(L)
    if L_int != L or L_int < 1:
        raise ValueError(f"L must be a positive int; got {L}")
    T_int = int(T)
    if T_int != T or T_int < 0:
        raise ValueError(f"T must be a non-negative int; got {T}")

    P = s_tr.shape[0]
    s64 = s_tr.to(torch.float64)
    w64 = omega.to(torch.float64)

    if gamma0 is None:
        g0 = torch.zeros(P, dtype=torch.float64)
    elif isinstance(gamma0, (int, float)):
        g0 = torch.full((P,), float(gamma0), dtype=torch.float64)
    else:
        if gamma0.shape != (P,):
            raise ValueError(
                f"gamma0 must have shape ({P},); got {tuple(gamma0.shape)}"
            )
        g0 = gamma0.to(torch.float64).clone()

    traj = torch.zeros(T_int + 1, P, dtype=torch.float64)
    traj[0] = g0
    exponent = 2 * L_int - 1
    eta_f = float(eta)
    s_sq = s64.pow(2)
    for t in range(T_int):
        g = traj[t]
        step = eta_f * w64 * s_sq * (1.0 - s64 * g / L_int).pow(exponent)
        traj[t + 1] = g + step
    return traj


def mode_trajectory_error(
    gamma_hat: torch.Tensor, gamma_star: torch.Tensor
) -> torch.Tensor:
    """Per-mode relative error

        | gamma_hat - gamma_star | / ( | gamma_star | + 1e-30 )

    element-wise. Output shape matches the inputs (either ``(P,)`` or
    ``(T, P)``).
    """
    if gamma_hat.shape != gamma_star.shape:
        raise ValueError(
            f"shape mismatch: gamma_hat {tuple(gamma_hat.shape)} vs "
            f"gamma_star {tuple(gamma_star.shape)}"
        )
    return (gamma_hat - gamma_star).abs() / (gamma_star.abs() + 1e-30)


def transfer_function_error(
    T_hat: torch.Tensor, T_star: torch.Tensor
) -> float:
    """L2(grid) error between two precomputed transfer functions::

        sqrt( sum_k | T_hat[k] - T_star[k] |^2 ).
    """
    if T_hat.shape != T_star.shape:
        raise ValueError(
            f"shape mismatch: T_hat {tuple(T_hat.shape)} vs T_star {tuple(T_star.shape)}"
        )
    diff = (T_hat - T_star).flatten()
    return float(torch.linalg.vector_norm(diff, ord=2).item())


# ---------------------------------------------------------------------------
# §7.4 Theorem-C commutant + grouped-trajectory metrics
# ---------------------------------------------------------------------------
# commutant_violation is re-exported from scripts.thesis.utils.commutants above.


def grouped_trajectory_error(
    q_hat: torch.Tensor, q_star: torch.Tensor
) -> torch.Tensor:
    """Per-block relative error

        | q_hat_b(t) - q_star_b(t) | / ( | q_star_b(t) | + 1e-30 )

    element-wise. Output shape matches the inputs.
    """
    if q_hat.shape != q_star.shape:
        raise ValueError(
            f"shape mismatch: q_hat {tuple(q_hat.shape)} vs q_star {tuple(q_star.shape)}"
        )
    return (q_hat - q_star).abs() / (q_star.abs() + 1e-30)


# ---------------------------------------------------------------------------
# §7.5 Oracle commutant minimum (C3, C6 reference)
# ---------------------------------------------------------------------------


def oracle_commutant_loss(
    lam: torch.Tensor,
    omega: torch.Tensor,
    partition: BlockPartition,
    L: int,
    *,
    q_init: torch.Tensor | None = None,
    optimizer: str = "lbfgs",
    max_iter: int = 500,
) -> dict[str, Any]:
    """Minimize the theorem-C population loss over the refined block-scalar
    commutant class (v4 §7.5)::

        L(q, L) = sum_b sum_{j in block b} omega_{b,j} * lambda_{b,j}
                  * (1 - q_b * lambda_{b,j} / L)^(2 * L),

    where ``q in R^{n_blocks}`` is a single scalar per block. The optimizer
    acts on the ``q`` vector only -- **not** on a generic block-diagonal
    matrix -- which is the v4 §6.3 / C6 definition of the oracle hybrid:
    direct optimization over the refined commutant class.

    Currently supported ``optimizer``: ``"lbfgs"`` (PyTorch L-BFGS with
    strong-Wolfe line search).

    Returns a dict with keys ``q_star`` (Tensor of shape ``(n_blocks,)``),
    ``loss_star`` (float), ``per_block_loss`` (Tensor of shape ``(n_blocks,)``),
    and ``converged`` (bool).
    """
    if optimizer != "lbfgs":
        raise ValueError(
            f"unsupported optimizer {optimizer!r}; only 'lbfgs' is implemented"
        )
    D = partition.D
    n = partition.n_blocks
    if lam.ndim != 1 or lam.shape[0] != D:
        raise ValueError(f"lam must have shape ({D},); got {tuple(lam.shape)}")
    if omega.ndim != 1 or omega.shape[0] != D:
        raise ValueError(f"omega must have shape ({D},); got {tuple(omega.shape)}")
    if not lam.is_floating_point() or not omega.is_floating_point():
        raise TypeError("lam and omega must be real floating-point")
    L_int = int(L)
    if L_int != L or L_int < 1:
        raise ValueError(f"L must be a positive int; got {L}")

    lam64 = lam.to(torch.float64).detach()
    omega64 = omega.to(torch.float64).detach()

    block_of_arr = torch.tensor(
        [partition.block_of(k) for k in range(D)], dtype=torch.long
    )

    if q_init is None:
        q = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    else:
        if q_init.ndim != 1 or q_init.shape[0] != n:
            raise ValueError(
                f"q_init must have shape ({n},); got {tuple(q_init.shape)}"
            )
        if not q_init.is_floating_point():
            raise TypeError(
                f"q_init must be real floating-point; got dtype {q_init.dtype}"
            )
        q = q_init.to(torch.float64).detach().clone().requires_grad_(True)

    opt = torch.optim.LBFGS(
        [q],
        max_iter=int(max_iter),
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        q_expanded = q[block_of_arr]
        residual = 1.0 - q_expanded * lam64 / L_int
        loss = (omega64 * lam64 * residual.pow(2 * L_int)).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        q_expanded = q[block_of_arr]
        per_mode = omega64 * lam64 * (1.0 - q_expanded * lam64 / L_int).pow(
            2 * L_int
        )
        loss_star = float(per_mode.sum().item())
        per_block_loss = torch.zeros(n, dtype=torch.float64)
        for b_idx, block in enumerate(partition.blocks):
            per_block_loss[b_idx] = per_mode[list(block)].sum()
        grad = q.grad
        grad_norm = float(grad.norm().item()) if grad is not None else float("inf")
        converged = bool(grad_norm < 1e-8)

    return {
        "q_star": q.detach().clone(),
        "loss_star": loss_star,
        "per_block_loss": per_block_loss,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# §7.6 Theorem-C contraction-depth overlay (C7 binding)
# ---------------------------------------------------------------------------


def contraction_depth_overlay(
    kappa_b: torch.Tensor | float | int, L_grid: torch.Tensor
) -> torch.Tensor:
    """Theorem-C contraction-rate overlay (v4 §7.6)::

        (rho_b^*)^(2 * L),   rho_b^* = (kappa_b - 1) / (kappa_b + 1).

    Shape convention:

    - If ``kappa_b`` is a scalar (Python number or 0-d tensor), the result
      has shape ``(len(L_grid),)``.
    - If ``kappa_b`` is a 1-D tensor of shape ``(n_blocks,)``, the result
      has shape ``(n_blocks, len(L_grid))``.

    Output dtype is ``float64``.
    """
    if isinstance(kappa_b, (int, float)):
        kappa_t = torch.tensor(float(kappa_b), dtype=torch.float64)
    elif isinstance(kappa_b, torch.Tensor):
        if not kappa_b.is_floating_point():
            kappa_t = kappa_b.to(torch.float64)
        else:
            kappa_t = kappa_b.to(torch.float64)
    else:
        raise TypeError(
            f"kappa_b must be a float, int, or Tensor; got {type(kappa_b).__name__}"
        )

    if L_grid.ndim != 1:
        raise ValueError(f"L_grid must be 1D; got shape {tuple(L_grid.shape)}")
    L_t = L_grid.to(torch.float64)

    rho = (kappa_t - 1.0) / (kappa_t + 1.0)
    if rho.ndim == 0:
        return rho.pow(2.0 * L_t)
    if rho.ndim == 1:
        return rho.unsqueeze(-1).pow(2.0 * L_t.unsqueeze(0))
    raise ValueError(
        f"kappa_b must be scalar or 1D; got shape {tuple(kappa_t.shape)}"
    )


# ---------------------------------------------------------------------------
# §7.7 OOD slope (B3)
# ---------------------------------------------------------------------------


def ood_slope(
    theta: torch.Tensor,
    loss: torch.Tensor,
    *,
    fit_window: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Log-linear (power-law) fit of ``loss`` vs. ``theta`` (v4 §7.7).

    Internally calls :func:`scripts.thesis.utils.fit_powerlaws.fit_loglog` with
    a user-supplied ``fit_window``; when ``fit_window`` is ``None`` the full
    observed range of ``theta`` is used. Returns
    ``{'slope', 'intercept', 'r2'}``.
    """
    if theta.shape != loss.shape:
        raise ValueError(
            f"shape mismatch: theta {tuple(theta.shape)} vs loss {tuple(loss.shape)}"
        )
    # Lazy import so this module loads before fit_powerlaws exists.
    from scripts.thesis.utils.fit_powerlaws import fit_loglog  # noqa: WPS433

    if fit_window is None:
        fit_window = (float(theta.min().item()), float(theta.max().item()))
    result = fit_loglog(theta, loss, fit_window=fit_window)
    return {
        "slope": float(result["slope"]),
        "intercept": float(result["intercept"]),
        "r2": float(result["r2"]),
    }


# ---------------------------------------------------------------------------
# §7.8 Scaling-law utilities (Phase VI)
# ---------------------------------------------------------------------------


def holdout_prediction_error(
    fit_result: dict[str, Any], x_val: torch.Tensor, y_val: torch.Tensor
) -> dict[str, float]:
    """Evaluate a power-law fit on held-out ``(x_val, y_val)`` points.

    The fit is parameterized by ``slope`` and ``intercept`` as produced by
    :func:`scripts.thesis.utils.fit_powerlaws.fit_loglog`; the model is

        y_pred = exp(intercept) * x_val^slope.

    Returns ``{'median_rel_err', 'max_rel_err'}`` with

        rel_err_k = | y_pred_k - y_val_k | / ( | y_val_k | + 1e-30 ).
    """
    if "slope" not in fit_result or "intercept" not in fit_result:
        raise ValueError("fit_result must contain 'slope' and 'intercept' keys")
    if x_val.shape != y_val.shape:
        raise ValueError(
            f"shape mismatch: x_val {tuple(x_val.shape)} vs y_val {tuple(y_val.shape)}"
        )
    slope = float(fit_result["slope"])
    intercept = float(fit_result["intercept"])
    y_pred = math.exp(intercept) * x_val.pow(slope)
    rel = (y_pred - y_val).abs() / (y_val.abs() + 1e-30)
    return {
        "median_rel_err": float(rel.median().item()),
        "max_rel_err": float(rel.max().item()),
    }


def frontier_regret(
    configs: list[dict[str, Any]],
    loss: torch.Tensor,
    compute: torch.Tensor,
    *,
    predicted_optimum: dict[str, Any],
) -> float:
    """Relative regret of a predicted compute-optimal allocation against the
    observed best loss in the sweep.

    ``predicted_optimum`` must contain an ``'index'`` field pointing into the
    shared ordering of ``configs``, ``loss``, and ``compute``. Returns

        max( 0, loss[idx] - min(loss) ) / min(loss),

    a non-negative Python float.
    """
    if "index" not in predicted_optimum:
        raise ValueError("predicted_optimum must contain an 'index' key")
    n = len(configs)
    if loss.ndim != 1 or loss.shape[0] != n:
        raise ValueError(
            f"loss must be 1D of length {n}; got shape {tuple(loss.shape)}"
        )
    if compute.ndim != 1 or compute.shape[0] != n:
        raise ValueError(
            f"compute must be 1D of length {n}; got shape {tuple(compute.shape)}"
        )
    idx = int(predicted_optimum["index"])
    if not (0 <= idx < n):
        raise ValueError(
            f"predicted_optimum['index'] = {idx} out of range [0, {n})"
        )
    predicted_loss = float(loss[idx].item())
    best_loss = float(loss.min().item())
    if best_loss <= 0:
        raise ValueError(
            f"best_loss must be positive for regret to be meaningful; got {best_loss}"
        )
    return max(0.0, (predicted_loss - best_loss) / best_loss)


__all__ = [
    "reduced_model_error",
    "ab_perturbation_bound",
    "gamma_star_trajectory_circulant",
    "mode_trajectory_error",
    "transfer_function_error",
    "commutant_violation",
    "grouped_trajectory_error",
    "oracle_commutant_loss",
    "contraction_depth_overlay",
    "ood_slope",
    "holdout_prediction_error",
    "frontier_regret",
]
