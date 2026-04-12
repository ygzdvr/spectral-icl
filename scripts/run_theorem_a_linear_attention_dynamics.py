from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "dynamics").exists():
    PROJECT_ROOT = Path.cwd()
DYNAMICS_DIR = PROJECT_ROOT / "dynamics"
UTILS_DIR = PROJECT_ROOT / "utils"
for path in (PROJECT_ROOT, DYNAMICS_DIR, UTILS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import OutputDir

# Prefer direct module imports to avoid package-level circular imports.
try:
    from linear_attention_dynamics import (
        init_params_isotropic,
        model_eval_isotropic,
        reduced_theory_four_var_linear_att_isotropic,
        sample_data_gauss_isotropic,
    )
except Exception:
    from dynamics.linear_attention_dynamics import (  # type: ignore
        init_params_isotropic,
        model_eval_isotropic,
        reduced_theory_four_var_linear_att_isotropic,
        sample_data_gauss_isotropic,
    )


Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]



def _relative_error(a: Tensor, b: Tensor, eps: float | None = None) -> Tensor:
    if eps is None:
        eps = torch.finfo(a.dtype).eps
    denom = torch.maximum(torch.linalg.norm(a), torch.linalg.norm(b)).clamp_min(eps)
    return torch.linalg.norm(a - b) / denom



def _mean_diag(M: Tensor) -> Tensor:
    diag = torch.diagonal(M, offset=0)
    return torch.mean(diag)



def _flatten_test(out: Tensor, p_tr: int) -> Tensor:
    return out[:, p_tr:].reshape(-1)


# -----------------------------------------------------------------------------
# Exact reduced-operator extraction for the isotropic frozen-embedding model
# -----------------------------------------------------------------------------

def extract_gamma_isotropic(
    params_tr: list[Tensor],
    Wy: Tensor,
    *,
    beta_model: float,
) -> dict[str, Tensor]:
    """Extract the exact reduced feature-space operator for model_eval_isotropic.

    For the isotropic frozen-embedding model in linear_attention_dynamics.py,
    the forward pass is

        hx = X W_x^T,
        q  = hx W_q^T,
        k  = hx W_k^T,
        v  = hy W_v^T,
        hy <- hy - (beta_model / L) * (A_masked @ v) / (P_tr * sqrt(d)),
        A  = k q^T.

    If u = W_y and W_v^T u = alpha_v u, the model is exactly equivalent to a
    reduced predictor with

        Gamma = beta_model * alpha_v / sqrt(d) * W_x^T W_k^T W_q W_x.

    In the factorized isotropic theory, the corresponding scalar gamma is

        gamma_fact = beta_model * (w_x^2 * w_q * w_k * w_v),

    where
        w_x = mean(diag(W_x)),
        w_q = mean(diag(W_q)),
        w_k = mean(diag(W_k)),
        w_v = (W_y^T W_v W_y) / sqrt(d).

    Returns both matrix-level and scalar diagnostics.
    """
    W_x, Wq, Wk, Wv = params_tr
    d = W_x.shape[1]
    N = W_x.shape[0]
    device = W_x.device
    dtype = W_x.dtype

    wy = Wy.to(device=device, dtype=dtype)
    wy_norm_sq = torch.dot(wy, wy).clamp_min(torch.finfo(dtype).eps)
    alpha_v = torch.dot(wy, Wv.T @ wy) / wy_norm_sq

    Gamma = beta_model * alpha_v * (W_x.T @ Wk.T @ Wq @ W_x) / math.sqrt(d)

    gamma_eff = torch.trace(Gamma) / d
    I_d = torch.eye(d, device=device, dtype=dtype)
    E_iso_raw = torch.linalg.norm(Gamma - gamma_eff * I_d) / torch.linalg.norm(Gamma).clamp_min(
        torch.finfo(dtype).eps
    )
    E_skew = torch.linalg.norm(Gamma - Gamma.T) / torch.linalg.norm(Gamma).clamp_min(
        torch.finfo(dtype).eps
    )

    wx = _mean_diag(W_x)
    wq = _mean_diag(Wq)
    wk = _mean_diag(Wk)
    wv = alpha_v / math.sqrt(d)
    gamma_fact = beta_model * (wx**2) * wq * wk * wv

    return {
        "Gamma": Gamma,
        "gamma_eff": gamma_eff,
        "gamma_fact": gamma_fact,
        "alpha_v": alpha_v,
        "wx": wx,
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "E_iso_raw": E_iso_raw,
        "E_skew": E_skew,
    }



def reduced_predictor_isotropic(
    Gamma: Tensor,
    X: Tensor,
    y: Tensor,
    *,
    L: int,
    P_test: int,
) -> Tensor:
    """Exact reduced predictor matching model_eval_isotropic output convention.

    The returned tensor matches the *network output convention* of
    model_eval_isotropic: on test positions it is the negative prediction used in
    the training loss (out/gamma + y)^2.

    If X has shape [B, P_tr + P_test, d], then for one batch element,

        K_tr = X_tr Gamma X_tr^T,
        K_te = X_te Gamma X_tr^T.

    The training-token outputs obey

        h^{ell+1} = (I - K_tr / (L P_tr)) h^ell,
        h^0 = y_tr,

    while the test-token outputs are

        f_te = - K_te / (L P_tr) * sum_{ell=0}^{L-1} h^ell.
    """
    device = X.device
    dtype = X.dtype
    Gamma = Gamma.to(device=device, dtype=dtype)

    B, seq_len, d = X.shape
    p_tr = seq_len - P_test
    x_tr = X[:, :p_tr, :]
    x_te = X[:, p_tr:, :]
    y_tr = y[:, :p_tr]

    K_tr = torch.einsum("bpd,df,bqf->bpq", x_tr, Gamma, x_tr)
    K_te = torch.einsum("bpd,df,bqf->bpq", x_te, Gamma, x_tr)

    h = y_tr.clone()
    accum = torch.zeros_like(h)
    step_scale = float(L * p_tr)

    for _ in range(L):
        accum = accum + h
        h = h - torch.bmm(K_tr, h.unsqueeze(-1)).squeeze(-1) / step_scale

    f_te = -torch.bmm(K_te, accum.unsqueeze(-1)).squeeze(-1) / step_scale
    return torch.cat([h, f_te], dim=1)



def audit_isotropic_model(
    params_tr: list[Tensor],
    Wy: Tensor,
    X_dbg: Tensor,
    y_dbg: Tensor,
    *,
    L: int,
    P_test: int,
    beta_model: float,
    gamma_out: float,
    store_outputs: bool = False,
) -> dict[str, Any]:
    """Compute theorem-A diagnostics on a fixed debug batch."""
    with torch.no_grad():
        out_net, _, _ = model_eval_isotropic(
            params_tr,
            Wy,
            X_dbg,
            y_dbg,
            L=L,
            P_test=P_test,
            beta=beta_model,
        )
        gamma_info = extract_gamma_isotropic(params_tr, Wy, beta_model=beta_model)
        Gamma = gamma_info["Gamma"]
        gamma_eff = gamma_info["gamma_eff"]
        Gamma_scalar = gamma_eff * torch.eye(Gamma.shape[0], device=Gamma.device, dtype=Gamma.dtype)

        out_gamma = reduced_predictor_isotropic(Gamma, X_dbg, y_dbg, L=L, P_test=P_test)
        out_scalar = reduced_predictor_isotropic(Gamma_scalar, X_dbg, y_dbg, L=L, P_test=P_test)

        p_tr = X_dbg.shape[1] - P_test
        test_net = _flatten_test(out_net, p_tr)
        test_gamma = _flatten_test(out_gamma, p_tr)
        test_scalar = _flatten_test(out_scalar, p_tr)
        y_test = y_dbg[:, p_tr:].reshape(-1)

        loss_net = torch.mean((test_net / gamma_out + y_test) ** 2)
        loss_gamma = torch.mean((test_gamma / gamma_out + y_test) ** 2)
        loss_scalar = torch.mean((test_scalar / gamma_out + y_test) ** 2)

        eps = torch.finfo(out_net.dtype).eps
        E_reduced_all = _relative_error(out_net, out_gamma, eps=eps)
        E_reduced_test = _relative_error(test_net, test_gamma, eps=eps)
        E_scalar_all = _relative_error(out_gamma, out_scalar, eps=eps)
        E_scalar_test = _relative_error(test_gamma, test_scalar, eps=eps)
        E_gamma_fact = torch.abs(gamma_info["gamma_eff"] - gamma_info["gamma_fact"]) / torch.maximum(
            torch.abs(gamma_info["gamma_eff"]), torch.abs(gamma_info["gamma_fact"])
        ).clamp_min(eps)

        result: dict[str, Any] = {
            "loss_net": float(loss_net.cpu()),
            "loss_gamma": float(loss_gamma.cpu()),
            "loss_scalar": float(loss_scalar.cpu()),
            "E_reduced_all": float(E_reduced_all.cpu()),
            "E_reduced_test": float(E_reduced_test.cpu()),
            "E_scalar_all": float(E_scalar_all.cpu()),
            "E_scalar_test": float(E_scalar_test.cpu()),
            "E_gamma_fact": float(E_gamma_fact.cpu()),
            "E_iso_raw": float(gamma_info["E_iso_raw"].cpu()),
            "E_skew": float(gamma_info["E_skew"].cpu()),
            "gamma_eff": float(gamma_info["gamma_eff"].cpu()),
            "gamma_fact": float(gamma_info["gamma_fact"].cpu()),
            "alpha_v": float(gamma_info["alpha_v"].cpu()),
            "wx": float(gamma_info["wx"].cpu()),
            "wq": float(gamma_info["wq"].cpu()),
            "wk": float(gamma_info["wk"].cpu()),
            "wv": float(gamma_info["wv"].cpu()),
        }
        if store_outputs:
            result["out_net"] = out_net.detach().cpu()
            result["out_gamma"] = out_gamma.detach().cpu()
            result["out_scalar"] = out_scalar.detach().cpu()
            result["Gamma"] = Gamma.detach().cpu()
        return result


# -----------------------------------------------------------------------------
# Training loop with theorem-A auditing
# -----------------------------------------------------------------------------

def train_model_isotropic_with_theorem_a(
    *,
    d: int,
    P_tr: int,
    P_test: int,
    B: int,
    N: int,
    L: int,
    beta_model: float,
    gamma_out: float,
    T: int,
    lr: float,
    lamb: float,
    sigma: float,
    debug_batch: int,
    debug_seed: int,
    audit_every: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    device = torch.device(device)
    params = init_params_isotropic(d, N, sigma=sigma, device=device, dtype=dtype)
    W_x, Wy, Wq, Wk, Wv, _w_out = params

    trainable = [
        torch.nn.Parameter(W_x.clone()),
        torch.nn.Parameter(Wq.clone()),
        torch.nn.Parameter(Wk.clone()),
        torch.nn.Parameter(Wv.clone()),
    ]
    optimizer = torch.optim.SGD(trainable, lr=lr)

    X_dbg, y_dbg = sample_data_gauss_isotropic(
        d,
        debug_batch,
        P_tr,
        P_test,
        seed=debug_seed,
        device=device,
        dtype=dtype,
    )

    losses: list[float] = []
    weight_norms: list[list[float]] = []
    audit_steps: list[int] = []
    audit_history: dict[str, list[float]] = {
        "loss_net": [],
        "loss_gamma": [],
        "loss_scalar": [],
        "E_reduced_all": [],
        "E_reduced_test": [],
        "E_scalar_all": [],
        "E_scalar_test": [],
        "E_gamma_fact": [],
        "E_iso_raw": [],
        "E_skew": [],
        "gamma_eff": [],
        "gamma_fact": [],
        "alpha_v": [],
        "wx": [],
        "wq": [],
        "wk": [],
        "wv": [],
    }

    def append_audit(step: int, *, store_outputs: bool = False) -> dict[str, Any]:
        metrics = audit_isotropic_model(
            [p.detach() for p in trainable],
            Wy.detach(),
            X_dbg,
            y_dbg,
            L=L,
            P_test=P_test,
            beta_model=beta_model,
            gamma_out=gamma_out,
            store_outputs=store_outputs,
        )
        audit_steps.append(step)
        for key in audit_history:
            audit_history[key].append(float(metrics[key]))
        return metrics

    # Save initial diagnostic state.
    init_metrics = append_audit(0, store_outputs=False)

    final_metrics: dict[str, Any] | None = None
    for t in range(T):
        X, y = sample_data_gauss_isotropic(
            d,
            B,
            P_tr,
            P_test,
            seed=t + 1,
            device=device,
            dtype=dtype,
        )

        optimizer.zero_grad(set_to_none=True)
        out, _, _ = model_eval_isotropic(
            trainable,
            Wy,
            X,
            y,
            L=L,
            P_test=P_test,
            beta=beta_model,
        )
        loss = torch.mean((out[:, P_tr:] / gamma_out + y[:, P_tr:]) ** 2)
        reg_loss = loss + lamb * sum(torch.sum(p * p) for p in trainable)
        reg_loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            W_x_cur, Wq_cur, Wk_cur, Wv_cur = trainable
            wn = [
                float(_mean_diag(W_x_cur).cpu()),
                float(_mean_diag(Wq_cur).cpu()),
                float(_mean_diag(Wk_cur).cpu()),
                float(torch.dot(Wy, Wv_cur @ Wy).cpu()),
            ]
        weight_norms.append(wn)

        step = t + 1
        if (step % audit_every == 0) or (step == T):
            final_metrics = append_audit(step, store_outputs=(step == T))

        if t % 100 == 0:
            print(f"step {t} , loss = {float(loss.detach().cpu()):.8f}")

    if final_metrics is None:
        final_metrics = append_audit(T, store_outputs=True)

    return {
        "losses": losses,
        "weight_norms": weight_norms,
        "audit_steps": audit_steps,
        "audit_history": audit_history,
        "init_metrics": init_metrics,
        "final_metrics": final_metrics,
        "final_params": [p.detach().cpu().clone() for p in trainable],
        "Wy": Wy.detach().cpu().clone(),
        "X_debug": X_dbg.detach().cpu().clone(),
        "y_debug": y_dbg.detach().cpu().clone(),
    }


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _setup_style() -> None:
    try:
        import seaborn as sns  # type: ignore

        sns.set(font_scale=1.2)
        sns.set_style("whitegrid")
        sns.set_palette("rocket")
    except Exception:
        plt.style.use("default")



def _plot_identity(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)


# -----------------------------------------------------------------------------
# Main experiment driver
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Theorem-A reparameterization-robustness experiment for isotropic "
            "linear-attention dynamics (alpha = P_train / d = 1 by default)."
        )
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--p-train", type=int, default=128)
    parser.add_argument("--p-test", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--debug-batch", type=int, default=256)
    parser.add_argument("--debug-seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--audit-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lamb", type=float, default=0.0)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--lvals", type=str, default="1,2,4,8")
    parser.add_argument("--lamb-grid-min", type=float, default=0.0025)
    parser.add_argument("--lamb-grid-max", type=float, default=7.0)
    parser.add_argument("--lamb-grid-points", type=int, default=800)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    d = args.d
    N = d
    T = args.steps
    sqrt_d = math.sqrt(d)
    lr_train = args.lr * sqrt_d
    alpha_ctx = float(args.p_train) / float(d)

    _setup_style()
    out = OutputDir(__file__, base=args.output_dir)
    lvals = parse_int_list(args.lvals)

    if abs(alpha_ctx - 1.0) > 1e-12:
        print(f"[warning] alpha = P_train / d = {alpha_ctx:.6f}, not exactly 1.0.")
    if abs(args.beta_model - 1.0) > 1e-12:
        print(
            "[warning] beta_model != 1. The full-theory loss overlay is exact only in the "
            "default beta_model=1 setting of the original script. Reduced-operator audits remain valid."
        )

    # ------------------------------------------------------------------
    # Train the full isotropic model and audit theorem-A quantities.
    # ------------------------------------------------------------------
    all_losses: list[list[float]] = []
    all_weight_norms: list[list[list[float]]] = []
    run_data: dict[int, dict[str, Any]] = {}

    print("=== Training isotropic linear-attention model with theorem-A audits ===")
    for L in lvals:
        print(f"\nL = {L}")
        bundle = train_model_isotropic_with_theorem_a(
            d=d,
            P_tr=args.p_train,
            P_test=args.p_test,
            B=args.batch,
            N=N,
            L=L,
            beta_model=args.beta_model,
            gamma_out=args.gamma,
            T=T,
            lr=lr_train,
            lamb=args.lamb,
            sigma=args.sigma,
            debug_batch=args.debug_batch,
            debug_seed=args.debug_seed,
            audit_every=args.audit_every,
            device=device,
            dtype=dtype,
        )
        run_data[L] = bundle
        all_losses.append(bundle["losses"])
        all_weight_norms.append(bundle["weight_norms"])

    # ------------------------------------------------------------------
    # Run the original four-variable reduced theory for comparison.
    # ------------------------------------------------------------------
    print("\n=== Running four-variable reduced theory ===")
    lamb_grid = torch.linspace(
        args.lamb_grid_min,
        args.lamb_grid_max,
        args.lamb_grid_points,
        device=device,
        dtype=dtype,
    )

    losses_th: list[list[float]] = []
    all_ws_th: list[list[list[float]]] = []
    gamma_th: list[np.ndarray] = []

    for L in lvals:
        print(f"\nL = {L}")
        loss_i, ws_i = reduced_theory_four_var_linear_att_isotropic(
            L,
            alpha_ctx,
            lamb_grid,
            eta=args.lr / sqrt_d,
            T=T,
            sigma=args.sigma,
        )
        losses_th.append(loss_i)
        all_ws_th.append(ws_i)
        gamma_i = np.asarray([args.beta_model * (w[0] ** 2) * w[1] * w[2] * w[3] for w in ws_i], dtype=np.float64)
        gamma_th.append(gamma_i)

    # ------------------------------------------------------------------
    # Assemble arrays for plotting / saving.
    # ------------------------------------------------------------------
    train_t = np.arange(1, T + 1, dtype=np.float64)

    # All runs share the same audit grid.
    audit_steps = np.asarray(run_data[lvals[0]]["audit_steps"], dtype=np.int64)
    for L in lvals[1:]:
        if not np.array_equal(audit_steps, np.asarray(run_data[L]["audit_steps"], dtype=np.int64)):
            raise RuntimeError("Audit grids differ across L; expected a shared audit schedule.")

    def hist_array(key: str) -> np.ndarray:
        return np.asarray([run_data[L]["audit_history"][key] for L in lvals], dtype=np.float64)

    arr_loss_net_dbg = hist_array("loss_net")
    arr_loss_gamma_dbg = hist_array("loss_gamma")
    arr_loss_scalar_dbg = hist_array("loss_scalar")
    arr_E_reduced = hist_array("E_reduced_test")
    arr_E_scalar = hist_array("E_scalar_test")
    arr_E_gamma_fact = hist_array("E_gamma_fact")
    arr_E_iso = hist_array("E_iso_raw")
    arr_E_skew = hist_array("E_skew")
    arr_gamma_eff = hist_array("gamma_eff")
    arr_gamma_fact = hist_array("gamma_fact")

    # ------------------------------------------------------------------
    # Figure 1: theorem-A main figure (loss, reduced exactness, scalarization,
    # gamma-factorization mismatch).
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for i, L in enumerate(lvals):
        loss_arr = np.asarray(all_losses[i], dtype=np.float64)
        loss_th_arr = np.asarray(losses_th[i], dtype=np.float64)
        axs[0, 0].semilogx(train_t, loss_arr / loss_arr[0], color=f"C{i}", label=rf"$L = {L}$")
        axs[0, 0].semilogx(train_t, loss_th_arr / loss_th_arr[0], "--", color="black", alpha=0.7)
    axs[0, 0].semilogx([], [], "--", color="black", label="Four-var theory")
    axs[0, 0].set_xlabel(r"$t$")
    axs[0, 0].set_ylabel(r"$\mathcal{L}(t) / \mathcal{L}(0)$")
    axs[0, 0].set_title(r"Normalized training loss, $\alpha = 1$")
    axs[0, 0].legend()

    for i, L in enumerate(lvals):
        axs[0, 1].semilogy(audit_steps, arr_E_reduced[i], color=f"C{i}", label=rf"$L = {L}$")
    axs[0, 1].set_xlabel(r"$t$")
    axs[0, 1].set_ylabel(r"$E_{\mathrm{red,test}}$")
    axs[0, 1].set_title(r"Reduced-model output gap: $f_{\rm net}$ vs. $f_{\Gamma}$")

    for i, L in enumerate(lvals):
        axs[1, 0].semilogy(audit_steps, arr_E_scalar[i], color=f"C{i}", label=rf"$L = {L}$")
    axs[1, 0].set_xlabel(r"$t$")
    axs[1, 0].set_ylabel(r"$E_{\mathrm{scalar,test}}$")
    axs[1, 0].set_title(r"Scalarization gap: $f_{\Gamma}$ vs. $f_{\gamma I}$")

    for i, L in enumerate(lvals):
        axs[1, 1].semilogy(audit_steps, arr_E_gamma_fact[i], color=f"C{i}", label=rf"$L = {L}$")
    axs[1, 1].set_xlabel(r"$t$")
    axs[1, 1].set_ylabel(r"$|\gamma_{\rm eff} - \gamma_{\rm fact}| / |\gamma_{\rm eff}|$")
    axs[1, 1].set_title(r"Scalar mismatch: $\gamma_{\rm eff}$ vs. $\gamma_{\rm fact}$")

    fig.suptitle(r"Theorem A on isotropic linear-attention dynamics ($\alpha = 1$)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out.png("theorem_a_linear_attention_main"), dpi=200, bbox_inches="tight")
    fig.savefig(out.pdf("theorem_a_linear_attention_main"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('theorem_a_linear_attention_main')}")

    # ------------------------------------------------------------------
    # Figure 2: deepest-L diagnostics (direct function comparison, gamma
    # trajectories, isotropy/skew, weight norms vs. theory).
    # ------------------------------------------------------------------
    L_deep = lvals[-1]
    idx_deep = len(lvals) - 1
    deep_bundle = run_data[L_deep]
    deep_final = deep_bundle["final_metrics"]

    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))

    loss_net_norm = arr_loss_net_dbg[idx_deep] / arr_loss_net_dbg[idx_deep][0]
    loss_gamma_norm = arr_loss_gamma_dbg[idx_deep] / arr_loss_gamma_dbg[idx_deep][0]
    loss_scalar_norm = arr_loss_scalar_dbg[idx_deep] / arr_loss_scalar_dbg[idx_deep][0]
    axs2[0, 0].plot(audit_steps, loss_net_norm, label=r"$f_{\rm net}$")
    axs2[0, 0].plot(audit_steps, loss_gamma_norm, "--", label=r"$f_{\Gamma}$")
    axs2[0, 0].plot(audit_steps, loss_scalar_norm, label=r"$f_{\gamma I}$")
    axs2[0, 0].set_xlabel(r"$t$")
    axs2[0, 0].set_ylabel(r"debug loss / init")
    axs2[0, 0].set_title(rf"Deepest depth $L = {L_deep}$: debug-batch losses")
    axs2[0, 0].legend()

    axs2[0, 1].plot(audit_steps, arr_gamma_eff[idx_deep], label=r"$\gamma_{\rm eff}$")
    axs2[0, 1].plot(audit_steps, arr_gamma_fact[idx_deep], label=r"$\gamma_{\rm fact}$")
    axs2[0, 1].plot(np.arange(1, T + 1), gamma_th[idx_deep], "--", color="black", label=r"$\gamma_{\rm theory}$")
    axs2[0, 1].set_xlabel(r"$t$")
    axs2[0, 1].set_ylabel(r"$\gamma$")
    axs2[0, 1].set_title(rf"Deepest depth $L = {L_deep}$: gamma trajectories")
    axs2[0, 1].legend()

    axs2[1, 0].semilogy(audit_steps, arr_E_iso[idx_deep], label=r"$E_{\rm iso,raw}$")
    axs2[1, 0].semilogy(audit_steps, arr_E_skew[idx_deep], label=r"$E_{\rm skew}$")
    axs2[1, 0].set_xlabel(r"$t$")
    axs2[1, 0].set_ylabel(r"relative error")
    axs2[1, 0].set_title(rf"Deepest depth $L = {L_deep}$: matrix diagnostics")
    axs2[1, 0].legend()

    # Weight norms: theory vs experiment, mirroring the original script.
    labels = [r"$W_x$", r"$W_q$", r"$W_k$", r"$W_v / \sqrt{d}$"]
    ws_final = np.asarray(all_ws_th[idx_deep], dtype=np.float64)  # (T, 4)
    wn_final = np.asarray(all_weight_norms[idx_deep], dtype=np.float64)  # (T, 4)
    for i in range(4):
        vals_th = ws_final[:, i].copy()
        if i == 0:
            vals_th = vals_th / math.sqrt(2.0)
        axs2[1, 1].plot(np.arange(1, T + 1), vals_th, label=labels[i])

        vals_ex = wn_final[:, i].copy()
        if i == 0:
            vals_ex = vals_ex / math.sqrt(2.0)
        elif i == 3:
            vals_ex = vals_ex / sqrt_d
        axs2[1, 1].plot(np.arange(1, T + 1), vals_ex, "--", color="black", alpha=0.55)
    axs2[1, 1].plot([], [], "--", color="black", label="experiment")
    axs2[1, 1].set_xlabel(r"$t$")
    axs2[1, 1].set_ylabel(r"normalized weight / scalar")
    axs2[1, 1].set_title(rf"Deepest depth $L = {L_deep}$: theory vs. experiment")
    axs2[1, 1].legend(loc="best", ncol=2)

    fig2.suptitle(r"Theorem A diagnostics for isotropic linear-attention dynamics", fontsize=18)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(out.png("theorem_a_linear_attention_deepest"), dpi=200, bbox_inches="tight")
    fig2.savefig(out.pdf("theorem_a_linear_attention_deepest"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('theorem_a_linear_attention_deepest')}")

    # ------------------------------------------------------------------
    # Figure 3: direct final-prediction scatter for deepest depth.
    # ------------------------------------------------------------------
    out_net_final = deep_final["out_net"].numpy()
    out_gamma_final = deep_final["out_gamma"].numpy()
    out_scalar_final = deep_final["out_scalar"].numpy()
    p_tr = args.p_train

    x1 = out_net_final[:, p_tr:].reshape(-1)
    y1 = out_gamma_final[:, p_tr:].reshape(-1)
    x2 = out_gamma_final[:, p_tr:].reshape(-1)
    y2 = out_scalar_final[:, p_tr:].reshape(-1)

    fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))
    axs3[0].scatter(x1, y1, s=6, alpha=0.35)
    _plot_identity(axs3[0], x1, y1)
    axs3[0].set_xlabel(r"$f_{\rm net}$")
    axs3[0].set_ylabel(r"$f_{\Gamma}$")
    axs3[0].set_title(rf"Final test outputs, $L = {L_deep}$")

    axs3[1].scatter(x2, y2, s=6, alpha=0.35)
    _plot_identity(axs3[1], x2, y2)
    axs3[1].set_xlabel(r"$f_{\Gamma}$")
    axs3[1].set_ylabel(r"$f_{\gamma I}$")
    axs3[1].set_title(rf"Final test outputs, $L = {L_deep}$")

    fig3.suptitle(r"Direct function comparisons on the fixed debug batch", fontsize=16)
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.savefig(out.png("theorem_a_linear_attention_final_scatter"), dpi=200, bbox_inches="tight")
    fig3.savefig(out.pdf("theorem_a_linear_attention_final_scatter"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('theorem_a_linear_attention_final_scatter')}")

    # ------------------------------------------------------------------
    # Save arrays and artifacts.
    # ------------------------------------------------------------------
    payload: dict[str, np.ndarray] = {
        "lvals": np.asarray(lvals, dtype=np.int64),
        "audit_steps": audit_steps.astype(np.int64),
        "train_t": train_t,
        "gamma_theory_t": np.asarray(gamma_th, dtype=np.float64),
        "loss_net_debug": arr_loss_net_dbg,
        "loss_gamma_debug": arr_loss_gamma_dbg,
        "loss_scalar_debug": arr_loss_scalar_dbg,
        "E_reduced_test": arr_E_reduced,
        "E_scalar_test": arr_E_scalar,
        "E_gamma_fact": arr_E_gamma_fact,
        "E_iso_raw": arr_E_iso,
        "E_skew": arr_E_skew,
        "gamma_eff": arr_gamma_eff,
        "gamma_fact": arr_gamma_fact,
    }
    for i, L in enumerate(lvals):
        payload[f"loss_exp_L_{L}"] = np.asarray(all_losses[i], dtype=np.float64)
        payload[f"loss_theory_L_{L}"] = np.asarray(losses_th[i], dtype=np.float64)
        payload[f"weight_norms_L_{L}"] = np.asarray(all_weight_norms[i], dtype=np.float64)
        payload[f"ws_theory_L_{L}"] = np.asarray(all_ws_th[i], dtype=np.float64)
    np.savez(out.numpy("theorem_a_linear_attention_dynamics_data"), **payload)
    print(f"Saved data to: {out.numpy('theorem_a_linear_attention_dynamics_data')}")

    artifacts = {
        "config": vars(args),
        "lvals": lvals,
        "all_losses": {L: torch.tensor(run_data[L]["losses"], dtype=torch.float64) for L in lvals},
        "all_weight_norms": {L: torch.tensor(run_data[L]["weight_norms"], dtype=torch.float64) for L in lvals},
        "audit_steps": torch.tensor(audit_steps, dtype=torch.int64),
        "audit_history": {
            L: {k: torch.tensor(v, dtype=torch.float64) for k, v in run_data[L]["audit_history"].items()}
            for L in lvals
        },
        "final_params": {L: run_data[L]["final_params"] for L in lvals},
        "Wy": {L: run_data[L]["Wy"] for L in lvals},
        "X_debug": {L: run_data[L]["X_debug"] for L in lvals},
        "y_debug": {L: run_data[L]["y_debug"] for L in lvals},
        "final_metrics": {
            L: {
                k: (v if not isinstance(v, torch.Tensor) else v.clone())
                for k, v in run_data[L]["final_metrics"].items()
            }
            for L in lvals
        },
        "theory_losses": {L: torch.tensor(losses_th[i], dtype=torch.float64) for i, L in enumerate(lvals)},
        "theory_ws": {L: torch.tensor(all_ws_th[i], dtype=torch.float64) for i, L in enumerate(lvals)},
    }
    torch.save(artifacts, out.torch("theorem_a_linear_attention_dynamics_artifacts"))
    print(f"Saved artifacts to: {out.torch('theorem_a_linear_attention_dynamics_artifacts')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
