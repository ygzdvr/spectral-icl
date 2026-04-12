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
if not (PROJECT_ROOT / "configs").exists() or not (PROJECT_ROOT / "dynamics").exists():
    PROJECT_ROOT = Path.cwd()
CONFIGS_DIR = PROJECT_ROOT / "configs"
DYNAMICS_DIR = PROJECT_ROOT / "dynamics"
for path in (PROJECT_ROOT, CONFIGS_DIR, DYNAMICS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import OutputDir
from train_configs import DecoupledTrainModelConfig
from linear_icl_dynamics import model_eval_decoupled_frozen_emb
from sgd_isotropic_dynamics import visualize_loss_landscape


Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]



def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]



def _as_tensor(x: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)



def _randn(shape: tuple[int, ...], seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)



def _build_y_mask(batch: int, seq_len: int, p_tr: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    mask = torch.ones((batch, seq_len), device=device, dtype=dtype)
    mask[:, p_tr:] = 0.0
    return mask



def _build_attn_mask(seq_len: int, p_tr: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    mask = torch.ones((seq_len, seq_len), device=device, dtype=dtype)
    mask[:, p_tr:] = 0.0
    return mask



def _sum_squares(params: list[Tensor]) -> Tensor:
    return sum(torch.sum(p * p) for p in params)



def _relative_error(a: Tensor, b: Tensor, eps: float | None = None) -> Tensor:
    if eps is None:
        eps = torch.finfo(a.dtype).eps
    denom = torch.maximum(torch.linalg.norm(a), torch.linalg.norm(b)).clamp_min(eps)
    return torch.linalg.norm(a - b) / denom


# -----------------------------------------------------------------------------
# Data generation and initialization copied locally from the baseline pipeline.
# We keep these local to avoid circular package-import issues.
# -----------------------------------------------------------------------------


def sample_data(
    d: int,
    B: int,
    P_tr: int,
    P_te: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    device = torch.device(device)
    X = _randn((B, P_tr + P_te, d), 2 * seed, device=device, dtype=dtype)
    betas = _randn((B, d), 2 * seed + 1, device=device, dtype=dtype)
    y = torch.einsum("bpd,bd->bp", X, betas) / math.sqrt(d)
    return X, y



def sample_data_spec(
    spec: Tensor,
    w_star: Tensor,
    B: int,
    P_tr: int,
    P_te: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    del w_star
    device = torch.device(device)
    spec = spec.to(device=device, dtype=dtype)
    d = spec.shape[0]

    X = _randn((B, P_tr + P_te, d), 2 * seed + 1, device=device, dtype=dtype)
    X = X * spec.sqrt().view(1, 1, d)
    betas = _randn((B, d), 2 * seed + 2, device=device, dtype=dtype)
    y = torch.einsum("bpd,bd->bp", X, betas)
    return X, y



def init_pretrain_params(
    d: int,
    N: int,
    *,
    sigma: float = 0.4,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    device = torch.device(device)
    W_x = math.sqrt(2.0) * math.sqrt(N) * sigma * torch.eye(N, d, device=device, dtype=dtype)
    W_y = torch.ones((N,), device=device, dtype=dtype)
    Wq = sigma * math.sqrt(N) * torch.eye(N, device=device, dtype=dtype)
    Wk = Wq.clone()
    Wv = sigma * math.sqrt(N) * torch.eye(N, device=device, dtype=dtype)
    w_out = W_y.clone()
    return [W_x, W_y, Wq, Wk, Wv, w_out]



def make_normalized_powerlaw_problem(
    d: int,
    alpha: float,
    beta: float,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    device = torch.device(device)
    coords = torch.linspace(1, d, d, device=device, dtype=dtype)
    spec = coords.pow(-alpha)
    spec = spec / torch.sum(spec)
    w_star = torch.sqrt(coords.pow(-alpha * beta - 1.0) / spec)
    w_star = w_star / torch.sqrt(torch.sum((w_star ** 2) * spec))
    return spec, w_star



def sample_batch_from_cfg(
    cfg: DecoupledTrainModelConfig,
    *,
    spec: Tensor | None,
    w_star: Tensor | None,
    seed: int,
    B_override: int | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    B = cfg.B if B_override is None else B_override

    if cfg.sample_mode == "iid":
        return sample_data(cfg.d, B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype)

    if spec is None or w_star is None:
        raise ValueError("spec and w_star are required for non-iid sample modes.")

    if cfg.sample_mode == "spec":
        return sample_data_spec(spec, w_star, B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype)

    raise ValueError(f"Unsupported sample_mode for theorem-A ISO closure: {cfg.sample_mode}")


# -----------------------------------------------------------------------------
# Exact theorem-A reduction helpers for the Figure-1 decoupled frozen-embedding
# model.
# -----------------------------------------------------------------------------


def linear_attention_token_mixer(
    W_x: Tensor,
    Wq: Tensor,
    Wk: Tensor,
    X: Tensor,
    P_test: int,
) -> dict[str, Tensor]:
    device = X.device
    dtype = X.dtype
    W_x = _as_tensor(W_x, device=device, dtype=dtype)
    Wq = _as_tensor(Wq, device=device, dtype=dtype)
    Wk = _as_tensor(Wk, device=device, dtype=dtype)

    N, _ = W_x.shape
    seq_len = X.shape[1]
    p_tr = seq_len - P_test

    hx = torch.einsum("btd,nd->btn", X, W_x)
    q = torch.einsum("btn,kn->btk", hx, Wq) / math.sqrt(N)
    k = torch.einsum("btn,kn->btk", hx, Wk) / math.sqrt(N)

    A = torch.einsum("btk,bsk->bts", k, q) / float(N)
    mask = _build_attn_mask(seq_len, p_tr, device=device, dtype=dtype)
    masked_A = A * mask

    H = (W_x.transpose(0, 1) @ Wk.transpose(0, 1) @ Wq @ W_x) / float(N * N)
    A_from_H = torch.einsum("btd,df,bsf->bts", X, H, X)

    return {
        "A": A,
        "masked_A": masked_A,
        "H": H,
        "A_from_H": A_from_H,
    }



def reduced_raw_output_from_gamma(
    X: Tensor,
    y: Tensor,
    Gamma_code: Tensor,
    *,
    L: int,
    P_test: int,
) -> Tensor:
    device = X.device
    dtype = X.dtype
    Gamma_code = _as_tensor(Gamma_code, device=device, dtype=dtype)

    seq_len = X.shape[1]
    p_tr = seq_len - P_test

    X_tr = X[:, :p_tr, :]
    X_te = X[:, p_tr:, :]
    y_tr = y[:, :p_tr]

    K_trtr = torch.einsum("bpd,df,bqf->bpq", X_tr, Gamma_code, X_tr)
    K_tetr = torch.einsum("bpd,df,bqf->bpq", X_te, Gamma_code, X_tr)

    state = y_tr.clone()
    sum_states = torch.zeros_like(state)
    step_scale = 1.0 / (float(L) * float(p_tr))

    for _ in range(L):
        sum_states = sum_states + state
        state = state - step_scale * torch.bmm(K_trtr, state.unsqueeze(-1)).squeeze(-1)

    out_te = -step_scale * torch.bmm(K_tetr, sum_states.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros((X.shape[0], seq_len), device=device, dtype=dtype)
    out[:, :p_tr] = state
    out[:, p_tr:] = out_te
    return out



def trace_theorem_a_decoupled_frozen_emb(
    params_tr: list[Tensor],
    Wy: Tensor,
    X: Tensor,
    y: Tensor,
    *,
    L: int,
    P_test: int,
    beta_model: float,
) -> tuple[Tensor, dict[str, float]]:
    W_x, Wq, Wk, Wv = params_tr
    device = X.device
    dtype = X.dtype
    eps = torch.finfo(dtype).eps

    W_x = _as_tensor(W_x, device=device, dtype=dtype)
    Wq = _as_tensor(Wq, device=device, dtype=dtype)
    Wk = _as_tensor(Wk, device=device, dtype=dtype)
    Wv = _as_tensor(Wv, device=device, dtype=dtype)
    Wy = _as_tensor(Wy, device=device, dtype=dtype)

    N, _ = W_x.shape
    seq_len = X.shape[1]
    p_tr = seq_len - P_test
    batch = X.shape[0]
    eta = float(beta_model) / (float(L) * float(p_tr))

    token = linear_attention_token_mixer(W_x, Wq, Wk, X, P_test)
    A = token["A"]
    masked_A = token["masked_A"]
    H = token["H"]
    A_from_H = token["A_from_H"]

    mask_y = _build_y_mask(batch, seq_len, p_tr, device=device, dtype=dtype)
    hy = torch.einsum("bt,n->btn", y * mask_y, Wy)

    wy_norm_sq = torch.dot(Wy, Wy).clamp_min(eps)
    wvTwy = Wv.transpose(0, 1) @ Wy
    alpha_v = torch.dot(Wy, wvTwy) / wy_norm_sq
    chi_v = alpha_v / math.sqrt(N)

    def output_coord(h: Tensor) -> Tensor:
        return torch.einsum("btn,n->bt", h, Wy) / float(N)

    def decompose(h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        coeff = torch.einsum("btn,n->bt", h, Wy) / wy_norm_sq
        parallel = torch.einsum("bt,n->btn", coeff, Wy)
        residual = h - parallel
        return coeff, parallel, residual

    kernel_err = _relative_error(A, A_from_H, eps).detach()

    o_history: list[Tensor] = [output_coord(hy).detach().clone()]
    exact_err_max = 0.0
    local_err_max = 0.0
    span_err_max = 0.0
    value_align_err_max = 0.0

    for _ in range(L):
        o = output_coord(hy)
        _, _, residual = decompose(hy)
        zeta = torch.einsum("btn,n->bt", residual, wvTwy) / (float(N) * math.sqrt(N))

        v = torch.einsum("btn,kn->btk", hy, Wv) / math.sqrt(N)
        update = torch.bmm(masked_A, v)
        hy_next = hy - eta * update
        o_next = output_coord(hy_next)

        o_exact = o - eta * torch.bmm(masked_A, (chi_v * o + zeta).unsqueeze(-1)).squeeze(-1)
        o_local_red = o - eta * torch.bmm(masked_A, (chi_v * o).unsqueeze(-1)).squeeze(-1)

        exact_err = float(_relative_error(o_next, o_exact, eps).detach().cpu())
        local_err = float(_relative_error(o_next, o_local_red, eps).detach().cpu())
        span_err = float((torch.linalg.norm(residual) / torch.linalg.norm(hy).clamp_min(eps)).detach().cpu())
        value_align_err = float(_relative_error(wvTwy, alpha_v * Wy, eps).detach().cpu())

        exact_err_max = max(exact_err_max, exact_err)
        local_err_max = max(local_err_max, local_err)
        span_err_max = max(span_err_max, span_err)
        value_align_err_max = max(value_align_err_max, value_align_err)

        hy = hy_next
        o_history.append(o_next.detach().clone())

    out = output_coord(hy)

    o_red = o_history[0].to(device=device, dtype=dtype)
    for _ in range(L):
        o_red = o_red - eta * torch.bmm(masked_A, (chi_v * o_red).unsqueeze(-1)).squeeze(-1)

    out_tr = out[:, :p_tr]
    red_tr = o_red[:, :p_tr]
    out_te = out[:, p_tr:]
    red_te = o_red[:, p_tr:]

    trace_summary = {
        "E_kernel": float(kernel_err.cpu()),
        "E_roll_all": float(_relative_error(out, o_red, eps).detach().cpu()),
        "E_roll_train": float(_relative_error(out_tr, red_tr, eps).detach().cpu()),
        "E_roll_test": float(_relative_error(out_te, red_te, eps).detach().cpu()),
        "E_exact_max": exact_err_max,
        "E_local_max": local_err_max,
        "E_span_max": span_err_max,
        "E_value_align_max": value_align_err_max,
    }
    return out, trace_summary



def extract_gamma_code(params_tr: list[Tensor], Wy: Tensor, beta_model: float) -> dict[str, Tensor]:
    W_x, Wq, Wk, Wv = params_tr
    device = W_x.device
    dtype = W_x.dtype

    W_x = _as_tensor(W_x, device=device, dtype=dtype)
    Wq = _as_tensor(Wq, device=device, dtype=dtype)
    Wk = _as_tensor(Wk, device=device, dtype=dtype)
    Wv = _as_tensor(Wv, device=device, dtype=dtype)
    Wy = _as_tensor(Wy, device=device, dtype=dtype)

    N, d = W_x.shape
    wy_norm_sq = torch.dot(Wy, Wy).clamp_min(torch.finfo(dtype).eps)
    alpha_v = torch.dot(Wy, Wv.transpose(0, 1) @ Wy) / wy_norm_sq
    chi_v = alpha_v / math.sqrt(N)

    Gamma_code = beta_model * chi_v * (W_x.transpose(0, 1) @ Wk.transpose(0, 1) @ Wq @ W_x) / float(N * N)
    gamma_eff = torch.trace(Gamma_code) / float(d)
    identity = torch.eye(d, device=device, dtype=dtype)
    eps = torch.finfo(dtype).eps

    E_iso_raw = torch.linalg.norm(Gamma_code - gamma_eff * identity) / torch.linalg.norm(Gamma_code).clamp_min(eps)
    E_skew = torch.linalg.norm(Gamma_code - Gamma_code.transpose(0, 1)) / torch.linalg.norm(Gamma_code).clamp_min(eps)

    return {
        "Gamma_code": Gamma_code,
        "gamma_eff": gamma_eff,
        "chi_v": chi_v,
        "alpha_v": alpha_v,
        "E_iso_raw": E_iso_raw,
        "E_skew": E_skew,
    }



def compute_theorem_a_metrics(
    params_tr: list[Tensor],
    Wy: Tensor,
    X_dbg: Tensor,
    y_dbg: Tensor,
    cfg: DecoupledTrainModelConfig,
) -> dict[str, float | Tensor]:
    device = X_dbg.device
    dtype = X_dbg.dtype
    p_tr = cfg.P_tr
    eps = torch.finfo(dtype).eps

    with torch.no_grad():
        full_raw, trace_summary = trace_theorem_a_decoupled_frozen_emb(
            params_tr,
            Wy,
            X_dbg,
            y_dbg,
            L=cfg.L,
            P_test=cfg.P_test,
            beta_model=cfg.beta_model,
        )

        gamma_info = extract_gamma_code(params_tr, Wy, cfg.beta_model)
        Gamma_code = gamma_info["Gamma_code"]
        gamma_eff = gamma_info["gamma_eff"]
        identity = torch.eye(cfg.d, device=device, dtype=dtype)

        reduced_raw = reduced_raw_output_from_gamma(X_dbg, y_dbg, Gamma_code, L=cfg.L, P_test=cfg.P_test)
        scalar_raw = reduced_raw_output_from_gamma(X_dbg, y_dbg, gamma_eff * identity, L=cfg.L, P_test=cfg.P_test)

        E_matrix_test = _relative_error(full_raw[:, p_tr:], reduced_raw[:, p_tr:], eps)
        E_scalar_test = _relative_error(reduced_raw[:, p_tr:], scalar_raw[:, p_tr:], eps)

        debug_loss_full = torch.mean((full_raw[:, p_tr:] / cfg.gamma + y_dbg[:, p_tr:]) ** 2)
        debug_loss_scalar = torch.mean((scalar_raw[:, p_tr:] / cfg.gamma + y_dbg[:, p_tr:]) ** 2)

        E_A = max(
            trace_summary["E_kernel"],
            trace_summary["E_roll_test"],
            trace_summary["E_exact_max"],
            trace_summary["E_local_max"],
            trace_summary["E_span_max"],
            trace_summary["E_value_align_max"],
            float(E_matrix_test.detach().cpu()),
        )

    return {
        "gamma_eff": float(gamma_eff.detach().cpu()),
        "E_A": float(E_A),
        "E_scalar_test": float(E_scalar_test.detach().cpu()),
        "E_iso_raw": float(gamma_info["E_iso_raw"].detach().cpu()),
        "E_skew": float(gamma_info["E_skew"].detach().cpu()),
        "debug_loss_full": float(debug_loss_full.detach().cpu()),
        "debug_loss_scalar": float(debug_loss_scalar.detach().cpu()),
        "Gamma_code": Gamma_code.detach().cpu(),
    }


# -----------------------------------------------------------------------------
# Scalar landscape helpers
# -----------------------------------------------------------------------------


def evaluate_landscape_curve(
    *,
    alpha_ctx: float,
    L: int,
    lamb_grid: Tensor,
    gamma_ratios: Tensor,
    theory_sigma: float,
) -> Tensor:
    losses = []
    for ratio in gamma_ratios:
        gamma = float(L) * float(ratio.item())
        val = visualize_loss_landscape(gamma, lamb_grid, alpha_ctx, L, sigma=theory_sigma)
        losses.append(val)
    return torch.stack(losses)



def find_gamma_star(
    *,
    alpha_ctx: float,
    L: int,
    lamb_grid: Tensor,
    gamma_ratio_max: float,
    gamma_points: int,
    gamma_refine_points: int,
    theory_sigma: float,
) -> dict[str, Tensor]:
    dtype = lamb_grid.dtype
    device = lamb_grid.device
    gamma_ratios = torch.linspace(0.0, gamma_ratio_max, gamma_points, device=device, dtype=dtype)
    losses = evaluate_landscape_curve(
        alpha_ctx=alpha_ctx,
        L=L,
        lamb_grid=lamb_grid,
        gamma_ratios=gamma_ratios,
        theory_sigma=theory_sigma,
    )
    idx = int(torch.argmin(losses).item())

    lo_idx = max(0, idx - 1)
    hi_idx = min(gamma_points - 1, idx + 1)
    lo = float(gamma_ratios[lo_idx].item())
    hi = float(gamma_ratios[hi_idx].item())

    if hi <= lo:
        gamma_ratio_star = gamma_ratios[idx]
        loss_star = losses[idx]
    else:
        refine_ratios = torch.linspace(lo, hi, gamma_refine_points, device=device, dtype=dtype)
        refine_losses = evaluate_landscape_curve(
            alpha_ctx=alpha_ctx,
            L=L,
            lamb_grid=lamb_grid,
            gamma_ratios=refine_ratios,
            theory_sigma=theory_sigma,
        )
        refine_idx = int(torch.argmin(refine_losses).item())
        gamma_ratio_star = refine_ratios[refine_idx]
        loss_star = refine_losses[refine_idx]

    gamma_star = float(L) * gamma_ratio_star
    return {
        "gamma_ratios": gamma_ratios.detach().cpu(),
        "losses": losses.detach().cpu(),
        "gamma_ratio_star": gamma_ratio_star.detach().cpu(),
        "gamma_star": gamma_star.detach().cpu(),
        "loss_star": loss_star.detach().cpu(),
    }


# -----------------------------------------------------------------------------
# Training with theorem-A closure audits
# -----------------------------------------------------------------------------


def train_with_closure_audits(
    cfg: DecoupledTrainModelConfig,
    *,
    spec: Tensor,
    w_star: Tensor,
    X_dbg: Tensor,
    y_dbg: Tensor,
    gamma_star: float,
    loss_star: float,
    lamb_grid: Tensor,
    alpha_ctx: float,
    theory_sigma: float,
    audit_every: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if cfg.unrestricted:
        raise ValueError("run_theorem_a_loss_landscape expects unrestricted=False.")

    device = torch.device(device)
    params = init_pretrain_params(cfg.d, cfg.N, sigma=cfg.sigma, device=device, dtype=dtype)
    W_x, Wy, Wq, Wk, Wv, _ = params

    trainable = [
        torch.nn.Parameter(W_x.clone()),
        torch.nn.Parameter(Wq.clone()),
        torch.nn.Parameter(Wk.clone()),
        torch.nn.Parameter(Wv.clone()),
    ]
    optimizer = torch.optim.SGD(trainable, lr=cfg.lr)

    train_losses: list[float] = []
    audit_steps: list[int] = []
    audits: list[dict[str, Any]] = []

    def run_audit(step: int) -> None:
        metrics = compute_theorem_a_metrics([p.detach() for p in trainable], Wy.detach(), X_dbg, y_dbg, cfg)
        gamma_eff = float(metrics["gamma_eff"])
        loss_at_gamma_eff = float(
            visualize_loss_landscape(gamma_eff, lamb_grid, alpha_ctx, cfg.L, sigma=theory_sigma).detach().cpu()
        )
        E_gamma = abs(gamma_eff - gamma_star) / max(abs(gamma_star), 1.0e-12)
        delta_L_theory = max(0.0, loss_at_gamma_eff - loss_star)
        audit_steps.append(step)
        audits.append(
            {
                **metrics,
                "gamma_star": float(gamma_star),
                "gamma_ratio_eff": float(gamma_eff / float(cfg.L)),
                "gamma_ratio_star": float(gamma_star / float(cfg.L)),
                "loss_star": float(loss_star),
                "loss_at_gamma_eff": float(loss_at_gamma_eff),
                "E_gamma": float(E_gamma),
                "delta_L_theory": float(delta_L_theory),
            }
        )

    run_audit(0)

    for t in range(cfg.T):
        X, y = sample_batch_from_cfg(
            cfg,
            spec=spec,
            w_star=w_star,
            seed=t if cfg.online else 0,
            device=device,
            dtype=dtype,
        )

        optimizer.zero_grad(set_to_none=True)
        out, _, _ = model_eval_decoupled_frozen_emb(
            trainable,
            Wy,
            X,
            y,
            L=cfg.L,
            P_test=cfg.P_test,
            beta=cfg.beta_model,
            qk_ln=False,
        )

        loss = torch.mean((out[:, cfg.P_tr:] / cfg.gamma + y[:, cfg.P_tr:]) ** 2)
        reg_loss = cfg.N * (cfg.gamma ** 2) * loss + cfg.lamb * _sum_squares(trainable)
        reg_loss.backward()
        optimizer.step()

        train_losses.append(float(loss.detach().cpu()))

        step = t + 1
        if step % audit_every == 0 or step == cfg.T:
            run_audit(step)

    final_params_cpu = [p.detach().cpu().clone() for p in trainable]
    return {
        "train_losses": train_losses,
        "audit_steps": audit_steps,
        "audits": audits,
        "Wy": Wy.detach().cpu().clone(),
        "params_tr": final_params_cpu,
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


def _save_axis_pair(fig: plt.Figure, ax: plt.Axes, out: OutputDir, stem: str, *, pad: float = 0.08) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).expanded(1.0 + pad, 1.0 + pad)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(out.png(stem), dpi=200, bbox_inches=bbox_inches)
    fig.savefig(out.pdf(stem), dpi=200, bbox_inches=bbox_inches)
    print(f"Saved {out.png(stem)}")


def _alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace("-", "m").replace(".", "p")


def _to_np_nested_matrix(results: dict[float, dict[int, dict[str, Any]]], key: str, alphas: list[float], lvals: list[int]) -> np.ndarray:
    arr = np.zeros((len(alphas), len(lvals)), dtype=np.float64)
    for i, alpha in enumerate(alphas):
        for j, L in enumerate(lvals):
            arr[i, j] = float(results[alpha][L]["audits"][-1][key])
    return arr



def _common_audit_steps(results: dict[float, dict[int, dict[str, Any]]], alpha_ref: float, L_ref: int) -> np.ndarray:
    return np.asarray(results[alpha_ref][L_ref]["audit_steps"], dtype=np.int64)



def plot_landscape_curves(
    out: OutputDir,
    *,
    alpha_values: list[float],
    lvals: list[int],
    theory: dict[float, dict[int, dict[str, Tensor]]],
    results: dict[float, dict[int, dict[str, Any]]],
    gamma_ratio_max: float,
) -> None:
    fig, axes = plt.subplots(1, len(alpha_values), figsize=(6.0 * len(alpha_values), 4.8), constrained_layout=True)
    if len(alpha_values) == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, alpha in zip(axes, alpha_values):
        for idx, L in enumerate(lvals):
            color = colors[idx % len(colors)]
            gamma_ratios = theory[alpha][L]["gamma_ratios"].numpy()
            losses = theory[alpha][L]["losses"].numpy()
            gamma_ratio_star = float(theory[alpha][L]["gamma_ratio_star"].item())
            loss_star = float(theory[alpha][L]["loss_star"].item())
            final_audit = results[alpha][L]["audits"][-1]
            gamma_ratio_eff = float(final_audit["gamma_ratio_eff"])
            loss_eff = float(final_audit["loss_at_gamma_eff"])

            ax.plot(gamma_ratios, losses, color=color, label=fr"$L={L}$")
            ax.scatter([gamma_ratio_star], [loss_star], marker="*", s=120, color=color, edgecolors="black", linewidths=0.5)
            ax.scatter([gamma_ratio_eff], [loss_eff], marker="o", s=48, color=color, edgecolors="black", linewidths=0.5)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$\gamma / L$")
        ax.set_title(fr"Loss landscape closure, $\alpha={alpha:.2f}$")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$\mathcal{L}(\gamma)$")
    axes[-1].legend(loc="best", fontsize=9)

    fig.savefig(out.png("theorem_a_loss_landscape_curves"), dpi=200, bbox_inches="tight")
    fig.savefig(out.pdf("theorem_a_loss_landscape_curves"), dpi=200, bbox_inches="tight")
    for ax, alpha in zip(axes, alpha_values):
        _save_axis_pair(fig, ax, out, f"theorem_a_loss_landscape_curves_alpha_{_alpha_tag(alpha)}")



def plot_gamma_closure_dynamics(
    out: OutputDir,
    *,
    alpha_values: list[float],
    lvals: list[int],
    theory: dict[float, dict[int, dict[str, Tensor]]],
    results: dict[float, dict[int, dict[str, Any]]],
) -> None:
    fig, axes = plt.subplots(1, len(alpha_values), figsize=(6.0 * len(alpha_values), 4.8), constrained_layout=True)
    if len(alpha_values) == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, alpha in zip(axes, alpha_values):
        for idx, L in enumerate(lvals):
            color = colors[idx % len(colors)]
            run = results[alpha][L]
            steps = np.asarray(run["audit_steps"], dtype=np.int64)
            gamma_ratio_eff = np.asarray([a["gamma_ratio_eff"] for a in run["audits"]], dtype=np.float64)
            gamma_ratio_star = float(theory[alpha][L]["gamma_ratio_star"].item())

            ax.plot(steps, gamma_ratio_eff, color=color, label=fr"$L={L}$")
            ax.axhline(gamma_ratio_star, color=color, linestyle="--", linewidth=1.2)

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\gamma_{\mathrm{eff}}(t) / L$")
        ax.set_title(fr"$\gamma$-closure dynamics, $\alpha={alpha:.2f}$")
        ax.grid(True, alpha=0.25)

    axes[-1].legend(loc="best", fontsize=9)
    fig.savefig(out.png("theorem_a_gamma_closure_dynamics"), dpi=200, bbox_inches="tight")
    fig.savefig(out.pdf("theorem_a_gamma_closure_dynamics"), dpi=200, bbox_inches="tight")
    for ax, alpha in zip(axes, alpha_values):
        _save_axis_pair(fig, ax, out, f"theorem_a_gamma_closure_dynamics_alpha_{_alpha_tag(alpha)}")



def plot_summary(
    out: OutputDir,
    *,
    alpha_values: list[float],
    lvals: list[int],
    results: dict[float, dict[int, dict[str, Any]]],
) -> None:
    x = np.asarray(alpha_values, dtype=np.float64)
    E_A = _to_np_nested_matrix(results, "E_A", alpha_values, lvals)
    E_gamma = _to_np_nested_matrix(results, "E_gamma", alpha_values, lvals)
    E_scalar = _to_np_nested_matrix(results, "E_scalar_test", alpha_values, lvals)
    delta_L = _to_np_nested_matrix(results, "delta_L_theory", alpha_values, lvals)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    panels = [
        (axes[0, 0], E_A, r"Final exactness $E_A$", True),
        (axes[0, 1], E_gamma, r"Final closure error $E_\gamma$", True),
        (axes[1, 0], E_scalar, r"Final scalarization error $E_{\mathrm{scalar,test}}$", True),
        (axes[1, 1], delta_L, r"Final theory gap $\mathcal{L}(\gamma_{\mathrm{eff}})-\mathcal{L}_\star$", True),
    ]

    for ax, arr, title, logy in panels:
        for idx, L in enumerate(lvals):
            color = colors[idx % len(colors)]
            ax.semilogx(x, arr[:, idx], marker="o", color=color, label=fr"$L={L}$")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha = P_{\mathrm{tr}}/d$")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(loc="best", fontsize=9)
    fig.savefig(out.png("theorem_a_loss_landscape_summary"), dpi=200, bbox_inches="tight")
    fig.savefig(out.pdf("theorem_a_loss_landscape_summary"), dpi=200, bbox_inches="tight")
    _save_axis_pair(fig, axes[0, 0], out, "theorem_a_loss_landscape_summary_exactness")
    _save_axis_pair(fig, axes[0, 1], out, "theorem_a_loss_landscape_summary_closure_error")
    _save_axis_pair(fig, axes[1, 0], out, "theorem_a_loss_landscape_summary_scalarization")
    _save_axis_pair(fig, axes[1, 1], out, "theorem_a_loss_landscape_summary_theory_gap")


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Theorem-A closure experiment for ISO Figure-1 training dynamics using the "
            "Figure-2 scalar loss landscape."
        )
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--p-test", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--debug-batch", type=int, default=64)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--spec-alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=0.125)
    parser.add_argument("--lamb", type=float, default=1.0e-14)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4, help="Parameter initialization scale.")
    parser.add_argument("--theory-sigma", type=float, default=0.0, help="Label-noise level in L(gamma; alpha, L).")
    parser.add_argument("--alphas", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16")
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--debug-seed", type=int, default=1234)
    parser.add_argument("--lamb-min", type=float, default=0.01)
    parser.add_argument("--lamb-max", type=float, default=10.0)
    parser.add_argument("--lamb-points", type=int, default=400)
    parser.add_argument("--gamma-ratio-max", type=float, default=1.25)
    parser.add_argument("--gamma-points", type=int, default=400)
    parser.add_argument("--gamma-refine-points", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    requested_device = torch.device(args.device)
    print(f"requested device: {requested_device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if requested_device.type == "cuda" and torch.cuda.is_available():
        current_idx = requested_device.index if requested_device.index is not None else torch.cuda.current_device()
        print(f"using cuda device: cuda:{current_idx}")
        print(f"cuda device name: {torch.cuda.get_device_name(current_idx)}")
    elif requested_device.type == "cuda":
        print("CUDA was requested but is not available.")

    out = OutputDir(__file__, base=args.output_dir)
    _setup_style()

    alpha_values_req = parse_float_list(args.alphas)
    lvals = parse_int_list(args.lvals)

    spec, w_star = make_normalized_powerlaw_problem(
        args.d,
        args.spec_alpha,
        args.beta,
        device=device,
        dtype=dtype,
    )

    lamb_grid = torch.linspace(args.lamb_min, args.lamb_max, args.lamb_points, device=device, dtype=dtype)

    alpha_values: list[float] = []
    p_trs: list[int] = []
    theory: dict[float, dict[int, dict[str, Tensor]]] = {}
    debug_batches: dict[float, tuple[Tensor, Tensor]] = {}
    results: dict[float, dict[int, dict[str, Any]]] = {}

    print("=== Building scalar loss landscapes and debug batches ===")
    for alpha_req in alpha_values_req:
        p_tr = max(1, int(round(alpha_req * args.d)))
        alpha_ctx = float(p_tr) / float(args.d)
        alpha_values.append(alpha_ctx)
        p_trs.append(p_tr)
        theory[alpha_ctx] = {}
        results[alpha_ctx] = {}

        cfg_dbg = DecoupledTrainModelConfig(
            d=args.d,
            P_tr=p_tr,
            P_test=args.p_test,
            B=args.batch,
            N=args.n,
            L=lvals[0],
            beta_model=args.beta_model,
            gamma=args.gamma,
            T=args.steps,
            lr=args.lr,
            lamb=args.lamb,
            alpha=args.spec_alpha,
            beta=args.beta,
            sigma=args.sigma,
            unrestricted=False,
            online=True,
            sample_mode="spec",
        )
        X_dbg, y_dbg = sample_batch_from_cfg(
            cfg_dbg,
            spec=spec,
            w_star=w_star,
            seed=args.debug_seed,
            B_override=args.debug_batch,
            device=device,
            dtype=dtype,
        )
        debug_batches[alpha_ctx] = (X_dbg, y_dbg)

        print(f"alpha requested={alpha_req:.4f}, actual={alpha_ctx:.4f}, P_tr={p_tr}")
        for L in lvals:
            theory[alpha_ctx][L] = find_gamma_star(
                alpha_ctx=alpha_ctx,
                L=L,
                lamb_grid=lamb_grid,
                gamma_ratio_max=args.gamma_ratio_max,
                gamma_points=args.gamma_points,
                gamma_refine_points=args.gamma_refine_points,
                theory_sigma=args.theory_sigma,
            )
            print(
                f"  L={L:>2d}: gamma_star/L={float(theory[alpha_ctx][L]['gamma_ratio_star']):.6f}, "
                f"L_star={float(theory[alpha_ctx][L]['loss_star']):.6e}"
            )

    print("\n=== Training theorem-A closure runs ===")
    for alpha_ctx, p_tr in zip(alpha_values, p_trs):
        X_dbg, y_dbg = debug_batches[alpha_ctx]
        for L in lvals:
            print(f"\nalpha={alpha_ctx:.4f}, P_tr={p_tr}, L={L}")
            cfg = DecoupledTrainModelConfig(
                d=args.d,
                P_tr=p_tr,
                P_test=args.p_test,
                B=args.batch,
                N=args.n,
                L=L,
                beta_model=args.beta_model,
                gamma=args.gamma,
                T=args.steps,
                lr=args.lr,
                lamb=args.lamb,
                alpha=args.spec_alpha,
                beta=args.beta,
                sigma=args.sigma,
                unrestricted=False,
                online=True,
                sample_mode="spec",
            )

            gamma_star = float(theory[alpha_ctx][L]["gamma_star"].item())
            loss_star = float(theory[alpha_ctx][L]["loss_star"].item())

            run = train_with_closure_audits(
                cfg,
                spec=spec,
                w_star=w_star,
                X_dbg=X_dbg,
                y_dbg=y_dbg,
                gamma_star=gamma_star,
                loss_star=loss_star,
                lamb_grid=lamb_grid,
                alpha_ctx=alpha_ctx,
                theory_sigma=args.theory_sigma,
                audit_every=args.audit_every,
                device=device,
                dtype=dtype,
            )
            results[alpha_ctx][L] = run

            final_audit = run["audits"][-1]
            print(
                "  final: "
                f"E_A={final_audit['E_A']:.3e}, "
                f"gamma_eff/L={final_audit['gamma_ratio_eff']:.6f}, "
                f"gamma_star/L={final_audit['gamma_ratio_star']:.6f}, "
                f"E_gamma={final_audit['E_gamma']:.3e}, "
                f"E_scalar_test={final_audit['E_scalar_test']:.3e}, "
                f"DeltaL={final_audit['delta_L_theory']:.3e}"
            )

    # ------------------------------------------------------------------
    # Save figures
    # ------------------------------------------------------------------
    plot_landscape_curves(
        out,
        alpha_values=alpha_values,
        lvals=lvals,
        theory=theory,
        results=results,
        gamma_ratio_max=args.gamma_ratio_max,
    )
    plot_gamma_closure_dynamics(
        out,
        alpha_values=alpha_values,
        lvals=lvals,
        theory=theory,
        results=results,
    )
    plot_summary(
        out,
        alpha_values=alpha_values,
        lvals=lvals,
        results=results,
    )

    # ------------------------------------------------------------------
    # Save compact numeric arrays
    # ------------------------------------------------------------------
    audit_steps = _common_audit_steps(results, alpha_values[0], lvals[0])
    A = len(alpha_values)
    Lc = len(lvals)
    S = len(audit_steps)
    G = args.gamma_points

    landscape_gamma_ratios = np.zeros((A, Lc, G), dtype=np.float64)
    landscape_losses = np.zeros((A, Lc, G), dtype=np.float64)
    gamma_star_arr = np.zeros((A, Lc), dtype=np.float64)
    gamma_ratio_star_arr = np.zeros((A, Lc), dtype=np.float64)
    loss_star_arr = np.zeros((A, Lc), dtype=np.float64)
    gamma_eff_traj = np.zeros((A, Lc, S), dtype=np.float64)
    gamma_ratio_eff_traj = np.zeros((A, Lc, S), dtype=np.float64)
    E_A_traj = np.zeros((A, Lc, S), dtype=np.float64)
    E_gamma_traj = np.zeros((A, Lc, S), dtype=np.float64)
    E_scalar_traj = np.zeros((A, Lc, S), dtype=np.float64)
    E_iso_raw_traj = np.zeros((A, Lc, S), dtype=np.float64)
    E_skew_traj = np.zeros((A, Lc, S), dtype=np.float64)
    delta_L_traj = np.zeros((A, Lc, S), dtype=np.float64)
    debug_loss_full_traj = np.zeros((A, Lc, S), dtype=np.float64)
    debug_loss_scalar_traj = np.zeros((A, Lc, S), dtype=np.float64)

    for i, alpha in enumerate(alpha_values):
        for j, L in enumerate(lvals):
            theory_ij = theory[alpha][L]
            run_ij = results[alpha][L]
            landscape_gamma_ratios[i, j] = theory_ij["gamma_ratios"].numpy()
            landscape_losses[i, j] = theory_ij["losses"].numpy()
            gamma_star_arr[i, j] = float(theory_ij["gamma_star"].item())
            gamma_ratio_star_arr[i, j] = float(theory_ij["gamma_ratio_star"].item())
            loss_star_arr[i, j] = float(theory_ij["loss_star"].item())

            for s_idx, audit in enumerate(run_ij["audits"]):
                gamma_eff_traj[i, j, s_idx] = float(audit["gamma_eff"])
                gamma_ratio_eff_traj[i, j, s_idx] = float(audit["gamma_ratio_eff"])
                E_A_traj[i, j, s_idx] = float(audit["E_A"])
                E_gamma_traj[i, j, s_idx] = float(audit["E_gamma"])
                E_scalar_traj[i, j, s_idx] = float(audit["E_scalar_test"])
                E_iso_raw_traj[i, j, s_idx] = float(audit["E_iso_raw"])
                E_skew_traj[i, j, s_idx] = float(audit["E_skew"])
                delta_L_traj[i, j, s_idx] = float(audit["delta_L_theory"])
                debug_loss_full_traj[i, j, s_idx] = float(audit["debug_loss_full"])
                debug_loss_scalar_traj[i, j, s_idx] = float(audit["debug_loss_scalar"])

    np.savez(
        out.numpy("theorem_a_loss_landscape_data"),
        alpha_req=np.asarray(alpha_values_req, dtype=np.float64),
        alpha_actual=np.asarray(alpha_values, dtype=np.float64),
        p_trs=np.asarray(p_trs, dtype=np.int64),
        lvals=np.asarray(lvals, dtype=np.int64),
        audit_steps=audit_steps,
        landscape_gamma_ratios=landscape_gamma_ratios,
        landscape_losses=landscape_losses,
        gamma_star=gamma_star_arr,
        gamma_ratio_star=gamma_ratio_star_arr,
        loss_star=loss_star_arr,
        gamma_eff_traj=gamma_eff_traj,
        gamma_ratio_eff_traj=gamma_ratio_eff_traj,
        E_A_traj=E_A_traj,
        E_gamma_traj=E_gamma_traj,
        E_scalar_test_traj=E_scalar_traj,
        E_iso_raw_traj=E_iso_raw_traj,
        E_skew_traj=E_skew_traj,
        delta_L_theory_traj=delta_L_traj,
        debug_loss_full_traj=debug_loss_full_traj,
        debug_loss_scalar_traj=debug_loss_scalar_traj,
    )

    # ------------------------------------------------------------------
    # Save richer artifacts
    # ------------------------------------------------------------------
    artifacts: dict[str, Any] = {
        "config": vars(args),
        "alpha_req": alpha_values_req,
        "alpha_actual": alpha_values,
        "p_trs": p_trs,
        "lvals": lvals,
        "spec": spec.detach().cpu(),
        "w_star": w_star.detach().cpu(),
        "lamb_grid": lamb_grid.detach().cpu(),
        "theory": theory,
        "results": results,
        "debug_batches": {alpha: (X.detach().cpu(), y.detach().cpu()) for alpha, (X, y) in debug_batches.items()},
    }
    torch.save(artifacts, out.torch("theorem_a_loss_landscape_artifacts"))

    print(f"Saved: {out.png('theorem_a_loss_landscape_curves')}")
    print(f"Saved: {out.png('theorem_a_gamma_closure_dynamics')}")
    print(f"Saved: {out.png('theorem_a_loss_landscape_summary')}")
    print(f"Saved: {out.numpy('theorem_a_loss_landscape_data')}")
    print(f"Saved: {out.torch('theorem_a_loss_landscape_artifacts')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
