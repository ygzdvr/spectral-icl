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

from train_configs import DecoupledTrainModelConfig, IsotropicDepthAlphaSweepConfig
from linear_icl_dynamics import model_eval_decoupled_frozen_emb


Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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
# Data generation (copied from the baseline pipeline, but kept local to avoid
# circular package-import issues)
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


def sample_data_spec_rotate(
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
    device = torch.device(device)
    spec = spec.to(device=device, dtype=dtype)
    w_star = w_star.to(device=device, dtype=dtype)
    d = spec.shape[0]

    X = _randn((B, P_tr + P_te, d), 2 * seed + 1, device=device, dtype=dtype)
    X = X * spec.sqrt().view(1, 1, d)

    O = torch.linalg.qr(_randn((B, d, d), 2 * seed, device=device, dtype=dtype)).Q
    betas = torch.einsum("bij,j->bi", O, w_star)
    y = torch.einsum("bpd,bd->bp", X, betas)
    return X, y


def sample_data_gauss_rotate(
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
    device = torch.device(device)
    spec = spec.to(device=device, dtype=dtype)
    w_star = w_star.to(device=device, dtype=dtype)
    d = spec.shape[0]

    O = _randn((B, d, d), 2 * seed, device=device, dtype=dtype) / math.sqrt(d)
    X = _randn((B, P_tr + P_te, d), 2 * seed + 1, device=device, dtype=dtype)
    X = X * spec.sqrt().view(1, 1, d)
    X = torch.einsum("bpk,blk->bpl", X, O)

    betas = torch.einsum("bij,j->bi", O, w_star)
    y = torch.einsum("bpd,bd->bp", X, betas)
    return X, y


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
    if cfg.sample_mode == "spec_rotate":
        return sample_data_spec_rotate(spec, w_star, B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype)
    if cfg.sample_mode == "gauss_rotate":
        return sample_data_gauss_rotate(spec, w_star, B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype)

    raise ValueError(f"Unsupported sample_mode: {cfg.sample_mode}")


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
    w_star = w_star / torch.sqrt(torch.sum((w_star**2) * spec))
    return spec, w_star


def isotropic_dmft(
    alpha: float | Tensor,
    gamma: float | Tensor,
    T: int,
    *,
    iters: int = 100,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    device = torch.device(device)
    alpha_t = torch.as_tensor(alpha, device=device, dtype=dtype)
    gamma_t = torch.as_tensor(gamma, device=device, dtype=dtype)

    ones_lower = torch.ones((T, T), device=device, dtype=dtype)
    theta = gamma_t * torch.tril(ones_lower, diagonal=-1)
    eye = torch.eye(T, device=device, dtype=dtype)

    H = torch.linalg.inv(eye + theta)
    for _ in range(iters):
        inner = torch.linalg.inv(eye + (H @ theta) / alpha_t)
        H = torch.linalg.inv(eye + theta @ inner)

    vs = H @ torch.ones((T,), device=device, dtype=dtype)
    return vs


# -----------------------------------------------------------------------------
# Theorem A diagnostics
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
        "hx": hx,
        "q": q,
        "k": k,
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
    store_full_tensors: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    W_x, Wq, Wk, Wv = params_tr
    device = X.device
    dtype = X.dtype
    eps = torch.finfo(dtype).eps

    W_x = _as_tensor(W_x, device=device, dtype=dtype)
    Wy = _as_tensor(Wy, device=device, dtype=dtype)
    Wq = _as_tensor(Wq, device=device, dtype=dtype)
    Wk = _as_tensor(Wk, device=device, dtype=dtype)
    Wv = _as_tensor(Wv, device=device, dtype=dtype)

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

    o_history = [output_coord(hy).detach().clone()]
    layer_metrics: list[dict[str, float]] = []
    zeta_history = []

    for _ in range(L):
        o = output_coord(hy)
        _, _, residual = decompose(hy)

        zeta = torch.einsum("btn,n->bt", residual, wvTwy) / (float(N) * math.sqrt(N))
        zeta_history.append(zeta.detach().clone())

        v = torch.einsum("btn,kn->btk", hy, Wv) / math.sqrt(N)
        update = torch.bmm(masked_A, v)
        hy_next = hy - eta * update
        o_next = output_coord(hy_next)

        o_exact = o - eta * torch.bmm(masked_A, (chi_v * o + zeta).unsqueeze(-1)).squeeze(-1)
        o_local_red = o - eta * torch.bmm(masked_A, (chi_v * o).unsqueeze(-1)).squeeze(-1)

        exact_err = _relative_error(o_next, o_exact, eps).detach()
        local_err = _relative_error(o_next, o_local_red, eps).detach()
        span_err = (torch.linalg.norm(residual) / torch.linalg.norm(hy).clamp_min(eps)).detach()
        value_align_err = _relative_error(wvTwy, alpha_v * Wy, eps).detach()

        layer_metrics.append(
            {
                "exact_err": float(exact_err.cpu()),
                "local_err": float(local_err.cpu()),
                "span_err": float(span_err.cpu()),
                "value_align_err": float(value_align_err.cpu()),
                "chi_v": float(chi_v.detach().cpu()),
                "alpha_v": float(alpha_v.detach().cpu()),
            }
        )

        hy = hy_next
        o_history.append(o_next.detach().clone())

    out = output_coord(hy)

    o_red = o_history[0].to(device=device, dtype=dtype)
    for _ in range(L):
        o_red = o_red - eta * torch.bmm(masked_A, (chi_v * o_red).unsqueeze(-1)).squeeze(-1)

    roll_err_all = _relative_error(out, o_red, eps).detach()

    out_tr = out[:, :p_tr]
    red_tr = o_red[:, :p_tr]
    out_te = out[:, p_tr:]
    red_te = o_red[:, p_tr:]

    roll_err_train = _relative_error(out_tr, red_tr, eps).detach()
    roll_err_test = _relative_error(out_te, red_te, eps).detach()

    trace: dict[str, Any] = {
        "kernel_err": float(kernel_err.cpu()),
        "roll_err_all": float(roll_err_all.cpu()),
        "roll_err_train": float(roll_err_train.cpu()),
        "roll_err_test": float(roll_err_test.cpu()),
        "layer_metrics": layer_metrics,
        "chi_v": float(chi_v.detach().cpu()),
        "alpha_v": float(alpha_v.detach().cpu()),
        "H": H.detach().cpu(),
    }

    if store_full_tensors:
        trace["A"] = A.detach().cpu()
        trace["A_from_H"] = A_from_H.detach().cpu()
        trace["masked_A"] = masked_A.detach().cpu()
        trace["o_history"] = [t.cpu() for t in o_history]
        trace["o_red_final"] = o_red.detach().cpu()
        trace["zeta_history"] = [t.cpu() for t in zeta_history]
        trace["Wy"] = Wy.detach().cpu()
        trace["wvTwy"] = wvTwy.detach().cpu()
        trace["out"] = out.detach().cpu()

    return out, trace


def summarize_theorem_a_trace(trace: dict[str, Any]) -> dict[str, float]:
    layer_metrics = trace["layer_metrics"]
    return {
        "E_kernel": trace["kernel_err"],
        "E_roll_all": trace["roll_err_all"],
        "E_roll_train": trace["roll_err_train"],
        "E_roll_test": trace["roll_err_test"],
        "E_exact_max": max(m["exact_err"] for m in layer_metrics),
        "E_local_max": max(m["local_err"] for m in layer_metrics),
        "E_span_max": max(m["span_err"] for m in layer_metrics),
        "E_value_align_max": max(m["value_align_err"] for m in layer_metrics),
        "chi_v": trace["chi_v"],
        "alpha_v": trace["alpha_v"],
    }


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
    wy_norm_sq = torch.dot(Wy, Wy)
    alpha_v = torch.dot(Wy, Wv.transpose(0, 1) @ Wy) / wy_norm_sq
    chi_v = alpha_v / math.sqrt(N)

    H = (W_x.transpose(0, 1) @ Wk.transpose(0, 1) @ Wq @ W_x) / float(N * N)
    Gamma_code = beta_model * chi_v * H
    Gamma_sym = 0.5 * (Gamma_code + Gamma_code.transpose(0, 1))
    gamma_eff = torch.trace(Gamma_sym) / float(d)
    identity = torch.eye(d, device=device, dtype=dtype)
    eps = torch.finfo(dtype).eps

    E_iso_sym = torch.linalg.norm(Gamma_sym - gamma_eff * identity) / torch.linalg.norm(Gamma_sym).clamp_min(eps)
    E_iso_raw = torch.linalg.norm(Gamma_code - gamma_eff * identity) / torch.linalg.norm(Gamma_code).clamp_min(eps)
    E_skew = torch.linalg.norm(Gamma_code - Gamma_code.transpose(0, 1)) / torch.linalg.norm(Gamma_code).clamp_min(eps)

    return {
        "Gamma_code": Gamma_code,
        "Gamma_sym": Gamma_sym,
        "gamma_eff": gamma_eff,
        "alpha_v": alpha_v,
        "chi_v": chi_v,
        "E_iso_sym": E_iso_sym,
        "E_iso_raw": E_iso_raw,
        "E_skew": E_skew,
    }


def compute_theorem_a_metrics(
    params_tr: list[Tensor],
    Wy: Tensor,
    X_dbg: Tensor,
    y_dbg: Tensor,
    cfg: DecoupledTrainModelConfig,
    *,
    store_full_tensors: bool = False,
) -> dict[str, Any]:
    device = X_dbg.device
    dtype = X_dbg.dtype
    p_tr = cfg.P_tr

    with torch.no_grad():
        full_raw, trace = trace_theorem_a_decoupled_frozen_emb(
            params_tr,
            Wy,
            X_dbg,
            y_dbg,
            L=cfg.L,
            P_test=cfg.P_test,
            beta_model=cfg.beta_model,
            store_full_tensors=store_full_tensors,
        )

        trace_summary = summarize_theorem_a_trace(trace)
        gamma_info = extract_gamma_code(params_tr, Wy, cfg.beta_model)

        Gamma_code = gamma_info["Gamma_code"]
        gamma_eff = gamma_info["gamma_eff"]
        identity = torch.eye(cfg.d, device=device, dtype=dtype)

        reduced_raw = reduced_raw_output_from_gamma(
            X_dbg,
            y_dbg,
            Gamma_code,
            L=cfg.L,
            P_test=cfg.P_test,
        )
        scalar_raw = reduced_raw_output_from_gamma(
            X_dbg,
            y_dbg,
            gamma_eff * identity,
            L=cfg.L,
            P_test=cfg.P_test,
        )

        eps = torch.finfo(dtype).eps
        E_matrix_all = _relative_error(full_raw, reduced_raw, eps)
        E_matrix_train = _relative_error(full_raw[:, :p_tr], reduced_raw[:, :p_tr], eps)
        E_matrix_test = _relative_error(full_raw[:, p_tr:], reduced_raw[:, p_tr:], eps)

        E_scalar_all = _relative_error(reduced_raw, scalar_raw, eps)
        E_scalar_train = _relative_error(reduced_raw[:, :p_tr], scalar_raw[:, :p_tr], eps)
        E_scalar_test = _relative_error(reduced_raw[:, p_tr:], scalar_raw[:, p_tr:], eps)

        debug_loss_full = torch.mean((full_raw[:, p_tr:] / cfg.gamma + y_dbg[:, p_tr:]) ** 2)
        debug_loss_matrix = torch.mean((reduced_raw[:, p_tr:] / cfg.gamma + y_dbg[:, p_tr:]) ** 2)
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

    summary: dict[str, Any] = {
        **trace_summary,
        "E_matrix_all": float(E_matrix_all.detach().cpu()),
        "E_matrix_train": float(E_matrix_train.detach().cpu()),
        "E_matrix_test": float(E_matrix_test.detach().cpu()),
        "E_scalar_all": float(E_scalar_all.detach().cpu()),
        "E_scalar_train": float(E_scalar_train.detach().cpu()),
        "E_scalar_test": float(E_scalar_test.detach().cpu()),
        "E_A": float(E_A),
        "E_iso_sym": float(gamma_info["E_iso_sym"].detach().cpu()),
        "E_iso_raw": float(gamma_info["E_iso_raw"].detach().cpu()),
        "E_skew": float(gamma_info["E_skew"].detach().cpu()),
        "gamma_eff": float(gamma_eff.detach().cpu()),
        "chi_v": float(gamma_info["chi_v"].detach().cpu()),
        "alpha_v": float(gamma_info["alpha_v"].detach().cpu()),
        "debug_loss_full": float(debug_loss_full.detach().cpu()),
        "debug_loss_matrix": float(debug_loss_matrix.detach().cpu()),
        "debug_loss_scalar": float(debug_loss_scalar.detach().cpu()),
    }

    if store_full_tensors:
        summary["Gamma_code"] = Gamma_code.detach().cpu()
        summary["Gamma_sym"] = gamma_info["Gamma_sym"].detach().cpu()
        summary["reduced_raw"] = reduced_raw.detach().cpu()
        summary["scalar_raw"] = scalar_raw.detach().cpu()
        summary["full_raw"] = full_raw.detach().cpu()
        summary["trace"] = trace

    return summary


# -----------------------------------------------------------------------------
# Sweep runner
# -----------------------------------------------------------------------------

def _final_window_mean_and_stderr(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    w = min(window, arr.shape[-1])
    tail = arr[..., -w:]
    mean = tail.mean(axis=-1)
    std = tail.std(axis=-1, ddof=0)
    stderr = std / math.sqrt(float(w))
    return mean, stderr


def _choose_reference_alpha_index(p_trs: list[int], d: int) -> int:
    alpha_ctx = np.asarray(p_trs, dtype=np.float64) / float(d)
    return int(np.argmin(np.abs(alpha_ctx - 1.0)))


def _metric_cube_from_nested_runs(runs: list[list[dict[str, Any]]], key: str) -> np.ndarray:
    num_p = len(runs)
    num_l = len(runs[0])
    num_t = len(runs[0][0]["audit_steps"])
    cube = np.zeros((num_p, num_l, num_t), dtype=np.float64)
    for i in range(num_p):
        for j in range(num_l):
            cube[i, j] = np.asarray(runs[i][j]["audit_metrics"][key], dtype=np.float64)
    return cube


def _loss_cube_from_nested_runs(runs: list[list[dict[str, Any]]]) -> np.ndarray:
    num_p = len(runs)
    num_l = len(runs[0])
    T = len(runs[0][0]["train_loss"])
    cube = np.zeros((num_p, num_l, T), dtype=np.float64)
    for i in range(num_p):
        for j in range(num_l):
            cube[i, j] = np.asarray(runs[i][j]["train_loss"], dtype=np.float64)
    return cube


def train_single_theorem_a_run(
    cfg: DecoupledTrainModelConfig,
    *,
    spec: Tensor | None,
    w_star: Tensor | None,
    audit_every: int,
    debug_seed: int,
    debug_batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    store_final_tensors: bool,
) -> dict[str, Any]:
    if cfg.unrestricted:
        raise ValueError("Theorem A should start with unrestricted=False for the Figure 1 architecture.")

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

    X_dbg, y_dbg = sample_batch_from_cfg(
        cfg,
        spec=spec,
        w_star=w_star,
        seed=debug_seed,
        B_override=debug_batch_size,
        device=device,
        dtype=dtype,
    )

    if audit_every <= 0:
        audit_steps_set = {0, cfg.T}
    else:
        audit_steps_set = set(range(0, cfg.T + 1, audit_every))
        audit_steps_set.add(cfg.T)
    audit_steps = sorted(audit_steps_set)

    audit_metrics: dict[str, list[float]] = {}
    train_loss: list[float] = []
    final_snapshot: dict[str, Any] | None = None

    def record(step: int, *, store_tensors: bool) -> None:
        nonlocal final_snapshot
        summary = compute_theorem_a_metrics(
            [p.detach() for p in trainable],
            Wy.detach(),
            X_dbg,
            y_dbg,
            cfg,
            store_full_tensors=store_tensors,
        )
        summary["step"] = step
        for key, value in summary.items():
            if isinstance(value, (float, int)):
                audit_metrics.setdefault(key, []).append(float(value))
        if store_tensors:
            final_snapshot = summary

    if 0 in audit_steps_set:
        record(0, store_tensors=False)

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
            norm_inputs=False,
        )
        loss = torch.mean((out[:, cfg.P_tr:] / cfg.gamma + y[:, cfg.P_tr:]) ** 2)
        reg_loss = cfg.N * (cfg.gamma**2) * loss + cfg.lamb * _sum_squares(trainable)
        reg_loss.backward()
        optimizer.step()

        train_loss.append(float(loss.detach().cpu()))

        step = t + 1
        if step in audit_steps_set:
            record(step, store_tensors=(store_final_tensors and step == cfg.T))

    return {
        "train_loss": train_loss,
        "audit_steps": audit_steps,
        "audit_metrics": audit_metrics,
        "X_debug": X_dbg.detach().cpu() if store_final_tensors else None,
        "y_debug": y_dbg.detach().cpu() if store_final_tensors else None,
        "Wy": Wy.detach().cpu() if store_final_tensors else None,
        "final_snapshot": final_snapshot,
        "final_params": [p.detach().cpu() for p in trainable] if store_final_tensors else None,
    }


def run_theorem_a_isotropic_depth_vs_alpha_sweep(
    cfg: IsotropicDepthAlphaSweepConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float64,
    audit_every: int = 100,
    debug_seed: int = 1234,
    debug_batch_size: int = 64,
    final_window: int = 10,
    include_theory: bool = True,
    store_final_tensors: bool = True,
) -> dict[str, Any]:
    spec, w_star = make_normalized_powerlaw_problem(cfg.d, cfg.alpha, cfg.beta, device=device, dtype=dtype)

    all_runs: list[list[dict[str, Any]]] = []
    for P_tr in cfg.p_trs:
        print("")
        print(f"P_tr = {P_tr}")
        runs_for_ptr: list[dict[str, Any]] = []
        for L in cfg.lvals:
            print(f"  L = {L}")
            run_cfg = DecoupledTrainModelConfig(
                d=cfg.d,
                P_tr=P_tr,
                P_test=cfg.P_test,
                B=cfg.B,
                N=cfg.N,
                L=L,
                beta_model=cfg.beta_model,
                gamma=cfg.gamma,
                T=cfg.T,
                lr=cfg.lr,
                lamb=cfg.lamb,
                alpha=cfg.alpha,
                beta=cfg.beta,
                sigma=cfg.sigma,
                random_rotate=False,
                unrestricted=cfg.unrestricted,
                online=True,
                sample_mode="spec",
            )
            run = train_single_theorem_a_run(
                run_cfg,
                spec=spec,
                w_star=w_star,
                audit_every=audit_every,
                debug_seed=debug_seed,
                debug_batch_size=debug_batch_size,
                device=device,
                dtype=dtype,
                store_final_tensors=store_final_tensors,
            )
            runs_for_ptr.append(run)
        all_runs.append(runs_for_ptr)

    alpha_ctx = np.asarray(cfg.p_trs, dtype=np.float64) / float(cfg.d)
    audit_steps = np.asarray(all_runs[0][0]["audit_steps"], dtype=np.int64)
    train_loss_cube = _loss_cube_from_nested_runs(all_runs)

    metric_names = [
        "E_A",
        "E_kernel",
        "E_roll_test",
        "E_span_max",
        "E_value_align_max",
        "E_exact_max",
        "E_local_max",
        "E_matrix_test",
        "E_scalar_test",
        "E_iso_sym",
        "E_iso_raw",
        "E_skew",
        "gamma_eff",
        "chi_v",
        "alpha_v",
        "debug_loss_full",
        "debug_loss_matrix",
        "debug_loss_scalar",
    ]
    metric_cubes = {name: _metric_cube_from_nested_runs(all_runs, name) for name in metric_names}

    final_stats: dict[str, np.ndarray] = {}
    for name, cube in metric_cubes.items():
        mean, stderr = _final_window_mean_and_stderr(cube, final_window)
        final_stats[f"{name}_mean"] = mean.transpose(1, 0)
        final_stats[f"{name}_stderr"] = stderr.transpose(1, 0)

    debug_loss_mean, debug_loss_stderr = _final_window_mean_and_stderr(metric_cubes["debug_loss_full"], final_window)
    train_loss_tail_mean = np.zeros((len(cfg.lvals), len(cfg.p_trs)), dtype=np.float64)
    train_loss_tail_stderr = np.zeros((len(cfg.lvals), len(cfg.p_trs)), dtype=np.float64)
    for i in range(len(cfg.p_trs)):
        for j in range(len(cfg.lvals)):
            mean, stderr = _final_window_mean_and_stderr(train_loss_cube[i, j][None, :], final_window)
            train_loss_tail_mean[j, i] = mean[0]
            train_loss_tail_stderr[j, i] = stderr[0]

    theory = None
    if include_theory:
        alpha_vals = torch.logspace(
            cfg.theory_alpha_min_exp,
            cfg.theory_alpha_max_exp,
            cfg.theory_alpha_points,
            device=torch.device(device),
            dtype=dtype,
        )
        loss_np = torch.stack(
            [
                isotropic_dmft(
                    alpha_val,
                    1.0 / (1.0 + 1.0 / alpha_val),
                    cfg.theory_T,
                    iters=cfg.theory_iters,
                    device=device,
                    dtype=dtype,
                )
                for alpha_val in alpha_vals
            ],
            dim=1,
        )
        theory = {
            "alpha_vals": alpha_vals.detach().cpu().numpy(),
            "loss_np": loss_np.detach().cpu().numpy(),
        }

    return {
        "config": cfg,
        "spec": spec.detach().cpu(),
        "w_star": w_star.detach().cpu(),
        "runs": all_runs,
        "alpha_ctx": alpha_ctx,
        "audit_steps": audit_steps,
        "train_loss_cube": train_loss_cube,
        "metric_cubes": metric_cubes,
        "final_stats": final_stats,
        "debug_loss_mean": debug_loss_mean.transpose(1, 0),
        "debug_loss_stderr": debug_loss_stderr.transpose(1, 0),
        "train_loss_tail_mean": train_loss_tail_mean,
        "train_loss_tail_stderr": train_loss_tail_stderr,
        "theory": theory,
        "p_trs": list(cfg.p_trs),
        "lvals": list(cfg.lvals),
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _safe_plot_values(arr: np.ndarray, floor: float = 1.0e-18) -> np.ndarray:
    return np.maximum(arr, floor)


def _plot_main_summary(
    results: dict[str, Any],
    *,
    output_path: Path,
    d: int,
    include_theory: bool,
) -> None:
    alpha_ctx = results["alpha_ctx"]
    lvals = results["lvals"]
    fs = results["final_stats"]
    theory = results["theory"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    if include_theory and theory is not None:
        for i, L in enumerate(lvals):
            ax.semilogx(theory["alpha_vals"], theory["loss_np"][2 * L], color=f"C{i}", alpha=0.8)
    for i, L in enumerate(lvals):
        ax.errorbar(
            alpha_ctx,
            results["debug_loss_mean"][i],
            results["debug_loss_stderr"][i],
            fmt="o",
            color=f"C{i}",
            label=f"L={L}",
        )
    if include_theory and theory is not None:
        ax.semilogx([], [], color="C0", label="DMFT")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha=P_{\rm tr}/d$")
    ax.set_ylabel("debug loss")
    ax.set_title("Final debug loss")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for i, L in enumerate(lvals):
        ax.errorbar(
            alpha_ctx,
            _safe_plot_values(fs["E_A_mean"][i]),
            _safe_plot_values(fs["E_A_stderr"][i]),
            fmt="o-",
            color=f"C{i}",
            label=f"L={L}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha=P_{\rm tr}/d$")
    ax.set_ylabel(r"$E_{\mathrm{A}}$")
    ax.set_title("Final exactness summary")

    ax = axes[1, 0]
    for i, L in enumerate(lvals):
        ax.errorbar(
            alpha_ctx,
            _safe_plot_values(fs["E_iso_sym_mean"][i]),
            _safe_plot_values(fs["E_iso_sym_stderr"][i]),
            fmt="o-",
            color=f"C{i}",
            label=f"L={L}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha=P_{\rm tr}/d$")
    ax.set_ylabel(r"$E_{\mathrm{iso,sym}}$")
    ax.set_title("Final isotropy error")

    ax = axes[1, 1]
    for i, L in enumerate(lvals):
        ax.errorbar(
            alpha_ctx,
            _safe_plot_values(fs["E_scalar_test_mean"][i]),
            _safe_plot_values(fs["E_scalar_test_stderr"][i]),
            fmt="o-",
            color=f"C{i}",
            label=f"L={L}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha=P_{\rm tr}/d$")
    ax.set_ylabel(r"$E_{\mathrm{scalar,test}}$")
    ax.set_title("Final scalarization error")

    fig.suptitle(f"Theorem A sweep on isotropic Figure-1 grid (d={d})", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamics_vary_alpha(
    results: dict[str, Any],
    *,
    output_path: Path,
    ref_l_index: int,
) -> None:
    alpha_ctx = results["alpha_ctx"]
    lvals = results["lvals"]
    audit_steps = results["audit_steps"]
    train_loss_cube = results["train_loss_cube"]
    metric_cubes = results["metric_cubes"]

    ref_l_index = ref_l_index % len(lvals)
    L_ref = lvals[ref_l_index]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for i, alpha_val in enumerate(alpha_ctx):
        ax.plot(audit_steps[1:], metric_cubes["debug_loss_full"][i, ref_l_index, 1:], label=rf"$\alpha={alpha_val:.2f}$")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("debug loss")
    ax.set_title(f"Vary $\\alpha$, fixed L={L_ref}")
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, alpha_val in enumerate(alpha_ctx):
        ax.plot(audit_steps, _safe_plot_values(metric_cubes["E_iso_sym"][i, ref_l_index]), label=rf"$\alpha={alpha_val:.2f}$")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$E_{\mathrm{iso,sym}}$")
    ax.set_title(f"Isotropy dynamics, L={L_ref}")

    ax = axes[2]
    for i, alpha_val in enumerate(alpha_ctx):
        ax.plot(audit_steps, _safe_plot_values(metric_cubes["E_scalar_test"][i, ref_l_index]), label=rf"$\alpha={alpha_val:.2f}$")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$E_{\mathrm{scalar,test}}$")
    ax.set_title(f"Scalarization dynamics, L={L_ref}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamics_vary_L(
    results: dict[str, Any],
    *,
    output_path: Path,
    ref_p_index: int,
) -> None:
    alpha_ctx = results["alpha_ctx"]
    lvals = results["lvals"]
    audit_steps = results["audit_steps"]
    metric_cubes = results["metric_cubes"]

    ref_p_index = ref_p_index % len(alpha_ctx)
    alpha_ref = alpha_ctx[ref_p_index]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for j, L in enumerate(lvals):
        ax.plot(audit_steps[1:], metric_cubes["debug_loss_full"][ref_p_index, j, 1:], label=rf"$L={L}$")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("debug loss")
    ax.set_title(rf"Vary $L$, fixed $\alpha={alpha_ref:.2f}$")
    ax.legend(fontsize=8)

    ax = axes[1]
    for j, L in enumerate(lvals):
        ax.plot(audit_steps, _safe_plot_values(metric_cubes["E_iso_sym"][ref_p_index, j]), label=rf"$L={L}$")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$E_{\mathrm{iso,sym}}$")
    ax.set_title(rf"Isotropy dynamics, $\alpha={alpha_ref:.2f}$")

    ax = axes[2]
    for j, L in enumerate(lvals):
        ax.plot(audit_steps, _safe_plot_values(metric_cubes["E_scalar_test"][ref_p_index, j]), label=rf"$L={L}$")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$E_{\mathrm{scalar,test}}$")
    ax.set_title(rf"Scalarization dynamics, $\alpha={alpha_ref:.2f}$")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Theorem A isotropic depth-vs-alpha sweep on the Figure-1 grid: "
            "exact reduction audit, isotropy audit, scalarization audit, and figure generation."
        )
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--p-test", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0, help="Spectral exponent for the data generator; alpha=0 gives flat spectrum.")
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=0.125)
    parser.add_argument("--lamb", type=float, default=1.0e-14)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--p-trs", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16")
    parser.add_argument("--theory-t", type=int, default=512)
    parser.add_argument("--theory-iters", type=int, default=100)
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--debug-seed", type=int, default=1234)
    parser.add_argument("--debug-batch-size", type=int, default=64)
    parser.add_argument("--final-window", type=int, default=10)
    parser.add_argument("--no-theory", action="store_true")
    parser.add_argument("--main-plot-path", type=str, default=None)
    parser.add_argument("--alpha-dynamics-plot-path", type=str, default=None)
    parser.add_argument("--l-dynamics-plot-path", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--artifacts-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    requested_device = torch.device(args.device)
    print(f"requested device: {requested_device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if requested_device.type == "cuda" and torch.cuda.is_available():
        current_idx = requested_device.index if requested_device.index is not None else torch.cuda.current_device()
        print(f"using cuda device: cuda:{current_idx}")
        print(f"cuda device name: {torch.cuda.get_device_name(current_idx)}")
    elif requested_device.type == "cuda":
        print("CUDA was requested but is not available.")

    p_trs = parse_int_list(args.p_trs)
    lvals = parse_int_list(args.lvals)

    cfg = IsotropicDepthAlphaSweepConfig(
        d=args.d,
        P_test=args.p_test,
        B=args.batch,
        N=args.n,
        alpha=args.alpha,
        beta=args.beta,
        T=args.steps,
        lr=args.lr,
        lamb=args.lamb,
        beta_model=args.beta_model,
        gamma=args.gamma,
        sigma=args.sigma,
        p_trs=tuple(p_trs),
        lvals=tuple(lvals),
        theory_T=args.theory_t,
        theory_iters=args.theory_iters,
        unrestricted=False,
    )

    results = run_theorem_a_isotropic_depth_vs_alpha_sweep(
        cfg,
        device=args.device,
        dtype=dtype,
        audit_every=args.audit_every,
        debug_seed=args.debug_seed,
        debug_batch_size=args.debug_batch_size,
        final_window=args.final_window,
        include_theory=not args.no_theory,
        store_final_tensors=True,
    )

    ref_l_index = len(lvals) - 1
    ref_p_index = _choose_reference_alpha_index(p_trs, args.d)

    main_plot_path = (
        Path(args.main_plot_path)
        if args.main_plot_path
        else PROJECT_ROOT / "outputs" / "theorem_a_isotropic_main.png"
    )
    alpha_dynamics_plot_path = (
        Path(args.alpha_dynamics_plot_path)
        if args.alpha_dynamics_plot_path
        else PROJECT_ROOT / "outputs" / "theorem_a_dynamics_vary_alpha.png"
    )
    l_dynamics_plot_path = (
        Path(args.l_dynamics_plot_path)
        if args.l_dynamics_plot_path
        else PROJECT_ROOT / "outputs" / "theorem_a_dynamics_vary_L.png"
    )
    metrics_path = (
        Path(args.metrics_path)
        if args.metrics_path
        else PROJECT_ROOT / "outputs" / "theorem_a_metrics.npz"
    )
    artifacts_path = (
        Path(args.artifacts_path)
        if args.artifacts_path
        else PROJECT_ROOT / "outputs" / "theorem_a_artifacts.pt"
    )

    _plot_main_summary(
        results,
        output_path=main_plot_path,
        d=args.d,
        include_theory=(not args.no_theory),
    )
    print(f"Saved theorem-A main plot to: {main_plot_path}")

    _plot_dynamics_vary_alpha(results, output_path=alpha_dynamics_plot_path, ref_l_index=ref_l_index)
    print(f"Saved theorem-A alpha dynamics plot to: {alpha_dynamics_plot_path}")

    _plot_dynamics_vary_L(results, output_path=l_dynamics_plot_path, ref_p_index=ref_p_index)
    print(f"Saved theorem-A L dynamics plot to: {l_dynamics_plot_path}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "alpha_ctx": np.asarray(results["alpha_ctx"], dtype=np.float64),
        "p_trs": np.asarray(results["p_trs"], dtype=np.int64),
        "lvals": np.asarray(results["lvals"], dtype=np.int64),
        "audit_steps": np.asarray(results["audit_steps"], dtype=np.int64),
        "train_loss_cube": np.asarray(results["train_loss_cube"], dtype=np.float64),
        "debug_loss_mean": np.asarray(results["debug_loss_mean"], dtype=np.float64),
        "debug_loss_stderr": np.asarray(results["debug_loss_stderr"], dtype=np.float64),
        "train_loss_tail_mean": np.asarray(results["train_loss_tail_mean"], dtype=np.float64),
        "train_loss_tail_stderr": np.asarray(results["train_loss_tail_stderr"], dtype=np.float64),
    }
    for key, cube in results["metric_cubes"].items():
        payload[f"metric_{key}"] = np.asarray(cube, dtype=np.float64)
    for key, arr in results["final_stats"].items():
        payload[f"final_{key}"] = np.asarray(arr, dtype=np.float64)
    if results["theory"] is not None:
        payload["theory_alpha_vals"] = np.asarray(results["theory"]["alpha_vals"], dtype=np.float64)
        payload["theory_loss_np"] = np.asarray(results["theory"]["loss_np"], dtype=np.float64)
    np.savez(metrics_path, **payload)
    print(f"Saved theorem-A metrics to: {metrics_path}")

    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, artifacts_path)
    print(f"Saved theorem-A artifacts to: {artifacts_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
