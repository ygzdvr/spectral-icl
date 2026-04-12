from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
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


Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Utilities
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



def _relative_error(a: Tensor, b: Tensor, eps: float | None = None) -> Tensor:
    if eps is None:
        eps = torch.finfo(a.dtype).eps
    denom = torch.maximum(torch.linalg.norm(a), torch.linalg.norm(b)).clamp_min(eps)
    return torch.linalg.norm(a - b) / denom



def _sum_squares(params: list[Tensor]) -> Tensor:
    return sum(torch.sum(p * p) for p in params)


# -----------------------------------------------------------------------------
# Data generation / initialization (copied from the theorem-A isotropic runner,
# kept local to avoid package-level circular imports)
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


# -----------------------------------------------------------------------------
# Reduced operator extraction
# -----------------------------------------------------------------------------

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

    H = (W_x.transpose(0, 1) @ Wk.transpose(0, 1) @ Wq @ W_x) / float(N * N)
    Gamma_code = beta_model * chi_v * H

    return {
        "Gamma_code": Gamma_code,
        "alpha_v": alpha_v,
        "chi_v": chi_v,
    }


# -----------------------------------------------------------------------------
# Mask families
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskSpec:
    key: str
    label: str
    readout_sign: float
    valid_theorem_a: bool


DEFAULT_MASK_SPECS: tuple[MaskSpec, ...] = (
    MaskSpec("same_sign_flip", "same sign + output flip", -1.0, True),
    MaskSpec("bordelon_signed", "Bordelon signed mask", +1.0, True),
    MaskSpec("same_sign_no_flip", "same sign, no flip", +1.0, False),
    MaskSpec("all_cols_flip", "allow test columns", -1.0, False),
    MaskSpec("window_train_only", "window train-only", -1.0, False),
    MaskSpec("block_train_only", "block train-only", -1.0, False),
    MaskSpec("sparse_train_only", "sparse train-only", -1.0, False),
)



def build_mask_matrix(
    spec: MaskSpec,
    *,
    seq_len: int,
    p_tr: int,
    window: int,
    block: int,
    sparse_prob: float,
    mask_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    M = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)

    if spec.key == "same_sign_flip":
        M[:, :p_tr] = 1.0
        return M

    if spec.key == "bordelon_signed":
        M[:p_tr, :p_tr] = 1.0
        M[p_tr:, :p_tr] = -1.0
        return M

    if spec.key == "same_sign_no_flip":
        M[:, :p_tr] = 1.0
        return M

    if spec.key == "all_cols_flip":
        M[:, :] = 1.0
        return M

    if spec.key == "window_train_only":
        width = max(1, min(window, p_tr))
        for row in range(seq_len):
            if row < p_tr:
                lo = max(0, row - width + 1)
                hi = min(p_tr, row + 1)
            else:
                lo = max(0, p_tr - width)
                hi = p_tr
            M[row, lo:hi] = 1.0
        return M

    if spec.key == "block_train_only":
        blk = max(1, min(block, p_tr))
        num_blocks = math.ceil(p_tr / blk)
        for row in range(seq_len):
            if row < p_tr:
                b = min(num_blocks - 1, row // blk)
            else:
                b = num_blocks - 1
            lo = b * blk
            hi = min(p_tr, (b + 1) * blk)
            M[row, lo:hi] = 1.0
        return M

    if spec.key == "sparse_train_only":
        gen = torch.Generator(device=device)
        gen.manual_seed(mask_seed)
        base = (torch.rand((seq_len, p_tr), generator=gen, device=device) < sparse_prob).to(dtype)
        # Ensure every row has at least one incoming training edge.
        row_sums = base.sum(dim=1)
        for row in range(seq_len):
            if row_sums[row] <= 0:
                col = min(row, p_tr - 1) if row < p_tr else (p_tr - 1)
                base[row, col] = 1.0
        M[:, :p_tr] = base
        return M

    raise ValueError(f"Unknown mask key: {spec.key}")


# -----------------------------------------------------------------------------
# Custom-mask forward passes and reference predictors
# -----------------------------------------------------------------------------

def model_eval_decoupled_frozen_emb_custom_mask(
    params_tr: list[Tensor],
    Wy: Tensor,
    X: Tensor,
    y: Tensor,
    *,
    L: int,
    P_test: int,
    beta_model: float,
    mask_matrix: Tensor,
) -> Tensor:
    W_x, Wq, Wk, Wv = params_tr
    device = X.device
    dtype = X.dtype

    W_x = _as_tensor(W_x, device=device, dtype=dtype)
    Wy = _as_tensor(Wy, device=device, dtype=dtype)
    Wq = _as_tensor(Wq, device=device, dtype=dtype)
    Wk = _as_tensor(Wk, device=device, dtype=dtype)
    Wv = _as_tensor(Wv, device=device, dtype=dtype)
    mask_matrix = _as_tensor(mask_matrix, device=device, dtype=dtype)

    N, _ = W_x.shape
    seq_len = X.shape[1]
    p_tr = seq_len - P_test
    batch = X.shape[0]

    hx = torch.einsum("btd,nd->btn", X, W_x)
    mask_y = _build_y_mask(batch, seq_len, p_tr, device=device, dtype=dtype)
    hy = torch.einsum("bt,n->btn", y * mask_y, Wy)

    step_scale = float(beta_model) / (float(L) * float(p_tr))
    masked = mask_matrix.unsqueeze(0)

    for _ in range(L):
        q = torch.einsum("btn,kn->btk", hx, Wq) / math.sqrt(N)
        k = torch.einsum("btn,kn->btk", hx, Wk) / math.sqrt(N)
        v = torch.einsum("btn,kn->btk", hy, Wv) / math.sqrt(N)
        A = torch.einsum("btk,bsk->bts", k, q) / float(N)
        update = torch.bmm(A * masked, v)
        hy = hy - step_scale * update

    raw_out = torch.einsum("btn,n->bt", hy, Wy) / float(N)
    return raw_out



def theorem_a_reference_predictions(
    X: Tensor,
    y: Tensor,
    Gamma_code: Tensor,
    *,
    L: int,
    P_test: int,
    gamma_out: float,
) -> Tensor:
    """Standard Bordelon / theorem-A predictor as positive label predictions.

    This is the predictor corresponding to the GD-compatible family:
      - training columns only,
      - signed train/query split (or equivalently same-sign plus output flip),
      - positive predictions returned on test tokens.
    """
    device = X.device
    dtype = X.dtype
    Gamma_code = _as_tensor(Gamma_code, device=device, dtype=dtype)

    seq_len = X.shape[1]
    p_tr = seq_len - P_test
    X_tr = X[:, :p_tr, :]
    X_te = X[:, p_tr:, :]
    y_tr = y[:, :p_tr]

    K_trtr = torch.einsum("btd,df,bsf->bts", X_tr, Gamma_code, X_tr)
    K_tetr = torch.einsum("btd,df,bsf->bts", X_te, Gamma_code, X_tr)

    state = y_tr.clone()
    accum = torch.zeros_like(state)
    step_scale = 1.0 / (float(L) * float(p_tr))

    for _ in range(L):
        accum = accum + state
        state = state - step_scale * torch.bmm(K_trtr, state.unsqueeze(-1)).squeeze(-1)

    preds = step_scale * torch.bmm(K_tetr, accum.unsqueeze(-1)).squeeze(-1) / float(gamma_out)
    return preds



def generic_mask_scalar_predictions(
    X: Tensor,
    y: Tensor,
    Gamma_code: Tensor,
    *,
    L: int,
    P_test: int,
    gamma_out: float,
    mask_matrix: Tensor,
    readout_sign: float,
) -> Tensor:
    """Generic scalar token recurrence induced by a custom mask.

    This is *not* the Bordelon theorem-A predictor. It is the broader scalar
    recurrence that any fixed token-side mask induces once the value path stays
    one-dimensional in Wy. Including this reference makes the converse precise:
    broken masks can still admit a scalar recurrence, but not the in-context-GD
    theorem-A recurrence.
    """
    device = X.device
    dtype = X.dtype
    Gamma_code = _as_tensor(Gamma_code, device=device, dtype=dtype)
    mask_matrix = _as_tensor(mask_matrix, device=device, dtype=dtype)

    seq_len = X.shape[1]
    p_tr = seq_len - P_test

    K_full = torch.einsum("btd,df,bsf->bts", X, Gamma_code, X)
    state = torch.zeros((X.shape[0], seq_len), device=device, dtype=dtype)
    state[:, :p_tr] = y[:, :p_tr]
    step_scale = 1.0 / (float(L) * float(p_tr))
    masked_kernel = K_full * mask_matrix.unsqueeze(0)

    for _ in range(L):
        state = state - step_scale * torch.bmm(masked_kernel, state.unsqueeze(-1)).squeeze(-1)

    preds = float(readout_sign) * state[:, p_tr:] / float(gamma_out)
    return preds



def network_test_predictions_under_mask(
    params_tr: list[Tensor],
    Wy: Tensor,
    X: Tensor,
    y: Tensor,
    *,
    L: int,
    P_test: int,
    beta_model: float,
    gamma_out: float,
    mask_matrix: Tensor,
    readout_sign: float,
) -> Tensor:
    raw_out = model_eval_decoupled_frozen_emb_custom_mask(
        params_tr,
        Wy,
        X,
        y,
        L=L,
        P_test=P_test,
        beta_model=beta_model,
        mask_matrix=mask_matrix,
    )
    p_tr = X.shape[1] - P_test
    return float(readout_sign) * raw_out[:, p_tr:] / float(gamma_out)


# -----------------------------------------------------------------------------
# Auditing
# -----------------------------------------------------------------------------

def audit_mask_family_bundle(
    params_tr: list[Tensor],
    Wy: Tensor,
    X_dbg: Tensor,
    y_dbg: Tensor,
    cfg: DecoupledTrainModelConfig,
    mask_specs: list[MaskSpec],
    *,
    window: int,
    block: int,
    sparse_prob: float,
    mask_seed: int,
    store_predictions: bool = False,
) -> dict[str, Any]:
    device = X_dbg.device
    dtype = X_dbg.dtype
    seq_len = X_dbg.shape[1]
    p_tr = seq_len - cfg.P_test
    eps = torch.finfo(dtype).eps

    gamma_info = extract_gamma_code(params_tr, Wy, cfg.beta_model)
    Gamma_code = gamma_info["Gamma_code"]
    pred_ref = theorem_a_reference_predictions(
        X_dbg,
        y_dbg,
        Gamma_code,
        L=cfg.L,
        P_test=cfg.P_test,
        gamma_out=cfg.gamma,
    )
    y_test = y_dbg[:, p_tr:]
    ref_loss = torch.mean((pred_ref - y_test) ** 2)

    out: dict[str, Any] = {
        "gamma_info": {k: v.detach().cpu() for k, v in gamma_info.items()},
        "ref_loss": float(ref_loss.detach().cpu()),
        "masks": {},
    }

    for spec in mask_specs:
        M = build_mask_matrix(
            spec,
            seq_len=seq_len,
            p_tr=p_tr,
            window=window,
            block=block,
            sparse_prob=sparse_prob,
            mask_seed=mask_seed,
            device=device,
            dtype=dtype,
        )

        pred_net = network_test_predictions_under_mask(
            params_tr,
            Wy,
            X_dbg,
            y_dbg,
            L=cfg.L,
            P_test=cfg.P_test,
            beta_model=cfg.beta_model,
            gamma_out=cfg.gamma,
            mask_matrix=M,
            readout_sign=spec.readout_sign,
        )
        pred_generic = generic_mask_scalar_predictions(
            X_dbg,
            y_dbg,
            Gamma_code,
            L=cfg.L,
            P_test=cfg.P_test,
            gamma_out=cfg.gamma,
            mask_matrix=M,
            readout_sign=spec.readout_sign,
        )

        E_general = _relative_error(pred_net, pred_generic, eps)
        E_theorem = _relative_error(pred_net, pred_ref, eps)
        test_loss = torch.mean((pred_net - y_test) ** 2)
        E_generic_to_theorem = _relative_error(pred_generic, pred_ref, eps)

        entry: dict[str, Any] = {
            "E_general_test": float(E_general.detach().cpu()),
            "E_theorem_test": float(E_theorem.detach().cpu()),
            "E_generic_to_theorem": float(E_generic_to_theorem.detach().cpu()),
            "test_loss": float(test_loss.detach().cpu()),
            "valid_theorem_a": bool(spec.valid_theorem_a),
            "readout_sign": float(spec.readout_sign),
        }
        if store_predictions:
            entry["pred_net"] = pred_net.detach().cpu()
            entry["pred_generic"] = pred_generic.detach().cpu()
            entry["mask_matrix"] = M.detach().cpu()
        out["masks"][spec.key] = entry

    if store_predictions:
        out["pred_ref"] = pred_ref.detach().cpu()
        out["y_test"] = y_test.detach().cpu()

    return out


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_baseline_with_mask_audits(
    cfg: DecoupledTrainModelConfig,
    *,
    spec: Tensor,
    w_star: Tensor,
    mask_specs: list[MaskSpec],
    debug_seed: int,
    debug_batch: int,
    audit_every: int,
    window: int,
    block: int,
    sparse_prob: float,
    mask_seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if cfg.unrestricted:
        raise ValueError("Mask ablation is intended for the decoupled frozen-embedding model.")

    device = torch.device(device)
    params = init_pretrain_params(cfg.d, cfg.N, sigma=cfg.sigma, device=device, dtype=dtype)
    W_x, Wy, Wq, Wk, Wv, _w_out = params

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
        B_override=debug_batch,
        device=device,
        dtype=dtype,
    )

    losses: list[float] = []
    audit_steps: list[int] = []
    history: dict[str, dict[str, list[float]]] = {
        spec_i.key: {
            "E_general_test": [],
            "E_theorem_test": [],
            "E_generic_to_theorem": [],
            "test_loss": [],
        }
        for spec_i in mask_specs
    }
    ref_losses: list[float] = []

    def append_audit(step: int, *, store_predictions: bool = False) -> dict[str, Any]:
        bundle = audit_mask_family_bundle(
            [p.detach() for p in trainable],
            Wy.detach(),
            X_dbg,
            y_dbg,
            cfg,
            mask_specs,
            window=window,
            block=block,
            sparse_prob=sparse_prob,
            mask_seed=mask_seed,
            store_predictions=store_predictions,
        )
        audit_steps.append(step)
        ref_losses.append(bundle["ref_loss"])
        for spec_i in mask_specs:
            entry = bundle["masks"][spec_i.key]
            for key in history[spec_i.key]:
                history[spec_i.key][key].append(float(entry[key]))
        return bundle

    init_bundle = append_audit(0, store_predictions=False)
    final_bundle: dict[str, Any] | None = None

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
        reg_loss = cfg.N * (cfg.gamma**2) * loss + cfg.lamb * _sum_squares(trainable)
        reg_loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        step = t + 1
        if (step % audit_every == 0) or (step == cfg.T):
            final_bundle = append_audit(step, store_predictions=(step == cfg.T))

        if t % 100 == 0:
            print(f"step {t} , loss = {float(loss.detach().cpu()):.8f}")

    if final_bundle is None:
        final_bundle = append_audit(cfg.T, store_predictions=True)

    return {
        "losses": losses,
        "audit_steps": audit_steps,
        "history": history,
        "ref_losses": ref_losses,
        "init_bundle": init_bundle,
        "final_bundle": final_bundle,
        "final_params": [p.detach().cpu().clone() for p in trainable],
        "Wy": Wy.detach().cpu().clone(),
        "X_debug": X_dbg.detach().cpu().clone(),
        "y_debug": y_dbg.detach().cpu().clone(),
    }


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _setup_style() -> None:
    try:
        import seaborn as sns  # type: ignore

        sns.set(font_scale=1.15)
        sns.set_style("whitegrid")
        sns.set_palette("rocket")
    except Exception:
        plt.style.use("default")


def _save_axis_pair(fig: plt.Figure, ax: plt.Axes, out: OutputDir, stem: str, *, pad: float = 0.08) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).expanded(1.0 + pad, 1.0 + pad)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(out.png(stem), dpi=220, bbox_inches=bbox_inches)
    fig.savefig(out.pdf(stem), dpi=220, bbox_inches=bbox_inches)
    print(f"Saved {out.png(stem)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Theorem-A mask-ablation / converse experiment. Trains the standard "
            "Figure-1 decoupled frozen-embedding model on the isotropic grid at "
            "alpha = P_tr / d = 1, then swaps only the token-side mask at audit "
            "time while keeping the learned feature-side Gamma fixed."
        )
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--p-train", type=int, default=32)
    parser.add_argument("--p-test", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--debug-batch", type=int, default=64)
    parser.add_argument("--debug-seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.125)
    parser.add_argument("--lamb", type=float, default=1.0e-14)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--alpha-spec", type=float, default=0.0)
    parser.add_argument("--beta-spec", type=float, default=1.75)
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--sparse-prob", type=float, default=0.25)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    out = OutputDir(__file__, base=args.output_dir)
    _setup_style()

    lvals = parse_int_list(args.lvals)
    mask_specs = list(DEFAULT_MASK_SPECS)
    deepest_L = max(lvals)

    spec, w_star = make_normalized_powerlaw_problem(
        args.d,
        args.alpha_spec,
        args.beta_spec,
        device=device,
        dtype=dtype,
    )

    runs: dict[int, dict[str, Any]] = {}
    final_E_general = np.zeros((len(mask_specs), len(lvals)), dtype=np.float64)
    final_E_theorem = np.zeros((len(mask_specs), len(lvals)), dtype=np.float64)
    final_E_generic_to_theorem = np.zeros((len(mask_specs), len(lvals)), dtype=np.float64)
    final_task_loss = np.zeros((len(mask_specs), len(lvals)), dtype=np.float64)
    final_ref_loss = np.zeros((len(lvals),), dtype=np.float64)

    for j, L in enumerate(lvals):
        print("\n" + "=" * 80)
        print(f"Mask-ablation baseline training, L = {L}")
        print("=" * 80)
        cfg = DecoupledTrainModelConfig(
            d=args.d,
            P_tr=args.p_train,
            P_test=args.p_test,
            B=args.batch,
            N=args.d,
            L=L,
            beta_model=args.beta_model,
            gamma=args.gamma,
            T=args.steps,
            lr=args.lr,
            lamb=args.lamb,
            alpha=args.alpha_spec,
            beta=args.beta_spec,
            sigma=args.sigma,
            random_rotate=False,
            unrestricted=False,
            online=True,
            sample_mode="spec",
        )
        run = train_baseline_with_mask_audits(
            cfg,
            spec=spec,
            w_star=w_star,
            mask_specs=mask_specs,
            debug_seed=args.debug_seed,
            debug_batch=args.debug_batch,
            audit_every=args.audit_every,
            window=args.window,
            block=args.block,
            sparse_prob=args.sparse_prob,
            mask_seed=args.mask_seed,
            device=device,
            dtype=dtype,
        )
        runs[L] = run

        final_bundle = run["final_bundle"]
        final_ref_loss[j] = float(final_bundle["ref_loss"])
        for i, spec_i in enumerate(mask_specs):
            metrics = final_bundle["masks"][spec_i.key]
            final_E_general[i, j] = metrics["E_general_test"]
            final_E_theorem[i, j] = metrics["E_theorem_test"]
            final_E_generic_to_theorem[i, j] = metrics["E_generic_to_theorem"]
            final_task_loss[i, j] = metrics["test_loss"]

    # ------------------------------------------------------------------
    # Main summary figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    lvals_np = np.asarray(lvals, dtype=np.float64)

    for i, spec_i in enumerate(mask_specs):
        axes[0, 0].semilogy(lvals_np, np.maximum(final_E_general[i], 1e-18), marker="o", label=spec_i.label)
        axes[0, 1].semilogy(lvals_np, np.maximum(final_E_theorem[i], 1e-18), marker="o", label=spec_i.label)
        axes[1, 0].semilogy(lvals_np, np.maximum(final_task_loss[i], 1e-18), marker="o", label=spec_i.label)

    axes[1, 0].semilogy(lvals_np, np.maximum(final_ref_loss, 1e-18), "--", color="black", label="theorem-A ref")

    deepest_idx = int(np.argmax(lvals_np))
    x = np.arange(len(mask_specs), dtype=np.float64)
    deepest_vals = np.maximum(final_E_theorem[:, deepest_idx], 1e-18)
    colors = [f"C{i}" for i in range(len(mask_specs))]
    axes[1, 1].scatter(x, deepest_vals, s=80, c=colors)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([m.key for m in mask_specs], rotation=35, ha="right")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel(r"$E_{\mathrm{theorem},\mathrm{test}}$")
    axes[1, 1].set_title(f"Deepest depth L={deepest_L}: mask separation")

    axes[0, 0].set_title("Final generic scalar-reduction error")
    axes[0, 0].set_xlabel(r"$L$")
    axes[0, 0].set_ylabel(r"$E_{\mathrm{general},\mathrm{test}}$")

    axes[0, 1].set_title("Final theorem-A reference error")
    axes[0, 1].set_xlabel(r"$L$")
    axes[0, 1].set_ylabel(r"$E_{\mathrm{theorem},\mathrm{test}}$")

    axes[1, 0].set_title("Final test MSE on fixed debug batch")
    axes[1, 0].set_xlabel(r"$L$")
    axes[1, 0].set_ylabel("test MSE")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=True)
    fig.suptitle(
        "Theorem-A mask ablation on isotropic decoupled attention (alpha = 1)",
        fontsize=18,
    )
    fig.savefig(out.png("theorem_a_mask_ablation_main"), dpi=220, bbox_inches="tight")
    fig.savefig(out.pdf("theorem_a_mask_ablation_main"), dpi=220, bbox_inches="tight")
    print(f"Saved {out.png('theorem_a_mask_ablation_main')}")
    _save_axis_pair(fig, axes[0, 0], out, "theorem_a_mask_ablation_main_general_error")
    _save_axis_pair(fig, axes[0, 1], out, "theorem_a_mask_ablation_main_theorem_error")
    _save_axis_pair(fig, axes[1, 0], out, "theorem_a_mask_ablation_main_test_mse")
    _save_axis_pair(fig, axes[1, 1], out, "theorem_a_mask_ablation_main_mask_separation")

    # ------------------------------------------------------------------
    # Deepest-L dynamics figure
    # ------------------------------------------------------------------
    deepest_run = runs[deepest_L]
    audit_steps = np.asarray(deepest_run["audit_steps"], dtype=np.int64)
    ref_hist = np.asarray(deepest_run["ref_losses"], dtype=np.float64)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for i, spec_i in enumerate(mask_specs):
        hist = deepest_run["history"][spec_i.key]
        axes2[0].semilogy(audit_steps, np.maximum(np.asarray(hist["E_general_test"], dtype=np.float64), 1e-18), label=spec_i.label)
        axes2[1].semilogy(audit_steps, np.maximum(np.asarray(hist["E_theorem_test"], dtype=np.float64), 1e-18), label=spec_i.label)
        axes2[2].semilogy(audit_steps, np.maximum(np.asarray(hist["test_loss"], dtype=np.float64), 1e-18), label=spec_i.label)

    axes2[2].semilogy(audit_steps, np.maximum(ref_hist, 1e-18), "--", color="black", label="theorem-A ref")

    axes2[0].set_title(f"Generic scalar reduction, L={deepest_L}")
    axes2[0].set_xlabel("t")
    axes2[0].set_ylabel(r"$E_{\mathrm{general},\mathrm{test}}$")

    axes2[1].set_title(f"Theorem-A reference error, L={deepest_L}")
    axes2[1].set_xlabel("t")
    axes2[1].set_ylabel(r"$E_{\mathrm{theorem},\mathrm{test}}$")

    axes2[2].set_title(f"Fixed-batch test MSE, L={deepest_L}")
    axes2[2].set_xlabel("t")
    axes2[2].set_ylabel("test MSE")

    handles2, labels2 = axes2[2].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc="upper center", ncols=2, frameon=True)
    fig2.savefig(out.png("theorem_a_mask_ablation_dynamics"), dpi=220, bbox_inches="tight")
    fig2.savefig(out.pdf("theorem_a_mask_ablation_dynamics"), dpi=220, bbox_inches="tight")
    print(f"Saved {out.png('theorem_a_mask_ablation_dynamics')}")
    _save_axis_pair(fig2, axes2[0], out, "theorem_a_mask_ablation_dynamics_general_error")
    _save_axis_pair(fig2, axes2[1], out, "theorem_a_mask_ablation_dynamics_theorem_error")
    _save_axis_pair(fig2, axes2[2], out, "theorem_a_mask_ablation_dynamics_test_mse")

    # ------------------------------------------------------------------
    # Final scatter figure at deepest L
    # ------------------------------------------------------------------
    final_bundle = deepest_run["final_bundle"]
    pred_ref = final_bundle["pred_ref"].numpy().reshape(-1)

    fig3, axes3 = plt.subplots(2, math.ceil(len(mask_specs) / 2), figsize=(18, 8), constrained_layout=True)
    axes3 = np.asarray(axes3).reshape(-1)
    for ax, spec_i in zip(axes3, mask_specs):
        pred_net = final_bundle["masks"][spec_i.key]["pred_net"].numpy().reshape(-1)
        ax.scatter(pred_ref, pred_net, s=8, alpha=0.5)
        lo = float(min(np.min(pred_ref), np.min(pred_net)))
        hi = float(max(np.max(pred_ref), np.max(pred_net)))
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
        ax.set_title(spec_i.label)
        ax.set_xlabel(r"$f_{\Gamma,\mathrm{std}}$")
        ax.set_ylabel(r"$f_{\mathrm{net}}^{(S)}$")
    for ax in axes3[len(mask_specs):]:
        ax.axis("off")
    fig3.savefig(out.png("theorem_a_mask_ablation_scatter"), dpi=220, bbox_inches="tight")
    fig3.savefig(out.pdf("theorem_a_mask_ablation_scatter"), dpi=220, bbox_inches="tight")
    print(f"Saved {out.png('theorem_a_mask_ablation_scatter')}")
    for ax, spec_i in zip(axes3, mask_specs):
        _save_axis_pair(fig3, ax, out, f"theorem_a_mask_ablation_scatter_{spec_i.key}")

    # ------------------------------------------------------------------
    # Save arrays / artifacts
    # ------------------------------------------------------------------
    payload: dict[str, np.ndarray] = {
        "lvals": np.asarray(lvals, dtype=np.int64),
        "mask_labels": np.asarray([m.label for m in mask_specs], dtype=object),
        "mask_keys": np.asarray([m.key for m in mask_specs], dtype=object),
        "mask_valid": np.asarray([m.valid_theorem_a for m in mask_specs], dtype=bool),
        "final_E_general": final_E_general,
        "final_E_theorem": final_E_theorem,
        "final_E_generic_to_theorem": final_E_generic_to_theorem,
        "final_task_loss": final_task_loss,
        "final_ref_loss": final_ref_loss,
        "deepest_audit_steps": audit_steps,
        "deepest_ref_loss": ref_hist,
    }
    for spec_i in mask_specs:
        hist = deepest_run["history"][spec_i.key]
        payload[f"deepest_E_general_{spec_i.key}"] = np.asarray(hist["E_general_test"], dtype=np.float64)
        payload[f"deepest_E_theorem_{spec_i.key}"] = np.asarray(hist["E_theorem_test"], dtype=np.float64)
        payload[f"deepest_E_generic_to_theorem_{spec_i.key}"] = np.asarray(hist["E_generic_to_theorem"], dtype=np.float64)
        payload[f"deepest_loss_{spec_i.key}"] = np.asarray(hist["test_loss"], dtype=np.float64)
    np.savez(out.numpy("theorem_a_mask_ablation_metrics"), **payload)
    print(f"Saved data to: {out.numpy('theorem_a_mask_ablation_metrics')}")

    torch.save(
        {
            "config": vars(args),
            "spec": spec.detach().cpu(),
            "w_star": w_star.detach().cpu(),
            "mask_specs": [spec_i.__dict__ for spec_i in mask_specs],
            "runs": runs,
        },
        out.torch("theorem_a_mask_ablation_artifacts"),
    )
    print(f"Saved artifacts to: {out.torch('theorem_a_mask_ablation_artifacts')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
