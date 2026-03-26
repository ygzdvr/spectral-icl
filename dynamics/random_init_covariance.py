"""Random weight initialization experiments for the hand-coded attention model.

This module evaluates the hand-coded linear attention model (from
``linear_icl_dynamics``) when initialized with random Gaussian weights instead
of the analytically optimal structure. This helps study:

  - How the model behaves before training (at random initialization).
  - The effect of weight scale (sigma) on initial ICL performance.
  - The interaction between random weights and power-law covariance structure.

The experiment flow is:
  1. Initialize random Gaussian weight matrices scaled by sigma
     (``init_random_covariance_params``).
  2. Generate a batch of tasks with power-law covariance
     (``sample_random_init_covariance_batch``).
  3. Run the model forward and collect train/test losses
     (``run_random_init_covariance_eval``).
"""
from __future__ import annotations

import math

import torch

from configs.eval_configs import RandomInitCovarianceEvalConfig
from .linear_icl_dynamics import model_eval


Tensor = torch.Tensor


def _randn(shape: tuple[int, ...], seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Generate a standard normal tensor with a deterministic CPU seed.

    Args:
        shape: Shape of the output tensor.
        seed: Random seed for reproducibility (seeded on CPU, then moved to device).
        device: Target device for the output tensor.
        dtype: Data type for the output tensor.

    Returns:
        Tensor of the given shape filled with i.i.d. N(0,1) values.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=dtype).to(device)


def init_random_covariance_params(
    cfg: RandomInitCovarianceEvalConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """Initialize attention parameters with random Gaussian weights.

    All weight matrices are drawn from N(0, sigma^2) element-wise, except
    w_out which is scaled by an additional factor of 0.25. Each matrix has
    its own deterministic random seed from the config.

    Weight shapes:
      - W_x: (d+1, d) -- input feature projection.
      - W_y: (d+1,) -- label embedding vector.
      - Wq: (d+1, d+1) -- query projection matrix.
      - Wk: (d+1, d+1) -- key projection matrix.
      - Wv: (d+1, d+1) -- value projection matrix.
      - w_out: (d+1,) -- output projection vector (scaled by 0.25*sigma).

    Args:
        cfg: Configuration dataclass containing dimension d, scale sigma,
            and per-parameter random seeds.
        device: Torch device for the parameters.
        dtype: Torch dtype for the parameters.

    Returns:
        List of 6 tensors: [W_x, W_y, Wq, Wk, Wv, w_out].
    """
    device = torch.device(device)
    d = cfg.d
    sigma = cfg.sigma

    W_x = sigma * _randn((d + 1, d), cfg.seed_wx, device=device, dtype=dtype)
    W_y = sigma * _randn((d + 1,), cfg.seed_wy, device=device, dtype=dtype)
    Wq = sigma * _randn((d + 1, d + 1), cfg.seed_wq, device=device, dtype=dtype)
    Wk = sigma * _randn((d + 1, d + 1), cfg.seed_wk, device=device, dtype=dtype)
    Wv = sigma * _randn((d + 1, d + 1), cfg.seed_wv, device=device, dtype=dtype)
    w_out = 0.25 * sigma * _randn((d + 1,), cfg.seed_wout, device=device, dtype=dtype)
    return [W_x, W_y, Wq, Wk, Wv, w_out]


def sample_random_init_covariance_batch(
    cfg: RandomInitCovarianceEvalConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a batch of ICL tasks with power-law covariance for random-init evaluation.

    For each task b in the batch:
      1. Determine the power-law exponent: either a fixed value (cfg.fixed_exp)
         or sampled from Uniform(0, cfg.exp_scale).
      2. Construct eigenvalue weights: powers[b, k] = k^{-exponent} for k=1..d.
      3. Generate isotropic Gaussian X and scale by powers to create anisotropic inputs.
      4. Generate random regression target beta and compute y = X @ beta / sqrt(d).

    The total sequence length per task is P + P_test tokens.

    Args:
        cfg: Configuration dataclass with dimension, batch size, sequence lengths,
            exponent settings, and random seeds.
        device: Torch device for the output tensors.
        dtype: Torch dtype for the output tensors.

    Returns:
        A tuple ``(X, y, powers)`` where:
            - ``X``: Input features of shape ``(B, P+P_test, d)``.
            - ``y``: Labels of shape ``(B, P+P_test)``.
            - ``powers``: Per-task eigenvalue weights of shape ``(B, d)``.
    """
    device = torch.device(device)
    seq_len = cfg.P + cfg.P_test
    coords = torch.linspace(1, cfg.d, cfg.d, device=device, dtype=dtype)

    if cfg.fixed_exp is None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed_exp)
        exps = cfg.exp_scale * torch.rand((cfg.B,), generator=gen, dtype=dtype).to(device)
    else:
        exps = torch.full((cfg.B,), cfg.fixed_exp, device=device, dtype=dtype)

    powers = coords.unsqueeze(0).pow(-exps.unsqueeze(1))
    X = _randn((cfg.B, seq_len, cfg.d), cfg.seed_x, device=device, dtype=dtype)
    X = X * powers.unsqueeze(1)
    betas = _randn((cfg.B, cfg.d), cfg.seed_beta, device=device, dtype=dtype)
    y = torch.einsum("bpd,bd->bp", X, betas) / math.sqrt(cfg.d)
    return X, y, powers


def run_random_init_covariance_eval(
    cfg: RandomInitCovarianceEvalConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, list[float], list[float], Tensor, Tensor, Tensor]:
    """Run the full random initialization covariance evaluation.

    Orchestrates the evaluation pipeline:
      1. Initialize randomly-scaled Gaussian weight matrices.
      2. Sample a batch of tasks with power-law covariance structures.
      3. Run the model forward for L layers and collect per-layer train and
         test losses.

    Args:
        cfg: Configuration dataclass with all experiment parameters.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        A tuple ``(out, train_losses, test_losses, X, y, powers)`` where:
            - ``out``: Model predictions of shape ``(B, P+P_test)``.
            - ``train_losses``: List of per-layer training losses (length L).
            - ``test_losses``: List of per-layer test losses (length L).
            - ``X``: Input features of shape ``(B, P+P_test, d)``.
            - ``y``: Labels of shape ``(B, P+P_test)``.
            - ``powers``: Per-task eigenvalue weights of shape ``(B, d)``.
    """
    params = init_random_covariance_params(cfg, device=device, dtype=dtype)
    X, y, powers = sample_random_init_covariance_batch(cfg, device=device, dtype=dtype)
    out, train_losses, test_losses = model_eval(
        params,
        X,
        y,
        L=cfg.L,
        P_test=cfg.P_test,
        beta=cfg.beta_model,
    )
    return out, train_losses, test_losses, X, y, powers
