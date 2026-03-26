"""Out-of-distribution covariance generalization evaluation for ICL models.

This module evaluates how the hand-coded linear attention model (from
``linear_icl_dynamics``) generalizes when the input covariance structure
varies across tasks. Instead of isotropic inputs, each task in the batch
has a power-law covariance with a randomly sampled exponent, creating an
out-of-distribution (OOD) test scenario.

The experiment flow is:
  1. Initialize the hand-coded attention parameters (``init_ood_covariance_params``).
  2. Generate a batch of tasks with heterogeneous covariance structures
     (``sample_ood_covariance_batch``).
  3. Run the model and collect train/test losses (``run_ood_covariance_eval``).

The covariance for task b has eigenvalues proportional to k^{-alpha_b} for
k = 1, ..., d, where alpha_b is drawn uniformly from [0, exp_scale]. When
exp_scale = 0, all tasks are isotropic.
"""
from __future__ import annotations

import math

import torch

from configs.eval_configs import OODCovarianceEvalConfig
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


def _rand(shape: tuple[int, ...], seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Generate a uniform [0,1) tensor with a deterministic CPU seed.

    Args:
        shape: Shape of the output tensor.
        seed: Random seed for reproducibility (seeded on CPU, then moved to device).
        device: Target device for the output tensor.
        dtype: Data type for the output tensor.

    Returns:
        Tensor of the given shape filled with i.i.d. Uniform(0,1) values.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.rand(shape, generator=gen, dtype=dtype).to(device)


def init_ood_covariance_params(
    d: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """Initialize the hand-coded attention parameters for OOD covariance evaluation.

    Creates the analytically optimal weight matrices for a single-layer linear
    attention model performing in-context linear regression. The structure
    encodes [x, Delta] in R^{d+1} where Delta tracks the residual prediction
    error.

    Weight structure:
      - W_x: (d+1, d) with sqrt(d)*I_d in the top-left block. Maps input
        features into the hidden representation.
      - W_y: (d+1,) with 1.0 in the last entry. Encodes the scalar label.
      - Wq, Wk: (d+1, d+1) with sqrt(d)*I_d in the top-left block. Query and
        key projections that compute attention scores based on feature similarity.
      - Wv: (d+1, d+1) with sqrt(d) in the bottom-right entry. Value projection
        that extracts the label component.
      - w_out: (d+1,) with sqrt(d) in the last entry. Output projection.

    Args:
        d: Ambient dimension of the linear regression problem.
        device: Torch device for the parameters.
        dtype: Torch dtype for the parameters.

    Returns:
        List of 6 tensors: [W_x, W_y, Wq, Wk, Wv, w_out].
    """
    device = torch.device(device)
    sqrt_d = math.sqrt(d)

    W_x = torch.zeros((d + 1, d), device=device, dtype=dtype)
    W_x[:d, :d] = torch.eye(d, device=device, dtype=dtype) * sqrt_d

    W_y = torch.zeros((d + 1,), device=device, dtype=dtype)
    W_y[-1] = 1.0

    Wq = torch.zeros((d + 1, d + 1), device=device, dtype=dtype)
    Wq[:d, :d] = torch.eye(d, device=device, dtype=dtype) * sqrt_d
    Wk = Wq.clone()

    Wv = torch.zeros((d + 1, d + 1), device=device, dtype=dtype)
    Wv[-1, -1] = sqrt_d

    w_out = torch.zeros((d + 1,), device=device, dtype=dtype)
    w_out[-1] = sqrt_d

    return [W_x, W_y, Wq, Wk, Wv, w_out]


def sample_ood_covariance_batch(
    cfg: OODCovarianceEvalConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a batch of ICL tasks with heterogeneous power-law covariances.

    For each task b in the batch:
      1. Sample a power-law exponent alpha_b ~ Uniform(0, exp_scale).
      2. Construct eigenvalue weights: powers[b, k] = k^{-alpha_b} for k=1..d.
      3. Generate isotropic X and scale by powers to create anisotropic inputs.
      4. Generate a random regression target beta and compute labels y = X @ beta / sqrt(d).

    The sequence for each task has P + P_test tokens total (P training, P_test test).

    Args:
        cfg: Configuration dataclass controlling dimensions, seeds, and exponent range.
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

    exps = cfg.exp_scale * _rand((cfg.B,), cfg.seed_exp, device=device, dtype=dtype)
    coords = torch.linspace(1, cfg.d, cfg.d, device=device, dtype=dtype)
    powers = coords.unsqueeze(0).pow(-exps.unsqueeze(1))

    X = _randn((cfg.B, seq_len, cfg.d), cfg.seed_x, device=device, dtype=dtype)
    X = X * powers.unsqueeze(1)

    betas = _randn((cfg.B, cfg.d), cfg.seed_beta, device=device, dtype=dtype)
    y = torch.einsum("bpd,bd->bp", X, betas) / math.sqrt(cfg.d)
    return X, y, powers


def run_ood_covariance_eval(
    cfg: OODCovarianceEvalConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, list[float], list[float], Tensor, Tensor, Tensor]:
    """Run the full OOD covariance generalization evaluation.

    Orchestrates the evaluation pipeline:
      1. Initialize hand-coded attention parameters.
      2. Sample a batch of tasks with heterogeneous covariance structures.
      3. Run the hand-coded model forward for L layers and collect per-layer
         train and test losses.

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
    params = init_ood_covariance_params(cfg.d, device=device, dtype=dtype)
    X, y, powers = sample_ood_covariance_batch(cfg, device=device, dtype=dtype)
    out, train_losses, test_losses = model_eval(
        params,
        X,
        y,
        L=cfg.L,
        P_test=cfg.P_test,
        beta=cfg.beta_model,
    )
    return out, train_losses, test_losses, X, y, powers
