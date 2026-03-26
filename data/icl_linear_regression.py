"""Synthetic data generation for in-context linear regression (ICL).

This module generates batches of synthetic linear regression tasks for studying
how transformers learn to perform linear regression via in-context learning.

Each batch contains B independent linear regression problems. For each problem:
  1. Draw random input vectors x_i ~ N(0, I_d) for i = 1..seq_len
  2. Draw a random regression vector beta ~ N(0, I_d)
  3. Compute labels y_i = <x_i, beta> / sqrt(d) + sigma * noise_i
  4. Zero out the last label y_{seq_len} (the query target)
  5. Pack inputs and labels into [x_i ; y_i] tokens of dimension d+1

The transformer receives the sequence of (x, y) tokens and must predict the
missing label for the last token — effectively performing ridge regression
from the context examples.
"""

from __future__ import annotations

import math

import torch

from configs.data_configs import LinearICLConfig


def _randn(shape: tuple[int, ...], seed: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate a seeded random normal tensor on CPU."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=dtype)


def generate_linear_icl_batch(
    batch_size: int,
    seq_len: int,
    xdim: int,
    sigma: float,
    seed_x: int,
    seed_beta: int,
    seed_noise: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of synthetic linear regression ICL sequences.

    Each sequence consists of seq_len tokens. Each token is [x_i; y_i] in R^{d+1}.
    The label of the last token is zeroed out (the query the model must predict).

    The regression relationship is y_i = <x_i, beta> / sqrt(d) + sigma * eps_i,
    where beta ~ N(0, I_d) and eps_i ~ N(0, 1).

    Args:
        batch_size: Number of independent regression tasks (B).
        seq_len: Number of tokens per sequence (P), including the query.
        xdim: Input dimensionality (d).
        sigma: Label noise standard deviation.
        seed_x: RNG seed for input vectors X ~ N(0, I).
        seed_beta: RNG seed for regression vectors beta ~ N(0, I).
        seed_noise: RNG seed for label noise eps ~ N(0, 1).
        device: Target device for output tensors.
        dtype: Floating-point precision.

    Returns:
        x_big: Tensor of shape [batch_size, seq_len, xdim + 1] — packed (x, y) tokens.
        targets: Tensor of shape [batch_size] — true labels for the last (query) token.
    """
    x = _randn((batch_size, seq_len, xdim), seed_x, dtype=dtype)
    betas = _randn((batch_size, xdim), seed_beta, dtype=dtype)

    y = torch.einsum("bsd,bd->bs", x, betas) / math.sqrt(xdim)
    y = y + sigma * _randn(tuple(y.shape), seed_noise, dtype=dtype)

    targets = y[:, -1].clone()
    y[:, -1] = 0.0
    x_big = torch.cat([x, y.unsqueeze(-1)], dim=-1)

    if device is not None:
        x_big = x_big.to(device)
        targets = targets.to(device)

    return x_big, targets


def make_train_test_batches(
    cfg: LinearICLConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create paired train and test batches from a LinearICLConfig.

    Uses separate seeds for train and test to ensure no data leakage.

    Args:
        cfg: Configuration specifying dimensions, batch sizes, and seeds.
        device: Target device for output tensors.
        dtype: Floating-point precision.

    Returns:
        x_train: Training tokens [train_batch_size, seq_len, xdim + 1].
        y_train: Training targets [train_batch_size].
        x_test: Test tokens [test_batch_size, seq_len, xdim + 1].
        y_test: Test targets [test_batch_size].
    """
    x_train, y_train = generate_linear_icl_batch(
        batch_size=cfg.train_batch_size,
        seq_len=cfg.seq_len,
        xdim=cfg.xdim,
        sigma=cfg.sigma,
        seed_x=cfg.train_seed_x,
        seed_beta=cfg.train_seed_beta,
        seed_noise=cfg.train_seed_noise,
        device=device,
        dtype=dtype,
    )
    x_test, y_test = generate_linear_icl_batch(
        batch_size=cfg.test_batch_size,
        seq_len=cfg.seq_len,
        xdim=cfg.xdim,
        sigma=cfg.sigma,
        seed_x=cfg.test_seed_x,
        seed_beta=cfg.test_seed_beta,
        seed_noise=cfg.test_seed_noise,
        device=device,
        dtype=dtype,
    )
    return x_train, y_train, x_test, y_test
