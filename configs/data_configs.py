"""Configuration dataclasses for data generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearICLConfig:
    """Configuration for generating linear ICL regression batches.

    Controls the data dimensionality, sequence length, noise level, batch sizes,
    and random seeds for reproducible train/test splits.

    Each batch contains B independent linear regression problems where
    y_i = <x_i, beta> / sqrt(d) + sigma * noise_i.

    Attributes:
        xdim: Dimensionality of the input vectors x_i (d).
        seq_len: Number of (x, y) examples per sequence, including the query.
            The first seq_len-1 examples have visible labels; the last has y=0.
        sigma: Standard deviation of additive Gaussian label noise.
        train_batch_size: Number of independent regression tasks in training batch.
        test_batch_size: Number of independent regression tasks in test batch.
        train_seed_x: RNG seed for training input vectors X.
        train_seed_beta: RNG seed for training regression vectors beta.
        train_seed_noise: RNG seed for training label noise.
        test_seed_x: RNG seed for test input vectors X.
        test_seed_beta: RNG seed for test regression vectors beta.
        test_seed_noise: RNG seed for test label noise.
    """

    xdim: int = 15
    seq_len: int = 50
    sigma: float = 5e-2
    train_batch_size: int = 250
    test_batch_size: int = 250
    train_seed_x: int = 0
    train_seed_beta: int = 1
    train_seed_noise: int = 10
    test_seed_x: int = 3
    test_seed_beta: int = 4
    test_seed_noise: int = 5
