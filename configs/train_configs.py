"""Configuration dataclasses for pretraining and sweep experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SampleMode = Literal["iid", "spec", "spec_rotate", "gauss_rotate"]


@dataclass(frozen=True)
class PretrainICLPowerLawConfig:
    """Configuration for pretraining an ICL attention model on power-law structured data.

    Controls the data generation, model architecture, and optimization hyperparameters
    for the full (non-decoupled) pretraining pipeline used by ``run_pretrain_icl_powerlaw``.

    The input covariance has a power-law eigenspectrum: spec_k = k^{-alpha}, and the
    teacher weight vector w_star is scaled according to both alpha and beta to produce
    a structured regression problem.

    Attributes:
        d: Input dimension of the linear regression problem.
        P_tr: Number of in-context training examples (prompt length).
        P_test: Number of in-context test examples appended after the prompt.
        B: Batch size (number of independent ICL tasks per gradient step).
        L: Number of attention layers (depth of the model).
        alpha: Power-law exponent for the eigenspectrum. spec_k = k^{-alpha}.
            Higher alpha means faster spectral decay (more low-rank structure).
        beta: Exponent controlling the teacher weight alignment with the spectrum.
            w_star_k ~ k^{(-alpha * beta - 1) / 2} / sqrt(spec_k), then normalized.
        T: Number of SGD training steps.
        lr: Learning rate for SGD optimizer.
        lamb: L2 regularization coefficient on all trainable parameters.
        beta_model: Residual connection scaling factor. Each layer's update is
            multiplied by beta_model / depth.
        n_multiplier: Hidden dimension N is set to int(n_multiplier * d).
        gamma: Output scaling factor. The model output is divided by gamma before
            computing the prediction loss.
        sigma: Standard deviation scale for parameter initialization.
        sample_mode: Data sampling strategy. One of:
            - "iid": Isotropic Gaussian inputs, random teacher per batch.
            - "spec": Power-law covariance in the canonical basis, random teacher.
            - "spec_rotate": Power-law covariance with random QR rotation per batch.
            - "gauss_rotate": Power-law covariance with Gaussian rotation per batch.
    """

    d: int = 60
    P_tr: int = 85
    P_test: int = 16
    B: int = 1024
    L: int = 4
    alpha: float = 1.5
    beta: float = 1.75
    T: int = 10000
    lr: float = 0.08
    lamb: float = 1.0e-14
    beta_model: float = 1.0
    n_multiplier: float = 1.4
    gamma: float = 1.0
    sigma: float = 0.4
    sample_mode: SampleMode = "spec_rotate"


@dataclass(frozen=True)
class DecoupledTrainModelConfig:
    """Configuration for training a decoupled ICL attention model.

    In the "decoupled" setting, the embedding weights (W_y, w_out) can optionally
    be frozen while only the attention weights (W_x, Wq, Wk, Wv) are trained.
    This is controlled by the ``unrestricted`` flag.

    The training objective is:
        reg_loss = N * gamma^2 * MSE(out / gamma + y_test) + lamb * ||params||^2

    where MSE is computed over the test positions only.

    Attributes:
        d: Input dimension of the linear regression problem.
        P_tr: Number of in-context training examples (prompt length).
        P_test: Number of in-context test examples appended after the prompt.
        B: Batch size (number of independent ICL tasks per gradient step).
        N: Hidden dimension (width) of the attention model's internal representation.
        L: Number of attention layers (depth).
        beta_model: Residual connection scaling factor (beta_model / depth per layer).
        gamma: Output scaling factor. Model output is divided by gamma before
            computing prediction loss.
        T: Number of SGD training steps.
        lr: Learning rate for SGD optimizer.
        lamb: L2 regularization coefficient on all trainable parameters.
        alpha: Power-law exponent for the input covariance eigenspectrum.
            spec_k = k^{-alpha}. When alpha=0, spectrum is flat (isotropic).
        beta: Exponent controlling teacher weight alignment with the spectrum.
        sigma: Standard deviation scale for parameter initialization.
        random_rotate: If True, apply a random QR rotation to data each batch
            (uses ``sample_data_spec_rotate`` instead of ``sample_data_spec``).
        unrestricted: If True, all six parameter matrices are trainable.
            If False, only W_x, Wq, Wk, Wv are trained (W_y and w_out are frozen).
        online: If True, use a fresh random seed per step (online SGD).
            If False, reuse the same data and separately evaluate on a held-out
            batch to track generalization.
        sample_mode: Data sampling strategy. One of "iid", "spec",
            "spec_rotate", "gauss_rotate".
    """

    d: int = 12
    P_tr: int = 1024
    P_test: int = 16
    B: int = 512
    N: int = 12
    L: int = 1
    beta_model: float = 1.0
    gamma: float = 1.0
    T: int = 10000
    lr: float = 0.125
    lamb: float = 1.0e-14
    alpha: float = 0.0
    beta: float = 1.75
    sigma: float = 0.4
    random_rotate: bool = False
    unrestricted: bool = False
    online: bool = True
    sample_mode: SampleMode = "spec"


@dataclass(frozen=True)
class IsotropicDepthAlphaSweepConfig:
    """Configuration for sweeping over depth and P_tr in the isotropic setting,
    comparing trained model loss against Dynamic Mean Field Theory (DMFT) predictions.

    This config drives ``run_isotropic_depth_vs_alpha_sweep``, which:
    1. Trains decoupled models for each (P_tr, L) combination in the grid
       defined by ``p_trs`` x ``lvals``.
    2. Computes DMFT theoretical predictions via ``isotropic_dmft`` over a
       log-spaced range of alpha values (the ratio P_tr / d).
    3. Returns both experimental final losses and theoretical curves for comparison.

    Attributes:
        d: Input dimension.
        P_test: Number of in-context test examples.
        B: Batch size per SGD step.
        N: Hidden dimension (width) of the attention model.
        alpha: Power-law exponent for eigenspectrum. alpha=0 gives isotropic (flat).
        beta: Teacher weight alignment exponent.
        T: Number of SGD training steps per (P_tr, L) run.
        lr: Learning rate for SGD.
        lamb: L2 regularization coefficient.
        beta_model: Residual connection scaling factor.
        gamma: Output scaling factor.
        sigma: Parameter initialization scale.
        p_trs: Tuple of P_tr values (prompt lengths) to sweep over.
        lvals: Tuple of L values (depths) to sweep over.
        unrestricted: If True, train all parameters; if False, freeze embeddings.
        theory_alpha_min_exp: Log10 of minimum alpha value for DMFT theory curve.
        theory_alpha_max_exp: Log10 of maximum alpha value for DMFT theory curve.
        theory_alpha_points: Number of alpha values in the DMFT theory curve.
        theory_T: Sequence length T used in DMFT fixed-point iteration.
        theory_iters: Number of fixed-point iterations in DMFT solver.
    """

    d: int = 32
    P_test: int = 32
    B: int = 512
    N: int = 32
    alpha: float = 0.0
    beta: float = 1.75
    T: int = 6000
    lr: float = 0.125
    lamb: float = 1.0e-14
    beta_model: float = 1.0
    gamma: float = 1.0
    sigma: float = 0.4
    p_trs: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    lvals: tuple[int, ...] = (1, 2, 4, 8, 16)
    unrestricted: bool = False
    theory_alpha_min_exp: float = -1.0
    theory_alpha_max_exp: float = 1.0
    theory_alpha_points: int = 100
    theory_T: int = 512
    theory_iters: int = 100
