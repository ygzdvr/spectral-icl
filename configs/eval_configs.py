"""Configuration dataclasses for model evaluation experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HandCodedEvalConfig:
    """Configuration for evaluating the hand-coded analytical attention model.

    This dataclass specifies problem dimensions, batch size, sequence lengths,
    model hyper-parameters, and random seeds for the isotropic linear regression
    ICL evaluation.

    Attributes:
        d: Feature dimensionality of the linear regression problem.
        B: Batch size (number of independent regression tasks sampled).
        P: Total sequence length (training + test examples).  The number of
            training examples is ``P - P_test``.
        L: Number of attention layers (depth) in the hand-coded model.
        P_test: Number of test (held-out) examples appended after the training
            examples in each sequence.
        beta: Step-size scaling factor for the attention update.  The per-layer
            update is scaled by ``beta / L``.
        seed_x: Random seed used to generate the input feature matrix *X*.
        seed_beta: Random seed used to generate the true regression
            coefficients *beta*.
    """

    d: int = 100
    B: int = 50
    P: int = 80
    L: int = 50
    P_test: int = 1
    beta: float = 100.0
    seed_x: int = 0
    seed_beta: int = 1


@dataclass(frozen=True)
class HardPowerLawDepthConfig:
    """Configuration for the power-law covariance depth evaluation.

    Controls the synthetic data distribution in which each feature coordinate
    *k* is scaled by ``k^{-exp_value}``, producing a power-law eigenspectrum.
    The hand-coded model is then run for ``5 * L`` layers to study how depth
    interacts with the spectral decay rate.

    Attributes:
        d: Feature dimensionality of the regression problem.
        B: Batch size (number of independent regression tasks).
        L: Base number of attention layers.  The actual evaluation runs
            ``5 * L`` layers (see :func:`run_hard_power_law_depth_eval`).
        P: Number of training examples in each sequence.
        P_test: Number of test examples appended after training examples.
        beta_model: Step-size scaling factor for the attention update.
        exp_value: Power-law exponent alpha.  Feature coordinate *k* is scaled
            by ``k^{-exp_value}``, so larger values produce faster spectral
            decay and a more anisotropic covariance.
        seed_x: Random seed for sampling the input matrix *X*.
        seed_beta: Random seed for sampling the true regression weights.
    """

    d: int = 100
    B: int = 50
    L: int = 40
    P: int = 120
    P_test: int = 40
    beta_model: float = 500.0
    exp_value: float = 1.0
    seed_x: int = 1
    seed_beta: int = 2


@dataclass(frozen=True)
class OODCovarianceEvalConfig:
    """Configuration for out-of-distribution covariance evaluation.

    Evaluates how the hand-coded linear attention model generalizes when
    input covariance structure varies across tasks with heterogeneous
    power-law exponents.

    Attributes:
        d: Ambient dimension of the linear regression problem.
        B: Batch size (number of tasks evaluated simultaneously).
        L: Number of attention layers (depth) in the hand-coded model.
        P: Number of training context tokens per task.
        P_test: Number of test tokens per task.
        seed_exp: Random seed for sampling the power-law exponents.
        seed_x: Random seed for sampling the input features X.
        seed_beta: Random seed for sampling the regression targets beta.
        exp_scale: Maximum power-law exponent. Each task's exponent is drawn
            from Uniform(0, exp_scale). Set to 0.0 for isotropic inputs.
        beta_model: Residual connection scaling parameter passed to model_eval.
    """

    d: int = 100
    B: int = 50
    L: int = 40
    P: int = 40
    P_test: int = 40
    seed_exp: int = 0
    seed_x: int = 1
    seed_beta: int = 2
    exp_scale: float = 0.0
    beta_model: float = 100.0


@dataclass(frozen=True)
class RandomInitCovarianceEvalConfig:
    """Configuration for random initialization covariance evaluation.

    Evaluates the hand-coded linear attention model when initialized with
    random Gaussian weights instead of analytically optimal structure.

    Attributes:
        d: Ambient dimension of the linear regression problem.
        B: Batch size (number of tasks evaluated simultaneously).
        L: Number of attention layers (depth) in the model.
        P: Number of training context tokens per task.
        P_test: Number of test tokens per task.
        sigma: Scale of random Gaussian initialization for weight matrices.
        beta_model: Residual connection scaling parameter passed to model_eval.
        exp_scale: Maximum power-law exponent when using random exponents.
            Ignored when fixed_exp is not None.
        fixed_exp: If not None, all tasks use this fixed power-law exponent.
            If None, exponents are sampled from Uniform(0, exp_scale).
        seed_exp: Random seed for sampling power-law exponents.
        seed_x: Random seed for sampling input features X.
        seed_beta: Random seed for sampling regression targets beta.
        seed_wx: Random seed for initializing W_x.
        seed_wy: Random seed for initializing W_y.
        seed_wq: Random seed for initializing Wq.
        seed_wk: Random seed for initializing Wk.
        seed_wv: Random seed for initializing Wv.
        seed_wout: Random seed for initializing w_out.
    """

    d: int = 100
    B: int = 50
    L: int = 40
    P: int = 120
    P_test: int = 40
    sigma: float = 0.1
    beta_model: float = 0.5
    exp_scale: float = 0.0
    fixed_exp: float | None = 1.0
    seed_exp: int = 0
    seed_x: int = 1
    seed_beta: int = 2
    seed_wx: int = 0
    seed_wy: int = 1
    seed_wq: int = 2
    seed_wk: int = 3
    seed_wv: int = 4
    seed_wout: int = 5
