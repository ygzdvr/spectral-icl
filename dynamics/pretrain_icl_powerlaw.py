from __future__ import annotations

import math

import torch

from configs.train_configs import (
    DecoupledTrainModelConfig,
    IsotropicDepthAlphaSweepConfig,
    PretrainICLPowerLawConfig,
    SampleMode,
)
from .linear_icl_dynamics import (
    model_eval,
    model_eval_decoupled_frozen_emb,
    model_eval_decoupled_softmax_frozen_emb,
    model_eval_decoupled_frozen_emb_trace,
)

from utils.theorem_a_utils import summarize_theorem_a_trace

Tensor = torch.Tensor


def _randn(shape: tuple[int, ...], seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Generate a seeded random normal tensor.

    Creates a fresh ``torch.Generator`` seeded with the given seed to produce
    reproducible Gaussian random tensors on the specified device.

    Args:
        shape: Shape of the output tensor.
        seed: Integer seed for the random number generator.
        device: Torch device on which to allocate the tensor.
        dtype: Data type of the output tensor.

    Returns:
        Tensor of shape ``shape`` with i.i.d. standard normal entries.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


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
    """Sample an i.i.d. isotropic linear regression ICL batch.

    Generates data for the "iid" sampling mode: input features X are drawn from
    a standard normal distribution (isotropic covariance), and per-batch teacher
    weights betas are also standard normal. Labels are computed as:

        y_{b,p} = (1 / sqrt(d)) * sum_k X_{b,p,k} * betas_{b,k}

    Args:
        d: Input feature dimension.
        B: Batch size (number of independent regression tasks).
        P_tr: Number of in-context training examples.
        P_te: Number of in-context test examples.
        seed: Random seed. Internally uses ``2*seed`` for X and ``2*seed+1`` for betas
            to ensure independence between inputs and teachers.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        X: Input features of shape ``[B, P_tr + P_te, d]``.
        y: Labels of shape ``[B, P_tr + P_te]``.
    """
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
    """Sample a linear regression ICL batch with fixed diagonal covariance.

    Generates data for the "spec" sampling mode. Input features X are drawn from
    N(0, diag(spec)) -- i.e., each coordinate k is scaled by sqrt(spec_k). The
    teacher weights betas are drawn i.i.d. standard normal per batch (w_star is
    unused in this mode). Labels are:

        y_{b,p} = sum_k X_{b,p,k} * betas_{b,k}

    Note: ``w_star`` is accepted for API compatibility but is not used. The
    covariance structure comes entirely from ``spec``.

    Args:
        spec: Eigenspectrum tensor of shape ``[d]``. Defines the diagonal
            covariance Sigma = diag(spec).
        w_star: Teacher weight vector of shape ``[d]``. Unused in this function
            (deleted immediately).
        B: Batch size.
        P_tr: Number of in-context training examples.
        P_te: Number of in-context test examples.
        seed: Random seed.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        X: Input features of shape ``[B, P_tr + P_te, d]`` with covariance
            diag(spec) along the feature axis.
        y: Labels of shape ``[B, P_tr + P_te]``.
    """
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
    """Sample an ICL batch with power-law covariance and random orthogonal teacher rotation.

    Generates data for the "spec_rotate" sampling mode. Input features X have
    diagonal covariance diag(spec) (same as ``sample_data_spec``). The teacher
    weights are obtained by applying a random orthogonal rotation (via QR
    decomposition) to the fixed w_star vector:

        O_b = QR(G_b).Q,   where G_b ~ N(0, I) of shape [d, d]
        betas_b = O_b @ w_star
        y_{b,p} = sum_k X_{b,p,k} * betas_{b,k}

    This ensures that each batch has a different teacher direction but preserves
    the norm structure encoded in w_star.

    Args:
        spec: Eigenspectrum tensor of shape ``[d]``.
        w_star: Teacher weight template of shape ``[d]``. Rotated by a random
            orthogonal matrix per batch element.
        B: Batch size.
        P_tr: Number of in-context training examples.
        P_te: Number of in-context test examples.
        seed: Random seed.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        X: Input features of shape ``[B, P_tr + P_te, d]``.
        y: Labels of shape ``[B, P_tr + P_te]``.
    """
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
    """Sample an ICL batch with Gaussian (non-orthogonal) random rotation of both inputs and teacher.

    Generates data for the "gauss_rotate" sampling mode. Unlike ``sample_data_spec_rotate``
    which uses an orthogonal rotation (QR), this mode uses a random Gaussian matrix
    O ~ N(0, 1/d) to rotate both the input features and the teacher weights:

        O_b ~ N(0, 1/d) of shape [d, d]
        Z_{b,p,k} ~ N(0, 1),  then scaled: Z * sqrt(spec_k)
        X_{b,p,l} = sum_k Z_{b,p,k} * O_{b,l,k}   (Gaussian rotation of inputs)
        betas_b = O_b @ w_star
        y_{b,p} = X_{b,p} . betas_b

    The effective covariance of X is O @ diag(spec) @ O^T, which is a random
    Wishart-like matrix rather than a diagonal.

    Args:
        spec: Eigenspectrum tensor of shape ``[d]``.
        w_star: Teacher weight template of shape ``[d]``.
        B: Batch size.
        P_tr: Number of in-context training examples.
        P_te: Number of in-context test examples.
        seed: Random seed.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        X: Input features of shape ``[B, P_tr + P_te, d]``, rotated by a
            random Gaussian matrix.
        y: Labels of shape ``[B, P_tr + P_te]``.
    """
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


def init_pretrain_params(
    d: int,
    N: int,
    *,
    sigma: float = 0.4,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """Initialize the six parameter matrices for the ICL attention model.

    Creates scaled identity-based initializations for the embedding, attention,
    and output parameters. The scaling follows:

        W_x = sqrt(2) * sqrt(N) * sigma * I_{N x d}
        W_y = ones(N)
        Wq = Wk = sigma * sqrt(N) * I_{N x N}
        Wv = sigma * sqrt(N) * I_{N x N}
        w_out = ones(N)   (clone of W_y)

    The model encodes input token [x_p, y_p] as h_p = W_x @ x_p + W_y * y_p,
    then applies attention with query/key/value projections, and extracts the
    scalar prediction via w_out.

    Args:
        d: Input feature dimension.
        N: Hidden dimension (width) of the model's internal representation.
        sigma: Initialization scale controlling the magnitude of weight matrices.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        List of six tensors ``[W_x, W_y, Wq, Wk, Wv, w_out]``:
            - W_x: shape ``[N, d]`` -- input feature embedding.
            - W_y: shape ``[N]`` -- label embedding.
            - Wq: shape ``[N, N]`` -- query projection.
            - Wk: shape ``[N, N]`` -- key projection.
            - Wv: shape ``[N, N]`` -- value projection.
            - w_out: shape ``[N]`` -- output readout vector.
    """
    device = torch.device(device)
    W_x = math.sqrt(2.0) * math.sqrt(N) * sigma * torch.eye(N, d, device=device, dtype=dtype)
    W_y = torch.ones((N,), device=device, dtype=dtype)
    Wq = sigma * math.sqrt(N) * torch.eye(N, device=device, dtype=dtype)
    Wk = Wq.clone()
    Wv = sigma * math.sqrt(N) * torch.eye(N, device=device, dtype=dtype)
    w_out = W_y.clone()
    return [W_x, W_y, Wq, Wk, Wv, w_out]


def make_powerlaw_problem(
    cfg: PretrainICLPowerLawConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, int]:
    """Construct a power-law eigenspectrum and corresponding teacher weight vector.

    Creates the problem structure for pretraining:

        spec_k = k^{-alpha}                          (unnormalized power-law spectrum)
        w_star_k = sqrt(k^{-alpha * beta - 1} / spec_k)
        w_star = w_star / sqrt(sum_k w_star_k^2 * spec_k)   (normalize so E[y^2]=1)

    The normalization ensures that ``sum_k w_star_k^2 * spec_k = 1``, so the
    signal variance is unit when X has covariance diag(spec). The hidden
    dimension N is computed as ``int(n_multiplier * d)``.

    Args:
        cfg: Configuration dataclass containing d, alpha, beta, and n_multiplier.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        spec: Eigenspectrum of shape ``[d]``. spec_k = k^{-alpha}.
        w_star: Normalized teacher weights of shape ``[d]``.
        N: Hidden dimension, equal to ``int(cfg.n_multiplier * cfg.d)``.
    """
    device = torch.device(device)
    d = cfg.d
    coords = torch.linspace(1, d, d, device=device, dtype=dtype)
    spec = coords.pow(-cfg.alpha)
    w_star = torch.sqrt(coords.pow(-cfg.alpha * cfg.beta - 1.0) / spec)
    w_star = w_star / torch.sqrt(torch.sum((w_star**2) * spec))
    N = int(cfg.n_multiplier * d)
    return spec, w_star, N


def make_normalized_powerlaw_problem(
    d: int,
    alpha: float,
    beta: float,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Construct a normalized power-law eigenspectrum and teacher weights.

    Like ``make_powerlaw_problem``, but additionally normalizes the spectrum so
    that ``sum_k spec_k = 1`` (i.e., trace of covariance equals 1):

        spec_k = k^{-alpha} / sum_j j^{-alpha}
        w_star_k = sqrt(k^{-alpha * beta - 1} / spec_k)
        w_star = w_star / sqrt(sum_k w_star_k^2 * spec_k)

    This normalization makes the total variance of each input feature vector
    equal to 1, which is useful for comparing across different alpha values.

    Args:
        d: Input dimension.
        alpha: Power-law exponent. spec_k ~ k^{-alpha} before normalization.
        beta: Teacher weight alignment exponent.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        spec: Normalized eigenspectrum of shape ``[d]`` summing to 1.
        w_star: Normalized teacher weights of shape ``[d]`` satisfying
            sum_k w_star_k^2 * spec_k = 1.
    """
    device = torch.device(device)
    coords = torch.linspace(1, d, d, device=device, dtype=dtype)
    spec = coords.pow(-alpha)
    spec = spec / torch.sum(spec)
    w_star = torch.sqrt(coords.pow(-alpha * beta - 1.0) / spec)
    w_star = w_star / torch.sqrt(torch.sum((w_star**2) * spec))
    return spec, w_star


def make_unnormalized_powerlaw_problem(
    d: int,
    alpha: float,
    beta: float,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Construct an unnormalized power-law eigenspectrum and teacher weights.

    Same as ``make_normalized_powerlaw_problem`` but without normalizing the
    spectrum to sum to 1. The raw power-law spectrum is used:

        spec_k = k^{-alpha}
        w_star_k = sqrt(k^{-alpha * beta - 1} / spec_k)
        w_star = w_star / sqrt(sum_k w_star_k^2 * spec_k)

    The teacher weights are still normalized so that the signal variance
    ``sum_k w_star_k^2 * spec_k = 1``, but the trace of the covariance
    ``sum_k spec_k`` is not constrained.

    This is identical to the spectrum construction in ``make_powerlaw_problem``
    but returns (spec, w_star) without the hidden dimension N.

    Args:
        d: Input dimension.
        alpha: Power-law exponent. spec_k = k^{-alpha}.
        beta: Teacher weight alignment exponent.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        spec: Unnormalized eigenspectrum of shape ``[d]``.
        w_star: Normalized teacher weights of shape ``[d]``.
    """
    device = torch.device(device)
    coords = torch.linspace(1, d, d, device=device, dtype=dtype)
    spec = coords.pow(-alpha)
    w_star = torch.sqrt(coords.pow(-alpha * beta - 1.0) / spec)
    w_star = w_star / torch.sqrt(torch.sum((w_star**2) * spec))
    return spec, w_star


def sample_pretrain_batch(
    cfg: PretrainICLPowerLawConfig,
    spec: Tensor,
    w_star: Tensor,
    *,
    seed: int,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Dispatch to the appropriate data sampling function based on cfg.sample_mode.

    Routes to one of ``sample_data``, ``sample_data_spec``,
    ``sample_data_spec_rotate``, or ``sample_data_gauss_rotate`` depending on
    the ``sample_mode`` field of the configuration.

    Args:
        cfg: Pretraining configuration. Uses ``sample_mode``, ``d``, ``B``,
            ``P_tr``, and ``P_test``.
        spec: Eigenspectrum tensor of shape ``[d]``.
        w_star: Teacher weight vector of shape ``[d]``.
        seed: Random seed for reproducible data generation.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        X: Input features of shape ``[B, P_tr + P_test, d]``.
        y: Labels of shape ``[B, P_tr + P_test]``.
    """
    if cfg.sample_mode == "iid":
        return sample_data(cfg.d, cfg.B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype)
    if cfg.sample_mode == "spec":
        return sample_data_spec(spec, w_star, cfg.B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype)
    if cfg.sample_mode == "gauss_rotate":
        return sample_data_gauss_rotate(
            spec, w_star, cfg.B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype
        )
    return sample_data_spec_rotate(
        spec, w_star, cfg.B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype
    )


def _sum_squares(params: list[Tensor]) -> Tensor:
    """Compute the total squared Frobenius norm of a list of parameter tensors.

    Calculates sum_i ||params_i||_F^2 = sum_i sum_j params_i[j]^2, used as the
    L2 regularization penalty in the training objective.

    Args:
        params: List of parameter tensors (any shape).

    Returns:
        Scalar tensor containing the sum of squared entries across all parameters.
    """
    return sum(torch.sum(p * p) for p in params)


def isotropic_dmft(
    alpha: float | Tensor,
    gamma: float | Tensor,
    T: int,
    *,
    iters: int = 100,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute the isotropic Dynamic Mean Field Theory (DMFT) prediction for ICL loss.

    Solves the DMFT self-consistent equations for a linear attention model
    performing in-context linear regression in the isotropic (flat spectrum) setting.
    The DMFT reduces the high-dimensional problem to a fixed-point iteration over
    T x T matrices.

    The iteration is:
        theta = gamma * L   (where L is the strict lower-triangular ones matrix)
        Initialize: H = (I + theta)^{-1}
        Repeat:
            inner = (I + (H @ theta) / alpha)^{-1}
            H = (I + theta @ inner)^{-1}

    The output ``vs = H @ 1`` gives the DMFT prediction for the residual error
    at each sequence position. ``vs[t]`` represents the expected prediction error
    after observing t in-context examples.

    Args:
        alpha: Ratio P_tr / d (number of in-context examples per dimension).
            Controls the statistical difficulty of the problem. Can be a scalar
            float or a 0-d tensor.
        gamma: Effective attention strength parameter, typically set to
            1 / (1 + 1/alpha) in the depth-vs-alpha sweep.
        T: Sequence length (matrix dimension for the DMFT system).
        iters: Number of fixed-point iterations. Default 100 is typically
            sufficient for convergence.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        vs: Tensor of shape ``[T]``. The DMFT-predicted residual error vector,
            where ``vs[t]`` is the expected squared prediction error after t
            in-context examples.
    """
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


def train_model(
    cfg: DecoupledTrainModelConfig,
    *,
    spec: Tensor | None = None,
    w_star: Tensor | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[list[float], list[float] | None]:
    """Train a decoupled linear-attention ICL model using SGD.

    Trains the attention model on synthetic linear regression ICL data with
    optional power-law spectral structure. The training objective is:

        loss = MSE(out[:, P_tr:] / gamma + y[:, P_tr:])
        reg_loss = N * gamma^2 * loss + lamb * ||trainable_params||^2

    where ``out`` is the model's prediction (negative residual) and the loss is
    computed only over the test positions (indices P_tr onward).

    Two training modes are supported:
    - **Online** (``cfg.online=True``): Each step uses a fresh random batch.
      Only training loss is tracked.
    - **Offline** (``cfg.online=False``): The same batch is reused every step.
      A separate held-out evaluation batch is generated to track test loss.

    Two model modes are supported:
    - **Unrestricted** (``cfg.unrestricted=True``): All 6 parameter matrices are
      trained, using ``model_eval`` for the forward pass.
    - **Decoupled** (``cfg.unrestricted=False``): Only W_x, Wq, Wk, Wv are
      trained; W_y and w_out are frozen, using
      ``model_eval_decoupled_frozen_emb``.

    Args:
        cfg: Training configuration dataclass.
        spec: Optional eigenspectrum tensor of shape ``[d]``. If provided, data
            is sampled with power-law covariance. If None, isotropic i.i.d. data
            is used.
        w_star: Optional teacher weight vector of shape ``[d]``. Required when
            spec is provided and ``random_rotate=True`` or in offline mode.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        pretrain_loss: List of length ``cfg.T`` containing the loss at each step.
            In online mode, this is the training loss. In offline mode, this is
            the held-out test loss.
        train_loss: In offline mode, list of length ``cfg.T`` with the training
            loss at each step. In online mode, returns None.
    """
    device = torch.device(device)
    params = init_pretrain_params(cfg.d, cfg.N, sigma=cfg.sigma, device=device, dtype=dtype)
    W_x, Wy, Wq, Wk, Wv, w_out = params

    if cfg.unrestricted:
        trainable = [
            torch.nn.Parameter(W_x.clone()),
            torch.nn.Parameter(Wy.clone()),
            torch.nn.Parameter(Wq.clone()),
            torch.nn.Parameter(Wk.clone()),
            torch.nn.Parameter(Wv.clone()),
            torch.nn.Parameter(w_out.clone()),
        ]
    else:
        trainable = [
            torch.nn.Parameter(W_x.clone()),
            torch.nn.Parameter(Wq.clone()),
            torch.nn.Parameter(Wk.clone()),
            torch.nn.Parameter(Wv.clone()),
        ]

    optimizer = torch.optim.SGD(trainable, lr=cfg.lr)
    pretrain_loss: list[float] = []
    train_loss: list[float] = []

    for t in range(cfg.T):
        seed_t = t if cfg.online else 0

        if spec is not None:
            if cfg.random_rotate:
                X, y = sample_data_spec_rotate(
                    spec, w_star, cfg.B, cfg.P_tr, cfg.P_test, seed=seed_t, device=device, dtype=dtype
                )
            else:
                X, y = sample_data_spec(
                    spec, w_star, cfg.B, cfg.P_tr, cfg.P_test, seed=seed_t, device=device, dtype=dtype
                )
        else:
            X, y = sample_data(cfg.d, cfg.B, cfg.P_tr, cfg.P_test, seed=t, device=device, dtype=dtype)

        optimizer.zero_grad(set_to_none=True)
        if cfg.unrestricted:
            out, _, _ = model_eval(
                trainable,
                X,
                y,
                L=cfg.L,
                P_test=cfg.P_test,
                beta=cfg.beta_model,
                qk_ln=False,
                divide_update_by_sqrt_d=False,
            )
        else:
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

        loss_value = float(loss.detach().cpu())
        if cfg.online:
            pretrain_loss.append(loss_value)
        else:
            assert spec is not None and w_star is not None
            X_eval, y_eval = sample_data_spec_rotate(
                spec, w_star, cfg.B, cfg.P_tr, cfg.P_test, seed=seed_t + 1, device=device, dtype=dtype
            )
            with torch.no_grad():
                if cfg.unrestricted:
                    out_eval, _, _ = model_eval(
                        trainable,
                        X_eval,
                        y_eval,
                        L=cfg.L,
                        P_test=cfg.P_test,
                        beta=cfg.beta_model,
                        qk_ln=False,
                        divide_update_by_sqrt_d=False,
                    )
                else:
                    out_eval, _, _ = model_eval_decoupled_frozen_emb(
                        trainable,
                        Wy,
                        X_eval,
                        y_eval,
                        L=cfg.L,
                        P_test=cfg.P_test,
                        beta=cfg.beta_model,
                        qk_ln=False,
                    )
                test_loss = torch.mean((out_eval[:, cfg.P_tr:] / cfg.gamma + y_eval[:, cfg.P_tr:]) ** 2)
            pretrain_loss.append(float(test_loss.detach().cpu()))
            train_loss.append(loss_value)

        if t % 100 == 0:
            if cfg.online:
                print(f"step {t} , loss = {loss_value:.8f}")
            else:
                print(
                    f"step {t} , train loss = {loss_value:.8f}, "
                    f"test loss = {pretrain_loss[-1]:.8f}"
                )

    if cfg.online:
        return pretrain_loss, None
    return pretrain_loss, train_loss


def run_depth_scaling_nonrotate_sweep(
    cfg: DecoupledTrainModelConfig,
    Lvals: list[int],
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Sweep over model depths without random rotation, using fixed diagonal covariance.

    For each depth L in ``Lvals``, trains a decoupled model with
    ``sample_mode="spec"`` (no rotation) and ``random_rotate=False``, keeping
    all other hyperparameters from ``cfg``. Uses the normalized power-law
    spectrum from ``make_normalized_powerlaw_problem``.

    Args:
        cfg: Base training configuration. The ``L`` field is overridden per sweep
            point; ``sample_mode`` is forced to "spec" and ``random_rotate`` to
            False.
        Lvals: List of depth values to sweep over.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        Dictionary with keys:
            - "spec": Tensor ``[d]`` -- the normalized power-law spectrum.
            - "w_star": Tensor ``[d]`` -- the normalized teacher weights.
            - "Lvals": The input list of depth values.
            - "all_losses": List of loss curves, one per L value. Each is a list
              of floats of length ``cfg.T``.
            - "all_train_losses": List of training loss curves (populated only
              when ``cfg.online=False``; empty list otherwise).
    """
    spec, w_star = make_normalized_powerlaw_problem(cfg.d, cfg.alpha, cfg.beta, device=device, dtype=dtype)
    all_losses: list[list[float]] = []
    all_train_losses: list[list[float]] = []

    for L in Lvals:
        print("")
        print(f"L = {L}")
        run_cfg = DecoupledTrainModelConfig(
            d=cfg.d,
            P_tr=cfg.P_tr,
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
            online=cfg.online,
            sample_mode="spec",
        )
        losses, train_losses_i = train_model(run_cfg, spec=spec, w_star=w_star, device=device, dtype=dtype)
        all_losses.append(losses)
        if train_losses_i is not None:
            all_train_losses.append(train_losses_i)

    return {
        "spec": spec.detach(),
        "w_star": w_star.detach(),
        "Lvals": Lvals,
        "all_losses": all_losses,
        "all_train_losses": all_train_losses,
    }


def run_ptr_scaling_sweep(
    cfg: DecoupledTrainModelConfig,
    p_trs: list[int],
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Sweep over P_tr (prompt length / number of in-context training examples).

    For each P_tr value, trains a decoupled model with fixed depth ``cfg.L``
    and ``sample_mode="spec"`` (no rotation), using the normalized power-law
    spectrum. Collects the full loss curve and the final converged loss for each
    P_tr value.

    Args:
        cfg: Base training configuration. The ``P_tr`` field is overridden per
            sweep point; ``sample_mode`` is forced to "spec" and
            ``random_rotate`` to False.
        p_trs: List of P_tr values (prompt lengths) to sweep over.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        Dictionary with keys:
            - "spec": Tensor ``[d]`` -- the normalized power-law spectrum.
            - "w_star": Tensor ``[d]`` -- the normalized teacher weights.
            - "p_trs": The input list of P_tr values.
            - "all_losses": List of loss curves, one per P_tr value.
            - "final_loss": List of final (converged) loss values, one per P_tr.
    """
    spec, w_star = make_normalized_powerlaw_problem(cfg.d, cfg.alpha, cfg.beta, device=device, dtype=dtype)
    all_losses: list[list[float]] = []

    for P_tr in p_trs:
        print("")
        print(f"L = {cfg.L}")
        run_cfg = DecoupledTrainModelConfig(
            d=cfg.d,
            P_tr=P_tr,
            P_test=cfg.P_test,
            B=cfg.B,
            N=cfg.N,
            L=cfg.L,
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
            online=cfg.online,
            sample_mode="spec",
        )
        losses, _ = train_model(run_cfg, spec=spec, w_star=w_star, device=device, dtype=dtype)
        all_losses.append(losses)

    final_loss = [loss[-1] for loss in all_losses]
    return {
        "spec": spec.detach(),
        "w_star": w_star.detach(),
        "p_trs": p_trs,
        "all_losses": all_losses,
        "final_loss": final_loss,
    }


def run_isotropic_depth_vs_alpha_sweep(
    cfg: IsotropicDepthAlphaSweepConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Run a joint sweep over (P_tr, depth) and compare against DMFT theory.

    This is the main experiment function for comparing trained ICL model
    performance against isotropic DMFT theoretical predictions. It:

    1. **Experimental phase**: For each (P_tr, L) pair in the grid
       ``cfg.p_trs x cfg.lvals``, trains a decoupled model with
       ``sample_mode="spec"`` and ``online=True``, collecting the full loss
       curve.
    2. **Theory phase**: Computes the DMFT prediction via ``isotropic_dmft``
       for a log-spaced range of alpha = P_tr / d values. For each alpha, the
       effective attention strength is gamma = 1 / (1 + 1/alpha).

    The experimental final losses and their standard deviations (estimated from
    the last 10 training steps) are returned alongside the theoretical curves
    for downstream plotting.

    Args:
        cfg: Sweep configuration containing grid dimensions, training
            hyperparameters, and theory solver settings.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        Dictionary with keys:
            - "spec": Tensor ``[d]`` -- the normalized power-law spectrum.
            - "w_star": Tensor ``[d]`` -- the normalized teacher weights.
            - "all_losses": Nested list of shape ``[len(p_trs)][len(lvals)][T]``
              containing loss curves for each (P_tr, L) pair.
            - "alpha_vals": Tensor ``[theory_alpha_points]`` -- log-spaced alpha
              values used for the theory curves.
            - "loss_np": Tensor ``[theory_T, theory_alpha_points]`` -- DMFT
              predicted error for each (time step, alpha) pair.
            - "all_losses_tr": Tensor ``[len(lvals), len(p_trs)]`` -- final
              experimental losses (transposed: rows=depths, cols=P_trs).
            - "all_losses_std": Tensor ``[len(lvals), len(p_trs)]`` -- standard
              error of the final losses estimated from the last 10 steps.
            - "p_trs": List of P_tr values used.
            - "lvals": List of depth values used.
    """
    spec, w_star = make_normalized_powerlaw_problem(cfg.d, cfg.alpha, cfg.beta, device=device, dtype=dtype)
    all_losses: list[list[list[float]]] = []

    for P_tr in cfg.p_trs:
        print("")
        print(f"P = {P_tr}")
        all_loss_i: list[list[float]] = []
        for L in cfg.lvals:
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
            losses, _ = train_model(run_cfg, spec=spec, w_star=w_star, device=device, dtype=dtype)
            all_loss_i.append(losses)
        all_losses.append(all_loss_i)

    alpha_vals = torch.logspace(
        cfg.theory_alpha_min_exp,
        cfg.theory_alpha_max_exp,
        cfg.theory_alpha_points,
        device=torch.device(device),
        dtype=dtype,
    )
    all_losses_th = [
        isotropic_dmft(
            alpha_val,
            1.0 / (1.0 + 1.0 / alpha_val),
            cfg.theory_T,
            iters=cfg.theory_iters,
            device=device,
            dtype=dtype,
        )
        for alpha_val in alpha_vals
    ]
    loss_np = torch.stack(all_losses_th, dim=1)

    all_losses_tr = torch.tensor(
        [[loss[-1] for loss in loss_i] for loss_i in all_losses],
        device=torch.device(device),
        dtype=dtype,
    ).T
    all_losses_std = torch.tensor(
        [
            [
                torch.tensor([loss[-k] for k in range(10)], device=torch.device(device), dtype=dtype).std(
                    correction=0
                )
                / math.sqrt(20.0)
                for loss in loss_i
            ]
            for loss_i in all_losses
        ],
        device=torch.device(device),
        dtype=dtype,
    ).T

    return {
        "spec": spec.detach(),
        "w_star": w_star.detach(),
        "all_losses": all_losses,
        "alpha_vals": alpha_vals.detach(),
        "loss_np": loss_np.detach(),
        "all_losses_tr": all_losses_tr.detach(),
        "all_losses_std": all_losses_std.detach(),
        "p_trs": list(cfg.p_trs),
        "lvals": list(cfg.lvals),
    }


def run_pretrain_icl_powerlaw(
    cfg: PretrainICLPowerLawConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Train a full (non-decoupled) ICL attention model on power-law data.

    This is the primary pretraining entry point for the full model (all 6
    parameter matrices are trainable). It:

    1. Constructs the power-law spectrum and teacher weights via
       ``make_powerlaw_problem``.
    2. Initializes model parameters via ``init_pretrain_params``.
    3. Trains for ``cfg.T`` steps of SGD with the objective:

           loss = MSE(out[:, P_tr:] / gamma + y[:, P_tr:])
           reg_loss = N * loss + lamb * ||params||^2

       Note: unlike ``train_model``, the regularized loss here uses ``N * loss``
       (not ``N * gamma^2 * loss``).
    4. Returns the trained parameters and loss history.

    Data is sampled fresh each step (online mode) using the sampling strategy
    specified by ``cfg.sample_mode``.

    Args:
        cfg: Pretraining configuration dataclass.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        Dictionary with keys:
            - "spec": Tensor ``[d]`` -- the power-law eigenspectrum.
            - "w_star": Tensor ``[d]`` -- the normalized teacher weights.
            - "N": int -- hidden dimension (width).
            - "params": List of 6 detached parameter tensors
              ``[W_x, W_y, Wq, Wk, Wv, w_out]``.
            - "pretrain_loss": List of length ``cfg.T`` with loss values.
            - "last_X": Tensor ``[B, P_tr + P_test, d]`` -- last training batch inputs.
            - "last_y": Tensor ``[B, P_tr + P_test]`` -- last training batch labels.
    """
    device = torch.device(device)
    spec, w_star, N = make_powerlaw_problem(cfg, device=device, dtype=dtype)

    params = init_pretrain_params(cfg.d, N, sigma=cfg.sigma, device=device, dtype=dtype)
    trainable = [torch.nn.Parameter(p.clone()) for p in params]
    optimizer = torch.optim.SGD(trainable, lr=cfg.lr)

    pretrain_loss: list[float] = []
    last_X: Tensor | None = None
    last_y: Tensor | None = None

    for t in range(cfg.T):
        X, y = sample_pretrain_batch(cfg, spec, w_star, seed=t, device=device, dtype=dtype)
        last_X, last_y = X, y

        optimizer.zero_grad(set_to_none=True)
        out, _, _ = model_eval(
            trainable,
            X,
            y,
            L=cfg.L,
            P_test=cfg.P_test,
            beta=cfg.beta_model,
            qk_ln=False,
            divide_update_by_sqrt_d=False,
        )
        loss = torch.mean((out[:, cfg.P_tr:] / cfg.gamma + y[:, cfg.P_tr:]) ** 2)
        reg_loss = N * loss + cfg.lamb * _sum_squares(trainable)
        reg_loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        pretrain_loss.append(loss_value)
        if t % 25 == 0:
            print(f"Loss step {t}, loss = {loss_value:.8f}")

    assert last_X is not None
    assert last_y is not None
    return {
        "spec": spec.detach(),
        "w_star": w_star.detach(),
        "N": N,
        "params": [p.detach() for p in trainable],
        "pretrain_loss": pretrain_loss,
        "last_X": last_X.detach(),
        "last_y": last_y.detach(),
    }


def run_powerlaw_depth_sweep(
    cfg: DecoupledTrainModelConfig,
    Lvals: list[int],
    *,
    normalize_spec: bool = True,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Sweep over model depths with power-law spectral structure.

    For each depth L in ``Lvals``, trains a decoupled model using the full
    configuration from ``cfg`` (including its ``sample_mode`` and
    ``random_rotate`` settings). Unlike ``run_depth_scaling_nonrotate_sweep``,
    this function preserves the original sampling mode and rotation settings.

    The spectrum can be either normalized (sum to 1) or unnormalized, controlled
    by the ``normalize_spec`` flag.

    Args:
        cfg: Base training configuration. The ``L`` field is overridden per
            sweep point; all other fields (including ``sample_mode`` and
            ``random_rotate``) are preserved from the input config.
        Lvals: List of depth values to sweep over.
        normalize_spec: If True, use ``make_normalized_powerlaw_problem``
            (spectrum sums to 1). If False, use
            ``make_unnormalized_powerlaw_problem`` (raw power-law spectrum).
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        Dictionary with keys:
            - "spec": Tensor ``[d]`` -- the power-law spectrum.
            - "w_star": Tensor ``[d]`` -- the normalized teacher weights.
            - "Lvals": The input list of depth values.
            - "all_losses": List of loss curves, one per L value.
    """
    if normalize_spec:
        spec, w_star = make_normalized_powerlaw_problem(
            cfg.d, cfg.alpha, cfg.beta, device=device, dtype=dtype
        )
    else:
        spec, w_star = make_unnormalized_powerlaw_problem(
            cfg.d, cfg.alpha, cfg.beta, device=device, dtype=dtype
        )
    all_losses: list[list[float]] = []

    for L in Lvals:
        print("")
        print(f"L = {L}")
        run_cfg = DecoupledTrainModelConfig(
            d=cfg.d,
            P_tr=cfg.P_tr,
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
            random_rotate=cfg.random_rotate,
            unrestricted=cfg.unrestricted,
            online=cfg.online,
            sample_mode=cfg.sample_mode,
        )
        losses, _ = train_model(run_cfg, spec=spec, w_star=w_star, device=device, dtype=dtype)
        all_losses.append(losses)

    return {
        "spec": spec.detach(),
        "w_star": w_star.detach(),
        "Lvals": Lvals,
        "all_losses": all_losses,
    }


def train_model_softmax(
    cfg: DecoupledTrainModelConfig,
    *,
    spec: Tensor | None = None,
    w_star: Tensor | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[list[float], list[float] | None]:
    """Train a decoupled ICL model using exponential (softmax) attention instead of linear attention.

    Identical to ``train_model`` in structure and training objective, but uses
    ``model_eval_decoupled_softmax_frozen_emb`` for the forward pass when
    ``unrestricted=False``. This replaces the linear dot-product attention with
    softmax-normalized attention (exponential kernel), enabling comparison between
    linear and softmax attention scaling behaviors.

    The training objective is the same as ``train_model``:

        loss = MSE(out[:, P_tr:] / gamma + y[:, P_tr:])
        reg_loss = N * gamma^2 * loss + lamb * ||trainable_params||^2

    When ``unrestricted=True``, falls back to ``model_eval`` (same as
    ``train_model``), so softmax is only used in the decoupled (frozen
    embedding) case.

    Args:
        cfg: Training configuration dataclass. Same fields as for ``train_model``.
        spec: Optional eigenspectrum tensor of shape ``[d]``. If provided, data
            is sampled with power-law covariance. If None, isotropic i.i.d. data
            is used.
        w_star: Optional teacher weight vector of shape ``[d]``.
        device: Torch device.
        dtype: Tensor data type.

    Returns:
        pretrain_loss: List of length ``cfg.T`` containing the loss at each step.
            In online mode, this is the training loss. In offline mode, this is
            the held-out test loss.
        train_loss: In offline mode, list of length ``cfg.T`` with the training
            loss at each step. In online mode, returns None.
    """
    device = torch.device(device)
    params = init_pretrain_params(cfg.d, cfg.N, sigma=cfg.sigma, device=device, dtype=dtype)
    W_x, Wy, Wq, Wk, Wv, w_out = params

    if cfg.unrestricted:
        trainable = [
            torch.nn.Parameter(W_x.clone()),
            torch.nn.Parameter(Wy.clone()),
            torch.nn.Parameter(Wq.clone()),
            torch.nn.Parameter(Wk.clone()),
            torch.nn.Parameter(Wv.clone()),
            torch.nn.Parameter(w_out.clone()),
        ]
    else:
        trainable = [
            torch.nn.Parameter(W_x.clone()),
            torch.nn.Parameter(Wq.clone()),
            torch.nn.Parameter(Wk.clone()),
            torch.nn.Parameter(Wv.clone()),
        ]

    optimizer = torch.optim.SGD(trainable, lr=cfg.lr)
    pretrain_loss: list[float] = []
    train_loss: list[float] = []

    for t in range(cfg.T):
        seed_t = t if cfg.online else 0

        if spec is not None:
            if cfg.random_rotate:
                X, y = sample_data_spec_rotate(
                    spec, w_star, cfg.B, cfg.P_tr, cfg.P_test, seed=seed_t, device=device, dtype=dtype
                )
            else:
                X, y = sample_data_spec(
                    spec, w_star, cfg.B, cfg.P_tr, cfg.P_test, seed=seed_t, device=device, dtype=dtype
                )
        else:
            X, y = sample_data(cfg.d, cfg.B, cfg.P_tr, cfg.P_test, seed=t, device=device, dtype=dtype)

        optimizer.zero_grad(set_to_none=True)
        if cfg.unrestricted:
            out, _, _ = model_eval(
                trainable, X, y, L=cfg.L, P_test=cfg.P_test,
                beta=cfg.beta_model, qk_ln=False, divide_update_by_sqrt_d=False,
            )
        else:
            out, _, _ = model_eval_decoupled_softmax_frozen_emb(
                trainable, Wy, X, y, L=cfg.L, P_test=cfg.P_test,
                beta=cfg.beta_model, qk_ln=False,
            )

        loss = torch.mean((out[:, cfg.P_tr:] / cfg.gamma + y[:, cfg.P_tr:]) ** 2)
        reg_loss = cfg.N * (cfg.gamma**2) * loss + cfg.lamb * _sum_squares(trainable)
        reg_loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if cfg.online:
            pretrain_loss.append(loss_value)
        else:
            assert spec is not None and w_star is not None
            X_eval, y_eval = sample_data_spec_rotate(
                spec, w_star, cfg.B, cfg.P_tr, cfg.P_test, seed=seed_t + 1, device=device, dtype=dtype
            )
            with torch.no_grad():
                if cfg.unrestricted:
                    out_eval, _, _ = model_eval(
                        trainable, X_eval, y_eval, L=cfg.L, P_test=cfg.P_test,
                        beta=cfg.beta_model, qk_ln=False, divide_update_by_sqrt_d=False,
                    )
                else:
                    out_eval, _, _ = model_eval_decoupled_softmax_frozen_emb(
                        trainable, Wy, X_eval, y_eval, L=cfg.L, P_test=cfg.P_test,
                        beta=cfg.beta_model, qk_ln=False,
                    )
                test_loss = torch.mean((out_eval[:, cfg.P_tr:] / cfg.gamma + y_eval[:, cfg.P_tr:]) ** 2)
            pretrain_loss.append(float(test_loss.detach().cpu()))
            train_loss.append(loss_value)

        if t % 100 == 0:
            if cfg.online:
                print(f"L = {cfg.L}, step {t} , loss = {loss_value:.8f}")
            else:
                print(
                    f"L = {cfg.L} , step {t} , train loss = {loss_value:.8f}, "
                    f"test loss = {pretrain_loss[-1]:.8f}"
                )

    if cfg.online:
        return pretrain_loss, None
    return pretrain_loss, train_loss

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

    if spec is None:
        raise ValueError("spec is required for non-iid sample modes.")
    if w_star is None:
        raise ValueError("w_star is required for non-iid sample modes.")

    if cfg.sample_mode == "spec":
        return sample_data_spec(spec, w_star, B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype)
    if cfg.sample_mode == "spec_rotate":
        return sample_data_spec_rotate(spec, w_star, B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype)
    if cfg.sample_mode == "gauss_rotate":
        return sample_data_gauss_rotate(spec, w_star, B, cfg.P_tr, cfg.P_test, seed=seed, device=device, dtype=dtype)

    raise ValueError(f"Unsupported sample_mode: {cfg.sample_mode}")

def train_model_with_checkpoints(
    cfg: DecoupledTrainModelConfig,
    *,
    spec: Tensor | None = None,
    w_star: Tensor | None = None,
    checkpoint_steps: tuple[int, ...] = (0,),
    debug_seed: int = 1234,
    debug_batch_size: int = 64,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    if cfg.unrestricted:
        raise ValueError("Theorem-A audit should start with unrestricted=False.")

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

    losses: list[float] = []
    checkpoints: dict[int, dict[str, object]] = {}

    def save_checkpoint(step: int) -> None:
        with torch.no_grad():
            _, trace = model_eval_decoupled_frozen_emb_trace(
                [p.detach() for p in trainable],
                Wy.detach(),
                X_dbg,
                y_dbg,
                L=cfg.L,
                P_test=cfg.P_test,
                beta=cfg.beta_model,
                store_full_tensors=False,
            )
            checkpoints[step] = {
                "params_tr": [p.detach().cpu().clone() for p in trainable],
                "Wy": Wy.detach().cpu().clone(),
                "summary": summarize_theorem_a_trace(trace),
            }

    if 0 in checkpoint_steps:
        save_checkpoint(0)

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

        losses.append(float(loss.detach().cpu()))

        step = t + 1
        if step in checkpoint_steps:
            save_checkpoint(step)

    return {
        "losses": losses,
        "checkpoints": checkpoints,
        "X_debug": X_dbg.detach().cpu(),
        "y_debug": y_dbg.detach().cpu(),
    }