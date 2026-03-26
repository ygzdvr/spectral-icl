"""Dimension-free and isotropic linear attention models for ICL dynamics.

This module implements two variants of linear attention models for studying
in-context learning dynamics, along with their corresponding reduced
four-variable theoretical models.

Isotropic Variant (``*_isotropic`` functions):
  - Uses W_v scaled by sqrt(d) and divides the attention update by P_tr * sqrt(d).
  - Designed for isotropic (identity covariance) data distributions.
  - The effective gamma factorizes as wx^2 * wq * wk * wv.

Dimension-Free Variant (``*_dim_free`` functions):
  - Uses W_v scaled by sqrt(N) and divides the attention update by P_tr only.
  - Designed for spectral (power-law covariance) data distributions.
  - Uses QR-rotated data with Bernoulli-signed targets.

Both variants share a "frozen embedding" architecture where the input
embedding weights (W_x, W_y) are fixed and only the attention weights
(W_q, W_k, W_v) are trained.

Reduced Theory (``reduced_theory_four_var_*`` functions):
  - Replaces the full matrix dynamics with 4 scalar variables
    [wx, wq, wk, wv] whose product wx^2 * wq * wk * wv = gamma.
  - The loss is computed analytically by integrating over the
    Marchenko-Pastur distribution (isotropic) or the given spectrum (spectral).
  - Gradient descent on the 4 variables captures the essential nonlinear
    dynamics of the full model.
"""
from __future__ import annotations

import math
import sys

import torch


Tensor = torch.Tensor


def init_params_isotropic(
    d: int,
    N: int,
    *,
    sigma: float = 0.4,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> list[Tensor]:
    """Initialize parameters for the isotropic linear attention model.

    Creates diagonal weight matrices with specific scaling conventions:
      - W_x = sqrt(2) * sigma * I_d    (d x d input projection)
      - W_y = ones(N) / sqrt(N)        (N-dim label embedding, normalized)
      - Wq  = sigma * I_N              (N x N query projection)
      - Wk  = sigma * I_N              (N x N key projection, same as Wq)
      - Wv  = sigma * sqrt(d) * I_N    (N x N value projection, sqrt(d) scaling)
      - w_out = W_y                    (output projection, same as label embedding)

    The sqrt(2) factor on W_x and the sqrt(d) on Wv are chosen so that the
    effective gamma = wx^2 * wq * wk * wv has the correct scaling for the
    isotropic model where the update is divided by P_tr * sqrt(d).

    Args:
        d: Input dimension of the linear regression problem.
        N: Hidden dimension (width) of the attention model.
        sigma: Initialization scale for the weight matrices.
        device: Torch device for the parameters.
        dtype: Torch dtype for the parameters.

    Returns:
        List of 6 tensors: [W_x, W_y, Wq, Wk, Wv, w_out].
    """
    device = torch.device(device)
    W_x = math.sqrt(2.0) * sigma * torch.eye(d, device=device, dtype=dtype)
    W_y = torch.ones(N, device=device, dtype=dtype) / math.sqrt(N)
    Wq = sigma * torch.eye(N, device=device, dtype=dtype)
    Wk = Wq.clone()
    Wv = sigma * math.sqrt(d) * torch.eye(N, device=device, dtype=dtype)
    w_out = W_y.clone()
    return [W_x, W_y, Wq, Wk, Wv, w_out]


def _build_y_mask(batch: int, seq_len: int, p_tr: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build a label mask that zeros out test-position labels.

    Creates a mask of ones for training positions (indices 0..p_tr-1) and
    zeros for test positions (indices p_tr..seq_len-1). This prevents the
    model from seeing test labels during the forward pass.

    Args:
        batch: Batch size.
        seq_len: Total sequence length (train + test).
        p_tr: Number of training positions.
        device: Torch device for the mask.
        dtype: Torch dtype for the mask.

    Returns:
        Tensor of shape ``(batch, seq_len)`` with 1s at train positions
        and 0s at test positions.
    """
    mask = torch.ones(batch, seq_len, device=device, dtype=dtype)
    mask[:, p_tr:] = 0.0
    return mask


def _build_attn_mask(seq_len: int, p_tr: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build an attention mask that prevents attending to test positions.

    Creates a (seq_len, seq_len) mask where columns corresponding to test
    positions (indices p_tr..seq_len-1) are zeroed out. This ensures that
    all tokens (including test tokens) can only attend to training tokens.

    Args:
        seq_len: Total sequence length (train + test).
        p_tr: Number of training positions (columns 0..p_tr-1 are unmasked).
        device: Torch device for the mask.
        dtype: Torch dtype for the mask.

    Returns:
        Tensor of shape ``(seq_len, seq_len)`` with 1s in columns 0..p_tr-1
        and 0s in columns p_tr..seq_len-1.
    """
    mask = torch.ones(seq_len, seq_len, device=device, dtype=dtype)
    mask[:, p_tr:] = 0.0
    return mask


def model_eval_isotropic(
    params_tr: list[Tensor],
    Wy: Tensor,
    X: Tensor,
    y: Tensor,
    L: int = 100,
    P_test: int = 1,
    beta: float = 1.0,
) -> tuple[Tensor, list[float], list[float]]:
    """Run the isotropic linear attention model forward for L layers.

    This is a "frozen embedding" model where the input/output embeddings
    (W_x, W_y) are fixed and only the attention weights (Wq, Wk, Wv) are
    trainable. The architecture uses linear (non-softmax) attention.

    The forward pass for each layer is:
      1. q = hx @ Wq^T,  k = hx @ Wk^T,  v = hy @ Wv^T
      2. A = k @ q^T  (linear attention scores, no softmax)
      3. update = A_masked @ v  (masked to attend only to training tokens)
      4. hy <- hy - (beta / L) * update / (P_tr * sqrt(d))

    Key normalizations specific to the isotropic variant:
      - No 1/sqrt(N) scaling on q, k, v projections.
      - No 1/N scaling on the attention matrix.
      - Update divided by P_tr * sqrt(d) (the sqrt(d) accounts for isotropic
        data scaling).
      - Output projection: out = hy @ W_y (no 1/N factor).

    Args:
        params_tr: List of 4 trainable tensors [W_x, Wq, Wk, Wv] where
            W_x is (N, d) and Wq, Wk, Wv are (N, N).
        Wy: Frozen label embedding vector of shape ``(N,)``.
        X: Input features of shape ``(batch, seq_len, d)``.
        y: Labels of shape ``(batch, seq_len)``.
        L: Number of attention layers to apply.
        P_test: Number of test tokens at the end of each sequence.
        beta: Residual connection scaling factor.

    Returns:
        A tuple ``(out, train_losses, test_losses)`` where:
            - ``out``: Predictions of shape ``(batch, seq_len)``.
            - ``train_losses``: Empty list (not computed in this variant).
            - ``test_losses``: Empty list (not computed in this variant).
    """
    W_x, Wq, Wk, Wv = params_tr
    device = X.device
    dtype = X.dtype

    N, d = W_x.shape
    seq_len = X.shape[1]
    p_tr = seq_len - P_test
    batch = X.shape[0]

    hx = torch.einsum("bpd,nd->bpn", X, W_x)
    mask_y = _build_y_mask(batch, seq_len, p_tr, device=device, dtype=dtype)
    hy = torch.einsum("bp,n->bpn", y * mask_y, Wy)
    mask = _build_attn_mask(seq_len, p_tr, device=device, dtype=dtype)

    sqrt_d = math.sqrt(d)
    for _ in range(L):
        q = torch.einsum("bpn,kn->bpk", hx, Wq)
        k = torch.einsum("bpn,kn->bpk", hx, Wk)
        v = torch.einsum("bpn,kn->bpk", hy, Wv)
        A = torch.einsum("bpk,bqk->bpq", k, q)

        masked_A = A * mask
        update = torch.bmm(masked_A, v)
        hy = hy - (beta / L) * update / (float(p_tr) * sqrt_d)

    out = torch.einsum("bpn,n->bp", hy, Wy)
    return out, [], []


def _randn(shape: tuple[int, ...], seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Generate a standard normal tensor with a deterministic device-local seed.

    Args:
        shape: Shape of the output tensor.
        seed: Random seed for reproducibility.
        device: Target device for the output tensor and generator.
        dtype: Data type for the output tensor.

    Returns:
        Tensor of the given shape filled with i.i.d. N(0,1) values.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


def sample_data_gauss_isotropic(
    d: int,
    B: int,
    P_tr: int,
    P_te: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of isotropic Gaussian ICL regression tasks.

    Generates B independent linear regression tasks where:
      - X ~ N(0, I_d), with P_tr + P_te tokens per task.
      - beta ~ N(0, I_d), one per task.
      - y = X @ beta / sqrt(d).

    Deterministic seeding: seed 2*seed for X, seed 2*seed+1 for betas.

    Args:
        d: Input dimension.
        B: Batch size (number of tasks).
        P_tr: Number of training context tokens.
        P_te: Number of test tokens.
        seed: Base random seed.
        device: Torch device for outputs.
        dtype: Torch dtype for outputs.

    Returns:
        A tuple ``(X, y)`` where:
            - ``X``: Input features of shape ``(B, P_tr+P_te, d)``.
            - ``y``: Labels of shape ``(B, P_tr+P_te)``.
    """
    device = torch.device(device)
    X = _randn((B, P_tr + P_te, d), 2 * seed, device=device, dtype=dtype)
    betas = _randn((B, d), 2 * seed + 1, device=device, dtype=dtype)
    y = torch.einsum("bpd,bd->bp", X, betas) / math.sqrt(d)
    return X, y


def train_model_isotropic(
    d: int,
    P_tr: int,
    P_test: int,
    B: int,
    N: int,
    L: int,
    beta: float,
    gamma: float,
    T: int,
    lr: float,
    lamb: float,
    *,
    online: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[list[float], list[list[float]]]:
    """Train the isotropic linear attention model via SGD.

    Trains the attention weights (W_x, Wq, Wk, Wv) while keeping the
    label embedding W_y frozen. Uses isotropic Gaussian data at each step
    (online mode) or a fixed dataset (offline mode).

    The training loss is:

        loss = mean((out[:, P_tr:] / gamma + y[:, P_tr:])^2)

    where gamma is a scaling factor applied to the model output. The sign
    convention (out/gamma + y) means the model output is expected to be
    approximately -gamma * y at test positions.

    L2 regularization is applied to all trainable parameters:

        reg_loss = loss + lamb * sum(||p||^2 for p in trainable)

    Weight norms are tracked at each step for monitoring the effective
    gamma = wx^2 * wq * wk * (Wy^T Wv Wy).

    Args:
        d: Input dimension.
        P_tr: Number of training context tokens.
        P_test: Number of test tokens.
        B: Batch size.
        N: Hidden dimension (width) of the attention model.
        L: Number of attention layers (depth).
        beta: Residual connection scaling factor.
        gamma: Output scaling factor (model output is divided by gamma).
        T: Number of training steps.
        lr: Learning rate for SGD.
        lamb: L2 regularization coefficient.
        online: If True, sample fresh data each step. If False, reuse the
            same data (seed=1) every step.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        A tuple ``(pretrain_loss, weight_norms)`` where:
            - ``pretrain_loss``: List of T floats, training loss at each step.
            - ``weight_norms``: List of T sublists, each containing 4 floats:
              [mean(diag(W_x)), mean(diag(Wq)), mean(diag(Wk)), Wy^T Wv Wy].
    """
    device = torch.device(device)
    params = init_params_isotropic(d, N, device=device, dtype=dtype)
    W_x, Wy, Wq, Wk, Wv, w_out = params

    trainable = [
        torch.nn.Parameter(W_x.clone()),
        torch.nn.Parameter(Wq.clone()),
        torch.nn.Parameter(Wk.clone()),
        torch.nn.Parameter(Wv.clone()),
    ]

    optimizer = torch.optim.SGD(trainable, lr=lr)
    pretrain_loss: list[float] = []
    weight_norms: list[list[float]] = []

    for t in range(T):
        seed_t = t if online else 0

        X, y = sample_data_gauss_isotropic(d, B, P_tr, P_test, seed=seed_t + 1, device=device, dtype=dtype)

        optimizer.zero_grad(set_to_none=True)
        out, _, _ = model_eval_isotropic(
            trainable, Wy, X, y, L=L, P_test=P_test, beta=beta,
        )

        loss = torch.mean((out[:, P_tr:] / gamma + y[:, P_tr:]) ** 2)
        reg_loss = loss + lamb * sum(torch.sum(p * p) for p in trainable)
        reg_loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        pretrain_loss.append(loss_value)

        # Track weight norms
        with torch.no_grad():
            W_x_cur, Wq_cur, Wk_cur, Wv_cur = trainable
            wn = [
                float(torch.mean(torch.diag(W_x_cur)).cpu()),
                float(torch.mean(torch.diag(Wq_cur)).cpu()),
                float(torch.mean(torch.diag(Wk_cur)).cpu()),
                float(torch.dot(Wy, Wv_cur @ Wy).cpu()),
            ]
        weight_norms.append(wn)

        if t % 100 == 0:
            print(f"step {t} , loss = {loss_value:.8f}")

    return pretrain_loss, weight_norms


def reduced_theory_four_var_linear_att_isotropic(
    L: int,
    alpha: float,
    lamb_grid: Tensor,
    eta: float,
    T: int,
    sigma: float = 0.4,
) -> tuple[list[float], list[list[float]]]:
    """Simulate four-variable reduced theory for isotropic linear attention.

    Reduces the full-matrix training dynamics to 4 scalar variables
    ws = [wx, wq, wk, wv] whose product gamma = wx^2 * wq * wk * wv
    represents the effective attention strength. The loss is computed by
    integrating over the Marchenko-Pastur (MP) spectral density:

        loss = integral rho_MP(lambda) * (1 - gamma*lambda/L)^{2L} d(lambda)
               + (1-alpha)  [if alpha < 1]

    where rho_MP is the Marchenko-Pastur density with aspect ratio alpha = P/d:

        rho_MP(lambda) = alpha/(2*pi*lambda) * sqrt((lambda+ - lambda)(lambda - lambda-))

    and lambda+/- = (1 +/- 1/sqrt(alpha))^2.

    The (1-alpha) correction accounts for the point mass at lambda=0 when
    the context length is less than the dimension (alpha < 1).

    Gradient descent is performed on the 4 variables via autograd, capturing
    the nonlinear dynamics of the factored gamma representation.

    Initial values: wx = sqrt(2)*sigma, wq = wk = wv = sigma.

    Args:
        L: Number of attention layers (depth).
        alpha: Aspect ratio P/d controlling the Marchenko-Pastur distribution.
        lamb_grid: 1D tensor of eigenvalue grid points for numerical integration.
        eta: Learning rate for gradient descent.
        T: Number of GD steps.
        sigma: Initialization scale (default 0.4).

    Returns:
        A tuple ``(losses, all_ws)`` where:
            - ``losses``: List of T floats, the loss at each step.
            - ``all_ws``: List of T sublists, each containing 4 floats
              [wx, wq, wk, wv] at that step.
    """
    device = lamb_grid.device
    dtype = lamb_grid.dtype

    sqrt_alpha = math.sqrt(alpha)
    lamb_p = (1.0 + 1.0 / sqrt_alpha) ** 2
    lamb_m = (1.0 - 1.0 / sqrt_alpha) ** 2

    diff = (lamb_p - lamb_grid) * (lamb_grid - lamb_m)
    in_bulk = ((lamb_grid < lamb_p) & (lamb_grid > lamb_m)).to(dtype)
    bulk = alpha / (2.0 * math.pi * lamb_grid) * torch.sqrt(torch.clamp(diff, min=0.0)) * in_bulk

    dlamb = (lamb_grid[-1] - lamb_grid[0]) / len(lamb_grid)

    alpha_correction = (1.0 - alpha) if alpha < 1.0 else 0.0

    ws = [
        torch.tensor(math.sqrt(2.0) * sigma, device=device, dtype=dtype, requires_grad=True),
        torch.tensor(sigma, device=device, dtype=dtype, requires_grad=True),
        torch.tensor(sigma, device=device, dtype=dtype, requires_grad=True),
        torch.tensor(sigma, device=device, dtype=dtype, requires_grad=True),
    ]

    losses: list[float] = []
    all_ws: list[list[float]] = []

    for t in range(T):
        gamma_val = ws[0] ** 2 * ws[1] * ws[2] * ws[3]
        decay = (1.0 - (1.0 / L) * gamma_val * lamb_grid).pow(2 * L)
        loss_t = torch.dot(bulk * dlamb, decay) + alpha_correction

        losses.append(float(loss_t.detach().cpu()))
        all_ws.append([float(w.detach().cpu()) for w in ws])

        grads = torch.autograd.grad(loss_t, ws)
        with torch.no_grad():
            ws = [
                (w - eta * g).detach().requires_grad_(True)
                for w, g in zip(ws, grads)
            ]

        if t % 100 == 0:
            sys.stdout.write(
                f"\r t = {t},  loss = {losses[-1]:.6f} , gamma = {float(gamma_val.detach().cpu()):.6f}"
            )

    print()
    return losses, all_ws


# =========================================================================
# Dimension-free variants (spectral data, no sqrt(d) in update)
# =========================================================================


def init_params_dim_free(
    d: int,
    N: int,
    *,
    sigma: float = 0.4,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> list[Tensor]:
    """Initialize parameters for the dimension-free linear attention model.

    Creates diagonal weight matrices similar to the isotropic variant, but
    with a key difference in W_v scaling:

      - W_x = sqrt(2) * sigma * I_d    (d x d input projection)
      - W_y = ones(N) / sqrt(N)        (N-dim label embedding, normalized)
      - Wq  = sigma * I_N              (N x N query projection)
      - Wk  = sigma * I_N              (N x N key projection)
      - Wv  = sigma * sqrt(N) * I_N    (N x N value projection, sqrt(N) scaling)
      - w_out = W_y                    (output projection)

    The key difference from the isotropic variant is that W_v is scaled by
    sqrt(N) instead of sqrt(d). Combined with the dimension-free forward pass
    (which divides by P_tr instead of P_tr * sqrt(d)), this makes the model's
    effective gamma independent of the input dimension d.

    Args:
        d: Input dimension of the linear regression problem.
        N: Hidden dimension (width) of the attention model.
        sigma: Initialization scale for the weight matrices.
        device: Torch device for the parameters.
        dtype: Torch dtype for the parameters.

    Returns:
        List of 6 tensors: [W_x, W_y, Wq, Wk, Wv, w_out].
    """
    device = torch.device(device)
    W_x = math.sqrt(2.0) * sigma * torch.eye(d, device=device, dtype=dtype)
    W_y = torch.ones(N, device=device, dtype=dtype) / math.sqrt(N)
    Wq = sigma * torch.eye(N, device=device, dtype=dtype)
    Wk = Wq.clone()
    Wv = sigma * math.sqrt(N) * torch.eye(N, device=device, dtype=dtype)
    w_out = W_y.clone()
    return [W_x, W_y, Wq, Wk, Wv, w_out]


def model_eval_dim_free(
    params_tr: list[Tensor],
    Wy: Tensor,
    X: Tensor,
    y: Tensor,
    L: int = 100,
    P_test: int = 1,
    beta: float = 1.0,
) -> tuple[Tensor, list[float], list[float]]:
    """Run the dimension-free linear attention model forward for L layers.

    This is a "frozen embedding" model identical in structure to the isotropic
    variant, but with a different normalization in the attention update. The
    key difference is that the update is divided by P_tr only (no sqrt(d)
    factor), making the model's behavior dimension-free.

    The forward pass for each layer is:
      1. q = hx @ Wq^T,  k = hx @ Wk^T,  v = hy @ Wv^T
      2. A = k @ q^T  (linear attention scores)
      3. update = A_masked @ v
      4. hy <- hy - (beta / L) * update / P_tr

    This normalization (dividing by P_tr instead of P_tr * sqrt(d)) is
    appropriate when combined with W_v ~ sqrt(N) (instead of sqrt(d)),
    yielding dimension-independent dynamics.

    Args:
        params_tr: List of 4 trainable tensors [W_x, Wq, Wk, Wv] where
            W_x is (N, d) and Wq, Wk, Wv are (N, N).
        Wy: Frozen label embedding vector of shape ``(N,)``.
        X: Input features of shape ``(batch, seq_len, d)``.
        y: Labels of shape ``(batch, seq_len)``.
        L: Number of attention layers to apply.
        P_test: Number of test tokens at the end of each sequence.
        beta: Residual connection scaling factor.

    Returns:
        A tuple ``(out, train_losses, test_losses)`` where:
            - ``out``: Predictions of shape ``(batch, seq_len)``.
            - ``train_losses``: Empty list (not computed in this variant).
            - ``test_losses``: Empty list (not computed in this variant).
    """
    W_x, Wq, Wk, Wv = params_tr
    device = X.device
    dtype = X.dtype

    N, d = W_x.shape
    seq_len = X.shape[1]
    p_tr = seq_len - P_test
    batch = X.shape[0]

    hx = torch.einsum("bpd,nd->bpn", X, W_x)
    mask_y = _build_y_mask(batch, seq_len, p_tr, device=device, dtype=dtype)
    hy = torch.einsum("bp,n->bpn", y * mask_y, Wy)
    mask = _build_attn_mask(seq_len, p_tr, device=device, dtype=dtype)

    for _ in range(L):
        q = torch.einsum("bpn,kn->bpk", hx, Wq)
        k = torch.einsum("bpn,kn->bpk", hx, Wk)
        v = torch.einsum("bpn,kn->bpk", hy, Wv)
        A = torch.einsum("bpk,bqk->bpq", k, q)

        masked_A = A * mask
        update = torch.bmm(masked_A, v)
        hy = hy - (beta / L) * update / float(p_tr)

    out = torch.einsum("bpn,n->bp", hy, Wy)
    return out, [], []


def sample_data_spec_rotate_bernoulli(
    spec: Tensor,
    w_star: Tensor,
    B: int,
    P_tr: int,
    P_te: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Sample spectral data with QR rotation and Bernoulli-signed targets.

    Generates a batch of ICL tasks with power-law covariance structure rotated
    by a random orthogonal matrix, and regression targets with Rademacher
    (+/-1) sign flips on w_star:

      1. Sample B random orthogonal matrices O via QR of Gaussian matrices
         (seed 3*seed).
      2. Draw X ~ N(0, I_d) and scale by sqrt(spec) to create spectral features
         (seed 3*seed+1).
      3. Rotate: OX = O^T @ X (mix eigendirections).
      4. Draw Bernoulli signs b ~ {-1, +1}^d (seed 3*seed+2).
      5. Compute labels: y = X @ (w_star * b) (in the un-rotated spectral basis).

    Note: The labels y are computed using the un-rotated X (not OX), while the
    features returned are the rotated OX. This is intentional -- the model must
    learn to undo the rotation.

    Args:
        spec: Eigenvalues of the input covariance. Shape ``(d,)``.
        w_star: Target weight norms per eigendirection. Shape ``(d,)``.
        B: Batch size.
        P_tr: Number of training context tokens.
        P_te: Number of test tokens.
        seed: Base random seed.
        device: Torch device for outputs.
        dtype: Torch dtype for outputs.

    Returns:
        A tuple ``(OX, y)`` where:
            - ``OX``: Rotated input features of shape ``(B, P_tr+P_te, d)``.
            - ``y``: Labels of shape ``(B, P_tr+P_te)``.
    """
    device = torch.device(device)
    spec = spec.to(device=device, dtype=dtype)
    w_star = w_star.to(device=device, dtype=dtype)
    d = spec.shape[0]

    O = torch.linalg.qr(
        _randn((B, d, d), 3 * seed, device=device, dtype=dtype)
    ).Q

    X = _randn((B, P_tr + P_te, d), 3 * seed + 1, device=device, dtype=dtype)
    X = X * spec.sqrt().unsqueeze(0).unsqueeze(0)
    OX = torch.einsum("bdc,bpc->bpd", O, X)  # O^T @ X

    gen_bern = torch.Generator(device=device)
    gen_bern.manual_seed(3 * seed + 2)
    bernoulli = torch.empty(B, d, device=device, dtype=dtype).bernoulli_(generator=gen_bern)
    bernoulli = 2.0 * bernoulli - 1.0
    w_sign = w_star.unsqueeze(0) * bernoulli

    y = torch.einsum("bpd,bd->bp", X, w_sign)
    return OX, y


def train_model_dim_free(
    d: int,
    P_tr: int,
    P_test: int,
    B: int,
    N: int,
    L: int,
    beta: float,
    gamma: float,
    T: int,
    lr: float,
    lamb: float,
    spec: Tensor,
    w_star: Tensor,
    *,
    sigma: float = 0.45,
    online: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[list[float], list[list[float]]]:
    """Train the dimension-free linear attention model via SGD on spectral data.

    Trains the attention weights (W_x, Wq, Wk, Wv) while keeping W_y frozen.
    Uses spectral (power-law covariance) data with QR rotation and Bernoulli
    sign flips via ``sample_data_spec_rotate_bernoulli``.

    The training loss is:

        loss = mean((out[:, P_tr:] / gamma + y[:, P_tr:])^2)

    L2 regularization is applied to all trainable parameters.

    Weight norms are tracked at each step for monitoring the effective
    gamma = wx^2 * wq * wk * (Wy^T Wv Wy).

    Args:
        d: Input dimension.
        P_tr: Number of training context tokens.
        P_test: Number of test tokens.
        B: Batch size.
        N: Hidden dimension (width) of the attention model.
        L: Number of attention layers (depth).
        beta: Residual connection scaling factor.
        gamma: Output scaling factor (model output is divided by gamma).
        T: Number of training steps.
        lr: Learning rate for SGD.
        lamb: L2 regularization coefficient.
        spec: Eigenvalues of the input covariance. Shape ``(d,)``.
        w_star: Target weight norms per eigendirection. Shape ``(d,)``.
        sigma: Initialization scale for weight matrices (default 0.45).
        online: If True, sample fresh data each step. If False, reuse data.
        device: Torch device for computation.
        dtype: Torch dtype for computation.

    Returns:
        A tuple ``(pretrain_loss, weight_norms)`` where:
            - ``pretrain_loss``: List of T floats, training loss at each step.
            - ``weight_norms``: List of T sublists, each containing 4 floats:
              [mean(diag(W_x)), mean(diag(Wq)), mean(diag(Wk)), Wy^T Wv Wy].
    """
    device = torch.device(device)
    spec = spec.to(device=device, dtype=dtype)
    w_star = w_star.to(device=device, dtype=dtype)

    params = init_params_dim_free(d, N, sigma=sigma, device=device, dtype=dtype)
    W_x, Wy, Wq, Wk, Wv, w_out = params

    trainable = [
        torch.nn.Parameter(W_x.clone()),
        torch.nn.Parameter(Wq.clone()),
        torch.nn.Parameter(Wk.clone()),
        torch.nn.Parameter(Wv.clone()),
    ]

    optimizer = torch.optim.SGD(trainable, lr=lr)
    pretrain_loss: list[float] = []
    weight_norms: list[list[float]] = []

    for t in range(T):
        seed_t = t if online else 0

        X, y = sample_data_spec_rotate_bernoulli(
            spec, w_star, B, P_tr, P_test, seed=seed_t + 1, device=device, dtype=dtype
        )

        optimizer.zero_grad(set_to_none=True)
        out, _, _ = model_eval_dim_free(
            trainable, Wy, X, y, L=L, P_test=P_test, beta=beta,
        )

        loss = torch.mean((out[:, P_tr:] / gamma + y[:, P_tr:]) ** 2)
        reg_loss = loss + lamb * sum(torch.sum(p * p) for p in trainable)
        reg_loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        pretrain_loss.append(loss_value)

        with torch.no_grad():
            W_x_cur, Wq_cur, Wk_cur, Wv_cur = trainable
            wn = [
                float(torch.mean(torch.diag(W_x_cur)).cpu()),
                float(torch.mean(torch.diag(Wq_cur)).cpu()),
                float(torch.mean(torch.diag(Wk_cur)).cpu()),
                float(torch.dot(Wy, Wv_cur @ Wy).cpu()),
            ]
        weight_norms.append(wn)

        if t % 100 == 0:
            print(f"step {t} , loss = {loss_value:.8f}")

    return pretrain_loss, weight_norms


def reduced_theory_four_var_linear_att_spec(
    spec: Tensor,
    w_star: Tensor,
    L: int,
    eta: float,
    T: int,
    sigma: float = 0.4,
) -> tuple[list[float], list[list[float]]]:
    """Simulate four-variable reduced theory for spectral linear attention.

    Similar to ``reduced_theory_four_var_linear_att_isotropic`` but uses the
    actual discrete spectrum and target weights instead of integrating over
    the Marchenko-Pastur density. The loss is:

        loss = sum_k spec_k * (1 - gamma * spec_k / L)^{2L} * w_star_k^2

    where gamma = wx^2 * wq * wk * wv.

    This is appropriate for the dimension-free model variant where the data
    has a known power-law spectrum, and the loss is a discrete sum rather
    than a continuous integral.

    Initial values: wx = sqrt(2)*sigma, wq = wk = wv = sigma.

    Args:
        spec: Eigenvalues of the input covariance. Shape ``(d,)``.
        w_star: Target weight norms per eigendirection. Shape ``(d,)``.
        L: Number of attention layers (depth).
        eta: Learning rate for gradient descent.
        T: Number of GD steps.
        sigma: Initialization scale (default 0.4).

    Returns:
        A tuple ``(losses, all_ws)`` where:
            - ``losses``: List of T floats, the loss at each step.
            - ``all_ws``: List of T sublists, each containing 4 floats
              [wx, wq, wk, wv] at that step.
    """
    device = spec.device
    dtype = spec.dtype

    ws = [
        torch.tensor(math.sqrt(2.0) * sigma, device=device, dtype=dtype, requires_grad=True),
        torch.tensor(sigma, device=device, dtype=dtype, requires_grad=True),
        torch.tensor(sigma, device=device, dtype=dtype, requires_grad=True),
        torch.tensor(sigma, device=device, dtype=dtype, requires_grad=True),
    ]

    losses: list[float] = []
    all_ws: list[list[float]] = []

    for t in range(T):
        gamma_val = ws[0] ** 2 * ws[1] * ws[2] * ws[3]
        decay = (1.0 - (1.0 / L) * gamma_val * spec).pow(2 * L)
        loss_t = torch.sum(spec * decay * w_star**2)

        losses.append(float(loss_t.detach().cpu()))
        all_ws.append([float(w.detach().cpu()) for w in ws])

        grads = torch.autograd.grad(loss_t, ws)
        with torch.no_grad():
            ws = [
                (w - eta * g).detach().requires_grad_(True)
                for w, g in zip(ws, grads)
            ]

        if t % 100 == 0:
            sys.stdout.write(
                f"\r t = {t},  loss = {losses[-1]:.6f} , gamma = {float(gamma_val.detach().cpu()):.6f}"
            )

    print()
    return losses, all_ws
