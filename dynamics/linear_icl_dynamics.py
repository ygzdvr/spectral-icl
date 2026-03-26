"""Hand-coded attention models that analytically solve in-context linear regression.

This module provides reference implementations of attention-based models whose
weight matrices are constructed analytically (rather than learned) to solve
linear regression in-context.  The core idea is to encode the model state as

    h(t) = [x, Delta(t)]^T   in  R^{d+1}

where *x* is the input feature vector and *Delta(t)* tracks the residual
prediction error after *t* attention layers.  The attention update rule
iteratively reduces the residual so that after sufficiently many layers the
model approximates ridge regression.

**Variants provided:**

* *Coupled* (``model_eval``) -- A single hidden state ``h`` mixes the x- and
  y-embeddings.  Parameters are ``[W_x, W_y, Wq, Wk, Wv, w_out]``.
* *Decoupled* (``model_eval_decoupled``) -- Separate x-embedding ``hx`` and
  y-embedding ``hy``; queries/keys come from ``hx``, values from ``hy``.
  Parameters are ``[W_x, W_y, Wq, Wk, Wv, w_out]`` (``w_out`` unused).
* *Decoupled frozen-embedding* (``model_eval_decoupled_frozen_emb``) --
  Like decoupled, but ``W_y`` and ``w_out`` are frozen; only
  ``[W_x, Wq, Wk, Wv]`` are trainable.
* *Decoupled frozen-embedding with softmax* (``model_eval_decoupled_softmax_frozen_emb``)
  -- Same as frozen-embedding but applies ``exp(A)`` instead of raw
  dot-product attention scores.

The module also contains helpers for sampling synthetic linear-regression tasks
(isotropic and power-law covariance) and convenience runners that combine data
generation with model evaluation.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from configs.eval_configs import HandCodedEvalConfig, HardPowerLawDepthConfig


Tensor = torch.Tensor


def _as_tensor(x: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Ensure *x* is a torch Tensor on the given device and dtype.

    If *x* is already a ``torch.Tensor`` it is moved in-place via ``.to()``;
    otherwise ``torch.as_tensor`` creates a new tensor.

    Args:
        x: Input value -- either a ``torch.Tensor`` or an array-like that
            ``torch.as_tensor`` can convert.
        device: Target device.
        dtype: Target floating-point dtype.

    Returns:
        A ``torch.Tensor`` with the requested device and dtype.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _build_y_mask(batch: int, seq_len: int, p_tr: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build a label mask that zeros out test-set labels.

    During in-context evaluation the model must not see the labels of the test
    examples.  This function creates a ``(batch, seq_len)`` mask that is 1 for
    the first ``p_tr`` (training) positions and 0 for the remaining test
    positions, so that multiplying ``y * mask_y`` hides the test labels.

    Args:
        batch: Batch dimension size.
        seq_len: Total sequence length (training + test).
        p_tr: Number of training positions (the first ``p_tr`` entries are 1).
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape ``(batch, seq_len)`` with ones at training positions
        and zeros at test positions.
    """
    mask = torch.ones((batch, seq_len), device=device, dtype=dtype)
    mask[:, p_tr:] = 0.0
    return mask


def _build_attn_mask(seq_len: int, p_tr: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build an attention mask that prevents attending to test positions.

    Returns a ``(seq_len, seq_len)`` matrix where column *j* is 1 if position
    *j* is a training example (j < p_tr) and 0 otherwise.  When multiplied
    element-wise with the attention scores, this prevents any token from
    attending to test-set positions, ensuring that predictions for test tokens
    are based solely on training examples.

    Args:
        seq_len: Total sequence length (training + test).
        p_tr: Number of training positions.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape ``(seq_len, seq_len)`` with ones in columns
        ``0..p_tr-1`` and zeros in columns ``p_tr..seq_len-1``.
    """
    mask = torch.ones((seq_len, seq_len), device=device, dtype=dtype)
    mask[:, p_tr:] = 0.0
    return mask


def _randn(shape: tuple[int, ...], seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Sample i.i.d. standard normal entries with a deterministic seed.

    Creates a fresh ``torch.Generator`` seeded with *seed* so that the draw is
    reproducible regardless of the global RNG state.

    Args:
        shape: Shape of the output tensor.
        seed: Manual seed for the local generator.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape *shape* with entries drawn from N(0, 1).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


def init_hand_coded_params(d: int, *, device: torch.device | str = "cuda", dtype: torch.dtype = torch.float32) -> list[Tensor]:
    """Create the analytically-designed weight matrices for the coupled model.

    Constructs parameters ``[W_x, W_y, Wq, Wk, Wv, w_out]`` such that, with
    the right step-size ``beta``, the model implements iterative least-squares
    regression via attention.  The hidden dimension is ``N = d + 1`` and the
    state encodes::

        h(t) = [x, Delta(t)]^T   in  R^{d+1}

    where ``Delta(t)`` is the residual prediction error at layer *t*.

    Weight structure:

    * ``W_x`` -- shape ``(d+1, d)``.  Embeds input *x* into the first *d*
      coordinates of *h*, scaled by ``sqrt(d)``.
    * ``W_y`` -- shape ``(d+1,)``.  Places the label *y* into the last
      coordinate of *h*.
    * ``Wq``, ``Wk`` -- shape ``(d+1, d+1)``.  Diagonal matrices with
      ``sqrt(d) * I_d`` in the top-left block and zero in the (d+1)-th
      coordinate, so queries and keys operate on the *x*-subspace.
    * ``Wv`` -- shape ``(d+1, d+1)``.  Picks out the residual coordinate:
      ``Wv[-1, -1] = sqrt(d)``, rest zero.
    * ``w_out`` -- shape ``(d+1,)``.  Reads out the residual coordinate:
      ``w_out[-1] = sqrt(d)``.

    Args:
        d: Feature dimensionality of the regression problem.
        device: Target device (string or ``torch.device``).
        dtype: Target floating-point dtype.

    Returns:
        A list ``[W_x, W_y, Wq, Wk, Wv, w_out]`` of tensors on the
        requested device and dtype.
    """
    device = torch.device(device)
    sqrt_d = math.sqrt(d)

    # encode h(t) = [x, Delta(t)]^T in R^{d+1}
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


def sample_linear_task(
    B: int,
    P: int,
    d: int,
    *,
    seed_x: int = 0,
    seed_beta: int = 1,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of isotropic linear regression tasks.

    For each task in the batch, draws:

    * Input features  X ~ N(0, I_d),  shape ``(B, P, d)``
    * True weights    beta ~ N(0, I_d),  shape ``(B, d)``
    * Labels          y = X @ beta / sqrt(d),  shape ``(B, P)``

    The ``1/sqrt(d)`` normalisation keeps the label variance O(1) as *d*
    grows.

    Args:
        B: Batch size (number of independent tasks).
        P: Sequence length (total number of examples per task).
        d: Feature dimensionality.
        seed_x: Deterministic seed for sampling *X*.
        seed_beta: Deterministic seed for sampling *beta*.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A tuple ``(X, y)`` where ``X`` has shape ``(B, P, d)`` and ``y`` has
        shape ``(B, P)``.
    """
    device = torch.device(device)
    X = _randn((B, P, d), seed_x, device=device, dtype=dtype)
    betas = _randn((B, d), seed_beta, device=device, dtype=dtype)
    y = torch.einsum("bpd,bd->bp", X, betas) / math.sqrt(d)
    return X, y


def model_eval(
    params: Sequence[Tensor],
    X: Tensor,
    y: Tensor,
    L: int = 100,
    P_test: int = 1,
    beta: float = 100.0,
    qk_ln: bool = False,
    norm_inputs: bool = False,
    divide_update_by_sqrt_d: bool = True,
) -> tuple[Tensor, list[float], list[float]]:
    """Evaluate the *coupled* hand-coded attention model on a linear ICL task.

    Runs ``L`` layers of the coupled attention update on the input batch.
    The state ``h`` in R^{d+1} is initialised by embedding inputs via ``W_x``
    and labels via ``W_y``, then iteratively updated:

    .. math::

        q = h W_q^T / sqrt(N),  \\quad k = h W_k^T / sqrt(N),  \\quad v = h W_v^T / sqrt(N)

        A = k q^T / N

        h \\leftarrow h - (\\beta / L) \\cdot (A \\odot \\text{mask}) \\, v \\;/\\; \\text{denom}

    where ``mask`` zeros out attention to test positions and ``denom`` equals
    ``P_{tr}`` (optionally multiplied by ``sqrt(d)``).

    The final prediction is read out as ``out = h @ w_out / N``.

    Args:
        params: List of six tensors ``[W_x, W_y, Wq, Wk, Wv, w_out]``.

            * ``W_x`` -- shape ``(N, d)``, input embedding matrix.
            * ``W_y`` -- shape ``(N,)``, label embedding vector.
            * ``Wq``  -- shape ``(N, N)``, query projection.
            * ``Wk``  -- shape ``(N, N)``, key projection.
            * ``Wv``  -- shape ``(N, N)``, value projection.
            * ``w_out`` -- shape ``(N,)``, output readout vector.

        X: Input features, shape ``(batch, seq_len, d)``.
        y: Labels, shape ``(batch, seq_len)``.  Test-position labels are masked
            internally so their values do not affect the forward pass.
        L: Number of attention layers (depth).
        P_test: Number of test examples at the end of each sequence.
        beta: Global step-size scaling factor.
        qk_ln: If ``True``, apply query-key LayerNorm (zero-mean, unit
            variance normalisation) to queries and keys before computing
            attention scores.
        norm_inputs: If ``True``, divide the initial hidden state ``h`` by
            ``sqrt(d)`` after the ``W_x`` embedding.
        divide_update_by_sqrt_d: If ``True`` (default), the attention update
            denominator is ``P_tr * sqrt(d)``; if ``False``, it is just
            ``P_tr``.

    Returns:
        A tuple ``(out, train_losses, test_losses)`` where:

        * ``out`` -- Tensor of shape ``(batch, seq_len)`` with the final
          model predictions.
        * ``train_losses`` -- List of length ``L`` with the mean squared
          residual on training positions at the *start* of each layer.
        * ``test_losses`` -- List of length ``L`` with the mean squared
          prediction error on test positions at the *start* of each layer.
    """
    W_x, W_y, Wq, Wk, Wv, w_out = params
    device = X.device
    dtype = X.dtype

    W_x = _as_tensor(W_x, device=device, dtype=dtype)
    W_y = _as_tensor(W_y, device=device, dtype=dtype)
    Wq = _as_tensor(Wq, device=device, dtype=dtype)
    Wk = _as_tensor(Wk, device=device, dtype=dtype)
    Wv = _as_tensor(Wv, device=device, dtype=dtype)
    w_out = _as_tensor(w_out, device=device, dtype=dtype)

    N, d = W_x.shape
    seq_len = X.shape[1]
    p_tr = seq_len - P_test
    batch = X.shape[0]

    h = torch.einsum("bpd,nd->bpn", X, W_x)
    if norm_inputs:
        h = h / math.sqrt(d)

    mask_y = _build_y_mask(batch, seq_len, p_tr, device=device, dtype=dtype)
    h = h + torch.einsum("bp,n->bpn", y * mask_y, W_y)

    mask = _build_attn_mask(seq_len, p_tr, device=device, dtype=dtype)

    train_losses: list[float] = []
    test_losses: list[float] = []
    eps = torch.finfo(dtype).eps

    for _ in range(L):
        train_losses.append(float((h[:, :p_tr, -1] ** 2).mean().detach().cpu()))
        test_losses.append(float(((h[:, p_tr:, -1] + y[:, p_tr:]) ** 2).mean().detach().cpu()))

        q = torch.einsum("bpn,kn->bpk", h, Wq) / math.sqrt(N)
        k = torch.einsum("bpn,kn->bpk", h, Wk) / math.sqrt(N)

        if qk_ln:
            q = q - q.mean(dim=-1, keepdim=True)
            k = k - k.mean(dim=-1, keepdim=True)
            q = q / torch.sqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
            k = k / torch.sqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)

        v = torch.einsum("bpn,kn->bpk", h, Wv) / math.sqrt(N)
        A = torch.einsum("bpk,bqk->bpq", k, q) / float(N)

        masked_A = A * mask
        # Notebook contraction order:
        # update[b, l, n] = sum_j masked_A[b, l, j] * v[b, j, n]
        update = torch.bmm(masked_A, v)
        denom = float(p_tr)
        if divide_update_by_sqrt_d:
            denom = denom * math.sqrt(float(d))
        h = h - (beta / L) * update / denom

    out = torch.einsum("bpn,n->bp", h, w_out) / float(N)
    return out, train_losses, test_losses


def model_eval_decoupled(
    params: Sequence[Tensor],
    X: Tensor,
    y: Tensor,
    L: int = 100,
    P_test: int = 1,
    beta: float = 100.0,
    qk_ln: bool = False,
    norm_inputs: bool = False,
) -> tuple[Tensor, list[float], list[float]]:
    """Evaluate the *decoupled* hand-coded attention model.

    Unlike the coupled variant (:func:`model_eval`), this formulation maintains
    two separate embedding streams:

    * ``hx = X @ W_x^T`` -- the *x*-embedding, used to form queries and keys.
      ``hx`` is **never updated** during the forward pass.
    * ``hy = (y * mask_y) . W_y`` -- the *y*-embedding, used to form values.
      ``hy`` is updated at each layer.

    At every layer the attention update modifies only ``hy``:

    .. math::

        q = hx \\, W_q^T / \\sqrt{N}, \\quad k = hx \\, W_k^T / \\sqrt{N},
        \\quad v = hy \\, W_v^T / \\sqrt{N}

        A = k \\, q^T / N

        hy \\leftarrow hy - (\\beta / L) \\cdot (A \\odot \\text{mask}) \\, v \\;/\\; P_{tr}

    The final output is ``out = hy @ W_y / N``.

    Args:
        params: List of six tensors ``[W_x, W_y, Wq, Wk, Wv, w_out]``.
            ``w_out`` is accepted for interface compatibility but is unused.

            * ``W_x`` -- shape ``(N, d)``, input embedding matrix.
            * ``W_y`` -- shape ``(N,)``, label embedding / readout vector.
            * ``Wq``  -- shape ``(N, N)``, query projection.
            * ``Wk``  -- shape ``(N, N)``, key projection.
            * ``Wv``  -- shape ``(N, N)``, value projection.
            * ``w_out`` -- unused (kept for signature compatibility).

        X: Input features, shape ``(batch, seq_len, d)``.
        y: Labels, shape ``(batch, seq_len)``.
        L: Number of attention layers (depth).
        P_test: Number of test examples at the end of each sequence.
        beta: Step-size scaling factor.
        qk_ln: Ignored in this variant (accepted for interface compatibility).
        norm_inputs: Ignored in this variant (accepted for interface
            compatibility).

    Returns:
        A tuple ``(out, train_losses, test_losses)`` where:

        * ``out`` -- Tensor of shape ``(batch, seq_len)`` with final
          predictions.
        * ``train_losses`` -- Empty list (losses are not tracked in the
          decoupled variant).
        * ``test_losses`` -- Empty list.
    """
    del qk_ln, norm_inputs

    W_x, W_y, Wq, Wk, Wv, _w_out = params
    device = X.device
    dtype = X.dtype

    W_x = _as_tensor(W_x, device=device, dtype=dtype)
    W_y = _as_tensor(W_y, device=device, dtype=dtype)
    Wq = _as_tensor(Wq, device=device, dtype=dtype)
    Wk = _as_tensor(Wk, device=device, dtype=dtype)
    Wv = _as_tensor(Wv, device=device, dtype=dtype)

    N, d = W_x.shape
    seq_len = X.shape[1]
    p_tr = seq_len - P_test
    batch = X.shape[0]

    hx = torch.einsum("bpd,nd->bpn", X, W_x)
    mask_y = _build_y_mask(batch, seq_len, p_tr, device=device, dtype=dtype)
    hy = torch.einsum("bp,n->bpn", y * mask_y, W_y)
    mask = _build_attn_mask(seq_len, p_tr, device=device, dtype=dtype)

    for _ in range(L):
        q = torch.einsum("bpn,kn->bpk", hx, Wq) / math.sqrt(N)
        k = torch.einsum("bpn,kn->bpk", hx, Wk) / math.sqrt(N)
        v = torch.einsum("bpn,kn->bpk", hy, Wv) / math.sqrt(N)
        A = torch.einsum("bpk,bqk->bpq", k, q) / float(N)

        masked_A = A * mask
        update = torch.bmm(masked_A, v)
        hy = hy - (beta / L) * update / float(p_tr)

    out = torch.einsum("bpn,n->bp", hy, W_y) / float(N)
    return out, [], []


def model_eval_decoupled_frozen_emb(
    params_tr: Sequence[Tensor],
    Wy: Tensor,
    X: Tensor,
    y: Tensor,
    L: int = 100,
    P_test: int = 1,
    beta: float = 100.0,
    qk_ln: bool = False,
    norm_inputs: bool = False,
) -> tuple[Tensor, list[float], list[float]]:
    """Evaluate the decoupled model with frozen label-embedding weights.

    This is a variant of :func:`model_eval_decoupled` designed for training
    experiments where only the attention parameters and the input embedding are
    learnable, while the label embedding ``W_y`` and the output readout
    ``w_out`` are held fixed (frozen).  The frozen ``Wy`` is passed as a
    separate argument rather than inside ``params_tr``.

    The forward pass is identical to the decoupled model:

    * ``hx = X @ W_x^T`` (from trainable ``W_x``).
    * ``hy = (y * mask_y) . Wy`` (using frozen ``Wy``).
    * Iterative attention update modifies ``hy`` using trainable
      ``Wq, Wk, Wv``.
    * Output: ``out = hy @ Wy / N``.

    Args:
        params_tr: List of four *trainable* tensors ``[W_x, Wq, Wk, Wv]``.

            * ``W_x`` -- shape ``(N, d)``, trainable input embedding.
            * ``Wq``  -- shape ``(N, N)``, trainable query projection.
            * ``Wk``  -- shape ``(N, N)``, trainable key projection.
            * ``Wv``  -- shape ``(N, N)``, trainable value projection.

        Wy: Frozen label embedding / readout vector, shape ``(N,)``.  Used
            both to embed *y* into the hidden state and to read out the final
            prediction.
        X: Input features, shape ``(batch, seq_len, d)``.
        y: Labels, shape ``(batch, seq_len)``.
        L: Number of attention layers (depth).
        P_test: Number of test examples at the end of each sequence.
        beta: Step-size scaling factor.
        qk_ln: Ignored (accepted for interface compatibility).
        norm_inputs: Ignored (accepted for interface compatibility).

    Returns:
        A tuple ``(out, train_losses, test_losses)`` where:

        * ``out`` -- Tensor of shape ``(batch, seq_len)`` with final
          predictions.
        * ``train_losses`` -- Empty list (losses are not tracked).
        * ``test_losses`` -- Empty list.
    """
    del qk_ln, norm_inputs

    W_x, Wq, Wk, Wv = params_tr
    device = X.device
    dtype = X.dtype

    W_x = _as_tensor(W_x, device=device, dtype=dtype)
    Wy = _as_tensor(Wy, device=device, dtype=dtype)
    Wq = _as_tensor(Wq, device=device, dtype=dtype)
    Wk = _as_tensor(Wk, device=device, dtype=dtype)
    Wv = _as_tensor(Wv, device=device, dtype=dtype)

    N, d = W_x.shape
    seq_len = X.shape[1]
    p_tr = seq_len - P_test
    batch = X.shape[0]

    hx = torch.einsum("bpd,nd->bpn", X, W_x)
    mask_y = _build_y_mask(batch, seq_len, p_tr, device=device, dtype=dtype)
    hy = torch.einsum("bp,n->bpn", y * mask_y, Wy)
    mask = _build_attn_mask(seq_len, p_tr, device=device, dtype=dtype)

    for _ in range(L):
        q = torch.einsum("bpn,kn->bpk", hx, Wq) / math.sqrt(N)
        k = torch.einsum("bpn,kn->bpk", hx, Wk) / math.sqrt(N)
        v = torch.einsum("bpn,kn->bpk", hy, Wv) / math.sqrt(N)
        A = torch.einsum("bpk,bqk->bpq", k, q) / float(N)

        masked_A = A * mask
        update = torch.bmm(masked_A, v)
        hy = hy - (beta / L) * update / float(p_tr)

    out = torch.einsum("bpn,n->bp", hy, Wy) / float(N)
    return out, [], []


def model_eval_decoupled_softmax_frozen_emb(
    params_tr: Sequence[Tensor],
    Wy: Tensor,
    X: Tensor,
    y: Tensor,
    L: int = 100,
    P_test: int = 1,
    beta: float = 100.0,
    qk_ln: bool = False,
    norm_inputs: bool = False,
) -> tuple[Tensor, list[float], list[float]]:
    """Evaluate the decoupled frozen-embedding model with softmax attention.

    Identical to :func:`model_eval_decoupled_frozen_emb` except that the
    raw dot-product attention scores are passed through ``exp()`` before
    masking, approximating softmax attention in the regime where the scores
    are moderate:

    .. math::

        A_{\\text{softmax}} = \\exp\\bigl(k \\, q^T / N\\bigr)

    This makes the attention weights strictly positive (rather than allowing
    negative values), which changes the optimisation landscape and the
    model's implicit bias.

    Args:
        params_tr: List of four *trainable* tensors ``[W_x, Wq, Wk, Wv]``.

            * ``W_x`` -- shape ``(N, d)``, trainable input embedding.
            * ``Wq``  -- shape ``(N, N)``, trainable query projection.
            * ``Wk``  -- shape ``(N, N)``, trainable key projection.
            * ``Wv``  -- shape ``(N, N)``, trainable value projection.

        Wy: Frozen label embedding / readout vector, shape ``(N,)``.
        X: Input features, shape ``(batch, seq_len, d)``.
        y: Labels, shape ``(batch, seq_len)``.
        L: Number of attention layers (depth).
        P_test: Number of test examples at the end of each sequence.
        beta: Step-size scaling factor.
        qk_ln: Ignored (accepted for interface compatibility).
        norm_inputs: Ignored (accepted for interface compatibility).

    Returns:
        A tuple ``(out, train_losses, test_losses)`` where:

        * ``out`` -- Tensor of shape ``(batch, seq_len)`` with final
          predictions.
        * ``train_losses`` -- Empty list.
        * ``test_losses`` -- Empty list.
    """
    del qk_ln, norm_inputs

    W_x, Wq, Wk, Wv = params_tr
    device = X.device
    dtype = X.dtype

    W_x = _as_tensor(W_x, device=device, dtype=dtype)
    Wy = _as_tensor(Wy, device=device, dtype=dtype)
    Wq = _as_tensor(Wq, device=device, dtype=dtype)
    Wk = _as_tensor(Wk, device=device, dtype=dtype)
    Wv = _as_tensor(Wv, device=device, dtype=dtype)

    N, d = W_x.shape
    seq_len = X.shape[1]
    p_tr = seq_len - P_test
    batch = X.shape[0]

    hx = torch.einsum("bpd,nd->bpn", X, W_x)
    mask_y = _build_y_mask(batch, seq_len, p_tr, device=device, dtype=dtype)
    hy = torch.einsum("bp,n->bpn", y * mask_y, Wy)
    mask = _build_attn_mask(seq_len, p_tr, device=device, dtype=dtype)

    for _ in range(L):
        q = torch.einsum("bpn,kn->bpk", hx, Wq) / math.sqrt(N)
        k = torch.einsum("bpn,kn->bpk", hx, Wk) / math.sqrt(N)
        v = torch.einsum("bpn,kn->bpk", hy, Wv) / math.sqrt(N)
        A = torch.einsum("bpk,bqk->bpq", k, q) / float(N)

        A = torch.exp(A)

        masked_A = A * mask
        update = torch.bmm(masked_A, v)
        hy = hy - (beta / L) * update / float(p_tr)

    out = torch.einsum("bpn,n->bp", hy, Wy) / float(N)
    return out, [], []


def run_hand_coded_eval(
    cfg: HandCodedEvalConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, list[float], list[float], Tensor, Tensor]:
    """End-to-end evaluation of the hand-coded model on an isotropic task.

    Convenience wrapper that:

    1. Builds the analytical weight matrices via :func:`init_hand_coded_params`.
    2. Samples an isotropic linear regression batch via
       :func:`sample_linear_task`.
    3. Runs the coupled model forward pass via :func:`model_eval`.

    Args:
        cfg: A :class:`HandCodedEvalConfig` specifying dimensions, batch size,
            depths, seeds, etc.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A tuple ``(out, train_losses, test_losses, X, y)`` where:

        * ``out`` -- Tensor of shape ``(B, P)`` with model predictions.
        * ``train_losses`` -- Per-layer mean squared residual on training
          positions (list of length ``L``).
        * ``test_losses`` -- Per-layer mean squared error on test positions
          (list of length ``L``).
        * ``X`` -- Input features, shape ``(B, P, d)``.
        * ``y`` -- Labels, shape ``(B, P)``.
    """
    params = init_hand_coded_params(cfg.d, device=device, dtype=dtype)
    X, y = sample_linear_task(
        B=cfg.B,
        P=cfg.P,
        d=cfg.d,
        seed_x=cfg.seed_x,
        seed_beta=cfg.seed_beta,
        device=device,
        dtype=dtype,
    )
    out, train_losses, test_losses = model_eval(
        params,
        X,
        y,
        L=cfg.L,
        P_test=cfg.P_test,
        beta=cfg.beta,
    )
    return out, train_losses, test_losses, X, y


def sample_hard_power_law_batch(
    cfg: HardPowerLawDepthConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a batch of linear regression tasks with power-law covariance.

    Each feature coordinate *k* (1-indexed) is scaled by ``k^{-alpha}`` where
    ``alpha = cfg.exp_value``.  This induces an anisotropic covariance
    ``Sigma`` whose eigenvalues decay as a power law:
    ``lambda_k ~ k^{-2*alpha}``.

    Concretely:

    * ``powers[b, k] = k^{-alpha}``  (same for all *b* in the batch).
    * ``X_raw ~ N(0, I)``, then ``X = X_raw * powers``.
    * ``y = X @ beta / sqrt(d)`` with ``beta ~ N(0, I)``.

    Args:
        cfg: A :class:`HardPowerLawDepthConfig` specifying the problem
            parameters (dimension, batch size, exponent, seeds, etc.).
        device: Target device.
        dtype: Target dtype.

    Returns:
        A tuple ``(X, y, powers)`` where:

        * ``X`` -- Input features with power-law covariance, shape
          ``(B, P + P_test, d)``.
        * ``y`` -- Labels, shape ``(B, P + P_test)``.
        * ``powers`` -- Per-coordinate scaling factors, shape ``(B, d)``.
    """
    device = torch.device(device)
    seq_len = cfg.P + cfg.P_test

    exps = torch.full((cfg.B,), cfg.exp_value, device=device, dtype=dtype)
    coords = torch.linspace(1, cfg.d, cfg.d, device=device, dtype=dtype)
    powers = coords.unsqueeze(0).pow(-exps.unsqueeze(1))

    X = _randn((cfg.B, seq_len, cfg.d), cfg.seed_x, device=device, dtype=dtype)
    X = X * powers.unsqueeze(1)

    betas = _randn((cfg.B, cfg.d), cfg.seed_beta, device=device, dtype=dtype)
    y = torch.einsum("bpd,bd->bp", X, betas) / math.sqrt(cfg.d)
    return X, y, powers


def run_hard_power_law_depth_eval(
    cfg: HardPowerLawDepthConfig,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, list[float], list[float], Tensor, Tensor, Tensor]:
    """End-to-end evaluation of the hand-coded model on a power-law task.

    Convenience wrapper that:

    1. Builds analytical weights via :func:`init_hand_coded_params`.
    2. Samples a power-law covariance batch via
       :func:`sample_hard_power_law_batch`.
    3. Runs the coupled model for ``5 * cfg.L`` layers with
       ``divide_update_by_sqrt_d=False`` via :func:`model_eval`.

    The 5x depth multiplier and the absence of the ``sqrt(d)`` denominator
    factor are specific to the power-law depth experiment, where the model
    needs more iterations to converge on anisotropic data.

    Args:
        cfg: A :class:`HardPowerLawDepthConfig` specifying dimensions, batch
            size, power-law exponent, seeds, etc.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A tuple ``(out, train_losses, test_losses, X, y, powers)`` where:

        * ``out`` -- Tensor of shape ``(B, P + P_test)`` with model
          predictions.
        * ``train_losses`` -- Per-layer mean squared residual on training
          positions (list of length ``5 * L``).
        * ``test_losses`` -- Per-layer mean squared error on test positions
          (list of length ``5 * L``).
        * ``X`` -- Input features, shape ``(B, P + P_test, d)``.
        * ``y`` -- Labels, shape ``(B, P + P_test)``.
        * ``powers`` -- Per-coordinate power-law scaling factors, shape
          ``(B, d)``.
    """
    params = init_hand_coded_params(cfg.d, device=device, dtype=dtype)
    X, y, powers = sample_hard_power_law_batch(cfg, device=device, dtype=dtype)
    out, train_losses, test_losses = model_eval(
        params,
        X,
        y,
        L=5 * cfg.L,
        P_test=cfg.P_test,
        beta=cfg.beta_model,
        divide_update_by_sqrt_d=False,
    )
    return out, train_losses, test_losses, X, y, powers
