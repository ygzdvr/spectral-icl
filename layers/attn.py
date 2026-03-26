"""Attention layers used by the transformer models.

This module currently provides a single attention block, :class:`Attn`, that is
designed to match the research notebook behavior while still supporting a modern
PyTorch fast-path. The layer exposes two execution modes:

1. Raw dot-product attention (`use_softmax=False`), which intentionally avoids
   softmax normalization to preserve the original theoretical setup.
2. Softmax attention (`use_softmax=True`), implemented through
   ``torch.nn.functional.scaled_dot_product_attention`` (SDPA), which can
   dispatch to optimized kernels such as FlashAttention when available.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Attn(nn.Module):
    """Single-head attention block with notebook-compatible and SDPA modes.

    This layer projects an input sequence into queries, keys, and values with
    learned linear maps, computes attention outputs, and projects back to the
    embedding space.

    Design goals:
    - Preserve the original experiment behavior when ``use_softmax=False``.
    - Provide a high-performance implementation path when ``use_softmax=True``.
    - Keep shape conventions explicit: inputs and outputs are always
      ``[batch, seq_len, embed_dim]``.

    Notes:
    - The layer is intentionally single-head to align with the analytical setup.
    - In raw mode, scaling follows the existing code path and notebook equations,
      not the canonical Transformer ``1/sqrt(d_k)`` softmax formulation.
    """

    def __init__(
        self,
        kq_dim: int = 64,
        embed_dim: int | None = None,
        use_softmax: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        """Initialize projection layers and attention behavior.

        Args:
            kq_dim: Internal dimension used for key/query/value projections.
                This also sets the feature size processed by attention scores.
            embed_dim: External token embedding dimension. If ``None``, defaults
                to ``kq_dim`` so input/output dimensions match internal width.
            use_softmax: If ``True``, use SDPA softmax attention. If ``False``,
                use the raw dot-product path used in the research notebooks.
            dropout_p: Attention dropout probability used only in softmax mode
                during training. It is automatically disabled at eval time.
        """
        super().__init__()
        # `embed_dim` is the public interface dimension; `kq_dim` is the latent
        # attention computation dimension.
        self.kq_dim = kq_dim
        self.embed_dim = embed_dim if embed_dim is not None else kq_dim
        self.use_softmax = use_softmax
        self.dropout_p = dropout_p

        # Single-head projections:
        #   q_proj, k_proj, v_proj: [embed_dim -> kq_dim]
        #   out_proj:               [kq_dim -> embed_dim]
        self.q_proj = nn.Linear(self.embed_dim, self.kq_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.kq_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.kq_dim, bias=False)
        self.out_proj = nn.Linear(self.kq_dim, self.embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run one forward pass of single-head attention.

        Args:
            inputs: Tensor of shape ``[batch, seq_len, embed_dim]``.

        Returns:
            Tensor of shape ``[batch, seq_len, embed_dim]`` after attention and
            output projection.

        Raises:
            ValueError: If ``inputs`` is not rank-3 or the last dimension does
                not match this layer's ``embed_dim``.
        """
        # Validate shape early so downstream einsum/SDPA errors are avoided.
        if inputs.dim() != 3:
            raise ValueError(f"expected [batch, seq, dim], got shape={tuple(inputs.shape)}")
        if inputs.size(-1) != self.embed_dim:
            raise ValueError(
                f"input dim {inputs.size(-1)} does not match embed_dim={self.embed_dim}"
            )

        # Project to attention space.
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        if self.use_softmax:
            # SDPA expects an explicit head axis: [B, H, S, D].
            # We inject H=1, compute attention, then remove the singleton head.
            out = F.scaled_dot_product_attention(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            ).squeeze(1)
        else:
            # Notebook-compatible raw attention path:
            # scale q/k/v by sqrt(input_dim), build unnormalized attention
            # scores with einsum, then mix values with those scores.
            q = q / math.sqrt(inputs.size(-1))
            k = k / math.sqrt(inputs.size(-1))
            v = v / math.sqrt(inputs.size(-1))

            # Matches notebook einsums:
            # A = einsum('ijk,ilk->ijl', k, q) / kq_dim
            # out = einsum('ijl,ilk->ijk', A, v)
            attn = torch.einsum("bsk,btk->bst", k, q) / float(self.kq_dim)
            out = torch.einsum("bst,btd->bsd", attn, v)

        # Final projection back to embedding dimension. The extra scaling keeps
        # magnitudes aligned with the original implementation.
        out = self.out_proj(out) / math.sqrt(v.size(-1))
        return out
