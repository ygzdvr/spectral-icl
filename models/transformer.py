"""Transformer model definitions used in spectral scaling experiments.

This module provides a compact transformer variant, :class:`SimpleTransformer`,
designed to mirror the behavior of the original research notebooks while using
standard PyTorch modules.

Implementation highlights:
    - Input projection is lazy (dimension inferred on first forward pass),
      matching notebook-style "shape inferred at call time" behavior.
    - Attention blocks are stacked with pre-norm residual connections.
    - Residual updates are scaled by ``beta / depth`` to keep depth-dependent
      dynamics consistent with the analytical setup.
    - Final output is a scalar per token (shape ``[B, S, 1]``).
"""

from torch import nn

try:
    from spectral_scaling.layers.attn import Attn
except ModuleNotFoundError:
    from layers.attn import Attn


class SimpleTransformer(nn.Module):
    """Minimal transformer-style model aligned with notebook experiments.

    Architecture:
        1. Project each input token to latent width ``N``.
        2. Apply ``depth`` blocks of pre-norm attention with residual updates.
        3. Apply a final norm and project to a single scalar output channel.

    The model is intentionally lightweight and single-purpose: most experiments
    in this repo rely on controlling depth/width/residual scaling rather than
    using large production transformer features.
    """

    def __init__(
        self,
        width: int = 64,
        heads: int = 4,
        depth: int = 8,
        beta: float = 1.0,
        out_scale: float = 0.1,
        use_softmax: bool = False,
    ) -> None:
        """Initialize model layers and experiment hyperparameters.

        Args:
            width: Hidden width ``N`` used for token embeddings and attention.
            heads: Reserved for compatibility with notebook/API conventions.
                This model currently uses a single attention head implementation
                in :class:`layers.Attn`; the value is stored for metadata.
            depth: Number of stacked attention residual blocks.
            beta: Residual scaling numerator. Each block update uses
                ``beta / depth`` to keep update magnitudes depth-aware.
            out_scale: Multiplicative scale applied to final scalar predictions.
            use_softmax: If True, attention layers use SDPA softmax mode.
                If False, they use the raw dot-product mode used in theory runs.
        """
        super().__init__()
        # Store constructor parameters for reproducibility/inspection.
        self.width = width
        self.heads = heads
        self.depth = depth
        self.beta = beta
        self.out_scale = out_scale

        # Flax/JAX notebook Dense layers infer input dimension on first call.
        # LazyLinear reproduces that behavior in PyTorch and avoids hardcoding
        # token input dimension in the constructor.
        self.input_proj = nn.LazyLinear(width, bias=False)

        # One LayerNorm + attention layer per depth block (pre-norm layout).
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.attn_layers = nn.ModuleList(
            [
                Attn(kq_dim=width, embed_dim=width, use_softmax=use_softmax)
                for _ in range(depth)
            ]
        )

        # Final normalization and scalar readout head.
        self.final_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, 1, bias=False)

    def forward(self, inputs):
        """Run a forward pass for a batch of token sequences.

        Args:
            inputs: Tensor shaped ``[batch, seq_len, input_dim]`` where
                ``input_dim`` is inferred by ``self.input_proj`` on first call.

        Returns:
            Tensor shaped ``[batch, seq_len, 1]`` containing per-token scalar
            outputs after residual attention processing and final projection.
        """
        # Project raw tokens into the model width.
        x = self.input_proj(inputs)

        # Depth-normalized residual step size used at every block.
        residual_scale = self.beta / max(self.depth, 1)

        # Pre-norm residual attention blocks:
        #   x <- x + (beta/depth) * Attn(LayerNorm(x))
        for norm, attn in zip(self.norms, self.attn_layers):
            x = x + residual_scale * attn(norm(x))

        # Final normalized scalar head.
        x = self.out_scale * self.out_proj(self.final_norm(x))
        return x


# Backward-compatible alias preserving original notebook symbol name.
simple_transformer = SimpleTransformer
