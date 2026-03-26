"""Layer building blocks for spectral scaling experiments.

Exports:
    Attn: Single-head attention block supporting raw and softmax modes.
"""

from .attn import Attn

__all__ = ["Attn"]
