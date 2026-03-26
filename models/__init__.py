"""Model package exports.

Exports:
    SimpleTransformer: Primary transformer module used in experiments.
    simple_transformer: Backward-compatible alias matching notebook naming.
"""

from .transformer import SimpleTransformer, simple_transformer

__all__ = ["SimpleTransformer", "simple_transformer"]
