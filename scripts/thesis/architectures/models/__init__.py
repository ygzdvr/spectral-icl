"""Trainable PyTorch models for the thesis architecture-aligned tier (§9).

Every model extends :class:`ICLRegressionModel`, which owns the embedding,
alignment enforcement, depth loop, and readout, and leaves the per-layer mixer
update to subclasses.
"""

from scripts.thesis.architectures.models.base import ICLRegressionModel
from scripts.thesis.architectures.models.hybrid import AdaptiveFirstHybrid
from scripts.thesis.architectures.models.linear_attention import LinearAttention
from scripts.thesis.architectures.models.spectral_attention import SpectralAttention
from scripts.thesis.architectures.models.spectral_filter import SpectralFilter
from scripts.thesis.architectures.models.stu_native import STUNative

__all__ = [
    "ICLRegressionModel",
    "AdaptiveFirstHybrid",
    "LinearAttention",
    "SpectralAttention",
    "SpectralFilter",
    "STUNative",
]
