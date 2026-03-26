"""Smoothing utilities for plotting training trajectories.

Most plotting scripts visualize noisy optimization curves. A lightweight moving
average smooth is often applied before plotting to improve readability while
preserving trend shape. This module centralizes that operation.
"""

from __future__ import annotations

import numpy as np


def moving_average(xs: np.ndarray, window: int) -> np.ndarray:
    """Apply a 1D simple moving average to an array.

    The output length follows NumPy ``convolve(..., mode="valid")`` semantics:
    for input length ``n`` and window ``w``, output length is ``n - w + 1``.

    Args:
        xs: 1D numeric array to smooth.
        window: Averaging window width. If ``window <= 1``, returns ``xs.copy()``.

    Returns:
        Smoothed array (or copy of original when no smoothing is requested).
    """
    if window <= 1:
        # Return a copy so callers can modify result without mutating input.
        return xs.copy()

    # Uniform kernel for simple moving average.
    kernel = np.ones(window, dtype=np.float64) / window

    # "valid" avoids padded boundary effects and keeps only fully-supported windows.
    return np.convolve(xs, kernel, mode="valid")
