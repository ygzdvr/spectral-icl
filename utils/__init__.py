"""Public utility surface for experiment entry-point scripts.

This package intentionally exposes small, composable helpers that recur across
many standalone scripts:

- parsing helpers for CLI list arguments,
- power-law problem constructors,
- smoothing utilities for plotting,
- device/error helpers for runtime selection and graceful fallback.

Scripts should import from this package-level module when possible:
    ``from utils import parse_int_list, moving_average, ...``
so shared behavior stays centralized and consistent.
"""

from .parsing import parse_float_list, parse_int_list
from .powerlaw import make_powerlaw_spec_and_wstar
from .device import is_cuda_oom, resolve_device
from .smoothing import moving_average
from .analysis import compute_loss_inf_depth, loss_landscape
from .output_dir import OutputDir

__all__ = [
    # Parsing
    "parse_int_list",
    "parse_float_list",
    # Problem construction
    "make_powerlaw_spec_and_wstar",
    # Plot processing
    "moving_average",
    # Analysis helpers
    "compute_loss_inf_depth",
    "loss_landscape",
    # Sweep runners
    "run_experiment",
    "run_sweep",
    # Runtime helpers
    "resolve_device",
    "is_cuda_oom",
    # Output organization
    "OutputDir",
]


def __getattr__(name: str):
    """Lazily import sweep runners to avoid package-level circular imports."""
    if name in {"run_experiment", "run_sweep"}:
        from .sgd_sweeps import run_experiment, run_sweep

        return {"run_experiment": run_experiment, "run_sweep": run_sweep}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
