"""Canonical thesis plotting style and helpers.

All thesis figures - exact theorem validations, architecture-aligned runs,
scaling-law analyses, and robustness tests - must be produced through this
module so that they are visually comparable across chapters. This is the
engineering realization of ``EXPERIMENT_PLAN_FINAL.MD`` Section 4.2: the
plotting utility's purpose is *not* aesthetic alone, it is to keep the
cross-chapter comparison from being muddled by inconsistent styling.

The style is deliberately close to the Bordelon figures so that reviewers can
overlay thesis plots on the original reproductions. Sequential sweeps (depth,
rank, context length) use the ``rocket`` palette; phase diagrams use ``mako``;
diverging quantities (OOD, signed residuals) use ``vlag``; categorical groups
use ``colorblind``.

Style is *not* applied on import. Call :func:`apply_thesis_style` explicitly
or use :func:`thesis_style` as a context manager. This keeps the Bordelon
scripts - which also call ``sns.set`` - unaffected when this module is merely
imported as a dependency.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Canonical rcParams and palettes
# ---------------------------------------------------------------------------


THESIS_RCPARAMS: dict[str, Any] = {
    # Font sizes - slightly larger than matplotlib defaults, matching thesis typography.
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    # Legend and frame conventions.
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.6",
    # Default figure sizing. Individual scripts override as needed.
    "figure.figsize": (5.5, 4.0),
    "figure.dpi": 120,
    # Save conventions. 300 dpi PNG + vector PDF with Type-42 fonts for LaTeX
    # embedding. ``pdf.fonttype`` / ``ps.fonttype`` = 42 means TrueType (Type-42)
    # rather than Type-3; this is required for selectable/searchable text in the
    # final PDF and avoids subset-font warnings from LaTeX engines.
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Grid (whitegrid via seaborn; these tweaks keep it subtle).
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    # Line and marker defaults.
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
}


# Palette names used across the thesis. They are strings so that seaborn can
# interpret them via ``sns.color_palette`` - that way the same name also works
# when an individual routine asks for a continuous colormap.
PALETTE_SEQUENTIAL: str = "rocket"       # ordered sweeps: depth, rank, P, t
PALETTE_PHASE: str = "mako"              # 2D phase diagrams
PALETTE_DIVERGING: str = "vlag"          # OOD shift, signed residuals
PALETTE_CATEGORICAL: str = "colorblind"  # unordered groups


def apply_thesis_style() -> None:
    """Apply the canonical rcParams and seaborn whitegrid style.

    Safe to call repeatedly; it replaces the relevant rcParams each time.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update(THESIS_RCPARAMS)


@contextlib.contextmanager
def thesis_style() -> Iterator[None]:
    """Context manager that applies the thesis style and restores on exit."""
    old = dict(plt.rcParams)
    try:
        apply_thesis_style()
        yield
    finally:
        plt.rcParams.update(old)


def sequential_colors(n: int, palette: str = PALETTE_SEQUENTIAL) -> list[tuple[float, float, float]]:
    """Return ``n`` evenly spaced RGB tuples from a sequential palette."""
    return list(sns.color_palette(palette, n_colors=n))


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_both(
    fig: Figure,
    run_dir,
    name: str,
    *,
    also_pdf: bool = True,
    **savefig_kw: Any,
) -> tuple[str, str | None]:
    """Save a figure to both ``figures/<name>.png`` and ``pdfs/<name>.pdf``.

    ``run_dir`` must expose ``.png(name) -> Path`` and ``.pdf(name) -> Path``
    (see :class:`scripts.thesis.utils.run_metadata.ThesisRunDir`).

    Returns the paths of the saved PNG and PDF (or ``None`` for the PDF if
    ``also_pdf=False``).
    """
    defaults = dict(
        dpi=plt.rcParams.get("savefig.dpi", 300),
        bbox_inches=plt.rcParams.get("savefig.bbox", "tight"),
    )
    defaults.update(savefig_kw)
    png_path = run_dir.png(name)
    fig.savefig(png_path, **defaults)
    pdf_path: str | None = None
    if also_pdf:
        pdf_path = run_dir.pdf(name)
        fig.savefig(pdf_path, **defaults)
    return str(png_path), (str(pdf_path) if pdf_path is not None else None)


# ---------------------------------------------------------------------------
# Overlays for theory curves
# ---------------------------------------------------------------------------


def overlay_powerlaw(
    ax: Axes,
    x: Sequence[float] | np.ndarray,
    *,
    coef: float,
    exponent: float,
    label: str | None = None,
    style: str = "--",
    color: str = "black",
    lw: float = 1.2,
    zorder: int = 10,
) -> None:
    """Overlay ``y = coef * x**exponent`` as a dashed reference line.

    Intended for theorem-predicted scaling exponents such as ``t**(-beta/(2+beta))``.
    """
    x_arr = np.asarray(x, dtype=float)
    y = coef * np.power(x_arr, exponent)
    ax.plot(x_arr, y, style, color=color, lw=lw, zorder=zorder, label=label)


def overlay_reference(
    ax: Axes,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    label: str | None = None,
    style: str = "--",
    color: str = "black",
    lw: float = 1.2,
    zorder: int = 10,
) -> None:
    """Overlay an arbitrary theory curve (DMFT solution, exact ODE, ...)."""
    ax.plot(np.asarray(x), np.asarray(y), style, color=color, lw=lw, zorder=zorder, label=label)


# ---------------------------------------------------------------------------
# Phase-diagram heatmap (Theorem-C C4 style)
# ---------------------------------------------------------------------------


def phase_heatmap(
    ax: Axes,
    values: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    cmap: str = PALETTE_PHASE,
    log_z: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    shading: str = "auto",
) -> tuple[Any, Colorbar]:
    """Draw a 2D phase-diagram heatmap.

    Parameters
    ----------
    ax
        Target axes.
    values
        2D array shaped ``(len(y_coords), len(x_coords))`` of values to color.
    x_coords, y_coords
        1D coordinate arrays for the two axes. Can be numeric or categorical
        indices; log-scaling requires positive numeric values.
    xlabel, ylabel, cbar_label
        Axis labels.
    cmap
        Seaborn/matplotlib colormap name.
    log_z
        If ``True``, use a :class:`~matplotlib.colors.LogNorm` colormap.
    log_x, log_y
        Use log scale on the corresponding axes.
    vmin, vmax
        Color-limit overrides.
    shading
        Passed through to :meth:`matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    (mesh, colorbar)
        The underlying pcolormesh artist and its colorbar, for further styling.
    """
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_z else Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(
        x_coords,
        y_coords,
        values,
        cmap=cmap,
        norm=norm,
        shading=shading,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)
    return mesh, cbar


# ---------------------------------------------------------------------------
# Mode trajectories (Theorem-B B1 and Theorem-C C2 style)
# ---------------------------------------------------------------------------


def mode_trajectories(
    ax: Axes,
    t: np.ndarray,
    modes: np.ndarray,
    mode_indices: Sequence[int] | None = None,
    *,
    cmap: str = PALETTE_SEQUENTIAL,
    loglog: bool = True,
    alpha: float = 0.85,
    label_fmt: str | None = None,
) -> list[tuple[float, float, float]]:
    """Plot per-mode trajectories (e.g., ``gamma_k(t)`` or Fourier-mode residuals).

    Parameters
    ----------
    ax
        Target axes.
    t
        1D array of time / step values (length ``T``).
    modes
        2D array of shape ``(T, K)`` with the mode values over time.
    mode_indices
        Sub-selection of modes to plot. Defaults to all of them.
    cmap
        Sequential palette name used to color modes by index.
    loglog
        If ``True``, both axes are log-scaled (the default for scaling curves).
    alpha
        Line alpha.
    label_fmt
        Optional format string. If provided, each line is labeled by
        ``label_fmt.format(k=mode_index)``.

    Returns
    -------
    List of RGB tuples used, so callers can match overlay colors.
    """
    modes_arr = np.asarray(modes)
    if modes_arr.ndim != 2:
        raise ValueError(
            f"expected modes.shape=(T, K); got shape={modes_arr.shape}"
        )
    T, K = modes_arr.shape
    t_arr = np.asarray(t)
    if t_arr.ndim != 1 or t_arr.shape[0] != T:
        raise ValueError(
            f"expected t.shape=(T,) matching modes, got t.shape={t_arr.shape}, T={T}"
        )
    idx = list(range(K)) if mode_indices is None else list(mode_indices)
    colors = sequential_colors(len(idx), palette=cmap)
    for color, k in zip(colors, idx):
        label = label_fmt.format(k=k) if label_fmt is not None else None
        ax.plot(t_arr, modes_arr[:, k], color=color, alpha=alpha, label=label)
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    return colors


# ---------------------------------------------------------------------------
# Compute-frontier plot (Theorem-D / scaling-law style)
# ---------------------------------------------------------------------------


def frontier_plot(
    ax: Axes,
    compute: np.ndarray,
    loss: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    predicted_frontier: tuple[np.ndarray, np.ndarray] | None = None,
    scatter_kw: dict[str, Any] | None = None,
    annotate: bool = False,
) -> None:
    """Plot ``(compute, loss)`` points with an optional predicted Pareto frontier.

    When ``predicted_frontier`` is provided it is drawn as a dashed black line,
    consistent with other theorem-predicted overlays.
    """
    scatter_kw = dict(scatter_kw or {})
    scatter_kw.setdefault("s", 22)
    scatter_kw.setdefault("alpha", 0.85)
    ax.scatter(compute, loss, **scatter_kw)
    if annotate and labels is not None:
        for c, l, lab in zip(compute, loss, labels):
            ax.annotate(
                lab, (c, l), fontsize=7, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )
    if predicted_frontier is not None:
        cx, cy = predicted_frontier
        ax.plot(np.asarray(cx), np.asarray(cy), "--", color="black", lw=1.2,
                label="predicted frontier")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compute")
    ax.set_ylabel("loss")


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------


def legend_compact(ax: Axes, *, ncol: int = 1, outside: bool = False, **kw: Any) -> Any:
    """Render a compact legend, optionally outside the plot area."""
    if outside:
        return ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=ncol,
            frameon=False,
            **kw,
        )
    return ax.legend(ncol=ncol, **kw)


__all__ = [
    "THESIS_RCPARAMS",
    "PALETTE_SEQUENTIAL",
    "PALETTE_PHASE",
    "PALETTE_DIVERGING",
    "PALETTE_CATEGORICAL",
    "apply_thesis_style",
    "thesis_style",
    "sequential_colors",
    "save_both",
    "overlay_powerlaw",
    "overlay_reference",
    "phase_heatmap",
    "mode_trajectories",
    "frontier_plot",
    "legend_compact",
]
