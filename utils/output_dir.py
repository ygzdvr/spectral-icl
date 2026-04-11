"""Structured per-experiment output directory helper."""

from __future__ import annotations

from pathlib import Path


class OutputDir:
    """Organized output: figures/, pdfs/, pt/, npz/ under outputs/<experiment>/.

    Usage::

        out = OutputDir(__file__)
        plt.savefig(out.png("loss_curve"), dpi=200, bbox_inches="tight")
        plt.savefig(out.pdf("loss_curve"), dpi=200, bbox_inches="tight")
        np.savez(out.numpy("losses"), **payload)
        torch.save(artifacts, out.torch("artifacts"))
    """

    def __init__(self, script_file: str, base: Path | str | None = None) -> None:
        stem = Path(script_file).stem
        if base is not None:
            root = Path(base) / stem
        else:
            root = Path(script_file).resolve().parents[1] / "outputs" / stem
        self.root = root
        self.figures = root / "figures"
        self.pdfs = root / "pdfs"
        self.pt = root / "pt"
        self.npz = root / "npz"
        for d in (self.figures, self.pdfs, self.pt, self.npz):
            d.mkdir(parents=True, exist_ok=True)

    def png(self, name: str) -> Path:
        """Return path for a PNG figure: <root>/figures/<name>.png"""
        return self.figures / f"{name}.png"

    def pdf(self, name: str) -> Path:
        """Return path for a PDF figure: <root>/pdfs/<name>.pdf"""
        return self.pdfs / f"{name}.pdf"

    def torch(self, name: str) -> Path:
        """Return path for a PyTorch artifact: <root>/pt/<name>.pt"""
        return self.pt / f"{name}.pt"

    def numpy(self, name: str) -> Path:
        """Return path for a NumPy archive: <root>/npz/<name>.npz"""
        return self.npz / f"{name}.npz"
