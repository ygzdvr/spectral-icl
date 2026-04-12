import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


_PROJ = str(Path(__file__).resolve().parents[1])
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from dynamics import visualize_loss_landscape
from utils import OutputDir


def main() -> None:
    """Run notebook-style loss-landscape studies across multiple settings.

    Executes several predefined experiments (varying depth and sigma), saves
    resulting figures, and exports arrays for offline analysis.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook loss landscape visualization."
    )
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--lamb-min", type=float, default=0.01)
    parser.add_argument("--lamb-max", type=float, default=10.0)
    parser.add_argument("--lamb-points", type=int, default=100)
    parser.add_argument("--gamma-points", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    out = OutputDir(__file__, base=args.output_dir)

    lamb_grid = torch.linspace(args.lamb_min, args.lamb_max, args.lamb_points, dtype=dtype)
    gamma_vals = torch.linspace(0, 1.0, args.gamma_points, dtype=dtype)
    gamma_np = gamma_vals.numpy()

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    # =====================================================================
    # Experiment 1: single case alpha=1.25, L=10 (semilogy diagnostic)
    # =====================================================================
    alpha = 1.25
    L = 10
    gamma_vals_50 = torch.linspace(0, 1.0, 50, dtype=dtype)
    losses_single = [
        float(visualize_loss_landscape(L * g.item(), lamb_grid, alpha, L))
        for g in gamma_vals_50
    ]
    print(f"Single case losses: {losses_single[:5]}...")

    plt.figure()
    plt.semilogy(gamma_vals_50.numpy(), losses_single)
    plt.ylim([1e-3, 1.2])
    plt.savefig(out.png("loss_landscape_single"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("loss_landscape_single"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('loss_landscape_single')}")

    # =====================================================================
    # Experiment 2: vary L, alpha=0.5, sigma=0.0
    # =====================================================================
    alpha = 0.5
    Lvals = [1, 2, 4, 8, 16, 32]

    all_losses_L: list[list[float]] = []
    for L in Lvals:
        losses_L = [
            float(visualize_loss_landscape(L * g.item(), lamb_grid, alpha, L))
            for g in gamma_vals
        ]
        all_losses_L.append(losses_L)

    plt.figure()
    sns.set_palette("rocket", n_colors=len(Lvals))
    for i, L in enumerate(Lvals):
        plt.plot(gamma_np, all_losses_L[i], label=f"$L = {L}$")
    plt.legend()
    plt.ylim([0, 1.01])
    plt.xlabel(r"$\gamma \ / \  L$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(\gamma)$", fontsize=20)
    plt.savefig(out.png(f"loss_landscape_gamma_alpha_{alpha}"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf(f"loss_landscape_gamma_alpha_{alpha}"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png(f'loss_landscape_gamma_alpha_{alpha}')}")

    # =====================================================================
    # Experiment 2b: vary L, alpha=2.0, sigma=0.0
    # =====================================================================
    alpha = 2.0

    all_losses_L2: list[list[float]] = []
    for L in Lvals:
        losses_L2 = [
            float(visualize_loss_landscape(L * g.item(), lamb_grid, alpha, L))
            for g in gamma_vals
        ]
        all_losses_L2.append(losses_L2)

    plt.figure()
    sns.set_palette("rocket", n_colors=len(Lvals))
    for i, L in enumerate(Lvals):
        plt.plot(gamma_np, all_losses_L2[i], label=f"$L = {L}$")
    plt.legend()
    plt.ylim([0, 1.01])
    plt.xlabel(r"$\gamma \ / \  L$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(\gamma)$", fontsize=20)
    plt.savefig(out.png(f"loss_landscape_gamma_alpha_{alpha}"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf(f"loss_landscape_gamma_alpha_{alpha}"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png(f'loss_landscape_gamma_alpha_{alpha}')}")

    # =====================================================================
    # Experiment 3: vary sigma, L=16, alpha=1.125
    # =====================================================================
    L = 16
    alpha = 1.125
    sigmas = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_losses_sigma: list[list[float]] = []
    for sigma in sigmas:
        losses_s = [
            float(
                visualize_loss_landscape(
                    L * g.item(), lamb_grid, alpha, L, sigma=sigma
                )
            )
            for g in gamma_vals
        ]
        all_losses_sigma.append(losses_s)

    plt.figure()
    sns.set_palette("rocket", n_colors=len(sigmas))
    for i, sigma in enumerate(sigmas):
        plt.plot(gamma_np, all_losses_sigma[i], label=f"$\\sigma = {sigma}$")
    plt.legend()
    plt.ylim([0, 1.01])
    plt.xlabel(r"$\gamma \ / \ L$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(\gamma)-\sigma^2$", fontsize=20)
    plt.savefig(out.png("loss_landscape_gamma_vary_sigma"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("loss_landscape_gamma_vary_sigma"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('loss_landscape_gamma_vary_sigma')}")

    # --- Save all data ---
    payload: dict[str, np.ndarray] = {
        "gamma_vals": gamma_np,
        "lamb_grid": lamb_grid.numpy(),
    }
    for i, L in enumerate(Lvals):
        payload[f"vary_L_L{L}"] = np.asarray(all_losses_L[i], dtype=np.float64)
    for i, L in enumerate(Lvals):
        payload[f"vary_L_alpha2_L{L}"] = np.asarray(all_losses_L2[i], dtype=np.float64)
    for i, sigma in enumerate(sigmas):
        payload[f"vary_sigma_s{sigma:.2f}"] = np.asarray(
            all_losses_sigma[i], dtype=np.float64
        )
    np.savez(out.numpy("loss_landscape_data"), **payload)
    print(f"Saved data to: {out.numpy('loss_landscape_data')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
