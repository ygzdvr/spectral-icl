import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import OutputDir, loss_landscape, make_powerlaw_spec_and_wstar, parse_int_list


def main() -> None:
    """Plot and save depth-conditioned analytical loss-vs-weight curves."""
    parser = argparse.ArgumentParser(
        description="Torch port of notebook loss-vs-w landscape plot."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--beta0", type=float, default=0.5)
    parser.add_argument("--lamb", type=float, default=0.0)
    parser.add_argument("--w-min-exp", type=float, default=0.0)
    parser.add_argument("--w-max-exp", type=float, default=0.7)
    parser.add_argument("--w-points", type=int, default=250)
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    out = OutputDir(__file__, base=args.output_dir)

    M = args.m
    spec, w_star = make_powerlaw_spec_and_wstar(
        M,
        args.alpha,
        args.beta,
        device=device,
        dtype=dtype,
        normalize_w_star=False,
    )

    w_grid = torch.logspace(args.w_min_exp, args.w_max_exp, args.w_points, device=device, dtype=dtype)
    lvals = parse_int_list(args.lvals)

    losses = [
        loss_landscape(w_grid, spec, w_star, l=L, beta0=args.beta0, lamb=args.lamb)
        for L in lvals
    ]

    # Print argmin for each L (matches notebook cell)
    for i, L in enumerate(lvals):
        amin = int(torch.argmin(losses[i]).item())
        print(f"L={L:4d}  argmin={amin}  w*={w_grid[amin].item():.4f}  loss*={losses[i][amin].item():.6e}")

    # --- Plot ---
    w_np = w_grid.cpu().numpy()

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    sns.set_palette("rocket", n_colors=len(lvals))

    plt.figure()
    for i, L in enumerate(lvals):
        plt.loglog(w_np, losses[i].cpu().numpy(), label=f"$L={L}$")
    plt.legend()
    plt.xlabel(r"$w$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(w)$", fontsize=20)
    plt.ylim([5e-5, 1e0])

    plt.savefig(out.png("loss_vs_w"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("loss_vs_w"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out.png('loss_vs_w')}")

    # --- Save losses ---
    npz_payload: dict[str, np.ndarray] = {
        "w_grid": w_np.astype(np.float64),
        "lvals": np.asarray(lvals, dtype=np.int64),
        "spec": spec.cpu().numpy().astype(np.float64),
        "w_star": w_star.cpu().numpy().astype(np.float64),
    }
    for i, L in enumerate(lvals):
        npz_payload[f"loss_L_{L}"] = losses[i].cpu().numpy().astype(np.float64)
    np.savez(out.numpy("loss_vs_w_losses"), **npz_payload)
    print(f"Saved losses to: {out.numpy('loss_vs_w_losses')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
