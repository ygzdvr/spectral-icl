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

from dynamics import (
    reduced_gamma_structured_sgd_rmt_isotropic_dynamics,
    powerlaw_loss_landscape,
)
from utils import make_powerlaw_spec_and_wstar, parse_int_list


def main() -> None:
    """Run reduced-gamma SGD dynamics and/or power-law loss landscapes.

    Depending on flags, this script executes a time-dynamics experiment,
    evaluates loss-vs-gamma landscapes for multiple depths, and writes
    publication-style plots plus `.npz` summaries.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook reduced-Gamma structured SGD dynamics + loss landscape."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    # SGD dynamics params
    parser.add_argument("--run-sgd", action="store_true", help="Run the structured SGD dynamics experiment")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--l", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--lamb", type=float, default=1e-3)

    # Loss landscape params
    parser.add_argument("--gamma-min-exp", type=float, default=-2.0)
    parser.add_argument("--gamma-max-exp", type=float, default=3.0)
    parser.add_argument("--gamma-points", type=int, default=500)
    parser.add_argument("--landscape-lvals", type=str, default="1,2,4,8,16,32,64")

    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    M = args.m
    alpha = args.alpha
    beta = args.beta

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec, w_star = make_powerlaw_spec_and_wstar(
        M,
        alpha,
        beta,
        device=device,
        dtype=dtype,
    )

    # =====================================================================
    # 0) Structured SGD dynamics (optional, --run-sgd)
    # =====================================================================
    if args.run_sgd:
        print(f"=== Running reduced-Gamma SGD dynamics ===")
        print(f"  M={M}, N={args.n}, L={args.l}, B={args.batch}, K={args.k}, P={args.p}")
        print(f"  T={args.steps}, eta={args.eta}, alpha={alpha}, beta={beta}")

        sgd_losses, sgd_mean_eigs, sgd_var_eigs = (
            reduced_gamma_structured_sgd_rmt_isotropic_dynamics(
                spec, w_star, N=args.n, L=args.l, B=args.batch, K=args.k, P=args.p,
                T=args.steps, eta=args.eta, lamb=args.lamb,
                device=args.device, dtype=dtype,
            )
        )

        t_axis = np.linspace(1, args.steps, args.steps)
        th_exp = beta / (2 + beta)

        plt.figure()
        plt.loglog(sgd_losses)
        plt.loglog(
            t_axis,
            1.3 * t_axis ** (-th_exp),
            "--",
            color="blue",
            label=rf"$t^{{-\beta/(2+\beta)}}$",
        )
        plt.legend()
        plt.xlabel(r"$t$", fontsize=20)
        plt.ylabel(r"Loss", fontsize=20)
        plt.savefig(output_dir / "reduced_gamma_sgd_loss.png", dpi=200, bbox_inches="tight")
        print("Saved reduced_gamma_sgd_loss.png")

        sgd_npz = output_dir / "reduced_gamma_sgd_data.npz"
        np.savez(
            sgd_npz,
            losses=np.asarray(sgd_losses, dtype=np.float64),
            mean_eigs=np.asarray(sgd_mean_eigs, dtype=np.float64),
            var_eigs=np.asarray(sgd_var_eigs, dtype=np.float64),
        )
        print(f"Saved SGD data to: {sgd_npz}")

    # =====================================================================
    # 1) Loss landscape: loss(gamma) for varying L
    # =====================================================================
    gammas = torch.logspace(
        args.gamma_min_exp, args.gamma_max_exp, args.gamma_points,
        device=device, dtype=dtype,
    )
    landscape_lvals = parse_int_list(args.landscape_lvals)

    landscape_losses = [
        powerlaw_loss_landscape(gammas, spec, w_star, L=L).cpu().numpy()
        for L in landscape_lvals
    ]

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    plt.figure()
    sns.set_palette("rocket", n_colors=len(landscape_lvals))
    gammas_np = gammas.cpu().numpy()
    for i, L in enumerate(landscape_lvals):
        plt.loglog(gammas_np, landscape_losses[i], label=f"$L = {L}$")
    plt.ylim([5e-3, 2.0])
    plt.legend()
    plt.xlabel(r"$\gamma$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(\gamma)$", fontsize=20)
    plt.savefig(output_dir / "loss_landscape_powerlaw.png", dpi=200, bbox_inches="tight")
    print("Saved loss_landscape_powerlaw.png")

    # =====================================================================
    # 2) Min loss vs L (power-law scaling)
    # =====================================================================
    min_losses = [float(np.min(ll)) for ll in landscape_losses]

    plt.figure()
    lvals_arr = np.asarray(landscape_lvals, dtype=np.float64)
    plt.loglog(lvals_arr, min_losses, "-o")
    plt.loglog(
        lvals_arr,
        0.8 * lvals_arr ** (-beta),
        "--",
        color="black",
        label=rf"$L^{{-\beta}}$",
    )
    plt.legend()
    plt.xlabel(r"$L$", fontsize=20)
    plt.ylabel(r"$\min_\gamma \mathcal{L}(\gamma)$", fontsize=20)
    plt.savefig(output_dir / "min_loss_vs_depth_powerlaw.png", dpi=200, bbox_inches="tight")
    print("Saved min_loss_vs_depth_powerlaw.png")

    # =====================================================================
    # Save data
    # =====================================================================
    npz_path = output_dir / "reduced_gamma_landscape.npz"
    payload: dict[str, np.ndarray] = {
        "gammas": gammas_np,
        "landscape_lvals": np.asarray(landscape_lvals, dtype=np.int64),
        "min_losses": np.asarray(min_losses, dtype=np.float64),
        "spec": spec.cpu().numpy(),
        "w_star": w_star.cpu().numpy(),
    }
    for i, L in enumerate(landscape_lvals):
        payload[f"landscape_L_{L}"] = landscape_losses[i]
    np.savez(npz_path, **payload)
    print(f"Saved data to: {npz_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
