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

from dynamics import reduced_gamma_structured_sgd_rmt_isotropic_dynamics
from utils import make_powerlaw_spec_and_wstar, parse_int_list


def main() -> None:
    """Run reduced-gamma depth sweeps and compare step/compute scaling views.

    This entry point builds a power-law problem, trains across ``lvals``,
    renders trajectory plots, and persists losses/spec/teacher tensors.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook reduced-Gamma depth sweep."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=45)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.75)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    M = args.m
    alpha = args.alpha
    beta = args.beta
    T = args.steps

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec, w_star = make_powerlaw_spec_and_wstar(
        M,
        alpha,
        beta,
        device=device,
        dtype=dtype,
    )

    lvals = parse_int_list(args.lvals)

    all_loss: list[list[float]] = []
    for L in lvals:
        print(f"L = {L}")
        losses, _, _ = reduced_gamma_structured_sgd_rmt_isotropic_dynamics(
            spec, w_star, N=args.n, L=L, B=args.batch, K=args.k, P=args.p,
            T=T, eta=args.eta, lamb=args.lamb,
            device=args.device, dtype=dtype,
        )
        all_loss.append(losses)

    t_axis = np.linspace(10, T, T)
    th_exp = beta / (2 + beta)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    # --- Plot 1: loss vs steps, vary L ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        plt.loglog(all_loss[i], label=f"$L = {L}$")
    plt.loglog(
        t_axis,
        0.6 * t_axis ** (-th_exp),
        "--",
        color="blue",
        label=r"$t^{-\beta/(2+\beta)}$",
    )
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t)$", fontsize=20)
    plt.legend()
    plt.savefig(output_dir / "losses_gamma_model_vary_L.png", dpi=200, bbox_inches="tight")
    print("Saved losses_gamma_model_vary_L.png")

    # --- Plot 2: loss vs compute C = L * t ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        n_steps = len(all_loss[i])
        compute_axis = L * np.linspace(1, n_steps, n_steps)
        plt.loglog(compute_axis, all_loss[i], label=f"$L = {L}$")
    last_L = lvals[-1]
    compute_theory = np.linspace(1, last_L * T, T)
    plt.loglog(
        compute_theory,
        0.5 * compute_theory ** (-beta / (beta + 3.0)),
        "--",
        color="red",
        label=r"$C^{- \beta/(3+\beta) }$",
    )
    plt.xlabel(r"$C$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(C)$", fontsize=20)
    plt.legend()
    plt.savefig(
        output_dir / "losses_gamma_model_compute_vary_L.png", dpi=200, bbox_inches="tight"
    )
    print("Saved losses_gamma_model_compute_vary_L.png")

    # --- Save data ---
    npz_path = output_dir / "reduced_gamma_depth_sweep.npz"
    payload: dict[str, np.ndarray] = {
        "lvals": np.asarray(lvals, dtype=np.int64),
        "spec": spec.cpu().numpy(),
        "w_star": w_star.cpu().numpy(),
    }
    for i, L in enumerate(lvals):
        payload[f"loss_L_{L}"] = np.asarray(all_loss[i], dtype=np.float64)
    np.savez(npz_path, **payload)
    print(f"Saved data to: {npz_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
