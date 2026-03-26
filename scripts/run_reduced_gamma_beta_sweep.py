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
from utils import make_powerlaw_spec_and_wstar, parse_float_list


def main() -> None:
    """Sweep beta in reduced-gamma dynamics and compare empirical/theory curves.

    For each beta value the script trains dynamics on a matched power-law
    problem, overlays scaling-law references, and exports losses to disk.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook reduced-Gamma beta sweep."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--l", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eta", type=float, default=5.0)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--beta-vals", type=str, default="0.5,0.75,1.0,1.5,2.0,4.0")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    M = args.m
    alpha = args.alpha
    T = args.steps
    beta_vals = parse_float_list(args.beta_vals)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    losses_vary_beta: list[list[float]] = []
    for beta in beta_vals:
        print(f"beta = {beta}")
        spec, w_star = make_powerlaw_spec_and_wstar(
            M,
            alpha,
            beta,
            device=device,
            dtype=dtype,
        )

        losses, _, _ = reduced_gamma_structured_sgd_rmt_isotropic_dynamics(
            spec, w_star, N=args.n, L=args.l, B=args.batch, K=args.k, P=args.p,
            T=T, eta=args.eta, lamb=args.lamb,
            device=args.device, dtype=dtype,
        )
        losses_vary_beta.append(losses)

    # --- Plot: loss vs steps for each beta, with per-beta theory lines ---
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    plt.figure()
    sns.set_palette("rocket", n_colors=len(beta_vals))
    t_axis = np.linspace(20, T, T)
    for i, beta in enumerate(beta_vals):
        loss_arr = np.asarray(losses_vary_beta[i], dtype=np.float64)
        plt.loglog(loss_arr, label=rf"$\beta = {beta:.2f}$")
        # Theory line: t^{-beta/(2+beta)}, normalized to match final loss
        beta_t = beta / (2 + beta)
        loss_th_i = t_axis ** (-beta_t)
        plt.loglog(
            t_axis,
            0.95 * loss_th_i / loss_th_i[-1] * loss_arr[-1],
            "--",
            color="black",
        )
    plt.loglog([], [], "--", color="black", label=r"$t^{- \beta/(2 + \beta)}$")
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t)$", fontsize=20)
    plt.legend()
    plt.savefig(output_dir / "losses_vary_beta.png", dpi=200, bbox_inches="tight")
    print("Saved losses_vary_beta.png")

    # --- Save data ---
    npz_path = output_dir / "reduced_gamma_beta_sweep.npz"
    payload: dict[str, np.ndarray] = {
        "beta_vals": np.asarray(beta_vals, dtype=np.float64),
    }
    for i, beta in enumerate(beta_vals):
        payload[f"loss_beta_{beta:.2f}"] = np.asarray(
            losses_vary_beta[i], dtype=np.float64
        )
    np.savez(npz_path, **payload)
    print(f"Saved data to: {npz_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
