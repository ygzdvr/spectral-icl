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
    reduced_gamma_structured_fixed_sgd_rmt_isotropic_dynamics,
    ood_loss_fixed_covariance,
)
from utils import make_powerlaw_spec_and_wstar, parse_int_list


def main() -> None:
    """Run fixed-covariance depth sweeps and OOD-rotation loss evaluations.

    The script executes the training dynamics across depth values, tracks
    eigenvalue evolution, evaluates OOD losses, and saves complete outputs.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook fixed-covariance dynamics + OOD generalization."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--beta", type=float, default=1.25)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--lamb", type=float, default=1e-6)
    parser.add_argument("--lvals", type=str, default="1,2,4,8")
    parser.add_argument("--eig-ks", type=str, default="1,2,4,8,16")
    parser.add_argument("--theta-points", type=int, default=250)
    parser.add_argument("--theta-max", type=float, default=0.25)
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
    eig_ks = parse_int_list(args.eig_ks)

    # =====================================================================
    # 1) Fixed-covariance depth sweep
    # =====================================================================
    all_loss: list[list[float]] = []
    all_eigs: list[list[torch.Tensor]] = []
    for L in lvals:
        print(f"L = {L}")
        losses, eigs = reduced_gamma_structured_fixed_sgd_rmt_isotropic_dynamics(
            spec, w_star, N=args.n, L=L, B=args.batch, K=args.k, P=args.p,
            T=T, eta=args.eta, lamb=args.lamb, ctx_sample=False,
            device=args.device, dtype=dtype,
        )
        all_loss.append(losses)
        all_eigs.append(eigs)

    t_axis = np.linspace(10, T, T)
    th_exp = beta / (alpha + beta * alpha + 1)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    # --- Plot 1: loss trajectories ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        plt.loglog(all_loss[i], label=f"$L={L}$")
    plt.loglog(
        t_axis,
        0.65 * t_axis ** (-th_exp),
        "--",
        color="red",
        label=r"$t^{- \frac{\beta}{\nu + \nu \beta + 1}}$",
    )
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t)$", fontsize=20)
    plt.legend()
    plt.savefig(output_dir / "losses_fixed_covariance.png", dpi=200, bbox_inches="tight")
    print("Saved losses_fixed_covariance.png")

    # --- Plot 2: eigenvalue evolution (last L) ---
    last_eigs = torch.stack(all_eigs[-1])  # (T, d)
    ts = np.linspace(1, len(all_eigs[-1]), len(all_eigs[-1]))
    spec_np = spec.cpu().numpy()
    w_star_np = w_star.cpu().numpy()

    plt.figure()
    sns.set_palette("rocket", n_colors=len(eig_ks))
    for j, k in enumerate(eig_ks):
        idx = k - 1  # 1-indexed to 0-indexed
        plt.loglog(ts, last_eigs[:, idx].numpy(), color=f"C{j}", label=f"$k = {k}$")

    # Theory lines: log(1 + 4*eta*spec_k^3 * w_star_k^2 * t) / (spec_k * 2)
    for j, k in enumerate(eig_ks):
        idx = k - 1
        theory = np.log(1.0 + 4 * args.eta * spec_np[idx] ** 3 * w_star_np[idx] ** 2 * ts) / (
            spec_np[idx] * 2.0
        )
        plt.loglog(ts, theory, "--", color="black")
    plt.loglog([], [], "--", color="black", label="Theory")
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\gamma_k(t)$", fontsize=20)
    plt.legend()
    plt.savefig(output_dir / "eig_evolution_fixed_covariance.png", dpi=200, bbox_inches="tight")
    print("Saved eig_evolution_fixed_covariance.png")

    # =====================================================================
    # 2) OOD loss under rotation
    # =====================================================================
    print("\n=== Computing OOD loss under rotation ===")
    thetas = torch.linspace(0, args.theta_max, args.theta_points, device=device, dtype=dtype)
    ood_losses = ood_loss_fixed_covariance(
        spec, w_star, lvals, thetas, device=device, dtype=dtype
    )

    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    thetas_np = thetas.cpu().numpy()
    for i, L in enumerate(lvals):
        plt.semilogy(thetas_np, ood_losses[i], label=f"$L = {L}$")
    plt.ylim([1e-2, 1e2])
    plt.legend()
    plt.xlabel(r"$\theta$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}_{OOD}$", fontsize=20)
    plt.savefig(output_dir / "ood_loss_fixed_covariance.png", dpi=200, bbox_inches="tight")
    print("Saved ood_loss_fixed_covariance.png")

    # --- Save data ---
    npz_path = output_dir / "fixed_covariance_data.npz"
    payload: dict[str, np.ndarray] = {
        "lvals": np.asarray(lvals, dtype=np.int64),
        "thetas": thetas_np,
        "spec": spec.cpu().numpy(),
        "w_star": w_star.cpu().numpy(),
    }
    for i, L in enumerate(lvals):
        payload[f"loss_L_{L}"] = np.asarray(all_loss[i], dtype=np.float64)
        payload[f"ood_L_{L}"] = np.asarray(ood_losses[i], dtype=np.float64)
        eig_arr = torch.stack(all_eigs[i]).numpy()
        payload[f"eigs_L_{L}"] = eig_arr
    np.savez(npz_path, **payload)
    print(f"Saved data to: {npz_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
