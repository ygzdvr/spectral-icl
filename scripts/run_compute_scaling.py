import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics import reduced_gamma_structured_sgd_rmt_isotropic_dynamics
from utils import OutputDir, make_powerlaw_spec_and_wstar


def main() -> None:
    """Run canonical compute-scaling sweeps across width/depth regimes.

    Executes three experiment families (fixed-L, fixed-N, joint N-L), renders
    corresponding compute plots, and writes all loss arrays to disk.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook compute-scaling experiments (vary N at fixed L, vary L at fixed N, joint N-L scaling)."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eta", type=float, default=0.075)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--n-base", type=int, default=8)
    parser.add_argument("--n-ratio", type=float, default=1.5)
    parser.add_argument("--n-count", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    M = args.m
    alpha = args.alpha
    beta = args.beta
    T = args.steps

    out = OutputDir(__file__, base=args.output_dir)

    spec, w_star = make_powerlaw_spec_and_wstar(
        M,
        alpha,
        beta,
        device=device,
        dtype=dtype,
    )

    Nvals = [int(args.n_base * args.n_ratio**k) for k in range(args.n_count)]
    t_steps = np.linspace(1, T, T)

    def run_sweep(label: str, configs: list[tuple[int, int]]) -> list[list[float]]:
        all_losses: list[list[float]] = []
        for N, L in configs:
            print(f"  {label}: N={N}, L={L}")
            losses, _, _ = reduced_gamma_structured_sgd_rmt_isotropic_dynamics(
                spec, w_star, N=N, L=L, B=args.batch, K=args.k, P=args.p,
                T=T, eta=args.eta, lamb=args.lamb, ctx_sample=False,
                device=args.device, dtype=dtype,
            )
            all_losses.append(losses)
        return all_losses

    # =====================================================================
    # 1) Fixed L=1, vary N
    # =====================================================================
    print("=== Sweep 1: vary N, fixed L=1 ===")
    configs_no_scale = [(N, 1) for N in Nvals]
    all_loss_no_scale = run_sweep("no_scale", configs_no_scale)

    # =====================================================================
    # 2) Fixed N=8, vary L
    # =====================================================================
    Lvals_fix = [1, 2, 4, 8, 16]
    N_fixed = args.n_base
    print(f"=== Sweep 2: vary L, fixed N={N_fixed} ===")
    configs_lin_scale = [(N_fixed, L) for L in Lvals_fix]
    all_loss_lin_scale = run_sweep("lin_scale", configs_lin_scale)

    # =====================================================================
    # 3) Joint N-L scaling: L = int(N / 8)
    # =====================================================================
    print("=== Sweep 3: joint N-L scaling, L = int(N/8) ===")
    configs_opt_scale = [(N, max(1, int(N / 8.0))) for N in Nvals]
    all_loss_opt_scale = run_sweep("opt_scale", configs_opt_scale)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    # --- Plot 1: compute scaling, fixed L=1 ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(Nvals))
    for i, N in enumerate(Nvals):
        compute = N**2 * t_steps
        plt.loglog(compute, all_loss_no_scale[i], label=f"$N = {N}$")
    plt.ylim([0.15, 1.0])
    plt.legend()
    plt.xlabel(r"Compute", fontsize=20)
    plt.ylabel(r"Loss", fontsize=20)
    plt.savefig(out.png("compute_scaling_fixed_L"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("compute_scaling_fixed_L"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out.png('compute_scaling_fixed_L')}")

    # --- Plot 2: compute scaling, fixed N ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(Lvals_fix))
    for i, L in enumerate(Lvals_fix):
        compute = L * N_fixed**2 * t_steps
        plt.loglog(compute, all_loss_lin_scale[i], label=f"$L = {L}$", color=f"C{i}")
    plt.legend()
    plt.ylim([0.15, 1.0])
    plt.xlabel(r"Compute", fontsize=20)
    plt.ylabel(r"Loss", fontsize=20)
    plt.savefig(out.png("compute_scaling_fixed_N"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("compute_scaling_fixed_N"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out.png('compute_scaling_fixed_N')}")

    # --- Plot 3: compute scaling, joint N-L ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(Nvals))
    for i, N in enumerate(Nvals):
        L = int((N / 4.0) ** alpha)
        compute = L * N**2 * t_steps
        plt.loglog(compute, all_loss_opt_scale[i], label=f"$N = {N}$", color=f"C{i}")
    plt.legend()
    plt.ylim([0.15, 1.0])
    plt.xlabel(r"Compute", fontsize=20)
    plt.ylabel(r"Loss", fontsize=20)
    plt.savefig(out.png("compute_scaling_joint_NL_scaling"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("compute_scaling_joint_NL_scaling"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out.png('compute_scaling_joint_NL_scaling')}")

    # --- Save data ---
    payload: dict[str, np.ndarray] = {
        "Nvals": np.asarray(Nvals, dtype=np.int64),
        "Lvals_fix": np.asarray(Lvals_fix, dtype=np.int64),
    }
    for i, N in enumerate(Nvals):
        payload[f"no_scale_N_{N}"] = np.asarray(all_loss_no_scale[i], dtype=np.float64)
        payload[f"opt_scale_N_{N}"] = np.asarray(all_loss_opt_scale[i], dtype=np.float64)
    for i, L in enumerate(Lvals_fix):
        payload[f"lin_scale_L_{L}"] = np.asarray(all_loss_lin_scale[i], dtype=np.float64)
    np.savez(out.numpy("compute_scaling_data"), **payload)
    print(f"Saved data to: {out.numpy('compute_scaling_data')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
