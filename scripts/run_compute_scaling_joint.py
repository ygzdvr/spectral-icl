import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics import (
    reduced_gamma_structured_sgd_rmt_isotropic_dynamics,
    solve_n_final,
)
from utils import OutputDir, make_powerlaw_spec_and_wstar, parse_int_list


def main() -> None:
    """Run joint N-L compute scaling experiments with theory diagnostics.

    Performs empirical joint scaling sweeps, overlays compute-law references,
    computes `solve_n_final` theory curves, and exports full results.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook joint N-L compute scaling with theory curves."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eta", type=float, default=0.85)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--nvals", type=str, default="4,8,16,32,64")
    parser.add_argument("--n-base-divisor", type=float, default=4.0,
                        help="L = int((N / divisor) ^ alpha)")
    # solve_N_final theory curve params
    parser.add_argument("--m-theory", type=int, default=4000)
    parser.add_argument("--n-theory-points", type=int, default=50)
    parser.add_argument("--n-theory-min-exp", type=float, default=0.0)
    parser.add_argument("--n-theory-max-exp", type=float, default=3.0)
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

    Nvals = parse_int_list(args.nvals)

    # =====================================================================
    # 1) Joint N-L sweep: L = int((N / divisor)^alpha)
    # =====================================================================
    all_loss: list[list[float]] = []
    configs: list[tuple[int, int]] = []
    for N in Nvals:
        L = max(1, int((N / args.n_base_divisor) ** alpha))
        configs.append((N, L))
        print(f"N = {N}, L = {L}")
        losses, _, _ = reduced_gamma_structured_sgd_rmt_isotropic_dynamics(
            spec, w_star, N=N, L=L, B=args.batch, K=args.k, P=args.p,
            T=T, eta=args.eta, lamb=args.lamb, ctx_sample=False,
            device=args.device, dtype=dtype,
        )
        all_loss.append(losses)

    t_steps = np.linspace(1, T, T)
    exp_th = alpha * beta / ((3 + beta) * alpha + 2.0)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    # --- Plot: loss vs compute C = N^2 * L * t ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(Nvals))
    last_N = Nvals[-1]
    last_L = max(1, int((last_N / args.n_base_divisor) ** alpha))
    for i, N in enumerate(Nvals):
        _, L = configs[i]
        compute = N**2 * int(N / args.n_base_divisor) * t_steps
        plt.loglog(compute, all_loss[i], label=f"$N = {N}$")

    # Theory line
    compute_theory = 0.01 * last_N**2 * int(last_N / args.n_base_divisor) * np.linspace(20, 100 * T, T)
    plt.plot(
        compute_theory,
        1.1 * np.linspace(20, 100 * T, T) ** (-exp_th),
        "--",
        color="red",
        label=r"$C^{ - \frac{\nu\beta}{(3+\beta)\nu + 2} }$",
    )
    plt.ylim([0.1, 1.05])
    plt.legend()
    plt.xlabel(r"Compute", fontsize=20)
    plt.ylabel(r"Loss", fontsize=20)
    plt.savefig(out.png("compute_scaling_joint_linear_more_NL"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("compute_scaling_joint_linear_more_NL"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out.png('compute_scaling_joint_linear_more_NL')}")

    # =====================================================================
    # 2) solve_N_final: loss floor vs N (theory curve)
    # =====================================================================
    print("\n=== Computing solve_N_final theory curve ===")
    loss_8 = solve_n_final(spec, w_star, N=8.0)
    print(f"loss_8 = {loss_8}")

    M_th = args.m_theory
    spec_th, w_star_th = make_powerlaw_spec_and_wstar(
        M_th,
        alpha,
        beta,
        device=device,
        dtype=dtype,
    )

    N_th = torch.logspace(
        args.n_theory_min_exp, args.n_theory_max_exp, args.n_theory_points,
        device=device, dtype=dtype,
    )
    loss_vs_N = [solve_n_final(spec_th, w_star_th, float(n)) for n in N_th]
    print(f"loss_vs_N = {loss_vs_N}")
    print(f"len(spec) = {len(spec)}")

    # --- Save data ---
    payload: dict[str, np.ndarray] = {
        "Nvals": np.asarray(Nvals, dtype=np.int64),
        "loss_8": np.asarray([loss_8], dtype=np.float64),
        "N_th": N_th.cpu().numpy(),
        "loss_vs_N": np.asarray(loss_vs_N, dtype=np.float64),
    }
    for i, N in enumerate(Nvals):
        payload[f"loss_N_{N}"] = np.asarray(all_loss[i], dtype=np.float64)
    np.savez(out.numpy("compute_scaling_joint_data"), **payload)
    print(f"Saved data to: {out.numpy('compute_scaling_joint_data')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
