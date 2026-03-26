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
from utils import compute_loss_inf_depth, make_powerlaw_spec_and_wstar, parse_int_list


def main() -> None:
    """Run depth-only compute scaling sweeps at fixed model width.

    Estimates the infinite-depth floor, executes depth sweeps, generates
    compute/raw-loss plots, and saves a structured result bundle.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook depth-only compute scaling (fixed N, vary L)."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--lamb", type=float, default=1e-12)
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16")
    parser.add_argument("--n-floor-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    M = args.m
    alpha = args.alpha
    beta = args.beta
    N = args.n
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

    # =====================================================================
    # 1) Compute L -> infinity loss floor
    # =====================================================================
    print(f"N = {N}")
    loss_inf = compute_loss_inf_depth(spec, w_star, N, n_samples=args.n_floor_samples)
    print(f"loss_inf (L -> inf, N={N}) = {loss_inf}")

    # =====================================================================
    # 2) Depth sweep at fixed N
    # =====================================================================
    all_loss_lin_scale: list[list[float]] = []
    for L in lvals:
        print(f"L = {L}")
        losses, _, _ = reduced_gamma_structured_sgd_rmt_isotropic_dynamics(
            spec, w_star, N=N, L=L, B=args.batch, K=args.k, P=args.p,
            T=T, eta=args.eta, lamb=args.lamb, ctx_sample=False,
            device=args.device, dtype=dtype,
        )
        all_loss_lin_scale.append(losses)

    t_steps = np.linspace(1, T, T)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    # --- Plot 1: loss vs compute (L * N^2 * t) ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        compute = 16**2 * L * t_steps
        plt.loglog(compute, all_loss_lin_scale[i], label=f"$L = {L}$")

    # L -> infinity floor
    compute_floor = 16**2 * np.linspace(1, 500 * T, T)
    plt.plot(
        compute_floor,
        loss_inf * np.ones(T),
        "--",
        color="red",
        label=r"$L \to \infty$",
    )
    plt.ylim([0.2, 1.05])
    plt.legend()
    plt.xlabel(r"Compute", fontsize=20)
    plt.ylabel(r"Loss", fontsize=20)
    plt.savefig(
        output_dir / "compute_scaling_fixed_N_more_depth.png",
        dpi=200,
        bbox_inches="tight",
    )
    print("Saved compute_scaling_fixed_N_more_depth.png")

    # --- Plot 2: raw loss vs steps ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        plt.loglog(all_loss_lin_scale[i], label=f"$L = {L}$")
    plt.legend()
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"Loss", fontsize=20)
    plt.savefig(
        output_dir / "depth_sweep_raw_loss.png", dpi=200, bbox_inches="tight"
    )
    print("Saved depth_sweep_raw_loss.png")

    # Print loss_final for reference
    loss_final = float(torch.sum((1.0 - spec) ** 2 * w_star**2 * spec).cpu())
    print(f"loss_final = {loss_final}")

    # --- Save data ---
    npz_path = output_dir / "compute_scaling_depth_data.npz"
    payload: dict[str, np.ndarray] = {
        "lvals": np.asarray(lvals, dtype=np.int64),
        "loss_inf": np.asarray([loss_inf], dtype=np.float64),
        "loss_final": np.asarray([loss_final], dtype=np.float64),
        "spec": spec.cpu().numpy(),
        "w_star": w_star.cpu().numpy(),
    }
    for i, L in enumerate(lvals):
        payload[f"loss_L_{L}"] = np.asarray(all_loss_lin_scale[i], dtype=np.float64)
    np.savez(npz_path, **payload)
    print(f"Saved data to: {npz_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
