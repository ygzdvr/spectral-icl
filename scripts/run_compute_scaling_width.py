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
    """Run width-only compute-scaling experiments at fixed depth.

    Sweeps model width, plots loss vs compute, marks asymptotic floor behavior,
    and saves all curves with problem tensors.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook width-only compute scaling (fixed L=1, vary N)."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=180)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--nvals", type=str, default="16,32,64,128,256")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    M = args.m
    alpha = args.alpha
    beta = args.beta
    T = args.steps
    L = args.l

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec, w_star = make_powerlaw_spec_and_wstar(
        M,
        alpha,
        beta,
        device=device,
        dtype=dtype,
    )

    Nvals = parse_int_list(args.nvals)

    # =====================================================================
    # Sweep: fixed L, vary N
    # =====================================================================
    all_loss_no_scale: list[list[float]] = []
    for N in Nvals:
        print(f"N = {N}")
        losses, _, _ = reduced_gamma_structured_sgd_rmt_isotropic_dynamics(
            spec, w_star, N=N, L=L, B=args.batch, K=args.k, P=args.p,
            T=T, eta=args.eta, lamb=args.lamb, ctx_sample=False,
            device=args.device, dtype=dtype,
        )
        all_loss_no_scale.append(losses)

    # N -> infinity loss floor: sum( (1 - 1.1*spec)^2 * w_star^2 * spec )
    loss_final = float(torch.sum((1.0 - 1.1 * spec) ** 2 * w_star**2 * spec).cpu())
    print(f"loss_final (N -> inf) = {loss_final}")

    t_steps = np.linspace(1, T, T)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    plt.figure()
    sns.set_palette("rocket", n_colors=len(Nvals))
    for i, N in enumerate(Nvals):
        compute = N**2 * t_steps
        plt.loglog(compute, all_loss_no_scale[i], label=f"$N = {N}$")

    # N -> infinity floor line
    compute_floor = 16**2 * np.linspace(1, 500 * T, T)
    plt.plot(
        compute_floor,
        loss_final * np.ones(T),
        "--",
        color="red",
        label=r"$N \to \infty$",
    )
    plt.ylim([0.2, 1.05])
    plt.legend()
    plt.xlabel(r"Compute", fontsize=20)
    plt.ylabel(r"Loss", fontsize=20)
    plt.savefig(
        output_dir / "compute_scaling_fixed_L_more_width.png",
        dpi=200,
        bbox_inches="tight",
    )
    print("Saved compute_scaling_fixed_L_more_width.png")

    # --- Save data ---
    npz_path = output_dir / "compute_scaling_width_data.npz"
    payload: dict[str, np.ndarray] = {
        "Nvals": np.asarray(Nvals, dtype=np.int64),
        "loss_final": np.asarray([loss_final], dtype=np.float64),
        "spec": spec.cpu().numpy(),
        "w_star": w_star.cpu().numpy(),
    }
    for i, N in enumerate(Nvals):
        payload[f"loss_N_{N}"] = np.asarray(all_loss_no_scale[i], dtype=np.float64)
    np.savez(npz_path, **payload)
    print(f"Saved data to: {npz_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
