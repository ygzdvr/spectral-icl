import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamics import train_model_dim_free, sample_data_spec_rotate_bernoulli
from utils import make_powerlaw_spec_and_wstar, parse_int_list


def main() -> None:
    """Run dimension-free linear-attention dynamics across depth values.

    Launches training for each configured depth, compares to scaling-law
    references, visualizes weight dynamics, and persists all results.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook dim-free linear attention dynamics."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=80)
    parser.add_argument("--p-train", type=int, default=256)
    parser.add_argument("--p-test", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.125)
    parser.add_argument("--lamb", type=float, default=0.0)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.45)
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    d = args.d
    N = d
    T = args.steps
    lr = args.lr
    alpha = args.alpha
    beta = args.beta
    sqrt_d = math.sqrt(d)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec, w_star = make_powerlaw_spec_and_wstar(
        d,
        alpha,
        beta,
        device=device,
        dtype=dtype,
    )

    print(f"sum of variance: {float(torch.sum(w_star**2 * spec).cpu())}")

    lvals = parse_int_list(args.lvals)

    # =====================================================================
    # Train dim-free model for each L
    # =====================================================================
    all_losses: list[list[float]] = []
    all_weight_norms: list[list[list[float]]] = []

    for L in lvals:
        print(f"\nL = {L}")
        pretrain_loss, weight_norms = train_model_dim_free(
            d=d, P_tr=args.p_train, P_test=args.p_test, B=args.batch,
            N=N, L=L, beta=args.beta_model, gamma=args.gamma,
            T=T, lr=lr * sqrt_d, lamb=args.lamb,
            spec=spec, w_star=w_star, sigma=args.sigma,
            online=True, device=args.device, dtype=dtype,
        )
        all_losses.append(pretrain_loss)
        all_weight_norms.append(weight_norms)

    # Check data variance
    X, y = sample_data_spec_rotate_bernoulli(
        spec, w_star, args.batch, args.p_train, args.p_test,
        seed=0, device=args.device, dtype=dtype,
    )
    print(f"\nmean(y^2) = {float(torch.mean(y**2).cpu())}")

    t_axis = np.linspace(10, T, T)
    th_exp = 5 * beta / (5 * beta + 2)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    # --- Plot 1: loss trajectories + theory ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        plt.loglog(all_losses[i], label=f"$L = {L}$")
    plt.loglog(
        t_axis,
        t_axis ** (-th_exp),
        "--",
        color="red",
        label=r"$t^{- 5\beta/(5\beta+2)}$",
    )
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t)$", fontsize=20)
    plt.legend()
    plt.savefig(
        output_dir / "dim_free_theory_weight_decoupled_vs_expt.png",
        dpi=200,
        bbox_inches="tight",
    )
    print("Saved dim_free_theory_weight_decoupled_vs_expt.png")

    # --- Plot 2: raw weight norms (last L) ---
    weight_arr = np.array(all_weight_norms[-1])
    plt.figure()
    for i in range(4):
        plt.plot(weight_arr[:, i])
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$|W_i(t)|$", fontsize=20)
    plt.savefig(output_dir / "dim_free_weight_norms_raw.png", dpi=200, bbox_inches="tight")
    print("Saved dim_free_weight_norms_raw.png")

    # --- Plot 3: normalized weight norms ---
    plt.figure()
    sns.set_palette("rocket", n_colors=4)
    labels = [r"$W_x$", r"$W_q$", r"$W_k$", r"$W_v$"]
    for i in range(4):
        vals = weight_arr[:, i].copy()
        if i == 0:
            vals = vals / math.sqrt(2.0)
        elif i == 3:
            vals = vals / sqrt_d
        plt.plot(vals, label=labels[i])
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$|W_i(t)|$", fontsize=20)
    plt.legend()
    plt.savefig(output_dir / "dim_free_weight_norms_normalized.png", dpi=200, bbox_inches="tight")
    print("Saved dim_free_weight_norms_normalized.png")

    # --- Save data ---
    npz_path = output_dir / "dim_free_dynamics_data.npz"
    payload: dict[str, np.ndarray] = {
        "lvals": np.asarray(lvals, dtype=np.int64),
        "spec": spec.cpu().numpy(),
        "w_star": w_star.cpu().numpy(),
    }
    for i, L in enumerate(lvals):
        payload[f"loss_L_{L}"] = np.array(all_losses[i], dtype=np.float64)
        payload[f"weight_norms_L_{L}"] = np.array(all_weight_norms[i], dtype=np.float64)
    np.savez(npz_path, **payload)
    print(f"Saved data to: {npz_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
