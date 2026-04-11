import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


_PROJ = str(Path(__file__).resolve().parents[1])
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from dynamics import pretrain_dynamics_two_var
from utils import make_powerlaw_spec_and_wstar, parse_float_list, OutputDir


def main() -> None:
    """Run toy two-variable dynamics across beta values.

    For each beta, trains dynamics, plots key parameter combinations and losses,
    and stores complete trajectory histories in `.npz` form.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook beta-sweep two-var dynamics (cells 62-63)."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--l", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--eta", type=float, default=0.005)
    parser.add_argument("--w0", type=float, default=0.25)
    parser.add_argument("--beta-vals", type=str, default="0.5,0.75,1.1,1.25,1.5,2.0")
    parser.add_argument("--multipliers", type=str, default="0.1,0.1,0.2,0.5,0.8,1.0")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    M = args.m
    alpha = args.alpha
    L = args.l
    T = args.steps
    beta_vals = parse_float_list(args.beta_vals)
    multipliers = parse_float_list(args.multipliers)

    out = OutputDir(__file__, base=args.output_dir)

    # Run two-var dynamics for each beta
    train_loss_and_params: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
    for i, beta in enumerate(beta_vals):
        print(f"beta = {beta}")
        spec, w_star = make_powerlaw_spec_and_wstar(
            M,
            alpha,
            beta,
            device=device,
            dtype=dtype,
        )

        losses, param_hist = pretrain_dynamics_two_var(
            spec, w_star, beta0=1.0, L=L, T=T, eta=args.eta, w0=args.w0
        )
        train_loss_and_params.append((losses, param_hist))

    t_axis = np.linspace(10, T, T)

    # --- Plot 1: w_y(t) * w_o(t) ---
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    sns.set_palette("rocket", n_colors=len(beta_vals))

    plt.figure()
    for i, beta in enumerate(beta_vals):
        ph = train_loss_and_params[i][1]
        wy_wo = (ph[1] * ph[5]).cpu().numpy()
        plt.semilogx(wy_wo)
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$w_o(t) w_y(t)$", fontsize=20)
    plt.savefig(out.png("beta_sweep_wy_wo"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("beta_sweep_wy_wo"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('beta_sweep_wy_wo')}")

    # --- Plot 2: w_x^2 * w_k * w_q * w_v + per-beta theory lines ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(beta_vals))
    for i, beta in enumerate(beta_vals):
        ph = train_loss_and_params[i][1]
        combo = (ph[0] ** 2 * ph[2] * ph[3] * ph[4]).cpu().numpy()
        plt.loglog(combo, label=rf"$\beta = {beta:.2f}$")
        th_exp = 5.0 / (5 * beta + 2)
        mult = multipliers[i] if i < len(multipliers) else 1.0
        plt.loglog(t_axis, 0.25 * mult * t_axis**th_exp, "--", color="blue")
    plt.legend()
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$w_x(t)^2 w_k(t) w_q(t) w_v(t)$", fontsize=20)
    plt.savefig(out.png("beta_sweep_wx2_wk_wq_wv"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("beta_sweep_wx2_wk_wq_wv"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('beta_sweep_wx2_wk_wq_wv')}")

    # --- Plot 3: loss trajectories ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(beta_vals))
    for i, beta in enumerate(beta_vals):
        plt.loglog(train_loss_and_params[i][0].cpu().numpy())
    plt.savefig(out.png("beta_sweep_loss"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("beta_sweep_loss"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('beta_sweep_loss')}")

    # --- Save losses ---
    npz_payload: dict[str, np.ndarray] = {
        "beta_vals": np.asarray(beta_vals, dtype=np.float64),
    }
    for i, beta in enumerate(beta_vals):
        tag = f"beta_{beta:.2f}"
        npz_payload[f"loss_{tag}"] = train_loss_and_params[i][0].cpu().numpy()
        for k, name in enumerate(["wx", "wy", "wk", "wq", "wv", "wo"]):
            npz_payload[f"{name}_{tag}"] = train_loss_and_params[i][1][k].cpu().numpy()
    np.savez(out.numpy("beta_sweep_dynamics_losses"), **npz_payload)
    print(f"Saved losses to: {out.numpy('beta_sweep_dynamics_losses')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
