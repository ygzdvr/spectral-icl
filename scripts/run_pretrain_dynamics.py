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

from dynamics import pretrain_dynamics, pretrain_dynamics_two_var
from utils import make_powerlaw_spec_and_wstar, parse_int_list


def main() -> None:
    """Run toy pretraining dynamics (single-var and six-var variants).

    This script reproduces notebook-style depth scaling analyses by launching
    both dynamics models, plotting trajectory/statistics diagnostics, and
    serializing full loss/parameter histories.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook toy-model pretrain dynamics (single & six-var)."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--w0-single", type=float, default=0.5)
    parser.add_argument("--w0-two", type=float, default=0.25)
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16,32,64,128")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--losses-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    M = args.m
    alpha = args.alpha
    beta = args.beta
    T = args.steps

    spec, w_star = make_powerlaw_spec_and_wstar(
        M,
        alpha,
        beta,
        device=device,
        dtype=dtype,
        normalize_w_star=False,
    )

    lvals = parse_int_list(args.lvals)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # 1) Six-variable dynamics (pretrain_dynamics_two_var)
    # =====================================================================
    print("=== Running six-variable dynamics ===")
    two_var_results: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
    for L in lvals:
        print(f"  L = {L}")
        losses, param_hist = pretrain_dynamics_two_var(
            spec, w_star, beta0=args.beta0, L=L, T=T, eta=args.eta, w0=args.w0_two
        )
        two_var_results.append((losses, param_hist))

    # --- Plot 1: loss trajectories + theory t^{-5β/(5β+2)} ---
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    sns.set_palette("rocket", n_colors=len(lvals))

    th_exp_two = 5 * beta / (5 * beta + 2)
    t_axis = np.linspace(10, T, T)

    plt.figure()
    for i, L in enumerate(lvals):
        plt.loglog(two_var_results[i][0].cpu().numpy())
    plt.loglog(t_axis, t_axis ** (-th_exp_two), "--", color="blue")
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"Loss", fontsize=20)
    plt.savefig(output_dir / "two_var_loss_trajectories.png", dpi=200, bbox_inches="tight")
    print(f"  Saved two_var_loss_trajectories.png")

    # --- Plot 2: w_y(t) * w_o(t) ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        ph = two_var_results[i][1]
        wy_wo = (ph[1] * ph[5]).cpu().numpy()
        plt.semilogx(wy_wo)
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$w_o(t) w_y(t)$", fontsize=20)
    plt.savefig(output_dir / "two_var_wy_wo.png", dpi=200, bbox_inches="tight")
    print(f"  Saved two_var_wy_wo.png")

    # --- Plot 3: w_x(t)^2 * w_k(t) * w_q(t) * w_v(t) + theory ---
    th_exp_param = 5.0 / (5 * beta + 2)

    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        ph = two_var_results[i][1]
        combo = (ph[0] ** 2 * ph[2] * ph[3] * ph[4]).cpu().numpy()
        plt.loglog(combo)
    plt.loglog(
        t_axis,
        0.25 * t_axis ** th_exp_param,
        "--",
        color="blue",
        label=r"$t^{5 / (5\beta + 2)}$",
    )
    plt.legend()
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$w_x(t)^2 w_k(t) w_q(t) w_v(t)$", fontsize=20)
    plt.savefig(output_dir / "two_var_wx2_wk_wq_wv.png", dpi=200, bbox_inches="tight")
    print(f"  Saved two_var_wx2_wk_wq_wv.png")

    # =====================================================================
    # 2) Single-variable dynamics (pretrain_dynamics)
    # =====================================================================
    print("=== Running single-variable dynamics ===")
    train_loss: list[torch.Tensor] = []
    for L in lvals:
        print(f"  L = {L}")
        losses = pretrain_dynamics(
            spec, w_star, beta0=args.beta0, L=L, T=T, eta=args.eta, w0=args.w0_single
        )
        train_loss.append(losses)

    print(f"alpha = {alpha} , beta = {beta}")
    th_exp_single = 7 * beta / (2 + 7 * beta)
    print(f"new exp = {th_exp_single}")

    # --- Plot 4: single-var loss trajectories + theory lines ---
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    sns.set_palette("rocket", n_colors=len(lvals))

    last_len = len(train_loss[-1])
    t_axis_single = np.linspace(10, last_len, last_len)

    plt.figure()
    for i, L in enumerate(lvals):
        plt.loglog(train_loss[i].cpu().numpy(), label=f"$L = {L}$")
    plt.loglog(
        t_axis_single,
        0.3 * t_axis_single ** (-th_exp_single),
        "--",
        color="blue",
        label=r"$t^{-7\beta/(7\beta+2)}$",
    )
    plt.loglog(
        t_axis_single,
        1.2 * t_axis_single ** (-beta),
        "--",
        color="red",
        label=r"$t^{-\beta}$",
    )
    plt.xlabel(r"Pretrain steps", fontsize=20)
    plt.ylabel(r"Test Loss", fontsize=20)
    plt.legend()
    plt.ylim([1e-5, 2.0])
    plt.tight_layout()
    plt.savefig(
        output_dir / "depth_scaling_pretrain_theory_ICL_powerlaw.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(f"  Saved depth_scaling_pretrain_theory_ICL_powerlaw.png")

    # --- Plot 5: final loss vs L ---
    final_losses = [float(loss_i[-1].cpu()) for loss_i in train_loss]
    # Exclude last 2 L values for fit (matches notebook: Lvals[:-2])
    fit_lvals = np.asarray(lvals[:-2], dtype=np.float64)
    fit_finals = final_losses[: len(lvals) - 2]

    plt.figure()
    plt.loglog(fit_lvals, fit_finals, "-o")
    plt.loglog(
        fit_lvals,
        0.1 * fit_lvals ** (-beta),
        "--",
        color="blue",
        label=r"$L^{-\beta}$",
    )
    plt.legend()
    plt.xlabel(r"$L$", fontsize=20)
    plt.ylabel(r"$\lim_{t \to \infty} \  \mathcal{L}(t,L)$", fontsize=20)
    plt.savefig(
        output_dir / "depth_scaling_final_loss_powerlaw.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(f"  Saved depth_scaling_final_loss_powerlaw.png")

    # =====================================================================
    # Save losses
    # =====================================================================
    losses_path = (
        Path(args.losses_path)
        if args.losses_path
        else output_dir / "pretrain_dynamics_losses.npz"
    )
    npz_payload: dict[str, np.ndarray] = {
        "lvals": np.asarray(lvals, dtype=np.int64),
        "spec": spec.cpu().numpy(),
        "w_star": w_star.cpu().numpy(),
    }
    for i, L in enumerate(lvals):
        npz_payload[f"single_loss_L_{L}"] = train_loss[i].cpu().numpy()
        npz_payload[f"two_var_loss_L_{L}"] = two_var_results[i][0].cpu().numpy()
        for k, name in enumerate(["wx", "wy", "wk", "wq", "wv", "wo"]):
            npz_payload[f"two_var_{name}_L_{L}"] = two_var_results[i][1][k].cpu().numpy()
    np.savez(losses_path, **npz_payload)
    print(f"Saved losses to: {losses_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
