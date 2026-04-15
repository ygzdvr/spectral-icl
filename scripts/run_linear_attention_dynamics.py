# run_linear_attention_dynamics.py
import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


_PROJ = str(Path(__file__).resolve().parents[1])
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from dynamics import (
    train_model_isotropic,
    reduced_theory_four_var_linear_att_isotropic,
)
from utils import parse_int_list, OutputDir


def main() -> None:
    """Compare isotropic linear-attention training dynamics against theory.

    Trains models across depth values, runs reduced-theory predictions, and
    saves side-by-side diagnostics for losses and weight norms.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook linear attention dynamics with theory comparison."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--p-train", type=int, default=128)
    parser.add_argument("--p-test", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lamb", type=float, default=0.0)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lvals", type=str, default="1,2,4,8")
    parser.add_argument("--lamb-grid-min", type=float, default=0.0025)
    parser.add_argument("--lamb-grid-max", type=float, default=7.0)
    parser.add_argument("--lamb-grid-points", type=int, default=800)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    d = args.d
    N = d  # N = d for isotropic case
    T = args.steps
    lr = args.lr
    sqrt_d = math.sqrt(d)

    out = OutputDir(__file__, base=args.output_dir)

    lvals = parse_int_list(args.lvals)

    # =====================================================================
    # 1) Train isotropic model for each L
    # =====================================================================
    print("=== Training isotropic model ===")
    all_losses: list[list[float]] = []
    all_weight_norms: list[list[list[float]]] = []

    for L in lvals:
        print(f"\nL = {L}")
        pretrain_loss, weight_norms = train_model_isotropic(
            d=d, P_tr=args.p_train, P_test=args.p_test, B=args.batch,
            N=N, L=L, beta=args.beta_model, gamma=args.gamma,
            T=T, lr=lr * sqrt_d, lamb=args.lamb,
            online=True, device=args.device, dtype=dtype,
        )
        all_losses.append(pretrain_loss)
        all_weight_norms.append(weight_norms)

    # =====================================================================
    # 2) Four-variable reduced theory
    # =====================================================================
    print("\n=== Running four-variable reduced theory ===")
    lamb_grid = torch.linspace(
        args.lamb_grid_min, args.lamb_grid_max, args.lamb_grid_points,
        device=device, dtype=dtype,
    )
    alpha = float(args.p_train) / d

    losses_th: list[list[float]] = []
    all_ws: list[list[list[float]]] = []

    for L in lvals:
        print(f"\nL = {L}")
        loss_i, ws_i = reduced_theory_four_var_linear_att_isotropic(
            L, alpha, lamb_grid, eta=lr / sqrt_d, T=T,
        )
        losses_th.append(loss_i)
        all_ws.append(ws_i)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    # --- Plot 1: weight norm dynamics (last L) ---
    plt.figure()
    sns.set_palette("rocket", n_colors=4)
    labels = [r"$W_x$", r"$W_q$", r"$W_k$", r"$W_v$"]

    ws_final = np.array(all_ws[-1])  # (T, 4)
    for i in range(4):
        vals = ws_final[:, i]
        if i == 0:
            vals = vals / math.sqrt(2.0)
        plt.plot(vals, label=labels[i])

    wn_final = np.array(all_weight_norms[-1])  # (T, 4)
    for i in range(4):
        vals = wn_final[:, i]
        if i == 0:
            vals = vals / math.sqrt(2.0)
        elif i == 3:
            vals = vals / sqrt_d
        plt.plot(vals, "--", color="black")

    plt.plot([], [], "--", color="black", label="Theory")
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$|W_i(t)|$", fontsize=20)
    plt.legend()
    plt.savefig(out.png("weight_norm_dynamics"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("weight_norm_dynamics"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('weight_norm_dynamics')}")

    # --- Plot 2: normalized loss comparison (theory vs experiment) ---
    t_axis = np.linspace(1, T, T)

    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        loss_arr = np.array(all_losses[i])
        loss_th_arr = np.array(losses_th[i])
        plt.semilogx(t_axis, loss_arr / loss_arr[0], label=f"$L = {L}$")
        plt.semilogx(t_axis, loss_th_arr / loss_th_arr[0], "--", color="black")

    plt.semilogx([], [], "--", color="black", label="Theory")
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t)$", fontsize=20)
    plt.legend()
    plt.savefig(out.png("isotropic_theory_vs_expt"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("isotropic_theory_vs_expt"), dpi=200, bbox_inches="tight")
    print(f"Saved {out.png('isotropic_theory_vs_expt')}")

    # --- Print final losses ---
    print(f"\nExperiment final losses: {[loss[-1] for loss in all_losses]}")
    print(f"Theory final losses: {[loss[-1] for loss in losses_th]}")

    # --- Save data ---
    payload: dict[str, np.ndarray] = {
        "lvals": np.asarray(lvals, dtype=np.int64),
    }
    for i, L in enumerate(lvals):
        payload[f"loss_L_{L}"] = np.array(all_losses[i], dtype=np.float64)
        payload[f"loss_th_L_{L}"] = np.array(losses_th[i], dtype=np.float64)
        payload[f"weight_norms_L_{L}"] = np.array(all_weight_norms[i], dtype=np.float64)
        payload[f"ws_th_L_{L}"] = np.array(all_ws[i], dtype=np.float64)
    np.savez(out.numpy("linear_attention_dynamics_data"), **payload)
    print(f"Saved data to: {out.numpy('linear_attention_dynamics_data')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
