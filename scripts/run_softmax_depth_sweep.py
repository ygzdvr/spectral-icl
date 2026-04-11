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

from configs import DecoupledTrainModelConfig
from dynamics import make_normalized_powerlaw_problem, train_model_softmax
from utils import OutputDir, parse_int_list


def main() -> None:
    """Run a softmax-attention depth sweep and save plots/loss arrays.

    The script builds a fixed power-law problem, trains one model per depth,
    visualizes training trajectories, and writes structured `.npz` outputs.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook softmax attention depth sweep."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--p-train", type=int, default=32)
    parser.add_argument("--p-test", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lamb", type=float, default=0.0)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.45)
    parser.add_argument("--lvals", type=str, default="1,2,4")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    out = OutputDir(__file__, base=args.output_dir)

    lvals = parse_int_list(args.lvals)
    spec, w_star = make_normalized_powerlaw_problem(
        args.d, args.alpha, args.beta, device=device, dtype=dtype
    )

    all_losses: list[list[float]] = []
    for L in lvals:
        cfg = DecoupledTrainModelConfig(
            d=args.d,
            P_tr=args.p_train,
            P_test=args.p_test,
            B=args.batch,
            N=args.n,
            L=L,
            beta_model=args.beta_model,
            gamma=args.gamma,
            T=args.steps,
            lr=args.lr,
            lamb=args.lamb,
            alpha=args.alpha,
            beta=args.beta,
            sigma=args.sigma,
            random_rotate=False,
            unrestricted=False,
            online=True,
            sample_mode="spec",
        )
        losses, _ = train_model_softmax(cfg, spec=spec, w_star=w_star, device=device, dtype=dtype)
        all_losses.append(losses)

    # --- Plot ---
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    plt.figure()
    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    for i, L in enumerate(lvals):
        c = colors[i % len(colors)]
        plt.plot(all_losses[i], color=c, label=f"L = {L}")
    plt.legend()
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"Loss", fontsize=20)

    plt.savefig(out.png("softmax_depth_sweep"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("softmax_depth_sweep"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out.png('softmax_depth_sweep')}")

    # --- Save ---
    payload: dict[str, np.ndarray] = {"lvals": np.asarray(lvals, dtype=np.int64)}
    for i, L in enumerate(lvals):
        payload[f"loss_L_{L}"] = np.asarray(all_losses[i], dtype=np.float64)

    # Final losses (last 10 steps averaged)
    final_losses = np.array(
        [np.mean(all_losses[i][-10:]) for i in range(len(lvals))], dtype=np.float64
    )
    payload["final_losses"] = final_losses
    print(f"Final losses: {final_losses}")

    np.savez(out.numpy("softmax_depth_sweep_losses"), **payload)
    print(f"Saved losses to: {out.numpy('softmax_depth_sweep_losses')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
