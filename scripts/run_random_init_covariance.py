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

from configs import RandomInitCovarianceEvalConfig
from dynamics import run_random_init_covariance_eval


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)


def main() -> None:
    """Run random-initialization covariance experiments for hand-coded attention.

    Builds a random-init evaluation config, executes the run, visualizes
    train/test trajectories, and writes reusable losses to disk.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of the notebook section: Randomly initialize Parameters."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--p-train", type=int, default=120)
    parser.add_argument("--p-test", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--beta-model", type=float, default=0.5)
    parser.add_argument("--fixed-exp", type=float, default=1.0)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--losses-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device

    cfg = RandomInitCovarianceEvalConfig(
        d=args.d,
        B=args.batch,
        L=args.layers,
        P=args.p_train,
        P_test=args.p_test,
        sigma=args.sigma,
        beta_model=args.beta_model,
        fixed_exp=args.fixed_exp,
    )

    out, train_losses, test_losses, X, y, powers = run_random_init_covariance_eval(
        cfg,
        device=device,
        dtype=dtype,
    )

    print(f"device: {device}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"powers shape: {tuple(powers.shape)}")
    print(f"out shape: {tuple(out.shape)}")

    plt.plot(train_losses, label="train")
    plt.plot(np.asarray(test_losses), label="test")
    plt.legend()

    plot_path = (
        Path(args.plot_path)
        if args.plot_path
        else PROJECT_ROOT / "outputs" / "random_init_covariance.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")

    losses_path = (
        Path(args.losses_path)
        if args.losses_path
        else PROJECT_ROOT / "outputs" / "random_init_covariance_losses.npz"
    )
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        losses_path,
        train=np.asarray(train_losses, dtype=np.float64),
        test=np.asarray(test_losses, dtype=np.float64),
    )
    print(f"Saved losses to: {losses_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
