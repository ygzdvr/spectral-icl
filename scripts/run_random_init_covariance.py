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
from utils import OutputDir


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
    parser.add_argument("--output-dir", type=str, default=None)
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

    odir = OutputDir(__file__, base=args.output_dir)

    result, train_losses, test_losses, X, y, powers = run_random_init_covariance_eval(
        cfg,
        device=device,
        dtype=dtype,
    )

    print(f"device: {device}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"powers shape: {tuple(powers.shape)}")
    print(f"out shape: {tuple(result.shape)}")

    plt.plot(train_losses, label="train")
    plt.plot(np.asarray(test_losses), label="test")
    plt.legend()

    plt.savefig(odir.png("covariance"), dpi=200, bbox_inches="tight")
    plt.savefig(odir.pdf("covariance"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {odir.png('covariance')}")

    np.savez(
        odir.numpy("losses"),
        train=np.asarray(train_losses, dtype=np.float64),
        test=np.asarray(test_losses, dtype=np.float64),
    )
    print(f"Saved losses to: {odir.numpy('losses')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
