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

from configs import OODCovarianceEvalConfig
from dynamics import run_ood_covariance_eval
from utils import OutputDir


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)
def main() -> None:
    """Run OOD covariance generalization evaluation for the hand-coded model.

    Configures the OOD experiment, executes evaluation, plots train/test
    trajectories with a reference baseline, and saves loss outputs.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of the notebook section: OOD Generalization over covariance matrices."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--p-train", type=int, default=40)
    parser.add_argument("--p-test", type=int, default=40)
    parser.add_argument("--exp-scale", type=float, default=0.0)
    parser.add_argument("--beta-model", type=float, default=100.0)
    parser.add_argument("--seed-exp", type=int, default=0)
    parser.add_argument("--seed-x", type=int, default=1)
    parser.add_argument("--seed-beta", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device

    cfg = OODCovarianceEvalConfig(
        d=args.d,
        B=args.batch,
        L=args.layers,
        P=args.p_train,
        P_test=args.p_test,
        seed_exp=args.seed_exp,
        seed_x=args.seed_x,
        seed_beta=args.seed_beta,
        exp_scale=args.exp_scale,
        beta_model=args.beta_model,
    )

    odir = OutputDir(__file__, base=args.output_dir)

    result, train_losses, test_losses, X, y, powers = run_ood_covariance_eval(
        cfg,
        device=device,
        dtype=dtype,
    )

    print(f"device: {device}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"powers shape: {tuple(powers.shape)}")
    print(f"out shape: {tuple(result.shape)}")

    ref_curve = (1.0 - cfg.P / cfg.d) * np.ones(len(test_losses), dtype=np.float64)

    plt.title(r"Hand Coded Solution", fontsize=20)
    plt.plot(train_losses)
    plt.plot(np.asarray(test_losses))
    plt.plot(ref_curve, "--", color="green")
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.ylabel(r"$|h^\ell_{tr,te}|^2$", fontsize=20)

    plt.savefig(odir.png("generalization"), dpi=200, bbox_inches="tight")
    plt.savefig(odir.pdf("generalization"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {odir.png('generalization')}")

    np.savez(
        odir.numpy("losses"),
        train=np.asarray(train_losses, dtype=np.float64),
        test=np.asarray(test_losses, dtype=np.float64),
        ref=ref_curve,
    )
    print(f"Saved losses to: {odir.numpy('losses')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
