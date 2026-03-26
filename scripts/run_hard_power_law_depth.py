import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Allow running the script directly: `python scripts/run_hard_power_law_depth.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import HardPowerLawDepthConfig
from dynamics import run_hard_power_law_depth_eval
from utils import is_cuda_oom, resolve_device


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)

def main() -> None:
    """Evaluate hard power-law depth behavior for the hand-coded model.

    Configures and runs the depth experiment, handles optional CUDA fallback,
    and writes plots/loss files/artifacts for post-hoc analysis.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook section: hard power law convergence in depth."
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--layers", type=int, default=40, help="Base L before evaluation uses 5*L.")
    parser.add_argument("--p-train", type=int, default=120)
    parser.add_argument("--p-test", type=int, default=40)
    parser.add_argument("--beta-model", type=float, default=500.0)
    parser.add_argument("--exp-value", type=float, default=1.0, help="Constant exponent in the power-law data.")
    parser.add_argument("--seed-x", type=int, default=1)
    parser.add_argument("--seed-beta", type=int, default=2)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--losses-path", type=str, default=None)
    parser.add_argument("--artifacts-path", type=str, default=None, help="Path to save tensors/results as .pt.")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    cfg = HardPowerLawDepthConfig(
        d=args.d,
        B=args.batch,
        L=args.layers,
        P=args.p_train,
        P_test=args.p_test,
        beta_model=args.beta_model,
        exp_value=args.exp_value,
        seed_x=args.seed_x,
        seed_beta=args.seed_beta,
    )
    device = resolve_device(args.device)

    try:
        out, train_losses, test_losses, X, y, powers = run_hard_power_law_depth_eval(
            cfg,
            device=device,
            dtype=dtype,
        )
    except (RuntimeError, torch.AcceleratorError) as exc:
        if not device.startswith("cuda") or not is_cuda_oom(exc):
            raise
        print(
            f"CUDA out of memory on device '{device}'. "
            "Retrying on CPU. Pass --device cpu to skip this retry."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = "cpu"
        out, train_losses, test_losses, X, y, powers = run_hard_power_law_depth_eval(
            cfg,
            device=device,
            dtype=dtype,
        )

    print(f"device: {device}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"powers shape: {tuple(powers.shape)}")
    print(f"out shape: {tuple(out.shape)}")
    print(f"effective eval layers: {5 * cfg.L}")

    steps = np.linspace(1.0, float(len(train_losses)), len(train_losses), dtype=np.float64)

    sns.set_palette("rocket", n_colors=2)
    plt.title(r"Hand Coded ICL Loss", fontsize=20)
    plt.loglog(steps, np.asarray(train_losses, dtype=np.float64), label="train")
    plt.loglog(steps, np.asarray(test_losses, dtype=np.float64), label="test")
    plt.xlabel(r"Layer", fontsize=20)
    plt.ylabel(r"$|h^\ell_{tr,te}|^2$", fontsize=20)

    plot_path = (
        Path(args.plot_path)
        if args.plot_path
        else PROJECT_ROOT / "outputs" / "hard_power_law_depth_icl_loss.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")

    losses_path = (
        Path(args.losses_path)
        if args.losses_path
        else PROJECT_ROOT / "outputs" / "hard_power_law_depth_icl_losses.npz"
    )
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        losses_path,
        steps=steps,
        train=np.asarray(train_losses, dtype=np.float64),
        test=np.asarray(test_losses, dtype=np.float64),
    )
    print(f"Saved losses to: {losses_path}")

    artifacts_path = (
        Path(args.artifacts_path)
        if args.artifacts_path
        else PROJECT_ROOT / "outputs" / "hard_power_law_depth_artifacts.pt"
    )
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": vars(args),
            "device_used": device,
            "effective_eval_layers": 5 * cfg.L,
            "out": out.detach().cpu(),
            "X": X.detach().cpu(),
            "y": y.detach().cpu(),
            "powers": powers.detach().cpu(),
            "train_losses": torch.tensor(train_losses, dtype=torch.float64),
            "test_losses": torch.tensor(test_losses, dtype=torch.float64),
        },
        artifacts_path,
    )
    print(f"Saved artifacts to: {artifacts_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
