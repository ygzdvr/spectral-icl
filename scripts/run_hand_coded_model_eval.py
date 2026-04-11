import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Allow running the script directly: `python scripts/run_hand_coded_model_eval.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import HandCodedEvalConfig
from dynamics import run_hand_coded_eval
from utils import is_cuda_oom, resolve_device, OutputDir


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)

def main() -> None:
    """Evaluate the hand-coded analytical attention model via CLI settings.

    Builds a `HandCodedEvalConfig`, runs the model (with CUDA OOM fallback),
    and saves the resulting trajectory plots and loss arrays.
    """
    parser = argparse.ArgumentParser(description="Torch port of hand-coded model_eval notebook block.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: e.g. auto, cpu, cuda, cuda:0.",
    )
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--seq", type=int, default=80)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--p-test", type=int, default=1)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--seed-x", type=int, default=0)
    parser.add_argument("--seed-beta", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    cfg = HandCodedEvalConfig(
        d=args.d,
        B=args.batch,
        P=args.seq,
        L=args.layers,
        P_test=args.p_test,
        beta=args.beta,
        seed_x=args.seed_x,
        seed_beta=args.seed_beta,
    )
    device = resolve_device(args.device)

    odir = OutputDir(__file__, base=args.output_dir)

    try:
        result, train_losses, test_losses, X, y = run_hand_coded_eval(cfg, device=device, dtype=dtype)
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
        result, train_losses, test_losses, X, y = run_hand_coded_eval(cfg, device=device, dtype=dtype)

    print(f"device: {device}")

    print(f"X shape: {tuple(X.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"out shape: {tuple(result.shape)}")

    plt.plot(train_losses)
    ts = torch.linspace(0, len(train_losses), len(train_losses))
    _ = ts  # intentionally kept to mirror notebook cell structure
    plt.plot(np.asarray(test_losses))
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.ylabel(r"In Context Loss", fontsize=20)

    plt.savefig(odir.png("eval"), dpi=200, bbox_inches="tight")
    plt.savefig(odir.pdf("eval"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {odir.png('eval')}")

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
