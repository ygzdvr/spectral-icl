import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch


if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import OutputDir, parse_float_list, run_experiment


def main() -> None:
    """Launch isotropic SGD sweeps across one or more alpha values.

    The script parses comma-separated tau/alpha lists, executes each alpha
    experiment, and emits reusable plot/data artifacts.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook simple SGD isotropic dynamics."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--taus", type=str, default="0.05,0.1,0.2,0.4")
    parser.add_argument("--alphas", type=str, default="0.5,2.0")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    taus = parse_float_list(args.taus)
    alphas = parse_float_list(args.alphas)

    out = OutputDir(__file__, base=args.output_dir)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    for alpha in alphas:
        run_experiment(args.d, alpha, taus, args.eta, args.steps, out, args.device, dtype)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
