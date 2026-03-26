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

from utils import run_sweep


def main() -> None:
    """Execute all configured RMT isotropic sweep experiments.

    This CLI entry point runs multiple predefined sweeps, each targeting a
    different control variable, and writes per-sweep figures plus loss dumps.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook simple SGD RMT isotropic dynamics."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    # Experiment 1: vary tau, alpha=1.0, kappa=4.0
    run_sweep(
        name="vary_tau_alpha1p0_kappa4p0",
        sweep_var="tau",
        sweep_vals=[0.1, 0.2, 0.4, 0.8, 1.6],
        d=args.d, tau_fixed=0.1, alpha_fixed=1.0, kappa_fixed=4.0,
        eta=args.eta, T=args.steps,
        output_dir=output_dir, device=args.device, dtype=dtype,
    )

    # Experiment 2: vary alpha, tau=1.0, kappa=4.0
    run_sweep(
        name="vary_alpha_tau1p0_kappa4p0",
        sweep_var="alpha",
        sweep_vals=[0.1, 0.2, 0.4, 0.8, 1.6],
        d=args.d, tau_fixed=1.0, alpha_fixed=1.0, kappa_fixed=4.0,
        eta=args.eta, T=args.steps,
        output_dir=output_dir, device=args.device, dtype=dtype,
    )

    # Experiment 3: vary kappa, tau=1.0, alpha=1.0
    run_sweep(
        name="vary_kappa_tau1p0_alpha1p0",
        sweep_var="kappa",
        sweep_vals=[0.1, 0.2, 0.4, 0.8, 1.6],
        d=args.d, tau_fixed=1.0, alpha_fixed=1.0, kappa_fixed=4.0,
        eta=args.eta, T=args.steps,
        output_dir=output_dir, device=args.device, dtype=dtype,
    )

    # Experiment 4: vary tau, alpha=8.0, kappa=2.0 (semilogy)
    run_sweep(
        name="vary_tau_alpha8p0_kappa2p0",
        sweep_var="tau",
        sweep_vals=[0.1, 0.2, 0.4, 0.8, 1.6],
        d=args.d, tau_fixed=0.1, alpha_fixed=8.0, kappa_fixed=2.0,
        eta=args.eta, T=args.steps,
        output_dir=output_dir, device=args.device, dtype=dtype,
        use_semilogy=True,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
