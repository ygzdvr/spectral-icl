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

from dynamics import reduced_gamma_decoupled_depth_structured_sgd_dynamics
from utils import make_powerlaw_spec_and_wstar, parse_int_list


def main() -> None:
    """Run decoupled reduced-gamma dynamics across depth values.

    Computes depth-conditioned trajectories, overlays expected scaling curves,
    and writes plot/data outputs for downstream analysis.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook decoupled-layer Gamma dynamics."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eta", type=float, default=4.0)
    parser.add_argument("--lamb", type=float, default=1e-12)
    parser.add_argument("--lvals", type=str, default="1,2,4,8")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    M = args.m
    alpha = args.alpha
    beta = args.beta
    T = args.steps

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec, w_star = make_powerlaw_spec_and_wstar(
        M,
        alpha,
        beta,
        device=device,
        dtype=dtype,
    )

    lvals = parse_int_list(args.lvals)

    all_loss: list[list[float]] = []
    for L in lvals:
        print(f"L = {L}")
        losses, _, _ = reduced_gamma_decoupled_depth_structured_sgd_dynamics(
            spec, w_star, N=args.n, L=L, B=args.batch, K=args.k, P=args.p,
            T=T, eta=args.eta, lamb=args.lamb, ctx_sample=False,
            device=args.device, dtype=dtype,
        )
        all_loss.append(losses)

    t_axis = np.linspace(10, T, T)
    th_exp = beta / (beta + 2)

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")

    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        plt.loglog(all_loss[i], label=f"$L = {L}$")
    plt.loglog(
        t_axis,
        t_axis ** (-th_exp),
        "--",
        color="red",
        label=r"$t^{-\beta/(\beta + 2)}$",
    )
    plt.legend()
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t)$", fontsize=20)
    plt.savefig(
        output_dir / "gamma_decoupled_layers_dynamics.png",
        dpi=200,
        bbox_inches="tight",
    )
    print("Saved gamma_decoupled_layers_dynamics.png")

    # --- Save data ---
    npz_path = output_dir / "decoupled_layers_data.npz"
    payload: dict[str, np.ndarray] = {"lvals": np.asarray(lvals, dtype=np.int64)}
    for i, L in enumerate(lvals):
        payload[f"loss_L_{L}"] = np.asarray(all_loss[i], dtype=np.float64)
    np.savez(npz_path, **payload)
    print(f"Saved data to: {npz_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
