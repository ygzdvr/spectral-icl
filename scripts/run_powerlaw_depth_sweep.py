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
from dynamics import run_powerlaw_depth_sweep
from utils import moving_average, parse_int_list


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)


def main() -> None:
    """Run rotated power-law depth sweeps and collect training trajectories.

    The script constructs per-depth configs, executes the sweep runner,
    visualizes smoothed losses, and stores reusable numeric artifacts.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook power-law depth sweep with rotation."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--p-train", type=int, default=30)
    parser.add_argument("--p-test", type=int, default=25)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.8)
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--lamb", type=float, default=1.0e-4)
    parser.add_argument("--beta-model", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--losses-path", type=str, default=None)
    parser.add_argument("--artifacts-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    requested_device = torch.device(args.device)
    print(f"requested device: {requested_device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if requested_device.type == "cuda" and torch.cuda.is_available():
        current_idx = (
            requested_device.index
            if requested_device.index is not None
            else torch.cuda.current_device()
        )
        print(f"using cuda device: cuda:{current_idx}")
        print(f"cuda device name: {torch.cuda.get_device_name(current_idx)}")
    elif requested_device.type == "cuda":
        print("CUDA was requested but is not available.")

    lvals = parse_int_list(args.lvals)

    cfg = DecoupledTrainModelConfig(
        d=args.d,
        P_tr=args.p_train,
        P_test=args.p_test,
        B=args.batch,
        N=args.n,
        L=lvals[0],
        beta_model=args.beta_model,
        gamma=args.gamma,
        T=args.steps,
        lr=args.lr,
        lamb=args.lamb,
        alpha=args.alpha,
        beta=args.beta,
        sigma=args.sigma,
        random_rotate=True,
        unrestricted=False,
        online=True,
        sample_mode="spec_rotate",
    )

    results = run_powerlaw_depth_sweep(
        cfg, lvals, normalize_spec=False, device=args.device, dtype=dtype
    )
    all_losses = results["all_losses"]

    # --- Smoothed pretrain loss per depth ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(all_losses))
    for i, loss in enumerate(all_losses):
        loss_np = np.asarray(loss, dtype=np.float64)
        smooth = moving_average(loss_np, 100)
        plt.loglog(smooth, label=f"$L = {lvals[i]}$")
    plt.ylim([5e-2, 1.5])
    plt.xlabel(r"Steps", fontsize=20)
    plt.ylabel(r"Pretrain loss", fontsize=20)
    plt.legend()

    plot_path = (
        Path(args.plot_path)
        if args.plot_path
        else PROJECT_ROOT / "outputs" / "depth_scaling_ICL_powerlaw.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")

    # --- Save losses ---
    losses_path = (
        Path(args.losses_path)
        if args.losses_path
        else PROJECT_ROOT / "outputs" / "depth_scaling_ICL_powerlaw_losses.npz"
    )
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    npz_payload: dict[str, np.ndarray] = {"lvals": np.asarray(lvals, dtype=np.int64)}
    for i, L in enumerate(lvals):
        npz_payload[f"loss_L_{L}"] = np.asarray(all_losses[i], dtype=np.float64)
    np.savez(losses_path, **npz_payload)
    print(f"Saved losses to: {losses_path}")

    # --- Save artifacts ---
    artifacts_path = (
        Path(args.artifacts_path)
        if args.artifacts_path
        else PROJECT_ROOT / "outputs" / "depth_scaling_ICL_powerlaw_artifacts.pt"
    )
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": vars(args),
            "lvals": lvals,
            "spec": results["spec"].detach().cpu(),
            "w_star": results["w_star"].detach().cpu(),
            "all_losses": [torch.tensor(l, dtype=torch.float64) for l in all_losses],
        },
        artifacts_path,
    )
    print(f"Saved artifacts to: {artifacts_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
