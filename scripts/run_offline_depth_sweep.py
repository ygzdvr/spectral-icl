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
from dynamics import run_depth_scaling_nonrotate_sweep
from utils import parse_int_list


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)


def main() -> None:
    """Run offline depth sweeps with paired train/test trajectory tracking.

    Uses fixed data (online=False), compares losses across depths, and saves
    side-by-side visualizations plus serialized artifacts.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook offline depth sweep (train vs test loss)."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--p-train", type=int, default=128)
    parser.add_argument("--p-test", type=int, default=16)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--lamb", type=float, default=1.0e-14)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--lvals", type=str, default="1,2,4,8")
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
        random_rotate=False,
        unrestricted=False,
        online=False,
    )

    results = run_depth_scaling_nonrotate_sweep(cfg, lvals, device=args.device, dtype=dtype)
    all_test = results["all_losses"]
    all_train = results["all_train_losses"]

    # --- side-by-side test / train plot ---
    fig = plt.figure(figsize=(10, 5))
    sns.set_palette("rocket", n_colors=len(lvals))

    plt.subplot(1, 2, 1)
    for i, L in enumerate(lvals):
        plt.loglog(all_test[i], label=f"L={L}", color=f"C{i}")
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t)$", fontsize=20)
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    for i, L in enumerate(lvals):
        plt.loglog(all_train[i], label=f"L={L}", color=f"C{i}")
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\hat{\mathcal{L}}(t)$", fontsize=20)
    plt.legend()
    plt.tight_layout()

    plot_path = (
        Path(args.plot_path)
        if args.plot_path
        else PROJECT_ROOT / "outputs" / "offline_depth_sweep_test_train.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")

    # --- save losses ---
    losses_path = (
        Path(args.losses_path)
        if args.losses_path
        else PROJECT_ROOT / "outputs" / "offline_depth_sweep_losses.npz"
    )
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    npz_payload: dict[str, np.ndarray] = {"lvals": np.asarray(lvals, dtype=np.int64)}
    for i, L in enumerate(lvals):
        npz_payload[f"test_L_{L}"] = np.asarray(all_test[i], dtype=np.float64)
        npz_payload[f"train_L_{L}"] = np.asarray(all_train[i], dtype=np.float64)
    np.savez(losses_path, **npz_payload)
    print(f"Saved losses to: {losses_path}")

    # --- save artifacts ---
    artifacts_path = (
        Path(args.artifacts_path)
        if args.artifacts_path
        else PROJECT_ROOT / "outputs" / "offline_depth_sweep_artifacts.pt"
    )
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": vars(args),
            "lvals": lvals,
            "spec": results["spec"].detach().cpu(),
            "w_star": results["w_star"].detach().cpu(),
            "all_test": [torch.tensor(t, dtype=torch.float64) for t in all_test],
            "all_train": [torch.tensor(t, dtype=torch.float64) for t in all_train],
        },
        artifacts_path,
    )
    print(f"Saved artifacts to: {artifacts_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
