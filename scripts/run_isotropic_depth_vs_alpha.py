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

from configs import IsotropicDepthAlphaSweepConfig
from dynamics import run_isotropic_depth_vs_alpha_sweep
from utils import parse_int_list


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)


def main() -> None:
    """Run isotropic depth-vs-alpha sweeps with DMFT theory overlays.

    This command executes a grid over prompt sizes and depth values, plots
    final-loss theory/experiment comparisons, and saves complete run artifacts.
    """
    parser = argparse.ArgumentParser(description="Torch port of notebook isotropic depth-vs-alpha sweep.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--p-test", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=0.125)
    parser.add_argument("--lamb", type=float, default=1.0e-14)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--p-trs", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16")
    parser.add_argument("--theory-t", type=int, default=512)
    parser.add_argument("--theory-iters", type=int, default=100)
    parser.add_argument("--main-plot-path", type=str, default=None)
    parser.add_argument("--alpha-plot-path", type=str, default=None)
    parser.add_argument("--l-plot-path", type=str, default=None)
    parser.add_argument("--losses-path", type=str, default=None)
    parser.add_argument("--artifacts-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    requested_device = torch.device(args.device)
    print(f"requested device: {requested_device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if requested_device.type == "cuda" and torch.cuda.is_available():
        current_idx = requested_device.index if requested_device.index is not None else torch.cuda.current_device()
        print(f"using cuda device: cuda:{current_idx}")
        print(f"cuda device name: {torch.cuda.get_device_name(current_idx)}")
    elif requested_device.type == "cuda":
        print("CUDA was requested but is not available.")

    p_trs = parse_int_list(args.p_trs)
    lvals = parse_int_list(args.lvals)

    cfg = IsotropicDepthAlphaSweepConfig(
        d=args.d,
        P_test=args.p_test,
        B=args.batch,
        N=args.n,
        alpha=args.alpha,
        beta=args.beta,
        T=args.steps,
        lr=args.lr,
        lamb=args.lamb,
        beta_model=args.beta_model,
        gamma=args.gamma,
        sigma=args.sigma,
        p_trs=tuple(p_trs),
        lvals=tuple(lvals),
        theory_T=args.theory_t,
        theory_iters=args.theory_iters,
    )

    results = run_isotropic_depth_vs_alpha_sweep(cfg, device=args.device, dtype=dtype)

    alpha_vals = results["alpha_vals"].detach().cpu().numpy()
    loss_np = results["loss_np"].detach().cpu().numpy()
    all_losses_tr = results["all_losses_tr"].detach().cpu().numpy()
    all_losses_std = results["all_losses_std"].detach().cpu().numpy()
    all_losses = results["all_losses"]

    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        plt.semilogx(alpha_vals, loss_np[2 * L], color=f"C{i}")
        plt.errorbar(
            np.asarray(p_trs, dtype=np.float64) / float(args.d),
            all_losses_tr[i],
            all_losses_std[i],
            fmt="o",
            color=f"C{i}",
            label=f"L={L}",
        )
    plt.semilogx([], [], color="C0", label="Theory")
    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(\alpha)$", fontsize=20)
    plt.legend()

    main_plot_path = (
        Path(args.main_plot_path)
        if args.main_plot_path
        else PROJECT_ROOT / "outputs" / "depth_vs_alpha_isotropic.png"
    )
    main_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(main_plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved main plot to: {main_plot_path}")

    plt.figure()
    sns.set_palette("rocket", n_colors=len(p_trs))
    for i, P in enumerate(p_trs):
        plt.plot(all_losses[i][-1], label=rf"$\alpha = {1.0 * P / args.d:0.1f}$")
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t, \alpha)$", fontsize=20)
    plt.legend()

    alpha_plot_path = (
        Path(args.alpha_plot_path)
        if args.alpha_plot_path
        else PROJECT_ROOT / "outputs" / "train_dynamics_vary_alpha.png"
    )
    alpha_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(alpha_plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved alpha dynamics plot to: {alpha_plot_path}")

    plt.figure()
    sns.set_palette("rocket", n_colors=len(lvals))
    for i, L in enumerate(lvals):
        plt.plot(all_losses[2][i], label=f"$L = {L}$")
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t, L)$", fontsize=20)
    plt.legend()

    l_plot_path = (
        Path(args.l_plot_path)
        if args.l_plot_path
        else PROJECT_ROOT / "outputs" / "train_dynamics_vary_L.png"
    )
    l_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(l_plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved L dynamics plot to: {l_plot_path}")

    losses_path = (
        Path(args.losses_path)
        if args.losses_path
        else PROJECT_ROOT / "outputs" / "isotropic_depth_vs_alpha_losses.npz"
    )
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "alpha_vals": alpha_vals,
        "p_trs": np.asarray(p_trs, dtype=np.int64),
        "lvals": np.asarray(lvals, dtype=np.int64),
        "loss_np": loss_np,
        "all_losses_tr": all_losses_tr,
        "all_losses_std": all_losses_std,
    }
    for i, p_tr in enumerate(p_trs):
        for j, L in enumerate(lvals):
            payload[f"loss_P_{p_tr}_L_{L}"] = np.asarray(all_losses[i][j], dtype=np.float64)
    np.savez(losses_path, **payload)
    print(f"Saved losses to: {losses_path}")

    artifacts_path = (
        Path(args.artifacts_path)
        if args.artifacts_path
        else PROJECT_ROOT / "outputs" / "isotropic_depth_vs_alpha_artifacts.pt"
    )
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": vars(args),
            "spec": results["spec"].detach().cpu(),
            "w_star": results["w_star"].detach().cpu(),
            "alpha_vals": results["alpha_vals"].detach().cpu(),
            "loss_np": results["loss_np"].detach().cpu(),
            "all_losses_tr": results["all_losses_tr"].detach().cpu(),
            "all_losses_std": results["all_losses_std"].detach().cpu(),
            "all_losses": [[torch.tensor(loss, dtype=torch.float64) for loss in loss_i] for loss_i in all_losses],
            "p_trs": p_trs,
            "lvals": lvals,
        },
        artifacts_path,
    )
    print(f"Saved artifacts to: {artifacts_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
