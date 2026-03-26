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

from configs import PretrainICLPowerLawConfig
from dynamics import run_pretrain_icl_powerlaw
from utils import moving_average


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)

def main() -> None:
    """Run the main ICL power-law pretraining experiment.

    The entry point configures the training run, executes optimization,
    overlays theoretical decay references, and saves plots/losses/artifacts.
    """
    parser = argparse.ArgumentParser(description="Torch port of the notebook ICL pretraining power-law block.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=60)
    parser.add_argument("--p-train", type=int, default=85)
    parser.add_argument("--p-test", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--lamb", type=float, default=1.0e-14)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--n-multiplier", type=float, default=1.4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="spec_rotate",
        choices=["iid", "spec", "spec_rotate", "gauss_rotate"],
    )
    parser.add_argument("--plot-linear-path", type=str, default=None)
    parser.add_argument("--plot-loglog-path", type=str, default=None)
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

    cfg = PretrainICLPowerLawConfig(
        d=args.d,
        P_tr=args.p_train,
        P_test=args.p_test,
        B=args.batch,
        L=args.layers,
        alpha=args.alpha,
        beta=args.beta,
        T=args.steps,
        lr=args.lr,
        lamb=args.lamb,
        beta_model=args.beta_model,
        n_multiplier=args.n_multiplier,
        gamma=args.gamma,
        sigma=args.sigma,
        sample_mode=args.sample_mode,
    )

    results = run_pretrain_icl_powerlaw(cfg, device=args.device, dtype=dtype)
    pretrain_loss = np.asarray(results["pretrain_loss"], dtype=np.float64)
    steps = np.arange(1, len(pretrain_loss) + 1, dtype=np.float64)
    smooth_10 = moving_average(pretrain_loss, 10)
    smooth_25 = moving_average(pretrain_loss, 25)
    smooth_10_steps = np.arange(1, len(smooth_10) + 1, dtype=np.float64)
    smooth_25_steps = np.arange(1, len(smooth_25) + 1, dtype=np.float64)

    th_exp = 7.0 * cfg.beta / (2.0 + 7.0 * cfg.beta)
    theory_steps = np.linspace(100.0, float(len(pretrain_loss)), len(pretrain_loss), dtype=np.float64)
    theory_fast = 1e1 * theory_steps ** (-th_exp)
    theory_beta = 1e3 * theory_steps ** (-cfg.beta)

    print(f"device: {requested_device}")
    print(f"N: {results['N']}")
    print(f"last batch X shape: {tuple(results['last_X'].shape)}")
    print(f"last batch y shape: {tuple(results['last_y'].shape)}")
    print(f"theory exponent: {th_exp:.8f}")

    plt.figure()
    plt.plot(smooth_10_steps, smooth_10)
    plt.ylim([0, 1.1])
    plt.xlabel(r"Steps", fontsize=20)
    plt.ylabel(r"Pretrain Loss", fontsize=20)
    plot_linear_path = (
        Path(args.plot_linear_path)
        if args.plot_linear_path
        else PROJECT_ROOT / "outputs" / "pretrain_icl_powerlaw_linear.png"
    )
    plot_linear_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_linear_path, dpi=200, bbox_inches="tight")
    print(f"Saved linear plot to: {plot_linear_path}")

    plt.figure()
    plt.loglog(smooth_25_steps, smooth_25)
    plt.loglog(theory_steps, theory_fast, "--", color="blue", label=r"$t^{-\frac{7\beta}{2 + 7\beta}}$")
    plt.loglog(theory_steps, theory_beta, "--", color="red", label=r"$t^{-\beta}$")
    plt.ylim([4e-3, 1.2])
    plt.xlabel(r"Steps", fontsize=20)
    plt.ylabel(r"Pretrain Loss", fontsize=20)
    plt.legend()
    plot_loglog_path = (
        Path(args.plot_loglog_path)
        if args.plot_loglog_path
        else PROJECT_ROOT / "outputs" / f"pretrain_ICL_powerlaw_L_{cfg.L}_beta_{cfg.beta}.png"
    )
    plot_loglog_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_loglog_path, dpi=200, bbox_inches="tight")
    print(f"Saved loglog plot to: {plot_loglog_path}")

    losses_path = (
        Path(args.losses_path)
        if args.losses_path
        else PROJECT_ROOT / "outputs" / "pretrain_icl_powerlaw_losses.npz"
    )
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        losses_path,
        pretrain_loss=pretrain_loss,
        steps=steps,
        smooth_10=smooth_10,
        smooth_10_steps=smooth_10_steps,
        smooth_25=smooth_25,
        smooth_25_steps=smooth_25_steps,
        theory_steps=theory_steps,
        theory_fast=theory_fast,
        theory_beta=theory_beta,
        theory_exponent=np.asarray([th_exp], dtype=np.float64),
    )
    print(f"Saved losses to: {losses_path}")

    artifacts_path = (
        Path(args.artifacts_path)
        if args.artifacts_path
        else PROJECT_ROOT / "outputs" / "pretrain_icl_powerlaw_artifacts.pt"
    )
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": vars(args),
            "N": results["N"],
            "spec": results["spec"].detach().cpu(),
            "w_star": results["w_star"].detach().cpu(),
            "params": [p.detach().cpu() for p in results["params"]],
            "last_X": results["last_X"].detach().cpu(),
            "last_y": results["last_y"].detach().cpu(),
            "pretrain_loss": torch.tensor(pretrain_loss, dtype=torch.float64),
            "smooth_10": torch.tensor(smooth_10, dtype=torch.float64),
            "smooth_25": torch.tensor(smooth_25, dtype=torch.float64),
            "theory_steps": torch.tensor(theory_steps, dtype=torch.float64),
            "theory_fast": torch.tensor(theory_fast, dtype=torch.float64),
            "theory_beta": torch.tensor(theory_beta, dtype=torch.float64),
        },
        artifacts_path,
    )
    print(f"Saved artifacts to: {artifacts_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
