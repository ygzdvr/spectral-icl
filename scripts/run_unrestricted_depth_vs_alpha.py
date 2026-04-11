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
from utils import OutputDir, moving_average, parse_int_list


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)

def main() -> None:
    """Run unrestricted depth-vs-alpha sweeps and generate analysis artifacts.

    This entry point parses CLI settings, launches the unrestricted training
    sweep/theory comparison pipeline, produces plots, and stores losses and
    serialized artifacts for downstream inspection.
    """
    parser = argparse.ArgumentParser(
        description="Torch port of notebook unrestricted depth-vs-alpha sweep."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--p-test", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lamb", type=float, default=1.0e-6)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--p-trs", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--lvals", type=str, default="1,2,4,8,16")
    parser.add_argument("--theory-lvals", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--theory-t", type=int, default=512)
    parser.add_argument("--theory-iters", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
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

    p_trs = parse_int_list(args.p_trs)
    lvals = parse_int_list(args.lvals)
    theory_lvals = parse_int_list(args.theory_lvals)
    out = OutputDir(__file__, base=args.output_dir)

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
        unrestricted=True,
        theory_T=args.theory_t,
        theory_iters=args.theory_iters,
    )

    results = run_isotropic_depth_vs_alpha_sweep(cfg, device=args.device, dtype=dtype)

    alpha_vals = results["alpha_vals"].detach().cpu().numpy()
    loss_np = results["loss_np"].detach().cpu().numpy()
    all_losses_tr = results["all_losses_tr"].detach().cpu().numpy()
    all_losses_std = results["all_losses_std"].detach().cpu().numpy()
    all_losses = results["all_losses"]

    # --- Theory-only plot (matches notebook cell) ---
    plt.figure()
    sns.set_palette("rocket", n_colors=len(theory_lvals))
    for i, L in enumerate(theory_lvals):
        plt.semilogx(alpha_vals, loss_np[2 * L], label=f"L={L}")
    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel("Final Loss", fontsize=20)
    plt.legend()

    plt.savefig(out.png("unrestricted_depth_vs_alpha_theory"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("unrestricted_depth_vs_alpha_theory"), dpi=200, bbox_inches="tight")
    print(f"Saved theory plot to: {out.png('unrestricted_depth_vs_alpha_theory')}")

    # --- Combined plot (theory + empirical errorbars) ---
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

    plt.savefig(out.png("unrestricted_depth_vs_alpha_combined"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("unrestricted_depth_vs_alpha_combined"), dpi=200, bbox_inches="tight")
    print(f"Saved combined plot to: {out.png('unrestricted_depth_vs_alpha_combined')}")

    # --- Smoothed pretrain loss per depth (one figure per P_tr) ---
    for pi, p_tr in enumerate(p_trs):
        plt.figure()
        sns.set_palette("rocket", n_colors=len(lvals))
        for li, L in enumerate(lvals):
            loss_arr = np.asarray(all_losses[pi][li], dtype=np.float64)
            smooth = moving_average(loss_arr, 100)
            plt.loglog(smooth, label=f"$L = {L}$")
        plt.xlabel(r"Steps", fontsize=20)
        plt.ylabel(r"Pretrain Loss", fontsize=20)
        plt.legend()
        plt.savefig(out.png(f"unrestricted_depth_scaling_P{p_tr}"), dpi=200, bbox_inches="tight")
        plt.savefig(out.pdf(f"unrestricted_depth_scaling_P{p_tr}"), dpi=200, bbox_inches="tight")
        print(f"Saved depth scaling plot (P={p_tr}) to: {out.png(f'unrestricted_depth_scaling_P{p_tr}')}")

    # --- Save losses ---
    payload: dict[str, np.ndarray] = {
        "alpha_vals": alpha_vals,
        "p_trs": np.asarray(p_trs, dtype=np.int64),
        "lvals": np.asarray(lvals, dtype=np.int64),
        "theory_lvals": np.asarray(theory_lvals, dtype=np.int64),
        "loss_np": loss_np,
        "all_losses_tr": all_losses_tr,
        "all_losses_std": all_losses_std,
    }
    for i, p_tr in enumerate(p_trs):
        for j, L in enumerate(lvals):
            payload[f"loss_P_{p_tr}_L_{L}"] = np.asarray(all_losses[i][j], dtype=np.float64)
    np.savez(out.numpy("unrestricted_depth_vs_alpha_losses"), **payload)
    print(f"Saved losses to: {out.numpy('unrestricted_depth_vs_alpha_losses')}")

    # --- Save artifacts ---
    torch.save(
        {
            "config": vars(args),
            "spec": results["spec"].detach().cpu(),
            "w_star": results["w_star"].detach().cpu(),
            "alpha_vals": results["alpha_vals"].detach().cpu(),
            "loss_np": results["loss_np"].detach().cpu(),
            "all_losses_tr": results["all_losses_tr"].detach().cpu(),
            "all_losses_std": results["all_losses_std"].detach().cpu(),
            "all_losses": [
                [torch.tensor(loss, dtype=torch.float64) for loss in loss_i]
                for loss_i in all_losses
            ],
            "p_trs": p_trs,
            "lvals": lvals,
            "theory_lvals": theory_lvals,
        },
        out.torch("unrestricted_depth_vs_alpha_artifacts"),
    )
    print(f"Saved artifacts to: {out.torch('unrestricted_depth_vs_alpha_artifacts')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
