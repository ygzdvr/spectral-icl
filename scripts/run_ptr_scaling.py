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
from dynamics import run_ptr_scaling_sweep
from utils import moving_average, parse_int_list, OutputDir


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)


def main() -> None:
    """Run fixed-depth prompt-length (P_tr) scaling experiments.

    This routine trains one model per prompt length, plots smoothed dynamics
    and final-loss curves against alpha=P/D, and saves numeric artifacts.
    """
    parser = argparse.ArgumentParser(description="Torch port of notebook P_tr scaling sweep at fixed depth.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--p-test", type=int, default=16)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--lamb", type=float, default=1.0e-14)
    parser.add_argument("--beta-model", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--p-trs", type=str, default="4,8,16,32,64,128")
    parser.add_argument("--output-dir", type=str, default=None)
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
    cfg = DecoupledTrainModelConfig(
        d=args.d,
        P_tr=p_trs[0],
        P_test=args.p_test,
        B=args.batch,
        N=args.n,
        L=args.layers,
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
        online=True,
    )

    out = OutputDir(__file__, base=args.output_dir)

    results = run_ptr_scaling_sweep(cfg, p_trs, device=args.device, dtype=dtype)
    all_losses = results["all_losses"]
    final_loss = np.asarray(results["final_loss"], dtype=np.float64)

    plt.figure()
    sns.set_palette("rocket", n_colors=len(all_losses))
    smoothed_losses: list[np.ndarray] = []
    smoothed_steps: list[np.ndarray] = []
    for i, loss in enumerate(all_losses):
        loss_np = np.asarray(loss, dtype=np.float64)
        smooth = moving_average(loss_np, 100)
        steps = np.arange(1, len(smooth) + 1, dtype=np.float64)
        smoothed_losses.append(smooth)
        smoothed_steps.append(steps)
        plt.loglog(steps, smooth, label=f"$P = {p_trs[i]}$")
    plt.xlabel(r"Steps", fontsize=20)
    plt.ylabel(r"Pretrain Loss", fontsize=20)
    plt.legend()

    plt.savefig(out.png("pretrain_loss"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("pretrain_loss"), dpi=200, bbox_inches="tight")
    print(f"Saved loss plot to: {out.png('pretrain_loss')}")

    plt.figure()
    alpha_points = np.asarray(p_trs, dtype=np.float64) / float(args.d)
    theory_alphas = np.logspace(-1, 1.25, 100)
    theory_depth_1 = 1.0 - (1.0 + 1.0 / theory_alphas) ** (-1)
    theory_depth_inf = np.maximum(0.0, 1.0 - theory_alphas)
    plt.semilogx(alpha_points, final_loss, "o")
    plt.semilogx(theory_alphas, theory_depth_1, color="blue", label=r"$L = 1$")
    plt.semilogx(theory_alphas, theory_depth_inf, color="red", label=r"$L = \infty$")
    plt.xlabel(r"$\alpha = P/D$", fontsize=20)
    plt.ylabel("Depth 1 Final Loss", fontsize=20)
    plt.legend()

    plt.savefig(out.png("final_loss"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("final_loss"), dpi=200, bbox_inches="tight")
    print(f"Saved final-loss plot to: {out.png('final_loss')}")

    npz_payload: dict[str, np.ndarray] = {
        "p_trs": np.asarray(p_trs, dtype=np.int64),
        "alpha_points": alpha_points,
        "final_loss": final_loss,
        "theory_alphas": theory_alphas,
        "theory_depth_1": theory_depth_1,
        "theory_depth_inf": theory_depth_inf,
    }
    for i, loss in enumerate(all_losses):
        npz_payload[f"loss_P_{p_trs[i]}"] = np.asarray(loss, dtype=np.float64)
        npz_payload[f"smooth_P_{p_trs[i]}"] = smoothed_losses[i]
        npz_payload[f"smooth_steps_P_{p_trs[i]}"] = smoothed_steps[i]
    np.savez(out.numpy("losses"), **npz_payload)
    print(f"Saved losses to: {out.numpy('losses')}")

    torch.save(
        {
            "config": vars(args),
            "p_trs": p_trs,
            "spec": results["spec"].detach().cpu(),
            "w_star": results["w_star"].detach().cpu(),
            "all_losses": [torch.tensor(loss, dtype=torch.float64) for loss in all_losses],
            "final_loss": torch.tensor(final_loss, dtype=torch.float64),
            "alpha_points": torch.tensor(alpha_points, dtype=torch.float64),
            "theory_alphas": torch.tensor(theory_alphas, dtype=torch.float64),
            "theory_depth_1": torch.tensor(theory_depth_1, dtype=torch.float64),
            "theory_depth_inf": torch.tensor(theory_depth_inf, dtype=torch.float64),
        },
        out.torch("artifacts"),
    )
    print(f"Saved artifacts to: {out.torch('artifacts')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
