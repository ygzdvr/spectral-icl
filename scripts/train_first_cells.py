import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Allow running the script directly: `python scripts/train_first_cells.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import LinearICLConfig
from data import make_train_test_batches
from models import SimpleTransformer
from utils import OutputDir


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)


def mse_last_token(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE on the final token prediction of a sequence model.

    Args:
        model: Sequence model returning token-wise scalar outputs.
        x: Input token tensor of shape ``[B, S, D]``.
        target: Final-token regression targets of shape ``[B]``.

    Returns:
        Scalar mean-squared error between predicted and target final-token values.
    """
    out = model(x)
    preds = out[:, -1, 0]
    return torch.mean((preds - target) ** 2)


def main() -> None:
    """Train the baseline transformer used in early notebook cells.

    Builds synthetic train/test batches, optimizes a `SimpleTransformer`,
    tracks train/test losses, and emits plot/data outputs.
    """
    parser = argparse.ArgumentParser(description="Torch port of first notebook cells.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--out-scale", type=float, default=0.1)
    parser.add_argument(
        "--use-softmax-attn",
        action="store_true",
        help="Use SDPA (FlashAttention-eligible on CUDA). Default reproduces raw notebook attention.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot window.")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = LinearICLConfig()
    x_train, y_train, x_test, y_test = make_train_test_batches(cfg, device=device)

    model = SimpleTransformer(
        width=args.width,
        depth=args.depth,
        out_scale=args.out_scale,
        use_softmax=args.use_softmax_attn,
    ).to(device)

    # Initialize lazy input projection before creating optimizer.
    with torch.no_grad():
        _ = model(x_train)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    print(f"x_big shape: {tuple(x_train.shape)}")
    print(f"target mean: {torch.mean(y_train**2).item():.6f}")
    print(f"test target mean: {torch.mean(y_test**2).item():.6f}")

    train_losses: list[float] = []
    test_losses: list[float] = []

    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = mse_last_token(model, x_train, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = mse_last_token(model, x_test, y_test)

        train_losses.append(float(train_loss.detach().cpu()))
        test_losses.append(float(test_loss.detach().cpu()))

        if step % 100 == 0:
            print(f"Loss step {step}: {train_losses[-1]:.6f}")

    out = OutputDir(__file__, base=args.output_dir)

    xs = torch.linspace(10, args.steps, args.steps).cpu().numpy()
    ref = (xs ** 0.5).tolist()

    plt.loglog(train_losses, label="train")
    plt.loglog(test_losses, label="test")
    plt.loglog(ref, label="sqrt(t)")
    plt.legend()
    plt.savefig(out.png("train_test"), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf("train_test"), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out.png('train_test')}")

    np.savez(
        out.numpy("losses"),
        train=np.asarray(train_losses, dtype=np.float64),
        test=np.asarray(test_losses, dtype=np.float64),
        ref=np.asarray(ref, dtype=np.float64),
    )
    print(f"Saved losses to: {out.numpy('losses')}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
