#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

export UV_CACHE_DIR="$SCRIPT_DIR/.cache/uv"
export XDG_CACHE_HOME="$SCRIPT_DIR/.cache"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
mkdir -p "$UV_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME"

echo "=== spectral_scaling installer ==="
echo "  Project: $SCRIPT_DIR"
echo "  Venv:    $VENV_DIR"
echo "  Cache:   $UV_CACHE_DIR"
echo ""

# Optional cluster modules.
module load cudatoolkit/12.6 2>/dev/null || true
module load gcc/11 2>/dev/null || true
echo "Tried loading cudatoolkit/12.6 and gcc/11 (if available)."
echo ""

# Create venv.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    uv venv "$VENV_DIR"
else
    echo "Venv exists, reusing."
fi

echo "Installing dependencies..."
uv sync --project "$SCRIPT_DIR" --python "$VENV_DIR/bin/python"

echo ""
echo "Verifying..."
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$VENV_DIR/bin/python" -c "
import torch
from models import SimpleTransformer
from data import make_train_test_batches, LinearICLConfig
x, y, _, _ = make_train_test_batches(LinearICLConfig())
m = SimpleTransformer(depth=1)
_ = m(x)
print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')
print('spectral_scaling OK')
"

echo ""
echo "Done. To use:"
echo "  source $SCRIPT_DIR/starter.sh"
