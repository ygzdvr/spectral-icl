#!/bin/bash
# Source this file to set up the spectral_scaling environment:
#   source starter.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Optional cluster modules.
module load cudatoolkit/12.6 2>/dev/null
module load gcc/11 2>/dev/null

# CUDA libs (if CUDA_HOME is defined on this machine).
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Add pip-installed NVIDIA shared libraries from this venv (if present).
if [ -d "$VENV_DIR/lib" ]; then
    while IFS= read -r _d; do
        export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done < <(find "$VENV_DIR/lib" -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null)
fi

# Keep caches in-repo.
export UV_CACHE_DIR="$SCRIPT_DIR/.cache/uv"
export PIP_CACHE_DIR="$SCRIPT_DIR/.cache/pip"
export XDG_CACHE_HOME="$SCRIPT_DIR/.cache"
export MPLCONFIGDIR="$SCRIPT_DIR/.cache/matplotlib"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" 2>/dev/null

# Ensure local package modules are importable without editable install.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "spectral_scaling ready ($(python -c 'import torch; print(f\"torch {torch.__version__}, cuda={torch.cuda.is_available()}\")' 2>/dev/null || echo 'torch not verified'))"
else
    echo "Venv not found at $VENV_DIR — run install.sh first"
fi
