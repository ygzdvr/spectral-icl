#!/bin/bash
#SBATCH --job-name=C7-depth-scaling
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Experiment C7: finite-depth scaling in the grouped band-RRS class.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §7.7.
#
# Operator-level only — NO learned architectures, NO projector
# estimation, NO hybrid training. Grouped spectral-only finite-L
# optimization; the primary theorem-level overlay is the contraction
# envelope (ρ★)^(2L), NOT a generic L^(-β_b) power law. For each
# (m, κ, L) cell the script runs one L-BFGS optimization over block
# scalars. Default 6 m × 6 κ × 7 L = 252 optimizations; ≤ 30 min on 1 GPU.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running C7: finite-depth scaling in the grouped band-RRS class..."
python -u scripts/thesis/theoremC/run_theoremC_depth_scaling.py \
    --device cuda \
    --dtype float64 \
    --no-show
