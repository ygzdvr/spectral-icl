#!/bin/bash
#SBATCH --job-name=B3-symbol-shift
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Experiment B3: symbol-native spectral OOD brittleness.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §6.4.
#
# Operator-level experiment. The matched-training recursion (per L, via
# metrics.gamma_star_trajectory_circulant) is the main cost center; OOD
# evaluation is a handful of Σ_k reductions per (α, L). Generic covariance
# rotations are deliberately excluded from this script and belong to a
# later bridge experiment toward theorem C. A single GPU and <= 20 min
# wallclock suffice for the default 5-L × 12-α × 8-seed sweep at T=100000.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running B3: symbol-native OOD brittleness..."
python -u scripts/thesis/theoremB/run_theoremB_symbol_shift.py \
    --device cuda \
    --dtype float64 \
    --no-show
