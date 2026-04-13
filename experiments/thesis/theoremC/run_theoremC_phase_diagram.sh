#!/bin/bash
#SBATCH --job-name=C4-phase-diagram
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

# Experiment C4: theorem-C heterogeneity phase diagram (headline figure).
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §7.4.
#
# Operator-level only — no learned architecture, no trained network. For
# each (m, κ, L) cell the script runs L-BFGS optimization over block
# scalars at the coarse partition (size m), the dyadically-finer
# partition (size m/2), and the singleton partition (full oracle, sanity
# check ≡ 0). Default 6 m × 7 κ × 5 L = 210 cells × 3 L-BFGS ≈ 630
# optimizations; ≤ 30 min on 1 GPU.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running C4: theorem-C heterogeneity phase diagram..."
python -u scripts/thesis/theoremC/run_theoremC_phase_diagram.py \
    --device cuda \
    --dtype float64 \
    --no-show
