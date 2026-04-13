#!/bin/bash
#SBATCH --job-name=C5-refinement-monotonicity
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

# Experiment C5: theorem-C refinement monotonicity ladder.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §7.5.
#
# Operator-level only — no learned architecture, no trained network. For
# each κ the script builds a G3 dyadic ladder (reference partition =
# coarsest single block), optimizes over every commutant class from
# coarsest to singleton via L-BFGS, and verifies refinement monotonicity.
# Default 7 κ × 7 levels × 2 L = 98 optimizations; ≤ 20 min on 1 GPU.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running C5: theorem-C refinement monotonicity ladder..."
python -u scripts/thesis/theoremC/run_theoremC_refinement_monotonicity.py \
    --device cuda \
    --dtype float64 \
    --no-show
