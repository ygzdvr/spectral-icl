#!/bin/bash
#SBATCH --job-name=A2-mask-perturbation
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Experiment A2: theorem-A perturbation around GD-compatibility with the
# full additive (A, B) reduced-operator bound.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §8.2.
#
# Operator-level deterministic forward-pass perturbation diagnostic.
# Empirical full-model error compared to the full additive (A, B) bound;
# A-side (telescoping ΔA term) and B-side (ΔB term) reported separately
# and never folded into each other. ≤ 15 min on 1 GPU.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running A2: theorem-A mask perturbation with full (A, B) bound..."
python -u scripts/thesis/theoremA/run_theoremA_mask_perturbation.py \
    --device cuda \
    --dtype float64 \
    --no-show
