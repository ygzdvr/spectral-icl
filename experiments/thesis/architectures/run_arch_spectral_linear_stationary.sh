#!/bin/bash
#SBATCH --job-name=arch-spectral-linear-stationary
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# §9.1 first architecture-aligned spectral-only experiment:
# trainable linear FFT-based spectral filter on matched stationary
# circulant ICL regression.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §9.1 (first script).
#
# This is the FIRST architecture-aligned validation script; theorem-B
# is already proven exactly (B1/B2/B3/B4). The qualitative
# architecture-aligned question is whether the trainable spectral
# filter preserves matched-stationary depth-irrelevance. Submit via
# SLURM; the login node enforces per-user cgroup memory caps that have
# previously SIGKILLed long-running scripts.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running §9.1 first arch-aligned spectral filter..."
python -u scripts/thesis/architectures/run_arch_spectral_linear_stationary.py \
    --device cuda \
    --dtype float64 \
    --no-show
