#!/bin/bash
#SBATCH --job-name=B1-circulant-modes
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

# Experiment B1: exact finite-P circulant mode closure.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §6.2.
#
# Operator-level experiment. Runs on CUDA by default: the matrix recursion
# uses torch.matrix_power on (P, P) tensors for every (P, L, symbol_kind)
# trial, which is O(P^3 * log(2L-1)) per step and benefits from GPU
# acceleration at P = 64. A single GPU and <= 15 min wallclock suffice for
# the default 3 x 4 x 3 sweep.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running B1: exact finite-P circulant mode closure..."
python -u scripts/thesis/theoremB/run_theoremB_circulant_modes.py \
    --device cuda \
    --dtype float64 \
    --no-show
