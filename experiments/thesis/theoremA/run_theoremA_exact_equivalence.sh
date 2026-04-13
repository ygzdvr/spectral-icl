#!/bin/bash
#SBATCH --job-name=A1-exact-equivalence
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

# Experiment A1: exact full structured model vs reduced (A_S, B_S) recursion.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §8.1.
#
# Operator-level deterministic forward-pass equivalence test. NO training,
# NO architecture search. Compares three structurally distinct forward
# routes (full L-layer simulation; reduced (A_S, B_S) closed form;
# feature-space reduced-Γ closed form) and verifies they agree to float64
# machine precision on a sweep over (D, P, K, L). ≤ 15 min on 1 GPU
# (each cell is a few small matmuls).

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running A1: exact theorem-A equivalence (full vs reduced (A_S, B_S))..."
python -u scripts/thesis/theoremA/run_theoremA_exact_equivalence.py \
    --device cuda \
    --dtype float64 \
    --no-show
