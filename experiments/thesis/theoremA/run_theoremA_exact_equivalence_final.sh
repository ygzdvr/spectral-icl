#!/bin/bash
#SBATCH --job-name=A1final-exact-equivalence
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

# Experiment A1final: four-way exact theorem-A forward-pass equivalence.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §8.1 (unifies A1 + A1b).
#
# Operator-level deterministic forward-pass equivalence test. NO training,
# NO architecture search. Compares four structurally distinct forward
# routes (R0 = true full-hidden-state aligned structured forward;
# R1 = iterative reduced (A_S, B_S) recursion; R2 = closed-form reduced
# (A_S, B_S); R3 = feature-space reduced-Γ) and verifies all six
# pairwise errors agree to float64 machine precision on a sweep over
# (D, P, K, L). ≤ 15 min on 1 GPU.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running A1final: four-way theorem-A equivalence (R0, R1, R2, R3)..."
python -u scripts/thesis/theoremA/run_theoremA_exact_equivalence_final.py \
    --device cuda \
    --dtype float64 \
    --no-show
