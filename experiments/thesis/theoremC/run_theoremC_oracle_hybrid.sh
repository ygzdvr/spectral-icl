#!/bin/bash
#SBATCH --job-name=C6-oracle-hybrid
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

# Experiment C6: oracle hybrid defined correctly.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §7.6.
#
# Operator-level only — NO learned architectures, NO trained networks,
# NO estimated projectors. The "oracle hybrid" here is DEFINED AT THE
# THEOREM LEVEL as direct optimization over the refined commutant class.
# Architecture experiments belong to §9 and will approximate this
# theorem-level reference with learned projectors. For each (m, κ, L)
# the script runs 3 L-BFGS optimizations: coarse, refined-commutant
# (= oracle hybrid), and singleton (= oracle ceiling). ≤ 30 min on 1 GPU.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running C6: oracle hybrid defined correctly (refined commutant, NOT learned)..."
python -u scripts/thesis/theoremC/run_theoremC_oracle_hybrid.py \
    --device cuda \
    --dtype float64 \
    --no-show
