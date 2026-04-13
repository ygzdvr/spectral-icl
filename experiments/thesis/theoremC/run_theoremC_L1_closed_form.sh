#!/bin/bash
#SBATCH --job-name=C3-L1-closed-form
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

# Experiment C3: exact L = 1 closed-form block-commutant lower bound.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §7.3.
#
# Operator-level only — no learned architecture. For each (m, κ) the
# script builds a G2 operator-only configuration, computes the theorem-C
# closed-form minimizer (q_b* = b_b / c_b, L_b* = a_b - b_b² / c_b),
# compares against L-BFGS numerical optimization of the same loss, and
# produces the κ-dependent obstruction visualization. ≤ 15 min on 1 GPU.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running C3: L = 1 closed-form block-commutant lower bound..."
python -u scripts/thesis/theoremC/run_theoremC_L1_closed_form.py \
    --device cuda \
    --dtype float64 \
    --no-show
