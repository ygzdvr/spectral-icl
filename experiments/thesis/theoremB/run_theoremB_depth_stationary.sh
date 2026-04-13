#!/bin/bash
#SBATCH --job-name=B2-depth-stationary
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

# Experiment B2: matched stationary depth-irrelevance.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §6.3.
#
# Operator-level experiment that uses the per-mode circulant recursion from
# `metrics.gamma_star_trajectory_circulant` (the single source validated at
# machine precision by B1). The recursion itself is a sequential Python loop
# over T steps; GPU offers no speedup at these problem sizes, but we still
# run on CUDA so the environment matches the canonical thesis launcher
# contract (per repo policy: always source starter.sh; always default to
# CUDA). A single GPU and <= 30 min wallclock suffice for the default
# 3 × 5 × 2 sweep at T = 20000.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running B2: matched stationary depth irrelevance..."
python -u scripts/thesis/theoremB/run_theoremB_depth_stationary.py \
    --device cuda \
    --dtype float64 \
    --no-show
