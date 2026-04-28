#!/bin/bash
#SBATCH --job-name=B2-depth-stationary
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Experiment B2: matched stationary depth-irrelevance (EXPERIMENT_PLAN_FINAL.MD §6.3).
#
# Theorem objects verified:
#   Corollary 4  — modewise ODE closed form vs discrete recursion
#   Theorem 3 Claim 1 — shift invariance E_L(Pi^m Q Pi^{-m}) = E_L(Q)
#   Theorem 3 Claim 2 — gradient flow stays in Circ_P
#
# Sweep: P in {32,64} x symbol in {power_law,multiband} x L in {1,2,4,8,16}
#   + long-context sub-sweep: P in {128,256} x power_law x L in {1,2,4,8,16}
#   = 30 trials total, T=100000 steps each.
#
# Memory note: the per-mode recursion runs in a Python loop over T steps.
# gamma_traj (T+1, P) float64 is freed immediately after extracting subsampled
# trajectories; peak memory is dominated by _matched_stationary_loss internals
# (~4 x (T+1, P) float64 briefly) + _loss_exact_traj_chunked (chunk x P only).
# 64G is ample for P=256, T=100000.
#
# GPU is used for the environment contract (starter.sh CUDA path); the recursion
# itself runs on CPU float64. Runtime: ~20 min on a V100.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running B2: matched stationary depth irrelevance (with theorem diagnostics)..."
python -u scripts/thesis/theoremB/run_theoremB_depth_stationary.py \
    --device cuda \
    --dtype float64 \
    --no-show
