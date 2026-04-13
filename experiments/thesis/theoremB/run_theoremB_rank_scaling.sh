#!/bin/bash
#SBATCH --job-name=B4-rank-scaling
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

# Experiment B4: spectral-rank bottleneck and joint spectral-shape sweep.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §6.5.
#
# Operator-level experiment. Rank r is the primary control variable; joint
# (r, L_S) grid tests the theorem-B pure-spectral shape law (matched
# stationary regime, BEFORE hybridization). Thanks to the mode-decoupling
# shortcut, we only train one unmasked trajectory per L — rank-r losses
# come from post-masking γ. Default 8 r × 4 L = 32 evaluations driven by
# 4 training trajectories (T = 100000 each), ≤ 20 min wallclock on 1 GPU.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running B4: spectral-rank bottleneck + joint (r, L_S) spectral-shape sweep..."
python -u scripts/thesis/theoremB/run_theoremB_rank_scaling.py \
    --device cuda \
    --dtype float64 \
    --no-show
