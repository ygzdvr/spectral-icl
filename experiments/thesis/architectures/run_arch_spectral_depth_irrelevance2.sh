#!/bin/bash
#SBATCH --job-name=arch-spectral-depth-irrelevance2
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# §9.1 v2 architecture-aligned depth-irrelevance experiment (tied
# weights, α = P/D sweep, long-context sqrt_D normalization,
# transfer-alignment figure). Plan reference: EXPERIMENT_PLAN_FINAL.MD
# §9.1. Theorem reference: thesis/theorem_b.txt Corollary
# "Modewise gradient-flow dynamics and long-context depth irrelevance"
# (cor:theoremB_modewise_ode).
#
# Supersedes v1 (run_arch_spectral_depth_irrelevance.py) which used
# sqrt_P normalization; v2 uses the population-style sqrt_D so the
# per-batch MSE at γ = 0 is P-independent, making the depth-floor
# ratio directly comparable across α. v2 also adds a transfer-
# function alignment figure (α = 1 vs α = 8) probing whether all L_S
# converge to the matched-stationary target.
#
# Submit via SLURM; the login node enforces per-user cgroup memory
# caps that have previously SIGKILLed long-running scripts.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running §9.1 v2 tied-weight α-sweep depth-irrelevance..."
python -u scripts/thesis/architectures/run_arch_spectral_depth_irrelevance2.py \
    --device cuda \
    --dtype float64 \
    --no-show
