#!/bin/bash
#SBATCH --job-name=arch-spectral-depth-irrelevance
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

# §9.1 architecture-aligned depth-irrelevance experiment (tied weights,
# α = P/D sweep). Plan reference: EXPERIMENT_PLAN_FINAL.MD §9.1. Theorem
# reference: thesis/theorem_b.txt Corollary
# "Modewise gradient-flow dynamics and long-context depth irrelevance"
# (cor:theoremB_modewise_ode).
#
# Definitive architecture-aligned theorem-B depth-irrelevance test. A
# single learnable γ ∈ ℝ^D is shared across all L_S layers (tied weights
# → single circulant Q, matching the theorem) and α = P/D is swept at
# fixed D = 32 through P ∈ {32, 64, 128, 256}. The depth-floor ratio
# max_{L_S}(final) / min_{L_S}(final) is expected to collapse toward 1
# as α grows (the population-level depth-irrelevance prediction, holding
# once the per-batch sample-space operator concentrates).
#
# Submit via SLURM; the login node enforces per-user cgroup memory caps
# that have previously SIGKILLed long-running scripts.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running §9.1 tied-weight α-sweep depth-irrelevance..."
python -u scripts/thesis/architectures/run_arch_spectral_depth_irrelevance.py \
    --device cuda \
    --dtype float64 \
    --no-show
