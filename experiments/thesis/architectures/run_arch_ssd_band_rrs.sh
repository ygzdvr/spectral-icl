#!/bin/bash
#SBATCH --job-name=arch-ssd-band-rrs
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# §9.2 architecture-aligned structured-mask / SSD experiment on G2
# sampled band-RRS contexts. Plan ref: EXPERIMENT_PLAN_FINAL.MD §9.2.
# Theorem refs: thesis/theorem_a.txt Proposition 3 (SSD realization)
# and thesis/theorem_c.txt Proposition 4 (grouped closure).
# First use of G2 sampled-context mode in the architecture tier.
#
# 5 m × 6 κ × 4 seeds = 120 cells, each 30k steps at D=P=64, L=4.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running §9.2 arch-SSD on band-RRS..."
python -u scripts/thesis/architectures/run_arch_ssd_band_rrs.py \
    --device cuda \
    --dtype float64 \
    --no-show
