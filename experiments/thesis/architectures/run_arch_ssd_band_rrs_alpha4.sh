#!/bin/bash
#SBATCH --job-name=arch-ssd-band-rrs-alpha4
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

# §9.2 α=4 variant: D=64 fixed, P=256 (up from 64), K=32 (up from 16).
# Same theorem-C spectral construction as the canonical α=1 run at
# outputs/thesis/architectures/run_arch_ssd_band_rrs/run_arch_ssd_band_rrs-20260414T233703Z-5b368495/
# The α=1 artifact is loaded for the three-panel comparison figure.
#
# Feature-space forward avoids the (B, P, P) matrix; per-step cost is
# ~2.5 ms at P=256 (same as P=64).

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"
ALPHA1_NPZ="$SCRIPT_DIR/outputs/thesis/architectures/run_arch_ssd_band_rrs/run_arch_ssd_band_rrs-20260414T233703Z-5b368495/npz/arch_ssd_band_rrs.npz"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running §9.2 arch-SSD band-RRS at α=4 (P=256, D=64)..."
python -u scripts/thesis/architectures/run_arch_ssd_band_rrs.py \
    --device cuda \
    --dtype float64 \
    --no-show \
    --P 256 \
    --K 32 \
    --D 64 \
    --L 4 \
    --stationary-bridge-thresh 0.10 \
    --filter-bridge-thresh 0.5 \
    --block-scalar-thresh 0.3 \
    --spearman-rho-min 0.3 \
    --alpha1-artifact "$ALPHA1_NPZ"
