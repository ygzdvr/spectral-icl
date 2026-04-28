#!/bin/bash
#SBATCH --job-name=arch-spectral-rank
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Architectures Section 9: spectral rank bottleneck on stationary circulant
# data. Trains SpectralAttention(gd, r) across r in {1, 2, 4, 8, 16, 32, 64}
# plus a LinearAttention baseline, each at 4 seeds, 8000 Adam steps per run
# = 32 total runs. Fits the rank-floor power law and compares to the
# analytical tail sum and the theoretical exponent -(nu-1) = -0.5.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge
module load gcc/11
module load cudatoolkit/12.6

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/starter.sh"

nvidia-smi

python -u scripts/thesis/architectures/run_arch_spectral_attn_rank.py \
    --device cuda \
    --no-show
