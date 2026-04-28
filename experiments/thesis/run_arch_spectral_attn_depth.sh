#!/bin/bash
#SBATCH --job-name=arch-spectral-attn-depth
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Architectures Section 9: first trained-architecture experiment.
# Trains LinearAttention and two SpectralAttention variants (gd init with r=P,
# zero init with r=8) across {alpha in {1, 2}} x {L in {1,2,4,8,16}} x 4 seeds
# = 120 Adam runs at 5000 steps each. Validates (i) bridge to LinearAttention
# under optimization, (ii) Bordelon ISO-regime depth-alpha interaction, and
# (iii) learning-from-scratch for the spectral-bottleneck variant.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge
module load gcc/11
module load cudatoolkit/12.6

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/starter.sh"

nvidia-smi

python -u scripts/thesis/architectures/run_arch_spectral_attn_depth.py \
    --device cuda \
    --no-show
