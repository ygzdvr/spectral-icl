#!/bin/bash
#SBATCH --job-name=spectral-icl
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge
module load gcc/11
module load cudatoolkit/12.6

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/starter.sh"

nvidia-smi

echo "Running reduced gamma depth sweep..."
python -u scripts/run_reduced_gamma_depth_sweep.py --no-show --device cuda

echo "Running reduced gamma dynamics..."
python -u scripts/run_reduced_gamma_dynamics.py --run-sgd --alpha 1.75 --beta 1.33 --no-show --device cuda

echo "Running compute scaling width..."
python -u scripts/run_compute_scaling_width.py --no-show --device cuda

echo "Running compute scaling depth..."
python -u scripts/run_compute_scaling_depth.py --no-show --device cuda

echo "Running compute scaling joint..."
python -u scripts/run_compute_scaling_joint.py --no-show --device cuda

echo "Running fixed covariance..."
python -u scripts/run_fixed_covariance.py --no-show --device cuda

python -u scripts/run_softmax_depth_sweep.py --device cuda --no-show

echo "Running linear attention dynamics..."
python -u scripts/run_linear_attention_dynamics.py --no-show --device cuda

echo "Running dim free dynamics..."
python -u scripts/run_dim_free_dynamics.py --no-show --device cuda

echo "Running decoupled layers..."
python -u scripts/run_decoupled_layers.py --no-show --device cuda

