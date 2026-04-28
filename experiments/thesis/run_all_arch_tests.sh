#!/bin/bash
#SBATCH --job-name=arch-tests
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Runs the full 6-suite architectures test battery on a dedicated GPU:
#   - test_base_and_linear_attention   (float64 algebraic)
#   - test_spectral_attention          (float64 algebraic)
#   - test_training_smoke              (float32 training)
#   - test_samplers                    (float32 stationary kernel check)
#   - test_stu_native                  (float32 + float64)
#   - test_hybrid                      (float32 training + float64 edge cases)
#
# Any FAIL flips the exit code to 1; FINDING / WARN are treated as non-failing.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge
module load gcc/11
module load cudatoolkit/12.6

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/starter.sh"

nvidia-smi

python -u -m scripts.thesis.architectures.tests.test_base_and_linear_attention
echo "==="
python -u -m scripts.thesis.architectures.tests.test_spectral_attention
echo "==="
python -u -m scripts.thesis.architectures.tests.test_training_smoke
echo "==="
python -u -m scripts.thesis.architectures.tests.test_samplers
echo "==="
python -u -m scripts.thesis.architectures.tests.test_stu_native
echo "==="
python -u -m scripts.thesis.architectures.tests.test_hybrid
echo "==="
echo "all 6 arch test suites completed"
