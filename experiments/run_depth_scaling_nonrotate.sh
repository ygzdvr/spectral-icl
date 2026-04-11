#!/bin/bash
#SBATCH --job-name=depth-scaling-nonrotate
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=0:30:00
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

echo "Running non-rotated depth scaling sweep..."
python -u scripts/run_depth_scaling_nonrotate.py --no-show --device cuda
#!/bin/bash
#SBATCH --job-name=depth-scaling-nonrotate
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=0:30:00
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

echo "Running depth scaling nonrotate sweep..."
python -u scripts/run_depth_scaling_nonrotate.py --no-show --device cuda
