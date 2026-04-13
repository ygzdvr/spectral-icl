#!/bin/bash
#SBATCH --job-name=A3-A4-semiseparable
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

# Experiment A3 + A4: explicit semiseparable / SSD realization (PRIMARY)
# plus negative controls outside the theorem-A class (SECONDARY).
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §8.3 (A3) + §8.4 (A4).
#
# Operator-level deterministic forward-pass validation. NO learned
# architecture, NO training dynamics. A3 compares the explicit
# theorem-consistent SSD realization against the reduced (A_S, B_S)
# theorem object at float64 machine precision over a (D, P, K, L) sweep.
# A4 includes two negative controls that are deliberately outside the
# theorem-A class — their non-trivial deviation is the experimental
# point, not a failure. Submit via SLURM (login-node cgroup limits kill
# long-running scripts).

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running A3 + A4: semiseparable realization + negative controls..."
python -u scripts/thesis/theoremA/run_theoremA_semiseparable.py \
    --device cuda \
    --dtype float64 \
    --no-show
