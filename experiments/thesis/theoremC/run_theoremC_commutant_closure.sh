#!/bin/bash
#SBATCH --job-name=C1C2-commutant-closure
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

# Experiments C1 + C2: band-RRS commutant closure + grouped scalar dynamics.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §7.1 (C1) + §7.2 (C2).
#
# Operator-level only — no learned architecture. The main cost is the
# per-step Python loop of the reduced-Γ recursion (D × 4 trajectories ×
# T steps). Acceptance is float64 machine precision on both the commutant
# violation (C1) and the matrix-ODE block-scalar agreement (C2). The
# naive per-F-mode dynamics is included as a hard negative control.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running C1 + C2: band-RRS commutant closure and grouped dynamics..."
python -u scripts/thesis/theoremC/run_theoremC_commutant_closure.py \
    --device cuda \
    --dtype float64 \
    --no-show
