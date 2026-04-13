#!/bin/bash
#SBATCH --job-name=A1b-full-model-vs-reduced
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Experiment A1b: full-hidden-state aligned structured forward vs reduced.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §8.1 (A1b extension of A1).
#
# Closes the gap left by A1's R1 (iterative reduced recursion). A1b adds
# R0 = TRUE full-hidden-state forward pass that builds the (P+K)×(P+K)
# bilinear score from (X, Γ), applies a GD-compatible signed mask, and
# runs L explicit residual-stream layer updates without consuming the GA
# generator's (A_S, B_S, T) as inputs. Compares R0 vs R2 vs R3 at float64
# machine precision over the same (D, P, K, L) sweep used in A1.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running A1b: full-hidden-state structured forward vs reduced..."
python -u scripts/thesis/theoremA/run_theoremA_exact_equivalence_full_model.py \
    --device cuda \
    --dtype float64 \
    --no-show
