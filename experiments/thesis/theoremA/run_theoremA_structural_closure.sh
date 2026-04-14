#!/bin/bash
#SBATCH --job-name=A-structural-closure
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

# Experiment A-structural: Proposition 2 + Proposition 5 + Remark 2.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §8 (theorem-A structural tier).
#
# Covers the remaining theorem-chapter objects not validated by A1, A1b,
# A2, A3, A4, or A1-general:
#   Part 1 — Proposition 2 rank-1 factorization of M^GD and its
#            semiseparable (Definition 3) reconstruction.
#   Part 2 — Proposition 5 Toeplitz closure of the Hadamard reduction.
#   Part 3 — Proposition 5 circulant closure, plus the correct Fourier
#            identity for a Hadamard product of circulants (circular
#            convolution of DFT eigenvalues, NOT elementwise product).
#   Part 4 — Proposition 5 semiseparable rank multiplicativity, verified
#            on canonical and random strictly-lower-triangular rectangular
#            submatrices and via the explicit Kronecker factorization.
#   Part 5 — Remark 2 untied-layer non-autonomous reduced-model
#            equivalence.
#
# Deterministic matrix-identity tier; no training, no architecture, no
# statistics. Acceptance is algebraic at float64 precision.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running A-structural: Prop 2 + Prop 5 + Remark 2..."
python -u scripts/thesis/theoremA/run_theoremA_structural_closure.py \
    --device cuda \
    --dtype float64 \
    --no-show
