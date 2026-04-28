#!/bin/bash
#SBATCH --job-name=A-Lemma1-hidden-state
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Experiment A-Lemma1: direct N-dim hidden-state-closure validation.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §8 (theorem-A exact tier).
#
# A1 / A1b / A1-general go straight to the scalar recurrence of eq. (3),
# verifying the reduced (A_S, B_S) block but NOT Lemma 1's statement that
# the N-dim aligned hidden state h_μ^ℓ remains in the invariant
# two-channel subspace {W_x x_μ + λ w_y : λ ∈ ℝ}.
#
# This script constructs explicit (W_x, w_y, W_q, W_k, W_v, w_o) satisfying
# Assumption 1 with residual dim N = D + 10, runs the full N-dim
# residual-stream forward pass of eq. (2), and verifies
#   (1) N-dim readout w_o^⊤ h_μ^L ≡ scalar R0 recurrence, and
#   (2) at every depth ℓ and token μ,
#       ‖ h_μ^ℓ − W_x x_μ − Δ_μ^ℓ w_y ‖_2 / ‖ h_μ^ℓ ‖_2 ≤ 1e-10.
# Tested under gd_compatible and random train-supported masks.
# Deterministic; float64; acceptance is algebraic at machine precision.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running A-Lemma1: N-dim hidden-state closure..."
python -u scripts/thesis/theoremA/run_theoremA_lemma1_hidden_state.py \
    --device cuda \
    --dtype float64 \
    --no-show
