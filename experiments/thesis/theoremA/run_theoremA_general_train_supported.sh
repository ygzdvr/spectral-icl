#!/bin/bash
#SBATCH --job-name=A1-general-train-supported
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hd0216@princeton.edu

# Experiment A1-general: Theorem-1 exactness across ALL train-supported mixers
# + Proposition-3 necessity of GD-compatibility.
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §8.1 (A1 generalization).
#
# Extends A1 / A1b — which validate Theorem 1 and Corollary 1 only in the
# GD-compatible special case — to every train-supported structured mixer.
# Sweeps:
#   mask_kind  ∈ {gd_compatible, lower_triangular, random_dense,
#                 near_gd (ε ∈ {0.01, 0.1, 0.5})}
#   gamma_kind ∈ {identity, random_symmetric, random_nonsymmetric}
#   sigma_kind ∈ {isotropic, structured (Σ = diag(k^{-1}), Ω = diag(k^{-0.5}))}
#   (D, P, K, L) = 4 × 3 × 2 × 4  = 96 cells per (mask, γ, σ) family
#   Total: 6 × 3 × 2 × 96 = 1728 cells.
#
# Four prediction routes per cell: R0 (full hidden-state forward), R1
# (iterative reduced AB), R2 (matrix-power reduced AB), R3 (feature-space
# reduced-Γ predictor). Gates:
#   theorem1_pass  — max(err_R0_R1, err_R0_R2, err_R1_R2) ≤ 1e-10 in every cell.
#   corollary1_pass — max(err_R0_R3, err_R2_R3) ≤ 1e-10 on every GD cell.
#   necessity_pass — per non-GD mask kind: max_cells err_R2_R3 ≥ 1e-3.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running A1-general: Theorem-1 + Proposition-3 across all train-supported mixers..."
python -u scripts/thesis/theoremA/run_theoremA_general_train_supported.py \
    --device cuda \
    --dtype float64 \
    --no-show
