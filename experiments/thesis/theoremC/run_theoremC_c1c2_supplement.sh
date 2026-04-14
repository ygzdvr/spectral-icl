#!/bin/bash
#SBATCH --job-name=C1C2-supplement
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

# Experiment C1+C2 supplement: theorem-C objects not directly validated by
# run_theoremC_commutant_closure. Validates five items:
#   1. Lemma 3.5 band-rotation invariance (paired Monte Carlo, N_mc=10000,
#      20 U samples, both Q ∈ C(B) and Q ∉ C(B), L ∈ {1, 2, 4, 8}).
#   2. Grouped loss formula eq. (3.2) vs matrix-product path (1e-12 tol).
#   3. Induced metric ‖Q‖_F² = Σ_b m_b q_b² (1e-12 tol).
#   4. Corollary 3.9 endpoint recovery (singletons / m=8 / single band).
#   5. Unequal partition (4,4,8,16,32).
#
# Plan reference: EXPERIMENT_PLAN_FINAL.MD §7.1–§7.2 (supplements C1+C2).
# Theorem reference: thesis/theorem_c.txt — Lemma 3.5, Theorem 3.8, Cor 3.9.
#
# Item 1 dominates wall time (≈ 160 MC evaluations at N=10000 block-Haar
# samples); expect a few minutes on a single GPU. Items 2–5 are cheap.

set -euo pipefail

SCRIPT_DIR="/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling"

module --force purge 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true

cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/starter.sh"

nvidia-smi 2>/dev/null || true

echo "Running C1+C2 supplement: Lemma 3.5 / eq. (3.2) / metric / Cor 3.9 / unequal partition ..."
python -u scripts/thesis/theoremC/run_theoremC_c1c2_supplement.py \
    --device cuda \
    --dtype float64 \
    --no-show
