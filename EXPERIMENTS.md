# Experiment Catalog

Reference for all available experiments in `scripts/`. Each entry describes what the experiment measures, the figures it produces, and the data it saves. All outputs are written to `outputs/<script_name>/` with subdirectories `figures/`, `pdfs/`, `pt/`, `npz/`.

Run any experiment via its Slurm script in `experiments/`:

```bash
sbatch experiments/run_<name>.sh
```

---

## Table of Contents

1. [Baseline and Model Evaluation](#1-baseline-and-model-evaluation)
2. [Isotropic SGD Dynamics](#2-isotropic-sgd-dynamics)
3. [Pretrain Dynamics (Toy Models)](#3-pretrain-dynamics-toy-models)
4. [Linear Attention Dynamics](#4-linear-attention-dynamics)
5. [Depth Scaling Sweeps](#5-depth-scaling-sweeps)
6. [Reduced-Gamma Framework](#6-reduced-gamma-framework)
7. [Compute Scaling Laws](#7-compute-scaling-laws)
8. [Depth-vs-Alpha Phase Diagrams](#8-depth-vs-alpha-phase-diagrams)
9. [Prompt-Length (P_tr) Scaling](#9-prompt-length-p_tr-scaling)
10. [Loss Landscape Analysis](#10-loss-landscape-analysis)
11. [Covariance and OOD Generalization](#11-covariance-and-ood-generalization)

---

## 2. Isotropic SGD Dynamics

### `run_sgd_isotropic.py`

**What it measures:** Simple SGD dynamics in the isotropic (identity covariance) data regime. Sweeps over noise scale tau at multiple alpha values. Validates the basic SGD convergence behavior predicted by the isotropic DMFT.

**Figures:**
- Per-alpha figure with loss curves for each tau value.

**Data:**
- Per-alpha `.npz` with loss arrays for each tau.

---

### `run_sgd_rmt_isotropic.py`

**What it measures:** SGD dynamics with random matrix theory (RMT) corrections in the isotropic regime. Runs four predefined sweep experiments, each varying one control variable (tau, alpha, or kappa) while fixing the other two. Tests how sample complexity ratio, spectral exponent, and condition number affect learning dynamics.

**Experiments:**
1. Vary tau (alpha=1.0, kappa=4.0)
2. Vary alpha (tau=1.0, kappa=4.0)
3. Vary kappa (tau=1.0, alpha=1.0)
4. Vary tau (alpha=8.0, kappa=2.0) with semilogy scale

**Figures:**
- One figure per sweep showing loss trajectories for each parameter value.

**Data:**
- One `.npz` per sweep with loss arrays.

---

## 3. Pretrain Dynamics (Toy Models)

### `run_pretrain_dynamics.py`

**What it measures:** Toy-model pretraining dynamics under two analytical parameterizations: a single-variable model and a six-variable (two-var factored) model. Sweeps across depth values L to study how depth affects convergence rate. Compares empirical trajectories to theoretical power-law scaling predictions: t^{-7beta/(7beta+2)} for single-var and t^{-5beta/(5beta+2)} for six-var.

**Figures:**
- `two_var_loss_trajectories` -- Log-log loss vs. t for each L, with t^{-5beta/(5beta+2)} theory line.
- `two_var_wy_wo` -- Semilog product w_y(t) * w_o(t) vs. t for each L.
- `two_var_wx2_wk_wq_wv` -- Log-log product w_x^2 * w_k * w_q * w_v vs. t, with t^{5/(5beta+2)} theory.
- `depth_scaling_pretrain_theory_ICL_powerlaw` -- Log-log single-var loss vs. t for each L, with both t^{-7beta/(7beta+2)} and t^{-beta} theory lines.
- `depth_scaling_final_loss_powerlaw` -- Log-log final loss vs. L, with L^{-beta} theory line.

**Data:**
- `pretrain_dynamics_losses.npz` -- Per-L loss arrays, parameter histories (w_x, w_y, w_k, w_q, w_v, w_o), spectrum, teacher.

---

### `run_beta_sweep_dynamics.py`

**What it measures:** Two-variable (six-parameter) dynamics across different spectral decay exponents beta. Probes how the spectral difficulty beta controls the learning rate of attention parameters and the overall loss decay. Overlays per-beta theoretical power-law reference lines t^{5/(5beta+2)}.

**Figures:**
- `beta_sweep_wy_wo` -- Semilog w_o(t) * w_y(t) vs. t for each beta.
- `beta_sweep_wx2_wk_wq_wv` -- Log-log product w_x^2 * w_k * w_q * w_v with per-beta theory dashes.
- `beta_sweep_loss` -- Log-log loss trajectories for each beta.

**Data:**
- `beta_sweep_dynamics_losses.npz` -- Per-beta loss arrays and all six parameter histories.

---

## 4. Linear Attention Dynamics

### `run_linear_attention_dynamics.py`

**What it measures:** Compares trained isotropic linear-attention model dynamics to the four-variable reduced theory predictions. Trains actual transformer parameters (W_x, W_q, W_k, W_v) at each depth L, then runs the closed-form reduced-theory ODE and overlays both. Validates that the DMFT theory accurately predicts both loss decay and individual weight norm evolution.

**Figures:**
- `weight_norm_dynamics` -- Weight norms |W_i(t)| for trained model (solid) vs. theory (dashed black) at the deepest L.
- `isotropic_theory_vs_expt` -- Normalized loss L(t)/L(0) for each L: experiment (solid) vs. theory (dashed black).

**Data:**
- `linear_attention_dynamics_data.npz` -- Per-L loss, theory loss, weight norms, theory weight norms.

---

### `run_dim_free_dynamics.py`

**What it measures:** Dimension-free linear-attention dynamics with rotated Bernoulli data sampling (`spec_rotate` mode). Verifies that the loss scaling exponent t^{-5beta/(5beta+2)} holds in the dimension-free (N=d) regime and examines how individual weight norms (W_x, W_q, W_k, W_v) evolve.

**Figures:**
- `dim_free_theory_weight_decoupled_vs_expt` -- Log-log loss vs. t for each L, with t^{-5beta/(5beta+2)} theory line.
- `dim_free_weight_norms_raw` -- Raw weight norms |W_i(t)| for deepest L.
- `dim_free_weight_norms_normalized` -- Normalized weight norms (W_x/sqrt(2), W_v/sqrt(d)) for deepest L.

**Data:**
- `dim_free_dynamics_data.npz` -- Per-L loss arrays, weight norm histories, spectrum, teacher.

---

## 5. Depth Scaling Sweeps

### `run_depth_scaling_nonrotate.py`

**What it measures:** Depth scaling behavior in the non-rotated (FS = feature-structured) data regime. Trains a decoupled attention model at each depth L with `sample_mode="spec"` (no random rotation), and tracks smoothed pretrain loss. Tests the finite-sample (non-isotropic) depth scaling law.

**Figures:**
- `depth_scaling_ICL_powerlaw_nonrotate` -- Log-log smoothed pretrain loss vs. steps for each L.

**Data:**
- `depth_scaling_ICL_powerlaw_nonrotate_losses.npz` -- Per-L raw and smoothed loss, step arrays.
- `depth_scaling_ICL_powerlaw_nonrotate_artifacts.pt` -- Config, spectrum, teacher, all loss tensors.

---

### `run_powerlaw_depth_sweep.py`

**What it measures:** Depth scaling in the rotated (RRS = random rotation structured) data regime. Uses `sample_mode="spec_rotate"` with `random_rotate=True`. Compares how the rotation of the data covariance eigenstructure affects depth scaling relative to the non-rotated case.

**Figures:**
- `depth_scaling` -- Log-log smoothed pretrain loss vs. steps for each L.

**Data:**
- `losses.npz` -- Per-L loss arrays.
- `artifacts.pt` -- Config, spectrum, teacher, all loss tensors.

---

### `run_offline_depth_sweep.py`

**What it measures:** Depth scaling with fixed (offline) data, tracking both train and test loss separately. Uses `online=False` so the same batch is reused across steps. Reveals the train/test gap as a function of depth and detects potential overfitting behavior.

**Figures:**
- `test_train` -- Side-by-side log-log subplots: test loss (left) and train loss (right) vs. steps for each L.

**Data:**
- `losses.npz` -- Per-L test and train loss arrays.
- `artifacts.pt` -- Config, spectrum, teacher, all test and train loss tensors.

---

### `run_softmax_depth_sweep.py`

**What it measures:** Depth scaling with softmax (SDPA) attention instead of raw linear attention. Tests whether the depth scaling predictions from linear attention theory transfer to the nonlinear softmax attention regime.

**Figures:**
- `softmax_depth_sweep` -- Loss vs. steps for each L (linear scale).

**Data:**
- `softmax_depth_sweep_losses.npz` -- Per-L loss arrays, final losses averaged over last 10 steps.

---

## 6. Reduced-Gamma Framework

These experiments use the reduced-gamma (structured SGD with RMT) dynamics framework, which models attention learning through an eigenvalue-level gamma parameter that tracks how well each spectral mode is learned.

### `run_reduced_gamma_dynamics.py`

**What it measures:** Core reduced-gamma SGD dynamics plus the loss-vs-gamma landscape at multiple depths. Optionally runs a time-dynamics experiment (with `--run-sgd`), then always computes the analytical loss landscape L(gamma) for each depth L. Verifies the t^{-beta/(2+beta)} temporal scaling and the L^{-beta} depth scaling floor.

**Figures:**
- `reduced_gamma_sgd_loss` (if `--run-sgd`) -- Log-log loss vs. t with t^{-beta/(2+beta)} theory.
- `loss_landscape_powerlaw` -- Log-log loss vs. gamma for each L.
- `min_loss_vs_depth_powerlaw` -- Log-log optimal loss vs. L with L^{-beta} theory.

**Data:**
- `reduced_gamma_sgd_data.npz` (if `--run-sgd`) -- Loss, mean eigenvalues, variance eigenvalues.
- `reduced_gamma_landscape.npz` -- Gamma grid, per-L landscape losses, min losses, spectrum, teacher.

---

### `run_reduced_gamma_depth_sweep.py`

**What it measures:** Reduced-gamma dynamics across depth values at fixed width N. Produces two complementary views: loss vs. training steps and loss vs. total compute (C = L * t). Validates both the temporal scaling t^{-beta/(2+beta)} and the compute-optimal scaling C^{-beta/(3+beta)}.

**Figures:**
- `losses_gamma_model_vary_L` -- Log-log loss vs. steps for each L, with t^{-beta/(2+beta)} theory.
- `losses_gamma_model_compute_vary_L` -- Log-log loss vs. compute C = L*t for each L, with C^{-beta/(3+beta)} theory.

**Data:**
- `reduced_gamma_depth_sweep.npz` -- Per-L loss arrays, spectrum, teacher.

---

### `run_reduced_gamma_beta_sweep.py`

**What it measures:** Reduced-gamma dynamics across different spectral exponents beta at fixed depth and width. For each beta, trains on a matched power-law problem and overlays the per-beta theory curve t^{-beta/(2+beta)}. Demonstrates that the scaling exponent continuously varies with the data difficulty.

**Figures:**
- `losses_vary_beta` -- Log-log loss vs. steps for each beta, with per-beta t^{-beta/(2+beta)} theory (dashed black).

**Data:**
- `reduced_gamma_beta_sweep.npz` -- Per-beta loss arrays.

---

### `run_decoupled_layers.py`

**What it measures:** Decoupled reduced-gamma dynamics where each layer has an independent gamma parameter (no weight sharing across layers). Tests the t^{-beta/(beta+2)} scaling prediction under the decoupled parameterization.

**Figures:**
- `gamma_decoupled_layers_dynamics` -- Log-log loss vs. steps for each L, with t^{-beta/(beta+2)} theory (red dashed).

**Data:**
- `decoupled_layers_data.npz` -- Per-L loss arrays.

---

## 7. Compute Scaling Laws

### `run_compute_scaling.py`

**What it measures:** Canonical compute scaling in three regimes: (1) fixed depth L=1 with varying width N, (2) fixed width N with varying depth L, (3) joint N-L scaling with L = int(N/8). Plots loss as a function of total compute C = N^2 * L * t. Tests whether the curves collapse when compute is the horizontal axis.

**Figures:**
- `compute_scaling_fixed_L` -- Log-log loss vs. compute (N^2 * t) for each N at L=1.
- `compute_scaling_fixed_N` -- Log-log loss vs. compute (L * N^2 * t) for each L at fixed N.
- `compute_scaling_joint_NL_scaling` -- Log-log loss vs. compute (L * N^2 * t) for joint N-L configs.

**Data:**
- `compute_scaling_data.npz` -- Per-config loss arrays for all three sweeps.

---

### `run_compute_scaling_depth.py`

**What it measures:** Depth-only compute scaling at fixed model width N. Computes the L -> infinity loss floor via `compute_loss_inf_depth`, then sweeps across L values. Visualizes both the raw loss dynamics and the compute-normalized view. Shows that increasing depth alone saturates at a width-dependent floor.

**Figures:**
- `compute_scaling_fixed_N_more_depth` -- Log-log loss vs. compute (L * N^2 * t) for each L, with L -> infinity floor (red dashed).
- `depth_sweep_raw_loss` -- Log-log raw loss vs. steps for each L.

**Data:**
- `compute_scaling_depth_data.npz` -- Per-L loss, infinite-depth floor, spectrum, teacher.

---

### `run_compute_scaling_width.py`

**What it measures:** Width-only compute scaling at fixed depth L=1. Sweeps model width N and plots loss vs. compute C = N^2 * t. Computes the N -> infinity loss floor. Shows that increasing width alone saturates at a depth-dependent floor.

**Figures:**
- `compute_scaling_fixed_L_more_width` -- Log-log loss vs. compute (N^2 * t) for each N, with N -> infinity floor (red dashed).

**Data:**
- `compute_scaling_width_data.npz` -- Per-N loss, width-infinity floor, spectrum, teacher.

---

### `run_compute_scaling_joint.py`

**What it measures:** Joint N-L compute scaling with a theory-guided allocation rule L = int((N / divisor)^alpha). Includes both the empirical sweep and the `solve_n_final` analytical theory curve (loss floor vs. N at infinite time). Overlays the compute-optimal exponent C^{-nu*beta / ((3+beta)*nu + 2)}.

**Figures:**
- `compute_scaling_joint_linear_more_NL` -- Log-log loss vs. compute for each N, with theory power-law reference.

**Data:**
- `compute_scaling_joint_data.npz` -- Per-N loss arrays, theory curve (N_th, loss_vs_N), loss_8 reference.

---

## 8. Depth-vs-Alpha Phase Diagrams

### `run_isotropic_depth_vs_alpha.py`

**What it measures:** Joint sweep over prompt-to-dimension ratio alpha = P_tr/d and model depth L in the isotropic (ISO) data regime. Computes DMFT theory curves for final loss vs. alpha at each depth, and overlays empirical training results with error bars. Produces phase-diagram-style plots showing how the loss surface depends on the data ratio and model depth.

**Figures:**
- `depth_vs_alpha_isotropic` -- Final loss vs. alpha (semilog-x): theory curves (solid) + empirical errorbars for each L.
- `train_dynamics_vary_alpha` -- Loss vs. steps for each alpha at deepest L.
- `train_dynamics_vary_L` -- Loss vs. steps for each L at a fixed alpha.

**Data:**
- `isotropic_depth_vs_alpha_losses.npz` -- Theory alpha grid, per-(P_tr, L) loss arrays, mean/std summary.
- `isotropic_depth_vs_alpha_artifacts.pt` -- Full config, spectrum, teacher, all tensors.

---

### `run_unrestricted_depth_vs_alpha.py`

**What it measures:** Same depth-vs-alpha sweep as above but with `unrestricted=True` (unconstrained attention parameterization). Compares unrestricted learning to the restricted case. Produces both a theory-only plot across more L values and a combined theory+experiment plot.

**Figures:**
- `unrestricted_depth_vs_alpha_theory` -- Final loss vs. alpha theory curves for extended L range.
- `unrestricted_depth_vs_alpha_combined` -- Theory + empirical errorbars for each L.
- `unrestricted_depth_scaling_P{P_tr}` (one per P_tr) -- Log-log smoothed pretrain loss vs. steps for each L.

**Data:**
- `unrestricted_depth_vs_alpha_losses.npz` -- Same structure as isotropic version.
- `unrestricted_depth_vs_alpha_artifacts.pt` -- Full config and all tensors.

---

## 9. Prompt-Length (P_tr) Scaling

### `run_ptr_scaling.py`

**What it measures:** How ICL loss scales with the number of training prompts P_tr at fixed model depth. Trains one model per P_tr value, plots smoothed loss dynamics, and compares the final-loss-vs-alpha curve to two analytical limits: the L=1 theory 1 - (1 + 1/alpha)^{-1} and the L=infinity theory max(0, 1 - alpha).

**Figures:**
- `pretrain_loss` -- Log-log smoothed pretrain loss vs. steps for each P_tr.
- `final_loss` -- Semilog-x final loss vs. alpha = P_tr/d, with L=1 and L=infinity theory curves.

**Data:**
- `losses.npz` -- Per-P_tr raw and smoothed loss, alpha points, theory curves.
- `artifacts.pt` -- Config, spectrum, teacher, all loss tensors.

---

## 10. Loss Landscape Analysis

### `run_loss_landscape.py`

**What it measures:** Analytical loss landscape L(gamma) as a function of the normalized gamma parameter, using `visualize_loss_landscape`. Three experiments: (1) single diagnostic at alpha=1.25, L=10; (2) vary depth L at alpha=0.5; (3) vary noise sigma at L=16, alpha=1.125. CPU-only (no `--device` flag).

**Figures:**
- `loss_landscape_single` -- Semilogy loss vs. gamma for a single (alpha, L) config.
- `loss_landscape_gamma_alpha_0.5` -- Loss vs. gamma/L for each depth L.
- `loss_landscape_gamma_vary_sigma` -- Loss vs. gamma/L for each noise level sigma.

**Data:**
- `loss_landscape_data.npz` -- Gamma grid, per-L and per-sigma loss arrays.

---

### `run_loss_vs_w.py`

**What it measures:** Analytical loss as a function of a scalar weight w at multiple depths L, using the `loss_landscape` utility. Visualizes how the loss surface changes with depth: deeper models have sharper, lower minima. Reports the optimal w* and minimum loss for each L.

**Figures:**
- `loss_vs_w` -- Log-log loss vs. w for each L.

**Data:**
- `loss_vs_w_losses.npz` -- w grid, per-L loss arrays, spectrum, teacher.

---

## 11. Covariance and OOD Generalization

### `run_fixed_covariance.py`

**What it measures:** Two-part experiment: (1) Fixed-covariance depth sweep using `reduced_gamma_structured_fixed_sgd_rmt_isotropic_dynamics`, tracking both loss trajectories and per-eigenvalue gamma evolution; (2) Out-of-distribution (OOD) loss under covariance rotation via `ood_loss_fixed_covariance`. Validates the eigenvalue-level theory gamma_k(t) ~ log(1 + 4*eta*lambda_k^3 * w*_k^2 * t) / (2*lambda_k) and measures generalization fragility under distribution shift.

**Figures:**
- `losses_fixed_covariance` -- Log-log loss vs. t for each L, with t^{-beta/(nu + nu*beta + 1)} theory.
- `eig_evolution_fixed_covariance` -- Log-log gamma_k(t) for selected eigenvalue indices k, experiment (colored) vs. theory (dashed black).
- `ood_loss_fixed_covariance` -- Semilogy OOD loss vs. rotation angle theta for each L.

**Data:**
- `fixed_covariance_data.npz` -- Per-L loss, OOD loss, eigenvalue histories, theta grid, spectrum, teacher.

---

## Quick Reference: Script to Slurm Mapping

| Script | Slurm Job | GPU |
|--------|-----------|-----|
| `run_beta_sweep_dynamics.py` | `run_beta_sweep_dynamics.sh` | Yes |
| `run_compute_scaling.py` | `run_compute_scaling.sh` | Yes |
| `run_compute_scaling_depth.py` | `run_compute_scaling_depth.sh` | Yes |
| `run_compute_scaling_joint.py` | `run_compute_scaling_joint.sh` | Yes |
| `run_compute_scaling_width.py` | `run_compute_scaling_width.sh` | Yes |
| `run_decoupled_layers.py` | `run_decoupled_layers.sh` | Yes |
| `run_depth_scaling_nonrotate.py` | `run_depth_scaling_nonrotate.sh` | Yes |
| `run_dim_free_dynamics.py` | `run_dim_free_dynamics.sh` | Yes |
| `run_fixed_covariance.py` | `run_fixed_covariance.sh` | Yes |
| `run_isotropic_depth_vs_alpha.py` | `run_isotropic_depth_vs_alpha.sh` | Yes |
| `run_linear_attention_dynamics.py` | `run_linear_attention_dynamics.sh` | Yes |
| `run_loss_landscape.py` | `run_loss_landscape.sh` | No (CPU) |
| `run_loss_vs_w.py` | `run_loss_vs_w.sh` | Yes |
| `run_offline_depth_sweep.py` | `run_offline_depth_sweep.sh` | Yes |
| `run_powerlaw_depth_sweep.py` | `run_powerlaw_depth_sweep.sh` | Yes |
| `run_pretrain_dynamics.py` | `run_pretrain_dynamics.sh` | Yes |
| `run_ptr_scaling.py` | `run_ptr_scaling.sh` | Yes |
| `run_reduced_gamma_beta_sweep.py` | `run_reduced_gamma_beta_sweep.sh` | Yes |
| `run_reduced_gamma_depth_sweep.py` | `run_reduced_gamma_depth_sweep.sh` | Yes |
| `run_reduced_gamma_dynamics.py` | `run_reduced_gamma_dynamics.sh` | Yes |
| `run_sgd_isotropic.py` | `run_sgd_isotropic.sh` | Yes |
| `run_sgd_rmt_isotropic.py` | `run_sgd_rmt_isotropic.sh` | Yes |
| `run_softmax_depth_sweep.py` | `run_softmax_depth_sweep.sh` | Yes |
| `run_unrestricted_depth_vs_alpha.py` | `run_unrestricted_depth_vs_alpha.sh` | Yes |

---

## Key Scaling Law Exponents Tested

| Exponent | Formula | Where Tested |
|----------|---------|--------------|
| Temporal (6-var) | t^{-5beta/(5beta+2)} | `run_pretrain_dynamics`, `run_beta_sweep_dynamics`, `run_dim_free_dynamics` |
| Temporal (1-var) | t^{-7beta/(7beta+2)} | `run_pretrain_dynamics` |
| Temporal (gamma) | t^{-beta/(2+beta)} | `run_reduced_gamma_dynamics`, `run_reduced_gamma_depth_sweep`, `run_reduced_gamma_beta_sweep`, `run_decoupled_layers` |
| Depth floor | L^{-beta} | `run_pretrain_dynamics`, `run_reduced_gamma_dynamics` |
| Compute-optimal | C^{-beta/(3+beta)} | `run_reduced_gamma_depth_sweep` |
| Joint compute | C^{-nu*beta/((3+beta)*nu+2)} | `run_compute_scaling_joint` |
| Fixed-cov temporal | t^{-beta/(nu+nu*beta+1)} | `run_fixed_covariance` |
