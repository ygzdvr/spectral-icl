# Spectral Scaling for Transformer In-Context Learning

This repository contains PyTorch research code for studying how transformer-like linear attention models solve in-context linear regression, with an emphasis on:

- spectral power-law data structure,
- depth/width scaling behavior,
- reduced dynamical models (Gamma and toy models),
- and comparison to analytic theory (including DMFT-style predictions).

The codebase mixes:

- reusable modules in `data`, `models`, `layers`, `dynamics`, `utils`, `configs`,
- and many CLI experiment drivers in `scripts/`.

## What This Repo Implements

At a high level, experiments generate synthetic linear-regression sequences:

- inputs `x_i` in `R^d`,
- labels `y_i = <x_i, beta>/sqrt(d) + noise`,
- and a masked final/query label that the model must predict from context.

Two main modeling tracks are included:

- **Hand-coded analytical attention models**: directly structured weights that implement iterative regression updates.
- **Trainable models/dynamics**: SGD-trained attention parameters under isotropic and power-law covariance assumptions, plus reduced scalar/matrix surrogates.

## Repository Structure

- `configs/` - dataclasses for data, training, and evaluation defaults.
- `data/` - synthetic ICL batch generation.
- `layers/` - attention block(s).
- `models/` - compact transformer implementation.
- `dynamics/` - core research logic (analytical models, SGD dynamics, reduced theories, sweeps).
- `utils/` - parsing, device handling, plotting/sweep helpers, analysis helpers.
- `scripts/` - runnable experiment entrypoints with argparse.
- `outputs/` - generated artifacts (`.png`, `.npz`, `.pt`, etc.).
- `logs/` - batch/HPC logs.
- `install.sh`, `starter.sh`, `submit.sh` - environment and cluster workflow scripts.

## Environment Setup

This project uses `uv` and a local `.venv`.

```bash
bash install.sh
source starter.sh
```

`install.sh` will:

- create/reuse `.venv`,
- set project-local cache dirs under `.cache/`,
- `uv sync` dependencies from `pyproject.toml`/`uv.lock`,
- run a lightweight smoke test import/forward pass.

Dependencies from `pyproject.toml`:

- `torch`
- `numpy`
- `matplotlib`
- `seaborn`

Python requirement: `>=3.10`.

## Quick Start Commands

Core examples (CPU/GPU via `--device`):

```bash
python scripts/run_pretrain_icl_powerlaw.py --device cuda --d 60 --layers 4 --alpha 1.5
python scripts/run_hand_coded_model_eval.py --d 100 --batch 50
python scripts/train_first_cells.py --depth 4 --steps 1000
python scripts/run_isotropic_depth_vs_alpha.py --p-trs 50,85,120 --lvals 1,2,4,8
```

Most scripts support:

- `--device` (`auto`, `cpu`, `cuda`, `cuda:0`, ...)
- `--dtype` (`float32`, `float64`)
- output paths (`--plot-path`, `--losses-path`, `--artifacts-path` or `--output-dir`)
- `--no-show` for headless runs

## Data and Model Conventions

- Sequence tensors are generally `[batch, seq, feature]`.
- ICL token packing often uses `[x; y]` with size `d+1`.
- Train/test split is positional: first `P_tr` are context points, final `P_test` are query/test points.
- Attention masks prevent using test labels or attending to test positions.
- Many modules use deterministic local RNG seeds (`torch.Generator`) for reproducibility.

## Core Packages (All Source `.py` Modules)

The source tree contains 60 Python files across packages + scripts. The list below covers all non-venv/non-cache Python modules in this repository.

### `configs/`

- `configs/__init__.py` - re-exports all config dataclasses.
- `configs/data_configs.py` - `LinearICLConfig` for synthetic batch generation.
- `configs/eval_configs.py` - eval configs:
  - `HandCodedEvalConfig`
  - `HardPowerLawDepthConfig`
  - `OODCovarianceEvalConfig`
  - `RandomInitCovarianceEvalConfig`
- `configs/train_configs.py` - train/sweep configs:
  - `PretrainICLPowerLawConfig`
  - `DecoupledTrainModelConfig`
  - `IsotropicDepthAlphaSweepConfig`
  - `SampleMode = {"iid","spec","spec_rotate","gauss_rotate"}`

### `data/`

- `data/__init__.py` - exports data APIs.
- `data/icl_linear_regression.py` - synthetic linear ICL data generation:
  - `generate_linear_icl_batch(...)`
  - `make_train_test_batches(...)`

### `layers/`

- `layers/__init__.py` - exports attention layer.
- `layers/attn.py` - `Attn` single-head block with:
  - raw linear attention mode (`use_softmax=False`)
  - SDPA softmax mode (`use_softmax=True`)

### `models/`

- `models/__init__.py` - exports `SimpleTransformer`.
- `models/transformer.py` - minimal transformer:
  - lazy input projection
  - pre-norm residual attention stack
  - residual scaling by `beta/depth`
  - scalar token output head

### `utils/`

- `utils/__init__.py` - public utility exports.
- `utils/parsing.py` - `parse_int_list`, `parse_float_list`.
- `utils/device.py` - `resolve_device`, `is_cuda_oom`.
- `utils/powerlaw.py` - `make_powerlaw_spec_and_wstar`.
- `utils/smoothing.py` - `moving_average`.
- `utils/analysis.py` - `compute_loss_inf_depth`, `loss_landscape`.
- `utils/sgd_sweeps.py` - reusable sweep runners for isotropic/RMT SGD scripts.

### `dynamics/`

Core research logic. Each module implements one aspect of the paper's theory or its extensions.

- `dynamics/__init__.py` - central export surface for most dynamics functions.
- `dynamics/linear_icl_dynamics.py` - Hand-coded analytical attention model that implements optimal preconditioned in-context GD (Section 2.2 / Appendix B.2). Constructs structured weight matrices {W_x, W_y, W_q, W_k, W_v, w_out} encoding the [x, Delta] residual-stream representation. Variants:
  - coupled / decoupled / frozen-embedding / softmax-frozen modes
  - synthetic task samplers for ISO/FS data
  - `run_hand_coded_eval`, `run_hard_power_law_depth_eval`
- `dynamics/pretrain_icl_powerlaw.py` - Main pretraining engine. Implements all three data settings from the paper:
  - `sample_data` (ISO, "iid"), `sample_data_spec` (FS, "spec"), `sample_data_spec_rotate` (RRS, "spec_rotate"), `sample_data_gauss_rotate` (Wishart variant, "gauss_rotate")
  - `make_powerlaw_problem` builds spectrum lambda_k ~ k^{-alpha} with source/capacity structure
  - `train_model` / `train_model_softmax` - online SGD training loops
  - Sweep runners for depth, P_tr, and joint sweeps
  - `isotropic_dmft` - solves the DMFT fixed-point equations for the ISO setting (Result 1 / Appendix C.2)
- `dynamics/linear_attention_dynamics.py` - Full weight dynamics for Result 9. Trains separate {W_x, W_q, W_k, W_v} with frozen {w_y, w_o}. Includes:
  - Isotropic and dimension-free (power-law) variants
  - Reduced 4-variable theory solver implementing the w^5 reparameterization
  - Predicts time exponent t^{-5beta/(5beta+2)}
- `dynamics/reduced_gamma_dynamics.py` - Reduced-Gamma model (Section 2.2): trains a D x D matrix Gamma directly via SGD on RRS data. The scalar reduction gamma(t) drives the loss landscape. Includes loss landscape computation.
- `dynamics/reduced_gamma_fixed_dynamics.py` - Reduced-Gamma on FS setting (Section 3.2). Implements eigenvalue decoupling (Result 4) and OOD rotation loss (Result 5): Sigma' = exp(theta S) Sigma exp(-theta S).
- `dynamics/reduced_gamma_decoupled_dynamics.py` - Untied layers: separate gamma^l per layer (Appendix G.1). Tests permutation symmetry and balance condition gamma^l(t) = gamma(t).
- `dynamics/sgd_isotropic_dynamics.py` - Exact SGD dynamics for shallow ISO model (Appendix C.1, Result 10). Computes the linear recursion C(t+1) = a*C(t) + b with dependence on (eta, alpha, kappa, tau). RMT variants use Marchenko-Pastur eigenvalue predictions.
- `dynamics/toy_model_dynamics.py` - Scalar and two-variable toy pretraining dynamics. Minimal models for the gradient flow ODE d gamma/dt = beta * gamma^{-beta-1}.
- `dynamics/solve_n_final.py` - Newton solver for the asymptotic N-dependent loss floor. Solves the width bottleneck equation from Appendix F.5: sum_k lambda_k / (i*omega + lambda_k) = N/D.
- `dynamics/ood_covariance.py` - OOD covariance generalization evaluation (Result 5). Tests brittleness under heterogeneous task covariances.
- `dynamics/random_init_covariance.py` - Random-initialization covariance evaluation. Studies loss landscape before training.

### Package Root

- `__init__.py` - top-level exports combining config/data/dynamics/model APIs.

## Script Catalog (`scripts/*.py`)

Each script is a CLI experiment entrypoint. Most write `.png` figures plus `.npz`/`.pt` artifacts. Scripts are organized by which aspect of the scaling law theory they test.

### Pretraining / Depth / Scaling (RRS and ISO settings)

These scripts train the full attention model on ICL tasks and study how depth `L`, context length `P`, and spectral exponents affect the loss.

- `scripts/run_pretrain_icl_powerlaw.py` - Full pretraining run on power-law **RRS** data. Trains all attention parameters with SGD under `spec_rotate` sampling. Plots loss curves with theoretical power-law reference lines (`t^{-beta/(2+beta)}` for reduced-Gamma, `t^{-7beta/(2+7beta)}` for full coupled parameterization). Core experiment for verifying Result 8/9 time exponents.
- `scripts/run_powerlaw_depth_sweep.py` - Depth sweep over `L` on **RRS** rotated power-law data. Tests the `L^{-beta}` depth scaling law (Result 8) by training models at multiple depths and comparing final losses.
- `scripts/run_depth_scaling_nonrotate.py` - Depth sweep on **FS** (fixed covariance, no rotation). Uses `spec` sampling mode with normalized power-law spectrum. Tests Result 3: depth should be unnecessary at long contexts for fixed covariance.
- `scripts/run_offline_depth_sweep.py` - Offline (fixed batch, not online SGD) variant of the non-rotated depth sweep. Tests whether overfitting effects from repeated data change the depth scaling.
- `scripts/run_ptr_scaling.py` - Sweep over context length `P_tr` at fixed depth. Tests the `P^{-nu*beta}` context scaling law from Result 8.
- `scripts/run_isotropic_depth_vs_alpha.py` - Joint sweep over `(P_tr, L)` grid on **ISO** data. Compares trained model losses against DMFT theory curves (`isotropic_dmft`). Tests Results 1-2: depth helps at finite alpha but is unnecessary as alpha -> infinity.
- `scripts/run_unrestricted_depth_vs_alpha.py` - Same as above but with all 6 parameters trainable (unrestricted), not just decoupled. Tests whether the unrestricted parameterization changes the ISO depth-vs-alpha landscape.
- `scripts/run_softmax_depth_sweep.py` - Depth sweep with **softmax attention** + Adam. Tests Section 5.2: whether the qualitative depth separation from linear attention theory persists under nonlinear attention.

### Analytical / Hand-Coded Evaluations

These scripts evaluate the analytically-constructed weight matrices (the "hand-coded" model that implements optimal preconditioned GD) rather than training.

- `scripts/run_hand_coded_model_eval.py` - Evaluates the hand-coded analytical model (`init_hand_coded_params`) on isotropic data. Provides the baseline loss that trained models should approach.
- `scripts/run_hard_power_law_depth.py` - Hand-coded model on power-law covariance data across depths. Studies how the analytical solution interacts with spectral structure and depth.
- `scripts/run_ood_covariance_generalization.py` - OOD evaluation of hand-coded model under heterogeneous task covariances (exponents sampled from Uniform(0, exp_scale)). Tests Result 5: brittleness of FS solutions to distribution shift.
- `scripts/run_random_init_covariance.py` - Evaluates the hand-coded model with random Gaussian weights (not optimal). Studies the loss landscape at random initialization before any training.

### Reduced-Gamma Model and Compute Scaling

These scripts work with the reduced-Gamma parameterization (a single D x D matrix Gamma, or its scalar reduction gamma), which is the key theoretical object from the paper (Section 2.2). The compute scaling scripts test Result 8's separable scaling law.

- `scripts/run_reduced_gamma_dynamics.py` - Trains the reduced Gamma matrix via SGD on **RRS** power-law data. Optionally computes and plots the loss landscape over gamma. Core experiment for the Gamma-model dynamics.
- `scripts/run_reduced_gamma_depth_sweep.py` - Depth sweep over `L` in the reduced-Gamma model. Tests the `L^{-beta}` scaling directly on the simplified model.
- `scripts/run_reduced_gamma_beta_sweep.py` - Sweep over source exponent `beta` in the reduced-Gamma model. Tests how the power-law exponents in the scaling law depend on source/capacity conditions.
- `scripts/run_decoupled_layers.py` - Decoupled-layer reduced-Gamma dynamics: separate `gamma^l` per layer instead of tied weights. Tests Result from Appendix G.1: that balanced init leads to `gamma^l(t) = gamma(t)` for all l (permutation symmetry).
- `scripts/run_fixed_covariance.py` - Reduced-Gamma model on **FS** (fixed covariance). Includes OOD evaluation under rotations `Sigma' = exp(theta S) Sigma exp(-theta S)`. Tests Results 4-5: eigenvalue decoupling and brittleness.
- `scripts/run_compute_scaling.py` - Width-scaling sweep in the reduced-Gamma setting. Tests how finite width `N` bottlenecks the loss.
- `scripts/run_compute_scaling_width.py` - Explicit width (`N`) scaling sweep. Tests the `N^{-nu*beta}` width scaling law from Result 8.
- `scripts/run_compute_scaling_depth.py` - Depth scaling with infinite-depth floor estimation. Computes asymptotic loss floor via Newton solver (`solve_n_final`).
- `scripts/run_compute_scaling_joint.py` - Joint width + depth scaling analysis. Tests the compute-optimal shape prediction `L ~ N^nu` from Result 8 by sweeping N and L simultaneously.
- `scripts/run_loss_vs_w.py` - Closed-form loss landscape over a scalar weight grid. Visualizes the 1D loss function that gradient flow descends (as in Result 1/6).
- `scripts/run_loss_landscape.py` - 2D reduced-Gamma landscape visualization over (`lambda`, `gamma`) grid. Shows how spectral structure shapes the optimization landscape.

### Full Linear Attention Dynamics (Result 9)

These scripts train the full set of attention weight matrices {W_x, W_k, W_q, W_v} separately (not the reduced-Gamma shortcut), testing the reparameterization theory from Result 9.

- `scripts/run_linear_attention_dynamics.py` - Compares isotropic linear-attention training dynamics against the 4-variable reduced theory. Tests weight balancing and the `gamma -> w^5` reparameterization prediction.
- `scripts/run_dim_free_dynamics.py` - Dimension-free linear-attention dynamics across depth values on power-law data. Tests the `t^{-5beta/(5beta+2)}` time exponent from Result 9 in the frozen-embedding setting.

### Isotropic / RMT SGD and Toy Dynamics

These scripts study the SGD dynamics in detail (finite batch effects, SGD noise floor) and simplified toy models for building intuition.

- `scripts/run_sgd_isotropic.py` - Isotropic SGD experiments across `(tau, alpha)` grid. Tests Result 10 (Appendix C.1): the exact SGD recursion for shallow L=1 ISO models, including dependence on batch size tau and context ratio alpha.
- `scripts/run_sgd_rmt_isotropic.py` - RMT (random matrix theory) isotropic SGD sweep. Tests the Marchenko-Pastur-based theoretical predictions against finite-dimensional simulations.
- `scripts/run_pretrain_dynamics.py` - Toy scalar/two-variable pretraining dynamics. Minimal models for understanding the interplay of time and depth exponents.
- `scripts/run_beta_sweep_dynamics.py` - Beta sweep for two-variable toy dynamics. Studies how the source exponent beta controls the loss power-law across the simplified dynamical system.

### SimpleTransformer Training

- `scripts/train_first_cells.py` - Trains a `SimpleTransformer` (with LayerNorm, optional softmax attention) directly on synthetic ICL data. Configurable depth/width/heads. Serves as a bridge between the analytical linear-attention theory and more realistic architectures.

## Typical Outputs

Artifacts are commonly saved under `outputs/` (or custom paths):

- `*.png` - plots of losses, sweeps, landscapes.
- `*.npz` - numeric loss trajectories and sweep payloads.
- `*.pt` - saved tensors/checkpoints in some scripts.

Batch logs are typically under `logs/`.

## Cluster / Batch Workflow

`submit.sh` is a Slurm job script that:

- loads cluster modules (`gcc`, `cudatoolkit`),
- activates environment via `starter.sh`,
- runs a sequence of experiment scripts on GPU.

Adjust this file to match your partition/account/runtime needs.

## Notes for Extending the Repo

- Add shared logic in `dynamics/` + `utils/`; keep scripts thin wrappers.
- Prefer adding new experiment defaults as dataclasses in `configs/`.
- Reuse parsing helpers from `utils/parsing.py` for comma-separated CLI lists.
- Keep device and dtype flags wired through to dynamics for reproducibility and portability.

## Reproducibility Checklist

- Set explicit seeds where available in script flags/configs.
- Save both plots and raw `.npz` payloads.
- Record `--device`, `--dtype`, and full CLI args in job logs.
- Use `--no-show` for remote/headless runs.
