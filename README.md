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

- `dynamics/__init__.py` - central export surface for most dynamics functions.
- `dynamics/linear_icl_dynamics.py` - hand-coded analytical attention variants + evaluators:
  - coupled / decoupled / frozen-embedding / softmax-frozen modes
  - synthetic task samplers
  - `run_hand_coded_eval`, `run_hard_power_law_depth_eval`
- `dynamics/pretrain_icl_powerlaw.py` - main pretraining & sweep engine:
  - sample modes (`iid`, `spec`, `spec_rotate`, `gauss_rotate`)
  - power-law problem builders
  - `train_model`, `train_model_softmax`
  - sweep runners (`run_*_sweep`)
  - DMFT helper (`isotropic_dmft`)
- `dynamics/linear_attention_dynamics.py` - isotropic and dimension-free frozen-embedding dynamics:
  - model eval + training for both variants
  - reduced 4-variable theory solvers
- `dynamics/reduced_gamma_dynamics.py` - reduced Gamma matrix SGD model + loss landscape.
- `dynamics/reduced_gamma_fixed_dynamics.py` - fixed-covariance reduced Gamma + OOD rotation loss.
- `dynamics/reduced_gamma_decoupled_dynamics.py` - decoupled-layer reduced Gamma depth dynamics.
- `dynamics/sgd_isotropic_dynamics.py` - isotropic and RMT SGD dynamics + theoretical trajectories.
- `dynamics/toy_model_dynamics.py` - scalar/two-variable toy pretraining dynamics.
- `dynamics/solve_n_final.py` - Newton solver for asymptotic `N`-dependent loss floor.
- `dynamics/ood_covariance.py` - OOD covariance generalization eval routines.
- `dynamics/random_init_covariance.py` - random-initialization covariance eval routines.

### Package Root

- `__init__.py` - top-level exports combining config/data/dynamics/model APIs.

## Script Catalog (`scripts/*.py`)

Each script is a CLI experiment entrypoint. Most write `.png` figures plus `.npz`/`.pt` artifacts.

### Pretraining / Depth / Scaling

- `scripts/run_pretrain_icl_powerlaw.py` - full pretraining run on power-law data (`run_pretrain_icl_powerlaw`).
- `scripts/run_powerlaw_depth_sweep.py` - depth sweep over `L` on rotated power-law data.
- `scripts/run_depth_scaling_nonrotate.py` - depth sweep without random rotations.
- `scripts/run_offline_depth_sweep.py` - offline variant of non-rotated depth sweep.
- `scripts/run_ptr_scaling.py` - sweep over context length `P_tr`.
- `scripts/run_isotropic_depth_vs_alpha.py` - isotropic depth-vs-`P_tr/d` sweep with theory curves.
- `scripts/run_unrestricted_depth_vs_alpha.py` - unrestricted-parameter variant of isotropic depth-vs-alpha sweep.
- `scripts/run_softmax_depth_sweep.py` - softmax-attention depth sweep (`train_model_softmax`).

### Analytical / Hand-Coded Evaluations

- `scripts/run_hand_coded_model_eval.py` - baseline hand-coded model evaluation.
- `scripts/run_hard_power_law_depth.py` - hard power-law covariance depth evaluation.
- `scripts/run_ood_covariance_generalization.py` - OOD covariance exponent sweep evaluation.
- `scripts/run_random_init_covariance.py` - random-initialization covariance evaluation.

### Reduced Gamma and Related Scaling

- `scripts/run_reduced_gamma_dynamics.py` - reduced Gamma SGD + optional landscape.
- `scripts/run_reduced_gamma_depth_sweep.py` - reduced Gamma depth sweep over `L`.
- `scripts/run_reduced_gamma_beta_sweep.py` - reduced Gamma sweep over `beta`.
- `scripts/run_decoupled_layers.py` - decoupled-layer reduced Gamma depth dynamics.
- `scripts/run_fixed_covariance.py` - fixed-covariance reduced Gamma + OOD theta sweep.
- `scripts/run_compute_scaling.py` - width-scaling style sweep for reduced Gamma setting.
- `scripts/run_compute_scaling_width.py` - explicit width (`N`) scaling sweep.
- `scripts/run_compute_scaling_depth.py` - depth scaling + infinite-depth floor estimate.
- `scripts/run_compute_scaling_joint.py` - joint scaling analysis with theoretical `N` solver.
- `scripts/run_loss_vs_w.py` - closed-form loss landscape over scalar weight grid.
- `scripts/run_loss_landscape.py` - reduced-Gamma landscape visualization (`lambda`, `gamma` grid).

### Isotropic / RMT SGD and Toy Dynamics

- `scripts/run_sgd_isotropic.py` - isotropic SGD experiments across `(tau, alpha)` grid.
- `scripts/run_sgd_rmt_isotropic.py` - RMT isotropic SGD sweep runner.
- `scripts/run_pretrain_dynamics.py` - toy pretraining dynamics (`pretrain_dynamics`, two-var variant).
- `scripts/run_beta_sweep_dynamics.py` - beta sweep for two-variable toy dynamics.

### Other Training Entry Point

- `scripts/train_first_cells.py` - trains a `SimpleTransformer` directly on synthetic ICL with configurable depth/width and optional softmax attention.

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
