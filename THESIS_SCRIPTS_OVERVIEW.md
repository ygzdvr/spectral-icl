# Thesis Scripts Overview

Inventory of `scripts/thesis/{utils,theoremA,theoremB,theoremC}/`. Architectures and scaling-laws are excluded — this file covers the frozen utility layer and the three operator-level theorem tiers.

Every script below is **operator-level deterministic**: no training of learned architectures, no learned projectors, no gradient descent over learned parameters. Tolerances are machine-precision (1e-10 to 1e-14 in float64) unless explicitly diagnostic.

All scripts share the standard CLI surface `--device {cuda,cpu,auto} --dtype {float32,float64} --no-show` and write to `outputs/thesis/<phase>/<script_stem>/<run_id>/{figures,pdfs,npz,pt,config.json,metadata.json,summary.txt}`.

---

## `scripts/thesis/utils/` — Frozen Utility & Generator Layer

Every thesis experiment depends on this layer. It enforces the mathematical contracts listed in `THESIS_EXPERIMENTS_STATE_DUMP.md` (real-valued by default, column-sample convention `X ∈ ℝ^{D×P}`, theorem-A sample-space primary, theorem-B exact trajectory lives in `metrics.py`, mass-preserving block spectra, operator-vs-sampled split, `label_norm ∈ {sqrt_D, sqrt_P}`).

### `__init__.py`
Empty contract file. Downstream callers import submodules explicitly (`from scripts.thesis.utils.metrics import gamma_star_trajectory_circulant`) to avoid import-order hazards.

### `run_metadata.py`
Run-directory lifecycle and metadata contract.
- `make_run_id(script_stem)` → `<stem>-<UTC-ISO8601>-<8 hex>`.
- `ThesisRunDir(script_file, phase, …)` — creates `outputs/thesis/<phase>/<stem>/<run_id>/{figures,pdfs,npz,pt}` with paths `.png/.pdf/.npz_path/.pt_path/.metadata_path/.config_path/.summary_path`.
- `RunContext(run_dir, config, seeds, notes)` — context manager. On `__enter__` writes `config.json` + initial `metadata.json` (status=`started`) **before** any computation. Records `.record_step_time(dt)`, `.record_compute_proxy(...)`, `.record_measured_compute(...)`. On `__exit__` updates metadata with status=`completed`/`failed` and wall-clock.
- `git_commit_hash`, `git_is_dirty`, `env_fingerprint` capture reproducibility metadata. Known phases: `controls`, `theoremA`, `theoremB`, `theoremC`, `architectures`, `scaling_laws`, `robustness`.

### `plotting.py`
Unified thesis plotting style.
- `apply_thesis_style()` / `thesis_style()` context manager — sets rcParams + seaborn whitegrid. NOT auto-applied on import.
- Palettes: `rocket` (sequential), `mako` (phase), `vlag` (diverging), `colorblind` (categorical). `sequential_colors(n, palette)`.
- `save_both(fig, run_dir, name, also_pdf=True)` → saves `figures/<name>.png` (300 dpi) + `pdfs/<name>.pdf` (vector, Type-42 fonts).
- `overlay_powerlaw(ax, x, coef, exponent, …)` — draws `y = coef · x^exponent` reference line.
- `overlay_reference(ax, x, y, …)` — arbitrary dashed reference.
- `phase_heatmap(ax, values, x_coords, y_coords, …, log_z, log_x, log_y, cmap="mako")` → mesh + colorbar.
- `mode_trajectories(ax, t, modes, mode_indices, loglog=True, …)`.
- `frontier_plot(ax, compute, loss, predicted_frontier=None)`.
- `legend_compact(ax, ncol, outside)`.

### `fourier_ops.py`
Real-valued Fourier / circulant helpers. Complex arithmetic is isolated to a handful of functions labelled `# complex - isolated`; all real-valued returns assert imaginary leakage `< 1e-10`.
- `freq_grid(P)`, `dft_matrix(P)`, `idft_matrix(P)`, `unitary_dft(x, dim)`, `unitary_idft(X, dim)` (all ortho-normalized, √P division).
- `real_spectral_basis(D, kind="dct2"|"identity")` → `D×D` real orthogonal float64 matrix.
- `circulant_from_symbol(s)` = `F^H diag(s) F` (asserts real-evenness of `s`); `symbol_of_circulant(C)` = inverse.
- `real_even_symbol_from_half(half, P)` — extend length-`P//2+1` half-spectrum to full real-even symbol.
- Symbol constructors: `symbol_power_law(P, ν, eps)` → `(1 + k★)^{-ν}` centered-circular, normalized to mean 1; `symbol_multiband(P, bands)`; `symbol_flat(P, v)`; `symbol_interpolate(s0, s1, α)` = `(1-α)s0 + α s1`; `frequency_permutation(s, seed)` — uniform permutation of positive frequencies preserving real-evenness (DC/Nyquist fixed).
- `off_diagonal_fourier_energy(M)` ∈ [0,1] — circulant diagnostic `||offdiag(FMF^H)||_F² / ||FMF^H||_F²`.

### `partitions.py`
Block partitions and mass-preserving heterogeneity.
- `BlockPartition(D, blocks)` — frozen dataclass; `.n_blocks`, `.sizes`, `.block_of(k)`, `.indicator_matrix()`, `.block_projector(b)`. Validates disjointness + coverage.
- `equal_blocks(D, m)`, `dyadic_ladder(D, J)` — requires `D = 2^J`; returns `J+1` levels (1, 2, 4, …, 2^J blocks). `custom_ladder(levels)` validates refinement chain.
- `mass_preserving_block_spectrum(partition, block_means, block_kappas, xi_shape="linear")` → `λ ∈ ℝ^D` with the mass-preserving formula
  ```
  λ_{b,j} = λ̄_b · κ_b^{ξ_j} / ((1/m_b) Σ_u κ_b^{ξ_u})
  ```
  which enforces `(1/m_b) Σ_j λ_{b,j} = λ̄_b` exactly, for all κ_b. For `m_b ≥ 2` under linear ξ the within-block condition number is exactly κ_b. Singleton blocks get `ξ=0`.
- `mass_preserving_block_task(...)` — same formula for task variance ω.

### `commutants.py`
Block-commutant class `C(B) = {Q : Q = Σ_b q_b P_b, q ∈ ℝ^{n_blocks}}`, where `P_b = Σ_{k∈b} e_k e_k^T`.
- `extract_block_scalars(Q, partition)` → `q_b = (1/m_b) Σ_{k∈b} Q[k,k]`.
- `reconstruct_from_block_scalars(q, partition)` → `Σ_b q_b P_b`.
- `commutant_projection(Q, partition)` = extract + reconstruct.
- `commutant_violation(Q, partition, normalize=True)` → `||Q − π_C(Q)||_F²` (optionally divided by `||Q||_F²`).
- `refines(fine, coarse)` — predicate: every fine block is contained in exactly one coarse block.

### `metrics.py`
Theorem-A/B/C metrics.
- `reduced_model_error(f_full, f_red)` → relative L2.
- `ab_perturbation_bound(A_θ, A_GD, B_θ, B_GD, T_θ, T_GD, L, y)` → dict with `delta_A_op`, `delta_B_op`, `S_theta_y_norm`, `B_side_bound`, `A_side_bound` (with telescoping coefficient), `total_bound`, `empirical_error`. Full additive (A,B) decomposition for Theorem A §7.2.
- `gamma_star_trajectory_circulant(s_tr, ω, L, η, T, γ0=None)` → `(T+1, P)` float64 tensor. **Single source of truth for Theorem B exact trajectory**: `γ_k(t+1) = γ_k(t) + η · ω_k · s_tr_k² · (1 − s_tr_k · γ_k(t) / L)^{2L−1}`. G1 generator deliberately does NOT return trajectory; this is always called separately.
- `mode_trajectory_error(γ̂, γ★)`, `transfer_function_error(T̂, T★)`.
- `grouped_trajectory_error(q̂, q★)`.
- `oracle_commutant_loss(λ, ω, partition, L, q_init, optimizer="lbfgs", max_iter=500)` — minimizes `L(q, L) = Σ_b Σ_{j∈b} ω_{b,j} λ_{b,j} (1 − q_b λ_{b,j} / L)^{2L}` over block-scalar commutant. Returns `q_star`, `loss_star`, `per_block_loss`, `converged`. L-BFGS + strong-Wolfe. Used by C3/C4/C5/C6/C7.
- `contraction_depth_overlay(κ_b, L_grid)` → `(ρ★)^{2L}` where `ρ★ = (κ−1)/(κ+1)`. Used by C7 as theory reference, **not** power-law fit.
- `ood_slope(θ, loss, fit_window)` → power-law slope on OOD.
- `holdout_prediction_error(fit_result, x_val, y_val)`, `frontier_regret(configs, loss, compute, predicted_optimum)`.

### `cost_models.py`
Canonical compute proxy — **hardcoded** adaptive-first then spectral hybrid.
- `phi_adaptive(P)` = `P²`.
- `phi_spectral_fft(P, r)` = `P log P + P r`; `phi_spectral_trunc_linear(P, r)` = `P r` (alternate).
- `compute_proxy(t, P, L_A, L_S, r, c_A=1, c_S=1, phi_S=phi_spectral_fft)` → `t · (c_A · P² · L_A + c_S · φ_S(P,r) · L_S)`.
- `WallClockMeter` context manager; `.step()`, `.total_seconds`, `.per_step_seconds`.
- `calibrate(runs, phi_S)` — linear least-squares fit of `(c_A, c_S)` to measured wall-clock across runs.

### `fit_powerlaws.py`
Single entry point for log-log fits. **Fit windows are mandatory** (never auto-selected) per §9.1 binding.
- `fit_loglog(x, y, fit_window=(lo, hi), heteroskedastic_weights=None)` → `{slope, intercept, r2, residuals, fit_x, fit_y}`.
- `bootstrap_exponent(x, y, fit_window, seed_axis, n_bootstrap=1000, α=0.05)` → quantile envelopes. Deterministic seed.
- `holdout_evaluate(x_fit, y_fit, x_val, y_val, fit_window)` → `{median_rel_err, max_rel_err, slope_fit, slope_val}`.

### `data_generators.py`
The four generators — operator-level primary, sampled-context secondary.
- **GA (Theorem-A masked context).** `GAConfig(D, P, K, B=1, Sigma_kind, ..., label_norm="sqrt_D")`, `ga_generate(cfg)` → primary `A_S_GD = −(1/P) X_tr^T Γ X_tr`, `B_S_GD = +(1/P) X_q^T Γ X_tr`, `T_GD = I_P + A_S_GD/L`; perturbed variants `(A_S_θ, B_S_θ, T_θ)`; sampled data `(X_train, X_query, y_train, y_query, β)`; mask matrix; covariances `(Σ, Ω, Γ)`. Feature-space helpers are diagnostic-only.
- **G1 (Theorem-B stationary circulant).** `G1Config(P, ..., label_norm="sqrt_P")`, `g1_generate(cfg)` → spectra `(s_tr, s_te, ω)`, circulant covariances via symbol; optional sampled data. Returns no trajectories — caller invokes `metrics.gamma_star_trajectory_circulant`. `query_mode ∈ {"full_window", "single_query"}`.
- **G2 (Theorem-C band-RRS).** `G2Config(D, partition_kind, block_means_lam, block_kappas_lam, ...)`. `g2_generate_operator(cfg)` → partition, spectra `(Λ, Ω)`, spectral basis F, per-block stats. `g2_generate_sampled(cfg, n_contexts, P, K)` — physical-basis data `Σ_c = F^T R_c diag(Λ) R_c^T F` with per-context block-Haar rotation `R_c` in spectral basis. `g2_to_spectral_basis(X, F)` returns `R_c @ z` (NOT the canonical diagonal z) — diagnostic only.
- **G3 (Refinement ladder).** `G3Config(D, ladder_kind="dyadic", reference_partition_index, ...)`. `g3_generate(λ, ω, ladder, F=None)` — direct API; asserts `Λ`, `Ω`, `F` bitwise-identical across levels. `g3_generate_from_config(cfg)` — constructive wrapper.
- Helpers: `cols_to_rows` / `rows_to_cols` (only places deviating from column-sample convention, explicit names), `_build_covariance`, `_build_gamma`, `_build_symbol`, `_build_mask`, `_sample_gaussian_columns`, `_sample_block_haar`.

### `_test_scaffold.py`
End-to-end smoke test: RunContext lifecycle, metadata contract, figure saving, per-step wall-clock, failure path. Exit 0 = all assertions passed.

### `_self_tests/run_all.py`
Full v4 §12 test harness. `run_exact(phase, name, fn)` (fail-hard) and `run_mc(phase, name, fn)` (statistical, reported). Exit code 0 iff all exact tests pass. Current state (per state dump): 51/51 exact + 6/6 MC green.

---

## `scripts/thesis/theoremA/` — Exact Structured Reduced-Operator Bridge

Theorem A formalizes when the standard reduced-Γ dynamics is exactly recovered from a structured attention model. Sample-space reduced operators `(A_S, B_S)` are primary; feature-space is diagnostic. All scripts are operator-level deterministic forward-pass tests with NO training. Defaults to float64.

### `run_theoremA_exact_equivalence.py` — A1
Three structurally distinct forward routes agree to machine precision in the GD-compatible setting.
- **Grid**: `D ∈ {8,16,32,64} × P ∈ {8,16,32,64} × K ∈ {4,8,16} × L ∈ {1,2,4,8}` = 192 cells × 3 routes = 576 forwards. `B=4` per cell. Generator: `ga_generate` with `mask_kind="gd_compatible"`.
- **Routes**: R1 iterative reduced `(A_S, B_S)`; R2 closed-form `f_red = (1/L) B_S Σ_ℓ T^ℓ y`; R3 feature-space reduced-Γ.
- **Gate**: max of all pairwise errors `err_R1_R2, err_R2_R3, err_R1_R3 ≤ 1e-10`.
- **Outputs**: pairwise-error heatmap, error histogram with 1e-10 line, error-vs-L diagnostic, `a1_sweep_table.npz`.

### `run_theoremA_exact_equivalence_full_model.py` — A1b
Adds route R0 — the TRUE full-hidden-state forward that builds the `(P+K)×(P+K)` bilinear score `S[μ,ν] = x_μ^T Γ x_ν / P` directly and applies the GD-compatible signed mask, without consuming GA-generator reduced operators. Closes the gap A1 left (R1 was already a reduced object).
- **Grid**: same as A1.
- **Gate**: `max(err_R0_R2, err_R2_R3) ≤ 1e-10`.

### `run_theoremA_mask_perturbation.py` — A2
Empirical full-model error vs. the full additive (A,B)-operator perturbation bound away from GD-compatibility.
- **Configs**: `(D,P,K,L) ∈ {(32,32,8,4), (64,32,8,4)}`, θ-grid `{0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2}`, 4 seeds, two modes (`A_only` canonical, `B_only` auxiliary) → 144 trials.
- **Mechanism**: in `A_only`, train-train mask block perturbed → `ΔA ∝ θ, ΔB = 0`. `B_only` perturbs test-train block (out-of-train-support, diagnostic only).
- **Bound** (via `metrics.ab_perturbation_bound`): total = B-side + A-side with telescoping propagator difference. **Both contributions reported separately** — never folded.
- **Gates**: (1) θ=0 sanity `||F_0 − F_GD|| ≤ 1e-10`; (2) `empirical ≤ total_bound + 1e-12` for all (config, seed, θ).
- **Outputs**: `empirical_vs_bound`, `decomposition` (A-side, B-side, total, empirical), `theta_zero_sanity`, `bound_slack` ratio heatmap.

### `run_theoremA_semiseparable.py` — A3 + A4
Two packages in one script.
- **A3 (primary)**: Explicit rank-1 semiseparable realization `M_signed = (1_test − 1_train) 1_train^T` supports a D-dimensional state recursion via rank-D outer product. Compared against reduced `(A_S, B_S)` closed form (R_AB) and full-hidden-state (R_full). Gate: `err_R_SSD_vs_R_AB ≤ 1e-10` across 192 cells.
- **A4 (negative controls)**: Fixed geometry `(D,P,K,L) = (32,32,8,4)`, 4 seeds. Two mixers deliberately outside the theorem class: **NC1** circular convolution with Gaussian-bump kernel, **NC2** non-GD mask variant. Gate: `min_seed(||F_NC − F_red|| / ||F_red||) ≥ 0.1` — deviation proves theorem hypotheses are necessary.

### `run_theoremA_general_train_supported.py` — A1-general
Extends A1 / A1b exactness to EVERY train-supported mixer (Theorem 1 general case) and validates Proposition 3 (necessity of GD-compatibility for reduced-Γ collapse).
- **Mask families**: `gd_compatible`, `lower_triangular`, `random_dense`, `near_gd@ε ∈ {0.01, 0.1, 0.5}` — 6 families.
- **Γ kinds**: `identity`, `random_symmetric`, `random_nonsymmetric` (the last tests correct W_q / W_k ordering).
- **Σ kinds**: `isotropic`, `structured` (Σ = diag(k^{-1}), Ω = diag(k^{-0.5})).
- **Geometry**: 96 cells per family. Total 6 × 3 × 2 × 96 = 1728 cells.
- **Routes**: R0, R1, R2, R3.
- **Gates**: (1) Theorem 1 — `max(err_R0_R1, err_R0_R2, err_R1_R2) ≤ 1e-10` every cell; (2) Corollary 1 — `max(err_R0_R3, err_R2_R3) ≤ 1e-10` on GD-compatible cells; (3) Proposition 3 — non-GD kinds must show `max_cells(err_R2_R3) ≥ 1e-3` (necessity).

### `run_theoremA_structural_closure.py` — A-structural
Matrix-identity tier: Proposition 2 (rank-1 factorization of M^GD + semiseparable reconstruction), Proposition 5 (Toeplitz / circulant / semiseparable closure under Hadamard with `K_Γ(X) = X^T Γ X`), Remark 2 (untied-layer non-autonomous reduced model). Five independent parts:
1. Prop 2 rank-1: `||M^GD − st^T|| < 1e-14` and generator reconstruction `< 1e-14`.
2. Prop 5 Toeplitz: `A_S = (1/P)(S_TT ⊙ K_Γ)` with both factors exactly Toeplitz → off-Toeplitz energy `< 1e-12`.
3. Prop 5 circulant + Fourier consistency: off-DFT energy `< 1e-12`; eigenvalue identity `eigvals(A_S) = (1/P²)(eigvals(S_TT) ⊛ eigvals(K))` (circular convolution) `< 1e-12`.
4. Prop 5 semiseparable rank: product has rank-(r₁r₂) on strict-lower blocks (`σ_{r₁r₂+1} < 1e-10`); per-entry Kronecker factorization `< 1e-12`.
5. Remark 2 untied: L distinct Γ_ℓ matrices; non-autonomous reduced recursion `r^{(ℓ+1)} = (I + L^{-1} A_S(X, Γ_ℓ)) r^{(ℓ)}` matches full-model `< 1e-10`.

### `__init__.py`
Empty; scripts are independent.

---

## `scripts/thesis/theoremB/` — Stationary / Circulant Closure

Theorem B generalizes Bordelon's FS regime to stationary circulant covariances. X-axis is **layer index ℓ** for B1 (finite-P discrete recursion) and **optimization time t** for B2 (continuous-time gradient flow). All operator-level; B2 recursion runs on CPU float64 for precision. B0 (structure-closure) is not present; B5 (LDS) is deferred.

### `run_theoremB_circulant_modes.py` — B1
Exact finite-P layer-index recursion vs. closed-form transfer function on circulant operators.
- **Grid**: `P ∈ {16,32,64} × L ∈ {1,2,4,8} × symbol ∈ {flat, power_law, multiband} × match ∈ {matched, mismatched}` = 72 trials. Generator: G1, `exact_mode=True`, operator-level only.
- **Recursion**: `y = e_0`; `r^{ℓ+1} = (I − G/L) r^ℓ` for ℓ = 0..L−1; `f = (1/L) G★ Σ r^ℓ`. Transform to unitary DFT.
- **Theory overlay**: residual `r̂_{th}[ℓ, k] = (1 − λ_k/L)^ℓ / √P`; matched transfer `h_{th}[k] = 1 − (1 − λ_k/L)^L`; general `h_{th}[k] = λ★_k φ_L(λ_k)` with numerically-stable `log1p/expm1` branch.
- **Gates** (all 72 trials): `residual_mode_rel_err ≤ 1e-10`, `transfer_rel_err ≤ 1e-10`, `train_offdiag_fourier_energy ≤ 1e-10`, `query_offdiag_fourier_energy ≤ 1e-10`.
- **Optional `--bridge-to-b2`**: runs stationary γ_k(t) gradient flow via `metrics.gamma_star_trajectory_circulant` and overlays on Fourier trajectories.

### `run_theoremB_depth_stationary.py` — B2
Matched-training gradient-flow dynamics. Primary claim: at long T, terminal loss is L-independent in the matched stationary regime.
- **Grid**: main `P ∈ {32,64} × symbol ∈ {power_law, multiband} × L ∈ {1,2,4,8,16}` = 20; long-context `P ∈ {128,256} × power_law × 5L` = 10. Total 30 trials, T = 100,000, η = 5e-5.
- **Discrete** (empirical): `γ_k(t+1) = γ_k(t) + η ω_k s_k² (1 − L^{-1} s_k γ_k(t))^{2L−1}`.
- **Continuous ODE** (theory, subsampled at 200 log-spaced times): `L=1` ⇒ `δ_k(t) = exp(−α_k t)`, `α_k = η ω_k s_k³`; `L>1` ⇒ `δ_k(t) = (1 + 2(L−1)/L · α_k t)^{-1/(2(L−1))}`; then `γ_k = (L/s_k)(1 − δ_k)`.
- **Loss exact**: `(1/P) Σ_k ω_k s_k δ_k(t)^{2L}`.
- **Gates**: (1) monotonicity `max(Δ loss) ≤ 1e-9`; (2) decay `loss_final < 0.2 · loss_init`; (3) ODE agreement `max_rel_err < 1e-5`; (4) loss-theory `< 5e-2`; (5) forward invariance `γ_k(t) ≤ L/s_k` (slack 1e-10); (6) circulant preservation `< 1e-10`; (7) shift invariance `E_L(Π^m Q Π^{-m}) = E_L(Q)` at float eps.
- **Figures**: `loss_vs_time`, `finite_time_loss_vs_depth`, `finite_time_P_dependence`, `terminal_residual_factor_spectrum`, `modewise_ode_trajectories`, `modewise_ode_normalized`, `loss_vs_time_theory_overlay`, `operator_target_error`, `equal_tolerance_collapse`, `circulant_preservation`.

### `run_theoremB_symbol_shift.py` — B3
OOD brittleness under spectral symbol mismatch (Corollary 5). Two families: **F1 structural** `s_te(α) = (1−α) s_tr + α s_flat` (attenuation regime); **F2 permutation** `s_te(α, seed) = (1−α) s_tr + α · permute_freq(s_tr, seed)` (amplification regime).
- **Grid**: P = 64, `L ∈ {1,2,4,8,16}`, 12 α values per family, 8 permutation seeds for F2. Generic covariance rotation OOD is **excluded** (that's Theorem C territory).
- **Theory (Corollary 5 at converged optimum Q★)**: `E_OOD(α, L) = (1/P) Σ_k ω_k s_te_k(α) |1 − s_te_k(α)/s_tr_k|^{2L}`. If `|1 − ratio| < 1` everywhere: attenuation, E_OOD decreases with L. If some modes cross > 1: amplification, grows with L.
- **Empirical (finite-time γ(T))**: `(1/P) Σ_k ω_k s_te_k(α) (1 − L^{-1} s_tr_k γ_k(T))^{2L}` with γ from matched training.
- **Gates**: (1) matched-baseline recovery at α=0 `≤ 1e-10` relative; (2) full-shift brittleness `max_L(f1_loss[L][α=1] / baseline[L]) ≥ 1.25` (any L suffices; typical 1.30–1.51×).

### `run_theoremB_rank_scaling.py` — B4
Spectral rank bottleneck (Corollary 6) and joint (r, L) collapse. Mode-decoupling shortcut: train ONE unmasked trajectory per L, then rank-mask post-hoc.
- **Grid**: P = 256, `r ∈ {1,2,4,8,16,32,64,128}`, `L ∈ {1,2,4,8}`, T = 100,000. 4 training trajectories + 32 evaluations.
- **Analytical floor** (L-independent by construction): `floor(r) = (1/P) Σ_{k ≥ r} ω_k s_k`.
- **Power-law fit** on `r ∈ [4, 64]`: empirical slope at L=1 vs. analytical-floor slope; continuum asymptote `1 − (ν + νβ)`.
- **Gates**: (1) floor power-law fit `|slope_emp − slope_ana| / |slope_ana| ≤ 0.15`; (2) depth collapse at r_max `loss(r_max, L_max) / loss(r_max, L_min) ≤ 5.0`.
- **Figures**: `rank_floor`, `loss_vs_depth_at_fixed_rank` (r ∈ {4,16,64}), `joint_rL_grid` heatmap (horizontal iso-contours = L-independence), `depth_independence_ratio`.

### `run_theoremB_supplementary_figures.py`
Pure post-processing — loads canonical B3/B4 NPZ and evaluates Corollary 5 / Corollary 6 closed forms directly. Fixes original figures that incorrectly compared converged-optimum formulas against finite-time γ(T). CLI: `--b3-run-dir`, `--b4-run-dir`. Qualitative acceptance (no hard gate): F1 E_OOD decreases with L; F2 E_OOD grows with L at α ≥ 0.5; C6 empirical clusters near theoretical floor with worst L-collapse ratio ≲ 1.3.

### `__init__.py`
Docstring pointer to EXPERIMENT_PLAN_FINAL.MD §6. No re-exports.

---

## `scripts/thesis/theoremC/` — Band-RRS Commutant Closure

Theorem C generalizes Bordelon's RRS to band-partitioned blocks with within-block heterogeneity κ. The commutant class is the **tightened** `C(B) = {Σ_b q_b P_b}` (not generic block-diagonal). "Oracle hybrid" = **direct optimization over a refined commutant** (NOT a learned projector — that's architecture tier §9). Seven core experiments + four patches + cleanup. All operator-level via G2/G3; uses `oracle_commutant_loss` (L-BFGS) heavily.

### `run_theoremC_commutant_closure.py` — C1 + C2
Two experiments in one script.
- **C1 band-RRS commutant closure**: R-averaged population-loss recursion `Γ(t+1) = Γ(t) + η · E_R[Ω_c · Σ_c² · (I − L^{-1} Σ_c Γ(t))^{2L−1}]` preserves `Γ(t) ∈ C(B)` exactly. Compared against naive per-F-mode recursion (no R-averaging) as negative control (commutant violation grows).
- **C2 grouped-scalar ODE**: `δq_b = η (1/m_b) Σ_{j∈b} ω_j λ_j² (1 − L^{-1} λ_j q_b)^{2L−1}`. Matrix path and direct ODE on q must match to float eps.
- **Config**: single fixed operator — D=64, m=8, κ=2.0, moderate mass-downward per-block spectra (λ 1.0→0.3, ω 1.0→0.65), `L ∈ {1,2,4,8}`, T=5000, η=5e-3.
- **Gates**: (1) R-averaged violation `≤ 1e-12`; (2) naive exceeds R-averaged by `≥ 1e8×`; (3) `max|q_mat − q_ode| ≤ 1e-12`; (4) MC Haar consistency — at N=50,000 block-Haar rotations, relative Frobenius error `≤ 2e-2` (4× margin over `1/√N ≈ 5e-3`).

### `run_theoremC_c1c2_supplement.py`
Five gap-fillers on top of C1/C2.
1. **Lemma 3.5 invariance**: 20 random `U ∈ G_B`, 10K MC block-Haar samples; `E_L(Q) = E_L(UQU^T)` for Q inside and outside commutant (paired-MC 5σ).
2. **Grouped-loss formula**: at 50 log-spaced checkpoints, matrix loss `tr(...)` vs. block-scalar formula agree at 1e-12.
3. **Induced metric**: `||Q(t)||_F² = Σ_b m_b q_b(t)²` at each checkpoint.
4. **Corollary 3.9 endpoint recovery**: rerun at m ∈ {1 (singletons), 64 (single block), 8 (main)}; compare against per-mode and scalar-isotropic ODE.
5. **Unequal-block partition robustness**: partition `(4,4,8,16,32)` summing to D=64; C1+C2 gates still hold.

### `run_theoremC_L1_closed_form.py` — C3
At L=1, block-commutant loss has closed form `q_b★ = b_b / c_b` with `a_b = Σ_j ω_j λ_j`, `b_b = Σ_j ω_j λ_j²`, `c_b = Σ_j ω_j λ_j³`. Obstruction `L★ = Σ_b [a_b − b_b²/c_b]`.
- **Grid**: `m ∈ {1,2,4,8,16,32} × κ ∈ {1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0}` = 48 trials. Mass-preserving linear ξ, `λ̄ = ω̄ = 1` per block, `κ_ω = κ_λ`.
- **Gates**: (1) closed-form vs. numerical (L-BFGS) relative error `≤ 1e-8` on L★>0 trials; (2) κ=1 degeneracy `|L★| < 1e-10` at all m; (3) monotonicity diagnostic — `L★(κ)` non-decreasing for m>1 (soft).
- **Figures**: `c3_closed_form_vs_numeric`, `c3_obstruction_vs_kappa`, `c3_obstruction_heatmap`, `c3_loss_landscape` (per-block parabola across κ).

### `run_theoremC_phase_diagram.py` — C4
**Headline thesis figure.** 2D `(m, κ)` phase diagram at multiple depths L of (a) coarse-class optimum, (b) dyadic-finer oracle, (c) refinement gain `gap = L_coarse − L_fine`.
- **Grid**: `m ∈ {1,2,4,8,16,32} × κ ∈ 7 values × L ∈ 5 depths` = 210 cells × (2 or 3 L-BFGS each) ≈ 420+ optimizations. Finishes 1–3 min CPU.
- **Gates**: (1) refinement nonnegativity `gap ≥ −1e-7` for m≥2 (weak monotonicity); (2) κ=1 degeneracy `L_coarse ≤ 1e-9`; (3) full-oracle ≡ 0 — singleton optimum `≤ 1e-7` everywhere (per-mode matched regime).
- **Figures**: `c4_phase_diagram_main` (3-panel), `c4_kappa_slices`, `c4_depth_interaction` at κ ∈ {1.2, 2.0, 5.0}, `c4_full_oracle_sanity`.

### `run_theoremC_refinement_monotonicity.py` — C5
Monotonicity over the full dyadic ladder D=64 → singletons (7 levels, 6 refinement steps).
- **Grid**: 7 κ values × L ∈ {1, 4} = 14 ladder sweeps. G3 with `ladder_kind="dyadic"`, `reference_partition_index=0`.
- **Gates**: (1) `L★(j+1) ≤ L★(j) + 1e-8` every consecutive pair; (2) κ=1 all-zero; (3) finest level `L★(j=6) ≤ 1e-8`; (4) diagnostic — for κ>1, strict drops on all 6 steps.
- **Figures**: `c5_refinement_ladder` (per-κ line), `c5_level_drops` (per-step bar), `c5_ladder_heatmap`, `c5_depth_comparison` (L=1 vs. L=4).

### `run_theoremC_oracle_hybrid.py` — C6
Captures three strictly operator-level objects: (a) coarse-class optimum, (b) oracle hybrid (refined-class at one dyadic step), (c) oracle ceiling (singleton optimum ≡ 0 in matched regime). Captured fraction `F = (L_coarse − L_hybrid) / L_coarse`.
- **Grid**: `m ∈ {2,4,8,16,32}` (excludes m=1 — refining singleton is a no-op), 7 κ, 5 L = 175 cells × 3 L-BFGS = 525 optimizations.
- **Gates**: (1) ordering `L_coarse ≥ L_hybrid ≥ L_unc` up to 1e-7; (2) oracle ceiling `L_unc ≤ 1e-7`; (3) m=2 boundary — refinement is singleton, `F ≡ 1` within 1e-3.

### `run_theoremC_depth_scaling.py` — C7
Finite-depth scaling with **contraction overlay** — emphatically NOT a generic `L^{-β}` power-law fit. Theorem-correct reference is `(ρ★)^{2(L−1)}` anchored at L=1, with `ρ★ = (κ−1)/(κ+1)`.
- **Grid**: 6 m × 6 κ × 7 depths `L ∈ {1,2,4,8,16,32,64}` = 252 L-BFGS. `max_iter=3000` (bumped from 500 for ill-conditioned deep-L cells; convergence not strictly required).
- **Gates**: (1) singleton `L★(m=1, ·, ·) ≤ 1e-8`; (2) κ=1 `L★ ≤ 1e-8`; (3) monotone non-increase in L within 1e-8 per (m,κ); convergence is a diagnostic, not a gate.
- **Figures**: `c7_loss_vs_depth`, `c7_contraction_overlay` (empirical lines typically *above* `(ρ★)^{2(L−1)}` — single-root polynomial is slower than Chebyshev-optimal, physically correct), `c7_interpolation` (smooth κ transition), `c7_m_sweep`, `c7_empirical_slope_vs_theory` (scatter vs. `2 log ρ★`, diagnostic only).

### Patches

#### `run_theoremC_c3_patch.py`
Three fixes to C3.
1. Plotting bug: m=1 (L★≡0) no longer rendered on log scale; textual note added.
2. Formal Corollary 3.12 gates: `m=1 ⇒ L★ < 1e-10 ∀κ`, and `m>1, κ>1 ⇒ L★ > 1e-15` strict.
3. Corollary 3.13 Chebyshev bound overlay: `cheby_b = (Σ_{i∈B_b} ω_i λ_i) · ((κ_b−1)/(κ_b+1))²` must satisfy `cheby ≥ L_cf` everywhere.

#### `run_theoremC_c4_strict_gain_patch.py`
Acceptance-record-only patch, no new figures. Loads canonical C4 NPZ (or `--recompute`). Gates the STRICT direction of Corollary 3.16: for m≥2 and κ>1, `gap > 1e-15` (strict positive); and for m=1 or κ=1, `|gap| < 1e-9` (degenerate zero region).

#### `run_theoremC_c5_strict_drops_patch.py`
Mirrors the C4 strict-gain patch for the ladder: for κ>1 and all L, every dyadic step must produce a strict drop (count = 6/6 for D=64); κ=1 has zero strict drops.

#### `run_theoremC_c7_chebyshev_bound_patch.py`
Replaces C7's heuristic anchored reference with the theorem-correct Corollary 3.13 bound `Σ_b (Σ_{i∈B_b} ω_i λ_i) · ρ_b^{2L}`. Adds formal gate `L★_observed ≤ bound + 1e-15` (raw), with resolved diagnostic on cells above `1e-9` optimizer floor. Regenerates G2 operators on the fly to extract per-block κ_b.

### `run_theoremC_cleanup.py`
Three non-blocking housekeeping items in one place.
1. Heatmap visual encoding — regenerate C3/C4/C5 heatmaps with explicit light-gray hatching for exact zeros (m=1 rows, κ=1 columns) instead of log-scale floor rendering, with corner annotation "≡0 (Cor. 3.12)".
2. Corollary 3.11 edge case — inactive block (ω_block = 0): verify `L_b(q_b) = 0` for all q_b and that closed-form treats it as zero (checks at q ∈ {−10, 0, 1, 100}, all `≤ 1e-12`).
3. κ-monotonicity artifact — confirms violations observed at m=2 do NOT occur at m≥4; emits LaTeX-ready footnote attributing to mass-preserving linear-ξ parameterization at small-m, extreme-κ.

### `__init__.py`
Empty; each script is independent.

---

## Appendix — Utility Dependencies per Script

| Script | Primary utils dependencies |
|---|---|
| A1 / A1b | `data_generators.ga_generate`, `metrics.reduced_model_error`, `plotting`, `run_metadata` |
| A2 | `ga_generate` (modes `gd_compatible`+`perturbed`), `metrics.ab_perturbation_bound` |
| A3+A4 | `ga_generate`, `metrics.reduced_model_error` |
| A1-general | `ga_generate` (all mask kinds + γ kinds + Σ kinds), `metrics.reduced_model_error` |
| A-structural | `ga_generate` (untied Γ), `metrics.reduced_model_error`, torch linear algebra |
| B1 | `G1Config, g1_generate`, `fourier_ops.circulant_from_symbol/off_diagonal_fourier_energy/unitary_dft`, `metrics.gamma_star_trajectory_circulant` (bridge only) |
| B2 | `g1_generate` (matched), `metrics.gamma_star_trajectory_circulant` (core), `fourier_ops.dft_matrix` |
| B3 | `g1_generate`, `metrics.gamma_star_trajectory_circulant`, `fourier_ops.symbol_interpolate/frequency_permutation/symbol_{power_law,multiband,flat}` |
| B4 | `g1_generate`, `metrics.gamma_star_trajectory_circulant`, `fit_powerlaws.fit_loglog` |
| B supplementary | `plotting` only (post-processing from B3/B4 NPZ) |
| C1+C2 | `G2Config, g2_generate_operator`, `commutants.{commutant_projection, violation, extract_block_scalars, reconstruct_from_block_scalars}` |
| C1C2 supplement | `G2Config, g2_generate_operator`, `commutants`, `metrics.oracle_commutant_loss`, `partitions.{BlockPartition, equal_blocks, mass_preserving_block_spectrum}` |
| C3 | `G2Config`, `metrics.oracle_commutant_loss`, `commutants.reconstruct_from_block_scalars` |
| C4 | `G2Config`, `partitions.equal_blocks`, `metrics.oracle_commutant_loss`, `plotting.phase_heatmap` |
| C5 | `G3Config, g3_generate_from_config`, `metrics.oracle_commutant_loss` |
| C6 | `G2Config`, `partitions.equal_blocks`, `metrics.oracle_commutant_loss` |
| C7 | `G2Config`, `metrics.oracle_commutant_loss` |
| C patches | canonical NPZ loaders + whichever utils the patched script uses |
| C cleanup | canonical NPZ loaders + `plotting.PALETTE_PHASE/sequential_colors` |

All experiments write `config.json`, `metadata.json`, `summary.txt`, and raw arrays under `outputs/thesis/<phase>/<script_stem>/<run_id>/`. Test gate: `python -u scripts/thesis/utils/_self_tests/run_all.py` (exit code 0 = all exact tests green). State dump reports 51/51 exact + 6/6 MC passing.
