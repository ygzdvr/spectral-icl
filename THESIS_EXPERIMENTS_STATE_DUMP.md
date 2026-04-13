# THESIS_EXPERIMENTS_STATE_DUMP.md

> **Authoritative project-memory export for the spectral-scaling thesis
> experimental program.** Self-contained: a future Claude (or human)
> should be able to pick up the project from this document alone after
> conversation compaction, without needing to reconstruct anything from
> earlier chat history. Treat this file as the single source of truth
> for what has been done, what was decided, what was deprecated, and
> what comes next.
>
> **Authoritative companion document**: `EXPERIMENT_PLAN_FINAL.MD`
> (the locked plan). This state dump records what was actually built
> against that plan, with all the approved interpretations and
> corrections.

---

## 1. Global thesis experiment status

The thesis experimental program follows a five-layer hierarchy, with
implementation order **B → C → A → architecture-aligned → scaling-law
→ robustness** (plan §13). Status as of this dump:

| Phase | Plan ref | Status |
|---|---|---|
| Step 1a — repo scaffolding, `run_metadata.py`, `plotting.py` | §4.1, §4.2 | **Complete and frozen.** |
| Step 1b — utility / generator layer (`fourier_ops`, `partitions`, `commutants`, `metrics`, `cost_models`, `fit_powerlaws`, `data_generators`, `_self_tests`) | §4.3–§4.8 | **Complete and frozen.** Self-tests green (51/51 exact + 6/6 MC). |
| Step 2 — Bordelon control suite freeze | §5 | **Complete, immutable.** Six controls archived under `outputs/thesis/controls/` by archive-by-copy with SHA-256 manifest and `0444` mode. |
| Theorem-B exact program (B1–B4) | §6.2–§6.5 | **Complete.** B0 (structure-closure diagnostic, §6.1) **pending — not yet implemented**. B5 (LDS, §6.6) deferred (optional). |
| Theorem-C exact program (C1, C2, C1-MC-Haar follow-up, C3–C7) | §7.1–§7.7 | **Complete.** Central experimental block of the thesis. |
| Theorem-A exact program (A1, A1b, A2, A3, A4) | §8.1–§8.4 | **Complete.** A1 was supplemented by A1b after a clarification: A1's R1 was iterative reduced, not full hidden-state. A1b adds the true full-model R0 bridge. A3 + A4 combined into one script with clear section separation. |
| Architecture-aligned validation tier | §9 | **Pending — next phase.** Three suites required: spectral-only, structured-mask/SSD, canonical adaptive-first hybrid. |
| Conditional scaling-law tier | §10 | **Pending.** Separate exponent estimation → additive separability → compute-frontier validation. |
| Robustness / natural-task extensions | §11 | **Pending (lowest priority).** Optional bonus tier. |

**Plan freeze rules** (per user constraints, not in plan but binding):

- The Step 1b utility / generator layer is **frozen** — no edits
  unless a real downstream bug is exposed.
- The Step 2 Bordelon control suite is **immutable archive-by-copy**;
  its files are `0444` and must not be re-run or overwritten.
- Each completed theorem-tier experiment script's **canonical run
  directory** (a single timestamped run dir per script under
  `outputs/thesis/<theorem>/<script_stem>/`) is kept; stale dev runs
  were systematically cleaned up after every approval.

---

## 2. Canonical experimental architecture of the thesis

Five validation layers (plan §1):

1. **Layer 1 — frozen control replication** (§5). Reproduced
   Bordelon scripts archived as the calibration baseline.
2. **Layer 2 — exact theorem-validation tier**. Theorem-B (§6),
   theorem-C (§7), theorem-A (§8). All deterministic / operator-level.
   Acceptance is algebraic (machine precision in `float64`), not
   statistical.
3. **Layer 3 — architecture-aligned validation** (§9). Three suites:
   spectral-only, structured-mask / SSD, canonical adaptive-first
   spectral hybrid. Genuine trainable architectures aligned to theorem
   assumptions; acceptance is qualitative theorem phenomenon
   reproduction.
4. **Layer 4 — conditional empirical scaling-law tier** (§10).
   Separate exponent estimation, additive-separability grid,
   compute-optimal frontier validation. Bootstrap CIs over seeds; held-
   out validation on grid points not used for fitting.
5. **Layer 5 — robustness / natural-task tier** (§11). Full STU,
   selective Mamba, softmax-attention hybrid, in-context LDS
   identification. Phenomenology checks; not the source of fitted
   exponents.

**Implementation order chosen** (plan §1 governing principle 2):
Theorem-B first (it forces the Fourier utilities, exact stationary
generator, matched/mismatched-symbol logic, and modewise diagnostics
to harden); then theorem-C (forces the band-RRS generator, refinement
ladders, block commutant utilities, heterogeneity diagnostics);
**then** theorem-A (benefits from both); then architecture-aligned;
then scaling-law; then robustness.

The thesis chapter ordering is independent and follows the logical
hierarchy A → B → C → scaling-law.

---

## 3. Non-negotiable invariants

These global rules must never be violated:

1. **Real-valued by default** (plan §1 governing principle 3).
   The thesis codebase is real-valued. Complex Fourier helpers may be
   used internally inside `fourier_ops.py` and theorem-B exact
   scripts, but they must convert outputs to real observables before
   any later layer consumes them.

2. **Column-sample convention** (plan §3 first). The canonical
   internal representation is `X ∈ ℝ^{D × P}` with **columns as
   samples**. Any row-sample helper must be isolated behind named
   conversion functions.

3. **Theorem-A sample-space reduced operators are primary** (plan §3
   sixth + §10.1.5). The GA generator returns
   `A_S_GD ∈ ℝ^{B×P×P}`, `B_S_GD ∈ ℝ^{B×K×P}`, `T_GD = I_P + A_S_GD/L`,
   plus their perturbed variants (`A_S_theta`, `B_S_theta`, `T_theta`).
   Acceptance for A1–A4 evaluates against these sample-space matrices.
   A `return_feature_space=True` mode exposes secondary `(D, D)`
   helpers — they are diagnostic only, never the primary acceptance
   target.

4. **Theorem-B exact trajectory lives in `metrics.py`, not in the
   generator** (plan §10.2). `g1_generate` is a **pure data /
   operator generator** with no depth / step-size / trajectory-horizon
   parameters. Trajectories are computed by
   `metrics.gamma_star_trajectory_circulant`. Theorem-B scripts derive
   transfer functions from that trajectory; the generator never returns
   callables.

5. **Theorem-C operator-only vs sampled-context split** (plan §3
   sixth, §10.3). Theorem-C exact validation (C1–C7) is **operator-
   level**: consumes only `Lambda`, `Omega`, `partition`, `F` from
   `g2_generate_operator`. Sampled-context mode
   (`g2_generate_sampled`) is reserved for architecture-aligned
   experiments in §9.

6. **Mass-preserving heterogeneity** (plan §3 fifth, §10.3.2).
   Within-block `λ_{b,j}` and `ω_{b,j}` are constructed via
   `mass_preserving_block_spectrum` so that changing the within-block
   condition number `κ_b` does NOT change the block's average mass.
   This is what makes κ a clean axis and the C3/C4/C5/C6/C7 results
   interpretable.

7. **Theorem-C oracle hybrid = refined commutant optimum** (plan §7.6,
   binding).
   In the exact theorem-C tier the term **"oracle hybrid"** means
   *direct optimization over the refined commutant class*, not a
   learned or estimated projector. Architecture experiments in §9 will
   *approximate* this theorem-level reference. Do not confuse the two.

8. **C7 contraction overlay is a reference scale, NOT a strict upper
   bound or a power law.** `(ρ_b★)^{2L}` with
   `ρ_b★ = (κ_b − 1) / (κ_b + 1)` is the theorem-level reference
   contraction scale. The single-root polynomial `(1 − qλ/L)^{2L}`
   accessible to a single-scalar `q` filter converges *slower* than
   the Chebyshev-optimal polynomial of the same degree, so empirical
   `L★(L)` typically lies *above* the anchored reference. Observed /
   overlay > 1 is the correct physical regime, not a violation. Also:
   **no `L^{−β_b}` power-law fit** is claimed as the theorem-level
   result of C7 (plan §7.7 binding).

9. **Canonical hybrid = adaptive-first then spectral** (plan §1
   governing principle 4). The §9 / §10 canonical architecture is
   `L_A` dense linear-attention layers (adaptive module) followed by
   `L_S` FFT-based spectral filter layers with bottleneck `r`.
   Interleaved / parallel / local-attention hybrids belong to the
   robustness tier.

10. **No callables in generator return dicts.** `g1_generate`,
    `g2_generate_operator`, `g2_generate_sampled`, `g3_generate`,
    `ga_generate` all return tensors and pure metadata dicts. No
    closures, no `lambda` outputs.

11. **Frozen controls and frozen utility layer must not be modified
    casually.** Any change to `scripts/thesis/utils/*` requires a
    real downstream bug as justification, plus rerunning the
    `_self_tests/run_all.py` harness.

12. **Default `label_norm`** (plan §3 third, plan §10.1):
    - Theorem-B exact (B1) uses `sqrt_P` (matches the finite-P theorem
      derivation).
    - Theorem-A scripts and architecture-aligned scripts default to
      `sqrt_D`.
    Every run must record which normalization was used.

13. **Always source `starter.sh` and prefer CUDA via SLURM.** Login-
    node cgroup memory caps kill long-running scripts; A2 and A3+A4
    were submitted via `sbatch` after foreground attempts hit
    `exit 137` (SIGKILL). Submit any sweep of nontrivial size via
    SLURM.

---

## 4. Full utility / generator layer inventory

All paths relative to project root
`/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/spectral-scaling/spectral_scaling/`.

### 4.1 `scripts/thesis/utils/run_metadata.py` — **Complete, frozen**
Provides `ThesisRunDir` and `RunContext`.
- `ThesisRunDir(__file__, phase=...)`: builds an output directory
  `outputs/thesis/<phase>/<script_stem>/<run_id>/` with subdirectories
  `figures/`, `pdfs/`, `npz/`, `pt/`. Methods `.png(name)`, `.pdf(name)`,
  `.npz_path(name)`, `.pt_path(name)`, plus `.config_path`,
  `.metadata_path`, `.summary_path`, `.root`.
- `RunContext(run_dir, config=cfg, seeds=[...], notes=...)`: context
  manager that writes `config.json` and an initial `metadata.json`
  (`status="started"`) **before** any computation begins, captures
  per-step times via `record_step_time`, captures compute proxies via
  `record_compute_proxy`, captures arbitrary key/value extras via
  `record_extra(name, value)`, and writes `summary.txt` plus a
  finalized `metadata.json` on context exit.
- Required dependency for **every** theorem script in the thesis
  namespace; do not bypass.

### 4.2 `scripts/thesis/utils/plotting.py` — **Complete, frozen**
Canonical thesis figure style.
- `apply_thesis_style()` / `thesis_style()`: rcParams + seaborn
  whitegrid.
- Palette constants: `PALETTE_SEQUENTIAL = "rocket"`,
  `PALETTE_PHASE = "mako"`, `PALETTE_DIVERGING = "vlag"`,
  `PALETTE_CATEGORICAL = "colorblind"`.
- `sequential_colors(n, palette=...) -> list[RGB]`.
- `save_both(fig, run_dir, name) -> (png_path, pdf_path)` — emits
  both 300-dpi PNG (`figures/`) and `pdf.fonttype=42` PDF (`pdfs/`)
  for LaTeX embedding.
- `overlay_powerlaw(ax, x, *, coef, exponent, ...)` — dashed black
  reference for `coef · x^exponent`.
- `overlay_reference(ax, x, y, label=None, style="--", ...)` — generic
  theory-curve overlay.
- `phase_heatmap(ax, values, x_coords, y_coords, *, xlabel, ylabel,
  cbar_label, log_z, log_x, log_y, ...)` — canonical 2-D phase-diagram
  heatmap; returns `(pcolormesh, colorbar)`.
- `mode_trajectories`, `frontier_plot`, `legend_compact` — additional
  helpers used by various theorem-B/C/A scripts.
- **Convention reminders**: theorem-overlay lines are dashed black at
  `zorder=10`; sequential L / r / κ sweeps use `rocket`; phase-
  diagram heatmaps use `mako`; all matplotlib mathtext must avoid
  `\le` (use `\leq`) and `\mathcal` (use plain `C` or similar) — both
  trip the parser in our matplotlib version.

### 4.3 `scripts/thesis/utils/fourier_ops.py` — **Complete, frozen**
Canonical real-valued Fourier helpers.
- `freq_grid(P)`, `dft_matrix(P)`, `idft_matrix(P)`,
  `unitary_dft(P)`, `unitary_idft(P)`, `real_spectral_basis(P, kind)`.
- `circulant_from_symbol(s)` / `symbol_of_circulant(C)` /
  `real_even_symbol_from_half(half, P)`.
- Real-even symbol constructors: `symbol_power_law(P, nu, eps=1e-6)`,
  `symbol_multiband(P, bands)`, `symbol_flat(P, value)`. **All
  normalized so `mean(s) == 1`** — peak amplitude `s_max` therefore
  grows with `P` (especially for multiband). This is why B2 / B3 split
  multiband to small-P and use power_law for long-context P sweeps.
- `symbol_interpolate(s0, s1, alpha)` — convex blend.
- `frequency_permutation(s, *, seed)` — real-even-preserving
  permutation of positive-frequency indices.
- `off_diagonal_fourier_energy(M)` — circulant-closure diagnostic
  used by B1.

### 4.4 `scripts/thesis/utils/partitions.py` — **Complete, frozen**
- `BlockPartition` (frozen dataclass): `D`, `blocks` (tuple of disjoint
  index tuples), `.n_blocks`, `.sizes`, `.block_of(k)`,
  `.indicator_matrix()`, `.block_projector(b)`.
- `equal_blocks(D, m) -> BlockPartition` — `D/m` consecutive blocks of
  size `m`.
- `dyadic_ladder(D, J=None) -> list[BlockPartition]` — dyadic
  refinement ladder, coarsest to finest. Requires `D` a power of 2.
- `custom_ladder(levels) -> list[BlockPartition]` — validates a
  refinement chain.
- `mass_preserving_block_spectrum(partition, block_means,
  block_kappas, *, xi_shape, xi_custom, dtype)` — **the canonical
  mass-preserving heterogeneity construction** for theorem-C
  experiments. With `xi_shape="linear"` and `kappa_b > 1`, gives
  `λ_{b,j} = λ̄_b · κ_b^{ξ_j} / mean_j(κ_b^{ξ_j})` so the within-block
  arithmetic mean is exactly preserved.
- `mass_preserving_block_task` — same construction for `ω`.

### 4.5 `scripts/thesis/utils/commutants.py` — **Complete, frozen**
- `extract_block_scalars(Q, partition) -> Tensor` — returns `q[b] = (1/m_b) Σ_{k ∈ b} Q[k,k]`.
- `reconstruct_from_block_scalars(q, partition) -> Tensor` — builds
  `Σ_b q[b] · Π_b` as a `(D, D)` diagonal matrix.
- `commutant_projection(Q, partition)` — equals
  `reconstruct(extract(Q, ...))`; project a square matrix into the
  block commutant `C(B)`.
- `commutant_violation(Q, partition, *, normalize=True)` — returns
  `||Q − Π_C(Q)||_F^2 / ||Q||_F^2`. Zero up to float eps iff `Q ∈ C(B)`.
- `refines(fine, coarse) -> bool` — refinement predicate.
- These five APIs are the operational realization of the theorem-C
  block commutant class.

### 4.6 `scripts/thesis/utils/metrics.py` — **Complete, frozen**
Centralized metrics for theorem A / B / C and the scaling-law tier.
Major exports:

- `reduced_model_error(f_full, f_red, eps=1e-30) -> float` — relative
  L2 error `||f_full − f_red||_2 / (||f_full||_2 + eps)`. Theorem-A
  A1/A1b primary metric.
- `ab_perturbation_bound(A_theta, A_GD, B_theta, B_GD, T_theta, T_GD,
  L, y) -> dict` — **the full additive `(A, B)`-operator bound from
  plan §7.2** used by A2. Returns `delta_A_op`, `delta_B_op`,
  `S_theta_y_norm`, `B_side_bound`, `telescoping_coeff`,
  `A_side_bound`, `total_bound`, `empirical_error`. The decomposition
  is strict `total = B-side + A-side` with no cross term; B-side
  depends only on `ΔB`, A-side only on `ΔA` via the telescoping
  identity. Accepts unbatched or batched inputs.
- `gamma_star_trajectory_circulant(s_tr, omega, *, L, eta, T,
  gamma0=None) -> Tensor` — **the single source of truth for the
  per-mode reduced-Γ trajectory** used by B1, B2, B3 (training run),
  C2 (analog: see C1 script's `_evolve_grouped_ode`). Returns shape
  `(T+1, P)`. CPU `float64` internally; sequential Python loop —
  acceptable because the operator-level dimensions are small.
- `mode_trajectory_error(gamma_hat, gamma_star)` — pointwise relative
  error per mode; pathological when `|gamma_star_k| ≪ 1` (B1's
  primary metric was changed to a **max-magnitude-scaled** rel error
  to avoid this; that variant is computed inline in B1, not exposed
  as a generic util).
- `transfer_function_error(T_hat, T_star)` — L2 grid error.
- `grouped_trajectory_error(q_hat, q_star)` — per-block relative
  error (theorem-C analog).
- `oracle_commutant_loss(lam, omega, partition, L, q_init=None,
  optimizer="lbfgs", max_iter=500) -> dict` — **canonical L-BFGS
  optimizer over block scalars** for the theorem-C block-commutant
  loss
  `L(q, L) = Σ_b Σ_{j∈b} ω_{b,j} · λ_{b,j} · (1 − q_b · λ_{b,j} / L)^{2L}`.
  Returns `{"q_star", "loss_star", "per_block_loss", "converged"}`.
  Used by C3 (numerical reference), C4, C5, C6, C7. Soft-converged
  (xtol-stopping) cells in C7 are acceptable per
  `convergence_required=False`.

### 4.7 `scripts/thesis/utils/cost_models.py` — **Complete, frozen**
Implements the canonical compute proxy of plan §4.4:
`C_proxy = t · (c_S · Φ_S(P, r) · L_S + c_A · Φ_A(P) · L_A)` with
`Φ_A(P) = P²` (dense linear attention) and
`Φ_S(P, r) = P log P + P r` (FFT spectral with bottleneck `r`).
Plus a wall-clock measurement helper so figures can compare analytic
proxy vs measured cost. **No theorem script under §6/§7/§8 uses it
yet**; the architecture-aligned and scaling-law tiers will.

### 4.8 `scripts/thesis/utils/fit_powerlaws.py` — **Complete, frozen**
Centralized log-log power-law fitting.
- `fit_loglog(x, y, *, fit_window, heteroskedastic_weights=None) ->
  dict` — weighted log-log LSQ with mandatory fit-window argument
  (no auto-windowing per plan §9.1). Returns `slope`, `intercept`,
  `r2`, `residuals`, `fit_x`, `fit_y`. Used by B4 (rank-floor fit).
- `bootstrap_exponent(...)`, `holdout_evaluate(...)` — for the
  scaling-law tier; not yet consumed.
- **Binding**: every exponent reported in the thesis must come from
  this utility. C7 explicitly does NOT fit `L^{−β_b}` because the
  theorem-C finite-block prediction is contraction-style, not a power
  law (plan §7.7).

### 4.9 `scripts/thesis/utils/data_generators.py` — **Complete, frozen**
Houses GA, G1, G2 (operator-only + sampled), G3. Detailed contracts in §5
below. **Frozen** unless a downstream theorem script exposes a real
bug; all changes must keep `_self_tests/run_all.py` green.

### 4.10 `scripts/thesis/utils/_self_tests/run_all.py` — **Complete, frozen**
Hard-gate self-test harness. Run via
`python -u scripts/thesis/utils/_self_tests/run_all.py`. Final state
reported in earlier conversation: `exact: 51/51`, `MC: 6/6`. Coverage
includes G1 covariance identity, query-mode correctness, GA mask
construction (gd_compatible / perturbed / non_gd_control), G2 block-
Haar structure, operator-only G2, commutant invariance, G3 refinement
validity, dyadic_ladder validity, bootstrap CI smoke-test, and oracle
commutant loss correctness on small problems. **Must remain green
before any theorem-script edit is merged.**

---

## 5. Generator contracts in theorem language

### 5.1 GA — theorem-A masked-context generator
- **API**: `GAConfig(D, P, K, B=1, Sigma_kind, Omega_kind, sigma,
  label_norm, mask_kind, mask_perturbation, non_gd_kind, Gamma_kind,
  Gamma_params, L, return_feature_space=False, seeds, dtype, device)` →
  `ga_generate(cfg) -> dict`.
- **Object**: produces the theorem-A masked-context regression problem.
  - `X_train ∈ ℝ^{B×D×P}`, `X_query ∈ ℝ^{B×D×K}` sampled with columns
    `~ N(0, Σ)`.
  - `beta ∈ ℝ^{B×D}` sampled `~ N(0, Ω)`.
  - `y_train`, `y_query` produced via `β · X / norm` with
    `norm = sqrt(D)` (default) or `sqrt(P)`.
  - `mask ∈ ℝ^{(P+K)×(P+K)}` is the **signed mask** (not a binary
    attention mask). For `gd_compatible`: `M[:P,:P] = −1`,
    `M[P:,:P] = +1`, zeros elsewhere. For `perturbed`:
    `M[:P,:P] = −1 + θ · Δ` where `Δ ∈ ℝ^{P×P}` is a Frobenius-
    normalized symmetric Gaussian seeded by `pattern_seed`. For
    `non_gd_control` with `signflip_testtest`: `M[P:,:P] = −1` (test→
    train sign flipped); with `nonzero_testblock`: `M[:P, P:] = +1`
    (train can see queries — outside theorem A).
  - **Sample-space reduced operators (PRIMARY)**:
    - `A_S_GD = −X_train^T · Γ · X_train / P ∈ ℝ^{B×P×P}`,
    - `B_S_GD = +X_query^T · Γ · X_train / P ∈ ℝ^{B×K×P}`,
    - `T_GD = I_P + A_S_GD / L`,
    - plus perturbed `A_S_theta`, `B_S_theta`, `T_theta`.
  - With `return_feature_space=True`, secondary `(D, D)` operators
    `A_feat_GD`, `B_feat_GD`, `A_feat_theta`, `B_feat_theta` are
    additionally exposed; **diagnostic only**, never the primary
    acceptance target.
- **Theorems / experiments**: A1, A1b, A2, A3, A4. The `mask_kind`
  switch is the central knob.

### 5.2 G1 — theorem-B exact stationary circulant generator
- **API**: `G1Config(P, D=None, B=1, query_mode, query_position,
  matched_query_realization, symbol_kind_tr, symbol_params_tr,
  symbol_kind_te, symbol_params_te, task_spec_kind, task_spec_params,
  sigma, label_norm, exact_mode, sample_data, population_mode,
  seeds, dtype)` → `g1_generate(cfg) -> dict`.
- **Object**: theorem-B stationary circulant data generator.
  - **Three distinct spectra**: `s_tr` (training symbol), `s_te` (test/
    query symbol; equals `s_tr` when `symbol_kind_te="matched"`),
    `omega` (teacher / task spectrum, set by `task_spec_kind`).
  - Operator-level returns: `s_tr`, `s_te`, `omega`,
    `Sigma_tr = circulant_from_symbol(s_tr)`,
    `Sigma_te = circulant_from_symbol(s_te)`.
  - With `sample_data=True`: also `X_train`, `X_query`, `y_train`,
    `y_query`, `beta`. With `population_mode=True`: forbids
    `sample_data=True`.
  - `query_mode="full_window"` (used by B1) or `"single_query"`.
  - `matched_query_realization`: `"independent"` (B1 default) or
    `"shared"`.
- **Theorems / experiments**: B1, B2, B3 (training phase), B4. **No
  trajectory parameters** — `gamma_star_trajectory_circulant` lives
  in `metrics.py`.

### 5.3 G2 — theorem-C band-RRS generator (operator-only AND sampled)
- **APIs**:
  - `G2Config(D, partition_kind, partition_params, block_means_lam,
    block_kappas_lam, block_means_omega, block_kappas_omega, xi_shape,
    spectral_basis_kind, spectral_basis_custom, label_norm, sigma,
    seeds, dtype)`.
  - `g2_generate_operator(cfg) -> dict` — **the operator-only mode**
    (used by C1–C7).
  - `g2_generate_sampled(cfg, n_contexts, P, K) -> dict` — sampled
    mode for §9 architecture-aligned experiments (not yet consumed).
- **Operator-only return**: `partition` (`BlockPartition`), `F`
  (real spectral basis matrix, default DCT-II), `Lambda` (length-D
  spectrum vector, mass-preserving per the partition), `Omega`
  (length-D teacher spectrum vector), `block_means_lam`,
  `block_kappas_lam`, `block_means_omega`, `block_kappas_omega`,
  `rho_star = (κ_b − 1)/(κ_b + 1)`. **No matrices** — Σ, Ω are
  represented as length-D diagonals, materialized inline by callers if
  needed.
- **Sampled return**: in addition to all operator-only keys, `X_train`,
  `X_query`, `y_train`, `y_query`, `beta`, and the per-context
  block-Haar rotation `R ∈ ℝ^{n_contexts × D × D}` for diagnostics.
  Per-context: `x_{c,μ} ~ N(0, F^T R_c diag(λ) R_c^T F)`.
- **Mass-preserving heterogeneity**: built via
  `mass_preserving_block_spectrum`; `xi_shape="linear"` is the
  canonical default.

### 5.4 G3 — refinement-ladder generator
- **APIs**:
  - `G3Config(D, ladder_kind, ladder_params, reference_partition_index,
    base_block_means_lam, base_block_kappas_lam, base_block_means_omega,
    base_block_kappas_omega, xi_shape, spectral_basis_kind, dtype)`.
  - `g3_generate(lam, omega, ladder, *, F=None, dtype) -> list[dict]`
    — direct-tensor API.
  - `g3_generate_from_config(cfg) -> list[dict]` — constructive API
    (builds ladder + (lam, omega) at the reference partition then
    delegates to `g3_generate`).
- **Refinement ladder invariant**: at every level `j`, the returned
  dict carries `partition`, `F`, `Lambda`, `Omega`, plus per-level
  block stats (`block_means_lam`, `block_kappas_lam`,
  `block_means_omega`, `block_kappas_omega`, `rho_star`). **`Lambda`,
  `Omega`, and `F` are bit-wise identical across all levels** — only
  the partition changes. The ladder is asserted to be a valid
  refinement chain at construction time (`refines(ladder[j+1],
  ladder[j])` for all `j`).
- **Theorems / experiments**: C5 (refinement monotonicity ladder).

---

## 6. Frozen Bordelon control baseline

- **Location**: `outputs/thesis/controls/`.
- **Six frozen controls** (per plan §5):
  1. `run_isotropic_depth_vs_alpha.py`
  2. `run_fixed_covariance.py`
  3. `run_reduced_gamma_dynamics.py`
  4. `run_compute_scaling_joint.py`
  5. `run_linear_attention_dynamics.py`
  6. `run_softmax_depth_sweep.py` (the optional sixth)
- **Freeze semantics**: **archive-by-copy of EXISTING reproduced
  outputs**, not a fresh re-run. The freezer
  (`scripts/thesis/freeze_bordelon_controls.py`) copies artifacts
  already under `outputs/<script_stem>/`, computes SHA-256 hashes for
  every file, marks every frozen file `0444` (read-only), writes a
  per-control `metadata.json` with `freeze_mode =
  "archive_by_copy_of_existing_outputs"`, `archival_note`, SHA-256
  manifest, git commit, env fingerprint, and `reproduction_command`.
- **Top-level index**: `outputs/thesis/controls/FROZEN.json`.
- **Per-directory README**: `outputs/thesis/controls/README.md`
  explicitly documents the archive-by-copy semantics.
- **Status**: **Immutable**. Do not re-freeze, do not overwrite, do
  not modify. They serve as the calibration package against which the
  new theorem-tier scripts are compared.

---

## 7. Theorem-B exact experimental block

**Block status**: B1, B2, B3, B4 complete. B0 pending. B5 deferred.

### 7.1 B0 — structure-closure diagnostic (PENDING)
- **Path** (planned): `scripts/thesis/theoremB/run_theoremB_structure_closure.py`.
- **Plan ref**: §6.1.
- **Role**: theorem-A bridge check that the reduced operator
  preserves circulant / Toeplitz-like structured classes in the
  settings relevant for theorem-B. For circulant exact mode, the
  off-circulant energy of `A_S` should be machine-precision small.
- **Status**: not yet implemented. Not on the critical path —
  user prioritized B1→B4 directly. Worth backfilling later.

### 7.2 B1 — exact finite-P circulant mode closure (COMPLETE)
- **Path**: `scripts/thesis/theoremB/run_theoremB_circulant_modes.py`.
- **Plan ref**: §6.2. **Plays the spectral analog of Bordelon Fig 3a.**
- **Role**: validates that when the training covariance is circulant,
  the reduced-Γ recursion **decouples exactly into per-Fourier-mode
  scalar ODEs**. Compares two structurally distinct paths to machine
  precision:
  1. **Matrix recursion** in `ℝ^{P×P}`:
     `Γ(t+1) = Γ(t) + η · Ω · Σ² · (I − L⁻¹ Σ Γ(t))^{2L−1}` with
     `Σ, Ω` real-symmetric circulant. Per-mode values extracted at
     each step via FFT of the first column of Γ (Γ stays circulant
     because it commutes with Σ, Ω, and `Γ(0) = 0`).
  2. **Per-mode recursion** from
     `metrics.gamma_star_trajectory_circulant`.
- **Sweep**: P ∈ {16, 32, 64} × L ∈ {1, 2, 4, 8} × symbol ∈
  {flat, power_law, multiband} = 36 trials.
- **Acceptance metrics** (max-magnitude-scaled relative error to avoid
  the eps-band pathology of pointwise relative error):
  - `mode_rel_err_max ≤ 1e-10`,
  - `transfer_rel_err_max ≤ 1e-10`,
  - `off_diagonal_fourier_energy ≤ 1e-10`.
- **Final result**: 36/36 trials passed. Max `mode_rel_err = 1.08e-14`,
  max `transfer_rel_err = 4.89e-15`, `off_diag = 0.00e+00`.
- **Important caveats / corrections**:
  - First implementation used `eta=0.005`, `power_law_nu=1.5` →
    NaN divergence at high-amplitude modes (transient overshoot of
    `(1 − sγ/L)^{2L−1}` with odd exponent amplifying). **Final config**
    `eta=1e-4`, `power_law_nu=0.5`, `task_spec_nu_beta=1.0`,
    `multiband=((0,2,1.0), (5,7,0.8))`, `T=100`. ~2 orders margin from
    the stability boundary `η · ω · s³ · (2L−1)/L < 2`.
  - Default device must be CUDA via `starter.sh`; `_resolve_device`
    raises if CUDA requested but unavailable.
- **Canonical artifact**:
  `outputs/thesis/theoremB/run_theoremB_circulant_modes/run_theoremB_circulant_modes-20260413T051032Z-3a5e4cb4/`.
- **Final approved interpretation**: the per-mode recursion in
  `metrics.gamma_star_trajectory_circulant` is verified bit-exactly
  against the matrix recursion across the sweep — this is the
  theorem-B exact circulant closure result and the calibration check
  that all later theorem-B experiments inherit.

### 7.3 B2 — long-context matched-stationary depth irrelevance (COMPLETE)
- **Path**: `scripts/thesis/theoremB/run_theoremB_depth_stationary.py`.
- **Plan ref**: §6.3. Spectral analog of Bordelon Fig 3b.
- **Role**: in the matched stationary regime (`s_tr = s_te`, G1
  population mode), demonstrate that increasing spectral depth
  beyond a shallow baseline does NOT introduce a depth-dependent
  asymptotic floor. Finite-T cross-L differences are transient-rate
  effects (L=1 exponential, L>1 polynomial of order
  `−1/(2L−2)`).
- **Sweep**: split into two parts to handle `mean(s)=1` symbol
  normalization (multiband peak amplitude grows like P):
  - **Main sweep**: P ∈ {32, 64} × symbol ∈ {power_law, multiband}
    × L ∈ {1, 2, 4, 8, 16} = 20 trials.
  - **Long-context sub-sweep**: P ∈ {128, 256} × power_law × L ∈
    {1, 2, 4, 8, 16} = 10 trials.
  - Total 30 trials. `T=100000`, `eta=5e-5` (3× margin from stability
    at the worst-case P=256 power_law L=16 cell).
- **Acceptance**:
  - **Monotonicity**: every loss trajectory is non-increasing within
    `1e-9` slack.
  - **Per-trial decay**: `loss(T) / loss(0) ≤ depth_decay_fraction =
    0.20` for every trial.
- **Final result**: 30/30 trials passed. Max decay fraction = **6.2%**
  (well under 0.20). Cross-L terminal ratio `L(L_max)/L(L_min)` is
  diagnostic only, not a gate (it can be huge when L=1 hits float
  eps and L=16 still has finite residual — that is a transient-rate
  effect, not a depth floor).
- **Canonical artifact**:
  `outputs/thesis/theoremB/run_theoremB_depth_stationary/run_theoremB_depth_stationary-20260413T060815Z-f2cbb0fb/`.
- **Final approved interpretation**: **Finite-time matched-stationary
  depth-irrelevance experiment**. No evidence of a depth-dependent
  asymptotic floor; finite-T cross-L differences are transient-rate
  effects (L=1 exponential, L>1 polynomial of order `−1/(2L−2)`).
  Per-trial decay is the operational form of the asymptote-collapse
  claim.

### 7.4 B3 — symbol-native OOD brittleness (COMPLETE)
- **Path**: `scripts/thesis/theoremB/run_theoremB_symbol_shift.py`.
- **Plan ref**: §6.4. Spectral analog of Bordelon Fig 3c.
- **Role**: theorem-B fixed-basis OOD brittleness under
  **symbol-native** shifts — explicitly NOT a generic covariance
  rotation (those belong to a later bridge experiment toward theorem
  C).
- **Two shift families** (both binding per plan §6.4):
  - **Family 1 — structural interpolation**:
    `s_te(α) = (1−α) · s_tr + α · s_other` with `s_other = symbol_flat`
    by default. α sweep `{0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4,
    0.5, 0.65, 0.8, 1.0}`.
  - **Family 2 — frequency permutation interpolation**:
    `s_te(α, seed) = (1−α) · s_tr + α · permute(s_tr, seed)` with 8
    permutation seeds.
- **Forward**: trains via matched recursion (`s_tr = s_te`), then
  evaluates the stationary loss formula with shifted `s_te`:
  `L_ood(γ, s_tr, s_te, ω, L) = Σ_k ω_k · s_te,k · (1 − L⁻¹ s_tr,k γ_k(T))^{2L}`.
- **Acceptance**:
  - **Matched-baseline recovery**: at `α=0`, `L_ood ≡ L_matched(γ(T))`
    to float eps (verified `1e-30` slack — exact).
  - **Full-shift brittleness gate**: at `brittleness_alpha = 1.0`,
    at least one L gives `L_ood / L_matched ≥ 1.25`.
- **Final result**: matched recovery `f1_err = 0.0`, `f2_err = 0.0`.
  Brittleness at α=1: ratios 1.30–1.51× across L for family 1;
  family 2 medians 1.27–1.48×. Both gates pass.
- **Wording requirement (binding)**: A2-style "full-shift gate"
  explanation is required in the summary because finite-time matched
  training in the B2 regime leaves the baseline partially converged
  on slow modes (`s_k^3` small), so small-α ratio looks modest. The
  full-shift gate avoids that confounder.
- **Category requirement (binding)**: the script is a **theorem-B
  fixed-basis symbol-native OOD experiment**. Generic covariance
  rotation is a separate later bridge experiment toward theorem C.
- **Canonical artifact**:
  `outputs/thesis/theoremB/run_theoremB_symbol_shift/run_theoremB_symbol_shift-20260413T063315Z-59725122/`.

### 7.5 B4 — spectral-rank bottleneck and joint (r, L_S) sweep (COMPLETE)
- **Path**: `scripts/thesis/theoremB/run_theoremB_rank_scaling.py`.
- **Plan ref**: §6.5.
- **Role**: pure-spectral-shape law BEFORE hybridization. Spectral
  rank `r` is the primary control variable; depth `L_S` is the
  secondary axis.
- **Mechanism**: spectral-rank-r bottleneck restricts the recursion
  to update only modes with `k_star = min(k, P-k) < r`. Mode-decoupling
  of the matched circulant recursion lets us train ONCE per L
  (unmasked) then post-mask γ for each r.
- **Sweep**: P=256 × r ∈ {1, 2, 4, 8, 16, 32, 64, 128} × L ∈
  {1, 2, 4, 8} = 32 evaluations from 4 training trajectories.
- **Acceptance**:
  - **Floor power-law fit (FINITE-P)**: `log L★(r, L=1)` vs `log r`
    fitted over `r ∈ [4, 64]` must agree with the analytical
    finite-P fit slope `Σ_{k_star ≥ r} ω_k · s_k` to within
    `floor_exponent_tol = 0.15` relative tolerance. **Acceptance is
    against the FINITE-P analytical slope, NOT the continuum
    asymptote.**
  - **Depth collapse** at r = max: `loss(L=8)/loss(L=1) ≤ 5×`.
- **Final result**: empirical L=1 slope = **−0.820**, analytical
  finite-P slope = **−0.833** (fit window [4, 64]), `err = 1.6%`
  (well under 15% tolerance), R² = 0.98 both. Depth collapse at
  r=128: ratio = 1.77 (under 5×).
- **Critical wording correction (binding, written into the script)**:
  "B4 validates the FINITE-P spectral-rank floor: primary acceptance
  compares empirical rank-floor slope to the analytical finite-P
  tail-sum slope `Σ_{k_star ≥ r} ω_k · s_k`. The continuum exponent
  `1 − (ν + νβ) = −0.50` is a REFERENCE asymptote only, not the
  acceptance target. The observed steeper slope at the chosen P=256
  and fit window [4, 64] is a finite-size effect as r approaches
  P/2 (tail sum shortens), NOT a theorem failure."
- **Canonical artifact**:
  `outputs/thesis/theoremB/run_theoremB_rank_scaling/run_theoremB_rank_scaling-20260413T074851Z-93d98cc1/`.

### 7.6 B5 — periodic LDS identification (DEFERRED)
- **Plan ref**: §6.6. Optional natural-task extension.
- **Status**: not implemented; per user's instruction not to block on
  optional work. Worth implementing only if the rest of the program
  is stable and time / compute remain.

---

## 8. Theorem-C exact experimental block

**Block status: Complete.** This is the empirical centerpiece of the
thesis. Every script in this block is operator-level only; no learned
architectures.

### 8.1 C1 + C2 + C1-MC-Haar follow-up (COMPLETE)
- **Path**: `scripts/thesis/theoremC/run_theoremC_commutant_closure.py`.
- **Plan refs**: C1 = §7.1, C2 = §7.2.
- **Combined into one script** (user-approved). Section labels in
  the figures keep C1 and C2 outputs clearly separated.
- **C1 role**: under the band-RRS regime with `Σ_c = R_c · diag(λ) · R_c^T`,
  `Ω_c = R_c · diag(ω) · R_c^T`, the population reduced-Γ recursion
  preserves the block commutant `C(B)` exactly. Three recursion paths
  are tracked at each step:
  1. **R-averaged matrix recursion** (the C1 PRIMARY): per-step
     update is the naïve per-F-mode gradient projected into `C(B)`
     by block-averaging. By construction lives in `C(B)`.
  2. **Naïve per-F-mode recursion** (NEGATIVE CONTROL): no
     R-averaging; under within-block heterogeneity `κ_b > 1` it
     leaves `C(B)`.
  3. (C2) **Grouped scalar ODE** on `q ∈ ℝ^{n_blocks}`:
     `δq_b = η · (1/m_b) Σ_{j∈b} ω_j λ_j² (1 − L⁻¹ λ_j q_b)^{2L−1}`.
- **C1 acceptance**: `commutant_violation(diag(γ_R(t)), partition) ≤
  1e-12` for every step; **observed `cv_r_max = 0.000e+00`** bit-exact.
  Negative control naïve `cv_n_max = 0.296` (≫ contrast threshold).
- **C2 acceptance**: matrix-extracted block scalars match grouped-
  ODE output to `1e-12`; **observed `q_err_max = 0.000e+00`**.
- **MC-Haar follow-up acceptance** (added later per user request to
  close the projection-vs-population-Haar-average gap): at the largest
  `mc_n_samples_list[-1] = 50000`, the relative Frobenius error
  between `Π_C(diag(naive_grad))` and the MC estimate of
  `E_R[R · diag(naive_grad) · R^T]` is `≤ mc_haar_tol = 2e-2`.
  Observed worst rel_err = **2.95e-3** at N=50000, with clean
  `1/√N` convergence (6.8e-2 → 2.9e-3 over N=100→50000).
- **Sweep**: D=64, partition equal × m=8 (n_blocks=8), L ∈ {1, 2, 4, 8},
  T=5000.
- **Heterogeneity setup**: block_means_lam = `(1.0, 0.9, ..., 0.3)`,
  block_kappas_lam = `(2.0,) × 8`, similar for omega.
- **Canonical artifact**:
  `outputs/thesis/theoremC/run_theoremC_commutant_closure/run_theoremC_commutant_closure-20260413T083220Z-849ef5fc/`.
- **Final approved interpretation**:
  - C1 validates the exact band-RRS population-averaged commutant
    dynamics.
  - The MC-Haar consistency check closes the projection-vs-population-
    average gap.
  - C2 validates the grouped scalar ODE against the matrix-level
    averaged path. **C2 is interpreted only AFTER C1 acceptance
    passes.**
  - The naïve non-averaged path is a **negative control only**.

### 8.2 C3 — L=1 closed-form block lower bound (COMPLETE)
- **Path**: `scripts/thesis/theoremC/run_theoremC_L1_closed_form.py`.
- **Plan ref**: §7.3.
- **Role**: at L=1 the block-commutant loss is quadratic in `q_b`
  per block. Closed form: `q_b★ = b_b / c_b`,
  `L_b★ = a_b − b_b² / c_b` with
  `(a, b, c) = (Σ ω·λ, Σ ω·λ², Σ ω·λ³)` over block `b`. Compared to
  the L-BFGS numerical optimum (`metrics.oracle_commutant_loss` at
  `L=1`).
- **Sweep**: D=64, m ∈ {1, 2, 4, 8, 16, 32} × κ ∈ {1.0, 1.1, 1.2,
  1.5, 2.0, 3.0, 5.0, 10.0} = 48 trials.
- **Acceptance**:
  - Closed form vs numerical: `max rel_err_loss ≤ 1e-8`,
    `max |q_cf − q_num| ≤ 1e-6`.
  - κ = 1: `|L★| ≤ 1e-10` for every m.
- **Final result**: max rel loss err = **1.42e-13**, max
  `|q_cf − q_num| = 8.88e-16` (float eps). κ=1 worst = **0.0** exact.
  Two diagnostic monotonicity violations at m=2 with κ ∈ {3, 5, 10}
  — explained: under the mass-preserving linear-ξ construction, at
  extreme κ with very small blocks one mode's weight concentrates
  enough that `L★` decreases again. Diagnostic only; not a theorem
  failure.
- **Canonical artifact**:
  `outputs/thesis/theoremC/run_theoremC_L1_closed_form/run_theoremC_L1_closed_form-20260413T083943Z-987650ca/`.
- **Final approved interpretation**: C3 validates the exact L=1 closed-
  form block-commutant lower bound against the numerical oracle
  optimum. The non-monotonicity at extreme κ for tiny blocks is a
  property of the heterogeneity parameterization, kept diagnostic only.

### 8.3 C4 — heterogeneity phase diagram (COMPLETE — HEADLINE FIGURE)
- **Path**: `scripts/thesis/theoremC/run_theoremC_phase_diagram.py`.
- **Plan ref**: §7.4. **Main headline figure of the thesis
  experiments** per plan.
- **Role**: 2D phase diagram in `(m, κ)` of the spectral-only
  obstruction and the **dyadic one-step refinement gain**. Sweep
  depth `L` as a secondary axis.
- **Sweep**: D=64, m ∈ {1, 2, 4, 8, 16, 32} × κ ∈ {1.0, 1.2, 1.5,
  2.0, 3.0, 5.0, 10.0} × L ∈ {1, 2, 4, 8, 16}. Three optimizations
  per cell: coarse (size m), dyadic-finer (size m/2), singleton (full
  oracle, sanity ≡ 0). 210 cells × 3 opts = 630 L-BFGS runs.
- **Three primary heatmaps**:
  - `(a) L_coarse(m, κ)` — spectral-only obstruction.
  - `(b) L_fine(m, κ)` — refined-class optimum at dyadic finer
    partition.
  - `(c) gap(m, κ) = L_coarse − L_fine` — refinement gain (one step).
  Plus contour overlays + κ-slice line panel + (m, L) depth-
  interaction sub-figure.
- **Acceptance**:
  - **Refinement nonnegativity at m ≥ 2**: `gap ≥ −1e-7` everywhere.
  - **κ = 1 degeneracy**: `|L_coarse| ≤ 1e-9` everywhere at κ=1.
  - **Full oracle ≡ 0**: singleton numerical optimum `≤ 1e-7`.
- **Final result**: worst neg gap = **−5.24e-12** (float eps), κ=1
  worst = **9.69e-12**, full-oracle worst = **2.16e-10**. All gates
  pass. Max refinement gain at L=4: **0.523** at (m=8, κ=10).
- **Canonical artifact**:
  `outputs/thesis/theoremC/run_theoremC_phase_diagram/run_theoremC_phase_diagram-20260413T090447Z-a95a1cc9/`.
- **Final approved interpretation**:
  - **C4 is the theorem-C headline figure** — exact operator-level
    phase diagram over (m, κ).
  - The coarse-vs-fine gain panel refers to the **dyadically-finer
    commutant class**, NOT the singleton oracle.
  - The singleton partition is a sanity / oracle reference only.

### 8.4 C5 — refinement monotonicity ladder (COMPLETE)
- **Path**: `scripts/thesis/theoremC/run_theoremC_refinement_monotonicity.py`.
- **Plan ref**: §7.5.
- **Role**: extends C4's one-step dyadic gap to a **full multi-level
  ladder** from coarsest single-block to singleton via G3.
- **Setup**: D=64 dyadic ladder = **7 ladder levels and 6 refinement
  steps** (block sizes 64 → 32 → 16 → 8 → 4 → 2 → 1). G3 reference
  partition index 0 (coarsest). κ sweep `{1.0, 1.2, 1.5, 2.0, 3.0,
  5.0, 10.0}`. L ∈ {1, 4} (secondary axis). Total 7×7×2 = 98 L-BFGS.
- **Acceptance**:
  - **Monotonicity along ladder**: `L★(j+1) ≤ L★(j) + 1e-8` for every
    (κ, L).
  - **κ=1 flat**: `|L★| ≤ 1e-9` everywhere at κ=1.
  - **Finest level ≡ 0**: at singleton `|L★| ≤ 1e-8`.
- **Final result**: worst monotonicity violation = **1.25e-13** (float
  eps); κ=1 worst = **6.81e-13**; finest worst = **7.94e-11**. Strict-
  drop counts at L=1: κ=1 → 0/6 (flat ladder, as predicted), every
  κ>1 → **6/6** strict drops (full staircase).
- **Canonical artifact**:
  `outputs/thesis/theoremC/run_theoremC_refinement_monotonicity/run_theoremC_refinement_monotonicity-20260413T094019Z-3f84bcd0/`.
- **Wording correction (binding)**: precise count is "**7 ladder
  levels and 6 refinement steps**" — not "6 levels". Singleton level
  is an oracle sanity endpoint only.

### 8.5 C6 — oracle hybrid defined correctly (COMPLETE)
- **Path**: `scripts/thesis/theoremC/run_theoremC_oracle_hybrid.py`.
- **Plan ref**: §7.6.
- **PRIMARY PURPOSE**: make the theorem-level definition of "oracle
  hybrid" precise: **oracle hybrid = direct optimization over the
  refined commutant class** (NOT a learned, estimated, or trained
  projector). Architecture experiments in §9 will *approximate* this
  reference.
- **Three operator-level objects per cell**:
  - **L_coarse** — coarse-class commutant optimum at partition size m.
  - **L_hybrid** — refined-class commutant optimum at partition size
    m/2 (= **the oracle hybrid**).
  - **L_unconstrained** — singleton-partition optimum (= oracle
    ceiling; ≡ 0 in matched regime).
- **Captured fraction**: `F = (L_coarse − L_hybrid) / L_coarse`.
- **Sweep**: D=64, m ∈ {2, 4, 8, 16, 32} × κ ∈ {1.0, 1.2, 1.5, 2.0,
  3.0, 5.0, 10.0} × L ∈ {1, 2, 4, 8, 16}. 175 cells × 3 opts = 525
  L-BFGS.
- **Acceptance**:
  - Three-way ordering `L_coarse ≥ L_hybrid ≥ L_unconstrained`
    everywhere within `1e-7`.
  - Oracle ceiling worst `|L_u| ≤ 1e-7`.
  - At m=2: `F ≡ 1` within `captured_fraction_m2_tol = 1e-3`
    (relaxed from 1e-6 because L-BFGS noise on small-L_coarse cells
    gives ~5e-6 deviation).
- **Final result**: worst (L_c − L_h) = **−5.24e-12**, worst
  (L_h − L_u) = **−1.06e-11**, worst |L_u| = **2.16e-10**, worst
  |F − 1| at m=2 = **5.08e-6**. All gates pass.
- **Canonical artifact**:
  `outputs/thesis/theoremC/run_theoremC_oracle_hybrid/run_theoremC_oracle_hybrid-20260413T094705Z-a62ff6f3/`.
- **Binding summary text fields** (preserved in summary.txt):
  `oracle_hybrid_definition`, `not_architecture_comparison`,
  `category`, `interpretation`. The "oracle hybrid is NOT a learned
  projector" line is the binding theorem-level definition.

### 8.6 C7 — finite-depth scaling in the grouped band-RRS class (COMPLETE)
- **Path**: `scripts/thesis/theoremC/run_theoremC_depth_scaling.py`.
- **Plan ref**: §7.7.
- **PRIMARY THEOREM-LEVEL FRAMING**: For a fixed block with finite
  condition number, the correct theorem-C finite-depth overlay is
  NOT a generic `L^{−β_b}` power law. It is the geometric/contractive
  law controlled by `ρ_b★ = (κ_b − 1) / (κ_b + 1)`, with the
  reference scale `(ρ_b★)^{2L}` anchored at L=1.
- **Sweep**: D=64, m ∈ {1, 2, 4, 8, 16, 32} × κ ∈ {1.0, 1.5, 2.0,
  3.0, 5.0, 10.0} × L ∈ {1, 2, 4, 8, 16, 32, 64} = 252 L-BFGS.
- **Acceptance**:
  - Singleton m=1 flat: `|L★| ≤ 1e-8` everywhere.
  - κ=1 flat: `|L★| ≤ 1e-8` everywhere.
  - Monotone non-increase in L for every (m, κ).
  - L-BFGS convergence is a soft diagnostic (236/252 converged via
    gradient norm; the rest are xtol-stopped at acceptable accuracy).
- **Final result**: singleton worst = **1.92e-11**, κ=1 worst =
  **1.92e-11**, monotonicity worst Δ = **7.64e-12** (all float eps).
  Contraction-overlay diagnostic: observed/envelope ratios at L=64
  range from ~7 (κ=10) to ~10^76 (κ=1.5). The overlay is more
  optimistic than the grouped-scalar single-root polynomial filter
  achieves.
- **CRITICAL WORDING (binding, written in script docstring + summary)**:
  - "(ρ★)^(2L) is a **reference contraction scale, NOT a strict upper
    bound**, and the single-root polynomial converges *slower* than
    the Chebyshev optimal."
  - "Observed/overlay > 1 is the **expected and physically correct
    regime**, not a violation."
  - "**No β_b power-law fit** is claimed as the theorem-level result"
    — only a diagnostic empirical-semi-log-slope-vs-2·log(ρ★)
    comparison is computed, explicitly labeled as DIAGNOSTIC.
- **Canonical artifact**:
  `outputs/thesis/theoremC/run_theoremC_depth_scaling/run_theoremC_depth_scaling-20260413T164624Z-846d8bc6/`.

---

## 9. Theorem-A exact experimental block

**Block status: Complete.** A1, A1b, A2, A3, A4 all done. A1b was
added after a clarification: A1's R1 was an iterative reduced
recursion, not a true full-model forward.

### 9.1 A1 — reduced-theory consistency (COMPLETE — kept for the record)
- **Path**: `scripts/thesis/theoremA/run_theoremA_exact_equivalence.py`.
- **Plan ref**: §8.1 (first attempt at the exact equivalence test).
- **Role**: deterministic forward-pass equivalence test of three
  routes:
  - **R1** — iterates `(A_S, B_S, T)` from GA on a length-(P, K) state
    pair: `z_test ← z_test + (1/L) B_S z_train`,
    `z_train ← z_train + (1/L) A_S z_train`. **In hindsight this is
    iterative reduced, not a full-model forward.**
  - **R2** — closed-form sample-space `(A_S, B_S)` recursion.
  - **R3** — feature-space reduced-Γ closed form (preconditioned-GD
    iterate).
- **Sweep**: D ∈ {8,16,32,64} × P ∈ {8,16,32,64} × K ∈ {4,8,16} × L ∈
  {1,2,4,8} = 192 cells, three routes per cell.
- **Acceptance**: pairwise rel err `≤ 1e-10`.
- **Final result**: worst R1-R2 = **3.78e-16**, worst R2-R3 =
  **1.03e-15**, worst R1-R3 = **1.14e-15**. All pass at float eps.
- **Status note**: A1's canonical result is **kept and unchanged** as
  a reduced-theory consistency experiment; it is **not** the
  primary theorem-A bridge result. That role moves to A1b.
- **Canonical artifact**:
  `outputs/thesis/theoremA/run_theoremA_exact_equivalence/run_theoremA_exact_equivalence-20260413T165457Z-a1a3f216/`.

### 9.2 A1b — true full-hidden-state bridge (COMPLETE — PRIMARY A1 RESULT)
- **Path**: `scripts/thesis/theoremA/run_theoremA_exact_equivalence_full_model.py`.
- **Plan ref**: §8.1 (extended/corrected). **This is the canonical
  theorem-A bridge result.**
- **Role**: closes the gap left by A1's R1. Adds a TRUE full hidden-
  state aligned structured forward pass that:
  - builds the full `(P+K)×(P+K)` bilinear score
    `S = X^T Γ X / P` from `X = [X_train | X_query]`,
  - applies the GD-compatible **signed mask** (`−1` on train×train
    rows, `+1` on test×train rows, `0` elsewhere), taken directly from
    `op["mask"]` returned by GA,
  - runs L explicit residual-stream layer updates on a length-(P+K)
    scalar hidden channel,
  - **does not consume `(A_S, B_S, T)` from GA** — those are produced
    implicitly by the structured forward.
- **Three routes**: R0 (full hidden state), R2 (reduced AB), R3
  (reduced Γ).
- **Sweep + acceptance**: same as A1.
- **Final result**: worst R0-R2 = **1.03e-15**, worst R2-R3 =
  **1.03e-15**, worst R0-R3 = **1.47e-15**. All 192 cells pass at
  float eps.
- **Canonical artifact**:
  `outputs/thesis/theoremA/run_theoremA_exact_equivalence_full_model/run_theoremA_exact_equivalence_full_model-20260413T171410Z-1cb9a436/`.
- **Final approved interpretation**:
  - **A1 = reduced-theory consistency experiment.**
  - **A1b = full-model theorem-A bridge experiment — the canonical
    end-to-end exactness result.**
  - Both preserved; **A1b is primary for the theorem-level writeup**.

### 9.3 A2 — perturbation around GD-compatibility (COMPLETE)
- **Path**: `scripts/thesis/theoremA/run_theoremA_mask_perturbation.py`.
- **Plan ref**: §8.2.
- **Role**: faithful theorem-A perturbation diagnostic. Compares
  empirical full-model error against the **full additive
  `(A, B)`-operator reduced perturbation bound** from
  `metrics.ab_perturbation_bound`. **B-side and A-side are reported
  separately and never folded into each other.**
- **Empirical route binding**: both `F_θ` and `F_GD` are computed via
  the **A1b R0 full-hidden-state forward**. The reduced operators
  `(A_S_θ, B_S_θ, T_θ)` from GA are consumed **only** by the bound
  computation, never to define the empirical error.
- **Two perturbation modes** (binding distinction):
  - **A_only — CANONICAL theorem-A perturbation family** (plan §8.2).
    GA `mask_kind="perturbed"`: train-train block `−1 + θ·Δ`; test-
    train left at GD value. Produces `ΔA ∝ θ`, `ΔB ≡ 0`. Bound
    dominated by A-side.
  - **B_only — AUXILIARY decomposition diagnostic** (NOT a theorem-A
    family). Manually-built mask: `+1 + θ·Δ_B` on test-train block;
    train-train at GD value. Produces `ΔA ≡ 0`, `ΔB ∝ θ`. Included
    solely so the additive bound's B-side term is exercised with
    non-zero values (the GA "perturbed" mode keeps `ΔB ≡ 0`). The
    thesis writeup MUST keep this distinction explicit.
- **Sweep**: 2 configs × 4 seeds × 9 θ × 2 modes = **144 trials**.
  Configs (D, P, K, L) ∈ {(32,32,8,4), (64,32,8,4)}. θ ∈ {0, 1e-5,
  1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2}. Same X, β across (GD,
  perturbed) so only the mask differs at a given seed.
- **Acceptance**:
  - θ=0: `||F_0 − F_GD||₂ ≤ 1e-10` for every seed/config.
  - For all (θ > 0, seed, config): empirical `≤ total_bound + 1e-12`.
  - A-side and B-side reported per cell.
- **Final verified result** (SLURM job 6937652, 7.5 s wall):
  - θ=0 worst empirical = **0.000e+00** across all 144 trials.
  - 0 bound violations.
  - A_only: max `emp/bound = 0.101`, max excess = **−2.02e-6**.
  - B_only: max `emp/bound = 0.445`, max excess = **−1.27e-6**.
  - Worst bound slack `min(total/empirical) = 2.246`.
- **Canonical artifact**:
  `outputs/thesis/theoremA/run_theoremA_mask_perturbation/run_theoremA_mask_perturbation-20260413T194034Z-6d128b1d/`.
- **Binding summary text fields**: `framing`, `primary_vs_auxiliary`,
  `empirical_route`, `bound_components`, `interpretation`,
  `per_mode_aggregates`.

### 9.4 A3 + A4 — semiseparable / SSD realization + negative controls (COMPLETE)
- **Path**: `scripts/thesis/theoremA/run_theoremA_semiseparable.py`.
  (Combined per user-approved approach with clearly separated
  sections.)
- **Plan refs**: A3 (PRIMARY) = §8.3; A4 (SECONDARY) = §8.4.

#### A3 — explicit semiseparable / SSD realization (PRIMARY)
- **Role**: validates the explicit theorem-consistent semiseparable
  realization of the GD-compatible structured mask. Insight: the
  signed mask `M_signed = (1_test − 1_train) · 1_train^T` is
  **rank-1** in the sequence dimension, so the layer update admits
  an explicit D-dimensional state recursion:
  `z^(ℓ+1) = z^(ℓ) + (1/(P·L)) · diag_sign · X^T · Γ · (X · Π_train · z^(ℓ))`.
  This is the SSD-style realization that theorem A asserts is
  equivalent to the reduced operator object.
- **Three routes** at every (D, P, K, L):
  - **R_SSD** (the new explicit semiseparable route).
  - **R_AB** — sample-space reduced (A_S, B_S) recursion.
  - **R_full** — A1b R0 full-hidden-state forward (cross-check).
- **Acceptance gate (PRIMARY)**: `R_SSD vs R_AB ≤ 1e-10`.
  Cross-check: `R_SSD vs R_full ≤ 1e-10`.
- **Sweep**: same as A1/A1b: 192 cells.
- **Final result**: worst R_SSD vs R_AB = **1.63e-15**, worst
  R_SSD vs R_full = **2.03e-15**, worst R_AB vs R_full = **6.57e-16**.

#### A4 — negative controls outside the theorem-A class (SECONDARY)
- **Role**: show that exact reduced-Γ behavior is NOT universal —
  the theorem-A hypotheses are load-bearing. **A4 results MUST be
  labeled as negative controls, NOT as theorem-A failures.**
- **Two negative controls**:
  - **NC1 — pure linear circulant convolutional mixer**. Fixed
    Gaussian-kernel circulant convolution applied as
    `z^(ℓ+1) = z^(ℓ) + (1/L) · C · z^(ℓ)`. No data-dependent
    bilinear K·Q structure, no dependence on (X, Γ). Outside theorem A.
  - **NC2 — non-GD-compatible structured mask**. GA `mask_kind=
    "non_gd_control"` with `non_gd_kind="signflip_testtest"`
    (`B_S_theta = −B_S_GD`). Bilinear structure preserved; mask
    violates GD-compatibility. Outside theorem A.
- **Sweep**: 4 seeds at (D=32, P=32, K=8, L=4).
- **Acceptance**: each NC's relative deviation from the GD reduced
  prediction ≥ `a4_min_relative_deviation = 0.1`.
- **Final result**: NC1 min rel.dev = **1.195** (≥ 0.1), NC2 min
  rel.dev = **2.000** (≥ 0.1). Both deviate non-trivially —
  exactly the experimental point.
- **Canonical artifact (combined)**:
  `outputs/thesis/theoremA/run_theoremA_semiseparable/run_theoremA_semiseparable-20260413T201529Z-61f65d3e/`.

---

## 10. Current repository additions outside `utils/`

### `scripts/thesis/theoremB/`
| File | Status | Purpose |
|---|---|---|
| `__init__.py` | ✓ | namespace |
| `run_theoremB_circulant_modes.py` | B1 ✓ | exact finite-P circulant mode closure |
| `run_theoremB_depth_stationary.py` | B2 ✓ | matched stationary depth-irrelevance |
| `run_theoremB_symbol_shift.py` | B3 ✓ | symbol-native OOD brittleness |
| `run_theoremB_rank_scaling.py` | B4 ✓ | spectral-rank floor + joint (r, L_S) |

### `scripts/thesis/theoremC/`
| File | Status | Purpose |
|---|---|---|
| `__init__.py` | ✓ | namespace |
| `run_theoremC_commutant_closure.py` | C1+C2+MC-Haar ✓ | commutant closure + grouped ODE + MC-Haar consistency |
| `run_theoremC_L1_closed_form.py` | C3 ✓ | L=1 closed-form block lower bound |
| `run_theoremC_phase_diagram.py` | C4 ✓ | heterogeneity phase diagram (HEADLINE) |
| `run_theoremC_refinement_monotonicity.py` | C5 ✓ | full dyadic refinement ladder |
| `run_theoremC_oracle_hybrid.py` | C6 ✓ | oracle hybrid defined correctly |
| `run_theoremC_depth_scaling.py` | C7 ✓ | finite-depth grouped scaling (contraction overlay) |

### `scripts/thesis/theoremA/`
| File | Status | Purpose |
|---|---|---|
| `__init__.py` | ✓ | namespace |
| `run_theoremA_exact_equivalence.py` | A1 ✓ (reduced-theory consistency) | reduced-theory three-way consistency |
| `run_theoremA_exact_equivalence_full_model.py` | A1b ✓ (PRIMARY bridge) | true full-hidden-state forward vs reduced |
| `run_theoremA_mask_perturbation.py` | A2 ✓ | full additive (A,B) perturbation bound |
| `run_theoremA_semiseparable.py` | A3+A4 ✓ | SSD realization + NC1 circ-conv + NC2 non-GD mask |

### `configs/thesis/`
Mirror configs for every script above (frozen-default reference; the
scripts do NOT auto-load these JSONs — CLI flags only):
- `controls/...` (control-suite freeze metadata index, not a config).
- `theoremB/run_theoremB_circulant_modes.json`
- `theoremB/run_theoremB_depth_stationary.json`
- `theoremB/run_theoremB_symbol_shift.json`
- `theoremB/run_theoremB_rank_scaling.json`
- `theoremC/run_theoremC_commutant_closure.json`
- `theoremC/run_theoremC_L1_closed_form.json`
- `theoremC/run_theoremC_phase_diagram.json`
- `theoremC/run_theoremC_refinement_monotonicity.json`
- `theoremC/run_theoremC_oracle_hybrid.json`
- `theoremC/run_theoremC_depth_scaling.json`
- `theoremA/run_theoremA_exact_equivalence.json`
- `theoremA/run_theoremA_exact_equivalence_full_model.json`
- `theoremA/run_theoremA_mask_perturbation.json`
- `theoremA/run_theoremA_semiseparable.json`

### `experiments/thesis/`
SLURM launchers, one per theorem script:
- `theoremB/run_theoremB_circulant_modes.sh`
- `theoremB/run_theoremB_depth_stationary.sh`
- `theoremB/run_theoremB_symbol_shift.sh`
- `theoremB/run_theoremB_rank_scaling.sh`
- `theoremC/run_theoremC_commutant_closure.sh`
- `theoremC/run_theoremC_L1_closed_form.sh`
- `theoremC/run_theoremC_phase_diagram.sh`
- `theoremC/run_theoremC_refinement_monotonicity.sh`
- `theoremC/run_theoremC_oracle_hybrid.sh`
- `theoremC/run_theoremC_depth_scaling.sh`
- `theoremA/run_theoremA_exact_equivalence.sh`
- `theoremA/run_theoremA_exact_equivalence_full_model.sh`
- `theoremA/run_theoremA_mask_perturbation.sh`
- `theoremA/run_theoremA_semiseparable.sh`

All launchers: `--partition=ailab`, `--gres=gpu:1`, `--mem=16G`,
`--time` ≤ 30 min, `--cpus-per-task=4`, source `starter.sh`. Use
`sbatch` rather than foreground execution for any script larger than
trivial size — login-node cgroups will SIGKILL.

---

## 11. Canonical output / artifact structure

Every theorem script writes to:
```
outputs/thesis/<phase>/<script_stem>/<run_id>/
├── figures/         300-dpi PNG figures
├── pdfs/            Type-42 PDF figures (LaTeX-embeddable)
├── npz/             numpy archives (raw arrays)
├── pt/              torch tensors / checkpoints (rare)
├── config.json      exact config used (written on RunContext __enter__)
├── metadata.json    run_id, seeds, git, env, timings, compute proxy, status, extras
├── summary.txt      human-readable summary written on RunContext __exit__
├── per_trial_summary.json (or per_cell_summary.json / per_alpha_summary.json / per_point_summary.json)
└── (no run.log — SLURM writes stdout to logs/)
```

`run_id = <script_stem>-<UTC timestamp>-<8 hex chars>`.

**Reproducibility contract** (plan §2): every run must save the exact
config, seed list, git commit hash, Python+CUDA env description, wall-
clock runtime, analytic compute proxy, measured per-step wall-clock,
plain-text summary of fitted quantities, and run ID. **Both raw arrays
and human-readable summaries** because the thesis must reconstruct
results long after.

Required summary text fields (per script type):
- All scripts: `plan_reference`, `category`, `interpretation`,
  `status` (status string captures gate pass/fail).
- Theorem-A scripts: also `framing`, plus
  `primary_vs_auxiliary` (A2) or `primary_vs_secondary` (A3+A4).
- C6: `oracle_hybrid_definition` and `not_architecture_comparison`
  binding fields.
- C7: `theorem_framing` and `no_power_law_fit` binding fields.
- B4: `framing` field with finite-P clarification.
- C5: `n_ladder_levels` and `n_refinement_steps` fields (NOT `J_used`).

---

## 12. Acceptance-gate philosophy

Layer 1 — **frozen control replication**: passes by archival hash
matching the original Bordelon outputs.

Layer 2 — **exact theorem-validation**: acceptance is **algebraic**,
not statistical. Two flavors:
- **Machine-precision exactness** — the result is an algebraic
  identity that should hold to `float64 eps` (`~1e-15`) up to
  arithmetic accumulation. Examples: B1 closure, A1b R0 vs R2,
  A3 R_SSD vs R_AB, C1 commutant violation. Tolerance set at
  `1e-10` (5 orders of margin).
- **Theorem-level exact acceptance with finite-precision slack** —
  the result holds to a stated finite tolerance reflecting numerical
  optimization noise. Examples: C3 closed form vs L-BFGS (1e-8),
  C4 refinement nonnegativity (1e-7), C6 m=2 captured-fraction
  boundary (1e-3 because L-BFGS noise on small `L_coarse` cells gives
  ~5e-6 deviation). The tolerance must be set high enough to
  accommodate the optimizer's stopping criterion, not so high that it
  hides actual theorem failures.

**Diagnostic-only overlays / metrics** (NOT acceptance gates):
- C7 contraction overlay `(ρ★)^{2L}` — reference scale only.
- B2 cross-L terminal ratio — diagnostic only (per-trial decay
  fraction is the gate).
- C3 monotonicity in κ at small m — diagnostic only.
- C7 empirical-semi-log-slope vs `2·log(ρ★)` — diagnostic only;
  explicitly NOT a `β_b` fit.

Layer 3 — **architecture-aligned qualitative support** (PENDING):
the architecture must reproduce the predicted theorem-level
**transitions** (e.g., the C4 phase boundary, the B4 rank floor
shape, the B2 depth-irrelevance asymptote). Acceptance is
"qualitative match within a stated relative tolerance averaged over
seeds." Statistical: report per-seed standard errors.

Layer 4 — **conditional scaling-law validation** (PENDING):
fit-window enforcement (no auto-windowing per plan §9.1), bootstrap
CIs over seeds, held-out validation on grid points not used for
fitting, frontier regret reporting.

---

## 13. Lessons learned / important corrections

### Theorem-B
- **B1 stability vs eta**: initial `eta=5e-3` with `power_law nu=1.5`
  caused NaN divergence on the largest-amplitude modes (transient
  overshoot through the odd `(1 − sγ/L)^{2L−1}` exponent). Final
  config drops `eta` to `1e-4` (~2 orders of margin) and `nu` to
  `0.5`. Lesson: **always check the stability boundary
  `η · max(ω · s³) · (2L−1)/L < 2` AND the transient overshoot
  regime (where the recursion can diverge before reaching the fixed
  point).**
- **B1 metric pathology**: `metrics.mode_trajectory_error` uses
  pointwise `|emp − star| / (|star| + 1e-30)` which blows up for
  modes where `|γ★| ≪ 1`. B1 instead computes a **max-magnitude-
  scaled** relative error inline: `mode_rel_err_max = abs_err_max /
  (max(|γ★|) + 1e-30)`. The frozen pointwise metric is kept as a
  diagnostic; the max-magnitude variant is the acceptance metric.
- **B2 depth interpretation**: original draft suggested deeper L is
  "slower to converge" pejoratively. **Correct interpretation**:
  no depth-dependent asymptotic floor; finite-T cross-L differences
  are transient-rate effects (L=1 exponential, L>1 polynomial of
  order `−1/(2L−2)`). Wording binding for the thesis writeup.
- **B3 OOD must be symbol-native first**: the plan §6.4 binding is
  that the primary OOD experiment must be **symbol-native**
  (interpolation + frequency permutation), NOT generic covariance
  rotation. Generic rotation is a later bridge experiment toward
  theorem C, separate from B3.
- **B3 brittleness gate** at small α was confounded by the matched
  baseline being partially converged on slow modes (B2 regime). Final
  gate fires at `brittleness_alpha = 1.0` (full shift), not at small
  α. Wording explanation kept in the script summary.
- **B4 finite-P vs continuum**: the rank-floor power-law fit's
  primary acceptance compares the **empirical slope to the analytical
  finite-P fit slope**, NOT to the continuum asymptote `1−(ν+νβ)`.
  The continuum exponent is reported as a reference only. The
  observed steeper slope at the chosen `(P, fit_window)` is a
  finite-size effect as `r → P/2` (tail sum shortens), NOT a theorem
  failure.

### Theorem-C
- **Exact tier must stay operator-level** (plan §3 sixth + §7
  binding). All of C1–C7 consume `g2_generate_operator`, NOT
  `g2_generate_sampled`. Sampled-context is for §9.
- **C1 negative control vs primary**: the naïve per-F-mode recursion
  is a **negative control only** showing that R-averaging is required
  to enforce the commutant; it is NOT a theorem-C failure. Wording
  binding.
- **C1 MC-Haar follow-up was REQUIRED before C3** to close the
  projection-vs-population-Haar-average gap. The algebraic
  `commutant_projection` is shown to equal `E_R[R · diag(u) · R^T]`
  to MC noise floor at large N. Without this, C1's "by construction"
  claim has a logical hole.
- **C3 non-monotonicity at extreme κ for tiny blocks** is a property
  of the mass-preserving linear-ξ parameterization (one mode
  concentrates), NOT a theorem failure. Diagnostic only.
- **C4 oracle vs refinement**: the C4 "refinement gain" panel
  refers to the **dyadically-finer commutant class**, NOT the
  singleton oracle. The singleton appears only as a sanity reference
  (`L_full_oracle ≡ 0`).
- **C5 wording**: precise count is "**7 ladder levels and 6
  refinement steps**" for the D=64 dyadic ladder; not "6 levels".
- **C6 binding definition**: "oracle hybrid" = direct optimization
  over the refined commutant class. NOT a learned, estimated, or
  trained projector. Binding written into the summary's
  `oracle_hybrid_definition` field.
- **C7 contraction reference is not a power law**: plan §7.7
  binding. The script must NOT default to `L^{−β_b}` fitting.
  The `(ρ★)^{2L}` overlay is a **reference contraction scale, NOT
  a strict upper bound**. Observed/overlay > 1 is the expected
  regime because the single-root polynomial converges slower than
  Chebyshev optimal. Earlier wording said "looser than the single-
  scalar q filter can achieve" — that was BACKWARDS and was
  corrected.

### Theorem-A
- **A1's R1 was iterative reduced, not full-model**. A1b was
  added to provide the true full-hidden-state aligned forward
  (build full (P+K)×(P+K) score from (X, Γ); apply signed mask;
  iterate L layers). A1b is the canonical theorem-A bridge result;
  A1 is kept as the reduced-theory consistency experiment. Binding
  distinction in summaries.
- **A2 must use full additive (A, B) bound**: not just an A-only
  comparison and not folding the B-side into `T_θ − T_GD`.
  `metrics.ab_perturbation_bound` does this correctly (no cross
  term; B-side depends only on `ΔB`, A-side only on `ΔA` via the
  telescoping identity). Binding.
- **A2 A_only vs B_only distinction**: A_only is the **canonical
  theorem-A perturbation family** (plan §8.2). B_only is an
  **auxiliary decomposition diagnostic** added solely so the
  additive bound's `ΔB` term is exercised with non-zero values.
  B_only is NOT a theorem-A perturbation family. Binding.
- **A2 empirical error must use the full-model R0 route from A1b**;
  reduced operators are used only by the bound computation. Binding.
- **A3 SSD must be theorem-consistent**: not a generic SSD family.
  The explicit theorem-consistent realization exploits the rank-1
  structure of the GD-compatible signed mask
  `M_signed = (1_test − 1_train) · 1_train^T`, yielding a D-dim
  state recursion `z^(ℓ+1) = z^(ℓ) + (1/(P·L)) · diag_sign · X^T · Γ ·
  (X · Π_train · z^(ℓ))`.
- **A4 negative controls are NOT theorem-A failures**: plan §8.4
  binding. NC1 (circular conv, no bilinear K·Q) and NC2 (non-GD
  mask) are deliberately outside the theorem-A class; their
  deviation is the experimental point. Wording binding.

### Process / engineering
- **`starter.sh` always sourced; CUDA via SLURM mandatory** for
  any sweep larger than trivial. Login-node cgroup memory limits
  killed the foreground A2 run twice (`exit 137`); SLURM submission
  fixed it in 7 s.
- **Wording matters in generated summaries**: every approval came
  with explicit wording corrections (e.g., "n_ladder_levels and
  n_refinement_steps" instead of "J_used"; "no β_b fit" disclaimer
  in C7; "category: theorem-B fixed-basis symbol-native OOD" in B3;
  "primary_vs_auxiliary" in A2). These are preserved in
  `ctx.write_summary` payloads of each canonical run.
- **Stale dev runs were systematically cleaned up** after each
  approval — a single canonical run dir per script, named with the
  most-recent timestamp. Pattern: `LATEST=$(ls -td .../*/ | head -1)`,
  `for d in $(ls -td .../*/ | tail -n +2); do rm -rf "$d"; done`.
- **NPZ size hygiene**: B2 initially saved full `(T+1, P)` γ
  trajectories per trial → 1.25 GB. Trimmed by saving only scalar
  loss trajectories + once-per-(P, symbol) `s_tr`, `omega`. Resulting
  npz is ~17 MB.
- **matplotlib mathtext** does NOT support `\le` (use `\leq`) or
  `\mathcal` (use plain `C` or similar). Both tripped the parser
  during early figure generation.

---

## 14. Current pending work

| Phase | Item | Notes |
|---|---|---|
| Theorem-B | **B0 structure-closure diagnostic** (§6.1) | not implemented; not on critical path; nice-to-have backfill once theorem-A reduced-operator code exists (it does now) |
| Theorem-B | **B5 periodic LDS identification** (§6.6) | optional; deferred unless time/compute remain |
| Theorem-A | **all complete** | A1, A1b, A2, A3, A4 done |
| Architecture-aligned (§9) | **§9.1 spectral-only suite** | canonical FFT spectral filter + STU + S4 surrogate; theorem-B-aligned tasks (stationary circulant regression, spectral OOD shift, spectral bottleneck sweeps) |
| Architecture-aligned (§9) | **§9.2 structured-mask / SSD suite** | explicit linear semiseparable / SSD realization (theorem-A bridge), Mamba-2-style SSD secondary; theorem-A bridge checks + theorem-C band-RRS grouped behavior |
| Architecture-aligned (§9) | **§9.3 canonical adaptive-first hybrid suite** | the most important architecture block — `L_A` dense linear-attention + `L_S` FFT-spectral with bottleneck `r`. Compares spectral-only / oracle-refined commutant optima / learned hybrid on band-RRS data. Measures projector-estimation error, loss gap to oracle, sensitivity to (L_A, L_S, r). Direct connection to the C4/C6 oracle objects. |
| Conditional scaling-law (§10) | **§10.1 separate exponent estimation** | β_S, β_r first (cleanest spectral models); then β_A, β_P (canonical hybrid with L_S, r large enough); then β_t (large architecture). Fixed fit windows in config files. |
| Conditional scaling-law (§10) | **§10.2 additive-separability grid** | factorial sweep over (L_S, L_A, r, P, t) with fit/validation split; report median + max relative error + exponent stability. |
| Conditional scaling-law (§10) | **§10.3 compute-frontier validation** | predicted-optimal vs brute-force best on the canonical compute model `C = t(c_S(P log P + Pr) L_S + c_A P² L_A)`; analytic vs measured frontier ordering. |
| Conditional scaling-law (§10) | **§10.4 oracle vs learned distinction** | every L_A frontier experiment must include an oracle refinement reference. Mandatory per plan §10.4. |
| Robustness (§11) | full STU, S4, selective Mamba, softmax-attention hybrid, in-context LDS extension | preserves theorem phenomena qualitatively; not the source of fitted exponents |

---

## 15. Immediate next step

**Implement §9.1 spectral-only architecture suite — the FFT-based
spectral filter on theorem-B-aligned tasks.**

- **Suggested filename**: `scripts/thesis/architectures/run_spectral_only_circulant.py` (or similar; the
  script ancestry is `run_depth_scaling_nonrotate.py`,
  `run_fixed_covariance.py`, `run_compute_scaling_width.py`).
- **Theorem role**: architecture-aligned validation — show that the
  theorem-B mechanisms (matched-stationary depth-irrelevance,
  symbol-native OOD brittleness, spectral-rank bottleneck) survive in
  a realistic trainable spectral architecture, not just at the
  operator level.
- **Data generator**: G1 with `sample_data=True` (the architecture-
  aligned tier consumes sampled contexts, not operator-only).
- **Architecture**: canonical FFT-based spectral filter with
  controllable bottleneck `r`. `L_S` spectral layers. Real-valued.
  Primary architecture. STU and S4 are secondary additions to be
  added in follow-up scripts under the same `architectures/`
  directory.
- **Acceptance**: qualitative match — the trained model must reproduce
  the theorem-B phenomena (depth-irrelevance asymptote shape, OOD
  brittleness curve shape, rank-floor scaling exponent within a stated
  relative tolerance). Statistical: per-seed standard errors required.
- **Constraints (do NOT change while implementing)**:
  - Keep all theorem-tier scripts (§6, §7, §8) and their canonical
    artifacts untouched.
  - Use the frozen Step-1b utility / generator layer as-is.
  - Default to CUDA via SLURM submission.
  - Use `cost_models.py` for compute-proxy reporting.
  - Use `fit_powerlaws.fit_loglog` for any exponent fit; fixed fit
    window from the config.
  - Per-seed standard errors required because this is the first tier
    where statistical (not algebraic) acceptance applies.

After §9.1, proceed to §9.2 (structured-mask / SSD suite), then §9.3
(canonical hybrid), then §10, then §11.

---

## 16. Guardrails for future Claude after compaction

Before writing any code, **read in this order**:

1. **`EXPERIMENT_PLAN_FINAL.MD`** — the canonical plan. It is the
   authoritative ordering, framing, and acceptance contract.
2. **This file** (`THESIS_EXPERIMENTS_STATE_DUMP.md`) — what is built,
   what each canonical artifact contains, and what corrections have
   been agreed on.
3. **`CLAUDE.md`** at project root — the existing project conventions.
4. **`scripts/thesis/README.md`** — the navigational aid.
5. The specific theorem script you intend to extend — its module
   docstring records every binding wording and every framing
   correction agreed on in conversation.

### Things NOT to reinterpret

- **C7's contraction overlay is a reference scale, NOT a strict upper
  bound, NOT a power law.** Do not "fix" C7 by adding a `L^{−β_b}`
  fit. The `(ρ★)^{2L}` reference is the theorem-level statement.
- **A1 is the reduced-theory consistency experiment; A1b is the
  canonical full-model bridge.** Do not collapse them.
- **A2's A_only is canonical; B_only is auxiliary.** Do not present
  B_only as a theorem-A perturbation family. Do not fold B-side into
  `T_θ − T_GD`. The bound has the additive structure
  `total = B-side + A-side` and the two are reported separately.
- **A4 NC1, NC2 are negative controls.** Do not present their
  deviations as theorem-A failures.
- **C6's "oracle hybrid" = refined commutant optimum.** Do not
  conflate with learned projectors.
- **Theorem-C exact tier is operator-level.** Do not introduce
  sampled-context experiments into C1–C7.
- **Generators do NOT return callables** and do NOT contain
  trajectory parameters.
- **Frozen controls are archive-by-copy and `0444`.** Do not re-run
  them. Do not overwrite. Do not re-freeze.

### Things NOT to "simplify"

- The **mass-preserving heterogeneity construction** —
  `block_means_lam`, `block_kappas_lam`, `xi_shape="linear"`. Changing
  this changes the meaning of every theorem-C result.
- The **GA generator's signed mask convention** — `−1` train×train,
  `+1` test×train, with `−1 + θ·Δ` for `perturbed`. The full-hidden-
  state forward depends on this signed-mask convention; do not
  switch to a binary mask.
- The **B1 max-magnitude-scaled relative error** — do not revert to
  the pointwise relative error from `metrics.mode_trajectory_error`.
- The **per-trial decay fraction acceptance in B2** — do not revert
  to a cross-L ratio; it is uninformative when L=1 reaches float eps.

### Things to treat as canonical

- `EXPERIMENT_PLAN_FINAL.MD` § ordering and binding wording.
- The 14 completed canonical artifact paths listed in §7, §8, §9 of
  this dump.
- The Step-1b utility / generator API surface (do not extend casually;
  add new utilities in new files if needed).
- The `RunContext` / `ThesisRunDir` metadata contract.
- The summary text fields with binding wording (`framing`,
  `interpretation`, `category`, `oracle_hybrid_definition`,
  `not_architecture_comparison`, `theorem_framing`,
  `no_power_law_fit`, `primary_vs_auxiliary`, `primary_vs_secondary`,
  `empirical_route`, `bound_components`, `acceptance_framing`).

### Things to treat as deprecated

- The **first B1 config** (`eta=5e-3`, `nu=1.5`, `device="cpu"`).
  Diverged. Replaced by `eta=1e-4`, `nu=0.5`, `device="cuda"`.
- **A1's R1 as a "full forward"** — it is iterative reduced, not full
  model. A1b is the corrected primary route.
- The **first C7 acceptance gate** (`contraction_envelope_max_ratio`)
  — it was wrong: the envelope is not a strict upper bound. Replaced
  by **monotonicity in L** as the gate; contraction overlay reported
  as a diagnostic.
- The **wording "contraction overlay is looser than the single-scalar
  q filter can achieve"** — BACKWARDS. Corrected to: "the overlay is
  more optimistic than the grouped scalar optimum; the single-root
  polynomial converges slower than Chebyshev optimal."
- The **A2 acceptance gate at `brittleness_alpha = 0.2`** in B3 — was
  too sensitive to the partially-converged matched baseline. Final
  gate fires at α=1 (full shift).
- The **first attempt to perturb the GA "perturbed" mask kind to
  produce ΔB ≠ 0** — the GA "perturbed" mode keeps `B_S` unchanged by
  design. Use the manual `B_only` path in A2 instead.

---

## Restart checklist (for a future Claude after compaction)

1. **Read `EXPERIMENT_PLAN_FINAL.MD` end-to-end.**
2. **Read this file end-to-end.** Pay special attention to §3
   invariants, §13 lessons, §16 guardrails.
3. **List the canonical artifact directory of the most recently
   completed script** to confirm the state is what this dump
   describes:
   `ls -td outputs/thesis/theoremA/run_theoremA_semiseparable/*/`
   should show one directory ending `-61f65d3e`.
4. **Read `scripts/thesis/utils/_self_tests/run_all.py`'s last
   reported state** to confirm utility-layer health
   (`exact: 51/51`, `MC: 6/6`).
5. **The next script to implement** is the §9.1 spectral-only
   architecture suite (see §15 above). Suggested path:
   `scripts/thesis/architectures/run_spectral_only_circulant.py`
   with companion config + SLURM launcher.
6. **Always submit nontrivial sweeps via SLURM**
   (`sbatch experiments/thesis/architectures/<launcher>.sh`); do not
   run on the login node.
7. **Always write the canonical summary text fields** (§11 of this
   dump) so the next future Claude after the next compaction can
   carry context forward.
8. **After every approved script**, clean up stale dev runs so the
   canonical artifact directory contains exactly one timestamped run
   dir.

End of state dump.
