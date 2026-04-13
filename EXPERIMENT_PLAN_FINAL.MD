# Final Merged Experimental Plan for the Thesis

## 0. Purpose and status of this document

This document is the final experimental implementation contract for the thesis. It supersedes every earlier roadmap. It is written for a setting in which the theorem chapters already exist in near-final form, the original Bordelon baselines have been reproduced successfully, and the thesis now needs an experimental program that is both mathematically faithful and practically implementable. The plan is therefore organized around the exact theorem hierarchy that has already been developed in the thesis, while also incorporating the strongest implementation-level corrections surfaced by repeated audits of the current code, the current generator draft, and the alternate experimental roadmaps that were reviewed.

The thesis has four mathematical layers. The problem formulation chapter introduces the masked in-context regression task, the linear teacher model, the reduced-\(\Gamma\) baseline, and the statistical regimes of interest. Theorem A proves the exact structured reduced-operator bridge and identifies when the standard reduced-\(\Gamma\) predictor is recovered. Theorem B proves exact fixed-basis stationary spectral closure in the circulant regime and its associated modewise dynamics, depth irrelevance at large context in the matched stationary limit, OOD brittleness under spectral mismatch, and a spectral bottleneck law. Theorem C proves the band-random-rotation symmetry restriction, grouped finite-depth dynamics, a finite-depth spectral-only obstruction controlled by within-block heterogeneity, and exact hybrid advantage via partition refinement. The final scaling-law chapter is not an exact theorem in the same sense as A–C; it is a conditional empirical chapter which assumes approximate resource separability and uses the exact mechanisms isolated by A–C to fit and validate a hybrid mixture scaling law and a compute-optimal allocation rule. The experimental plan must mirror that structure exactly. The experiments must not begin with the largest or fanciest architectures. They must begin with exact theorem objects, then move to architecture-aligned models, then move to the conditional scaling-law layer, and only after that perform robustness and natural-task extensions. This is the same mechanism-first logic that structures the original Bordelon–Letey–Pehlevan paper and its codebase: exact reduced model first, fixed-structure and random-rotation regimes second, scaling laws third, realistic architectures last.

This final plan also has a second purpose. Claude Code will implement from this document, and Claude knows only the original Bordelon repository plus the files that have already been reviewed in this conversation. That means the plan cannot rely on any tacit knowledge. Every theorem experiment must state exactly what object it is validating, what data generator it consumes, what model class it uses, what metric determines success or failure, what original Bordelon script should be cloned or altered, and what figure or table it is meant to produce. This document therefore includes mathematical reasoning, implementation dependencies, acceptance criteria, code ancestry, output contracts, and phase-completion gates. Nothing important should be inferred; it must be written explicitly.

The plan uses the following principle throughout. Whenever a theorem asserts an exact algebraic or operator-level statement, the first experiment validating that theorem must also be exact at the algebraic or operator level. Whenever a later experiment uses a learned architecture, that experiment must be labeled as an architecture-aligned validation or a robustness test, not as a direct proof-of-theorem surrogate. This distinction is critical for the thesis. Without it, the experimental chapter would blur together theorem validation, model misspecification, optimization effects, and architecture choice.

## 1. The governing principles that every phase must respect

The first governing principle is that the entire experimental program is split into five validation layers. The first layer is a frozen control-replication layer built from the reproduced Bordelon scripts. The second layer is the exact theorem-validation layer for A, B, and C. The third layer is the architecture-aligned validation layer where STU, S4, SSD / semiseparable / Mamba-2-style structured mixers, and explicit spectral-plus-attention hybrids are tested against the mechanisms already validated exactly. The fourth layer is the conditional empirical scaling-law layer where the exponents and constants of the hybrid mixture law are estimated, validated on held-out grids, and converted into compute-frontier predictions. The fifth layer is the robustness and natural-task layer where more expressive models such as selective Mamba or softmax-based hybrids are asked only to preserve the qualitative theorem phenomena, not to serve as the primary source of exponents.

The second governing principle is that the experimental implementation order is **not** the same as the theorem numbering. The theorem statements were developed logically in the order A \(\rightarrow\) B \(\rightarrow\) C \(\rightarrow\) conditional scaling law. The implementation order should instead be **B first, then C, then A, then architecture-aligned models, then the conditional scaling-law suite, then robustness and natural tasks**. The reason is pragmatic and mathematically justified. Theorem B forces the Fourier utilities, exact stationary generator, matched versus mismatched symbol logic, and modewise diagnostics to stabilize. Theorem C forces the band-RRS generator, refinement ladders, block commutant utilities, and heterogeneity diagnostics to stabilize. Theorem A then benefits from both, because its exact reduced-operator scripts need good masked-context utilities and good structured-operator comparison machinery, but they do not themselves harden the generator layer as broadly as B and C do. This execution order is therefore the safest engineering order, even though A remains the logical bridge theorem in the thesis text.

The third governing principle is that the thesis codebase is **real-valued by default**. Exact theorem-B finite-\(P\) circulant validation may use complex Fourier helpers internally if that materially simplifies exact diagonalization, but those helpers must be isolated inside `fourier_ops.py` or theorem-B exact scripts and must not become the global convention of the codebase. Theorem A and Theorem C are written in a real-valued, transpose-based setting with orthogonal group symmetries, and the architecture layer will use real-valued PyTorch models. Therefore the default generator path, the default metrics, and the default model interfaces must be real-valued. If an exact theorem-B helper mode is complex, it must convert outputs to real observables before any later layer consumes them.

The fourth governing principle is that the hybrid architecture used in the main scaling-law tier is fixed explicitly. The canonical scaling-law hybrid is **adaptive-first then spectral**: \(L_A\) basis-adaptive layers followed by \(L_S\) spectral layers. This choice is not arbitrary. It is the architecture whose decomposition best matches the theorem-C conditional hybrid proposition: the adaptive module estimates or refines basis/projector structure, after which the spectral backbone solves the now-more-stationary problem. Interleaved hybrids, parallel hybrids, and local-attention hybrids are useful robustness models, but they are not the primary architecture for the conditional scaling-law chapter. For the main scaling-law tier, the canonical adaptive module will be a **dense linear attention block** (or its exact linearized equivalent) so that the adaptive-side cost is explicit and easy to account for; the canonical spectral backbone will be a **linear FFT-based spectral filter with controllable spectral bottleneck \(r\)**, because that is the cleanest way to operationalize theorem-B width/rank bottlenecks and theorem-D cost accounting. Later architecture-aligned sections may substitute STU or S4 as spectral backbones and local/sliding-window attention as adaptive modules, but the main scaling-law chapter must have one canonical architecture whose cost model and decomposition are fixed unambiguously.

The fifth governing principle is that the conditional scaling-law chapter stays in the thesis, but it is not treated as an exact theorem on the same footing as A–C. The directory names and file names may still refer historically to theorem D if that matches the evolving codebase, but the chapter and experiments must be written and interpreted as an empirical or conditional scaling-law layer built on top of exact mechanisms. The theorem-level claims stop with A–C. The scaling-law chapter estimates exponents, tests approximate separability, and validates compute-frontier predictions out of sample.

## 2. Repository structure, naming, and output contracts

Before any new code is written, the repository must contain a thesis-specific namespace that does not pollute or overwrite the original Bordelon scripts. The correct directory structure is `scripts/thesis/`, `experiments/thesis/`, `configs/thesis/`, and `outputs/thesis/`. Inside `scripts/thesis/`, there must be subdirectories `utils/`, `theoremA/`, `theoremB/`, `theoremC/`, `architectures/`, and `scaling_laws/`. The directory should be called `scaling_laws/`, not `theoremD/`, to remain faithful to the mathematical status of the final chapter. The `experiments/thesis/` directory must contain one launcher script per experiment script. The `configs/thesis/` directory must contain one YAML config per experiment family. The `outputs/thesis/` directory must mirror the theorem or phase structure so that every run can be found, rerun, and frozen without ambiguity.

Every experiment script in phases III–VI of this document must produce the same minimal metadata contract. Each run must save the exact configuration as JSON or YAML, the list of seeds used, the git commit hash, the Python and CUDA environment description, the wall-clock runtime, the analytic compute proxy used by the script, the measured per-step wall-clock cost, a plain-text summary file with the key fitted quantities, and the run ID. The plan requires both raw arrays and human-readable summaries because a PhD thesis eventually needs to reconstruct results long after scripts were first run.

Every theorem script must save outputs in a standardized directory such as
```text
outputs/thesis/<phase_or_theorem>/<script_name>/<run_id>/
```
with subdirectories or files for `figures/`, `pdfs/`, `npz/`, `pt/`, `config.json`, `summary.txt`, and `metadata.json`. The launcher scripts must write logs into the same run directory. This is not bureaucracy; it is what makes the thesis reproducible months later when the manuscript is being finalized.

## 3. The pre-experiment mathematical audit

Before implementing any theorem-specific experiment, a short but non-negotiable mathematical audit must be completed. This audit must produce a single checked-in note or README describing the canonical conventions of the thesis experiment stack. The note must answer the following questions explicitly.

First, what is the canonical sample orientation? The theorem chapters often use \(X \in \mathbb{R}^{D\times P}\) with columns as samples. Some exact Fourier constructions are easier with row-sample matrices. The codebase must pick one canonical internal representation and define explicit conversion helpers if any module deviates. Silent shifts in sample orientation are prohibited. The recommended choice is to make **columns are samples** the canonical thesis convention and to isolate any row-sample exact Fourier helper behind named conversion functions.

Second, what is the default scalar field? The answer must be: **real-valued by default**, with complex helpers isolated to theorem-B exact mode only if needed. All later architecture-facing generators must be real-valued.

Third, how is label normalization defined? The audit note must state that the generator interface exposes a configuration switch `label_norm` with allowable values at least `sqrt_D` and `sqrt_P`, with theorem-B exact finite-\(P\) scripts intentionally using `sqrt_P` and theorem-A plus architecture-facing scripts defaulting to `sqrt_D`. Every run must record which normalization was used.

Fourth, how are the spectra parameterized? The note must state that a data generator may need a **training symbol**, a **test/query symbol**, and a **teacher/task spectrum**, and that these are distinct objects. This matters especially for theorem-B matched versus OOD symbol-shift experiments. No script is allowed to infer train/test mismatch from one overloaded spectrum variable.

Fifth, how is theorem-C heterogeneity parameterized? The note must state explicitly that the band-RRS heterogeneity parameterization is **mass-preserving inside each block**, so that changing a within-block condition number does not change the mean spectral mass of the block.

Sixth, what is the distinction between operator-level experiments and sampled-context experiments? The note must state that theorem-C exact validation is primarily operator-level and only secondarily sampled-context-based, because the grouped commutant and grouped-loss theorems are population/operator results.

The audit must be completed and signed off before any theorem-B or theorem-C script is allowed to proceed. This is one of the places where earlier plans were not detailed enough, and the thesis should not repeat that mistake.

## 4. The shared utility and generator layer

The generator and utility layer is not a convenience tier. It is a theorem-validation tier in disguise. If its contracts are wrong, later theorem scripts cannot be interpreted.

### 4.1 `run_metadata.py`

This file must be implemented first or alongside the first generator module. It must provide utilities to capture run IDs, commit hashes, environment fingerprints, seed lists, and wall-clock summaries. Every experiment script must call this utility at startup and write the metadata to disk before training begins, not only after completion.

### 4.2 `plotting.py`

This file must define the canonical style of the thesis figures. The Bordelon ancestry of many experiments means that several thesis figures will visually mirror Bordelon Figures 3, 4, 5, and 6. The plotting utility should therefore encode a consistent palette, linewidths, marker styles, font sizes, legend conventions, and helper functions for saving both PNG and PDF. It should also provide helpers for theory overlays, phase-diagram heatmaps, spectral mode trajectories, and compute-frontier comparisons. The purpose is not aesthetic alone; it is to ensure that the theorem-exact plots and the architecture/robustness plots are visually comparable.

### 4.3 `fit_powerlaws.py`

This file must centralize every power-law or separability fit performed in the thesis. It must implement log-log least squares with optional heteroskedastic weighting, bootstrap confidence intervals over seeds, fit-window enforcement, and held-out predictive evaluation. No experiment script should fit exponents on its own. Every exponent reported in the thesis must be reproducible through this one utility. The utility must also allow fixed-window fits specified a priori in the config file, because the thesis may not choose fitting windows after looking at plots.

### 4.4 `cost_models.py`

This file must implement the canonical compute model. Since the main scaling-law architecture is adaptive-first then spectral, with \(L_A\) dense linear-attention layers followed by \(L_S\) FFT-based spectral layers, the main analytic cost model must be
\[
C_{\mathrm{proxy}}
=
t\Bigl(
c_S\,\Phi_S(P,r)\,L_S
+
c_A\,\Phi_A(P)\,L_A
\Bigr),
\]
with
\[
\Phi_A(P)=P^2
\]
for the canonical dense linear-attention adaptive module, and
\[
\Phi_S(P,r)=P\log P + P r
\]
for the canonical FFT-based spectral filter with \(r\) active learnable modes or effective spectral bottleneck. If the spectral layer is implemented in a simpler truncated-Fourier way whose cost is effectively \(Pr\), the utility should support that as a secondary option, but the plan must record which proxy is used in each figure. This explicit cost model is required for theorem-D empirical frontiers and must not be left implicit. The same utility must also compute measured wall-clock cost so that figures can compare analytic proxies and measured costs directly.

### 4.5 `metrics.py`

This file must contain the canonical implementations of every theorem- and scaling-facing metric. These include, at minimum, the relative reduced-model error for theorem A, the full \((A,B)\)-operator perturbation bound for theorem A mask perturbation, Fourier transfer-function error and off-diagonal Fourier energy for theorem B, commutant violation and grouped-coordinate trajectory error for theorem C, projector-estimation error for learned hybrids, OOD degradation slopes, fit residuals and held-out prediction errors for the scaling-law chapter, and frontier regret for compute-allocation experiments. The theorem-A perturbation metric is especially important: experiment A2 must compare empirical error not only qualitatively but also against the perturbation upper bound involving both \(A_\theta-A_{\mathrm{GD}}\) and \(B_\theta-B_{\mathrm{GD}}\).

### 4.6 `partitions.py` and `commutants.py`

These files must encode all partition- and commutant-related logic needed for theorem C. `partitions.py` must provide dyadic ladders, equal-block divisor chains, custom refinement ladders, explicit block index sets, and heterogeneity constructors that preserve average block mass. `commutants.py` must provide projection into the block commutant, extraction of grouped block scalars \(q_b\), block-commutant violation metrics, and refinement containment checks. These files are not abstract utilities; they are the code-level realization of theorem-C operator classes.

### 4.7 `fourier_ops.py`

This file must contain the canonical unitary DFT, inverse transforms, real-even spectrum constructors, exact circulant diagonalization helpers, frequency permutation helpers, symbol interpolation constructors, and optional isolated complex exact helpers. If theorem-B exact scripts rely on complex diagonalization internally, the implementation must live here and nowhere else.

### 4.8 `data_generators.py`

This file must be rebuilt or revised to satisfy the exact mathematical needs of the thesis. Four generators are required.

The first generator, which we will call **GA**, is the theorem-A exact masked-context generator. It must create the train/query split, explicit GD-compatible masks, train-supported perturbations, aligned teacher/features, and all objects needed by theorem-A full forward and reduced-operator exact comparisons.

The second generator, **G1**, is the theorem-B exact stationary circulant generator. It must expose exact finite-\(P\) mode and population mode, must implement full-window query mode and single-query mode explicitly, must separate training symbol, test symbol, and teacher/task spectrum, and must return both the operator-level spectral objects and the sampled context data if requested.

The third generator, **G2**, is the theorem-C band-RRS generator. It must expose both operator-only and sampled-context modes. It must implement block-Haar rotations, mass-preserving within-block heterogeneity, and return all objects needed for both exact grouped operator-level experiments and architecture-facing sampled-context experiments.

The fourth generator, **G3**, is the refinement-ladder generator. It must produce canonical coarse-to-fine ladders and exact refinement checks.

The self-test for `data_generators.py` must validate the algebraic identities promised by the theorem-exact experiments. It must test the exact G1 covariance identity, G1 matched and mismatched symbol handling, real-valued default operation, G1 query-mode correctness, theorem-A mask generation, G2 block-Haar structure, operator-only G2 mode, commutant invariance, block-average preservation under heterogeneity sweeps, and G3 refinement validity. The current draft already has some strong self-tests, but the revised plan requires these additional ones before any theorem script may trust the generators.

## 5. The frozen Bordelon control suite

The first true experiment phase is not new experiments. It is the freezing of a minimal reproduced Bordelon control suite. The repository already contains the original script catalog, and those scripts have clear figure and output mappings. The thesis must preserve that provenance. The minimal frozen suite should consist of `run_isotropic_depth_vs_alpha.py`, `run_fixed_covariance.py`, `run_reduced_gamma_dynamics.py`, `run_compute_scaling_joint.py`, and `run_linear_attention_dynamics.py`, with `run_softmax_depth_sweep.py` optionally added as a sixth frozen control. These scripts cover the original paper’s ISO depth-vs-alpha behavior, fixed-covariance depth and OOD results, reduced-\(\Gamma\) dynamics, compute-scaling frontiers, and full-model-vs-reduced-theory overlays. They therefore span the main methodological and plotting families you will reuse.

These control runs must be rerun once with fixed seeds and fixed configs, then frozen. Their outputs should be placed under `outputs/thesis/controls/` and never overwritten. The thesis experiments chapter should contain a short subsection stating that the reproduced controls were frozen and used as the calibration package for all new theorem and scaling-law experiments.

## 6. Exact theorem-B experimental program

Although theorem A is the bridge theorem logically, theorem B should be implemented first. The theorem-B exact program validates the Fourier/circulant machinery, the matched versus mismatched symbol interfaces, and the earliest new thesis-specific figures. It also provides the first nontrivial theorem-exact validation that departs from the original Bordelon setting while still inheriting its Figure-3 logic.

### 6.1 Experiment B0: structure-closure diagnostic inherited from theorem A

This experiment exists to validate the theorem-A proposition that the reduced operator preserves structured classes under the reduction in the settings relevant for theorem B. Before any dynamics or training sweep is run, this script should construct theorem-A reduced operators \(A_S(X,\Gamma)\) in cases where the token-side mixer and kernel-side operator are both circulant or Toeplitz-like, and numerically measure how much of the resulting operator remains in the corresponding structured class. For the circulant exact mode, the off-circulant energy of \(A_S\) should be machine-precision small. For semiseparable or Toeplitz-like negative controls, the diagnostic may be approximate or qualitative.

The purpose of this experiment is not to produce a headline figure but to ensure that the theorem-A bridge and theorem-B stationary closure really talk about the same operators. This experiment should be implemented very early, alongside the theorem-B utilities, because it depends only on the reduced-operator code and the G1 exact generators. The original code ancestor is `run_linear_attention_dynamics.py` for full-vs-reduced comparison logic and `run_fixed_covariance.py` for per-eigenmode structure inspection.

### 6.2 Experiment B1: exact finite-\(P\) circulant mode closure

This is the first major theorem-B figure. It must test the exact finite-\(P\) circulant theorem in the full-window query setting. The training operator and query operator are both circulant; therefore both are diagonalized exactly by the same Fourier basis. The experiment should compare the empirical residual mode trajectories and query transfer function obtained from the reduced recursion with the exact theorem formulas for selected Fourier modes and for the full transfer spectrum.

The generator must be G1 exact mode, with `query_mode="full_window"` and `K=P`. The experiment must sweep several window lengths \(P\), several depths \(L\), and several spectral symbols, including at least a flat symbol, a power-law symbol, and a multiband symbol. The primary metrics are the relative error between empirical and theoretical residual mode trajectories, the transfer-function error, and the off-diagonal Fourier energy of the reduced operator. The acceptance criterion for the exact-mode scripts is machine precision up to numerical integration and floating-point tolerances. If those errors are not tiny, the generator or recursion implementation is wrong.

The primary figure should mirror the role of Bordelon Figure 3a, but in the stationary spectral language. It should show Fourier mode trajectories with exact theory overlays. A secondary figure should show the transfer-function spectrum itself. The code ancestry is `run_fixed_covariance.py` for per-mode/eigenvalue dynamics and `run_reduced_gamma_dynamics.py` for reduced-theory time-dynamics overlays.

### 6.3 Experiment B2: long-context depth irrelevance in the matched stationary regime

Once exact finite-\(P\) closure is validated, theorem B’s fixed-basis depth-unnecessary claim must be tested in its natural large-context stationary regime. This experiment should use the G1 population or long-context mode with matched training and test symbols. The point is to show that, once the spectral model can cache the correct inverse filter in the fixed basis, increasing spectral depth beyond a shallow baseline no longer improves the asymptotic matched stationary loss, even though depth can affect transient optimization behavior or the speed at which the optimum is approached.

The visual role of this figure is the direct spectral analogue of Bordelon Figure 3b. It should show loss-vs-time curves for several depths at large context and a final-loss-vs-depth panel showing collapse of the stationary matched asymptote. The code ancestry is `run_fixed_covariance.py` and `run_depth_scaling_nonrotate.py`.

### 6.4 Experiment B3: spectral OOD brittleness under symbol mismatch

Theorem B predicts fixed-basis OOD brittleness in the stationary spectral regime. The primary OOD experiment must therefore be **symbol-native**, not generic covariance rotation. This script should train in a matched stationary regime and then evaluate under at least two families of symbol shift: direct symbol interpolation and frequency permutation or reallocation. These are the cleanest spectral analogues of fixed-covariance train/test mismatch. Generic rotational mismatch may be added later as a bridge experiment toward theorem C, but it should not be the primary theorem-B OOD test.

The main figure should play the same conceptual role as Bordelon Figure 3c, but with the horizontal axis being a spectral shift parameter. The code ancestry is `run_fixed_covariance.py`, modified so that the mismatch family is expressed in terms of spectra rather than arbitrary rotations.

### 6.5 Experiment B4: spectral-rank bottleneck and joint spectral shape sweep

The theorem-B exact program should not stop at separate depth and width/rank sweeps. It must also test the spectral shape relation underlying the later scaling-law chapter. The first half of this experiment is the spectral-rank bottleneck law, where only the first \(r\) modes or an equivalent spectral bottleneck class are learnable and the loss floor is expected to follow a power law in \(r\). The second half is a **joint \(r\)–\(L_S\) sweep** designed to test the compute-optimal spectral-shape relation analogous to the Bordelon width-depth relation. This joint sweep is important because later theorem-D frontiers will treat spectral depth and spectral bottleneck as separate resources, and theorem B should already provide the clean spectral-only version of that relation.

The code ancestry comes from `run_compute_scaling_width.py` and `run_compute_scaling_joint.py`, but with the width bottleneck replaced by an explicit spectral truncation or spectral rank bottleneck. The main outputs are a spectral-rank floor figure, a loss-vs-depth at fixed rank figure, and a joint \(r\)–\(L_S\) grid whose best path is compared to the theorem-B spectral shape prediction.

### 6.6 Optional experiment B5: periodic LDS identification

This experiment is not required for the core theorem-B story, but it is the best optional extension because it tests the fixed-basis spectral closure in a naturally spectral task. It should only be implemented after B1–B4 succeed. The data generator for LDS is a separate optional generator; the model classes can reuse the same spectral backbones used in B1–B4. The point of this experiment is not to fit exponents but to show that the theorem-B phenomena are not artifacts of static regression.

## 7. Exact theorem-C experimental program

Theorem C is the empirical centerpiece of the thesis. The exact theorem-C program must therefore be both broader and more careful than any other theorem experiment. It must not begin immediately with trained architectures. It must begin with the exact population/operator objects and only then move to architecture-aligned models.

### 7.1 Experiment C1: band-RRS commutant closure at the operator level

This is the first theorem-C experiment and it is exact. Using G2 operator-only mode, optimize or evolve the effective sample-space operator \(Q(t)\) under the theorem-C population loss from symmetric initialization and verify that it remains in the block commutant associated with the chosen band partition. The primary metric is commutant violation measured after projecting into the Fourier basis and removing the block-scalar part. This error must remain at machine precision in the exact theorem-C operator code. If it does not, the theorem-C implementation is wrong or the generator is not producing the intended band-RRS ensemble.

This experiment should produce a grouped-operator heatmap figure and a commutant-violation trajectory. The purpose is to verify the structural heart of theorem C before any lower bound or phase diagram is drawn.

### 7.2 Experiment C2: grouped ODE and grouped-loss validation

Once commutant closure is established, the grouped scalar dynamics and grouped loss formula must be tested directly. This experiment should compare the empirical grouped block coordinates \(q_b(t)\) obtained from the operator-level trajectory with the grouped theorem-C ODE and with the exact grouped-loss evaluation. This is the theorem-C analogue of theorem-B’s modewise closure figure. The grouped trajectory error must be small in the exact operator-level code.

The main output is a blockwise trajectory comparison plot and, where appropriate, grouped-loss curves parameterized by \(q_b\). The code ancestry comes from `run_reduced_gamma_dynamics.py`, but now the scalar reduced variable is replaced by a grouped vector \(q\).

### 7.3 Experiment C3: exact \(L=1\) closed-form block lower bound

Theorem C has a particularly sharp special case at \(L=1\). This experiment must validate the closed-form optimum for the spectral-only commutant-restricted class at \(L=1\). The correct object being optimized is the theorem-C block-commutant loss, not a trained network. The experiment must compare the closed-form minimizer and closed-form loss to direct numerical optimization over the block-scalar parameters. The primary figure should show exact agreement and should also illustrate how the bound depends on within-block heterogeneity.

This figure is extremely valuable because it gives the first fully explicit lower-bound-style picture of the theorem-C obstruction. The code ancestry comes from `run_loss_vs_w.py` and `run_loss_landscape.py` in the sense of scalar or few-variable exact landscape plotting.

### 7.4 Experiment C4: heterogeneity phase diagram

Only after C1–C3 have validated the mechanism exactly should the theorem-C phase diagram be drawn. This is the main headline figure of the thesis experiments. It should sweep both the block size \(m\) and the within-block heterogeneity parameter \(\kappa\), while holding block average scale fixed through the normalized band-RRS generator. The key outputs are the spectral-only optimum in the coarse block-commutant class, the oracle refined optimum in a finer class, and the gap between them. This phase diagram is not meant to be architecture-specific initially. It is the theorem-level map of where hybrid refinement matters.

The figure should include at least two heatmaps: the spectral-only fixed-depth optimum and the oracle refinement gain. It is appropriate to add contour lines or slices at fixed \(m\) or fixed \(\kappa\). This figure should be produced from exact operator-level optimization, not from trained networks.

### 7.5 Experiment C5: refinement monotonicity

This experiment validates the exact theorem-C refinement monotonicity statement. It must generate a refinement ladder \(\mathfrak B^{(0)} \succeq \mathfrak B^{(1)} \succeq \dots \succeq \mathfrak B^{(J)}\), optimize over each corresponding commutant class, and verify that the optimum loss is monotone nonincreasing, with strict improvements when heterogeneous modes are separated into finer blocks. The main output is a loss-versus-refinement-level figure. This experiment gives the exact theorem-level interpretation of hybrid advantage before any learned adaptive module is introduced.

### 7.6 Experiment C6: oracle hybrid defined correctly

This is where the plan must be precise. In the theorem-exact tier, the term “oracle hybrid” means **direct optimization over the refined commutant class**, not a learned or estimated projector. That is the correct theorem-level interpretation. Therefore experiment C6 should compare the coarse-class optimum, the refined-class optimum, and possibly an attention-only unconstrained optimum if desired, but the “oracle hybrid” must refer specifically to the refined commutant optimum. Projector-estimation and learned basis adaptation belong later, in the architecture-aligned tier.

This clarification is essential because earlier drafts risked treating “oracle hybrid” as if it meant already estimated projectors. That belongs to a different layer.

### 7.7 Experiment C7: finite-depth scaling in the grouped band-RRS class

Theorem C does imply a meaningful depth dependence, but the correct finite-block overlay is not a generic blockwise power law in depth. For a fixed block with finite condition number, the natural theorem-C prediction is a geometric or contractive depth law controlled by the block condition number, for example through the bound involving
\[
\rho_b^\star = \frac{\kappa_b-1}{\kappa_b+1}.
\]
Therefore C7 should not default to fitting a generic \(L^{-\beta_b}\) power law inside each block. Instead, the primary overlay should be the exact grouped finite-\(L\) optimum or the condition-number contraction bound. Only in later asymptotic regimes where block spectra themselves scale with system size should one expect a power law. The exact theorem-C finite-depth experiment must respect this.

The output of C7 should therefore show how grouped spectral-only performance improves with depth in a manner governed by within-block heterogeneity and condition numbers, interpolating between theorem-B-like depth irrelevance in singleton blocks and slower grouped optimization in coarser heterogeneous blocks.

## 8. Exact theorem-A experimental program

Theorem A remains logically foundational, but its clean exact experiments can be implemented after the theorem-B and theorem-C infrastructure has stabilized.

### 8.1 Experiment A1: exact full structured model versus reduced \((A_S,B_S)\) recursion

This is the main theorem-A exactness experiment. It must construct exact theorem-A masked contexts using GA, build fully aligned linear structured mixers, and compare the full forward pass against the exact reduced \((A_S,B_S)\) recursion. In the GD-compatible case it must also compare against the reduced-\(\Gamma\) special case. The script should sweep over \(D\), \(P\), \(K\), and \(L\), and it should save the exact forward predictions and relative errors. The theorem-A exact acceptance criterion is machine precision or its floating-point equivalent.

The script should be based on `run_linear_attention_dynamics.py` because that script already overlays full-model and reduced-theory dynamics, but it must be simplified to deterministic theorem-A forward validation rather than optimization dynamics.

### 8.2 Experiment A2: perturbation away from GD-compatibility with the full reduced \((A,B)\)-operator bound

The theorem-A perturbation experiment must be stricter than a qualitative “it gets worse as the mask is perturbed” statement. It must interpolate between the exact GD-compatible mask and structured perturbations of that mask, then compare the empirical difference between the full model and the GD-compatible reduced-\(\Gamma\) predictor against the perturbative upper bound written at the level of **both** reduced operators \(A\) and \(B\). In other words, the experiment should compute not only the empirical prediction error but also the perturbation bound based on \(A_\theta-A_{\mathrm{GD}}\), \(B_\theta-B_{\mathrm{GD}}\), and the telescoping difference between the corresponding propagators \(T_\theta^\ell\) and \(T_{\mathrm{GD}}^\ell\). This is the faithful theorem-A perturbation diagnostic.

This experiment is valuable because it distinguishes exact theorem-A failure from controlled approximation. It should produce both a main error-vs-perturbation-strength curve and a theory-bound-vs-empirical-error comparison.

### 8.3 Experiment A3: exact semiseparable / SSD realization

This experiment should validate the explicit semiseparable or SSD realization proposition used in theorem A. The primary test is again deterministic and exact: an explicit semiseparable realization of the GD-compatible structured mask is compared against the reduced-\((S,\Gamma)\) theorem object. The experiment may include a more generic SSD parameterization as a secondary comparison, but the main exact experiment must use the explicit theorem-consistent realization, not a broad SSD family whose relation to the theorem may be indirect.

### 8.4 Experiment A4: negative controls outside the exact theorem class

Theorem A should also include at least one negative-control experiment. The right negative controls are a pure linear circulant convolutional mixer with no data-dependent bilinear key-query structure and a structured mixer with a clearly non-GD-compatible train/query mask. These are not theorem-A in-class models; they are deliberately outside the class. Therefore their role is to show that exact reduced-\(\Gamma\) behavior is not universal and that the theorem-A hypotheses matter. They should be labeled as negative controls, not as theorem-A failures.

## 9. Architecture-aligned validation layer

After the exact theorem programs for B, C, and A are stable, the thesis should move to architecture-aligned models. These models are still relatively simple and aligned to the theorem assumptions, but they are genuine trainable architectures rather than exact operator objects.

### 9.1 Spectral-only architecture suite

The primary spectral-only architecture for this suite should be the canonical linear FFT-based spectral filter already used in the exact and scaling-law tiers. Then, more realistic spectral-only models should be added: STU as the main realistic spectral-only model, and S4 or another LTI SSM surrogate as a secondary one. The tasks here should be theorem-B aligned: stationary circulant regression, spectral OOD shift, and spectral bottleneck sweeps. The purpose is to show that the theorem-B mechanisms survive in realistic spectral models, not to prove theorem B again.

The original Bordelon ancestor scripts are `run_depth_scaling_nonrotate.py`, `run_fixed_covariance.py`, and `run_compute_scaling_width.py`.

### 9.2 Structured-mask / SSD suite

The primary structured-mask architecture should be an explicit linear semiseparable or SSD realization consistent with theorem A. A Mamba-2-style SSD-inspired linearization may be added as a secondary model if available. The tasks here should be theorem-A exact bridge checks and theorem-C band-RRS grouped behavior. The purpose is to show that the theorem-A bridge and theorem-C commutant logic survive in a realizable structured family.

The original Bordelon ancestor script is `run_linear_attention_dynamics.py` for full-vs-reduced overlays.

### 9.3 Canonical hybrid architecture suite

This is the most important architecture-aligned block because it connects theorem C to the scaling-law chapter. The canonical hybrid architecture must be explicitly specified as:
\[
\underbrace{\text{adaptive module}}_{L_A\ \text{dense linear attention layers}}
\;\longrightarrow\;
\underbrace{\text{spectral backbone}}_{L_S\ \text{FFT-based spectral layers with bottleneck } r}.
\]
This architecture is the default for all main scaling-law and compute-frontier experiments. Alternatives such as interleaved hybrids or local-attention hybrids belong to the robustness tier unless otherwise stated. The experiment program here should compare spectral-only, oracle refined commutant optima, and learned adaptive-first spectral hybrids on band-RRS data. It should measure projector-estimation error, loss gap to oracle, and sensitivity to \(L_A\), \(L_S\), and \(r\).

The code ancestry comes primarily from `run_compute_scaling_joint.py` and `run_linear_attention_dynamics.py`, with architecture wrappers added.

## 10. Conditional empirical scaling-law program

The scaling-law chapter is a conditional empirical layer, not an exact theorem layer. The experiments must say so explicitly.

### 10.1 Separate exponent estimation

The spectral-side exponents \(\beta_S\) and \(\beta_r\) must be estimated first, before any joint fit, using the cleanest exact or architecture-aligned spectral models. The adaptive-side exponents \(\beta_A\) and \(\beta_P\) must be estimated next using the canonical adaptive-first spectral hybrid, with \(L_S\) and \(r\) set large enough that spectral error is not dominant. The optimization-time exponent \(\beta_t\) must then be estimated at sufficiently large architecture size so that optimization error is the main varying term. These estimates must be done in separate scripts or script modes and must use fixed fit windows specified in config files.

### 10.2 Additive separability grid

After the separate exponents are estimated, the thesis must test the approximate separability assumption on a factorial grid over \((L_S,L_A,r,P,t)\). This grid must be split into a fit subset and a validation subset. The additive law may be fit on the fit subset, but its error must be reported on held-out validation points. The key metrics are not just \(R^2\), but also median relative error, maximum relative error, and exponent stability across train/validation splits.

### 10.3 Compute-optimal frontier validation

Once the additive law is fit, the thesis must validate whether the predicted compute-optimal allocations actually work. This is the culminating empirical statement of the scaling-law chapter. The canonical compute model must be used:
\[
C=t\bigl(c_S(P\log P + Pr)L_S + c_A P^2 L_A\bigr)
\]
for the main adaptive-first dense-linear-attention plus FFT-spectral hybrid. The same experiments should also record measured wall-clock cost and compare analytic proxy and measured frontier ordering. For a grid of budgets \(C\), the thesis must compare the predicted optimal allocation to the actual best configuration found in a brute-force or dense sweep over the canonical architecture. The frontier regret and allocation mismatch should be reported. This is the empirical test that makes the scaling-law chapter genuinely useful.

### 10.4 Required distinction between oracle and learned hybrid in exponent fits

Every adaptive-side exponent fit and every frontier experiment involving \(L_A\) must include an oracle refinement reference. Otherwise a poor learned basis estimator can make a correct theorem-C mechanism look false. This is one of the strongest practical lessons of the theorem hierarchy, and the plan must treat it as mandatory.

## 11. Robustness and natural-task extension layer

After the exact theorem, architecture-aligned, and scaling-law tiers are complete, the thesis should run a robustness layer. Here the models are more expressive and less theorem-aligned. Full STU, full S4, selective Mamba, and a softmax-attention hybrid are appropriate here. Their purpose is to show that the phase distinctions from theorem B and theorem C survive qualitatively and that the conditional scaling-law decomposition continues to be useful as a design lens. These models must not be the primary source of fitted exponents.

An optional but valuable natural-task extension is in-context LDS identification. It should use the same theorem-B and theorem-C stationary and band-RRS generator logic as far as possible, but with linear dynamical systems replacing static regression. This should only be implemented if the core thesis figures are already stable. It is a bonus, not a prerequisite.

## 12. Statistical protocol and acceptance gates

Because this is a PhD thesis, the statistical and reproducibility protocol is part of the scientific claim. Exact reduced/operator-level experiments should use many seeds where randomness remains, but their primary acceptance criteria are algebraic, not statistical. Architecture-aligned experiments should use enough seeds to report standard errors, and the number of seeds must be recorded in every caption or summary table. Scaling-law fits must use bootstrap confidence intervals over seeds, not a single unqualified exponent estimate.

Every phase of the experiment program has a gate. The generator phase is complete only when all self-tests pass, including theorem-A masks, theorem-B query modes, theorem-C heterogeneity normalization, and theorem-C operator-only mode. The theorem-B exact phase is complete only when the exact finite-\(P\) circulant closure errors are at machine precision and the long-context stationary depth-unnecessary story is clearly visible. The theorem-C exact phase is complete only when commutant closure, grouped ODEs, \(L=1\) lower bounds, heterogeneity phase diagrams, and refinement monotonicity have all been validated at the operator level. The theorem-A exact phase is complete only when full-vs-reduced and GD-special-case comparisons are exact and the perturbation experiment matches the theorem-A perturbation bound qualitatively or quantitatively. The architecture-aligned phase is complete only when the canonical hybrid and at least one realistic spectral-only model show the predicted qualitative transitions. The scaling-law phase is complete only when separate exponent fits are stable and the held-out compute-frontier predictions show small regret. Robustness and natural-task experiments only begin after all prior gates are satisfied.

## 13. Implementation order

The actual order Claude should code is now fixed.

First, implement the repository scaffolding, metadata utilities, and plotting utilities.

Second, implement `fourier_ops.py`, `partitions.py`, `commutants.py`, `metrics.py`, and `cost_models.py`.

Third, revise and complete `data_generators.py` so that GA, G1, G2, and G3 exist, are documented, and all required self-tests pass.

Fourth, freeze the Bordelon control suite outputs.

Fifth, implement theorem-B exact scripts in the order B0, B1, B2, B3, B4, then optional B5.

Sixth, implement theorem-C exact scripts in the order C1, C2, C3, C4, C5, C6, C7.

Seventh, implement theorem-A exact scripts A1, A2, A3, A4.

Eighth, implement the architecture-aligned suites in the order spectral-only, structured-mask / SSD, canonical adaptive-first hybrid.

Ninth, implement the conditional scaling-law scripts for separate exponent estimation, additive-law validation, and compute-frontier prediction.

Tenth, implement robustness and natural-task extensions if time and compute remain.

This is the final implementation order. It combines the theorem-first discipline of the thesis with the engineering reality that theorem-B and theorem-C exact code stabilize the generator layer first.

## 14. Mapping to the original Bordelon code

This final plan must remain grounded in the original repository because Claude Code knows only that codebase and the thesis files you have developed here. The Bordelon experiment catalog documents the existing scripts, their outputs, and the figures they produce. The thesis scripts should therefore be understood as descendants of those originals, not as an independent codebase. `run_linear_attention_dynamics.py` is the direct ancestor for theorem-A full-vs-reduced overlays. `run_fixed_covariance.py` is the direct ancestor for theorem-B fixed-basis depth and OOD experiments. `run_reduced_gamma_dynamics.py` is the direct ancestor for reduced-operator dynamics and theorem-C grouped ODE plotting. `run_compute_scaling_width.py`, `run_compute_scaling_depth.py`, and `run_compute_scaling_joint.py` are the direct ancestors of the conditional scaling-law and compute-frontier scripts. `run_isotropic_depth_vs_alpha.py` is the direct ancestor of theorem-C two-dimensional heterogeneity phase-diagram plotting. `run_softmax_depth_sweep.py` is the direct ancestor of the robustness softmax-hybrid experiments. The experiment catalog and the original paper must be treated as methodological anchors throughout implementation.

## 15. Immediate next move

The immediate next move is not to write theorem-A or hybrid scripts. It is to revise the generator and utility layer until it satisfies the exact contracts written above. The first code target after this plan is therefore the generator revision and self-test completion. The first theorem script to implement after that is `run_theoremB_circulant_modes.py`, because it is the fastest, cleanest, and most revealing theorem-specific figure that depends directly on the generator layer and immediately tests whether the new thesis code really extends beyond the original Bordelon baselines in a mathematically controlled way.

This is the final merged plan. Any future deviation from it should be treated as a deliberate amendment and not as an implicit improvisation.
