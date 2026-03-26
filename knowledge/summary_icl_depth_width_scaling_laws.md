# Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time

**Paper:** Bordelon, Letey, Pehlevan (Harvard) -- arXiv 2510.01098 (ICLR 2026)

---

## 1. Problem Statement and Motivation

The paper addresses a fundamental open question: **how should width and depth of a transformer scale under a compute budget, and do scaling laws depend on width and depth separately (not just total parameters)?**

Prior scaling law theories (Kaplan, Chinchilla, etc.) treat model size as a single number. Infinite-width (muP) and infinite-depth scaling limits exist but no theory captures the **relative gains from width vs. depth at fixed compute**. This paper provides the first solvable model where width $N$ and depth $L$ enter the scaling law as separate terms.

---

## 2. Architecture: Deep Linear Self-Attention

### 2.1 Full Model

Data matrix $D$ contains $P$ labeled examples $(x_\mu, y_\mu)$ and $K$ masked evaluation points $(x_\mu^\star, *)$. The depth-$L$ residual linear attention model:

$$h^1_\mu = W_x x_\mu + w_y y_\mu$$

$$h^{\ell+1}_\mu = h^\ell_\mu + \frac{1}{LP} \sum_{\nu=1}^P (k^\ell_\nu \cdot q^\ell_\mu) v^\ell_\nu, \quad \ell \in [L]$$

$$f_\mu = w_o \cdot h^L_\mu$$

where $q = W_q h$, $k = W_k h$, $v = W_v h$. Loss: $\mathcal{L}(D) = \frac{1}{K}\sum_{\mu=P+1}^{P+K}(f_\mu - y_\mu)^2$.

**Critical:** This is **linear attention** (dot product $q \cdot k$, not softmax). The $1/L$ factor in the residual connection is the depth-wise scaling that ensures a stable infinite-depth limit.

### 2.2 Reduced-Gamma Model

Under alignment assumptions ($W_x^\top w_y = 0$, $W_v \propto w_y w_y^\top$), the model reduces to a single matrix $\Gamma \in \mathbb{R}^{D \times D}$:

$$\Gamma \equiv (w_o^\top W_v w_y) W_x^\top W_k^\top W_q W_x$$

$$f(x_\star) = \frac{1}{LP} x_\star^\top \Gamma \sum_{\ell=0}^{L-1} (I - L^{-1} \hat\Sigma \Gamma)^\ell X^\top y$$

where $\hat\Sigma = \frac{1}{P} X X^\top$. This is precisely **$L$ steps of preconditioned gradient descent** on the in-context linear regression problem with preconditioner $\Gamma$ and step size $1/L$.

For looped/universal transformers ($W_i^\ell = W_i^{\ell'}$ for all layers), a single $\Gamma$ suffices.

### 2.3 Equivalence to In-Context Gradient Descent

The residual stream variable $\Delta_\mu^\ell = w_o \cdot h^\ell_\mu$ satisfies:
- **Training points:** $\Delta^{\ell+1}_\mu = \Delta^\ell_\mu - \frac{1}{LP}\sum_\nu x_\nu^\top \Gamma x_\mu \Delta^\ell_\nu$ (gradient descent on residuals)
- **Test points:** $\Delta^{\ell+1}_\mu = \Delta^\ell_\mu + \frac{1}{LP}\sum_\nu x_\nu^\top \Gamma x_\mu \Delta^\ell_\nu$ (accumulate predictions)

The sign flip comes from the positional masking matrix $M$ which has $-1_{P}1_P^\top$ in the train-train block and $+1_K 1_P^\top$ in the test-train block.

---

## 3. Three ICL Data Settings

### 3.1 Isotropic Covariates and Tasks (ISO)

$$x_{\mu,c} \sim \mathcal{N}(0, I), \quad \beta_c \sim \mathcal{N}(0, I), \quad y_{\mu,c} = \frac{1}{\sqrt{D}}\beta_c \cdot x_\mu + \sigma\epsilon$$

**Proportional asymptotics:** $P, K, B, D \to \infty$ with $P/D = \alpha$, $K/D = \kappa$, $B/D = \tau$.

**Key insight:** Only $\mathcal{O}(D^2)$ total tokens needed to converge (vs. $\mathcal{O}(D^3)$ in prior work by Lu et al.), because $K = \Theta(D)$ evaluation points per context provide multiple error signals.

**Result 1 (Gradient Flow, ISO):** From $\Gamma = 0$, gradient flow maintains $\Gamma(t) = \gamma(t) I$ where:

$$\frac{d\gamma}{dt} = \int d\lambda\, \rho(\lambda)\, \lambda\, (1 - L^{-1}\gamma\lambda)^{2L-1}$$

with $\rho(\lambda)$ the Marchenko-Pastur density. The final loss equals the optimal step-size MSE for $L$ steps of GD:

$$\mathcal{L}_\star(\alpha) = \min_\gamma \int \rho(\lambda)(1 - L^{-1}\gamma\lambda)^{2L} \, d\lambda$$

- $L = 1$: $\mathcal{L}_\star = (1+\alpha)^{-2}$
- $L \to \infty$: $\mathcal{L}_\star = [1-\alpha]_+$

**Result 2: Depth is unnecessary for long contexts.** If $\alpha = P/D \to \infty$, then $L=1$ achieves the minimal loss. Any $L \geq 2$ achieves the same (zero if $\sigma^2=0$) loss.

### 3.2 Fixed Structured Covariance (FS)

Population covariance $\langle xx^\top \rangle = \Sigma$ (arbitrary) and task covariance $\langle \beta\beta^\top \rangle = \Omega$.

**Result 3: Depth still unnecessary for long contexts.** For $\alpha \to \infty$, setting $\Gamma = L\Sigma^{-1}$ achieves zero loss at any depth. The model memorizes the whitening transform.

**Result 4: Decoupled eigenvalue dynamics.** When $\Omega$ and $\Sigma$ commute (eigenvalues $\omega_k$, $\lambda_k$):

$$\frac{d\gamma_k}{dt} = \omega_k \lambda_k^2 (1 - L^{-1}\lambda_k\gamma_k)^{2L-1}$$

At $L \to \infty$: $\gamma_k(t) = \frac{1}{2\lambda_k}\ln(1 + 4\omega_k\lambda_k^3 t)$, giving:

$$\mathcal{L}(t) = \sum_k \frac{\omega_k\lambda_k}{1 + 4\omega_k\lambda_k^3 t}$$

Under power-law source/capacity ($\lambda_k \sim k^{-\nu}$, $\sum_{\ell>k}\lambda_\ell\omega_\ell \sim k^{-\nu\beta}$):

$$\mathcal{L}(t) \sim t^{-\beta/(\nu + \nu\beta + 1)}$$

**Result 5: Brittleness to distribution shift.** The FS solution is specialized to the training covariance. OOD loss under $\Sigma \to \Sigma' = e^{\theta S}\Sigma e^{-\theta S}$:

$$\mathcal{L}_\text{OOD} = \text{tr}\, \Omega' [(I - \Sigma^{-1}\Sigma')^L]^\top \Sigma' (I - \Sigma^{-1}\Sigma')^L$$

This **grows** with depth $L$ for $\Sigma' \neq \Sigma$, making deep FS models fragile.

### 3.3 Random Rotated and Structured Covariances (RRS) -- THE KEY SETTING

Each context $c$ has randomly rotated covariance:

$$x_c^\mu \sim \mathcal{N}(0, \Sigma_c), \quad \Sigma_c = O_c \Lambda O_c^\top, \quad \Omega_c = O_c \Omega O_c^\top$$

where $O_c \sim \text{Haar}(O(D))$.

**Result 6: Gradient flow generates isotropic $\Gamma$.** From zero init, $\Gamma(t) = \gamma(t) I$ with:

$$\frac{d\gamma}{dt} = \text{tr}\, \Lambda^2 \Omega (I - L^{-1}\gamma\Lambda)^{2L-1}$$

The random rotation forces the model to learn **unconditioned in-context GD** (not a specialized preconditioner). This is why depth becomes essential: the model cannot memorize a whitening transform since the covariance changes every context.

---

## 4. Compute-Optimal Neural Scaling Laws (Main Result)

### 4.1 Width Bottleneck via Projection

Introduce width through a projection $A \in \mathbb{R}^{N \times D}$. The model operates on $\tilde{x} = Ax \in \mathbb{R}^N$, giving $\Gamma(t) = \gamma(t)(AA^\top)$.

### 4.2 Two-Point Deterministic Equivalent (DMFT)

The loss averaged over the Haar-random $O$ is computed via a dynamical mean field theory (DMFT) approach. The key technical result expresses the loss as:

$$\mathcal{L}(t,N,L,P) = \int \frac{d\omega\,d\omega'}{(2\pi)^2} (1 + L^{-1}\gamma i\omega)^L (1 + L^{-1}\gamma i\omega')^L \mathcal{C}(\omega,\omega')$$

where $\mathcal{C}(\omega,\omega')$ involves deterministic functions $\Psi_{v\chi}(\omega)$ and $\Psi_{\chi\chi}(\omega,\omega')$ determined by the spectra of $\hat\Sigma$ and $A^\top A$. These are computed from a saddle-point equation:

$$i\omega_{(A^\top A)^2}(\tau) \cdot i\omega_{\hat\Sigma}(\tau) = \frac{\tau - 1}{\tau} i\omega$$

using the $\tau$-transform $\tau_M(i\omega) = \text{tr}\, M(i\omega + M)^{-1}$.

### 4.3 The Scaling Law (Result 8)

Under source and capacity conditions for eigenvalues:

$$\sum_{\ell > k} \lambda_\ell \omega_k \sim k^{-\nu\beta}, \quad \lambda_k \sim k^{-\nu}$$

**The loss follows a separable Chinchilla-type scaling law:**

$$\boxed{\mathcal{L}(t, N, L, P) \approx c_t\, t^{-\beta/(2+\beta)} + c_N\, N^{-\nu\beta} + c_L\, L^{-\beta} + c_P\, P^{-\nu\beta}}$$

Each resource (time $t$, width $N$, depth $L$, context length $P$) contributes an **independent bottleneck** with its own power-law exponent.

### 4.4 Derivation of Individual Scaling Exponents

**Time scaling** ($N, L, P \to \infty$):
$$\frac{d\gamma}{dt} \approx \beta\gamma^{-\beta-1} \implies \gamma(t) \sim t^{1/(\beta+2)} \implies \mathcal{L} \sim t^{-\beta/(2+\beta)}$$

**Depth scaling** ($t, N, P \to \infty$):
Optimal $\gamma_\star \approx L/\lambda_1$, then:
$$\mathcal{L} \approx \sum_k \lambda_k(\beta_k^\star)^2 (1 - \lambda_k/\lambda_1)^L \approx \int dk\, k^{-\nu\beta-1} e^{-Lk^{-\nu}} \approx L^{-\beta}$$

**Width scaling** ($t, L, P \to \infty$):
Rank-$N$ projection only learns top $N$ eigendirections:
$$\mathcal{L} \sim \sum_k \frac{k^{-\nu\beta-1}}{1 + k^{-\nu}N^\nu} \approx N^{-\nu\beta}$$

**Context scaling** ($t, N, L \to \infty$):
Rank-$P$ empirical covariance limits learning similarly: $\mathcal{L} \sim P^{-\nu\beta}$.

### 4.5 Compute-Optimal Shapes

At fixed compute $C = t P^2 N^2 L$, the optimal width-depth ratio is:

$$\boxed{L \propto N^\nu}$$

where $\nu$ is the spectral decay exponent ($\lambda_k \sim k^{-\nu}$). For harder problems (faster eigenvalue decay), depth should scale faster relative to width.

---

## 5. Extensions to More Realistic Models

### 5.1 Untied Layers

With separate $\Gamma^\ell$ per layer (upscaled learning rate $\eta = \eta_0 L$), the loss is permutation symmetric in $\{\gamma^\ell\}$. Gradient flow from zero init maintains balance $\gamma^\ell = \gamma$ for all $\ell$, yielding **identical dynamics** to the recurrent model.

### 5.2 Full Linear Attention (Result 9)

Training separate $\{W_x, W_k, W_q, W_v\}$ with $w_y = w_o$ fixed at unit norm. From small balanced init $|W_i| = \sigma$, the dynamics maintain balance with a single scalar $w(t)$ evolving as:

$$\frac{dw}{dt} = w^4 \text{tr}\, \Lambda^2\Omega (I - w^5\Lambda)^{2L-1}$$

This is gradient flow on $\mathcal{L}$ with reparameterization $\gamma \to w^5$, giving a **modified time exponent**:

$$\mathcal{L}(t, L) \sim c_t\, t^{-5\beta/(5\beta+2)} + c_L\, L^{-\beta}$$

The depth exponent $L^{-\beta}$ is **unchanged** by reparameterization. The time exponent changes from $\beta/(2+\beta)$ to $5\beta/(5\beta+2)$ due to the 5th-power relationship $\gamma = w^5$.

### 5.3 Softmax Attention

Experiments with softmax attention trained with Adam show qualitatively similar depth-dependent scaling, consistent with the linear attention theory. Also tested with MLP blocks on the residual stream.

---

## 6. Technical Machinery: Two-Point Deterministic Equivalent

The DMFT computation is the paper's main technical innovation. Starting from a linear dynamical system $\partial_t v = -Mv + \delta(t)v_0$ with $M = OBO^\top A$ (free product), they:

1. Write the partition function $Z$ as a path integral over auxiliary fields $v(\omega), \chi(\omega)$ in Fourier space
2. Average over the Haar-random $O$ to obtain an effective action $\mathcal{S}[\Sigma, \Psi]$
3. Take the saddle point at $N \to \infty$ to get self-consistent equations for order parameters
4. The off-diagonal blocks decouple per frequency and satisfy: $i\omega_A(\tau) \cdot i\omega_B(\tau) = \frac{1-\tau}{\tau} i\omega$
5. The diagonal blocks (especially $\Psi_{\chi\chi}$) capture the two-point correlation needed for the loss

The deterministic equivalent is:
$$v(\omega) v(\omega')^\top \simeq (i\omega + \Psi_{v\chi}(\omega)A)^{-1}[v_0 v_0^\top - \Psi_{\chi\chi}(\omega,\omega')I](i\omega' + \Psi_{v\chi}(\omega')A)^{-1}$$

---

## 7. Connections to the spectral_scaling Codebase

This paper is essentially the theoretical foundation that the codebase is built around. Here are the explicit correspondences:

### 7.1 Data Generation Modes

| Paper Setting | Codebase `SampleMode` | Function |
|---|---|---|
| **ISO** | `"iid"` | `sample_data()` |
| **FS** | `"spec"` | `sample_data_spec()` |
| **RRS** | `"spec_rotate"` | `sample_data_spec_rotate()` |
| (variant) | `"gauss_rotate"` | `sample_data_gauss_rotate()` |

The codebase's `"spec_rotate"` mode exactly implements the RRS setting: each batch draws a random QR-rotation $O_b$ from a Gaussian matrix, rotates the canonical eigenvectors, and generates data with the rotated covariance.

### 7.2 Power-Law Spectrum Construction

The paper's $\lambda_k \sim k^{-\nu}$ and source condition $\omega_k\lambda_k \sim k^{-\nu\beta-1}$ maps to the codebase's `make_powerlaw_problem()` which constructs:
- `spec_k = k^{-alpha}` (the paper's $\nu$ is the codebase's `alpha`)
- `w_star_k = k^{-(alpha*beta + 1 - alpha)/2} / sqrt(spec_k)`, normalized so $\sum w_k^2 \lambda_k = 1$

### 7.3 Models

| Paper Concept | Codebase Implementation |
|---|---|
| Reduced-$\Gamma$ model | `reduced_gamma_dynamics.py` -- trains a $D \times D$ matrix $\Gamma$ directly |
| Hand-coded optimal weights | `linear_icl_dynamics.py: init_hand_coded_params()` |
| Full linear attention training | `train_model()` with coupled/decoupled modes |
| Width bottleneck $A \in \mathbb{R}^{N \times D}$ | `make_powerlaw_problem(d, alpha, beta, n_multiplier)` sets $N = \lfloor n_\text{mult} \cdot d \rfloor$ |
| Depth-scaled residual $\beta/L$ | `model_eval()` uses `beta_model / L` scaling |

### 7.4 DMFT Theory

The codebase's `isotropic_dmft()` function solves the fixed-point equations for the ISO setting:
- Constructs $\theta = \gamma \cdot L_\text{strict}$ (strict lower triangular matrix of ones)
- Iteratively solves $H = (I + \theta \cdot (I + H\theta/\alpha)^{-1})^{-1}$
- Returns the theoretical prediction curve $v_s[t]$ for comparison with trained models

### 7.5 Key Experiments

| Paper Figure | Codebase Script |
|---|---|
| Fig 1: ISO dynamics & depth vs alpha | `run_isotropic_depth_vs_alpha.py` |
| Fig 2: FS eigenvalue evolution & OOD | `run_ood_covariance_generalization.py` |
| Fig 3: Power-law time/depth scaling | `run_pretrain_icl_powerlaw.py`, `run_powerlaw_depth_sweep.py` |
| Fig 4: Width & depth compute scaling | (joint N,L sweep -- the scaling law experiments) |
| Fig 5: Full attention dynamics | `run_pretrain_icl_powerlaw.py` with decoupled training |

### 7.6 Predicted Power-Law Exponents

The codebase plots theoretical reference lines. The paper predicts:
- **Reduced-$\Gamma$ model:** $\mathcal{L} \sim t^{-\beta/(2+\beta)}$ (time), $L^{-\beta}$ (depth)
- **Full attention (5 parameters):** $\mathcal{L} \sim t^{-5\beta/(5\beta+2)}$ (time), $L^{-\beta}$ (depth)
- **Width bottleneck:** $\mathcal{L} \sim N^{-\nu\beta}$
- **Context bottleneck:** $\mathcal{L} \sim P^{-\nu\beta}$

The codebase's `run_pretrain_icl_powerlaw.py` plots reference lines for both $t^{-\beta}$ and $t^{-7\beta/(2+7\beta)}$ (the 7 comes from 7 trainable scalar degrees of freedom in the coupled weight parameterization, analogous to the 5 in the decoupled case).

---

## 8. Key Takeaways for Scaling Law Research

1. **Depth and width serve fundamentally different functions.** Width ($N$) controls how many spectral directions can be represented; depth ($L$) controls how many steps of in-context GD are executed. They bottleneck performance independently.

2. **Task diversity drives the need for depth.** With fixed covariance (FS), a shallow model memorizes the whitening transform and depth is unnecessary. With diverse covariances (RRS), the model must learn a generic algorithm (unconditioned GD), and depth directly controls how many steps of that algorithm are executed.

3. **The optimal shape $L \propto N^\nu$ depends on spectral decay.** For steeper eigenvalue decay (larger $\nu$), depth should scale faster relative to width. This provides a principled answer to "how should we shape the network?"

4. **The separable scaling law** $\mathcal{L} = c_t t^{-\beta_t} + c_N N^{-\beta_N} + c_L L^{-\beta_L} + c_P P^{-\beta_P}$ with all exponents determined by the source/capacity parameters $(\beta, \nu)$ is a concrete, testable prediction extending Chinchilla laws to width-depth decomposition.

5. **Reparameterization changes time exponents but not resource exponents.** Going from $\Gamma$ to $w^5$ changes the time scaling from $\beta/(2+\beta)$ to $5\beta/(5\beta+2)$ but depth scaling remains $L^{-\beta}$ and width scaling remains $N^{-\nu\beta}$.

6. **The two-point deterministic equivalent** is the technical workhorse. It extends one-point (spectral) random matrix theory to correlations between two resolvents, needed because the loss involves $|(\text{polynomial in } M) \beta|^2$, a quadratic form in the random matrix $M = O(A^\top A)^2 O^\top \hat\Sigma$.
