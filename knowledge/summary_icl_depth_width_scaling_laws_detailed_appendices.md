# Theory of Scaling Laws for In-Context Regression -- Detailed Appendix-by-Appendix Analysis

**Paper:** Bordelon, Letey, Pehlevan (Harvard) -- arXiv 2510.01098 (ICLR 2026)

This document supplements `summary_icl_depth_width_scaling_laws.md` with a proof-by-proof walkthrough of every appendix.

---

## Full Technical Summary

### Setup and Motivation

The paper asks: *when does depth help a transformer, and what is the compute-optimal shape (width vs. depth)?* Existing scaling law theory almost exclusively addressed width and training time; depth was largely ignored theoretically. This paper fills that gap using a solvable toy model.

The vehicle is **in-context learning (ICL) of linear regression** with **deep linear self-attention**. Linear attention (no softmax) is essential for tractability; the paper verifies qualitatively similar behavior holds for softmax + Adam in Section 5.2.

---

### Architecture (Section 2)

A depth-L linear attention model with residual stream:

h^1_mu = W_x x_mu + w_y y_mu,    h^{l+1}_mu = h^l_mu + (1/LP) sum_{nu=1}^P M_{mu,nu} (k^l_nu . q^l_mu) v^l_nu

with q^l = W_q h^l, k^l = W_k h^l, v^l = W_v h^l, and f_mu = w_o . h^L_mu. The 1/LP normalization is deliberate -- it makes the residual contributions O(1) total regardless of depth, analogous to a depth-scaled learning rate.

The context D contains P labeled pairs and K evaluation points. The positional masking matrix M in R^{(P+K)x(P+K)} has the block structure:

    M = [ -1_P 1_P^T   0  ]
        [  1_K 1_P^T   0  ]

which gives opposite signs on training vs. test tokens, enabling the model to implement gradient descent residuals.

**The reduced-Gamma model.** Under alignment conditions W_x^T w_y = 0, W_v proportional to w_y w_y^T, and the tied-layer ansatz W_j^l = W_j for all l, the entire model collapses to a single DxD matrix:

    Gamma = (w_o^T W_v w_y) W_x^T W_k^T W_q W_x

with predictor

    f(x*) = (1/LP) x*^T Gamma sum_{l=0}^{L-1} (I - L^{-1} Sigma_hat Gamma)^l X^T y

This is precisely **L steps of preconditioned in-context gradient descent** on the least-squares loss, with preconditioner Gamma and step size 1/L.

---

### Setting 1: Isotropic Covariates (ISO) -- Section 3.1 + Appendix C

x_{mu,c} ~ N(0, I), beta_c ~ N(0, I), y_{mu,c} = beta_c . x_{mu,c}/sqrt(D) + sigma epsilon.

Proportional asymptotics: P, D -> infinity with P/D = alpha fixed.

**Result 1.** Gradient flow from zero initialization preserves isotropy: Gamma(t) = gamma(t) I, with scalar dynamics

    d gamma/dt = -d/d gamma L(gamma, alpha)
    L(gamma, alpha) = integral d lambda rho(lambda, alpha) [(1 - L^{-1} gamma lambda)^{2L} + (sigma^2 / (alpha lambda)) (1 - (1 - L^{-1} gamma lambda)^L)^2]

where rho(lambda, alpha) is the Marchenko-Pastur law. The optimized loss L*(alpha) = min_gamma L(gamma, alpha) equals the loss for L steps of GD on the in-context problem with optimal step size:

- L=1: L* = (1+alpha)^{-2} (at sigma^2=0)
- L->infinity: L* = [1-alpha]_+, which is 0 when alpha >= 1

**Result 2.** As alpha -> infinity, L=1 achieves zero loss. Depth is irrelevant for long contexts in the ISO setting.

**SGD analysis (Appendix C.1).** For finite batch size B = tau D, the shallow model's SGD noise creates a residual floor C(infinity) = b/(1-a) that vanishes as eta/tau -> 0. Crucially, only O(D^2) total tokens are needed for convergence (vs. O(D^3) in prior work), because the paper scales K = kappa D masked points per context rather than K=1.

---

### Setting 2: Fixed Structured Covariance (FS) -- Section 3.2 + Appendix D

Now <xx^T> = Sigma (fixed across all contexts), <beta beta^T> = Omega.

**Result 3.** At alpha -> infinity, zero loss is achieved by Gamma = L Sigma^{-1} regardless of depth. Depth L=1 is already sufficient.

**Result 4.** When Omega and Sigma are codiagonalizable (eigenvalues omega_k, lambda_k), gradient flow decouples per eigenmode:

    d gamma_k / dt = omega_k lambda_k^2 (1 - L^{-1} lambda_k gamma_k)^{2L-1}

At L->infinity this solves exactly as gamma_k(t) = (1/(2 lambda_k)) ln(1 + 4 omega_k lambda_k^3 t), giving:

    L(t) = sum_k omega_k lambda_k / (1 + 4 omega_k lambda_k^3 t)

Under power laws lambda_k ~ k^{-nu}, omega_k lambda_k ~ k^{-nu*beta - 1}: L(t) ~ t^{-beta/(nu + nu*beta + 1)}.

**Result 5 (Brittleness).** The solution learned from FS training is specialized to Sigma. The OOD loss when tested on Sigma', Omega' is:

    L_OOD = tr Omega' [(I - Sigma^{-1} Sigma')^L]^T Sigma' (I - Sigma^{-1} Sigma')^L

This grows as Sigma' deviates from Sigma and is **not reduced** by increasing L.

---

### Setting 3: Randomly Rotated Structured Covariance (RRS) -- Section 3.3 + Appendix E

Each context c has Sigma_c = O_c Lambda O_c^T, Omega_c = O_c Omega O_c^T, with O_c drawn from the Haar measure.

**Result 6.** By rotational symmetry, gradient flow from zero initialization maintains Gamma(t) = gamma(t) I:

    d gamma/dt = tr(Lambda^2 Omega (I - L^{-1} gamma Lambda)^{2L-1})

This is plain (unpreconditioned) in-context GD. The critical consequence: even at alpha -> infinity, there is a nontrivial loss because the isotropic Gamma = gamma I cannot exploit any eigenstructure. More depth genuinely reduces this loss.

---

## Appendix A: Two-Point Deterministic Equivalent (DMFT)

### Goal

Compute the averaged correlation function

    C(omega, omega') = <[i omega + M]^{-1} Omega [i omega' + M^T]^{-1}>_O,    M = O B O^T A

where O is Haar-random orthogonal, and A, B have known spectra. This is needed because the loss at any depth L factors through C via the Cauchy integral representation.

### Why This is Hard

If M were symmetric, e^{-Mt} e^{-M^T t} = e^{-2Mt} and a single resolvent would suffice. But M = O B O^T A is generically **asymmetric** (since A and B need not commute), so left and right resolvents evaluated at different frequencies omega != omega' are coupled. This requires the full two-point DMFT machinery.

### Step 1: Cauchy Representation of the Depth-L Loss

The paper first motivates why C(omega, omega') is what you need. For finite depth L, using the Cauchy integral formula to represent the polynomial (I - gamma L^{-1} M)^L:

    <(v^L)(v^L)^T> = integral (d omega d omega' / (2 pi)^2) (1 + gamma L^{-1} i omega)^L (1 + gamma L^{-1} i omega')^L C(omega, omega')

where v^L = (I - gamma L^{-1} M)^L beta* and the contour encloses the positive imaginary axis.

At infinite depth L -> infinity, the polynomial (1 - gamma L^{-1} i omega)^L -> e^{-gamma i omega}, so the two-point function computes the loss via a Fourier-Laplace convolution.

### Step 2: Path Integral for the Free Product

For M = O B O^T A. The key tool is the **Martin-Siggia-Rose (MSR) path integral**: one writes Z = 1 by inserting delta functions enforcing the ODE:

    Z = integral Dv D chi exp(-integral dt chi(t) . [d_t v + M v - v_0 delta(t)]) = 1

Here chi(t) are the **response fields** (conjugate to v(t)), and the integration measure D chi is over the imaginary axis. Moving to Fourier space, this becomes an integral over pairs (v(omega), chi(omega)) for all frequencies omega.

**Averaging over O:** The Haar average over orthogonal matrices O acts only on the coupling chi^T M v = chi^T O B O^T A v. The result is that the average over O can be expressed as a **saddle-point action** over order parameters:

    <Z>_O = integral D Sigma D Psi exp(-N S[Sigma, Psi])

### Step 3: The Action and Its Block Structure

The action S[Sigma, Psi] has two contributions:

1. A "bulk" term -Tr Sigma Psi - (1/N) ln Z_A(Psi) involving a single-site partition function Z_A
2. A "disorder" term from the average over B

The order parameters Sigma and Psi are 2x2 block matrices (in the v/chi sector) indexed by frequency pairs (omega, omega'):

    Sigma = [Sigma_vv    Sigma_v_chi]     Psi = [0           Psi_v_chi  ]
            [Sigma_v_chi  0          ]           [Psi_v_chi   Psi_chi_chi]

The zero blocks are enforced by causality.

### Step 4: Saddle-Point Equations

Taking dS/dPsi = 0 and dS/dSigma = 0 gives self-consistency equations: Sigma is determined by averaging over the Gaussian measure Z_A which itself depends on Psi, which depends on Sigma.

### Step 5: Solving the Off-Diagonal Blocks (Response Function)

The off-diagonal blocks Sigma_{v chi}(omega, omega') = delta(omega - omega') Sigma_{v chi}(omega) decouple across frequencies. Introducing the tau-transform tau_M(i omega) = Tr M(i omega + M)^{-1}, the saddle point gives:

    tau = Sigma_{v chi} Psi_{v chi} = tau_A(i omega_A) = tau_B(i omega_B)

with i omega_A = i omega / Psi_{v chi} and i omega_B = (tau^{-1} - 1) Psi_{v chi}. Substituting:

    i omega_A(tau) . i omega_B(tau) = ((1 - tau) / tau) . i omega    ... (*)

This is the **free product tau-transform equation** -- the key equation to solve for tau(omega). Once tau(omega) is found, Psi_{v chi}(omega) = i omega / i omega_A(tau(omega)).

**Intuition:** The tau-transform of a matrix M is essentially its Stieltjes transform (up to a change of variables). The free product rule (*) is the analogue of the free multiplication rule for Stieltjes transforms.

### Step 6: Solving the Diagonal Blocks (Correlation Function)

The diagonal blocks Sigma_{vv}(omega, omega') couple different frequencies and satisfy:

    Sigma_{vv}(omega, omega') = [numerator] / [1 - [Sigma_{v chi}^{-1} Sigma_{v chi}'^{-1} - B(omega, omega')^{-1}] A(omega, omega')]

where A(omega, omega') = Tr A^2 (i omega + Psi_{v chi} A)^{-1} (i omega' + Psi_{v chi}' A)^{-1} and B(omega, omega') = Tr [Sigma_hat_{v chi} + B]^{-1} [Sigma_hat_{v chi}' + B]^{-1}.

The denominator factor 1 - [...] A is a **Dyson resummation**: it sums the ladder diagram series from the disorder average over O.

### Step 7: Final Deterministic Equivalent

    <v(omega) v(omega')^T> ~= (i omega + Psi_{v chi} A)^{-1} [v_0 v_0^T - Psi_{chi chi}(omega, omega') I] (i omega' + Psi_{v chi}' A)^{-1}

The ~= holds under trace against any fixed test matrix. Psi_{v chi} encodes the effective renormalized frequency after the disorder average, and Psi_{chi chi} is a two-frequency noise kernel.

---

## Appendix B: Model and Reduced-Gamma Model

### B.1: Masking

Two properties:
1. **Causal masking**: test tokens only receive attention from training tokens.
2. **Sign flip**: training tokens receive -sum(...) while test tokens receive +sum(...). Training tokens track y_mu - f^l_mu (residuals); test tokens accumulate +f^l_mu (predictions).

### B.2: Deriving the Reduced-Gamma Model

**Alignment assumptions:**
    W_x^T w_y = 0,  W_x^T w_o = 0,  w_y . w_y = |w_y|^2
    W_x^T (W_k^l)^T W_q^l W_x proportional to Gamma^l
    W_v^l W_x = 0,  W_v^l w_y proportional to w_y

x-information and y-information live in **orthogonal subspaces** of the residual stream.

**Residual stream dynamics.** Define Delta^l_mu = w_o . h^l_mu:

    Delta^{l+1}_mu = Delta^l_mu + (1/LP) sum_{nu=1}^P M_{mu,nu} x_nu^T Gamma^l x_mu Delta^l_nu

**Training tokens** (mu in [P], M_{mu,nu} = -1):
    Delta^l = (I - (1/LP) X^T Gamma X)^l y = (I - L^{-1} Sigma_hat Gamma)^l y

**Test tokens** (mu in {P+1,...,P+K}, M_{mu,nu} = +1):
    f* = (1/LP) x*^T Gamma X sum_{l=0}^{L-1} (I - L^{-1} Sigma_hat Gamma)^l y

**GD equivalence.** In-context GD on L_hat(beta) = (1/2P) ||X^T beta/sqrt(D) - y||^2 with preconditioner Gamma and step size D/L generates the same residual dynamics.

**Test error formula.** For target y = X^T beta*/sqrt(D) + sigma epsilon, the test loss splits into **bias** (approximation error) and **variance** (noise propagation) terms.

---

## Appendix C: ISO Setting

### C.1: SGD Dynamics (L = 1)

**Mean gradient.** Under isotropic x and beta:
    <G> = I - (1 + alpha^{-1} + sigma^2 alpha^{-1}) Gamma

The factor 1 + alpha^{-1}(1 + sigma^2) comes from Stein's lemma / Gaussian integration. The mean gradient drives Gamma toward Gamma* = [1 + alpha^{-1}(1 + sigma^2)]^{-1} I.

**Gradient variance.** Second-order term:
    tr <G^T G> = tr <G>^T <G> + (1/tau)(1 + kappa^{-1})(1 + alpha^{-1}(1+sigma^2))^2 [C(t) + ...]

The 1/tau factor means larger batches reduce noise. The 1/kappa factor is from masked evaluation points per context.

**Linear recursion for C(t) = (1/D)|Gamma(t) - Gamma*|^2:**
    C(t+1) = a(eta, alpha, kappa, tau) C(t) + b(eta, alpha, kappa, tau)

with:
    a = [1 - eta(1 + alpha^{-1}(1+sigma^2))]^2 + (eta^2/tau)(1+kappa^{-1})(1+alpha^{-1}(1+sigma^2))^2
    b = (eta^2/tau)(1+kappa^{-1}) alpha^{-1}(1+sigma^2)

Convergence requires a < 1. The floor C(infinity) = b/(1-a) is the **SGD noise floor**.

**Data efficiency.** Total tokens proportional to tau D . alpha D = O(D^2). Prior work required O(D^3) with K=1.

### C.2: Gradient Flow, Deep L >= 1

**Isotropy preserved.** At Gamma = gamma I, the gradient involves averages of <Sigma_hat f(Sigma_hat)> where under isotropic x, Sigma_hat has Marchenko-Pastur distribution which is rotationally invariant. By Schur's lemma, <Sigma_hat f(Sigma_hat)> = c . I.

**Scalar ODE.** Gradient flow on gamma is a gradient descent on a one-dimensional loss landscape involving the Marchenko-Pastur density.

---

## Appendix D: FS Setting

**Setup.** At alpha -> infinity, Sigma_hat -> Sigma deterministically.

**Eigenbasis.** When Omega and Sigma are codiagonalizable, the gradient is diagonal and the dynamics decouple:
    d gamma_k / dt = 2 omega_k lambda_k^2 (1 - L^{-1} lambda_k gamma_k)^{2L-1}

**Fixed point.** gamma_k* = L / lambda_k, i.e., Gamma* = L Sigma^{-1}. This makes (I - L^{-1} Gamma* Sigma) = 0.

**Infinite depth limit.** At L -> infinity, the ODE becomes:
    d gamma_k / dt = 2 omega_k lambda_k^2 exp(-2 lambda_k gamma_k)

Solution: 2 lambda_k gamma_k(t) = ln(1 + 4 omega_k lambda_k^3 t).

Loss: L(t) = sum_k omega_k lambda_k / (1 + 4 omega_k lambda_k^3 t).

**Powerlaw scaling.** Under lambda_k ~ k^{-nu} and omega_k lambda_k ~ k^{-nu*beta - 1}, the sum is dominated by a crossover at k* = t^{1/(nu*beta+1+2*nu)}, yielding L(t) ~ t^{-nu*beta/(nu*beta+2*nu+1)}.

**Brittleness (Result 5).** The fully trained model has Gamma = L Sigma^{-1}. If tested on Sigma', the error (I - Sigma^{-1} Sigma')^L **grows** as Sigma' deviates from Sigma. The model has hard-coded Sigma^{-1} as its preconditioner.

---

## Appendix E: RRS Setting

### E.1: Symmetry in Gradient Updates

**Rotational invariance of the loss.** Under Gamma -> V Gamma V^T for any fixed orthogonal V, the loss is unchanged because O is Haar-random: O =^d VO.

**Gradient isotropy.** At Gamma = gamma I, M = I - gamma Sigma_hat is independent of O. The remaining <O(...)O^T> average gives <O C O^T> = (tr C / N) . I by Schur's lemma.

**Why depth matters here but not in ISO/FS.** In ISO and FS, the optimal Gamma is anisotropic (Gamma* = L Sigma^{-1}), and even L=1 can reach zero loss. In RRS, the constraint Gamma = gamma I is **permanent** (forced by symmetry), so the model can only perform isotropic GD. More depth = more GD steps = lower loss regardless of context length.

---

## Appendix F: Scaling Law Theory

### F.1: Non-Proportional Scaling

The key modification: i omega_{(A^T A)^2}(tau) . i omega_{Sigma_hat}(tau) = (D/tau - 1) i omega.

For rank-N projection: tau = Tr (A^T A)^2 (i omega + (A^T A)^2)^{-1} = N/(i omega + 1), so i omega_{(A^T A)^2} = N/tau - 1.

### F.2: Dynamics at Infinite N, L, P

    d gamma/dt = -d/d gamma sum_k lambda_k omega_k exp(-2 gamma lambda_k) approx -d/d gamma gamma^{-beta}

ODE d gamma/dt = beta gamma^{-beta-1} has solution gamma(t) ~ t^{1/(beta+2)}, giving L(t) ~ t^{-beta/(2+beta)}.

### F.3: Depth Bottleneck

At finite depth L, stability requires gamma < 2L/lambda_1. Optimal gamma* approx L/lambda_1. Then:

    L approx sum_k lambda_k omega_k (1 - lambda_k/lambda_1)^{2L}
          approx integral dk k^{-nu*beta-1} exp(-2L k^{-nu})
          ~ L^{-beta}

The Laplace approximation: at large L, the integral is dominated by k ~ L^{1/nu}.

### F.4: DMFT Mapping to the Loss

Using the Cauchy integral formula: (I - gamma L^{-1} M)^L = contour integral (d omega / 2 pi) [i omega + M]^{-1} (1 + gamma L^{-1} i omega)^L. This converts the depth-L recursion into a two-point resolvent integral.

### F.5: Width and Context Scaling Laws

**Width bottleneck.** At t, L, P -> infinity, tau = N. For powerlaw lambda_k ~ k^{-nu}: i omega_{Sigma_hat} approx N^{-nu}. The loss:

    L(N) approx sum_k k^{-nu*beta-1} / (1 + k^{-nu} N^nu)^2 ~ N^{-nu*beta}

The loss comes from modes k > N not captured by the rank-N projection.

**Context bottleneck.** Setting N = D but finite P. Define r approx P^{-nu}:

    L(P) approx sum_k k^{-nu*beta-1} / (1 + k^{-nu} P^nu)^2 ~ P^{-nu*beta}

**Rank deficiency interpretation.** Both width and context create the same type of bottleneck: rank-N or rank-P matrices only allow the top N or P eigendirections to be learned.

**Compute-optimal shape.** At fixed compute C = t P^2 N^2 L, balancing N^{-nu*beta} = L^{-beta} gives L = N^nu.

---

## Appendix G: Enhancing Realism

### G.1.1: Untied Layers Under RRS

**Permutation symmetry.** The predictor involves sum_{l=1}^L gamma_l prod_{k<l}(I - gamma_k Sigma_hat / LP). Under RRS covariates, each Gamma^l = gamma^l I (rotational symmetry), and the product commutes. The predictor is a **symmetric function** of {gamma^l}.

**Consequence.** All gradients are equal at any symmetric point. Starting from gamma^l(0) = 0, gradient flow maintains gamma^l(t) = gamma(t) for all l. The dynamics are identical to the tied model.

**Subtlety.** If initial conditions are asymmetric, the permutation symmetry breaks and dynamics can diverge.

### G.2: Full Attention Weights

Under gradient flow on {W_x, W_k, W_q, W_v, w_o, w_y}, the symmetry arguments show W_x, W_k, W_q remain isotropic and W_v stays in the w_y w_o^T direction. The loss reduces to:

    L(w_v, w_q, w_k, w_x) = tr Omega Lambda (I - L^{-1} w_v w_k w_q (w_x)^2 Lambda)^{2L}

**Reduced scalar ODE.** Combined scale factor is w(t)^5 = w_v . w_k . w_q . w_x^2. The factor w^4 in the gradient arises from the chain rule through five weight matrices:

    dw/dt = w^4 tr Lambda^2 Omega (I - L^{-1} w^5 Lambda)^{2L-1}

Under power-law conditions: w(t) ~ t^{1/(5*beta+2)}, giving L(t) ~ t^{-5*beta/(5*beta+2)}.

The depth scaling L^{-beta} is **unchanged** by reparameterization.

### G.3: Softmax Experiments

Uses CompleteP (muP with depth scaling, Dey et al. 2025) for learning rate Theta_L(1) in softmax attention + Adam. All experiments confirm:
1. Clear depth separation consistent with linear theory
2. Softmax + MLP: similar qualitative behavior
3. Multi-head attention: depth separation persists but weakens with more heads

---

## Summary Table: What Each Appendix Proves

| Appendix | Core Proof |
|---|---|
| **A** | DMFT free-product deterministic equivalent via path integral + saddle point |
| **B.1** | Masking matrix design: opposite signs allow GD residual encoding |
| **B.2** | Alignment conditions -> Gamma-model; GD equivalence; test error expansion |
| **C.1** | SGD linear recursion for shallow ISO; gradient mean/variance; O(D^2) data efficiency |
| **C.2** | Gradient flow preserves isotropy via rotational invariance; MP law integral |
| **D** | FS eigenvalue ODEs; infinite-depth solution; powerlaw exponent; brittleness |
| **E.1** | RRS gradient isotropy via Schur's lemma; scalar gamma ODE; why depth always helps |
| **F.1** | Non-proportional DMFT equations; rank-N projection tau-transform |
| **F.2** | Infinite-resource limit: gamma ~ t^{1/(beta+2)}, L ~ t^{-beta/(2+beta)} |
| **F.3** | Finite-depth loss landscape; optimal gamma* approx L/lambda_1; L ~ L^{-beta} |
| **F.4** | Cauchy integral converts depth-L recursion to two-point resolvent |
| **F.5** | Rank-N projection -> N^{-nu*beta}; rank-P context -> P^{-nu*beta}; compute-optimal L proportional to N^nu |
| **G.1.1** | Permutation symmetry of predictor -> tied = untied at symmetric init |
| **G.2** | Full weight balancing; w^5 reparameterization; time exponent 5*beta/(5*beta+2) |
| **G.3** | Softmax + Adam experiments confirming depth separation |

---

## The Three-Setting Summary

| Setting | Fixed pt. of Gamma | Depth helps at alpha->infinity? | Robust to OOD? |
|---|---|---|---|
| ISO | gamma I | No | Yes (isotropic) |
| FS | L Sigma^{-1} | No | No (brittle) |
| RRS | gamma I (forced) | **Yes** | Yes |

The crucial distinction: in RRS, the model *cannot* learn a preconditioner because O_c is random per context. It is forced to implement bare GD, and more depth = more GD steps = lower loss, indefinitely.
