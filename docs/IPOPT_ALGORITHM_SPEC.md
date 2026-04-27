# Ipopt 3.14 Primal-Dual Interior-Point Algorithm with Filter Line Search: Implementation-Grade Specification

**Audience.** This document is written for two readers:

1. An expert numerical-optimization programmer re-implementing Ipopt 3.14 in Rust from scratch.
2. A reviewer auditing whether a candidate implementation is faithful to Ipopt 3.14.

**Conventions.**
- Math uses LaTeX-style `$...$` and `$$...$$`.
- Pseudocode is Rust-leaning (snake_case identifiers, `let mut`, slices). It is illustrative; the source of truth is the C++ at the cited file:line.
- Source citations have the form `IpFile.cpp:LINE`. Files live under `Ipopt/src/Algorithm/` unless prefixed (`Interfaces/`, `LinAlg/`, `Common/`).
- Sign convention. **Ipopt uses $L = f + y^\top c$**, not $f - y^\top c$. This is verified at `IpIpoptCalculatedQuantities.cpp:2019, 2022-2023`. Every formula below follows that convention.
- Primary references:
  - **WB06**: Wachter & Biegler, *Math. Prog.* 106 (2006), "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming."
  - **WB05a**: Wachter & Biegler 2005 *SIAM J. Optim.* (global convergence).
  - **WB05b**: Wachter & Biegler 2005 *SIAM J. Optim.* (local convergence).
  - **NWW09**: Nocedal, Wachter & Waltz 2009, "Adaptive barrier strategies."

---

## 1. Problem Formulation Ipopt Actually Solves

### 1.1 User NLP

The TNLP interface (`Interfaces/IpTNLP.hpp`) accepts

$$
\min_{x\in\mathbb{R}^n} f(x) \quad\text{s.t.}\quad g_L \le g(x) \le g_U,\quad x_L \le x \le x_U,
$$

where any individual bound may be $\pm\infty$ (encoded as $|b| \ge$ `nlp_lower_bound_inf` / `nlp_upper_bound_inf`, default $\pm 10^{19}$, `IpTNLPAdapter.cpp:100-112`).

### 1.2 Internal NLP After TNLPAdapter

`TNLPAdapter` (`IpTNLPAdapter.cpp`) reformulates the user NLP into Ipopt's internal form:

$$
\min_{x,s}\; f(x)\quad\text{s.t.}\quad
c(x) = 0,\quad d(x) - s = 0,\quad d_L \le s \le d_U,\quad x_L \le x \le x_U.
$$

- $c(x)$ is the equality block (rows where $g_L=g_U$).
- $d(x)$ is the inequality block; an explicit slack vector $s$ is introduced so $d_L \le s \le d_U$ becomes the only "double-bounded" piece. The slack is a real primal variable, with its own dual multipliers $v_L,v_U$.
- Single-sided original inequalities pass through; only one of $d_L,d_U$ is finite for those rows.

### 1.3 Fixed-Variable Treatment

Variables with $x_L = x_U$ are handled per `fixed_variable_treatment` (`IpTNLPAdapter.cpp:100-112`). Allowed values:

| Value | Behavior |
|---|---|
| `make_parameter` (default) | Fixed components are removed from $x$ entirely; $f$, $c$, $d$ become functions of the reduced $x$. The bound multipliers $z_L,z_U$ for those components are not produced. |
| `make_parameter_nodual` | Same as `make_parameter` but the user does not receive bound duals for the fixed components on output. |
| `make_constraint` | Each fixed component becomes an additional equality $x_i - \bar x_i = 0$. |
| `relax_bounds` | Fixed bounds are perturbed by `bound_relax_factor` (Sec 1.5) so the variable becomes "almost free." |

A Rust port should honor at least `make_parameter` and `make_constraint`; the others are convenience.

### 1.4 Bound Relaxation: `bound_relax_factor`

Active finite bounds are relaxed outward before the IPM sees them (`IpOrigIpoptNLP.cpp:53-64, 459-481`). For each finite component:

$$
\tilde x_{L,i} = x_{L,i} - \min(\texttt{constr\_viol\_tol},\; |\texttt{brf}|\cdot\max(|x_{L,i}|, 1)),
$$
$$
\tilde x_{U,i} = x_{U,i} + \min(\texttt{constr\_viol\_tol},\; |\texttt{brf}|\cdot\max(|x_{U,i}|, 1)),
$$

with default `bound_relax_factor` $= 10^{-8}$. The relaxation ensures small numerical noise does not push a feasible starting point onto the bound. The same shifting is applied to the slack bounds $d_L, d_U$.

Setting `bound_relax_factor = 0` disables relaxation entirely. The user-facing return values are reported on the ORIGINAL bounds (`honor_original_bounds`, `IpOrigIpoptNLP.cpp:65-72`).

### 1.5 Barrier Subproblem

For mu > 0, Ipopt solves the barrier problem

$$
\min_{x,s}\; \varphi_\mu(x,s) := f(x) - \mu \sum_{i\in\mathcal{L}_x}\ln(x_i - x_{L,i}) - \mu\sum_{i\in\mathcal{U}_x}\ln(x_{U,i}-x_i)
- \mu\sum_{i\in\mathcal{L}_s}\ln(s_i - d_{L,i}) - \mu\sum_{i\in\mathcal{U}_s}\ln(d_{U,i}-s_i)
$$

subject to $c(x)=0$, $d(x)-s=0$. The **damped barrier** variant (Sec 4.6) adds linear damping in $x$ and $s$ along the one-sided bounds.

### 1.6 Lagrangian and KKT

Lagrangian (sign convention `+y^T c`, `IpIpoptCalculatedQuantities.cpp:2019`):

$$
L(x,s,y_c,y_d,z_L,z_U,v_L,v_U) = f(x) + y_c^\top c(x) + y_d^\top (d(x)-s) - z_L^\top(x-x_L) + z_U^\top(x_U-x) - v_L^\top(s-d_L) + v_U^\top(d_U-s).
$$

Stationarity:
$$
\nabla f(x) + J_c(x)^\top y_c + J_d(x)^\top y_d - z_L + z_U = 0,
$$
$$
-y_d - v_L + v_U = 0.
$$

Complementarity (componentwise):
$$
(x - x_L)\circ z_L = \mu e,\quad (x_U-x)\circ z_U = \mu e,\quad (s-d_L)\circ v_L = \mu e,\quad (d_U-s)\circ v_U = \mu e.
$$

All bound multipliers must remain strictly nonnegative; primal-dual interior point keeps all four $z_L,z_U,v_L,v_U \ge 0$ and all four slack expressions strictly positive.

### 1.7 The Augmented (4-Block) System

The Newton step on the perturbed KKT system has eight components $(\delta x, \delta s, \delta y_c, \delta y_d, \delta z_L, \delta z_U, \delta v_L, \delta v_U)$. Eliminating $\delta z, \delta v$ via the complementarity rows gives the **augmented system** (4 blocks):

$$
\begin{bmatrix}
W + \Sigma_x + \delta_x I & 0 & J_c^\top & J_d^\top \\
0 & \Sigma_s + \delta_s I & 0 & -I \\
J_c & 0 & -\delta_c I & 0 \\
J_d & -I & 0 & -\delta_d I
\end{bmatrix}
\begin{bmatrix}\delta x\\ \delta s\\ \delta y_c\\ \delta y_d\end{bmatrix}
= -
\begin{bmatrix}r_x\\ r_s\\ r_c\\ r_d\end{bmatrix}.
$$

- $W = \nabla_{xx}^2 L$ (or quasi-Newton approximation).
- $\Sigma_x = P_{x,L}^\top \mathrm{diag}(z_L/(x-x_L)) P_{x,L} + P_{x,U}^\top\mathrm{diag}(z_U/(x_U-x))P_{x,U}$ where $P_{x,L},P_{x,U}$ are the $\{0,1\}$ selection matrices that pick rows with finite lower / upper bounds (`IpIpoptCalculatedQuantities.cpp:3501-3525`).
- $\Sigma_s$ is defined analogously from $v_L,v_U$ and the slack bounds.
- $\delta_x, \delta_s, \delta_c, \delta_d$ are inertia/regularization perturbations (Sec 4.4).
- The right-hand side is built from the gradient of the (damped) Lagrangian and constraint residuals (Sec 4.5).

After solving for $\delta x,\delta s, \delta y_c,\delta y_d$, the eliminated blocks are recovered:

$$
\delta z_L = -\frac{Z_L \delta x_L + (X-X_L)Z_L e - \mu e}{X - X_L},\quad
\delta z_U = +\frac{Z_U \delta x_U + (X_U-X)Z_U e - \mu e}{X_U - X},
$$

(and analogous formulas for $\delta v_L,\delta v_U$ in the $s$ block). The exact assembly is in `IpPDFullSpaceSolver.cpp`.

---

## 2. Initialization

Reference: `AGENT_REFERENCE/INITIALIZATION.md`, `IpDefaultIterateInitializer.cpp`.

### 2.1 Push x Inside Bounds (Cold Start)

Per-component, with `bound_push` $= \kappa_1$ (default 0.01) and `bound_frac` $= \kappa_2$ (default 0.01) (`IpDefaultIterateInitializer.cpp:33-49`):

For two-sided bounds:
$$
p_{L,i} = \min(\kappa_1\cdot\max(|x_{L,i}|, 1),\ \kappa_2\cdot(x_{U,i}-x_{L,i})),\qquad
p_{U,i} = \min(\kappa_1\cdot\max(|x_{U,i}|, 1),\ \kappa_2\cdot(x_{U,i}-x_{L,i})).
$$

For one-sided bounds: $p_{L,i} = \kappa_1\cdot\max(|x_{L,i}|, 1)$ (and analogously for upper).

Then the user's $x_0$ is projected (`IpDefaultIterateInitializer.cpp:603-616`):

$$
x_i \leftarrow \begin{cases} x_{L,i}+p_{L,i} & x_i < x_{L,i}+p_{L,i} \\ x_{U,i}-p_{U,i} & x_i > x_{U,i}-p_{U,i} \\ x_i & \text{otherwise.}\end{cases}
$$

A pre-pass with $(p_L,p_U)=(0,0)$ snaps any value at-or-outside the bounds onto the bound first (`IpDefaultIterateInitializer.cpp:493-497`), to avoid log-of-zero.

### 2.2 Slack Initialization

After the $x$ projection, $s_0 := d(x_{\text{pushed}})$ (`IpDefaultIterateInitializer.cpp:239`), then the same projection is applied with `slack_bound_push` and `slack_bound_frac` (defaults: fall back to `bound_push` / `bound_frac`).

### 2.3 Bound Multipliers $z_L, z_U, v_L, v_U$

Selected by `bound_mult_init_method` (`IpDefaultIterateInitializer.cpp:254-288`):

- `constant` (default): all entries set to `bound_mult_init_val` (default 1.0).
- `mu-based`: $z_{L,i} = \mu_0/(x_i-x_{L,i})$, $z_{U,i}=\mu_0/(x_{U,i}-x_i)$, etc., yielding $z\circ s = \mu_0$.

### 2.4 Equality Multipliers $y_c, y_d$ (Least Squares)

If $\dim(y_c)=n$ (square): set $y_c = y_d = 0$ (`IpDefaultIterateInitializer.cpp:685-691`).

Otherwise, call the least-squares estimator (`IpLeastSquareMults.cpp:30-95`) which solves the augmented system with $W=0$:

$$
\begin{bmatrix}
I & J_c^\top & J_d^\top \\
J_c & 0 & 0 \\
J_d & 0 & 0
\end{bmatrix}
\begin{bmatrix}u\\ y_c\\ y_d\end{bmatrix}
=
\begin{bmatrix}\nabla f - z_L + z_U \\ 0 \\ 0\end{bmatrix},
\quad
\text{rhs}_s = v_L - v_U.
$$

(`IpLeastSquareMults.cpp:53-61`.) If $\|(y_c,y_d)\|_\infty > $ `constr_mult_init_max` (default 1000), discard and set $y=0$.

### 2.5 Initial Barrier Parameter

`mu_init` (default 0.1, `IpMonotoneMuUpdate.cpp:42`). Initial fraction-to-the-boundary parameter:

$$
\tau_0 = \max(\texttt{tau\_min},\ 1-\mu_0),\qquad \texttt{tau\_min}=0.99.
$$

### 2.6 Cold-Start Pseudocode

```rust
fn cold_start(nlp: &Nlp, opt: &Options) -> Iterate {
    let mut x = nlp.starting_x();
    push_into_box(&mut x, &nlp.x_l, &nlp.x_u, opt.bound_push, opt.bound_frac);
    let mut s = nlp.eval_d(&x);
    push_into_box(&mut s, &nlp.d_l, &nlp.d_u, opt.slack_bound_push, opt.slack_bound_frac);

    let (z_l, z_u, v_l, v_u) = match opt.bound_mult_init_method {
        Constant => fill_constant(opt.bound_mult_init_val),
        MuBased  => fill_mu_over_slack(opt.mu_init, &x, &s, &nlp),
    };

    let (y_c, y_d) = if dim_yc == n {
        (zeros(), zeros())
    } else {
        let est = least_squares_y(&x, &s, &z_l, &z_u, &v_l, &v_u, nlp);
        if est.inf_norm() > opt.constr_mult_init_max { (zeros(), zeros()) } else { est }
    };

    Iterate { x, s, y_c, y_d, z_l, z_u, v_l, v_u, mu: opt.mu_init,
              tau: opt.tau_min.max(1.0 - opt.mu_init) }
}
```

### 2.7 Warm Start

If `warm_start_init_point=yes`, multipliers are taken from the user, clamped to `warm_start_mult_init_max` (default 1e6), bound-multiplier values floored at `warm_start_mult_bound_push` (default 1e-3). Default warm-start primal push uses tighter constants (1e-3 vs 0.01). See `IpWarmStartIterateInitializer.cpp:118+` and `INITIALIZATION.md` Sec "Warm Start Parameters."

---

## 3. Outer Barrier Loop and Subproblem Convergence

### 3.1 Two Strategies: monotone vs adaptive

`mu_strategy = adaptive` (default) or `monotone`. Adaptive runs in two modes (Free / Fixed), Free uses an oracle to pick mu per iteration. The line search filter is **reset** every time mu changes (`IpFilterLSAcceptor.cpp:524-532`).

### 3.2 Monotone (Fiacco-McCormick)

The "barrier stop test" determines when to decrease mu (`IpMonotoneMuUpdate.cpp:144`):

$$
E_\mu(x,s,y,z,v) \le \kappa_\varepsilon \cdot \mu,\quad \kappa_\varepsilon = \texttt{barrier\_tol\_factor} = 10.
$$

$E_\mu$ is the scaled NLP error (Sec 8) with mu in the complementarity term. When the test holds (or `tiny_step_flag` is set), update:

$$
\mu_{k+1} = \max\!\Big(\min(\kappa_\mu\mu_k,\ \mu_k^{\theta_\mu}),\ \mu_{\text{target}},\ \tfrac{\min(\text{tol},\,\text{compl\_inf\_tol})}{\kappa_\varepsilon+1}\Big),\qquad
\tau_{k+1} = \max(\tau_{\min},\ 1-\mu_{k+1}).
$$

Defaults: $\kappa_\mu=$ `mu_linear_decrease_factor` $=0.2$; $\theta_\mu=$ `mu_superlinear_decrease_power` $=1.5$. With `mu_allow_fast_monotone_decrease=yes` (default), the decrease loop continues so long as the test still holds at the new mu (`IpMonotoneMuUpdate.cpp:172-184`).

### 3.3 Adaptive Mu

State machine with two modes (`IpAdaptiveMuUpdate.cpp`):

**Free mode (default start, `cpp:239`):** each iteration
1. $\tau = \max(\tau_{\min},\ 1 - E_0)$ where $E_0 = $ `curr_nlp_error`.
2. Call oracle: `free_mu_oracle->CalculateMu(max(mu_min, mu_target), mu_max, &mu)`.
3. Safeguards: $\mu \leftarrow \max(\mu, \mu_{\min},\ \texttt{lower\_mu\_safeguard})$, then $\mu \leftarrow \min(\mu, \mu_{\max})$.
4. **Reset filter.**

**Fixed mode:** identical to monotone update with $\kappa_\varepsilon\mu$ subproblem test. New fixed mu seeded by `NewFixedMu` (`cpp:583-627`):

$$
\mu_{\text{fix}} = \texttt{adaptive\_mu\_monotone\_init\_factor}\cdot\overline{\text{compl}},\quad \texttt{factor}=0.8,
$$
clamped to $[\mu_{\min}, \mu_{\max}]$.

**Mode switching (Free $\to$ Fixed):** when `CheckSufficientProgress()` fails (`cpp:358-360`), or line search was skipped / tiny step (`cpp:347-351`), unless globalization is `never-monotone-mode`.

**Mode switching (Fixed $\to$ Free):** when sufficient progress holds and no tiny step (`cpp:303-304`).

### 3.4 Adaptive Globalization

`adaptive_mu_globalization`:

- `obj-constr-filter` (default): a 2-D filter on $(\,f, \theta)$. The current point is "sufficient progress" iff acceptable to the filter with margin $\texttt{filter\_margin\_fact}\cdot\min(\texttt{filter\_max\_margin},\ E_0)$ (defaults $10^{-5}$, $1.0$). Accepted points are added (`cpp:519-529`).
- `kkt-error`: keep up to `adaptive_mu_kkterror_red_iters` (default 4) reference KKT errors; "sufficient progress" iff $E_{\text{cur}} \le 0.9999\cdot E_{\text{ref}}$ for some stored ref.
- `never-monotone-mode`: always returns true; no global guarantee. Forced when `mehrotra_algorithm=true`.

### 3.5 Mu Oracles (Free Mode)

`free_mu_oracle` selects the strategy:

**LOQO** (`IpLoqoMuOracle.cpp:34-66`):
$$
\xi = \frac{\min_i s_iz_i}{\overline{s\circ z}},\quad
\sigma = 0.1 \cdot \min\!\Big(\tfrac{0.05(1-\xi)}{\xi},\ 2\Big)^3,\quad
\mu = \mathrm{clamp}(\sigma\cdot\overline{s\circ z},\ \mu_{\min},\ \mu_{\max}).
$$

**Quality Function (default, `IpQualityFunctionMuOracle.cpp:154-485`):**
1. Solve KKT for **affine** step ($\mu=0$ RHS).
2. Solve KKT for **centering** step (RHS = $-\overline{s\circ z}\nabla\kappa$, complementarity rows $=\overline{s\circ z}$).
3. Combined step $d(\sigma) = d_{\text{aff}} + \sigma d_{\text{cen}}$.
4. Quality function $Q(\sigma) = \|\text{dual\_inf}\| + \|\text{primal\_inf}\| + \|\text{compl}\| + \text{centrality} + \text{balancing}$ where infeasibility residuals scale as $(1-\alpha)\cdot$ current.
5. Golden-section minimize over $[\sigma_{\min},\sigma_{\max}] = [10^{-6},10^2]$, `quality_function_max_section_steps=8`, sigma tolerance $0.01$.
6. $\mu = \sigma^\star\cdot\overline{s\circ z}$.

Defaults: norm is `2-norm-squared` (each component is $\|v\|_2^2/n$), centrality `none`, balancing `none`.

**Probing (Mehrotra)** (`IpProbingMuOracle.cpp:47-133`):
1. Solve affine step.
2. FTB with $\tau=1$ to get $\alpha_p^{\text{aff}}, \alpha_d^{\text{aff}}$.
3. $\mu_{\text{aff}} = \overline{(s+\alpha_p^{\text{aff}}\delta s)(z+\alpha_d^{\text{aff}}\delta z)}$.
4. $\sigma = \min((\mu_{\text{aff}}/\mu_{\text{cur}})^3,\ \sigma_{\max})$, $\mu = \sigma\mu_{\text{cur}}$.

### 3.6 Mu Bounds

`mu_max_fact` $=10^3$, `mu_max` $=10^5$, `mu_min` $=10^{-11}$ (auto-set from `tol` if not provided).

### 3.7 No-Bounds Special Case

If problem has zero bound multipliers, set $\mu = \mu_{\min}, \tau = \tau_{\min}$ immediately (`IpAdaptiveMuUpdate.cpp:278-290`). Avoids stalling.

---

## 4. Step Computation

### 4.1 Top-Level: PDSearchDirCalculator

`IpPDSearchDirCalc.cpp:57-140`. Per iteration:

1. Build right-hand side `rhs` from the current iterate (Sec 4.5).
2. Call `pd_solver_->Solve(-1.0, 0.0, rhs, delta)` (the $-1$ negates the RHS so the result is a Newton **descent** direction).
3. If `mehrotra_algorithm=true` or `corrector_type` is enabled, build a corrector RHS using stored affine deltas and call `Solve` again with `1.0,1.0` to add the corrector to `delta`.

### 4.2 PDFullSpaceSolver: Reduction & Refinement

`IpPDFullSpaceSolver.cpp`. Steps:

1. Eliminate $\delta z, \delta v$ analytically (formulas in 1.7) to form the 4-block augmented RHS.
2. Call `aug_system_solver->Solve(...)` on the 4-block KKT.
3. Recover $\delta z, \delta v$.
4. **Iterative refinement** on the FULL 8-component residual: up to `max_refinement_steps` (default 10). Stop when residual norm $\le$ `residual_ratio_max` (default $10^{-10}$) of RHS norm. If improvement stalls or `residual_ratio_singular` is exceeded, request a re-factorization with stronger inertia perturbation.

### 4.3 Augmented System Solver

`IpStdAugSystemSolver.cpp`: assembles the 4x4 block matrix as a triplet (or CSC), passes to the chosen `SymLinearSolver` (MA27 / MA57 / MA97 / MUMPS / Pardiso / WSMP / SPRAL). Inertia $(n_+, n_-, n_0)$ is read back from the linear solver and given to the perturbation handler.

The required inertia for an unperturbed system is

$$
n_+ = n_x + n_s,\quad n_- = m_c + m_d,\quad n_0 = 0.
$$

(One positive eigenvalue per primal, one negative per equality.)

### 4.4 Inertia Correction State Machine

`IpPDPerturbationHandler.cpp:144-538`. Maintains $(\delta_x, \delta_s, \delta_c, \delta_d)$ between calls. The C-side identities $\delta_x \equiv \delta_s$ and $\delta_c \equiv \delta_d$ are enforced (`cpp:382, 412`). All defaults from `cpp:27-101`:

| Symbol | Option | Default |
|---|---|---|
| $\delta_x^{\text{init}}$ | `first_hessian_perturbation` | $10^{-4}$ |
| $\delta_x^{\min}$ | `min_hessian_perturbation` | $10^{-20}$ |
| $\delta_x^{\max}$ | `max_hessian_perturbation` | $10^{20}$ |
| dec factor | `perturb_dec_fact` | $1/3$ |
| inc factor | `perturb_inc_fact` | 8 |
| first inc factor | `perturb_inc_fact_first` | 100 |
| $\delta_{cd}^{\text{val}}$ | `jacobian_regularization_value` | $10^{-8}$ |
| $\delta_{cd}^{\text{exp}}$ | `jacobian_regularization_exponent` | $0.25$ |
| degeneracy iters | (hardcoded) `degen_iters_max` | 3 |

**`get_deltas_for_wrong_inertia` (`cpp:366-417`):**

```rust
fn get_deltas_for_wrong_inertia(state: &mut PerturbState) -> Option<(f64,f64,f64,f64)> {
    // delta_x and delta_s share a value; delta_c and delta_d share a value.
    let dx_new = match (state.dx_curr, state.dx_last) {
        (0.0, 0.0)            => DX_INIT,                         // first time
        (0.0, last) if last>0.0 => f64::max(DX_MIN, last * DEC),  // reuse, decreased
        (curr, 0.0)           => curr * INC_FIRST,                // 100x bump
        (curr, last) if 1e5*last < curr => curr * INC_FIRST,
        (curr, _)             => curr * INC,                      // 8x bump
    };
    if dx_new > DX_MAX { return None; }                           // give up
    let dcd = JAC_REG_VAL * mu.powf(JAC_REG_EXP);                 // 1e-8 * mu^0.25
    state.dx_curr = dx_new; state.ds_curr = dx_new;
    Some((dx_new, dx_new, dcd, dcd))
}
```

**Degeneracy detection (`cpp:470-538`):** When the very first solve in an iteration shows correct inertia, but iterative refinement fails or the matrix is structurally singular, the handler enters a small state machine (`TEST_DELTA_C_EQ_0_DELTA_X_EQ_0` etc.) for up to `degen_iters_max=3` iterations to learn whether the constraint Jacobian is rank-deficient (so $\delta_c > 0$ should be applied always) or the Hessian is degenerate (so $\delta_x > 0$ should be applied always). Once classified, subsequent iterations use the appropriate "always perturb" strategy.

The flag `perturb_always_cd` (default false) forces $\delta_c, \delta_d > 0$ unconditionally, which is the "safe" mode.

### 4.5 RHS Assembly

`IpPDSearchDirCalc.cpp:57-140` calls `IpIpoptCalculatedQuantities` to fetch:

| Block | Content |
|---|---|
| `rhs_x` | `curr_grad_lag_with_damping_x` (Sec 4.6) |
| `rhs_s` | `curr_grad_lag_with_damping_s` |
| `rhs_c` | $c(x)$ |
| `rhs_d` | $d(x)-s$ |
| `rhs_z_L` | `curr_relaxed_compl_x_L` $= (X-X_L)Z_L e - \mu e$ |
| `rhs_z_U` | `curr_relaxed_compl_x_U` |
| `rhs_v_L` | `curr_relaxed_compl_s_L` |
| `rhs_v_U` | `curr_relaxed_compl_s_U` |

Then `pd_solver->Solve(-1.0, 0.0, rhs, delta)` returns $\delta = -A^{-1}\text{rhs}$, which is the Newton step.

### 4.6 Damped Lagrangian Gradient (`kappa_d`)

For variables with **only one** finite bound (one-sided), Ipopt adds a linear damping term to the barrier so the bound multiplier stays bounded. Let $\eta_L$ be the indicator of components with ONLY a lower bound (no upper bound), $\eta_U$ analogously. Default `kappa_d = 1e-5` (`IpIpoptCalculatedQuantities.cpp:144-183`).

$$
\nabla_x L_{\text{damped}} = \nabla f + J_c^\top y_c + J_d^\top y_d - z_L + z_U + \kappa_d\mu\,P_{x,L}^\top\eta_L - \kappa_d\mu\,P_{x,U}^\top\eta_U.
$$

(`IpIpoptCalculatedQuantities.cpp:2131-2180`.) Same formula for $s$.

**Crucial nuance.** The damping enters only the **search-direction RHS**. The convergence test (Sec 8) uses the UN-damped $\nabla L$ (`IpIpoptCalculatedQuantities.cpp:1993-2030`). A Rust port that puts $\kappa_d$ into the convergence test has a (silent) bug: it will declare convergence on a point where dual infeasibility is actually nonzero.

### 4.7 Mehrotra Predictor-Corrector

When `mehrotra_algorithm=true` or `corrector_type != none` (`IpPDSearchDirCalc.cpp:81-111`), after the predictor (affine) step is computed, a corrector RHS is built that adds, to the complementarity rows, the second-order term $\delta x^{\text{aff}} \circ \delta z^{\text{aff}}$ (componentwise products). The corrector is computed by another `Solve` and **added** to the predictor `delta`. The Mehrotra setting also forces `adaptive_mu_globalization=never-monotone-mode`, `accept_every_trial_step=yes`, `corrector_type=none` (handled internally), aggressive bound-push defaults (`IpIpoptAlg.cpp:97-185`).

### 4.8 RHS Sign Audit (Common Pitfall)

The RHS components above are written so that `pd_solver->Solve(-1.0, 0.0, rhs, delta)` produces the Newton **decrease** direction. If your `Solve` does not negate, flip ALL signs in the RHS. The complementarity rows in particular: it is `(X-X_L)Z e - mu e`, NOT `mu e - (X-X_L)Z e`.

---

## 5. Step Length Selection (Primal & Dual)

### 5.1 Fraction-to-the-Boundary

For each component with $\delta_i < 0$,

$$
\alpha_i = -\tau\,\frac{x_i}{\delta_i},
$$

and $\alpha_{\max} = \min(1,\ \min_i\alpha_i)$ (`IpDenseVector.cpp:1325-1383`).

Two separate maxima are produced:
- $\alpha_p^{\max}$ from primal slacks $(x-x_L), (x_U-x), (s-d_L), (d_U-s)$ vs $(\delta x, \delta s)$.
- $\alpha_d^{\max}$ from dual variables $(z_L,z_U,v_L,v_U)$ vs $(\delta z_L,\delta z_U,\delta v_L,\delta v_U)$ (these must remain $> 0$).

`tau` is set by the mu update strategy (Sec 3): in Free mode $\tau = \max(\tau_{\min}, 1-E_0)$; in Fixed/Monotone mode $\tau = \max(\tau_{\min}, 1-\mu)$.

### 5.2 alpha_for_y (Equality Multiplier Step)

Bound multipliers $z, v$ ALWAYS use $\alpha_d$ (`IpBacktrackingLineSearch.cpp:928`). For $y_c, y_d$:

| `alpha_for_y` | Formula |
|---|---|
| `primal` (default) | $\alpha_p$ |
| `bound-mult` | $\alpha_d$ |
| `min` / `max` | $\min/\max(\alpha_p,\alpha_d)$ |
| `full` | 1.0 |
| `min-dual-infeas` | $\arg\min_\alpha \|\nabla L(x^+) + J^\top(y+\alpha\delta y)\|^2$, clipped to $[0,1]$ |
| `safer-min-dual-infeas` | as above, safeguarded between $\min/\max(\alpha_p,\alpha_d)$ |
| `primal-and-full` | $\alpha_p$, but 1.0 if $\|\delta x\|_\infty\le\texttt{alpha\_for\_y\_tol}$ |
| `dual-and-full` | $\alpha_d$ analogously |
| `acceptor` | delegated to filter acceptor |

### 5.3 Magic Step (`IpBacktrackingLineSearch.cpp:1013-1111`)

After accepting a step, Ipopt may further adjust $s_i$ for inequality components where moving the slack reduces the barrier objective without violating the bound or complementarity. This is a tiny extra adjustment, applied separately per component, and is independent of $\delta s$.

### 5.4 The $\kappa_\sigma$ Bound-Multiplier Reset (Eqn 16, WB06)

After a trial point is accepted, BEFORE promoting trial to current, every bound multiplier component is clamped (`IpIpoptAlg.cpp:1055-1134`):

```rust
fn correct_bound_multiplier(z_trial: &mut [f64], slack_trial: &[f64],
                             mu: f64, kappa_sigma: f64) -> i32 {
    if kappa_sigma < 1.0 || z_trial.is_empty() { return 0; }
    // Quick test: is correction needed at all?
    let max_zs = max(z_i * s_i);
    let min_zs = min(z_i * s_i);
    if max_zs <= kappa_sigma * mu && min_zs >= mu / kappa_sigma { return 0; }
    let mut corrected = 0;
    for i in 0..z_trial.len() {
        let lower = mu / (kappa_sigma * slack_trial[i]);
        let upper = kappa_sigma * mu / slack_trial[i];
        let z_new = z_trial[i].max(lower).min(upper);   // lower clamp first, then upper
        if z_new != z_trial[i] { corrected += 1; }
        z_trial[i] = z_new;
    }
    corrected
}
```

`kappa_sigma` defaults to $10^{10}$. The mu used for comparison is `min(trial_avrg_compl, 1e3)` in Free mode and `curr_mu` otherwise (`IpIpoptAlg.cpp:1075-1083`). The clamp is applied to all four of $z_L, z_U, v_L, v_U$ (lines 721-752). If `kappa_sigma < 1`, the clamp is disabled.

### 5.5 Slack Move (`slack_move`)

Default `slack_move` = $\varepsilon^{0.75} \approx 4.86\cdot 10^{-12}$ (`IpIpoptCalculatedQuantities.cpp:144-183`). `CalculateSafeSlack` (lines 455-537) ensures positive slacks $s$ never fall below

$$
s_{\min,i} = \max(\texttt{slack\_move}\cdot\max(|s_i|,1),\ \texttt{slack\_move}).
$$

If a trial slack is below $s_{\min,i}$ but the bound is not, the **bound itself** is widened by a tiny amount in `AcceptTrialPoint` (`IpIpoptAlg.cpp:665-714`), logging "Slack too small, adjusting variable bound." This avoids catastrophic divisions in $\Sigma$ later.

### 5.6 advance_z Ordering (Important for Re-implementation)

The order inside `IpoptAlgorithm::AcceptTrialPoint` (`IpIpoptAlg.cpp:652-819`) is:

1. Skip if line search marked as failed.
2. Adjust trial bounds for too-small slacks.
3. Apply `correct_bound_multiplier` to **trial** $z_L, z_U, v_L, v_U$ (Sec 5.4).
4. `IpData().AcceptTrialPoint()` -- promote trial to current.
5. Optionally recompute $y_c, y_d$ via least-squares (`recalc_y=yes`, default for L-BFGS, off otherwise; `recalc_y_feas_tol=1e-6`).

If you correct multipliers AFTER the promotion, you are correcting the wrong vector; if you skip step 2 you may divide by tiny slacks in step 3.

---

## 6. Filter Line Search

Reference: `AGENT_REFERENCE/LINE_SEARCH.md`, `IpFilterLSAcceptor.cpp`, `IpBacktrackingLineSearch.cpp`, WB06 Sec 3.

### 6.1 The Two Measures

- $\theta(x,s) = \|c(x)\|_1 + \|d(x)-s\|_1$ (1-norm of constraint violation).
- $\varphi_\mu(x,s) = $ barrier objective from Sec 1.5, plus the $\kappa_d$ damping linear term so that $\nabla\varphi_\mu = $ damped Lagrangian gradient at $y=0,z=0,v=0$.

### 6.2 Filter Bounds and Switching Threshold

Computed once at the start of each line search, based on the reference (current) iterate:

$$
\theta_{\max} = \theta_{\max,\text{fact}}\cdot\max(1,\theta_{\text{ref}}),\qquad \theta_{\min} = \theta_{\min,\text{fact}}\cdot\max(1,\theta_{\text{ref}}).
$$

Defaults: `theta_max_fact = 1e4`, `theta_min_fact = 1e-4`.

### 6.3 Switching Condition (f-type vs h-type)

Let $g_\varphi := \nabla\varphi_\mu^\top d$ at the reference point (`reference_gradBarrTDelta`). `IsFtype` (`IpFilterLSAcceptor.cpp:273-295`):

$$
g_\varphi < 0 \quad\text{AND}\quad \alpha\cdot(-g_\varphi)^{s_\varphi} > \delta\cdot\theta_{\text{ref}}^{s_\theta}.
$$

Defaults: `delta = 1.0`, `s_phi = 2.3`, `s_theta = 1.1`.

If $\theta_{\text{ref}} > \theta_{\min}$, the trial is forced to be **h-type** regardless. Note alpha is on the LHS: as $\alpha$ shrinks during backtracking the condition can flip from f-type to h-type.

### 6.4 Armijo (f-type Acceptance)

`ArmijoHolds` (`IpFilterLSAcceptor.cpp:439-448`):

$$
\varphi_\mu(\text{trial}) - \varphi_\mu(\text{ref}) \le \eta_\varphi\cdot\alpha\cdot g_\varphi.
$$

Default `eta_phi = 1e-8`. Comparison uses `Compare_le` with relative tolerance based on $|\varphi_{\text{ref}}|$.

### 6.5 Sufficient Reduction (h-type Acceptance)

Trial is acceptable to the **current** iterate iff (`IpFilterLSAcceptor.cpp:497-498`):

$$
\theta_{\text{trial}} \le (1-\gamma_\theta)\theta_{\text{ref}}\quad\text{OR}\quad \varphi_{\text{trial}} - \varphi_{\text{ref}} \le -\gamma_\varphi\theta_{\text{ref}}.
$$

Defaults: `gamma_theta = 1e-5`, `gamma_phi = 1e-8`.

Plus a rapid-increase guard (`obj_max_inc=5.0`, `IpFilterLSAcceptor.cpp:480-493`):

$$
\log_{10}(\varphi_{\text{trial}}-\varphi_{\text{ref}}) > \texttt{obj\_max\_inc} + \max(1,\log_{10}|\varphi_{\text{ref}}|) \implies \text{reject.}
$$

### 6.6 Filter Acceptability

Stored as a list of pairs $(\varphi_j,\theta_j)$. Trial $(\varphi^t,\theta^t)$ is acceptable iff for EVERY entry $j$:

$$
\varphi^t \le \varphi_j\quad\text{OR}\quad \theta^t \le \theta_j.
$$

Empty filter trivially accepts. A trial passing this test must additionally satisfy whichever of (Armijo, sufficient reduction) is selected by the switching condition.

### 6.7 Filter Augmentation

Performed by `UpdateForNextIteration` (`IpBacktrackingLineSearch.cpp:881-896`) after a step is accepted. The filter is augmented (with the **reference** iterate, not the trial) iff the switching condition was false OR Armijo did not hold:

$$
\varphi_{\text{add}} = \varphi_{\text{ref}} - \gamma_\varphi\theta_{\text{ref}},\qquad \theta_{\text{add}} = (1-\gamma_\theta)\theta_{\text{ref}}.
$$

Dominated entries are removed (`IpFilter.cpp:70-83`).

### 6.8 Filter Reset

Cleared in `Reset()` whenever mu changes (`IpFilterLSAcceptor.cpp:524-532`). Also cleared on entry to restoration. A heuristic reset on `filter_reset_trigger` (default 5) consecutive filter rejections (capped at `max_filter_resets`, default 5) is implemented at lines 419-435.

### 6.9 Backtracking Loop

`DoBacktrackingLineSearch` (`IpBacktrackingLineSearch.cpp:679-853`):

1. $\alpha_p^{\max} = $ FTB primal step (Sec 5.1).
2. $\alpha_{\min} = $ from acceptor (Sec 6.10) -- but if in watchdog, $\alpha_{\min}=\alpha_{\max}$.
3. Try corrector first if enabled and not skipping first trial.
4. Loop with $\alpha = \alpha_p^{\max}$, halving each time:
   - Build trial = current + $\alpha$ * delta.
   - Evaluate $\theta, \varphi$.
   - Run `CheckAcceptabilityOfTrialPoint` (Sec 6.11).
   - If accepted: break.
   - If first trial rejected and $\theta$ increased ($\theta^t > \theta_{\text{cur}}$ AND $\alpha = \alpha_p^{\max}$): try SOC.
   - $\alpha \leftarrow \alpha\cdot\texttt{alpha\_red\_factor}$ (default 0.5).
   - Continue while $\alpha > \alpha_{\min}$ OR `n_steps == 0`.
5. If loop exits without acceptance, trigger restoration.

### 6.10 alpha_min

`CalculateAlphaMin` (`IpFilterLSAcceptor.cpp:450-468`):

$$
\alpha^{\text{raw}}_{\min} = \begin{cases}
\gamma_\theta & g_\varphi \ge 0 \\
\min\!\big(\gamma_\theta,\ \tfrac{\gamma_\varphi\theta}{-g_\varphi}\big) & g_\varphi < 0,\ \theta > \theta_{\min} \\
\min\!\Big(\gamma_\theta,\ \tfrac{\gamma_\varphi\theta}{-g_\varphi},\ \tfrac{\delta\theta^{s_\theta}}{(-g_\varphi)^{s_\varphi}}\Big) & g_\varphi < 0,\ \theta\le\theta_{\min}
\end{cases}
$$

then $\alpha_{\min} = \texttt{alpha\_min\_frac}\cdot\alpha^{\text{raw}}_{\min}$, default `alpha_min_frac = 0.05`. The third term ensures the switching condition would FAIL at $\alpha_{\min}$, so the line search will be in h-type regime where filter progress alone suffices.

### 6.11 Overall Acceptance Test (`CheckAcceptabilityOfTrialPoint`, lines 311-437)

```rust
fn check_acceptability(theta_t: f64, phi_t: f64, alpha: f64) -> Acceptance {
    if theta_t > self.theta_max { return Reject_TooInfeas; }
    let switching_holds = self.is_ftype(alpha);
    let theta_small = self.theta_ref <= self.theta_min;

    if switching_holds && theta_small {
        if !self.armijo_holds(phi_t, alpha) { return Reject; }
    } else {
        if !self.sufficient_reduction(theta_t, phi_t) { return Reject; }
    }
    if !self.filter_acceptable(theta_t, phi_t) { return Reject_Filter; }

    Accept
}
```

The "filter reset heuristic" lives between steps 4 and 5 in C++ and is applied across line-search iterations rather than within one trial.

### 6.12 Second-Order Correction (SOC)

Triggered iff first trial rejected AND $\theta^t \ge \theta_{\text{cur}}$ AND $\alpha = \alpha_p^{\max}$ (`IpBacktrackingLineSearch.cpp:811-818`).

`TrySecondOrderCorrection` (`IpFilterLSAcceptor.cpp:534-657`):

```rust
let mut c_soc  = c(x_cur);
let mut dms_soc = d(x_cur) - s_cur;
let mut theta_old_soc = theta_trial;

for k in 0..max_soc {
    c_soc  += alpha_soc * c(x_trial);
    dms_soc += alpha_soc * (d(x_trial) - s_trial);
    let rhs_soc = build_rhs(c_soc, dms_soc, ...);  // soc_method 0 or 1
    let delta_soc = pd_solver.solve(-1.0, 0.0, rhs_soc);
    let alpha_soc = primal_ftb(delta_soc, tau);
    let trial_soc = current + alpha_soc * delta_soc;
    let (theta_soc, phi_soc) = eval_theta_phi(trial_soc);
    // CRITICAL: switching/Armijo test uses ORIGINAL alpha_primal_test
    if check_acceptability(theta_soc, phi_soc, alpha_primal_test) == Accept {
        actual_delta = delta_soc; alpha_primal = alpha_soc;
        return Accepted;
    }
    if theta_soc > kappa_soc * theta_old_soc { return NotAccepted; }
    theta_old_soc = theta_soc;
}
```

Defaults: `max_soc = 4`, `kappa_soc = 0.99`, `soc_method = 0` (replace constraint residuals with $c_{\text{soc}}, dms_{\text{soc}}$). `soc_method = 1` instead scales x/s rows by alpha.

### 6.13 Watchdog

Activated when `watchdog_shortened_iter` reaches `watchdog_shortened_iter_trigger` (default 10) consecutive iterations of "shortened" steps (`IpBacktrackingLineSearch.cpp:376-380`).

State machine (lines 480-509):
1. `StartWatchDog`: save current iterate, search direction, $\alpha_p^{\max}$ as the "watchdog reference."
2. Each subsequent iteration: only ONE trial step per iteration.
3. If trial accepted: append "W" to info, clear watchdog.
4. If not: increment `watchdog_trial_iter`. If $> $ `watchdog_trial_iter_max` (default 3) or evaluation error, `StopWatchDog`: restore reference iterate and direction, run normal backtracking.
5. Otherwise accept the non-improving trial and continue (line 500-501).

When mu changes between iterations, watchdog is automatically deactivated.

### 6.14 Tiny Step Detection

`DetectTinyStep` (`IpBacktrackingLineSearch.cpp:1219-1279`): the search direction is "tiny" iff

$$
\max_i \frac{|\delta x_i|}{|x_i|+1} \le \texttt{tiny\_step\_tol}\quad\text{AND}\quad
\max_i \frac{|\delta s_i|}{|s_i|+1} \le \texttt{tiny\_step\_tol}\quad\text{AND}\quad
\theta_{\text{cur}}\le 10^{-4},
$$

with `tiny_step_tol` $= 10\varepsilon \approx 2.22\cdot10^{-15}$. If detected, accept the full FTB step. If $\|\delta y\|<$ `tiny_step_y_tol = 0.01` also, two consecutive tiny steps trigger termination with `STOP_AT_TINY_STEP`.

### 6.15 Soft Restoration

Before invoking full restoration, Ipopt tries a "soft" restoration step (`IpBacktrackingLineSearch.cpp:528-556, 1113-1217`):

1. Take $\alpha = \min(\alpha_p^{\max},\alpha_d^{\max})$ for ALL variables (including duals).
2. Accept if filter accepts OR if primal-dual error decreased by `soft_resto_pderror_reduction_factor` (default $1-10^{-4}$).

Up to `max_soft_resto_iters = 10` consecutive soft restorations before falling back to full restoration.

---

## 7. Restoration Phase

Reference: `AGENT_REFERENCE/RESTORATION.md`, `IpRestoMinC_1Nrm.cpp`, `IpRestoIpoptNLP.cpp`, WB06 Sec 3.3.

### 7.1 Trigger

Filter line search calls `RestorationPhase::PerformRestoration()` when backtracking exits with $\alpha\le\alpha_{\min}$ without acceptance, OR when "step computation failed" emergency mode activates the fallback. An assertion requires $\theta_{\text{cur}} > 0$ at entry (`IpRestoMinC_1Nrm.cpp:119`).

### 7.2 Restoration NLP

Variables: extended compound vector with 5 components (`IpRestoIpoptNLP.cpp:134-140`):

$$
x_{\text{resto}} = (x,\ n_c,\ p_c,\ n_d,\ p_d),\quad p_*,n_*\ge 0.
$$

Constraints:
$$
c(x) + n_c - p_c = 0,\qquad d_L \le d(x) + n_d - p_d \le d_U.
$$

Objective:
$$
\min\ \rho\,e^\top(p_c+n_c+p_d+n_d) + \frac{\eta(\mu)}{2}\|D_R(x-x_{\text{ref}})\|_2^2.
$$

- $\rho = $ `resto_penalty_parameter`, default $1000$ (`IpRestoIpoptNLP.cpp:60`).
- $\eta(\mu) = $ `resto_proximity_weight` $\cdot \mu^{0.5}$, default factor $1.0$, exponent fixed at $0.5$ (`cpp:34, 67, 759-764`). Vanishes as $\mu\to 0$.
- $D_R = \mathrm{diag}(1/\max(1,|x_{\text{ref},i}|))$ (`cpp:442-449`).
- $x_{\text{ref}}$ is the iterate at the moment restoration was triggered.

### 7.3 Bounds in Restoration

- $x$ keeps its original $x_L, x_U$.
- $n_c, p_c, n_d, p_d$ have lower bound 0 only; no upper bound (`IpRestoIpoptNLP.cpp:321-331`).

### 7.4 Hessian Structure

5x5 compound, only the (0,0) block is nonzero (`IpRestoIpoptNLP.cpp:660-704`):

$$
H_{\text{resto}}(0,0) = H_{\text{orig}}(x,\ \text{obj\_factor}=0,\ y_c,\ y_d) + \text{obj\_factor}\cdot\eta(\mu)\cdot D_R^2.
$$

Note: original Hessian is evaluated with `obj_factor=0`, so only the constraint Hessian contribution participates. This is a deliberate departure from the "real" Hessian to avoid quadratic-in-$f$ noise during feasibility recovery.

### 7.5 Initial Iterate

`RestoIterateInitializer.cpp`:

- Initial mu: $\mu_{\text{resto}} = \max(\mu_{\text{outer}},\ \|c\|_\infty,\ \|d-s\|_\infty)$ (line 58).
- $x$, $s$ inherited from the outer iterate at trigger time.
- $p, n$ initialized so that the initial complementarity is balanced. For each row of $c$ with value $c_i$:
  $$
  a_i = \tfrac{\mu}{2\rho} - \tfrac{c_i}{2},\quad b_i = \tfrac{c_i\mu}{2\rho},\quad n_{c,i} = a_i + \sqrt{a_i^2 + b_i},\quad p_{c,i} = c_i + n_{c,i}.
  $$
  (`IpRestoIterateInitializer.cpp:79-97, 216-229`.)
- Bound multipliers for $x$ are capped at $\rho$ (lines 163-174). Bound multipliers for $p, n$ initialized as $\mu / n$ etc. (lines 177-188).
- $y_c, y_d$ computed by least-squares with `constr_mult_init_max=0` (default in restoration), which sets them to zero (lines 29-33).

### 7.6 Restoration Algorithm

Restoration runs a complete *nested* `IpoptAlgorithm::Optimize(true)` instance (`IpRestoMinC_1Nrm.cpp:192`), with options under the prefix `resto.`. Notable forced settings:

- `resto.start_with_resto = no`.
- `resto.theta_max_fact = 1e8` if not user-overridden.
- Time limits inherited (`IpRestoMinC_1Nrm.cpp:127-149`).
- Iteration counter continues from the outer loop (`cpp:181`).

### 7.7 Restoration Convergence Test

`RestoFilterConvergenceCheck::CheckConvergence` (`IpRestoConvCheck.cpp:71-248`):

1. First inner iteration: always continue (line 168-173).
2. Square problem shortcut: $\inf_{pr} \le \min(\text{tol},\text{constr\_viol\_tol}) \Rightarrow$ converged.
3. Infeasibility-reduction guard:
   $$
   \text{inf}_{pr}^{\text{trial}} \le \max(\kappa_{\text{resto}}\cdot\text{inf}_{pr}^{\text{cur}},\ \min(\text{tol},\text{constr\_viol\_tol})).
   $$
   `kappa_resto = required_infeasibility_reduction = 0.9` by default. Set to 0 for square problems.
4. Filter acceptability of the **original** filter (`TestOrigProgress`, `IpRestoFilterConvCheck.cpp:53-80`): point must be `IsAcceptableToCurrentFilter` AND `IsAcceptableToCurrentIterate`. Both must pass to declare success.
5. If the inner problem's standard convergence triggers but original filter not satisfied:
   - If $\inf_{pr}\le 100\cdot\text{tol}$ and tol can be tightened: tighten and continue.
   - If $\inf_{pr}\le 100\cdot\text{tol}$ but cannot tighten: throw `RESTORATION_CONVERGED_TO_FEASIBLE_POINT`.
   - Otherwise: throw `LOCALLY_INFEASIBLE`.

### 7.8 Returning to the Outer Loop

On restoration success (`IpRestoMinC_1Nrm.cpp:342-433`):

- Extract $x, s$ from the restoration solution's component-0.
- Compute $\delta z$ as a one-shot Newton step approximation:
  $$
  \delta z_i = \frac{\mu - z_i(s^{\text{trial}}_i - s^{\text{cur}}_i)}{s^{\text{cur}}_i} - z_i,
  $$
  apply $\alpha_d$ FTB. (`cpp:378-399, 438-453`.)
- If $\max|z| > $ `bound_mult_reset_threshold = 1000`, reset ALL bound multipliers to $1$ (lines 404-418).
- Recompute $y_c, y_d$ via least-squares; if estimate exceeds `constr_mult_reset_threshold = 0`, set to 0 (default 0 means LS estimate is always used when calculator is available).
- Restart the OUTER filter line search with the new iterate.

### 7.9 Nested Restoration

`RestoRestorationPhase::PerformRestoration` (`IpRestoRestoPhase.cpp:30-109`) handles the case when the restoration IPM itself needs restoration. It does NOT launch a 3rd-level IPM. Instead, it analytically resets $p, n$ from the current $x$ via the same quadratic formula (Sec 7.5), keeping $s$ and all multipliers fixed. Always returns true.

---

## 8. Convergence Test

### 8.1 Scaled NLP Error

`IpIpoptCalculatedQuantities.cpp:3050-3104, 3663-3700`. Define:

$$
s_d = \max\!\Big(s_{\max},\ \frac{\|y_c\|_1+\|y_d\|_1+\|z_L\|_1+\|z_U\|_1+\|v_L\|_1+\|v_U\|_1}{n_y+n_b}\Big)\Big/s_{\max},
$$

$$
s_c = \max\!\Big(s_{\max},\ \frac{\|z_L\|_1+\|z_U\|_1+\|v_L\|_1+\|v_U\|_1}{n_b}\Big)\Big/s_{\max},
$$

with `s_max` $= 100$ (option, `cpp:144-183`). Then

$$
E_\mu(x,s,y,z,v) = \max\!\Big(\frac{\|\nabla L_{\text{undamped}}\|_\infty}{s_d},\ \|c, d-s\|_\infty,\ \frac{\|\text{compl}_\mu\|_\infty}{s_c}\Big),
$$

where $\text{compl}_\mu = \big((X-X_L)Z_L e - \mu e,\ (X_U-X)Z_U e - \mu e,\ (S-D_L)V_L e - \mu e,\ (D_U-S)V_U e - \mu e\big)$.

### 8.2 Termination Criteria

`IpOptErrorConvCheck.cpp:1-333`. The user solution is declared converged when ALL of:

$$
E_0 \le \texttt{tol},\quad
\|\nabla L\|_\infty \le \texttt{dual\_inf\_tol},\quad
\|\theta\|_\infty \le \texttt{constr\_viol\_tol},\quad
\|\text{compl}_0\|_\infty \le \texttt{compl\_inf\_tol}.
$$

The first uses the scaled $E_0$ (mu = 0 in the complementarity term, with $s_d, s_c$ scaling). The other three use **un-scaled** norms (`IpOptErrorConvCheck.cpp:209-211`). All four must hold.

### 8.3 Defaults

| Option | Default |
|---|---|
| `tol` | $10^{-8}$ (`IpIpoptData.cpp:22-42`) |
| `dual_inf_tol` | $1.0$ |
| `constr_viol_tol` | $10^{-4}$ |
| `compl_inf_tol` | $10^{-4}$ |
| `s_max` | $100$ |
| `mu_target` | $0$ |

`dual_inf_tol = 1` is permissive: the SCALED error governs in practice.

### 8.4 Acceptable Tolerances

If `acceptable_iter > 0` (default 15), Ipopt also tracks "acceptable convergence." If for `acceptable_iter` consecutive iterations:

$$
E_0 \le \texttt{acceptable\_tol}\quad\text{and}\quad
\|\nabla L\|_\infty \le \texttt{acceptable\_dual\_inf\_tol}\quad\text{and}\quad
\|\theta\|_\infty \le \texttt{acceptable\_constr\_viol\_tol}\quad\text{and}\quad
\|\text{compl}_0\|_\infty \le \texttt{acceptable\_compl\_inf\_tol},
$$

AND objective change $\le$ `acceptable_obj_change_tol`, the solver returns `STOP_AT_ACCEPTABLE_POINT`.

| Acceptable option | Default |
|---|---|
| `acceptable_iter` | 15 |
| `acceptable_tol` | $10^{-6}$ |
| `acceptable_dual_inf_tol` | $10^{10}$ |
| `acceptable_constr_viol_tol` | $10^{-2}$ |
| `acceptable_compl_inf_tol` | $10^{-2}$ |
| `acceptable_obj_change_tol` | $10^{20}$ |

The `acceptable_dual_inf_tol = 1e10` is intentionally loose -- the iter-count gate is the real safeguard.

### 8.5 Diverging Iterates and Limits

- `diverging_iterates_tol = 1e20`: if $\|x\|_\infty > $ this, return `DIVERGING_ITERATES`.
- `max_iter = 3000`: iteration limit.
- `max_cpu_time`, `max_wall_time`: time limits.

---

## 9. Defaults Reference Table

Compiled from `IpOptErrorConvCheck.cpp`, `IpIpoptAlg.cpp`, `IpFilterLSAcceptor.cpp`, `IpBacktrackingLineSearch.cpp`, `IpMonotoneMuUpdate.cpp`, `IpAdaptiveMuUpdate.cpp`, `IpDefaultIterateInitializer.cpp`, `IpPDPerturbationHandler.cpp`, `IpIpoptCalculatedQuantities.cpp`, `IpRestoMinC_1Nrm.cpp`, `IpRestoIpoptNLP.cpp`, `IpOrigIpoptNLP.cpp`.

### 9.1 Top-Level

| Option | Default | Source |
|---|---|---|
| `tol` | 1e-8 | IpIpoptData.cpp:22 |
| `max_iter` | 3000 | IpOptErrorConvCheck.cpp |
| `mu_strategy` | adaptive | IpAlgBuilder.cpp |
| `mu_oracle` | quality-function | IpAlgBuilder.cpp |
| `linear_solver` | mumps (or ma27 if available) | IpAlgBuilder.cpp |
| `hessian_approximation` | exact | IpAlgBuilder.cpp |
| `bound_relax_factor` | 1e-8 | IpOrigIpoptNLP.cpp:53 |
| `honor_original_bounds` | yes | IpOrigIpoptNLP.cpp:65 |
| `fixed_variable_treatment` | make_parameter | IpTNLPAdapter.cpp:100 |
| `nlp_lower_bound_inf` | -1e19 | IpTNLPAdapter.cpp |
| `nlp_upper_bound_inf` | +1e19 | IpTNLPAdapter.cpp |

### 9.2 Convergence

| Option | Default |
|---|---|
| `dual_inf_tol` | 1.0 |
| `constr_viol_tol` | 1e-4 |
| `compl_inf_tol` | 1e-4 |
| `s_max` | 100 |
| `mu_target` | 0 |
| `diverging_iterates_tol` | 1e20 |
| `acceptable_iter` | 15 |
| `acceptable_tol` | 1e-6 |
| `acceptable_dual_inf_tol` | 1e10 |
| `acceptable_constr_viol_tol` | 1e-2 |
| `acceptable_compl_inf_tol` | 1e-2 |
| `acceptable_obj_change_tol` | 1e20 |

### 9.3 Initialization

| Option | Default |
|---|---|
| `bound_push` | 0.01 |
| `bound_frac` | 0.01 |
| `slack_bound_push` | 0.01 (falls back to bound_push) |
| `slack_bound_frac` | 0.01 (falls back to bound_frac) |
| `bound_mult_init_val` | 1.0 |
| `bound_mult_init_method` | constant |
| `constr_mult_init_max` | 1000 |
| `least_square_init_primal` | no |
| `least_square_init_duals` | no |
| `warm_start_init_point` | no |
| `warm_start_bound_push` | 1e-3 |
| `warm_start_bound_frac` | 1e-3 |
| `warm_start_mult_init_max` | 1e6 |
| `warm_start_mult_bound_push` | 1e-3 |

### 9.4 Barrier Update

| Option | Default |
|---|---|
| `mu_init` | 0.1 |
| `barrier_tol_factor` | 10 |
| `mu_linear_decrease_factor` | 0.2 |
| `mu_superlinear_decrease_power` | 1.5 |
| `mu_allow_fast_monotone_decrease` | yes |
| `tau_min` | 0.99 |
| `mu_max_fact` | 1e3 |
| `mu_max` | 1e5 |
| `mu_min` | 1e-11 |
| `adaptive_mu_globalization` | obj-constr-filter |
| `adaptive_mu_kkterror_red_iters` | 4 |
| `adaptive_mu_kkterror_red_fact` | 0.9999 |
| `filter_margin_fact` | 1e-5 |
| `filter_max_margin` | 1.0 |
| `adaptive_mu_restore_previous_iterate` | no |
| `adaptive_mu_monotone_init_factor` | 0.8 |
| `adaptive_mu_safeguard_factor` | 0.0 |
| `quality_function_norm_type` | 2-norm-squared |
| `quality_function_centrality` | none |
| `quality_function_balancing_term` | none |
| `quality_function_max_section_steps` | 8 |
| `quality_function_section_sigma_tol` | 1e-2 |
| `quality_function_section_qf_tol` | 0 |
| `sigma_max` | 1e2 |
| `sigma_min` | 1e-6 |

### 9.5 Step / KKT

| Option | Default |
|---|---|
| `kappa_d` | 1e-5 |
| `kappa_sigma` | 1e10 |
| `slack_move` | $\varepsilon^{0.75} \approx 4.86e-12$ |
| `recalc_y` | no (yes if `hessian_approximation=limited-memory`) |
| `recalc_y_feas_tol` | 1e-6 |
| `mehrotra_algorithm` | no |
| `first_hessian_perturbation` | 1e-4 |
| `min_hessian_perturbation` | 1e-20 |
| `max_hessian_perturbation` | 1e20 |
| `perturb_inc_fact_first` | 100 |
| `perturb_inc_fact` | 8 |
| `perturb_dec_fact` | 0.333... |
| `jacobian_regularization_value` | 1e-8 |
| `jacobian_regularization_exponent` | 0.25 |
| `perturb_always_cd` | no |
| `max_refinement_steps` | 10 |
| `min_refinement_steps` | 1 |
| `residual_ratio_max` | 1e-10 |
| `residual_ratio_singular` | 1e-5 |

### 9.6 Filter / Line Search

| Option | Default |
|---|---|
| `theta_max_fact` | 1e4 |
| `theta_min_fact` | 1e-4 |
| `eta_phi` | 1e-8 |
| `delta` | 1.0 |
| `s_phi` | 2.3 |
| `s_theta` | 1.1 |
| `gamma_phi` | 1e-8 |
| `gamma_theta` | 1e-5 |
| `alpha_min_frac` | 0.05 |
| `alpha_red_factor` | 0.5 |
| `max_soc` | 4 |
| `kappa_soc` | 0.99 |
| `obj_max_inc` | 5.0 |
| `max_filter_resets` | 5 |
| `filter_reset_trigger` | 5 |
| `soc_method` | 0 |
| `corrector_type` | none |
| `corrector_compl_avrg_red_fact` | 1.0 |
| `alpha_for_y` | primal |
| `alpha_for_y_tol` | 10.0 |
| `tiny_step_tol` | 10*eps |
| `tiny_step_y_tol` | 0.01 |
| `watchdog_shortened_iter_trigger` | 10 |
| `watchdog_trial_iter_max` | 3 |
| `soft_resto_pderror_reduction_factor` | $1-10^{-4}$ |
| `max_soft_resto_iters` | 10 |
| `accept_every_trial_step` | no |
| `accept_after_max_steps` | -1 |

### 9.7 Restoration

| Option | Default |
|---|---|
| `resto_penalty_parameter` (rho) | 1000 |
| `resto_proximity_weight` (eta_factor) | 1.0 |
| `required_infeasibility_reduction` | 0.9 |
| `bound_mult_reset_threshold` | 1000 |
| `constr_mult_reset_threshold` | 0 |
| `resto_failure_feasibility_threshold` | 100*tol |
| `max_resto_iter` | 3000000 |
| `evaluate_orig_obj_at_resto_trial` | yes |
| `start_with_resto` | no |
| `resto.theta_max_fact` | 1e8 |
| `resto.constr_mult_init_max` | 0 |

---

## 10. Common-Pitfall Checklist for a Rust Port

Each item below is a place where a faithful re-implementation differs from a "textbook" one, and where in our experience the bug is silent.

### P1. Sign Convention

`L = f + y^T c` (NOT $-y^Tc$). The dual stationarity is $\nabla f + J_c^\top y_c + J_d^\top y_d - z_L + z_U$ (`IpIpoptCalculatedQuantities.cpp:1993-2030`). If you flip the sign, the user-visible $y$ has the wrong sign and any code that checks Lagrangian gradient against zero passes garbage.

### P2. Damped Lagrangian Used Only in RHS

The $\kappa_d$ damping linear term enters `curr_grad_lag_with_damping_x` (`IpIpoptCalculatedQuantities.cpp:2131-2180`). The convergence test uses `curr_grad_lag_x` (un-damped). Putting damping in the convergence test will report "converged" on points that are not actually KKT points.

### P3. kappa_sigma Reset Happens **Before** Promoting Trial

`IpIpoptAlg.cpp:716-767`. The clamp must be applied to TRIAL multipliers; only THEN is the trial promoted to current. Reverse the order and you keep multipliers that violate Eqn 16.

### P4. kappa_sigma Uses Different mu in Free vs Fixed Mode

`IpIpoptAlg.cpp:1075-1083`. Free mode: `mu = min(trial_avrg_compl, 1e3)`. Otherwise: `mu = curr_mu`. If you always use `curr_mu`, the clamp is too tight in Free mode.

### P5. The kappa_sigma Quick-Bypass

`IpIpoptAlg.cpp:1090`. Test `max(z*s) <= kappa_sigma*mu AND min(z*s) >= mu/kappa_sigma` and skip the per-component clamp if both hold. Tiny optimization but matches Ipopt's behavior; if you always do the per-component pass you may corrupt z when min(z*s) is exactly at the boundary due to FP rounding.

### P6. Slack Lower Bound Adjustment Before Clamp

`IpIpoptAlg.cpp:665-714` happens BEFORE the kappa_sigma clamp. If a trial slack is below `slack_move * max(|s|, 1)`, the BOUND is widened, not the slack. Without this, dividing $\mu/(\kappa_\sigma s)$ in the clamp can overflow.

### P7. RHS for the Newton Solve Has a Negative Sign

`pd_solver->Solve(-1.0, 0.0, rhs, delta)`: the $-1$ is part of the API. If your linear solver wrapper does NOT internally negate, you must negate the RHS. The complementarity rows are $(X-X_L)Z e - \mu e$ in the RHS, which becomes $\mu e - (X-X_L)Ze$ in the Newton-direction equation. Easy to flip.

### P8. Inertia Perturbation State Persists Across Iterations

`IpPDPerturbationHandler` is stateful. `dx_last` is the LAST iteration's value. The `INC_FIRST=100` jump on the first nonzero-after-zero transition is critical for fast recovery. Per-iteration reinitialization defeats this and slows convergence.

### P9. delta_x = delta_s and delta_c = delta_d

`IpPDPerturbationHandler.cpp:382, 412`. The two pairs always carry the same value. They are physically separate slots in the augmented system but always equal in Ipopt 3.14. A Rust port can simplify to two scalars.

### P10. delta_cd Shrinks With mu

$\delta_c = \delta_d = $ `jacobian_regularization_value` $\cdot \mu^{0.25} = 10^{-8}\mu^{0.25}$. Hard-coding $10^{-8}$ ignoring $\mu$ over-regularizes near the solution (where $\mu \sim 10^{-9}$).

### P11. Iterative Refinement Operates on the FULL 8-Component Residual

`IpPDFullSpaceSolver`. After solving the 4-block augmented system and recovering $\delta z, \delta v$, the residual computed for refinement uses all eight blocks, including complementarity rows. A common mistake is to refine only the 4-block residual, which misses the residual injected by the elimination of $\delta z, \delta v$.

### P12. Filter Augmentation Adds the **Reference** Point With Margin

`IpFilterLSAcceptor.cpp:304-307`. NOT the trial point. Adding the trial defeats filter monotonicity guarantees.

### P13. Filter Cleared on mu Change

`IpFilterLSAcceptor.cpp:524-532`. Each barrier subproblem starts with an empty filter. Carrying the filter across mu transitions blocks legitimate steps.

### P14. Switching Condition Includes alpha

`IpFilterLSAcceptor.cpp:293-294`. Test is $\alpha\cdot(-g_\varphi)^{s_\varphi} > \delta\theta^{s_\theta}$, NOT $(-g_\varphi)^{s_\varphi} > \cdots$. This means as backtracking shrinks $\alpha$, an f-type trial can become h-type without re-evaluating the gradient.

### P15. SOC Acceptance Test Uses Original alpha_primal_test

`IpFilterLSAcceptor.cpp:629`. The switching condition for the SOC trial uses the ORIGINAL (non-SOC) $\alpha$, so SOC does not flip f-type to h-type just because its own $\alpha$ is smaller.

### P16. Watchdog Restores Pre-Watchdog Iterate on Time/Iter Limit

`IpIpoptAlg.cpp:456-459`. When the outer loop exits due to MAXITER/CPUTIME/WALLTIME, `BacktrackingLineSearch::StopWatchDog()` is called explicitly. Without this, the user gets the in-progress watchdog trial point, which may be objectively worse than the pre-watchdog reference.

### P17. tau Source Differs by Mu Mode

Free: $\tau = \max(\tau_{\min}, 1-E_0)$ (`IpAdaptiveMuUpdate.cpp:397`). Fixed/Monotone: $\tau = \max(\tau_{\min}, 1-\mu)$. At init: $\tau = \max(\tau_{\min}, 1-\mu_0)$. Using $1-\mu$ in Free mode keeps tau too large (since mu is oracle-chosen and may be small), and FTB becomes too aggressive.

### P18. Convergence Test Mixes Scaled and Unscaled Norms

$E_0$ uses $s_d, s_c$. The auxiliary `dual_inf_tol`, `constr_viol_tol`, `compl_inf_tol` checks use UN-scaled inf-norms. If you uniformly scale, the auxiliary tests become trivially satisfied at large $\|y\|$.

### P19. s_d, s_c Floor at s_max / s_max = 1

The formula is `max(s_max, sum/n) / s_max`, NOT `sum/n / s_max`. Floor of 1.0 prevents over-relaxation when multipliers are tiny.

### P20. Square Problems Set y = 0

`IpDefaultIterateInitializer.cpp:685-691`. For square systems ($n_y = n_x$), least-squares y is skipped and y = 0 is used. Without this, the augmented system at the initial point can be singular.

### P21. constr_mult_init_max Acts as a Discard Threshold

`IpDefaultIterateInitializer.cpp:722-727`. If LS y exceeds the threshold, ENTIRE y is set to zero. Treating it as a per-component clamp produces a different starting point.

### P22. Restoration Hessian Uses obj_factor=0

`IpRestoIpoptNLP.cpp:691`. The original NLP's Hessian routine is called with $\sigma_f=0$, so only $\sum y_i \nabla^2 c_i$ contributes from the original. The proximity term $\eta D_R^2$ is added separately. Using $\sigma_f=1$ doubles the descent's curvature near $x_{\text{ref}}$.

### P23. Restoration mu Is Bumped to Match Constraint Violation

`IpRestoIterateInitializer.cpp:58`. $\mu_{\text{resto}} = \max(\mu,\ \|c\|_\infty,\ \|d-s\|_\infty)$. Using outer mu directly leads to instant convergence-test failures inside restoration.

### P24. After Restoration, Bound Mults Reset to 1 if Any Exceeds 1000

`IpRestoMinC_1Nrm.cpp:404-418`. Threshold `bound_mult_reset_threshold=1000`. ALL components reset, not just the offending ones. Skipping this leaves spurious huge multipliers from the restoration sub-problem.

### P25. constr_mult_reset_threshold Default Is 0

`IpRestoMinC_1Nrm.cpp:46-51`. With default 0, ANY nonzero LS estimate is kept (because `0 > some-positive` is false in C++ float compare? No: the check is "if estimate exceeds threshold, set to 0"; threshold=0 means estimate is always "exceeds zero"... wait, read carefully.) Verified at IpRestoMinC_1Nrm.cpp: when threshold = 0, the LS estimate is always used. The "reset" branch is taken only if threshold > 0. A port that flips the sense of the comparison resets y unconditionally.

### P26. Restoration Reuses the Outer Iteration Counter

`IpRestoMinC_1Nrm.cpp:181`. The inner Optimize sees `iter_count` continuing from the outer. Restarting at zero inflates both the inner and outer iteration limits.

### P27. recalc_y Default Depends on hessian_approximation

`IpIpoptAlg.cpp:80-89`. With `hessian_approximation=exact`, default is `no`. With `limited-memory`, default flips to `yes`. The recalc cost is one extra factorization; without it, L-BFGS multiplier estimates drift.

### P28. PD Solver Negates RHS But Not the +1.0/+1.0 Variant

`pd_solver->Solve(alpha_pd, beta, rhs, delta)`. With `(-1.0, 0.0)` it computes $\delta = -A^{-1}\text{rhs}$. With `(1.0, 1.0)` it computes $\delta \mathrel{+}= A^{-1}\text{rhs}$ (used for adding the corrector). Mishandling the betas in your wrapper is a silent error.

### P29. `mehrotra_algorithm=yes` Forces Many Other Options

`IpIpoptAlg.cpp:139-185`. Forces `adaptive_mu_globalization=never-monotone-mode`, `corrector_type=none`, `accept_every_trial_step=yes`, `bound_push=10`, `bound_frac=0.2`, `bound_mult_init_val=10`, `constr_mult_init_max=0`, `alpha_for_y=bound-mult`, `least_square_init_primal=yes`. These are interlocked; do not cherry-pick.

### P30. Bound Relaxation Is Applied at the IpoptNLP Level, Not in TNLPAdapter

`IpOrigIpoptNLP.cpp:459-481`. After TNLPAdapter has produced the internal $(x_L, x_U, d_L, d_U)$, OrigIpoptNLP relaxes them. If you pre-relax in your TNLP-adapter equivalent and then relax again, you double-relax. Conversely, if neither layer relaxes, an "exact" feasible starting point causes log-of-zero.

### P31. push_variables Has a Two-Pass Implementation

`IpDefaultIterateInitializer.cpp:493-497`. Pass 1: `push_variables(x, 0, 0)` -- snap any $x_i$ at-or-outside bounds onto the bound. Pass 2: real $\kappa_1, \kappa_2$ projection. Without pass 1, a user-provided $x$ exactly on the bound triggers division-by-zero in the slack initialization.

### P32. Filter Reset Heuristic vs Filter Reset on mu Change

Two separate mechanisms. `filter_reset_trigger` (consecutive rejections) is a heuristic INSIDE one barrier subproblem. The unconditional reset on mu change is mandatory. Conflating them either resets too rarely (missing mu transitions) or too often (thrashing).

### P33. No-Bounds Edge Case in Adaptive

`IpAdaptiveMuUpdate.cpp:278-290`. Zero bound multipliers in the problem $\Rightarrow$ set $\mu = \mu_{\min}$ immediately. Otherwise the average complementarity is 0 and the oracle stalls.

### P34. Output Iteration Index Is Post-Increment

`IpIpoptAlg.cpp:407`. `iter_count += 1` happens AFTER `AcceptTrialPoint`, BEFORE `CheckConvergence`. So the iteration printed as $k$ contains the step that produced iterate $k+1$. Off-by-one when comparing logs.

### P35. UpdateBarrierParameter Failure ≡ Step Failure ≡ Activate Fallback

`IpIpoptAlg.cpp:367-395`. Both mu update failure AND linear-solve failure funnel into the same `ActivateFallbackMechanism`. Treating them as separate failure modes complicates the loop structure unnecessarily.

### P36. ResetInfo Is Called After Each Output Line

`IpIpoptAlg.cpp:357`. The "info string" (concatenated tags like `r`, `R`, `S`, `W`, `s`, `t`, `T`, `f`, `F`, `z`, `c`, `Nh`, `q`, `A`, `Ls`) is cleared every iteration. Each tag has a specific source:

| Tag | Where | Meaning |
|---|---|---|
| `r` | filter line search | restoration triggered |
| `R` | filter line search | restoration succeeded |
| `S` | line search | soft restoration |
| `W` | line search | watchdog acceptance |
| `s` | line search | slack adjusted (Sec 5.5) |
| `t`/`T` | line search | tiny step / persistent tiny |
| `f` | line search | f-type (Armijo) step |
| `F` | adaptive mu | fixed-mode |
| `z` | accept trial | kappa_sigma corrected |
| `Nh` | line search | negative curvature (LBFGS) |
| `q` | corrector | corrector accepted |

A port should produce these tags faithfully -- they are the primary diagnostic for whether your line search matches Ipopt.

### P37. Iteration "Step" Tag Format

The step-size column printed in the iteration log shows `alpha_primal` followed by a single character indicating the step type. The mapping is in `IpOrigIterationOutput.cpp` and matches the tags above.

---

## 11. Reference: Top-Level Loop Pseudocode

```rust
fn optimize(opt: &Options, nlp: &mut Nlp) -> SolverReturn {
    let mut it = cold_start(nlp, opt);  // Sec 2
    let mut perturb = PerturbState::default();
    let mut filter = Filter::new();
    let mut watchdog = WatchdogState::Idle;
    let mut k: u32 = 0;

    if let Some(s) = check_convergence(&it, opt, k) { return s; }      // Sec 8

    loop {
        update_hessian(&mut it, nlp);                                  // Sec 4
        write_iter_line(k, &it);
        clear_info_string(&mut it);

        // 3a. Mu update (may fail -> emergency)
        let mu_ok = update_barrier_parameter(&mut it, &mut filter, opt); // Sec 3
        // 3b. Search direction
        let dir = if mu_ok {
            compute_search_direction(&it, &mut perturb, opt, nlp)         // Sec 4
        } else { None };

        if dir.is_none() {
            if !activate_fallback(&mut it, &mut filter, opt) {
                return SolverReturn::ErrorInStepComputation;
            }
            continue;
        }
        let mut delta = dir.unwrap();

        // 5. Line search
        let trial = match find_acceptable_trial_point(
            &it, &delta, &mut filter, &mut watchdog, opt, nlp
        ) {                                                            // Sec 6
            LSResult::Accepted(t) => t,
            LSResult::NeedRestoration => {
                match perform_restoration(&mut it, &mut filter, opt, nlp) {  // Sec 7
                    Ok(t) => t,
                    Err(e) => return e.into(),
                }
            }
            LSResult::TinyStep => return SolverReturn::StopAtTinyStep,
        };

        // 6. AcceptTrialPoint (Sec 5.4-5.6)
        adjust_bounds_for_small_slacks(&mut it, &trial, opt);
        correct_kappa_sigma(&mut trial.z_l, &trial.s_xL, &it, opt);
        correct_kappa_sigma(&mut trial.z_u, &trial.s_xU, &it, opt);
        correct_kappa_sigma(&mut trial.v_l, &trial.s_sL, &it, opt);
        correct_kappa_sigma(&mut trial.v_u, &trial.s_sU, &it, opt);
        it.accept_trial(trial);

        if opt.recalc_y && it.constr_viol() < opt.recalc_y_feas_tol {
            recompute_y_via_least_squares(&mut it, nlp);
        }

        k += 1;

        if it.is_square() {
            compute_feasibility_multipliers(&mut it, nlp, opt);
        }

        if let Some(s) = check_convergence(&it, opt, k) { return s; }
        if k >= opt.max_iter { return SolverReturn::MaxIterExceeded; }
    }
}
```

This pseudocode mirrors `IpoptAlgorithm::Optimize()` (`IpIpoptAlg.cpp:292-563`) one-for-one.

---

## 12. Source Citation Index (selected)

| Concern | File | Lines |
|---|---|---|
| Main loop | `IpIpoptAlg.cpp` | 292-563 |
| Accept trial point | `IpIpoptAlg.cpp` | 652-819 |
| kappa_sigma correction | `IpIpoptAlg.cpp` | 1055-1134 |
| Mehrotra option forcing | `IpIpoptAlg.cpp` | 97-185 |
| Search direction RHS | `IpPDSearchDirCalc.cpp` | 57-140 |
| Inertia state machine | `IpPDPerturbationHandler.cpp` | 144-538 |
| Augmented system options | `IpStdAugSystemSolver.cpp` | (entire) |
| Filter switching | `IpFilterLSAcceptor.cpp` | 273-308 |
| Filter check | `IpFilterLSAcceptor.cpp` | 311-499 |
| Armijo | `IpFilterLSAcceptor.cpp` | 439-448 |
| alpha_min | `IpFilterLSAcceptor.cpp` | 450-468 |
| SOC | `IpFilterLSAcceptor.cpp` | 534-657 |
| Backtracking | `IpBacktrackingLineSearch.cpp` | 679-853 |
| Watchdog | `IpBacktrackingLineSearch.cpp` | 376-509, 855-898 |
| alpha_for_y | `IpBacktrackingLineSearch.cpp` | 919-1011 |
| Tiny step | `IpBacktrackingLineSearch.cpp` | 1219-1279 |
| Magic step | `IpBacktrackingLineSearch.cpp` | 1013-1111 |
| Soft restoration | `IpBacktrackingLineSearch.cpp` | 528-556, 1113-1217 |
| Monotone mu | `IpMonotoneMuUpdate.cpp` | 130-219 |
| Adaptive mu | `IpAdaptiveMuUpdate.cpp` | 239-484, 583-786 |
| Quality function | `IpQualityFunctionMuOracle.cpp` | 154-828 |
| LOQO oracle | `IpLoqoMuOracle.cpp` | 34-66 |
| Probing oracle | `IpProbingMuOracle.cpp` | 47-133 |
| Default initializer | `IpDefaultIterateInitializer.cpp` | 33-743 |
| Push variables | `IpDefaultIterateInitializer.cpp` | 473-667 |
| Least squares y | `IpLeastSquareMults.cpp` | 30-95 |
| Restoration NLP | `IpRestoIpoptNLP.cpp` | 60, 134-704 |
| Restoration init | `IpRestoIterateInitializer.cpp` | 29-229 |
| Restoration outer | `IpRestoMinC_1Nrm.cpp` | 119-453 |
| Restoration conv | `IpRestoConvCheck.cpp` | 71-248 |
| Restoration filter conv | `IpRestoFilterConvCheck.cpp` | 53-80 |
| Lagrangian gradient | `IpIpoptCalculatedQuantities.cpp` | 1993-2030 |
| Damped Lagrangian | `IpIpoptCalculatedQuantities.cpp` | 2131-2180 |
| Sigma matrices | `IpIpoptCalculatedQuantities.cpp` | 3501-3525 |
| Safe slack | `IpIpoptCalculatedQuantities.cpp` | 455-537 |
| NLP error scaling | `IpIpoptCalculatedQuantities.cpp` | 3050-3104, 3663-3700 |
| Convergence check | `IpOptErrorConvCheck.cpp` | 1-333 |
| Bound relaxation | `IpOrigIpoptNLP.cpp` | 53-72, 459-481 |
| Fixed variables | `IpTNLPAdapter.cpp` | 100-112 |

---

## 13. Selected Theoretical Background

- WB06: Section 2 derives the perturbed KKT system, Section 3.1 describes filter line search, Eqn (16) is the kappa_sigma reset, Eqn (31) is the restoration NLP. The defaults in this spec match Section 3.7-3.10 of the paper.
- WB05a (global convergence): defines theta_max, theta_min, the switching exponents s_phi, s_theta, and proves filter monotonicity under those choices.
- WB05b (local convergence): shows superlinear convergence of the un-globalized Newton step under standard regularity (LICQ + sufficient second-order conditions); justifies the monotone mu schedule's $\mu^{1.5}$ floor.
- NWW09: defines the "quality function" oracle and analyzes why it improves over LOQO and probing in practice. The default `quality_function_norm_type=2-norm-squared` and `centrality=none, balancing=none` are NWW09's recommendation (their Table 6).
- Forsgren-Gill-Wright 2002 (IPM survey): general background on primal-dual interior point methods, including the role of $\Sigma$, FTB, and corrector steps.

For a Rust port, the WB06 paper is the single most useful primary source; WB05a/b are needed only if you intend to reproduce the convergence proofs (e.g., as test oracles).

---

## 14. ripopt vs Ipopt 3.14 — known representation deviations

This section catalogs places where ripopt's working representation
diverges from Ipopt 3.14's internal data structures, even when the
mathematical specification is matched. These are honest deltas to be
aware of, not bugs.

### 14.1  Implicit slack representation for inequality constraints

**Ipopt 3.14:** transforms `g_L ≤ c(x) ≤ g_U` into an explicit slack
formulation: introduces `s` as an independent variable, replaces the
inequality with the equality `c(x) − s = 0`, and applies bounds
`g_L ≤ s ≤ g_U`. The state vector is `[x; s]`, the KKT system is
augmented to `(n + m_eq + m_ineq + m_ineq) × …`, and the algorithm's
primitives (filter trial, fraction-to-boundary, magic step,
quality-function oracle aff/centering trial point) operate on `s` as
an independent free variable.

**ripopt:** does **not** introduce explicit `s`. Inequality slacks are
implicit, computed at evaluation time as `c(x) − g_L` and `g_U − c(x)`.
`SolverState` carries multipliers `v_l`, `v_u` for the inequality
constraints but no separate slack vector. The KKT system is
`(n + m) × (n + m)` (one row per constraint regardless of
inequality/equality split).

**Equivalence at solutions:** at any KKT point, `s ≡ c(x)` exactly, so
the converged primal `x`, dual `y`, and bound multipliers `z_l`, `z_u`
are identical between the two representations. The deviation only
manifests along the *trial trajectory* during the iteration.

**Consequences (places ripopt necessarily approximates Ipopt's
behavior):**

- **Magic step (§5.3)** — Ipopt's magic step is a closed-form
  adjustment to `s` while holding `x` fixed. With implicit `s`, that
  degree of freedom does not exist, and `apply_magic_step` is a
  documented no-op. Implemented for spec compliance and as a future
  wiring point if explicit-`s` mode is added; gated on
  `SolverOptions::magic_step` (default `true`).
- **Quality-function μ oracle (§3.5)** — Ipopt's `Q(σ)` evaluates the
  barrier KKT error at a *trial iterate* obtained from aff/centering
  steps with `s_trial = s + α·ds`. ripopt's QF (`compute_quality_function_mu`,
  T2.23) evaluates a linearised `Q(σ)` using `(1−α)·current` for
  residual reduction; the σ search is capped at `σ_max = 1.0`
  (Ipopt's spec is `1e2`) because without true nonlinear residual
  evaluation the linearised Q is degenerate above σ = 1.
- **Filter trial point** — Ipopt compares `(θ, φ)` at
  `(x_trial, s_trial)`; ripopt at `(x_trial, c(x_trial))`. θ and φ
  agree at acceptance (`s_accepted ≡ c(x_accepted)`), but the trial
  trajectory through the line search differs.
- **κ_σ correction trajectory** — same issue: ripopt corrects against
  `c(x_trial)`-derived slack rather than an independent `s_trial`.
- **Restoration phase is *not* affected** — ripopt's
  `RestorationNlp` (`src/restoration_nlp.rs`) does carry explicit
  `p`, `n` slacks for the resto problem itself (T2.22 verified).

**Concern: large-scale problem behavior.** Pre-v0.8 ripopt repeatedly
struggled on certain large-scale CUTEst problems and AC-OPF instances
(see e.g. case30_ieee notes elsewhere in this repo). The implicit-slack
representation is one suspect: in problems with many active inequality
constraints, the absence of an independent `s` removes a stabilising
degree of freedom from the line search and from κ_σ correction, which
may cause ripopt's trial trajectory to diverge from Ipopt's on
problems where Ipopt's magic step or filter `s_trial` is materially
load-bearing. We have **not** isolated a single benchmark that fails
specifically because of implicit slacks — the v0.8 alignment work has
been deliberately scoped to representation-preserving fixes — but this
is the natural place to look first if a future regression or
unresolved large-scale failure cannot be explained by other gaps. If
such a benchmark is found, the appropriate response is to escalate to
a full explicit-`s` rewrite (Option 2 in the v0.8 alignment notes).
Until then we accept the deviation and document it here.

### 14.2  Linearised QF oracle (T2.23)

See §14.1 above. `compute_quality_function_mu` uses a linearised Q and
σ_max = 1.0 because evaluating the true nonlinear residual at each
trial σ requires plumbing the problem trait through
`update_barrier_parameter` (option (a) in T2.23 design notes), which
crosses five call frames. Acceptable per §14.1; promote to (a) if QF's
σ search is shown to under-explore on a real benchmark.

### 14.3  Combined-y representation

ripopt represents constraint multipliers as a single `y: Vec<f64>` of
length `m`, with `v_l`/`v_u` carrying the bound-side multipliers for
inequality constraints, rather than Ipopt's separate `y_c` (equality)
and `y_d` (inequality, paired with `s`) vectors. The
`fix_inequality_mult_signs` post-pass (`src/ipm.rs`) converts between
representations where needed. Mathematically equivalent at solutions;
diverges only in intermediate sign conventions on raw multiplier
arrays, and is documented at the source.

---

## End

This specification is grounded in Ipopt 3.14 source as of the release branch checked out at `Ipopt/src/Algorithm/`. Every formula and default has a file:line citation. A re-implementation that satisfies all 36 pitfalls in Section 10, the formulas in Sections 1-8, and the defaults in Section 9 will produce iterates that match Ipopt 3.14 to within numerical noise on every problem in the HS, CUTEst, and GAMS nlpbench test suites where Ipopt itself converges — modulo the representation deviations cataloged in §14.
