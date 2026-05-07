/// Treatment of variables with `x_L[i] == x_U[i]` (fixed variables).
///
/// Mirrors Ipopt 3.14's `fixed_variable_treatment` (`TNLPAdapter`).
/// Both Ipopt and ripopt default to `MakeParameter` (Phase 11 alignment):
/// fixed variables are physically removed from the optimization before
/// the IPM sees them. ripopt achieves this via the preprocessor wrapper
/// `PreprocessedProblem::new_fixed_only`, which eliminates fixed vars
/// without performing full preprocessing (no redundant-constraint
/// detection, no bound tightening). The IPM then operates on the
/// reduced problem and `state.x` has length `n_full − n_fixed`,
/// matching Ipopt's TNLPAdapter `MAKE_PARAMETER` mode.
///
/// `RelaxBounds` is the pre-Phase-11 legacy default; it widens
/// `[x_L, x_U]` by ±1e-8·max(|c|, 1) so the IPM has a non-empty
/// interior on the fixed slot, paying one degree of freedom per fixed
/// variable. Retained for backward compatibility and as a fallback for
/// callers that don't want preprocessor-layer elimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FixedVariableTreatment {
    /// Widen `[x_L, x_U]` by ±1e-8·max(|c|, 1) around the fixed value
    /// `c = x_L = x_U` so the IPM has a non-empty interior. Adds one
    /// degree of freedom per fixed variable.
    RelaxBounds,
    /// Remove fixed variables from the working problem before solving.
    /// Routes the user's NLP through `PreprocessedProblem::new_fixed_only`,
    /// which eliminates fixed vars without performing full preprocessing
    /// (no redundant-constraint detection, no bound tightening). Mirrors
    /// Ipopt 3.14's TNLPAdapter `make_parameter` mode.
    MakeParameter,
}

impl Default for FixedVariableTreatment {
    fn default() -> Self {
        // Phase 11: align default with Ipopt 3.14's TNLPAdapter
        // (`fixed_variable_treatment = make_parameter`).
        Self::MakeParameter
    }
}

/// Bound multiplier initialization method.
///
/// Mirrors Ipopt 3.14's `bound_mult_init_method`
/// (`IpDefaultIterateInitializer.cpp:254-288`). Ipopt's default is
/// `Constant` with `bound_mult_init_val = 1.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum BoundMultInitMethod {
    /// `z_l = z_u = bound_mult_init_val` for every finite bound.
    /// Ipopt default.
    Constant,
    /// `z_l = μ_init / (x − x_l)`, `z_u = μ_init / (x_u − x)`.
    /// Mirrors Ipopt's `mu-based`. Pre-v0.8 ripopt default.
    MuBased,
}

impl Default for BoundMultInitMethod {
    fn default() -> Self {
        Self::Constant
    }
}

/// Choice of linear solver for the KKT system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum LinearSolverChoice {
    /// Direct multifrontal LDL^T (default). Exact solve, provides inertia.
    Direct,
    /// Iterative MINRES with incomplete LDL^T preconditioner.
    /// Suitable for large problems where direct factorization is too expensive.
    Iterative,
    /// Hybrid: starts with direct solver, switches to iterative if the direct
    /// solver fails or becomes too slow (factorization > 1 second).
    /// Switches back to direct if MINRES fails to converge repeatedly.
    Hybrid,
}

impl Default for LinearSolverChoice {
    fn default() -> Self {
        Self::Direct
    }
}

/// Step length used to update the equality multipliers `y` after a
/// Newton step. Mirrors Ipopt 3.14 `alpha_for_y` option
/// (`IpBacktrackingLineSearch.cpp:84-104` and the closed-form 1D
/// minimizer at `:969-998`). All seven Ipopt modes are implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum AlphaForY {
    /// `alpha_y = alpha_primal` (Ipopt default).
    #[default]
    Primal,
    /// `alpha_y = alpha_dual_max` (the bound-multiplier step length).
    BoundMult,
    /// `alpha_y = min(alpha_primal, alpha_dual_max)`.
    Min,
    /// `alpha_y = max(alpha_primal, alpha_dual_max)`.
    Max,
    /// `alpha_y = 1.0` (full step, ignoring fraction-to-boundary).
    Full,
    /// Use full step when alpha_primal exceeds the alpha_for_y_tol gate;
    /// fall back to alpha_primal otherwise.
    PrimalAndFull,
    /// Use full step when alpha_dual_max exceeds the gate; else
    /// alpha_dual_max.
    DualAndFull,
    /// Closed-form 1D minimizer of `||grad_lag(trial, y + α·dy)||²`
    /// over α ∈ [0, 1]. Costs one fresh grad_f + Jacobian evaluation
    /// at the trial point per accepted step. Mirrors Ipopt's
    /// `min_dual_infeas` (`IpBacktrackingLineSearch.cpp:969-996`).
    MinDualInfeas,
    /// Same minimizer as `MinDualInfeas` but clipped to
    /// `[min(α_p, α_d), max(α_p, α_d)]` instead of `[0, 1]`. Mirrors
    /// Ipopt's `safer_min_dual_infeas` (line 992) — guarantees α_y is
    /// bracketed between the primal and dual step lengths.
    SaferMinDualInfeas,
}

/// NLP scaling method. Mirrors Ipopt 3.14 `nlp_scaling_method`
/// registered in `IpAlgBuilder.cpp:343-353` and dispatched in
/// `IpAlgBuilder.cpp:678-696`. The `Equilibration` variant (Curtis-Reid
/// via Harwell MC19) is omitted — ripopt has no MC19 binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum NlpScalingMethod {
    /// No automatic scaling. `obj_scaling_factor` is still applied
    /// multiplicatively (matches Ipopt's `NoNLPScalingObject` inheriting
    /// from `StandardScalingBase`).
    None,
    /// Compute `obj_scaling`/`g_scaling` from the gradient and Jacobian
    /// at the initial point (Ipopt default). Mirrors `GradientScaling::
    /// DetermineScalingParametersImpl` (`IpGradientScaling.cpp:69-233`).
    #[default]
    Gradient,
    /// Use scaling values supplied via `user_obj_scaling`,
    /// `user_g_scaling`, `user_x_scaling`. Mirrors Ipopt's
    /// `UserScaling` which forwards to `TNLP::get_scaling_parameters`.
    User,
}

/// Solver options matching Ipopt defaults.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SolverOptions {
    /// Convergence tolerance for optimality.
    pub tol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Initial barrier parameter.
    pub mu_init: f64,
    /// Minimum barrier parameter.
    pub mu_min: f64,
    /// Multiplier on the initial average complementarity used as the
    /// upper bound on adaptive μ (`μ_max = mu_max_fact * initial_avg_compl`).
    /// Mirrors Ipopt 3.14 `mu_max_fact` (default `1000`,
    /// `IpAdaptiveMuUpdate.cpp:267-273`). The cap is applied to both
    /// the LOQO and quality-function oracles. ripopt computes
    /// `initial_avg_compl` lazily on the first Free-mode call. When
    /// adaptive mode is off this option has no effect.
    pub mu_max_fact: f64,
    /// Maximum permitted log10 increase in the barrier objective per line
    /// search trial step. Mirrors Ipopt 3.14 `obj_max_inc` (default `5.0`,
    /// `IpFilterLSAcceptor.cpp:132-139`). The trial is rejected when the
    /// trial barrier objective `phi_trial` exceeds the reference and
    /// `log10(phi_trial - phi_ref) > obj_max_inc + basval`, where
    /// `basval = max(1.0, log10(|phi_ref|))`. Catches NaN-adjacent or
    /// blow-up trials that the filter alone may admit on degenerate
    /// problems.
    pub obj_max_inc: f64,
    /// Multiplier on the line-search minimum step size. Mirrors Ipopt 3.14
    /// `alpha_min_frac` (default `0.05`, `IpFilterLSAcceptor.cpp:113, 222,
    /// 468`). Ipopt computes
    /// `alpha_min_frac * min(gamma_theta, gamma_phi*theta/(-gBD), [δ*θ^sθ/(-gBD)^sφ])`
    /// and rejects line searches that backtrack below it (entry to
    /// restoration). Smaller values let backtracking continue further; the
    /// Ipopt default 0.05 means restoration triggers after ~13 halvings.
    pub alpha_min_frac: f64,
    /// T3.10: number of consecutive accepted steps whose preceding
    /// line search ended with a filter-based rejection that triggers a
    /// filter reset. Mirrors Ipopt 3.14 `filter_reset_trigger`
    /// (default 5, `IpFilterLSAcceptor.cpp:142`).
    pub filter_reset_trigger: u32,
    /// T3.10: maximum number of times the filter may be reset across
    /// the solve. Mirrors Ipopt 3.14 `max_filter_resets` (default 5;
    /// 0 disables the heuristic, `IpFilterLSAcceptor.cpp:152`).
    pub max_filter_resets: u32,
    /// DEV-36: scaling factor for the filter `theta_min` initialization
    /// from the initial constraint violation. Mirrors Ipopt 3.14
    /// `theta_min_fact` (default `1e-4`, `IpFilterLSAcceptor.cpp:118`).
    /// `theta_min = theta_min_fact * max(1, theta_init)`.
    pub theta_min_fact: f64,
    /// DEV-36: scaling factor for the filter `theta_max` initialization
    /// from the initial constraint violation. Mirrors Ipopt 3.14
    /// `theta_max_fact` (default `1e4`, `IpFilterLSAcceptor.cpp:120`).
    /// `theta_max = theta_max_fact * max(1, theta_init)`.
    pub theta_max_fact: f64,
    /// Fraction-to-boundary parameter minimum.
    pub tau_min: f64,
    /// Barrier parameter reduction factor (monotone mode).
    pub mu_linear_decrease_factor: f64,
    /// Barrier parameter superlinear decrease power.
    pub mu_superlinear_decrease_power: f64,
    /// Print level (0 = silent, 5 = verbose).
    pub print_level: u8,
    /// Outward relaxation factor applied to every finite variable
    /// bound `x_l`/`x_u` and constraint bound `g_l`/`g_u` at problem
    /// setup, before `bound_push` and `relax_fixed_variable_bounds`.
    /// Mirrors Ipopt 3.14's `bound_relax_factor` (default `1e-8`).
    /// For each finite bound `b`, the relaxation magnitude is
    /// `min(constr_viol_tol, bound_relax_factor * max(|b|, 1.0))`.
    /// Set to `0.0` to disable.
    pub bound_relax_factor: f64,
    /// Linear damping coefficient for variables with exactly one finite
    /// bound (XOR). Mirrors Ipopt 3.14's `kappa_d` (default `1e-5`).
    /// Adds `+ kappa_d * mu * slack` to the barrier objective and
    /// `± kappa_d * mu` to the corresponding row of the dual residual
    /// (Newton RHS), keeping one-sided-bounded variables from drifting
    /// to infinity along the barrier's unbounded direction. Convergence
    /// (`||r_d||_inf`) is computed from the UN-damped gradient.
    /// Set to `0.0` to disable.
    pub kappa_d: f64,
    /// Bound push for initial point (kappa_1 in Ipopt).
    pub bound_push: f64,
    /// Bound fraction for initial point (kappa_2 in Ipopt).
    pub bound_frac: f64,
    /// Slack bound push.
    pub slack_bound_push: f64,
    /// Slack bound fraction.
    pub slack_bound_frac: f64,
    /// Constraint violation tolerance for convergence.
    pub constr_viol_tol: f64,
    /// Dual infeasibility tolerance for convergence.
    pub dual_inf_tol: f64,
    /// Complementarity tolerance for convergence.
    pub compl_inf_tol: f64,
    /// Use adaptive barrier parameter update (vs monotone).
    pub mu_strategy_adaptive: bool,
    /// Maximum number of second-order correction steps.
    pub max_soc: usize,
    /// Warm-start initialization enabled.
    pub warm_start: bool,
    /// Warm-start bound push.
    pub warm_start_bound_push: f64,
    /// Warm-start bound fraction.
    pub warm_start_bound_frac: f64,
    /// Warm-start multiplier initial value.
    pub warm_start_mult_bound_push: f64,
    /// Target barrier parameter for warm-start initialization. When
    /// `warm_start = true` and this is `Some(mu)`, the IPM begins at
    /// `state.mu = mu` regardless of `mu_init`. Mirrors Ipopt's
    /// `warm_start_target_mu` (default in Ipopt: -1.0, meaning use
    /// `mu_init`). Set this to your previous solve's final `mu` to
    /// resume parametric / MPC sweeps without recentering through a
    /// large mu first. Default: `None` (falls through to `mu_init`).
    pub warm_start_target_mu: Option<f64>,
    /// Any bound less than this value is treated as -infinity (no bound).
    /// Set to a finite value to add artificial lower bounds on unbounded variables.
    pub nlp_lower_bound_inf: f64,
    /// Any bound greater than this value is treated as +infinity (no bound).
    /// Set to a finite value to add artificial upper bounds on unbounded variables.
    pub nlp_upper_bound_inf: f64,
    /// Adaptive barrier parameter divisor (kappa in mu = avg_compl / kappa).
    /// Higher values reduce mu faster. Default: 10.0.
    pub kappa: f64,
    /// Allow the adaptive barrier rule to increase mu when complementarity is large
    /// (e.g., after restoration or stall recovery). Default: true.
    pub mu_allow_increase: bool,
    /// Use least-squares estimate for initial constraint multipliers.
    ///
    /// Mirrors the basic LS-y init that Ipopt 3.14's
    /// `IpDefaultIterateInitializer::SetInitialIterates` runs at
    /// `cpp:340 → least_square_mults() (cpp:669-743)`. This LS solve
    /// runs **unconditionally** in Ipopt as long as
    /// `constr_mult_init_max > 0` (default 1000); the stronger
    /// `least_square_init_duals` option (`cpp:102-111`, default `no`)
    /// is a separate, additional LS solve that re-initializes z and v
    /// as well — that one is *not* what this flag controls.
    ///
    /// With `true`, ripopt solves `min ‖∇f − z_L + z_U + J^T y‖²`; the
    /// estimate is rejected if `‖y_LS‖_∞ > constr_mult_init_max`, in
    /// which case y is set to 0. Disabling this flag leaves y at the
    /// constant `v_U − v_L` post-correction (≈ ±1 for one-sided
    /// inequalities), which produces a 1000× larger iter-0 dual
    /// residual than Ipopt on problems like Mittelmann arki0003 where
    /// the Jacobian has columns with O(1e3) coefficients summed across
    /// inequality rows.
    ///
    /// Default: **true** (matches Ipopt 3.14's cold-start behavior).
    pub least_squares_mult_init: bool,
    /// Maximum absolute value for LS multiplier init; if exceeded, fall back to zero. Default: 1000.0.
    pub constr_mult_init_max: f64,
    /// Threshold for resetting equality / inequality multipliers after the
    /// restoration phase (mirrors Ipopt 3.14's `constr_mult_reset_threshold`,
    /// `IpRestoMinC_1Nrm.cpp:46-51`, default **0.0**).
    ///
    /// On a successful restoration exit, Ipopt calls
    /// `DefaultIterateInitializer::least_square_mults` with this option as
    /// the magnitude cap. The LS branch (`IpDefaultIterateInitializer.cpp:692`)
    /// requires `cap > 0.0`; with the default 0.0 the function falls through
    /// to the `else` branch (`cpp:734-737`) and **sets y_c = y_d = 0**.
    ///
    /// ripopt mirrors this: with `≤ 0.0`, `recompute_y_after_restoration`
    /// zeros y unconditionally (no LS solve). Setting this >0 allows the
    /// LS estimate to be kept when `‖y_LS‖_∞ ≤ threshold`. The Ipopt
    /// default of 0.0 is the principled handoff because the resto inner
    /// y's are meaningless to the parent (they solve a different
    /// stationarity), and a non-zero LS y at a poorly-scaled restored
    /// iterate biases the parent's first Newton direction — observed on
    /// arki0003 as a post-restoration dual-residual blow-up.
    pub constr_mult_reset_threshold: f64,
    /// Include constraint slack log-barriers in the filter merit function. Default: true.
    pub constraint_slack_barrier: bool,
    /// Maximum wall-clock time in seconds. 0.0 means no limit.
    pub max_wall_time: f64,
    /// Number of consecutive shortened steps before activating watchdog. Default: 10.
    pub watchdog_shortened_iter_trigger: usize,
    /// Maximum trial iterations during watchdog mode. Default: 3 (Ipopt).
    pub watchdog_trial_iter_max: usize,
    /// KKT dimension threshold for switching to sparse solver.
    /// When n + m >= sparse_threshold, use sparse LDLT instead of dense.
    /// Default: 100.
    pub sparse_threshold: usize,
    /// Barrier tolerance factor for fixed-mode mu decrease. Default: 10.0.
    pub barrier_tol_factor: f64,
    /// Allow multiple consecutive μ decreases per outer iteration in
    /// monotone (Fixed) mode when the barrier subproblem is already
    /// solved at the new μ. Mirrors Ipopt 3.14's
    /// `mu_allow_fast_monotone_decrease` (default `yes`). Capped at
    /// 4 consecutive decreases per iteration to bound work.
    /// Set to `false` to revert to one decrease per iteration.
    pub mu_allow_fast_monotone_decrease: bool,
    /// Initial factor for mu in fixed mode: mu = this * avg_compl. Default: 0.8.
    pub adaptive_mu_monotone_init_factor: f64,
    /// On a Free→Fixed mode switch, restore the most recent iterate that
    /// was accepted while in Free mode (snapshot taken whenever
    /// `CheckSufficientProgress` returned true). Mirrors Ipopt option
    /// `adaptive_mu_restore_previous_iterate` (default `false`,
    /// `IpAdaptiveMuUpdate.cpp:175,308-311,362-370`). When `false` the
    /// switch keeps the current iterate; only mu/tau change.
    pub adaptive_mu_restore_previous_iterate: bool,
    /// Maximum iterations for restoration NLP subproblem. Default: 200.
    pub restoration_max_iter: usize,
    /// Disable NLP restoration (prevents recursion in inner solve). Default: false.
    pub disable_nlp_restoration: bool,
    /// Required infeasibility reduction for restoration to be declared a
    /// success. Mirrors Ipopt's `required_infeasibility_reduction` option
    /// (a.k.a. `kappa_resto`) used by `IpRestoFilterConvCheck::CheckProgress`
    /// (`IpRestoConvCheck.cpp:71-248`, spec §7.7). Restoration accepts a
    /// trial point when
    ///   `theta_trial <= max(kappa_resto * theta_entry, min(tol, constr_viol_tol))`
    /// Default: 0.9. Set to 0.0 to require true feasibility (Ipopt's
    /// square-problem convention; ripopt's `RestorationPhase` enforces
    /// this automatically when `is_square`).
    pub kappa_resto: f64,
    /// Restoration proximity weight (Ipopt's `resto_proximity_weight`). The
    /// restoration objective is min ρ·(p+n) + (η/2)·||D_R(x − x_R)||² with
    /// η = `resto_proximity_weight` · √μ. Default: 1.0 (Ipopt 3.14).
    pub resto_proximity_weight: f64,
    /// Enable preprocessing to internally solve and remove auxiliary equality
    /// systems, eliminate fixed variables, and remove redundant constraints.
    /// Default: true.
    pub enable_preprocessing: bool,
    /// Maximum accepted residual for internal auxiliary equality-system solves
    /// during preprocessing. Default: 1e-8.
    pub auxiliary_tol: f64,
    /// Detect linear constraints and skip their Hessian contribution.
    /// Default: true.
    pub detect_linear_constraints: bool,
    /// Use L-BFGS Hessian approximation inside the IPM instead of exact Hessian.
    /// When enabled, `hessian_structure()` and `hessian_values()` are never called.
    /// Equivalent to Ipopt's `hessian_approximation = "limited-memory"`.
    /// Default: false.
    pub hessian_approximation_lbfgs: bool,
    /// Enable Mehrotra predictor-corrector for barrier parameter selection.
    ///
    /// After factoring the KKT system, solves an affine-scaling predictor step
    /// (μ=0 in the RHS) to probe the Newton direction. From this probe, computes
    /// a better centering parameter σ = (μ_aff/μ)³ and rebuilds the corrector RHS
    /// with μ_new = σ·μ. Costs one extra triangular solve (not a re-factorization).
    ///
    /// Expected effect: 20–40% fewer iterations on convex-like problems.
    /// Applies to the full sparse/dense KKT path.
    /// Default: false (matches Ipopt 3.14's `mehrotra_algorithm = no`).
    pub mehrotra_pc: bool,
    /// Maximum number of Gondzio multiple centrality corrections per iteration.
    ///
    /// After the main (Mehrotra-corrected) direction is computed, performs up to
    /// this many additional centrality corrections. Each correction uses the same
    /// factored KKT matrix (one backsolve each) to drive outlier complementarity
    /// pairs (z·s far from μ) back toward the central path.
    ///
    /// Set to 0 to disable. Typical value: 3.
    /// Default: 3.
    pub gondzio_mcc_max: usize,
    /// Choice of sparse linear solver. Default: Direct.
    pub linear_solver: LinearSolverChoice,
    /// Recompute equality multipliers `y` via the augmented least-squares
    /// system after each accepted step once the iterate is sufficiently
    /// feasible (`||c||_∞ < recalc_y_feas_tol`). Mirrors Ipopt 3.14
    /// `IpIpoptAlg.cpp:652-819` step 5 (spec §5 / P27). The effective
    /// runtime gate is `(recalc_y || hessian_approximation_lbfgs)` so
    /// that L-BFGS callers get the recompute by default — without it,
    /// quasi-Newton multiplier estimates drift.
    /// Default: `false` (auto-on with L-BFGS Hessian).
    pub recalc_y: bool,
    /// Constraint-violation threshold below which `recalc_y` actually
    /// recomputes y. Above this threshold, recomputing y from a
    /// least-squares fit to a non-feasible iterate hurts more than it
    /// helps (the LS multipliers absorb infeasibility into y, biasing
    /// the Newton direction).
    /// Default: 1e-6.
    pub recalc_y_feas_tol: f64,
    /// Step length policy for the equality multipliers `y` after a Newton
    /// step. Mirrors Ipopt 3.14 `alpha_for_y` (`IpBacktrackingLineSearch.cpp:84-104`).
    /// Default: `Primal` (the Ipopt default).
    pub alpha_for_y: AlphaForY,
    /// Threshold for `PrimalAndFull` / `DualAndFull` modes: take the full
    /// step on `y` when the gating step length exceeds this tolerance,
    /// else fall back to that step length. Mirrors Ipopt
    /// `alpha_for_y_tol` (default 10.0).
    pub alpha_for_y_tol: f64,
    /// Tolerance on the relative dual-step `‖Δy‖_∞ / (1 + ‖y‖_∞)` below
    /// which a tiny x-step is allowed to *latch* the tiny-step flag.
    /// Mirrors Ipopt 3.14 `tiny_step_y_tol` (`IpBacktrackingLineSearch.cpp:421-424`).
    /// Default: 1e-2.
    pub tiny_step_y_tol: f64,
    /// Use quality function for barrier parameter selection in adaptive mode.
    /// Evaluates Q(mu) = barrier KKT error for several candidate mu values and
    /// picks the minimizer. Allows more aggressive mu decreases than the Loqo
    /// oracle when the iterate is well-centered.
    /// Default: true.
    pub mu_oracle_quality_function: bool,
    /// Add a centrality penalty term `1 / xi` to the quality function,
    /// where `xi = min(z·s) / avg(z·s)` is evaluated at the candidate mu.
    /// Mirrors `IpQualityFunctionMuOracle` `centrality=reciprocal` mode
    /// (`IpQualityFunctionMuOracle.cpp:622`). Off by default, matching
    /// Ipopt 3.14's default of `centrality=none`. Enable on problems
    /// where the iterate drifts off-center and the plain quality
    /// function picks aggressive small-mu candidates.
    /// Default: false.
    pub quality_function_centrality: bool,
    /// Maximum number of golden-section steps used by the Quality Function
    /// μ oracle (`IpQualityFunctionMuOracle.cpp:548`,
    /// `quality_function_max_section_steps`). Each step adds one extra
    /// linearised Q(σ) evaluation; Ipopt's default is 8 and that's plenty
    /// for the [1e-6, 1e2] log-bracket because each shrink eliminates
    /// ~38% of the bracket width.
    /// Default: 8.
    pub quality_function_max_section_steps: usize,
    /// NLP scaling method. Mirrors Ipopt 3.14 `nlp_scaling_method`
    /// (`IpAlgBuilder.cpp:343-353`). Default `Gradient`.
    /// When set to `User`, the solver uses values supplied in
    /// `user_obj_scaling` / `user_g_scaling` / `user_x_scaling`.
    /// For backwards compatibility, supplying any of those user-scaling
    /// fields while `nlp_scaling_method = Gradient` (the default) still
    /// short-circuits the gradient algorithm.
    pub nlp_scaling_method: NlpScalingMethod,
    /// Multiplicative factor applied on top of the computed objective
    /// scale (mirrors Ipopt 3.14 `obj_scaling_factor`,
    /// `IpNLPScaling.cpp:236-243, 276`). Default `1.0`. A negative value
    /// is the canonical "maximize" idiom: the IPM still minimizes
    /// `obj_scaling * f(x)`, so `obj_scaling_factor = -1.0` flips the
    /// sense. Applied even when `nlp_scaling_method = None`.
    pub obj_scaling_factor: f64,
    /// Target gradient magnitude used by gradient-based scaling
    /// (`nlp_scaling_max_gradient`, `IpGradientScaling.cpp:18-27`). When
    /// `||grad_f(x0)||_inf` exceeds this, the objective is scaled down by
    /// `nlp_scaling_max_gradient / max_grad`. Same threshold is applied
    /// per-row to the constraint Jacobian. Default `100.0`.
    pub nlp_scaling_max_gradient: f64,
    /// Floor applied after computing gradient-based scales
    /// (`nlp_scaling_min_value`, `IpGradientScaling.cpp:46-54`). Default
    /// `1e-8`. Protects against tiny scales when the initial gradient is
    /// astronomical.
    pub nlp_scaling_min_value: f64,
    /// Per-objective override (`nlp_scaling_obj_target_gradient`,
    /// `IpGradientScaling.cpp:28-36`). When `> 0`, unconditionally sets
    /// `obj_scaling = target / max_grad` (skipping the
    /// `max_grad > nlp_scaling_max_gradient` gate). Default `0.0`
    /// (disabled).
    pub nlp_scaling_obj_target_gradient: f64,
    /// Per-constraint override (`nlp_scaling_constr_target_gradient`,
    /// `IpGradientScaling.cpp:37-45`). When `> 0`, every constraint row
    /// receives the same scale `target / max(row_amax)` regardless of
    /// the threshold. Default `0.0` (disabled).
    pub nlp_scaling_constr_target_gradient: f64,
    /// User-provided objective scaling factor. When `Some`, bypasses automatic
    /// gradient-based scaling and uses this value directly.
    pub user_obj_scaling: Option<f64>,
    /// User-provided constraint scaling factors (length m). When `Some`, bypasses
    /// automatic gradient-based constraint scaling.
    pub user_g_scaling: Option<Vec<f64>>,
    /// User-provided variable scaling factors (length n). When
    /// `Some(dx)`, the solver wraps the NLP with `XScaledProblem` and
    /// runs the IPM in the internal coordinate `x' = D_x · x` (where
    /// `D_x = diag(dx)`), then unscales the result on return
    /// (`x_user = x' / dx`, `z_L_user = dx · z_L_internal`,
    /// `z_U_user = dx · z_U_internal`; constraint multipliers and
    /// constraint values are invariant). Mirrors Ipopt 3.14
    /// `IpScaledNLP` / `IpStandardScalingBase` restricted to
    /// x-scaling; objective and constraint scaling are independent
    /// (see `user_obj_scaling`, `user_g_scaling`). Entries must be
    /// strictly positive and finite; invalid input returns
    /// `SolveStatus::InternalError`. `None` or `Some(vec![])` is a
    /// pass-through (use automatic gradient-based scaling).
    pub user_x_scaling: Option<Vec<f64>>,
    /// Initial constraint multipliers for warm starting.
    pub warm_start_y: Option<Vec<f64>>,
    /// Initial lower-bound multipliers for warm starting.
    pub warm_start_z_l: Option<Vec<f64>>,
    /// Initial upper-bound multipliers for warm starting.
    pub warm_start_z_u: Option<Vec<f64>>,
    /// B9: optional initial constraint slack iterate `s` for warm
    /// starting. When supplied (and `warm_start = true`), values are
    /// copied into `state.s` after the default slack-push initializer
    /// has run; equality rows are kept at their sentinel `s = g_l`.
    /// Out-of-bound entries are projected back into a strict interior
    /// of `[g_l, g_u]` so the IPM's barrier remains well-defined.
    pub warm_start_s: Option<Vec<f64>>,
    /// B9: optional initial constraint-slack lower-bound multipliers
    /// `v_l` for warm starting. Mirrors `warm_start_z_l`.
    pub warm_start_v_l: Option<Vec<f64>>,
    /// B9: optional initial constraint-slack upper-bound multipliers
    /// `v_u` for warm starting. Mirrors `warm_start_z_u`.
    pub warm_start_v_u: Option<Vec<f64>>,
    /// T0.12 (Ipopt 3.14 alignment): enable iterative refinement on the
    /// full augmented KKT system in `kkt::solve_for_direction`. When
    /// `false`, only a single backsolve is performed (no refinement
    /// loop). Default: `true`, matching Ipopt's `IpPDFullSpaceSolver`
    /// behavior of always running at least
    /// `iterative_refinement_steps_required` IR steps.
    pub use_ic_refinement: bool,
    /// T0.12: minimum iterative-refinement steps required per KKT
    /// solve. Mirrors Ipopt's `min_refinement_steps` (default 1):
    /// always perform this many IR iterations even when the residual
    /// is already small. Additional IR steps run up to the internal
    /// `max_refinement_steps` cap when the residual exceeds the
    /// acceptance threshold. Default: 1.
    pub iterative_refinement_steps_required: usize,
    /// T-MIT-F: hard cap on IR steps per KKT solve. Mirrors Ipopt's
    /// `max_refinement_steps` (`IpPDFullSpaceSolver.cpp:48-54`,
    /// default 10).
    pub max_refinement_steps: usize,
    /// T-MIT-F: residual-ratio acceptance threshold for the IR loop.
    /// IR continues while `residual_ratio > residual_ratio_max`.
    /// Mirrors Ipopt's `residual_ratio_max`
    /// (`IpPDFullSpaceSolver.cpp:55-62`, default 1e-10).
    pub residual_ratio_max: f64,
    /// T-MIT-F: residual-ratio threshold above which an IR-failed
    /// solve is declared singular and the perturbation handler is
    /// asked to step the singular branch. Below this, an IR-stalled
    /// solve is accepted ("S" code in Ipopt's iteration log).
    /// Mirrors `residual_ratio_singular`
    /// (`IpPDFullSpaceSolver.cpp:63-70`, default 1e-5).
    pub residual_ratio_singular: f64,
    /// T-MIT-F: stagnation factor for the IR give-up condition. IR
    /// gives up when iters > min and either iters > max or
    /// `residual_ratio > improvement_factor * residual_ratio_old`.
    /// Mirrors Ipopt's `residual_improvement_factor`
    /// (`IpPDFullSpaceSolver.cpp:72-79`, default 0.999999999).
    pub residual_improvement_factor: f64,
    /// T3.23: when true, after the augmented (4-block) IR loop converges,
    /// expand `(dx, ds, dy_c, dy_d)` to the full primal-dual step
    /// `(dx, ds, dy_c, dy_d, dz_L, dz_U, dv_L, dv_U)` via the analytic
    /// Fiacco recovery and re-check the residual on the unsymmetric
    /// 8-block system (in ripopt's condensed augmented system the
    /// v-blocks are already eliminated, leaving the 6 stationarity
    /// + Jacobian + z-complementarity rows). If the residual ratio
    /// exceeds `residual_ratio_max`, perform ONE extra back-solve
    /// correction. Mirrors Ipopt 3.14 `IpPDFullSpaceSolver::ComputeResiduals`
    /// (lines 666-793).
    ///
    /// Default: `false`. **Honest finding**: in ripopt's condensed
    /// augmented formulation the rows the 8-block check would catch
    /// (z-complementarity rows 5-6) are zero by construction in exact
    /// arithmetic — the augmented system uses the same Fiacco recovery
    /// formula that produces them. Floating-point cancellation in
    /// `(μ − z·s)/s` is the only source of nonzero residual, and that
    /// can't be improved by another augmented back-solve (which uses
    /// the same elimination). Plumbing lives in
    /// `kkt::solve_for_direction_with_ir_full` for a future
    /// unsymmetric-IR pathway; the IPM main loop currently does not
    /// dispatch to it. Flip to `true` only when an unsymmetric KKT
    /// solver lands.
    pub ir_residual_full_8_block: bool,
    /// T3.25 follow-up: enable the factorization cache (`dummy_cache_`
    /// analog from `IpPDFullSpaceSolver.cpp:430-450`). When `true`, the
    /// IPM threads a single `FactorCache` through every
    /// `factor_with_inertia_correction_cached` call site (main step,
    /// QF mu oracle, condensed-fallback retries) so that consecutive
    /// solves whose 13-tag dependency fingerprint matches skip the
    /// underlying supernodal-LDLᵀ refactor and replay the cached
    /// `(δ_w, δ_c)`.
    ///
    /// Default: `false`. Plumbing landed in T3.25; the cache is opt-in
    /// pending benchmark validation. Cache hits are bit-identical with
    /// the no-cache path by construction (the underlying solver still
    /// holds the matching factorization), so flipping this on is
    /// expected to be a pure win for problems where multiple solves
    /// per iteration share a matrix.
    pub factor_cache_enabled: bool,
    /// If set, serialize each KKT matrix (main IPM loop only) to this directory
    /// after factorization. Writes two files per iteration:
    ///   `<kkt_dump_name>_<iter:04>.mtx`  — Matrix Market format, symmetric, lower triangle
    ///   `<kkt_dump_name>_<iter:04>.json` — Metadata: n, m, iteration, rhs, inertia, status
    /// The directory is created with `create_dir_all` if it does not exist.
    /// No-op when `None` (default). IO errors are logged as warnings, never abort the solve.
    pub kkt_dump_dir: Option<std::path::PathBuf>,
    /// Problem name used in dump filenames. Defaults to `"problem"`.
    pub kkt_dump_name: String,
    /// Threshold below which a primal slack `x[i] - x_l[i]` (or its
    /// upper-bound mirror) is considered too small. After every
    /// accepted iterate, undersized slacks are widened by nudging the
    /// corresponding *bound* outward to
    /// `min(max(mu/z, slack_move), slack_move * max(1, |bound|))`.
    /// Mirrors Ipopt 3.14's `slack_move` option. Default
    /// `f64::EPSILON.powf(0.75) ≈ 1.83e-12`.
    pub slack_move: f64,
    /// Enable Ipopt 3.14's "magic step" — a closed-form per-component
    /// adjustment of the inequality-constraint slack vector `s` after a
    /// step is accepted (`IpBacktrackingLineSearch.cpp:1013-1111`). The
    /// magic step pushes `s_i` toward `d(x)_i` only along the side(s)
    /// allowed by the slack's bounds (`d_L`, `d_U`), holding `x` and all
    /// multipliers fixed. For doubly-bounded slacks it suppresses the
    /// step if it would not reduce centering against `d_L + d_U`.
    ///
    /// Architectural note (T2.24): ripopt uses an implicit-slack
    /// formulation (see `.crucible/wiki/concepts/implicit-slack-formulation.org`),
    /// so it has no explicit slack vector `s` distinct from `x`. With no
    /// slack to adjust, the magic step has no degree of freedom to act
    /// on in the standard solve path; `apply_magic_step` is therefore a
    /// no-op there. The flag is exposed for spec compliance and for
    /// future explicit-slack paths (see `slack_formulation.rs`).
    /// Default: `true`, matching Ipopt 3.14's `magic_steps` default.
    pub magic_step: bool,
    /// Method used to initialize the bound multipliers `z_l`, `z_u`.
    /// Default: `Constant` (Ipopt 3.14 default).
    pub bound_mult_init_method: BoundMultInitMethod,
    /// Initial value used when `bound_mult_init_method = Constant`.
    /// Default: 1.0 (Ipopt 3.14 default).
    pub bound_mult_init_val: f64,
    /// Treatment of fixed variables (`x_L[i] == x_U[i]`).
    /// Default: `RelaxBounds`. `MakeParameter` activates fixed-var
    /// elimination via `PreprocessedProblem::new_fixed_only` regardless
    /// of `enable_preprocessing`.
    pub fixed_variable_treatment: FixedVariableTreatment,
    /// Acceptable-level relative objective-change gate. The acceptable
    /// status only fires when `|f_k - f_{k-1}| / max(1, |f_k|) ≤
    /// acceptable_obj_change_tol`. Default 1e20 disables the gate
    /// (matches Ipopt 3.14 `IpOptErrorConvCheck.cpp:115`).
    pub acceptable_obj_change_tol: f64,
    /// Number of consecutive iterations meeting the acceptable
    /// thresholds required to terminate with `Acceptable`. Setting to
    /// `0` disables the acceptable termination entirely
    /// (`IpOptErrorConvCheck.cpp:241`). Default `15` matches Ipopt 3.14.
    pub acceptable_iter: usize,
    /// Threshold on `‖x‖_∞` above which the iterate is declared
    /// diverging. Default 1e20 matches Ipopt
    /// `IpOptErrorConvCheck.cpp:123`.
    pub diverging_iterates_tol: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 3000,
            mu_init: 0.1,
            mu_min: 1e-11,
            mu_max_fact: 1000.0,
            obj_max_inc: 5.0,
            alpha_min_frac: 0.05,
            filter_reset_trigger: 5,
            max_filter_resets: 5,
            theta_min_fact: 1e-4,
            theta_max_fact: 1e4,
            tau_min: 0.99,
            mu_linear_decrease_factor: 0.2,
            mu_superlinear_decrease_power: 1.5,
            print_level: 5,
            bound_relax_factor: 1e-8,
            kappa_d: 1e-5,
            bound_push: 1e-2,
            bound_frac: 1e-2,
            slack_bound_push: 1e-2,
            slack_bound_frac: 1e-2,
            constr_viol_tol: 1e-4,
            dual_inf_tol: 1.0,
            compl_inf_tol: 1e-4,
            // A8.11.1: Ipopt's default `mu_strategy = "monotone"`
            // (`IpAlgBuilder.cpp:355-362`). ripopt previously defaulted
            // to `adaptive` which mismatches Ipopt's iter-0 mu trajectory
            // on hard problems (arki0003 stayed at lg(mu)=-1.0 with
            // monotone, oscillated under adaptive). Switch to monotone.
            mu_strategy_adaptive: false,
            max_soc: 4,
            warm_start: false,
            warm_start_bound_push: 1e-3,
            warm_start_bound_frac: 1e-3,
            warm_start_mult_bound_push: 1e-3,
            warm_start_target_mu: None,
            nlp_lower_bound_inf: -1e19,
            nlp_upper_bound_inf: 1e19,
            kappa: 10.0,
            mu_allow_increase: true,
            least_squares_mult_init: true,
            constr_mult_init_max: 1000.0,
            constr_mult_reset_threshold: 0.0,
            constraint_slack_barrier: true,
            max_wall_time: 0.0,
            watchdog_shortened_iter_trigger: 10,
            // Ipopt default `watchdog_trial_iter_max = 3`
            // (`IpBacktrackingLineSearch.cpp:170-176`).
            watchdog_trial_iter_max: 3,
            sparse_threshold: 110,
            barrier_tol_factor: 10.0,
            mu_allow_fast_monotone_decrease: true,
            adaptive_mu_monotone_init_factor: 0.8,
            adaptive_mu_restore_previous_iterate: false,
            restoration_max_iter: 200,
            disable_nlp_restoration: false,
            kappa_resto: 0.9,
            resto_proximity_weight: 1.0,
            enable_preprocessing: true,
            auxiliary_tol: 1e-8,
            detect_linear_constraints: true,
            hessian_approximation_lbfgs: false,
            mehrotra_pc: false,
            gondzio_mcc_max: 3,
            linear_solver: LinearSolverChoice::default(),
            recalc_y: false,
            recalc_y_feas_tol: 1e-6,
            alpha_for_y: AlphaForY::default(),
            alpha_for_y_tol: 10.0,
            tiny_step_y_tol: 1e-2,
            mu_oracle_quality_function: true,
            quality_function_centrality: false,
            quality_function_max_section_steps: 8,
            nlp_scaling_method: NlpScalingMethod::Gradient,
            obj_scaling_factor: 1.0,
            nlp_scaling_max_gradient: 100.0,
            nlp_scaling_min_value: 1e-8,
            nlp_scaling_obj_target_gradient: 0.0,
            nlp_scaling_constr_target_gradient: 0.0,
            user_obj_scaling: None,
            user_g_scaling: None,
            user_x_scaling: None,
            warm_start_y: None,
            warm_start_z_l: None,
            warm_start_z_u: None,
            warm_start_s: None,
            warm_start_v_l: None,
            warm_start_v_u: None,
            kkt_dump_dir: None,
            kkt_dump_name: "problem".to_string(),
            slack_move: {
                // f64::EPSILON.powf(0.75) ≈ 1.83e-12
                const EPS: f64 = f64::EPSILON;
                EPS.powf(0.75)
            },
            magic_step: true,
            use_ic_refinement: true,
            iterative_refinement_steps_required: 1,
            max_refinement_steps: 10,
            residual_ratio_max: 1e-10,
            residual_ratio_singular: 1e-5,
            residual_improvement_factor: 0.999999999,
            // T3.23: default OFF. Analysis (see kkt.rs::compute_full_residual_ratio
            // doc) shows the complementarity residual rows (5)-(6) are zero by
            // construction in exact arithmetic when the analytic Fiacco recovery
            // is used — the elimination is the same formula. The check therefore
            // measures only floating-point cancellation in `(μ − z·s)`, which is
            // tiny on well-conditioned problems and not actionable via another
            // augmented back-solve (the corrective solve uses the same elimination).
            // The plumbing is in place; flip to true once a true unsymmetric-IR
            // path is added (T3.23 follow-up).
            ir_residual_full_8_block: false,
            // T3.25: HS-suite bisect (cache=on vs cache=off) on the v0.8 baseline
            // showed 0 status changes, 0 iteration deltas across 120 problems with
            // a small but real ~2% wall-clock speedup. Bit-identical by construction
            // (cache hits just skip the supernodal refactor; the held factorization
            // is the same one the no-cache path would have produced). Defaulted ON.
            factor_cache_enabled: true,
            bound_mult_init_method: BoundMultInitMethod::Constant,
            bound_mult_init_val: 1.0,
            fixed_variable_treatment: FixedVariableTreatment::RelaxBounds,
            acceptable_obj_change_tol: 1e20,
            acceptable_iter: 15,
            diverging_iterates_tol: 1e20,
        }
    }
}
