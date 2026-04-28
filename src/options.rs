/// Treatment of variables with `x_L[i] == x_U[i]` (fixed variables).
///
/// Mirrors Ipopt 3.14's `fixed_variable_treatment` (`TNLPAdapter`).
/// Ipopt's default is `MakeParameter`: fixed variables are removed from
/// the optimization. ripopt defaults to `RelaxBounds` for backward
/// compatibility with pre-v0.8 callers; `MakeParameter` activates a
/// fixed-var-only preprocessing pass (`PreprocessedProblem::new_fixed_only`)
/// even when `enable_preprocessing` is off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        Self::RelaxBounds
    }
}

/// Bound multiplier initialization method.
///
/// Mirrors Ipopt 3.14's `bound_mult_init_method`
/// (`IpDefaultIterateInitializer.cpp:254-288`). Ipopt's default is
/// `Constant` with `bound_mult_init_val = 1.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// (`IpBacktrackingLineSearch.cpp:84-104`). T3.32 ports the simple
/// modes; min/max-dual-infeas variants are deferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
}

/// NLP scaling method. Mirrors Ipopt 3.14 `nlp_scaling_method`
/// registered in `IpAlgBuilder.cpp:343-353` and dispatched in
/// `IpAlgBuilder.cpp:678-696`. The `Equilibration` variant (Curtis-Reid
/// via Harwell MC19) is omitted — ripopt has no MC19 binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
#[derive(Debug, Clone)]
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
    /// Use least-squares estimate for initial constraint multipliers. Default: true.
    pub least_squares_mult_init: bool,
    /// Maximum absolute value for LS multiplier init; if exceeded, fall back to zero. Default: 1000.0.
    pub constr_mult_init_max: f64,
    /// Include constraint slack log-barriers in the filter merit function. Default: true.
    pub constraint_slack_barrier: bool,
    /// Maximum wall-clock time in seconds. 0.0 means no limit.
    pub max_wall_time: f64,
    /// Number of consecutive shortened steps before activating watchdog. Default: 10.
    pub watchdog_shortened_iter_trigger: usize,
    /// Maximum trial iterations during watchdog mode. Default: 5.
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
    /// Enable slack variable fallback for inequality problems. When the initial
    /// solve fails, retry with explicit slack variables (g(x)-s=0, bounds on s).
    /// Default: true.
    pub enable_slack_fallback: bool,
    /// Enable L-BFGS fallback for unconstrained problems. When IPM fails with
    /// MaxIterations or NumericalError, retry with L-BFGS. Default: true.
    pub enable_lbfgs_fallback: bool,
    /// Enable preprocessing to eliminate fixed variables and redundant constraints.
    /// Default: true.
    pub enable_preprocessing: bool,
    /// Detect linear constraints and skip their Hessian contribution.
    /// Default: true.
    pub detect_linear_constraints: bool,
    /// Use L-BFGS Hessian approximation inside the IPM instead of exact Hessian.
    /// When enabled, `hessian_structure()` and `hessian_values()` are never called.
    /// Equivalent to Ipopt's `hessian_approximation = "limited-memory"`.
    /// Default: false.
    pub hessian_approximation_lbfgs: bool,
    /// Automatic L-BFGS Hessian fallback. When the exact-Hessian IPM fails
    /// (MaxIterations, NumericalError, or RestorationFailed), retry with
    /// L-BFGS Hessian approximation. Useful when the user-provided Hessian
    /// is ill-conditioned, buggy, or overly sparse.
    /// Default: true.
    pub enable_lbfgs_hessian_fallback: bool,
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
    /// Enable proactive infeasibility detection.
    ///
    /// Monitors constraint violation (θ) in the main loop. If θ has stagnated
    /// (< 1% relative change over the history window) and the gradient of θ is
    /// stationary, declares LocalInfeasibility earlier instead of wasting iterations
    /// before restoration eventually fires.
    ///
    /// Default: true.
    pub proactive_infeasibility_detection: bool,
    /// Choice of sparse linear solver. Default: Direct.
    pub linear_solver: LinearSolverChoice,
    /// Maximum consecutive iterations without 1% improvement in primal or dual
    /// infeasibility before declaring stall (NumericalError). 0 = disable stall detection.
    /// Default: 0 (disabled — Ipopt 3.14 has no equivalent stall_iter path; see
    /// docs/IPOPT_ALGORITHM_SPEC.md §14.x). Retained as an opt-in escape hatch.
    pub stall_iter_limit: usize,
    /// Maximum wall-clock seconds allowed for the first few iterations.
    /// If the solver has completed fewer than 3 iterations after this many seconds,
    /// it returns NumericalError to trigger fallback strategies.
    /// 0.0 disables early stall detection.
    /// Default: 10.0.
    pub early_stall_timeout: f64,
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
            mu_strategy_adaptive: true,
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
            constraint_slack_barrier: true,
            max_wall_time: 0.0,
            watchdog_shortened_iter_trigger: 10,
            watchdog_trial_iter_max: 5,
            sparse_threshold: 110,
            barrier_tol_factor: 10.0,
            mu_allow_fast_monotone_decrease: true,
            adaptive_mu_monotone_init_factor: 0.8,
            adaptive_mu_restore_previous_iterate: false,
            restoration_max_iter: 200,
            disable_nlp_restoration: false,
            kappa_resto: 0.9,
            enable_slack_fallback: true,
            enable_lbfgs_fallback: true,
            enable_preprocessing: true,
            detect_linear_constraints: true,
            hessian_approximation_lbfgs: false,
            enable_lbfgs_hessian_fallback: true,
            mehrotra_pc: false,
            gondzio_mcc_max: 3,
            proactive_infeasibility_detection: false,
            linear_solver: LinearSolverChoice::default(),
            stall_iter_limit: 0,
            early_stall_timeout: 120.0,
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
            bound_mult_init_method: BoundMultInitMethod::Constant,
            bound_mult_init_val: 1.0,
            fixed_variable_treatment: FixedVariableTreatment::RelaxBounds,
            acceptable_obj_change_tol: 1e20,
            acceptable_iter: 15,
            diverging_iterates_tol: 1e20,
        }
    }
}
