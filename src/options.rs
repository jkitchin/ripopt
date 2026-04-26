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
    /// Fraction-to-boundary parameter minimum.
    pub tau_min: f64,
    /// Barrier parameter reduction factor (monotone mode).
    pub mu_linear_decrease_factor: f64,
    /// Barrier parameter superlinear decrease power.
    pub mu_superlinear_decrease_power: f64,
    /// Print level (0 = silent, 5 = verbose).
    pub print_level: u8,
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
    /// Initial factor for mu in fixed mode: mu = this * avg_compl. Default: 0.8.
    pub adaptive_mu_monotone_init_factor: f64,
    /// Maximum iterations for restoration NLP subproblem. Default: 200.
    pub restoration_max_iter: usize,
    /// Disable NLP restoration (prevents recursion in inner solve). Default: false.
    pub disable_nlp_restoration: bool,
    /// Enable slack variable fallback for inequality problems. When the initial
    /// solve fails, retry with explicit slack variables (g(x)-s=0, bounds on s).
    /// Default: true.
    pub enable_slack_fallback: bool,
    /// Enable L-BFGS fallback for unconstrained problems. When IPM fails with
    /// MaxIterations or NumericalError, retry with L-BFGS. Default: true.
    pub enable_lbfgs_fallback: bool,
    /// Enable Augmented Lagrangian fallback for equality-only problems. When IPM
    /// fails, retry with AL method using L-BFGS inner solver. Default: true.
    pub enable_al_fallback: bool,
    /// Enable preprocessing to eliminate fixed variables and redundant constraints.
    /// Default: true.
    pub enable_preprocessing: bool,
    /// Detect linear constraints and skip their Hessian contribution.
    /// Default: true.
    pub detect_linear_constraints: bool,
    /// Enable SQP fallback for constrained problems when IPM/AL/slack fail.
    /// Default: true.
    pub enable_sqp_fallback: bool,
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
    /// Default: true.
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
    /// Default: 30.
    pub stall_iter_limit: usize,
    /// Maximum wall-clock seconds allowed for the first few iterations.
    /// If the solver has completed fewer than 3 iterations after this many seconds,
    /// it returns NumericalError to trigger fallback strategies.
    /// 0.0 disables early stall detection.
    /// Default: 10.0.
    pub early_stall_timeout: f64,
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
    /// If set, serialize each KKT matrix (main IPM loop only) to this directory
    /// after factorization. Writes two files per iteration:
    ///   `<kkt_dump_name>_<iter:04>.mtx`  — Matrix Market format, symmetric, lower triangle
    ///   `<kkt_dump_name>_<iter:04>.json` — Metadata: n, m, iteration, rhs, inertia, status
    /// The directory is created with `create_dir_all` if it does not exist.
    /// No-op when `None` (default). IO errors are logged as warnings, never abort the solve.
    pub kkt_dump_dir: Option<std::path::PathBuf>,
    /// Problem name used in dump filenames. Defaults to `"problem"`.
    pub kkt_dump_name: String,
    /// Acceptable-level relative objective-change gate. The acceptable
    /// status only fires when `|f_k - f_{k-1}| / max(1, |f_k|) ≤
    /// acceptable_obj_change_tol`. Default 1e20 disables the gate
    /// (matches Ipopt 3.14 `IpOptErrorConvCheck.cpp:115`). Lowering
    /// this below 1.0 forces the acceptable check to also see the
    /// objective settle, useful when quasi-Newton stalls leave dual
    /// infeasibility large but the iterate has plateaued.
    pub acceptable_obj_change_tol: f64,
    /// Threshold on `‖x‖_∞` above which the iterate is declared
    /// diverging. Default 1e20 matches Ipopt
    /// `IpOptErrorConvCheck.cpp:123`. Lower this on bounded problems
    /// where you want an early abort if the IPM walks far outside the
    /// expected feasible region.
    pub diverging_iterates_tol: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 3000,
            mu_init: 0.1,
            mu_min: 1e-11,
            tau_min: 0.99,
            mu_linear_decrease_factor: 0.2,
            mu_superlinear_decrease_power: 1.5,
            print_level: 5,
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
            nlp_lower_bound_inf: -1e20,
            nlp_upper_bound_inf: 1e20,
            kappa: 10.0,
            mu_allow_increase: true,
            least_squares_mult_init: true,
            constr_mult_init_max: 1000.0,
            constraint_slack_barrier: true,
            max_wall_time: 0.0,
            watchdog_shortened_iter_trigger: 10,
            watchdog_trial_iter_max: 3,
            sparse_threshold: 110,
            barrier_tol_factor: 10.0,
            adaptive_mu_monotone_init_factor: 0.8,
            restoration_max_iter: 200,
            disable_nlp_restoration: false,
            enable_slack_fallback: true,
            enable_lbfgs_fallback: true,
            enable_al_fallback: true,
            enable_preprocessing: true,
            detect_linear_constraints: true,
            enable_sqp_fallback: true,
            hessian_approximation_lbfgs: false,
            enable_lbfgs_hessian_fallback: true,
            mehrotra_pc: true,
            gondzio_mcc_max: 3,
            proactive_infeasibility_detection: false,
            linear_solver: LinearSolverChoice::default(),
            stall_iter_limit: 30,
            early_stall_timeout: 120.0,
            mu_oracle_quality_function: true,
            quality_function_centrality: false,
            user_obj_scaling: None,
            user_g_scaling: None,
            user_x_scaling: None,
            warm_start_y: None,
            warm_start_z_l: None,
            warm_start_z_u: None,
            kkt_dump_dir: None,
            kkt_dump_name: "problem".to_string(),
            acceptable_obj_change_tol: 1e20,
            diverging_iterates_tol: 1e20,
        }
    }
}
