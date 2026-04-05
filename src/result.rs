use crate::logging::rip_log;

/// Status of the solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    /// Converged to optimal solution within tolerance.
    Optimal,
    /// Problem is infeasible.
    Infeasible,
    /// Local infeasibility detected: constraint violation is at a stationary
    /// point (gradient of violation ≈ 0) but violation is still large.
    /// For NE-to-LS reformulations, this means the system is inconsistent
    /// and x* is the best least-squares solution.
    LocalInfeasibility,
    /// Reached maximum number of iterations.
    MaxIterations,
    /// Numerical difficulties (e.g., singular KKT system).
    NumericalError,
    /// Problem appears unbounded below.
    Unbounded,
    /// Restoration phase failed.
    RestorationFailed,
    /// User callback returned `false`, indicating evaluation failure at the
    /// current iterate (not during line search, where failures cause step
    /// rejection instead).
    EvaluationError,
    /// Intermediate callback returned `false`, requesting early termination.
    UserRequestedStop,
    /// Internal error.
    InternalError,
}

/// Structured diagnostic summary from a solve.
///
/// Captures counts of key solver events (restoration entries, barrier parameter
/// mode switches, filter rejects, etc.) and final convergence measures.
/// Useful for automated analysis and solver tuning.
#[derive(Debug, Clone, Default)]
pub struct SolverDiagnostics {
    /// Number of GN (Gauss-Newton) restoration entries.
    pub restoration_count: usize,
    /// Number of full NLP restoration entries.
    pub nlp_restoration_count: usize,
    /// Number of mu mode switches (Free↔Fixed).
    pub mu_mode_switches: usize,
    /// Number of filter rejects (line search exhausted backtracking).
    pub filter_rejects: usize,
    /// Number of watchdog activations.
    pub watchdog_activations: usize,
    /// Number of second-order corrections (SOC) applied.
    pub soc_corrections: usize,
    /// Final barrier parameter mu.
    pub final_mu: f64,
    /// Final primal infeasibility.
    pub final_primal_inf: f64,
    /// Final dual infeasibility (iterative z, used in unscaled gate).
    pub final_dual_inf: f64,
    /// Final dual infeasibility (z_opt, used in scaled gate).
    pub final_dual_inf_scaled: f64,
    /// Final complementarity error.
    pub final_compl: f64,
    /// Dual scaling factor s_d.
    pub final_s_d: f64,
    /// Total wall-clock time in seconds.
    pub wall_time_secs: f64,
    /// Fallback strategy used, if any.
    pub fallback_used: Option<String>,
}

impl SolverDiagnostics {
    /// Print a structured diagnostic summary to stderr.
    pub fn print_summary(&self, status: SolveStatus, iterations: usize) {
        rip_log!("\n--- ripopt diagnostics ---");
        rip_log!("status: {:?}", status);
        rip_log!("iterations: {}", iterations);
        rip_log!("wall_time: {:.3}s", self.wall_time_secs);
        rip_log!("final_mu: {:.2e}", self.final_mu);
        rip_log!("final_primal_inf: {:.2e}", self.final_primal_inf);
        rip_log!("final_dual_inf: {:.2e}", self.final_dual_inf);
        rip_log!("final_compl: {:.2e}", self.final_compl);
        rip_log!("restoration_count: {}", self.restoration_count);
        rip_log!("nlp_restoration_count: {}", self.nlp_restoration_count);
        rip_log!("mu_mode_switches: {}", self.mu_mode_switches);
        rip_log!("filter_rejects: {}", self.filter_rejects);
        rip_log!("watchdog_activations: {}", self.watchdog_activations);
        rip_log!("soc_corrections: {}", self.soc_corrections);
        if let Some(ref fb) = self.fallback_used {
            rip_log!("fallback_used: {}", fb);
        }
        rip_log!("--- end diagnostics ---");
    }
}

/// Result of solving an NLP.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Optimal primal variables x*.
    pub x: Vec<f64>,
    /// Optimal objective value f(x*).
    pub objective: f64,
    /// Constraint multipliers (lambda).
    pub constraint_multipliers: Vec<f64>,
    /// Lower bound multipliers (z_L).
    pub bound_multipliers_lower: Vec<f64>,
    /// Upper bound multipliers (z_U).
    pub bound_multipliers_upper: Vec<f64>,
    /// Constraint values g(x*).
    pub constraint_values: Vec<f64>,
    /// Solve status.
    pub status: SolveStatus,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Structured solver diagnostics.
    pub diagnostics: SolverDiagnostics,
}
