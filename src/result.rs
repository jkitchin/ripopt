use crate::logging::rip_log;

/// Status of the solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum SolveStatus {
    /// Converged to optimal solution within tolerance.
    Optimal,
    /// Converged to an acceptable solution: KKT residuals within the
    /// relaxed acceptable-level tolerances (matching Ipopt's
    /// `Solved_To_Acceptable_Level`). Counted as "solved" by the
    /// benchmark reporter — the iterate is at a stationary point that
    /// Ipopt's default user-facing settings would also return.
    Acceptable,
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
    /// Iterates diverged: ‖x‖_∞ exceeded `diverging_iterates_tol`
    /// (default 1e20). Mirrors Ipopt 3.14's
    /// `Diverging_Iterates` (`IpReturnCodes_inc.h`); the legitimate
    /// signature of an unbounded NLP after the bound-relaxation push.
    DivergingIterates,
    /// Restoration phase failed.
    RestorationFailed,
    /// User callback returned `false`, indicating evaluation failure at the
    /// current iterate (not during line search, where failures cause step
    /// rejection instead).
    EvaluationError,
    /// Intermediate callback returned `false`, requesting early termination.
    UserRequestedStop,
    /// Search direction has become too small to make further progress.
    /// Mirrors Ipopt's `STOP_AT_TINY_STEP` from `IpBacktrackingLineSearch.cpp`,
    /// reported as `Search_Direction_Becomes_Too_Small` in
    /// `IpReturnCodes_inc.h`. The current iterate is the best ripopt can
    /// produce with the current barrier subproblem; downstream code is
    /// expected to treat it analogously to acceptable termination.
    StopAtTinyStep,
    /// Internal error.
    InternalError,
}

/// Structured diagnostic summary from a solve.
///
/// Captures counts of key solver events (restoration entries, barrier parameter
/// mode switches, filter rejects, etc.) and final convergence measures.
/// Useful for automated analysis and solver tuning.
#[derive(Debug, Clone, Default, serde::Serialize)]
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
    /// Final dual infeasibility (iterative z).
    pub final_dual_inf: f64,
    /// Final complementarity error (iterative z).
    pub final_compl: f64,
    /// Dual scaling factor s_d.
    pub final_s_d: f64,
    /// Total wall-clock time in seconds.
    pub wall_time_secs: f64,
    /// Fallback strategy used, if any.
    pub fallback_used: Option<String>,
    /// T3.25 follow-up: number of `factor_with_inertia_correction_cached`
    /// invocations whose 13-tag fingerprint matched the previous
    /// successful factor and so skipped the underlying `solver.factor`.
    /// Always 0 when `SolverOptions::factor_cache_enabled = false`.
    pub factor_cache_hits: u64,
    /// T3.25 follow-up: number of cached-entry calls that fell through
    /// to a real factorization. Counted even when the cache is enabled
    /// but the fingerprint changed (atag bump or first call).
    pub factor_cache_misses: u64,
    /// T3.25 follow-up: total number of times `solver.factor` was
    /// invoked through the cached entry point. Exercised by every
    /// caller of `factor_with_inertia_correction_cached`, including
    /// the cache-disabled path where every call factors.
    pub factor_cache_factor_calls: u64,
    /// B11: cumulative NLP-callback counts surfaced in the final
    /// summary. ripopt's `constraints` and `jacobian_values` fill the
    /// joint c/d block in one call, so `n_constr_evals`/`n_jac_evals`
    /// count both the equality and the inequality side; the printed
    /// summary repeats the same value on the equality and inequality
    /// rows for Ipopt-format compatibility.
    pub n_obj_evals: usize,
    pub n_grad_evals: usize,
    pub n_constr_evals: usize,
    pub n_jac_evals: usize,
    pub n_hess_evals: usize,
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
