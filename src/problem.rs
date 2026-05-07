/// Trait defining a nonlinear programming problem.
///
/// All methods use a buffer-filling pattern to avoid allocations in the hot loop.
/// The caller provides pre-allocated slices and the implementation fills them.
///
/// The `new_x` parameter on evaluation methods indicates whether `x` has changed
/// since the last evaluation call. When `new_x` is `false`, cached intermediate
/// results (e.g., phase equilibria, shared subexpressions) can be reused. This
/// matches the semantics of IPOPT's C interface `new_x` flag.
///
/// Evaluation methods return `bool`: `true` on success, `false` when the function
/// cannot be evaluated at the given point (e.g., log of a negative number). This
/// matches the IPOPT C interface convention. During line search, the solver treats
/// evaluation failure as trial-point rejection and shortens the step. If evaluation
/// fails at the current iterate (not during line search), the solver returns
/// `SolveStatus::EvaluationError`.
pub trait NlpProblem {
    /// Number of primal variables.
    fn num_variables(&self) -> usize;

    /// Number of constraints.
    fn num_constraints(&self) -> usize;

    /// Fill variable bounds: x_l\[i\] <= x\[i\] <= x_u\[i\].
    /// Use f64::NEG_INFINITY / f64::INFINITY for unbounded.
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]);

    /// Fill constraint bounds: g_l\[i\] <= g(x)\[i\] <= g_u\[i\].
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]);

    /// Fill initial primal point.
    fn initial_point(&self, x0: &mut [f64]);

    /// Optional: fill initial dual multipliers for warm-starting.
    ///
    /// Called by the IPM only when `options.warm_start` is true. The default
    /// implementation returns `false`, leaving the solver to compute its own
    /// initial multipliers (least-squares estimate for constraint multipliers,
    /// `mu / slack` for bound multipliers).
    ///
    /// When overridden to return `true`, `lam_g` must be filled with the
    /// constraint multipliers (one per constraint; sign follows the standard
    /// Lagrangian `L = f + lam_g^T g`), and `z_l` / `z_u` with the bound
    /// multipliers (one per variable; must be non-negative — the solver will
    /// floor them to `warm_start_mult_bound_push`). Entries for unbounded
    /// variables are ignored.
    fn initial_multipliers(
        &self,
        _lam_g: &mut [f64],
        _z_l: &mut [f64],
        _z_u: &mut [f64],
    ) -> bool {
        false
    }

    /// Evaluate objective f(x) into `obj`.
    /// `new_x` is `true` when `x` differs from the previous evaluation point.
    /// Return `true` on success, `false` if evaluation fails at this point.
    fn objective(&self, x: &[f64], new_x: bool, obj: &mut f64) -> bool;

    /// Fill gradient of objective: grad\[i\] = df/dx_i.
    /// `new_x` is `true` when `x` differs from the previous evaluation point.
    /// Return `true` on success, `false` if evaluation fails at this point.
    fn gradient(&self, x: &[f64], new_x: bool, grad: &mut [f64]) -> bool;

    /// Evaluate constraints: g\[i\] = g_i(x).
    /// `new_x` is `true` when `x` differs from the previous evaluation point.
    /// Return `true` on success, `false` if evaluation fails at this point.
    fn constraints(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool;

    /// Return the sparsity structure of the constraint Jacobian.
    /// Returns (row_indices, col_indices) in triplet format.
    /// Only the non-zero entries need to be specified.
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>);

    /// Fill Jacobian values at x in the same order as jacobian_structure().
    /// `new_x` is `true` when `x` differs from the previous evaluation point.
    /// Return `true` on success, `false` if evaluation fails at this point.
    fn jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool;

    /// Return the sparsity structure of the Lagrangian Hessian (lower triangle only).
    /// Returns (row_indices, col_indices) in triplet format.
    /// This is the Hessian of: obj_factor * f(x) + sum_i lambda\[i\] * g_i(x).
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>);

    /// Fill Hessian values at x with the given obj_factor and constraint multipliers lambda.
    /// Only lower triangle entries in the same order as hessian_structure().
    /// `new_x` is `true` when `x` differs from the previous evaluation point.
    /// Return `true` on success, `false` if evaluation fails at this point.
    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool;

    /// Optional hook: notify the problem of the current barrier parameter μ.
    ///
    /// Default: no-op. Override when the objective depends on μ (the
    /// restoration NLP has `(η/2)·‖D_R(x − x_R)‖²` with η = η_factor·√μ
    /// per Ipopt `RestoIpoptNLP::Eta`; see `IpRestoIpoptNLP.cpp:759`).
    /// The IPM calls this once per outer iteration, before any objective
    /// / gradient / Hessian evaluations.
    fn notify_mu(&self, _mu: f64) {}

    /// Optional hook: per-iteration early-exit test. Default returns
    /// `false` (never exits early). The IPM consults this once per
    /// outer iteration after the trial step has been accepted; a `true`
    /// return ends the solve with `SolveStatus::Optimal` at the current
    /// iterate.
    ///
    /// Used by `RestorationNlp` to implement Ipopt's
    /// `IpRestoFilterConvCheck::TestOrigProgress`
    /// (`IpRestoFilterConvCheck.cpp:53-80`): when the parent's
    /// constraint violation at the restored x has dropped below
    /// `kappa_resto · theta_entry`, the inner restoration solve exits
    /// early so the parent can resume — instead of running the inner
    /// solve to its own KKT convergence (which targets resto-NLP
    /// optimality, not parent feasibility recovery).
    fn resto_early_exit(&self, _x: &[f64]) -> bool { false }
}
