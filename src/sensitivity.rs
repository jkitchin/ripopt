//! Parametric sensitivity analysis for nonlinear programs.
//!
//! Implements sIPOPT-style post-optimal sensitivity: given a converged solution,
//! computes how the optimal point changes when problem parameters are perturbed.
//! The core equation is `ds/dp = -M⁻¹ · Nₚ` — just one backsolve reusing the
//! factored KKT matrix.

use crate::linear_solver::dense::DenseLdl;
use crate::linear_solver::{KktMatrix, LinearSolver};
use crate::options::SolverOptions;
use crate::problem::NlpProblem;
use crate::result::{SolveResult, SolveStatus};

/// Extension of [`NlpProblem`] for problems with explicit parameters.
///
/// Mirrors sIPOPT 2.0's parametric interface. Users implement this trait
/// to provide derivatives of the objective, constraints, and Lagrangian
/// with respect to parameters `p`.
pub trait ParametricNlpProblem: NlpProblem {
    /// Number of parameters p.
    fn num_parameters(&self) -> usize;

    /// Sparsity of ∂g/∂p (constraint Jacobian w.r.t. parameters).
    /// Returns (row_indices, col_indices) in COO format.
    fn jacobian_p_structure(&self) -> (Vec<usize>, Vec<usize>);

    /// Fill ∂g/∂p values at x.
    fn jacobian_p_values(&self, x: &[f64], vals: &mut [f64]);

    /// Sparsity of ∇²ₓₚL (cross-Hessian of Lagrangian w.r.t. x and p).
    /// Returns (row_indices, col_indices) where rows index x, cols index p.
    fn hessian_xp_structure(&self) -> (Vec<usize>, Vec<usize>);

    /// Fill ∇²ₓₚL values at (x, obj_factor, lambda).
    fn hessian_xp_values(&self, x: &[f64], obj_factor: f64, lambda: &[f64], vals: &mut [f64]);
}

/// Retains the factored KKT matrix from a converged solve for sensitivity analysis.
pub struct SensitivityContext {
    /// The solve result (solution point, multipliers, status).
    pub result: SolveResult,
    /// Factored unregularized KKT matrix.
    solver: Box<dyn LinearSolver>,
    /// Number of primal variables.
    n: usize,
    /// Number of constraints.
    m: usize,
    /// Variable lower bounds.
    x_l: Vec<f64>,
    /// Variable upper bounds.
    x_u: Vec<f64>,
}

/// Result of sensitivity computation.
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// dx/dp matrix: `dx_dp[k]` is the n-vector of sensitivities for perturbation k.
    pub dx_dp: Vec<Vec<f64>>,
    /// dlambda/dp matrix: `dlambda_dp[k]` is the m-vector for perturbation k.
    pub dlambda_dp: Vec<Vec<f64>>,
    /// dz_l/dp (lower bound multiplier sensitivities).
    pub dz_l_dp: Vec<Vec<f64>>,
    /// dz_u/dp (upper bound multiplier sensitivities).
    pub dz_u_dp: Vec<Vec<f64>>,
}

/// Solve the NLP and retain a factored KKT for subsequent sensitivity analysis.
///
/// This calls the normal `solve()`, then reassembles the unregularized KKT at the
/// solution and factors it. The cost is one extra Hessian/Jacobian evaluation and
/// one factorization — negligible compared to the solve itself.
pub fn solve_with_sensitivity<P: ParametricNlpProblem>(
    problem: &P,
    options: &SolverOptions,
) -> SensitivityContext {
    let result = crate::solve(problem, options);

    let n = problem.num_variables();
    let m = problem.num_constraints();
    let dim = n + m;

    let x = &result.x;
    let y = &result.constraint_multipliers;
    let z_l = &result.bound_multipliers_lower;
    let z_u = &result.bound_multipliers_upper;

    // Get bounds
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    problem.bounds(&mut x_l, &mut x_u);

    // Build barrier diagonal Σ
    let mut sigma = vec![0.0; n];
    for i in 0..n {
        if x_l[i].is_finite() {
            let sl = x[i] - x_l[i];
            if sl > 0.0 {
                sigma[i] += z_l[i] / sl;
            }
        }
        if x_u[i].is_finite() {
            let su = x_u[i] - x[i];
            if su > 0.0 {
                sigma[i] += z_u[i] / su;
            }
        }
    }

    // Get Hessian at solution
    let (hess_rows, hess_cols) = problem.hessian_structure();
    let mut hess_vals = vec![0.0; hess_rows.len()];
    // Ignore eval failure here — sensitivity is post-solve, if evals fail the
    // sensitivity results will be meaningless but the solver result is already returned.
    let _ = problem.hessian_values(x, true, 1.0, y, &mut hess_vals);

    // Get Jacobian at solution
    let (jac_rows, jac_cols) = problem.jacobian_structure();
    let mut jac_vals = vec![0.0; jac_rows.len()];
    let _ = problem.jacobian_values(x, true, &mut jac_vals);

    // Assemble unregularized KKT:
    // [W + Σ,  J^T]
    // [J,       0  ]
    let mut matrix = KktMatrix::zeros_dense(dim);

    // (1,1) block: Hessian + Σ
    for (idx, (&r, &c)) in hess_rows.iter().zip(hess_cols.iter()).enumerate() {
        matrix.add(r, c, hess_vals[idx]);
    }
    for i in 0..n {
        matrix.add(i, i, sigma[i]);
    }

    // (2,1) block: J
    for (idx, (&r, &c)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        matrix.add(n + r, c, jac_vals[idx]);
    }

    // Factor
    let mut lin_solver: Box<dyn LinearSolver> = Box::new(DenseLdl::new());
    if let Err(e) = lin_solver.factor(&matrix) {
        log::warn!("Sensitivity KKT factorization failed: {}", e);
    }

    SensitivityContext {
        result,
        solver: lin_solver,
        n,
        m,
        x_l,
        x_u,
    }
}

impl SensitivityContext {
    /// Compute ds/dp for one or more parameter perturbations Δp.
    ///
    /// Each entry in `delta_p` is a slice of length `num_parameters`.
    /// Returns sensitivities `dx/dp`, `dlambda/dp`, `dz_l/dp`, `dz_u/dp`
    /// for each perturbation.
    pub fn compute_sensitivity<P: ParametricNlpProblem>(
        &mut self,
        problem: &P,
        delta_p: &[&[f64]],
    ) -> Result<SensitivityResult, String> {
        if !matches!(
            self.result.status,
            SolveStatus::Optimal
        ) {
            return Err(format!(
                "Sensitivity requires a converged solution, got {:?}",
                self.result.status
            ));
        }

        let n = self.n;
        let m = self.m;
        let dim = n + m;
        let n_p = problem.num_parameters();
        let x = &self.result.x;
        let y = &self.result.constraint_multipliers;
        let z_l = &self.result.bound_multipliers_lower;
        let z_u = &self.result.bound_multipliers_upper;

        // Get parameter derivative structures
        let (jac_p_rows, jac_p_cols) = problem.jacobian_p_structure();
        let mut jac_p_vals = vec![0.0; jac_p_rows.len()];
        problem.jacobian_p_values(x, &mut jac_p_vals);

        let (hess_xp_rows, hess_xp_cols) = problem.hessian_xp_structure();
        let mut hess_xp_vals = vec![0.0; hess_xp_rows.len()];
        problem.hessian_xp_values(x, 1.0, y, &mut hess_xp_vals);

        let mut dx_dp = Vec::with_capacity(delta_p.len());
        let mut dlambda_dp = Vec::with_capacity(delta_p.len());
        let mut dz_l_dp = Vec::with_capacity(delta_p.len());
        let mut dz_u_dp = Vec::with_capacity(delta_p.len());

        for dp in delta_p {
            if dp.len() != n_p {
                return Err(format!(
                    "delta_p has length {}, expected {}",
                    dp.len(),
                    n_p
                ));
            }

            // Assemble RHS: -Nₚ·Δp
            // Nₚ·Δp = [∇²ₓₚL · Δp]  (n-vector)
            //          [∂g/∂p · Δp ]  (m-vector)
            let mut rhs = vec![0.0; dim];

            // ∇²ₓₚL · Δp contribution to first n entries
            for (idx, (&r, &c)) in hess_xp_rows.iter().zip(hess_xp_cols.iter()).enumerate() {
                rhs[r] += hess_xp_vals[idx] * dp[c];
            }

            // ∂g/∂p · Δp contribution to entries n..n+m
            for (idx, (&r, &c)) in jac_p_rows.iter().zip(jac_p_cols.iter()).enumerate() {
                rhs[n + r] += jac_p_vals[idx] * dp[c];
            }

            // Negate: solve M·[dx; dy] = -Nₚ·Δp
            for v in rhs.iter_mut() {
                *v = -*v;
            }

            // Solve
            let mut sol = vec![0.0; dim];
            self.solver
                .solve(&rhs, &mut sol)
                .map_err(|e| format!("Sensitivity backsolve failed: {}", e))?;

            let dx: Vec<f64> = sol[..n].to_vec();
            let dy: Vec<f64> = sol[n..].to_vec();

            // Recover bound multiplier sensitivities:
            // dz_l[i] = (z_l[i] - sigma_l[i] * dx[i]) / s_l[i]
            // where sigma_l[i] = z_l[i] / s_l[i], so dz_l[i] = -sigma_l[i] * dx[i]
            // More precisely: from z_l*(x-x_l) = mu, differentiating:
            //   dz_l*(x-x_l) + z_l*dx = 0  =>  dz_l = -z_l*dx/(x-x_l)
            let mut dzl = vec![0.0; n];
            let mut dzu = vec![0.0; n];
            for i in 0..n {
                if self.x_l[i].is_finite() {
                    let sl = self.result.x[i] - self.x_l[i];
                    if sl > 0.0 {
                        dzl[i] = -z_l[i] * dx[i] / sl;
                    }
                }
                if self.x_u[i].is_finite() {
                    let su = self.x_u[i] - self.result.x[i];
                    if su > 0.0 {
                        // z_u*(x_u-x) = mu => dz_u*(x_u-x) + z_u*(-dx) = 0
                        // => dz_u = z_u*dx/(x_u-x)
                        dzu[i] = z_u[i] * dx[i] / su;
                    }
                }
            }

            dx_dp.push(dx);
            dlambda_dp.push(dy);
            dz_l_dp.push(dzl);
            dz_u_dp.push(dzu);
        }

        Ok(SensitivityResult {
            dx_dp,
            dlambda_dp,
            dz_l_dp,
            dz_u_dp,
        })
    }

    /// Extract the reduced Hessian of the Lagrangian projected onto the
    /// null space of active constraints.
    ///
    /// For unconstrained problems (m=0), this is simply `(W + Σ)⁻¹`.
    /// For constrained problems, computes `Z^T (W+Σ) Z` where Z spans
    /// the null space of J. Useful for covariance estimation in parameter
    /// estimation problems.
    ///
    /// Returns an (n_free × n_free) matrix where n_free = n - rank(J_active).
    /// For simplicity, currently returns the full n×n inverse of the (1,1)
    /// block when m=0, or the Schur complement inverse when m>0.
    pub fn reduced_hessian(&mut self) -> Result<Vec<Vec<f64>>, String> {
        if !matches!(
            self.result.status,
            SolveStatus::Optimal
        ) {
            return Err(format!(
                "Reduced Hessian requires a converged solution, got {:?}",
                self.result.status
            ));
        }

        let n = self.n;
        let dim = n + self.m;

        // Compute M⁻¹ column by column using the factored KKT
        let mut inv_cols = vec![vec![0.0; dim]; n];
        for j in 0..n {
            let mut e_j = vec![0.0; dim];
            e_j[j] = 1.0;
            self.solver
                .solve(&e_j, &mut inv_cols[j])
                .map_err(|e| format!("Reduced Hessian solve failed: {}", e))?;
        }

        // Extract the top-left n×n block of M⁻¹
        // This is (W + Σ - J^T·0⁻¹·J)⁻¹ for the Schur complement,
        // which equals the projected Hessian inverse on the primal space.
        let mut reduced = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                reduced[i][j] = inv_cols[j][i];
            }
        }

        Ok(reduced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple QP with parameter in objective:
    ///   min  0.5*x^2 - p*x
    ///   s.t. x >= 0
    /// Solution: x* = p (when p > 0), dx/dp = 1
    struct ParametricQP {
        p: f64,
    }

    impl NlpProblem for ParametricQP {
        fn num_variables(&self) -> usize {
            1
        }
        fn num_constraints(&self) -> usize {
            0
        }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.0;
            x_u[0] = f64::INFINITY;
        }
        fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = 0.5 * x[0] * x[0] - self.p * x[0];
            true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = x[0] - self.p;
            true
        }
        fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![], vec![])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = obj_factor; // d²f/dx² = 1;
            true
        }
    }

    impl ParametricNlpProblem for ParametricQP {
        fn num_parameters(&self) -> usize {
            1
        }
        fn jacobian_p_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![], vec![]) // no constraints
        }
        fn jacobian_p_values(&self, _x: &[f64], _vals: &mut [f64]) {}
        fn hessian_xp_structure(&self) -> (Vec<usize>, Vec<usize>) {
            // ∂²L/∂x∂p = ∂(x - p)/∂p = -1
            (vec![0], vec![0])
        }
        fn hessian_xp_values(
            &self,
            _x: &[f64],
            obj_factor: f64,
            _lambda: &[f64],
            vals: &mut [f64],
        ) {
            vals[0] = -obj_factor; // ∂²f/∂x∂p = -1
        }
    }

    #[test]
    fn test_parametric_qp_sensitivity() {
        let problem = ParametricQP { p: 2.0 };
        let options = SolverOptions {
            print_level: 0,
            ..SolverOptions::default()
        };

        let mut ctx = solve_with_sensitivity(&problem, &options);
        assert!(
            matches!(ctx.result.status, SolveStatus::Optimal),
            "Expected converged, got {:?}",
            ctx.result.status
        );
        assert!(
            (ctx.result.x[0] - 2.0).abs() < 1e-4,
            "x* should be ~2.0, got {}",
            ctx.result.x[0]
        );

        // Compute sensitivity for Δp = [1.0]
        let dp = [1.0];
        let sens = ctx
            .compute_sensitivity(&problem, &[&dp])
            .expect("sensitivity should succeed");

        // dx/dp should be ~1.0 (x* = p, so dx*/dp = 1)
        assert!(
            (sens.dx_dp[0][0] - 1.0).abs() < 1e-4,
            "dx/dp should be ~1.0, got {}",
            sens.dx_dp[0][0]
        );

        // Verify by finite differences: solve at p+Δp and compare
        let problem2 = ParametricQP { p: 2.001 };
        let result2 = crate::solve(&problem2, &options);
        let fd_dx_dp = (result2.x[0] - ctx.result.x[0]) / 0.001;
        assert!(
            (fd_dx_dp - sens.dx_dp[0][0]).abs() < 1e-2,
            "FD dx/dp = {}, analytical = {}",
            fd_dx_dp,
            sens.dx_dp[0][0]
        );
    }

    /// Constrained QP with parameter in constraint bound:
    ///   min  0.5*(x1^2 + x2^2)
    ///   s.t. x1 + x2 = p
    /// Solution: x1* = x2* = p/2, dx1/dp = dx2/dp = 0.5
    struct ConstrainedParametricQP {
        p: f64,
    }

    impl NlpProblem for ConstrainedParametricQP {
        fn num_variables(&self) -> usize {
            2
        }
        fn num_constraints(&self) -> usize {
            1
        }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY;
            x_l[1] = f64::NEG_INFINITY;
            x_u[0] = f64::INFINITY;
            x_u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = self.p;
            g_u[0] = self.p;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
            x0[1] = 1.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = 0.5 * (x[0] * x[0] + x[1] * x[1]);
            true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = x[0];
            grad[1] = x[1];
            true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] + x[1];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0], vec![0, 1])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 1.0;
            vals[1] = 1.0;
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = obj_factor;
            vals[1] = obj_factor;
            true
        }
    }

    impl ParametricNlpProblem for ConstrainedParametricQP {
        fn num_parameters(&self) -> usize {
            1
        }
        fn jacobian_p_structure(&self) -> (Vec<usize>, Vec<usize>) {
            // The parameter p appears in the constraint bound, not in g(x) itself.
            // g(x) = x1+x2, constraint is g(x) = p.
            // To model this as ∂g/∂p: the constraint is g(x) - p = 0,
            // so effectively ∂(g-p)/∂p = -1, but we handle it via the RHS.
            // Actually in sIPOPT, the parameter perturbation enters the RHS as:
            //   ∂g/∂p · Δp where the "constraint" includes the bound.
            // For bound perturbation: g(x) = target = p, perturbing p by Δp
            // means the constraint residual changes by -Δp.
            // So ∂g/∂p = 0 (g doesn't depend on p), but we need a -Δp in the RHS.
            // We model this as: the constraint is g(x) - p = 0, ∂(g-p)/∂p = -1.
            (vec![0], vec![0])
        }
        fn jacobian_p_values(&self, _x: &[f64], vals: &mut [f64]) {
            // ∂(g - p)/∂p = -1
            vals[0] = -1.0;
        }
        fn hessian_xp_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![], vec![]) // no cross terms
        }
        fn hessian_xp_values(
            &self,
            _x: &[f64],
            _obj_factor: f64,
            _lambda: &[f64],
            _vals: &mut [f64],
        ) {
        }
    }

    #[test]
    fn test_constrained_sensitivity() {
        let problem = ConstrainedParametricQP { p: 2.0 };
        let options = SolverOptions {
            print_level: 0,
            ..SolverOptions::default()
        };

        let mut ctx = solve_with_sensitivity(&problem, &options);
        assert!(
            matches!(ctx.result.status, SolveStatus::Optimal),
            "Expected converged, got {:?}",
            ctx.result.status
        );
        assert!(
            (ctx.result.x[0] - 1.0).abs() < 1e-4,
            "x1* should be ~1.0, got {}",
            ctx.result.x[0]
        );
        assert!(
            (ctx.result.x[1] - 1.0).abs() < 1e-4,
            "x2* should be ~1.0, got {}",
            ctx.result.x[1]
        );

        // Sensitivity for Δp = [1.0]
        let dp = [1.0];
        let sens = ctx
            .compute_sensitivity(&problem, &[&dp])
            .expect("sensitivity should succeed");

        // dx1/dp = dx2/dp = 0.5
        assert!(
            (sens.dx_dp[0][0] - 0.5).abs() < 1e-4,
            "dx1/dp should be ~0.5, got {}",
            sens.dx_dp[0][0]
        );
        assert!(
            (sens.dx_dp[0][1] - 0.5).abs() < 1e-4,
            "dx2/dp should be ~0.5, got {}",
            sens.dx_dp[0][1]
        );

        // Verify by finite differences
        let problem2 = ConstrainedParametricQP { p: 2.001 };
        let result2 = crate::solve(&problem2, &options);
        let fd0 = (result2.x[0] - ctx.result.x[0]) / 0.001;
        let fd1 = (result2.x[1] - ctx.result.x[1]) / 0.001;
        assert!(
            (fd0 - sens.dx_dp[0][0]).abs() < 1e-2,
            "FD dx1/dp = {}, analytical = {}",
            fd0,
            sens.dx_dp[0][0]
        );
        assert!(
            (fd1 - sens.dx_dp[0][1]).abs() < 1e-2,
            "FD dx2/dp = {}, analytical = {}",
            fd1,
            sens.dx_dp[0][1]
        );
    }

    #[test]
    fn test_reduced_hessian_unconstrained() {
        let problem = ParametricQP { p: 2.0 };
        let options = SolverOptions {
            print_level: 0,
            ..SolverOptions::default()
        };

        let mut ctx = solve_with_sensitivity(&problem, &options);
        let rh = ctx.reduced_hessian().expect("reduced hessian should succeed");

        // For min 0.5*x^2 - p*x, the Hessian is 1.0,
        // so the reduced Hessian inverse (what we return) should be ~1.0
        assert!(
            (rh[0][0] - 1.0).abs() < 1e-4,
            "Reduced Hessian inverse should be ~1.0, got {}",
            rh[0][0]
        );
    }

    #[test]
    fn test_multiple_perturbations() {
        let problem = ParametricQP { p: 2.0 };
        let options = SolverOptions {
            print_level: 0,
            ..SolverOptions::default()
        };

        let mut ctx = solve_with_sensitivity(&problem, &options);

        let dp1 = [1.0];
        let dp2 = [0.5];
        let dp3 = [-1.0];
        let sens = ctx
            .compute_sensitivity(&problem, &[&dp1, &dp2, &dp3])
            .expect("sensitivity should succeed");

        assert_eq!(sens.dx_dp.len(), 3);
        // All should have dx/dp ≈ 1.0 (times Δp gives the perturbation)
        // The sensitivity dx/dp is constant for this linear problem
        assert!((sens.dx_dp[0][0] - 1.0).abs() < 1e-4);
        assert!((sens.dx_dp[1][0] - 0.5).abs() < 1e-4);
        assert!((sens.dx_dp[2][0] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_sensitivity_predicts_perturbed_solution() {
        // Integration test: solve, predict via sensitivity, verify against re-solve
        let problem = ConstrainedParametricQP { p: 3.0 };
        let options = SolverOptions {
            print_level: 0,
            ..SolverOptions::default()
        };

        let mut ctx = solve_with_sensitivity(&problem, &options);

        // Predict solution at p = 3.1
        let delta = 0.1;
        let dp = [delta];
        let sens = ctx
            .compute_sensitivity(&problem, &[&dp])
            .expect("sensitivity should succeed");

        let x_predicted: Vec<f64> = ctx
            .result
            .x
            .iter()
            .zip(sens.dx_dp[0].iter())
            .map(|(x, dx)| x + dx)
            .collect();

        // Actually solve at p = 3.1
        let problem2 = ConstrainedParametricQP { p: 3.0 + delta };
        let result2 = crate::solve(&problem2, &options);

        for i in 0..2 {
            assert!(
                (x_predicted[i] - result2.x[i]).abs() < 1e-4,
                "Predicted x[{}] = {}, actual = {}",
                i,
                x_predicted[i],
                result2.x[i]
            );
        }
    }
}
