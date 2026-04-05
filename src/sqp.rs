//! Simple SQP (Sequential Quadratic Programming) fallback solver.
//!
//! Equality-constrained SQP: at each iteration, solve the KKT system of the
//! QP subproblem for a search direction, then do backtracking line search
//! with an l1 merit function.
//!
//! Used as a fallback when IPM/AL/slack fail.

use crate::logging::rip_log;
use crate::linear_solver::dense::DenseLdl;
use crate::linear_solver::{KktMatrix, LinearSolver};
use crate::options::SolverOptions;
use crate::problem::NlpProblem;
use crate::result::{SolveResult, SolveStatus};

/// Maximum SQP iterations.
const MAX_SQP_ITER: usize = 500;

/// Solve a constrained NLP using a simple SQP method.
///
/// Handles bound constraints via clamping and equality/inequality constraints
/// via the KKT system. Inequality constraints are treated with target = g_l
/// for lower-bounded, g_u for upper-bounded, or (g_l+g_u)/2 for two-sided.
pub fn solve<P: NlpProblem>(problem: &P, options: &SolverOptions) -> SolveResult {
    let n = problem.num_variables();
    let m = problem.num_constraints();
    let dim = n + m;

    if m == 0 {
        // SQP needs constraints; return failure for unconstrained problems
        return SolveResult {
            x: vec![0.0; n],
            objective: 0.0,
            constraint_multipliers: vec![],
            bound_multipliers_lower: vec![0.0; n],
            bound_multipliers_upper: vec![0.0; n],
            constraint_values: vec![],
            status: SolveStatus::NumericalError,
            iterations: 0,
            diagnostics: Default::default(),
        };
    }

    // Get bounds
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    problem.bounds(&mut x_l, &mut x_u);

    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);

    // Compute constraint targets (for the violation c = g - target)
    let mut target = vec![0.0; m];
    for i in 0..m {
        if (g_l[i] - g_u[i]).abs() < 1e-20 {
            target[i] = g_l[i]; // equality
        } else if g_l[i].is_finite() && g_u[i].is_finite() {
            target[i] = (g_l[i] + g_u[i]) / 2.0;
        } else if g_l[i].is_finite() {
            target[i] = g_l[i];
        } else if g_u[i].is_finite() {
            target[i] = g_u[i];
        }
    }

    // Initialize
    let mut x = vec![0.0; n];
    problem.initial_point(&mut x);
    // Clamp to bounds
    for i in 0..n {
        if x_l[i].is_finite() {
            x[i] = x[i].max(x_l[i]);
        }
        if x_u[i].is_finite() {
            x[i] = x[i].min(x_u[i]);
        }
    }

    let mut lambda = vec![0.0; m];

    // Buffers
    let mut grad_f = vec![0.0; n];
    let mut g = vec![0.0; m];
    let mut c = vec![0.0; m]; // constraint violation
    let (jac_rows, jac_cols) = problem.jacobian_structure();
    let jac_nnz = jac_rows.len();
    let mut jac_vals = vec![0.0; jac_nnz];
    let (hess_rows, hess_cols) = problem.hessian_structure();
    let hess_nnz = hess_rows.len();
    let mut hess_vals = vec![0.0; hess_nnz];

    let max_iter = MAX_SQP_ITER.min(options.max_iter);
    let tol = options.tol;

    let mut status = SolveStatus::MaxIterations;
    let mut iter = 0;

    for k in 0..max_iter {
        iter = k + 1;

        // Evaluate functions
        let mut f = 0.0;
        if !problem.objective(&x, true, &mut f)
            || !problem.gradient(&x, true, &mut grad_f)
            || !problem.constraints(&x, true, &mut g)
            || !problem.jacobian_values(&x, true, &mut jac_vals)
            || !problem.hessian_values(&x, true, 1.0, &lambda, &mut hess_vals)
        {
            status = SolveStatus::EvaluationError;
            break;
        }

        // Compute constraint violation c = g - target
        for i in 0..m {
            c[i] = g[i] - target[i];
        }

        // Compute grad_L = grad_f + J^T * lambda
        let mut grad_l = grad_f.clone();
        for (k_jac, (&r, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
            grad_l[col] += jac_vals[k_jac] * lambda[r];
        }

        // Check convergence
        let grad_l_inf = grad_l.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let c_inf = c.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        if options.print_level >= 7 {
            rip_log!(
                "  SQP iter {}: f={:.6e}, ||grad_L||={:.2e}, ||c||={:.2e}",
                k, f, grad_l_inf, c_inf
            );
        }

        if grad_l_inf < tol && c_inf < tol {
            status = SolveStatus::Optimal;
            break;
        }
        if grad_l_inf < 100.0 * options.tol && c_inf < 10.0 * options.constr_viol_tol {
            status = SolveStatus::NumericalError;
            break;
        }

        // Assemble KKT system:
        // [H  J'] [dx    ] = [-grad_L]
        // [J  0 ] [dlambda]   [-c     ]
        let mut kkt = KktMatrix::zeros_dense(dim);

        // H block (upper-left n×n) from Hessian values
        for (k_h, (&r, &col)) in hess_rows.iter().zip(hess_cols.iter()).enumerate() {
            if r >= col {
                kkt.add(r, col, hess_vals[k_h]);
            } else {
                kkt.add(col, r, hess_vals[k_h]);
            }
        }

        // J block (lower-left m×n): J[i,j] goes to kkt[n+i, j]
        for (k_jac, (&r, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
            let kkt_row = n + r;
            let kkt_col = col;
            if kkt_row >= kkt_col {
                kkt.add(kkt_row, kkt_col, jac_vals[k_jac]);
            } else {
                kkt.add(kkt_col, kkt_row, jac_vals[k_jac]);
            }
        }

        // Inertia correction: add delta*I to H block if needed
        let mut delta = 0.0;
        let mut factored = false;
        let mut solver = DenseLdl::new();

        for attempt in 0..20 {
            let mut kkt_try = kkt.clone();
            if delta > 0.0 {
                kkt_try.add_diagonal_range(0, n, delta);
                kkt_try.add_diagonal_range(n, dim, -1e-8);
            }

            if let Ok(Some(inertia)) = solver.factor(&kkt_try) {
                if inertia.positive == n && inertia.negative == m && inertia.zero == 0 {
                    factored = true;
                    break;
                }
            }

            if attempt == 0 {
                delta = 1e-4;
            } else {
                delta *= 10.0;
            }
            if delta > 1e10 {
                break;
            }
        }

        if !factored {
            status = SolveStatus::NumericalError;
            break;
        }

        // RHS = [-grad_L; -c]
        let mut rhs = vec![0.0; dim];
        for i in 0..n {
            rhs[i] = -grad_l[i];
        }
        for i in 0..m {
            rhs[n + i] = -c[i];
        }

        let mut sol = vec![0.0; dim];
        let _ = solver.solve(&rhs, &mut sol);

        let dx = &sol[..n];
        let lambda_new: Vec<f64> = sol[n..].iter().zip(lambda.iter()).map(|(dl, l)| l + dl).collect();

        // l1 merit function line search
        let rho = lambda_new
            .iter()
            .map(|l| l.abs())
            .fold(0.0f64, f64::max)
            + 1.0;

        let phi0 = f + rho * c.iter().map(|ci| ci.abs()).sum::<f64>();
        let dphi0 = grad_f.iter().zip(dx.iter()).map(|(g, d)| g * d).sum::<f64>()
            - rho * c.iter().map(|ci| ci.abs()).sum::<f64>();

        let mut alpha = 1.0;
        let mut x_trial = vec![0.0; n];
        let mut g_trial = vec![0.0; m];
        let eta = 1e-4;
        let mut ls_success = false;

        for _ls in 0..30 {
            for i in 0..n {
                x_trial[i] = x[i] + alpha * dx[i];
                if x_l[i].is_finite() {
                    x_trial[i] = x_trial[i].max(x_l[i]);
                }
                if x_u[i].is_finite() {
                    x_trial[i] = x_trial[i].min(x_u[i]);
                }
            }

            let mut f_trial = f64::INFINITY;
            let eval_ok = problem.objective(&x_trial, true, &mut f_trial);
            problem.constraints(&x_trial, true, &mut g_trial);
            if !eval_ok {
                alpha *= 0.5;
                if alpha < 1e-16 {
                    break;
                }
                continue;
            }
            let c_trial_norm: f64 = g_trial
                .iter()
                .zip(target.iter())
                .map(|(gi, ti)| (gi - ti).abs())
                .sum();
            let phi_trial = f_trial + rho * c_trial_norm;

            if phi_trial <= phi0 + eta * alpha * dphi0 || phi_trial <= phi0 - 1e-10 {
                ls_success = true;
                break;
            }

            alpha *= 0.5;
            if alpha < 1e-16 {
                break;
            }
        }

        if !ls_success {
            // Accept step anyway with smallest alpha tried
            for i in 0..n {
                x_trial[i] = x[i] + alpha * dx[i];
                if x_l[i].is_finite() {
                    x_trial[i] = x_trial[i].max(x_l[i]);
                }
                if x_u[i].is_finite() {
                    x_trial[i] = x_trial[i].min(x_u[i]);
                }
            }
        }

        x.copy_from_slice(&x_trial);
        lambda.copy_from_slice(&lambda_new);
    }

    // Final evaluation
    let mut obj = 0.0;
    problem.objective(&x, true, &mut obj);
    problem.constraints(&x, true, &mut g);

    SolveResult {
        x,
        objective: obj,
        constraint_multipliers: lambda,
        bound_multipliers_lower: vec![0.0; n],
        bound_multipliers_upper: vec![0.0; n],
        constraint_values: g,
        status,
        iterations: iter,
        diagnostics: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    /// HS071: min x0*x3*(x0+x1+x2) + x2
    /// s.t. x0*x1*x2*x3 >= 25, x0^2+x1^2+x2^2+x3^2 = 40
    /// 1 <= xi <= 5
    struct HS071;

    impl NlpProblem for HS071 {
        fn num_variables(&self) -> usize {
            4
        }
        fn num_constraints(&self) -> usize {
            2
        }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            for i in 0..4 {
                x_l[i] = 1.0;
                x_u[i] = 5.0;
            }
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 25.0;
            g_u[0] = f64::INFINITY;
            g_l[1] = 40.0;
            g_u[1] = 40.0;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
            x0[1] = 5.0;
            x0[2] = 5.0;
            x0[3] = 1.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool { *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]; true }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
            grad[1] = x[0] * x[3];
            grad[2] = x[0] * x[3] + 1.0;
            grad[3] = x[0] * (x[0] + x[1] + x[2]);
            true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] * x[1] * x[2] * x[3];
            g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (
                vec![0, 0, 0, 0, 1, 1, 1, 1],
                vec![0, 1, 2, 3, 0, 1, 2, 3],
            )
        }
        fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = x[1] * x[2] * x[3];
            vals[1] = x[0] * x[2] * x[3];
            vals[2] = x[0] * x[1] * x[3];
            vals[3] = x[0] * x[1] * x[2];
            vals[4] = 2.0 * x[0];
            vals[5] = 2.0 * x[1];
            vals[6] = 2.0 * x[2];
            vals[7] = 2.0 * x[3];
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (
                vec![0, 1, 1, 2, 2, 3, 3, 3, 3],
                vec![0, 0, 1, 0, 2, 0, 1, 2, 3],
            )
        }
        fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = obj_factor * 2.0 * x[3] + lambda[1] * 2.0; // d2f/dx0^2
            vals[1] = obj_factor * x[3] + lambda[0] * x[2] * x[3]; // d2f/dx0dx1
            vals[2] = lambda[1] * 2.0; // d2f/dx1^2
            vals[3] = obj_factor * x[3] + lambda[0] * x[1] * x[3]; // d2f/dx0dx2
            vals[4] = lambda[1] * 2.0; // d2f/dx2^2
            vals[5] = obj_factor * (2.0 * x[0] + x[1] + x[2]) + lambda[0] * x[1] * x[2]; // d2f/dx0dx3
            vals[6] = obj_factor * x[0] + lambda[0] * x[0] * x[2]; // d2f/dx1dx3
            vals[7] = obj_factor * x[0] + lambda[0] * x[0] * x[1]; // d2f/dx2dx3
            vals[8] = lambda[1] * 2.0; // d2f/dx3^2
            true
        }
    }

    #[test]
    fn test_sqp_hs071() {
        let prob = HS071;
        let opts = SolverOptions {
            print_level: 0,
            ..Default::default()
        };
        let result = solve(&prob, &opts);
        // SQP is a fallback — for inequality problems it may not converge tightly
        // but should get within 5% of the optimal (obj ≈ 17.014)
        assert!(
            (result.objective - 17.014).abs() < 0.5,
            "SQP obj={:.4e}, expected ~17.014 (±3%)",
            result.objective
        );
    }

    /// Simple equality-constrained QP: min x0^2 + x1^2, s.t. x0 + x1 = 1
    struct SimpleQP;

    impl NlpProblem for SimpleQP {
        fn num_variables(&self) -> usize {
            2
        }
        fn num_constraints(&self) -> usize {
            1
        }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY;
            x_u[0] = f64::INFINITY;
            x_l[1] = f64::NEG_INFINITY;
            x_u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 1.0;
            g_u[0] = 1.0;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 0.0;
            x0[1] = 0.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool { *obj = x[0] * x[0] + x[1] * x[1]; true }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 2.0 * x[0];
            grad[1] = 2.0 * x[1];
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
            vals[0] = 2.0 * obj_factor;
            vals[1] = 2.0 * obj_factor;
            true
        }
    }

    #[test]
    fn test_sqp_simple_qp() {
        let prob = SimpleQP;
        let opts = SolverOptions {
            print_level: 0,
            ..Default::default()
        };
        let result = solve(&prob, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        // Optimal: x = (0.5, 0.5), obj = 0.5
        assert!((result.x[0] - 0.5).abs() < 1e-6);
        assert!((result.x[1] - 0.5).abs() < 1e-6);
        assert!((result.objective - 0.5).abs() < 1e-6);
    }
}
