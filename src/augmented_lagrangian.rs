//! Augmented Lagrangian method for constrained problems.
//!
//! Used as a fallback for constrained problems where IPM stalls.
//! The method decouples constraints using penalty and multiplier updates,
//! solving unconstrained subproblems with L-BFGS.
//!
//! Handles both equality and inequality constraints:
//! - Equalities (g_l == g_u): penalize g(x) - target
//! - Inequalities: penalize max(g_l - g(x), 0) and max(g(x) - g_u, 0)

use crate::lbfgs;
use crate::options::SolverOptions;
use crate::problem::NlpProblem;
use crate::result::{SolveResult, SolveStatus};
use crate::logging::rip_log;

/// Maximum outer iterations for the augmented Lagrangian loop.
const MAX_OUTER_ITER: usize = 50;

/// Initial penalty parameter.
const RHO_INIT: f64 = 10.0;

/// Penalty increase factor when violation is not decreasing fast enough.
const RHO_INCREASE: f64 = 5.0;

/// Maximum penalty parameter.
const RHO_MAX: f64 = 1e10;

/// Compute constraint violation for each constraint, considering bounds.
/// For equalities (g_l == g_u): violation = g - g_l
/// For inequalities: violation = max(g_l - g, 0) - max(g - g_u, 0)
///   (positive when g < g_l, negative when g > g_u, zero when feasible)
fn constraint_violation_i(g_i: f64, g_l_i: f64, g_u_i: f64) -> f64 {
    if (g_l_i - g_u_i).abs() < 1e-20 {
        // Equality constraint
        g_i - g_l_i
    } else {
        // Inequality: return signed violation
        if g_l_i.is_finite() && g_i < g_l_i {
            g_i - g_l_i // negative
        } else if g_u_i.is_finite() && g_i > g_u_i {
            g_i - g_u_i // positive
        } else {
            0.0 // feasible
        }
    }
}

/// For inequality constraints, the effective multiplier/violation for the
/// augmented Lagrangian uses: c_hat = max(c + lambda/rho, 0) when lower-bounded,
/// min(c + lambda/rho, 0) when upper-bounded (Powell-Hestenes-Rockafellar).
fn effective_sigma(
    g_i: f64,
    g_l_i: f64,
    g_u_i: f64,
    lambda_i: f64,
    rho: f64,
) -> f64 {
    if (g_l_i - g_u_i).abs() < 1e-20 {
        // Equality: sigma = lambda + rho * (g - target)
        lambda_i + rho * (g_i - g_l_i)
    } else {
        // Inequality: use PHR formulation
        // For lower bound: penalize only when violated or lambda pulls
        // For upper bound: penalize only when violated or lambda pulls
        let mut sigma = 0.0;
        if g_l_i.is_finite() {
            let c_l = g_l_i - g_i; // positive when violated
            let s_l = (lambda_i + rho * c_l).max(0.0);
            sigma -= s_l; // gradient contribution: -J^T * s_l (push g up)
        }
        if g_u_i.is_finite() {
            let c_u = g_i - g_u_i; // positive when violated
            let s_u = (lambda_i + rho * c_u).max(0.0);
            sigma += s_u; // gradient contribution: +J^T * s_u (push g down)
        }
        sigma
    }
}

/// Wrapper that presents the augmented Lagrangian function as an unconstrained problem.
struct AugLagProblem<'a, P: NlpProblem> {
    inner: &'a P,
    lambda: Vec<f64>,
    rho: f64,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    n: usize,
    m: usize,
}

impl<P: NlpProblem> NlpProblem for AugLagProblem<'_, P> {
    fn num_variables(&self) -> usize {
        self.n
    }

    fn num_constraints(&self) -> usize {
        0
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        self.inner.bounds(x_l, x_u);
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        self.inner.initial_point(x0);
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let mut g = vec![0.0; self.m];
        self.inner.constraints(x, _new_x, &mut g);
        let f = self.inner.objective(x, _new_x);

        let mut aug = f;
        for i in 0..self.m {
            let is_eq = (self.g_l[i] - self.g_u[i]).abs() < 1e-20;
            if is_eq {
                let c = g[i] - self.g_l[i];
                aug += self.lambda[i] * c + 0.5 * self.rho * c * c;
            } else {
                // PHR for inequalities
                if self.g_l[i].is_finite() {
                    let c_l = self.g_l[i] - g[i];
                    let s_l = (self.lambda[i] + self.rho * c_l).max(0.0);
                    aug += s_l * c_l + 0.5 * s_l * s_l / self.rho.max(1e-20);
                    // Simplified: (1/(2*rho)) * (max(lambda + rho*c, 0)^2 - lambda^2)
                }
                if self.g_u[i].is_finite() {
                    let c_u = g[i] - self.g_u[i];
                    let s_u = (self.lambda[i] + self.rho * c_u).max(0.0);
                    aug += s_u * c_u + 0.5 * s_u * s_u / self.rho.max(1e-20);
                }
            }
        }
        aug
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let m = self.m;
        let n = self.n;

        self.inner.gradient(x, _new_x, grad);

        let mut g = vec![0.0; m];
        self.inner.constraints(x, _new_x, &mut g);

        let (jac_rows, jac_cols) = self.inner.jacobian_structure();
        let nnz = jac_rows.len();
        let mut jac_vals = vec![0.0; nnz];
        self.inner.jacobian_values(x, _new_x, &mut jac_vals);

        let sigma: Vec<f64> = (0..m)
            .map(|i| {
                effective_sigma(g[i], self.g_l[i], self.g_u[i], self.lambda[i], self.rho)
            })
            .collect();

        for k in 0..nnz {
            let row = jac_rows[k];
            let col = jac_cols[k];
            if col < n {
                grad[col] += jac_vals[k] * sigma[row];
            }
        }
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) {}
}

/// Solve a constrained problem using the Augmented Lagrangian method
/// with L-BFGS as the inner solver.
pub fn solve<P: NlpProblem>(problem: &P, options: &SolverOptions) -> SolveResult {
    let n = problem.num_variables();
    let m = problem.num_constraints();

    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);

    let mut lambda = vec![0.0; m];
    let mut rho = RHO_INIT;

    let mut x_current = vec![0.0; n];
    problem.initial_point(&mut x_current);

    let print_level = options.print_level;
    // Use constr_viol_tol for the outer violation check, not tol.
    // tol (1e-8) is too tight: the AL stabilizes at ~1e-6 violation which is
    // practically feasible but never beats 1e-8, wasting all 50 outer iterations.
    let tol = options.constr_viol_tol.max(options.tol);

    let mut total_iters = 0;
    let mut prev_violation = f64::INFINITY;

    for outer in 0..MAX_OUTER_ITER {
        let al_problem = AugLagProblem {
            inner: problem,
            lambda: lambda.clone(),
            rho,
            g_l: g_l.clone(),
            g_u: g_u.clone(),
            n,
            m,
        };

        let mut inner_opts = options.clone();
        inner_opts.max_iter = 1000;
        inner_opts.print_level = 0;

        let inner_result = solve_with_x0(&al_problem, &inner_opts, &x_current);
        total_iters += inner_result.iterations;
        x_current = inner_result.x;

        // Evaluate constraint violation
        let mut g = vec![0.0; m];
        problem.constraints(&x_current, true, &mut g);

        let violation: f64 = (0..m)
            .map(|i| constraint_violation_i(g[i], g_l[i], g_u[i]).abs())
            .fold(0.0, f64::max);

        let f_val = problem.objective(&x_current, false);

        if print_level >= 5 {
            rip_log!(
                "AL outer {}: f={:.6e}, violation={:.6e}, rho={:.2e}",
                outer, f_val, violation, rho
            );
        }

        // Check convergence
        if violation < tol {
            let status = SolveStatus::Optimal;

            if print_level >= 5 {
                rip_log!("AL converged: {:?}", status);
            }

            return SolveResult {
                x: x_current,
                objective: f_val,
                constraint_multipliers: lambda,
                bound_multipliers_lower: vec![0.0; n],
                bound_multipliers_upper: vec![0.0; n],
                constraint_values: g,
                status,
                iterations: total_iters,
                diagnostics: Default::default(),
            };
        }

        // Update multipliers
        for i in 0..m {
            let is_eq = (g_l[i] - g_u[i]).abs() < 1e-20;
            if is_eq {
                lambda[i] += rho * (g[i] - g_l[i]);
            } else {
                // For inequalities, update using PHR rule
                if g_l[i].is_finite() {
                    let c_l = g_l[i] - g[i];
                    lambda[i] = (lambda[i] + rho * c_l).max(0.0);
                }
                if g_u[i].is_finite() {
                    let c_u = g[i] - g_u[i];
                    lambda[i] = (lambda[i] + rho * c_u).max(0.0);
                }
            }
        }

        if violation > 0.25 * prev_violation {
            rho = (rho * RHO_INCREASE).min(RHO_MAX);
        }
        prev_violation = violation;
    }

    // Did not converge
    let f_val = problem.objective(&x_current, false);
    let mut g = vec![0.0; m];
    problem.constraints(&x_current, true, &mut g);

    SolveResult {
        x: x_current,
        objective: f_val,
        constraint_multipliers: lambda,
        bound_multipliers_lower: vec![0.0; n],
        bound_multipliers_upper: vec![0.0; n],
        constraint_values: g,
        status: SolveStatus::MaxIterations,
        iterations: total_iters,
        diagnostics: Default::default(),
    }
}

/// Solve with L-BFGS using a specific starting point.
fn solve_with_x0<P: NlpProblem>(
    problem: &P,
    options: &SolverOptions,
    x0: &[f64],
) -> SolveResult {
    let wrapper = StartPointWrapper {
        inner: problem,
        x0: x0.to_vec(),
    };
    lbfgs::solve(&wrapper, options)
}

/// Wrapper that overrides the initial point.
struct StartPointWrapper<'a, P: NlpProblem> {
    inner: &'a P,
    x0: Vec<f64>,
}

impl<P: NlpProblem> NlpProblem for StartPointWrapper<'_, P> {
    fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }
    fn num_constraints(&self) -> usize {
        self.inner.num_constraints()
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        self.inner.bounds(x_l, x_u);
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        self.inner.constraint_bounds(g_l, g_u);
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0.copy_from_slice(&self.x0);
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        self.inner.objective(x, _new_x)
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        self.inner.gradient(x, _new_x, grad);
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        self.inner.constraints(x, _new_x, g);
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.inner.jacobian_structure()
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        self.inner.jacobian_values(x, _new_x, vals);
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.inner.hessian_structure()
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        self.inner.hessian_values(x, _new_x, obj_factor, lambda, vals);
    }
}
