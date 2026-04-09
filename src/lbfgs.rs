//! L-BFGS unconstrained minimizer.
//!
//! Used as a fallback for unconstrained problems where IPM with mu=mu_min
//! gets stuck (wrong basin due to pathological Hessians). L-BFGS builds
//! a positive-definite curvature approximation from gradients alone.

use crate::logging::rip_log;
use crate::options::SolverOptions;
use crate::problem::NlpProblem;
use crate::result::{SolveResult, SolveStatus};

/// L-BFGS memory size.
const LBFGS_M: usize = 10;

/// Wolfe line search parameters.
const C1: f64 = 1e-4;
const C2: f64 = 0.9;
const MAX_LS_ITER: usize = 40;

/// Solve an unconstrained (or bound-constrained) problem using L-BFGS.
///
/// Uses only `objective()`, `gradient()`, `bounds()`, and `initial_point()`
/// from the NlpProblem trait. No Hessian is needed.
pub fn solve<P: NlpProblem>(problem: &P, options: &SolverOptions) -> SolveResult {
    let n = problem.num_variables();
    let m = problem.num_constraints();

    // Get bounds
    let mut x_l = vec![f64::NEG_INFINITY; n];
    let mut x_u = vec![f64::INFINITY; n];
    problem.bounds(&mut x_l, &mut x_u);

    // Initialize x
    let mut x = vec![0.0; n];
    problem.initial_point(&mut x);
    project_bounds(&mut x, &x_l, &x_u);

    let mut grad = vec![0.0; n];
    let mut f = 0.0;
    if !problem.objective(&x, true, &mut f) || !f.is_finite()
        || !problem.gradient(&x, true, &mut grad) || grad.iter().any(|v| !v.is_finite())
    {
        let g_out = vec![0.0; m];
        return SolveResult {
            x,
            objective: f64::NAN,
            constraint_multipliers: vec![0.0; m],
            bound_multipliers_lower: vec![0.0; n],
            bound_multipliers_upper: vec![0.0; n],
            constraint_values: g_out,
            status: SolveStatus::EvaluationError,
            iterations: 0,
            diagnostics: Default::default(),
        };
    }

    let print_level = options.print_level;
    let tol = options.tol;
    let max_iter = options.max_iter;

    if print_level >= 5 {
        rip_log!(
            "L-BFGS: start, n={}, f={:.6e}, ||g||={:.6e}",
            n,
            f,
            inf_norm(&grad)
        );
    }

    // L-BFGS storage: s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k
    let mut s_store: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_M);
    let mut y_store: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_M);
    let mut rho_store: Vec<f64> = Vec::with_capacity(LBFGS_M);

    let mut status = SolveStatus::MaxIterations;
    let mut acceptable_count = 0;
    let mut iter = 0;
    // Stall detection: track best gradient norm seen
    let mut stall_best_grad: f64 = f64::INFINITY;
    let mut stall_no_progress: usize = 0;
    let stall_limit: usize = 30;

    for k in 0..max_iter {
        iter = k;
        let grad_norm = inf_norm(&grad);

        if grad_norm < tol {
            status = SolveStatus::Optimal;
            if print_level >= 5 {
                rip_log!(
                    "L-BFGS iter {}: converged (optimal), f={:.6e}, ||g||={:.6e}",
                    k, f, grad_norm
                );
            }
            break;
        }

        let near_tol = 100.0 * tol;
        if grad_norm < near_tol {
            acceptable_count += 1;
            if acceptable_count >= 10 {
                // Near-tolerance: trigger promotion strategies in caller
                status = SolveStatus::NumericalError;
                if print_level >= 5 {
                    rip_log!(
                        "L-BFGS iter {}: near-tolerance but not optimal, f={:.6e}, ||g||={:.6e}",
                        k, f, grad_norm
                    );
                }
                break;
            }
        } else {
            acceptable_count = 0;
        }

        // Stall detection: if gradient hasn't improved by 1% in stall_limit iterations,
        // check if the point is near-acceptable and return Acceptable rather than
        // burning the full iteration budget.
        if k > 50 {
            if grad_norm < 0.99 * stall_best_grad {
                stall_best_grad = grad_norm;
                stall_no_progress = 0;
            } else {
                stall_no_progress += 1;
                if stall_no_progress >= stall_limit {
                    // Relaxed acceptable: gradient below absolute or relative threshold.
                    // For ill-conditioned problems (e.g., MEYER3), the gradient can be
                    // large in absolute terms even at the correct solution because the
                    // Hessian eigenvalues span many orders of magnitude. Use |f| as a
                    // reference scale: if ||g||/|f| < 0.5 and the solver is stuck, the
                    // point is a practical minimum.
                    let grad_thresh = (options.tol * 10000.0).max(1.0).max(0.5 * f.abs());
                    if grad_norm < grad_thresh {
                        status = SolveStatus::NumericalError;
                        if print_level >= 5 {
                            rip_log!(
                                "L-BFGS iter {}: stalled but near-acceptable (||g||={:.2e}, thresh={:.2e})",
                                k, grad_norm, grad_thresh
                            );
                        }
                        break;
                    }
                    stall_no_progress = 0; // reset and keep trying
                }
            }
        }

        // Two-loop recursion to compute search direction d = -H_k * grad
        let d = two_loop_recursion(&grad, &s_store, &y_store, &rho_store);

        // Wolfe line search
        let (alpha, f_new, grad_new) =
            match wolfe_line_search(problem, &x, &d, f, &grad, &x_l, &x_u) {
                Some(result) => result,
                None => {
                    // Line search failed — try steepest descent
                    let mut d_sd = grad.clone();
                    for v in d_sd.iter_mut() {
                        *v = -*v;
                    }
                    match wolfe_line_search(problem, &x, &d_sd, f, &grad, &x_l, &x_u) {
                        Some(result) => {
                            // Reset L-BFGS history since we used steepest descent
                            s_store.clear();
                            y_store.clear();
                            rho_store.clear();
                            result
                        }
                        None => {
                            if print_level >= 5 {
                                rip_log!(
                                    "L-BFGS iter {}: line search failed even with steepest descent",
                                    k
                                );
                            }
                            status = SolveStatus::NumericalError;
                            break;
                        }
                    }
                }
            };

        // Compute s_k and y_k
        let mut s_k = vec![0.0; n];
        let mut y_k = vec![0.0; n];
        for i in 0..n {
            let x_new_i = clamp(x[i] + alpha * d[i], x_l[i], x_u[i]);
            s_k[i] = x_new_i - x[i];
            y_k[i] = grad_new[i] - grad[i];
        }

        let sy: f64 = s_k.iter().zip(y_k.iter()).map(|(s, y)| s * y).sum();

        // Only store curvature pair if sy > 0 (positive curvature)
        if sy > 1e-20 {
            if s_store.len() == LBFGS_M {
                s_store.remove(0);
                y_store.remove(0);
                rho_store.remove(0);
            }
            rho_store.push(1.0 / sy);
            s_store.push(s_k);
            y_store.push(y_k);
        }

        // Update x
        for i in 0..n {
            x[i] = clamp(x[i] + alpha * d[i], x_l[i], x_u[i]);
        }
        f = f_new;
        grad.copy_from_slice(&grad_new);

        // Check for tiny steps
        let step_norm: f64 = (0..n)
            .map(|i| (alpha * d[i]).abs())
            .fold(0.0, |a, b| if b > a || b.is_nan() { b } else { a });
        if step_norm < 1e-15 {
            if grad_norm < 100.0 * tol {
                status = SolveStatus::NumericalError;
                if print_level >= 5 {
                    rip_log!("L-BFGS iter {}: step too small ({:.2e}), acceptable", k, step_norm);
                }
                break;
            }
            // Gradient is still large — try steepest descent restart
            if !s_store.is_empty() {
                if print_level >= 5 {
                    rip_log!(
                        "L-BFGS iter {}: step too small ({:.2e}) but ||g||={:.2e}, restarting with steepest descent",
                        k, step_norm, grad_norm
                    );
                }
                s_store.clear();
                y_store.clear();
                rho_store.clear();
                // Try a steepest descent step
                let mut d_sd = grad.clone();
                for v in d_sd.iter_mut() {
                    *v = -*v;
                }
                match wolfe_line_search(problem, &x, &d_sd, f, &grad, &x_l, &x_u) {
                    Some((a_sd, f_sd, g_sd)) => {
                        for i in 0..n {
                            x[i] = clamp(x[i] + a_sd * d_sd[i], x_l[i], x_u[i]);
                        }
                        f = f_sd;
                        grad.copy_from_slice(&g_sd);
                        continue; // Continue L-BFGS loop with fresh history
                    }
                    None => {
                        if print_level >= 5 {
                            rip_log!("L-BFGS iter {}: steepest descent restart also failed", k);
                        }
                        break;
                    }
                }
            } else {
                if print_level >= 5 {
                    rip_log!("L-BFGS iter {}: step too small ({:.2e})", k, step_norm);
                }
                break;
            }
        }

        if print_level >= 5 && k % 100 == 0 {
            rip_log!(
                "L-BFGS iter {}: f={:.6e}, ||g||={:.6e}, alpha={:.2e}",
                k, f, grad_norm, alpha
            );
        }
    }

    // Build result
    let mut g_out = vec![0.0; m];
    if m > 0 {
        let _ = problem.constraints(&x, true, &mut g_out);
    }

    SolveResult {
        x,
        objective: f,
        constraint_multipliers: vec![0.0; m],
        bound_multipliers_lower: vec![0.0; n],
        bound_multipliers_upper: vec![0.0; n],
        constraint_values: g_out,
        status,
        iterations: iter,
        diagnostics: Default::default(),
    }
}

/// Two-loop recursion for L-BFGS direction.
fn two_loop_recursion(
    grad: &[f64],
    s_store: &[Vec<f64>],
    y_store: &[Vec<f64>],
    rho_store: &[f64],
) -> Vec<f64> {
    let n = grad.len();
    let k = s_store.len();

    let mut q = grad.to_vec();
    let mut alpha_vals = vec![0.0; k];

    // First loop: from most recent to oldest
    for i in (0..k).rev() {
        let a: f64 = rho_store[i]
            * s_store[i]
                .iter()
                .zip(q.iter())
                .map(|(s, q)| s * q)
                .sum::<f64>();
        alpha_vals[i] = a;
        for j in 0..n {
            q[j] -= a * y_store[i][j];
        }
    }

    // Initial Hessian approximation: H0 = gamma * I
    // gamma = s^T y / y^T y for most recent pair
    let gamma = if k > 0 {
        let sy: f64 = s_store[k - 1]
            .iter()
            .zip(y_store[k - 1].iter())
            .map(|(s, y)| s * y)
            .sum();
        let yy: f64 = y_store[k - 1].iter().map(|y| y * y).sum();
        if yy > 0.0 {
            sy / yy
        } else {
            1.0
        }
    } else {
        1.0
    };

    let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

    // Second loop: from oldest to most recent
    for i in 0..k {
        let b: f64 = rho_store[i]
            * y_store[i]
                .iter()
                .zip(r.iter())
                .map(|(y, r)| y * r)
                .sum::<f64>();
        for j in 0..n {
            r[j] += s_store[i][j] * (alpha_vals[i] - b);
        }
    }

    // Negate for descent direction
    for v in r.iter_mut() {
        *v = -*v;
    }
    r
}

/// Wolfe line search with cubic interpolation.
///
/// Returns `Some((alpha, f_new, grad_new))` on success, `None` on failure.
fn wolfe_line_search<P: NlpProblem>(
    problem: &P,
    x: &[f64],
    d: &[f64],
    f0: f64,
    grad0: &[f64],
    x_l: &[f64],
    x_u: &[f64],
) -> Option<(f64, f64, Vec<f64>)> {
    let n = x.len();
    let dg0: f64 = grad0.iter().zip(d.iter()).map(|(g, d)| g * d).sum();

    // Direction must be descent
    if dg0 >= 0.0 {
        return None;
    }

    let mut x_trial = vec![0.0; n];
    let mut grad_trial = vec![0.0; n];

    let mut alpha = 1.0;
    let mut alpha_lo = 0.0;
    let mut alpha_hi = f64::INFINITY;
    let mut f_lo = f0;
    let mut dg_lo = dg0;

    for _ in 0..MAX_LS_ITER {
        // Compute trial point with bound projection
        for i in 0..n {
            x_trial[i] = clamp(x[i] + alpha * d[i], x_l[i], x_u[i]);
        }

        let mut f_trial = 0.0;
        let obj_ok = problem.objective(&x_trial, true, &mut f_trial);

        // Treat eval failure or NaN/Inf as Armijo violation
        if !obj_ok || !f_trial.is_finite() {
            alpha_hi = alpha;
            alpha = if alpha_lo > 0.0 {
                (alpha_lo + alpha) / 2.0
            } else {
                alpha * 0.1
            };
            continue;
        }

        if !problem.gradient(&x_trial, true, &mut grad_trial)
            || grad_trial.iter().any(|v| !v.is_finite())
        {
            alpha_hi = alpha;
            alpha = if alpha_lo > 0.0 {
                (alpha_lo + alpha) / 2.0
            } else {
                alpha * 0.1
            };
            continue;
        }
        let dg_trial: f64 = grad_trial.iter().zip(d.iter()).map(|(g, d)| g * d).sum();

        // Check sufficient decrease (Armijo)
        if f_trial > f0 + C1 * alpha * dg0 {
            // Went too far — bracket [alpha_lo, alpha]
            alpha_hi = alpha;
            // Cubic interpolation between alpha_lo and alpha_hi
            alpha = cubic_interp(alpha_lo, alpha_hi, f_lo, f_trial, dg_lo, dg_trial);
            continue;
        }

        // Check curvature condition
        if dg_trial < C2 * dg0 {
            // Not enough curvature — need to go further
            alpha_lo = alpha;
            f_lo = f_trial;
            dg_lo = dg_trial;
            if alpha_hi.is_infinite() {
                alpha *= 2.0;
            } else {
                alpha = cubic_interp(alpha_lo, alpha_hi, f_lo, f_trial, dg_lo, dg_trial);
            }
            continue;
        }

        // Both conditions satisfied
        return Some((alpha, f_trial, grad_trial));
    }

    // Fall back to best Armijo step if we found one
    if alpha_lo > 0.0 {
        for i in 0..n {
            x_trial[i] = clamp(x[i] + alpha_lo * d[i], x_l[i], x_u[i]);
        }
        let mut f_trial = 0.0;
        if !problem.objective(&x_trial, true, &mut f_trial) {
            return None;
        }
        if !problem.gradient(&x_trial, true, &mut grad_trial) {
            return None;
        }
        return Some((alpha_lo, f_trial, grad_trial));
    }

    None
}

/// Cubic interpolation between two points.
fn cubic_interp(
    a_lo: f64,
    a_hi: f64,
    f_lo: f64,
    f_hi: f64,
    dg_lo: f64,
    dg_hi: f64,
) -> f64 {
    let d = a_hi - a_lo;
    if d.abs() < 1e-20 {
        return (a_lo + a_hi) / 2.0;
    }

    let theta = 3.0 * (f_lo - f_hi) / d + dg_lo + dg_hi;
    let s = [theta.abs(), dg_lo.abs(), dg_hi.abs()]
        .iter()
        .cloned()
        .fold(0.0, f64::max);

    let theta_s = theta / s;
    let dg_lo_s = dg_lo / s;
    let dg_hi_s = dg_hi / s;

    let gamma_sq = theta_s * theta_s - dg_lo_s * dg_hi_s;
    if gamma_sq < 0.0 {
        return (a_lo + a_hi) / 2.0;
    }
    let gamma = s * gamma_sq.sqrt();

    let p = gamma - dg_lo + theta;
    let q = gamma - dg_lo + gamma + dg_hi;

    if q.abs() < 1e-20 {
        return (a_lo + a_hi) / 2.0;
    }

    let r = p / q;
    let alpha = a_lo + r * d;

    // Safeguard: keep within bracket
    let lo = a_lo + 0.1 * d;
    let hi = a_lo + 0.9 * d;
    clamp(alpha, lo, hi)
}

#[inline]
fn clamp(val: f64, lo: f64, hi: f64) -> f64 {
    if val < lo {
        lo
    } else if val > hi {
        hi
    } else {
        val
    }
}

fn project_bounds(x: &mut [f64], x_l: &[f64], x_u: &[f64]) {
    for i in 0..x.len() {
        x[i] = clamp(x[i], x_l[i], x_u[i]);
    }
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NlpProblem, SolverOptions};


    /// min (x-3)^2 + (y-4)^2, unconstrained, start at (0,0)
    struct QuadraticProblem;

    impl NlpProblem for QuadraticProblem {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 0 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY;
            x_u[0] = f64::INFINITY;
            x_l[1] = f64::NEG_INFINITY;
            x_u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 0.0;
            x0[1] = 0.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = (x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2);
            true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 2.0 * (x[0] - 3.0);
            grad[1] = 2.0 * (x[1] - 4.0);
            true
        }
        fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 1]) }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor;
            vals[1] = 2.0 * obj_factor;
            true
        }
    }

    #[test]
    fn lbfgs_direct_solve_quadratic() {
        let prob = QuadraticProblem;
        let options = SolverOptions::default();
        let result = solve(&prob, &options);

        assert!(
            result.status == crate::SolveStatus::Optimal,
            "expected Optimal, got {:?}",
            result.status
        );
        assert!((result.x[0] - 3.0).abs() < 1e-4, "x[0]={}, expected ~3.0", result.x[0]);
        assert!((result.x[1] - 4.0).abs() < 1e-4, "x[1]={}, expected ~4.0", result.x[1]);
        assert!(result.objective < 1e-6, "obj={}, expected ~0", result.objective);
    }

    /// min (x-3)^2, x <= 1.0, start at 0
    struct BoundConstrainedProblem;

    impl NlpProblem for BoundConstrainedProblem {
        fn num_variables(&self) -> usize { 1 }
        fn num_constraints(&self) -> usize { 0 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY;
            x_u[0] = 1.0;
        }
        fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool { *obj = (x[0] - 3.0).powi(2); true }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0 * (x[0] - 3.0); true }
        fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor;
            true
        }
    }

    #[test]
    fn lbfgs_bound_constrained() {
        let prob = BoundConstrainedProblem;
        let mut options = SolverOptions::default();
        options.max_iter = 100; // L-BFGS stalls at active bounds (unprojected grad != 0)
        let result = solve(&prob, &options);

        // L-BFGS reaches x=1.0 (the bound) quickly but can't converge because
        // the unprojected gradient is -4.0. Verify x is at the bound.
        assert!((result.x[0] - 1.0).abs() < 1e-6, "x[0]={}, expected ~1.0 (bound)", result.x[0]);
        assert!((result.objective - 4.0).abs() < 1e-6, "obj={}, expected ~4.0", result.objective);
    }
}
