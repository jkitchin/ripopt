use std::time::Instant;

use crate::convergence;
use crate::linear_solver::dense::DenseLdl;
use crate::linear_solver::{LinearSolver, SymmetricMatrix};
use crate::options::SolverOptions;

/// State for the restoration phase.
///
/// When the filter line search fails completely, the restoration phase
/// attempts to find a point that is acceptable to the filter by
/// minimizing constraint violation using Gauss-Newton steps on ||violation||^2.
pub struct RestorationPhase {
    /// Maximum iterations in restoration.
    max_iter: usize,
    /// Whether restoration is currently active.
    active: bool,
}

impl RestorationPhase {
    pub fn new(max_iter: usize) -> Self {
        Self {
            max_iter,
            active: false,
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Attempt restoration: minimize constraint violation subject to bounds.
    ///
    /// Uses Gauss-Newton steps on 0.5*||violation||^2, which provides quadratic
    /// convergence for nonlinear equality constraints (unlike gradient descent
    /// which converges linearly). Falls back to gradient descent if the
    /// Gauss-Newton system is singular.
    ///
    /// Returns (new_x, success) where success indicates whether a point
    /// with sufficiently small constraint violation was found.
    #[allow(clippy::too_many_arguments)]
    pub fn restore(
        &mut self,
        x: &[f64],
        x_l: &[f64],
        x_u: &[f64],
        g_l: &[f64],
        g_u: &[f64],
        jac_rows: &[usize],
        jac_cols: &[usize],
        n: usize,
        m: usize,
        options: &SolverOptions,
        is_acceptable_to_filter: &dyn Fn(f64, f64) -> bool,
        eval_constraints: &dyn Fn(&[f64], &mut [f64]) -> bool,
        eval_jacobian: &dyn Fn(&[f64], &mut [f64]) -> bool,
        eval_objective: Option<&dyn Fn(&[f64], &mut f64) -> bool>,
        deadline: Option<Instant>,
    ) -> (Vec<f64>, bool) {
        self.active = true;

        if m == 0 {
            // No constraints: nothing to restore.
            self.active = false;
            return (x.to_vec(), true);
        }

        let mut x_rest = x.to_vec();
        let mut g = vec![0.0; m];
        let jac_nnz = jac_rows.len();
        let mut jac_vals = vec![0.0; jac_nnz];

        // Record initial constraint violation for stricter success criteria
        if !eval_constraints(&x_rest, &mut g) {
            self.active = false;
            return (x.to_vec(), false);
        }
        let theta_initial = convergence::primal_infeasibility(&g, g_l, g_u);

        // Track consecutive failed line searches — only break after 3 (not 1)
        let mut consecutive_ls_failures: usize = 0;
        // Previous theta for detecting stalled progress
        let mut prev_theta = theta_initial;

        for _iter in 0..self.max_iter {
            // Check deadline every 10 iterations to avoid spending too long
            if _iter % 10 == 0 {
                if let Some(dl) = deadline {
                    if Instant::now() >= dl {
                        break;
                    }
                }
            }

            // Evaluate constraints at current point
            if !eval_constraints(&x_rest, &mut g) { break; }

            // Compute constraint violation
            let theta = convergence::primal_infeasibility(&g, g_l, g_u);

            if theta < options.tol {
                self.active = false;
                return (x_rest, true);
            }

            // Evaluate Jacobian at current point
            if !eval_jacobian(&x_rest, &mut jac_vals) { break; }

            // Compute violation vector for each constraint
            let mut violation = vec![0.0; m];
            let mut active = vec![false; m];
            for i in 0..m {
                if convergence::is_equality_constraint(g_l[i], g_u[i]) {
                    violation[i] = g[i] - g_l[i];
                    active[i] = true;
                } else if g_l[i].is_finite() && g[i] < g_l[i] {
                    violation[i] = g[i] - g_l[i];
                    active[i] = true;
                } else if g_u[i].is_finite() && g[i] > g_u[i] {
                    violation[i] = g[i] - g_u[i];
                    active[i] = true;
                }
            }

            // Collect active constraint indices
            let active_indices: Vec<usize> = (0..m).filter(|&i| active[i]).collect();
            let m_active = active_indices.len();

            if m_active == 0 {
                break;
            }

            // Try Gauss-Newton step with adaptive Levenberg-Marquardt regularization.
            // Skip for large problems: gauss_newton_step forms a dense m_active × m_active
            // matrix (J*J^T) which is O(m²) memory and O(m³) factorization.
            let mut gn_step = None;
            if m_active <= 5000 {
                let mut eps_factor = 1e-8;
                while eps_factor <= 1e-2 {
                    gn_step = self.gauss_newton_step(
                        &jac_rows,
                        &jac_cols,
                        &jac_vals,
                        &violation,
                        &active_indices,
                        n,
                        eps_factor,
                    );
                    if gn_step.is_some() {
                        break;
                    }
                    eps_factor *= 10.0;
                }
            }

            let mut step = match gn_step {
                Some(s) => s,
                None => {
                    // Fall back to gradient descent: step = -J^T * violation
                    let mut grad_step = vec![0.0; n];
                    for (idx, (&row, &col)) in
                        jac_rows.iter().zip(jac_cols.iter()).enumerate()
                    {
                        if active[row] {
                            grad_step[col] -= jac_vals[idx] * violation[row];
                        }
                    }
                    grad_step
                }
            };

            // Proximity regularization: when progress stalls (consecutive LS failures
            // or theta not decreasing), add a pull toward the starting point to prevent
            // wandering. Strength increases with consecutive failures.
            if consecutive_ls_failures > 0 || (theta > 0.95 * prev_theta && _iter > 2) {
                let eta = 1e-4 * (consecutive_ls_failures as f64 + 1.0);
                for i in 0..n {
                    step[i] -= eta * (x_rest[i] - x[i]);
                }
            }
            prev_theta = theta;

            // Normalize step if too large
            let step_norm: f64 = step.iter().map(|s| s * s).sum::<f64>().sqrt();
            if step_norm < 1e-20 {
                break;
            }
            let scale = if step_norm > 10.0 {
                10.0 / step_norm
            } else {
                1.0
            };

            // Backtracking line search on theta
            let mut alpha = scale;
            let mut x_trial = vec![0.0; n];
            let mut g_trial = vec![0.0; m];
            let mut found_decrease = false;

            for _ls in 0..30 {
                for i in 0..n {
                    x_trial[i] = x_rest[i] + alpha * step[i];
                    if x_l[i].is_finite() {
                        x_trial[i] = x_trial[i].max(x_l[i] + 1e-8);
                    }
                    if x_u[i].is_finite() {
                        x_trial[i] = x_trial[i].min(x_u[i] - 1e-8);
                    }
                }

                if !eval_constraints(&x_trial, &mut g_trial) {
                    alpha *= 0.5;
                    continue; // Eval failed, try shorter step
                }
                let theta_trial = convergence::primal_infeasibility(&g_trial, g_l, g_u);

                if theta_trial < (1.0 - 1e-4 * alpha) * theta {
                    x_rest.copy_from_slice(&x_trial);
                    found_decrease = true;
                    break;
                }

                alpha *= 0.5;
            }

            if found_decrease {
                consecutive_ls_failures = 0;
            } else {
                // Gauss-Newton line search failed — try gradient descent as fallback
                // step_gd = -J_a^T * violation_a (steepest descent on 0.5*||violation||^2)
                let mut grad_step = vec![0.0; n];
                for (idx, (&row, &col)) in
                    jac_rows.iter().zip(jac_cols.iter()).enumerate()
                {
                    if active[row] {
                        grad_step[col] -= jac_vals[idx] * violation[row];
                    }
                }

                let gd_norm: f64 = grad_step.iter().map(|s| s * s).sum::<f64>().sqrt();
                if gd_norm < 1e-20 {
                    break;
                }
                let gd_scale = if gd_norm > 10.0 {
                    10.0 / gd_norm
                } else {
                    1.0
                };

                let mut gd_alpha = gd_scale;
                let mut gd_found = false;
                for _ls in 0..30 {
                    for i in 0..n {
                        x_trial[i] = x_rest[i] + gd_alpha * grad_step[i];
                        if x_l[i].is_finite() {
                            x_trial[i] = x_trial[i].max(x_l[i] + 1e-8);
                        }
                        if x_u[i].is_finite() {
                            x_trial[i] = x_trial[i].min(x_u[i] - 1e-8);
                        }
                    }

                    if !eval_constraints(&x_trial, &mut g_trial) {
                        gd_alpha *= 0.5;
                        continue; // Eval failed, try shorter step
                    }
                    let theta_trial = convergence::primal_infeasibility(&g_trial, g_l, g_u);

                    if theta_trial < (1.0 - 1e-4 * gd_alpha) * theta {
                        x_rest.copy_from_slice(&x_trial);
                        gd_found = true;
                        break;
                    }

                    gd_alpha *= 0.5;
                }

                if gd_found {
                    consecutive_ls_failures = 0;
                } else {
                    consecutive_ls_failures += 1;
                    if consecutive_ls_failures >= 3 {
                        break;
                    }
                    // Don't break yet — the Jacobian at the next point may yield a better direction
                }
            }
        }

        // Check constraint violation after GN phase
        if !eval_constraints(&x_rest, &mut g) {
            self.active = false;
            return (x_rest, false);
        }
        let theta_after_gn = convergence::primal_infeasibility(&g, g_l, g_u);

        // If GN didn't achieve adequate reduction, try penalty-regularized fallback.
        // Minimizes 0.5*||violation||^2 + rho*||x - x_ref||^2 with increasing rho.
        // The trust-region-like penalty prevents wandering too far from the starting point.
        if theta_after_gn > options.constr_viol_tol && theta_after_gn > 0.5 * theta_initial {
            let x_ref = x_rest.clone();
            for &rho in &[1e-6, 1e-4, 1e-2] {
                for _pen_iter in 0..50 {
                    if !eval_constraints(&x_rest, &mut g) { break; }
                    let theta_pen = convergence::primal_infeasibility(&g, g_l, g_u);
                    if theta_pen < options.constr_viol_tol {
                        break;
                    }

                    if !eval_jacobian(&x_rest, &mut jac_vals) { break; }

                    // Compute violation
                    let mut violation_pen = vec![0.0; m];
                    let mut any_active = false;
                    for i in 0..m {
                        if convergence::is_equality_constraint(g_l[i], g_u[i]) {
                            violation_pen[i] = g[i] - g_l[i];
                            any_active = true;
                        } else if g_l[i].is_finite() && g[i] < g_l[i] {
                            violation_pen[i] = g[i] - g_l[i];
                            any_active = true;
                        } else if g_u[i].is_finite() && g[i] > g_u[i] {
                            violation_pen[i] = g[i] - g_u[i];
                            any_active = true;
                        }
                    }
                    if !any_active { break; }

                    // Gradient = J^T * violation + rho * (x - x_ref)
                    let mut pen_grad = vec![0.0; n];
                    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
                        pen_grad[col] += jac_vals[idx] * violation_pen[row];
                    }
                    for i in 0..n {
                        pen_grad[i] += rho * (x_rest[i] - x_ref[i]);
                    }

                    let grad_norm: f64 = pen_grad.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if grad_norm < 1e-20 { break; }

                    // Steepest descent with Armijo backtracking
                    let scale = if grad_norm > 10.0 { 10.0 / grad_norm } else { 1.0 };
                    let mut pen_alpha = scale;
                    let mut pen_found = false;
                    let mut x_trial_pen = vec![0.0; n];
                    let mut g_trial_pen = vec![0.0; m];
                    for _ls in 0..20 {
                        for i in 0..n {
                            x_trial_pen[i] = x_rest[i] - pen_alpha * pen_grad[i];
                            if x_l[i].is_finite() {
                                x_trial_pen[i] = x_trial_pen[i].max(x_l[i] + 1e-8);
                            }
                            if x_u[i].is_finite() {
                                x_trial_pen[i] = x_trial_pen[i].min(x_u[i] - 1e-8);
                            }
                        }
                        if !eval_constraints(&x_trial_pen, &mut g_trial_pen) {
                            pen_alpha *= 0.5;
                            continue;
                        }
                        let theta_trial_pen = convergence::primal_infeasibility(&g_trial_pen, g_l, g_u);
                        if theta_trial_pen < (1.0 - 1e-4 * pen_alpha) * theta_pen {
                            x_rest.copy_from_slice(&x_trial_pen);
                            pen_found = true;
                            break;
                        }
                        pen_alpha *= 0.5;
                    }
                    if !pen_found { break; }
                }
            }
        }

        // Check final constraint violation
        if !eval_constraints(&x_rest, &mut g) {
            self.active = false;
            return (x_rest, false);
        }
        let theta_final = convergence::primal_infeasibility(&g, g_l, g_u);

        self.active = false;

        // First-iteration protection: if less than 1% improvement, always fail.
        // The point must have genuinely improved — returning the same point wastes an iteration.
        if theta_initial > options.tol && theta_final >= 0.99 * theta_initial {
            return (x_rest, false);
        }

        // Success criteria:
        // 1. Constraint violation is below solver tolerance (feasible)
        // 2. At least 10% infeasibility reduction (kappa_resto = 0.9) AND either
        //    feasible enough or a meaningful absolute reduction
        let feasible = theta_final < options.constr_viol_tol;
        let kappa_resto_met = theta_final <= 0.9 * theta_initial;
        let large_reduction = theta_final < 0.5 * theta_initial;
        let abs_reduction = (theta_initial - theta_final) > options.tol;

        let mut success = feasible
            || large_reduction
            || (kappa_resto_met && (feasible || abs_reduction));

        // Filter acceptance check: if we have an objective evaluator, verify the
        // restored point is acceptable to the filter before declaring full success.
        if success {
            if let Some(eval_obj) = eval_objective {
                let mut phi_rest = f64::INFINITY;
                if !eval_obj(&x_rest, &mut phi_rest) {
                    // Objective eval failed; treat as filter-rejected
                    if !feasible {
                        success = false;
                    }
                } else if !is_acceptable_to_filter(theta_final, phi_rest) {
                    // Point is not acceptable to the filter.
                    // Still succeed if we achieved feasibility (filter will be reset),
                    // but fail if we only got partial reduction.
                    if !feasible {
                        success = false;
                    }
                }
            }
        }

        (x_rest, success)
    }

    /// Compute Gauss-Newton step: dx = -J_a^T * (J_a * J_a^T + eps*I)^{-1} * v_a
    ///
    /// where J_a is the Jacobian restricted to active (violated) constraints
    /// and v_a is the violation vector for active constraints.
    /// `eps_factor` controls the Levenberg-Marquardt regularization strength.
    fn gauss_newton_step(
        &self,
        jac_rows: &[usize],
        jac_cols: &[usize],
        jac_vals: &[f64],
        violation: &[f64],
        active_indices: &[usize],
        n: usize,
        eps_factor: f64,
    ) -> Option<Vec<f64>> {
        let m_active = active_indices.len();
        if m_active == 0 {
            return None;
        }

        // Map from original constraint index to active index
        let mut active_map = vec![usize::MAX; violation.len()];
        for (ai, &orig) in active_indices.iter().enumerate() {
            active_map[orig] = ai;
        }

        // Form J_a * J_a^T (m_active x m_active)
        // Group Jacobian entries by column for efficient J*J^T computation
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
            if active_map[row] != usize::MAX {
                col_entries[col].push((active_map[row], jac_vals[idx]));
            }
        }

        let mut jjt = SymmetricMatrix::zeros(m_active);
        for col_ents in &col_entries {
            for &(ai, val_i) in col_ents {
                for &(aj, val_j) in col_ents {
                    if ai >= aj {
                        jjt.add(ai, aj, val_i * val_j);
                    }
                }
            }
        }

        // Add Levenberg-Marquardt regularization for numerical stability
        let jjt_diag_max = (0..m_active)
            .map(|i| jjt.get(i, i).abs())
            .fold(0.0f64, f64::max);
        let eps = eps_factor * jjt_diag_max.max(1.0);
        jjt.add_diagonal(eps);

        // Solve (J_a * J_a^T + eps*I) * w = violation_a
        let mut solver = DenseLdl::new();
        if solver.bunch_kaufman_factor(&jjt).is_err() {
            return None;
        }

        let v_active: Vec<f64> = active_indices.iter().map(|&i| violation[i]).collect();
        let mut w = vec![0.0; m_active];
        if solver.solve(&v_active, &mut w).is_err() {
            return None;
        }

        // step = -J_a^T * w
        let mut step = vec![0.0; n];
        for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
            let ai = active_map[row];
            if ai != usize::MAX {
                step[col] -= jac_vals[idx] * w[ai];
            }
        }

        Some(step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;

    fn default_opts() -> SolverOptions {
        SolverOptions {
            print_level: 0,
            ..SolverOptions::default()
        }
    }

    #[test]
    fn test_restoration_no_constraints() {
        let mut phase = RestorationPhase::new(50);
        let x = vec![1.0, 2.0];
        let x_l = vec![f64::NEG_INFINITY; 2];
        let x_u = vec![f64::INFINITY; 2];
        let opts = default_opts();

        let (x_new, success) = phase.restore(
            &x, &x_l, &x_u, &[], &[],
            &[], &[], 2, 0, &opts,
            &|_theta, _phi| true,
            &|_x, _g| true,
            &|_x, _vals| true,
            None,
            None,
        );

        assert!(success, "No constraints → immediate success");
        assert!((x_new[0] - 1.0).abs() < 1e-15);
        assert!((x_new[1] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_restoration_already_feasible() {
        let mut phase = RestorationPhase::new(50);
        let x = vec![0.5, 0.5];
        let x_l = vec![f64::NEG_INFINITY; 2];
        let x_u = vec![f64::INFINITY; 2];
        let g_l = vec![1.0]; // g = x0 + x1 = 1.0 = g_l → feasible
        let g_u = vec![1.0];
        let jac_rows = vec![0, 0];
        let jac_cols = vec![0, 1];
        let opts = default_opts();

        let (_, success) = phase.restore(
            &x, &x_l, &x_u, &g_l, &g_u,
            &jac_rows, &jac_cols, 2, 1, &opts,
            &|_theta, _phi| true,
            &|x, g| { g[0] = x[0] + x[1]; true },
            &|_x, vals| { vals[0] = 1.0; vals[1] = 1.0; true },
            None,
            None,
        );

        assert!(success, "Already feasible → success");
    }

    #[test]
    fn test_restoration_linear_equality() {
        // g(x) = x0 + x1 = 1, from x = (0, 0)
        let mut phase = RestorationPhase::new(100);
        let x = vec![0.0, 0.0];
        let x_l = vec![f64::NEG_INFINITY; 2];
        let x_u = vec![f64::INFINITY; 2];
        let g_l = vec![1.0];
        let g_u = vec![1.0];
        let jac_rows = vec![0, 0];
        let jac_cols = vec![0, 1];
        let opts = default_opts();

        let (x_new, success) = phase.restore(
            &x, &x_l, &x_u, &g_l, &g_u,
            &jac_rows, &jac_cols, 2, 1, &opts,
            &|_theta, _phi| true,
            &|x, g| { g[0] = x[0] + x[1]; true },
            &|_x, vals| { vals[0] = 1.0; vals[1] = 1.0; true },
            None,
            None,
        );

        assert!(success, "Linear equality should be restored");
        let g_val = x_new[0] + x_new[1];
        assert!((g_val - 1.0).abs() < 1e-6,
            "Constraint should be satisfied: g = {}", g_val);
    }

    #[test]
    fn test_restoration_linear_inequality() {
        // g(x) = x0 >= 2, from x = (0)
        let mut phase = RestorationPhase::new(100);
        let x = vec![0.0];
        let x_l = vec![f64::NEG_INFINITY];
        let x_u = vec![f64::INFINITY];
        let g_l = vec![2.0];
        let g_u = vec![f64::INFINITY];
        let jac_rows = vec![0];
        let jac_cols = vec![0];
        let opts = default_opts();

        let (x_new, success) = phase.restore(
            &x, &x_l, &x_u, &g_l, &g_u,
            &jac_rows, &jac_cols, 1, 1, &opts,
            &|_theta, _phi| true,
            &|x, g| { g[0] = x[0]; true },
            &|_x, vals| { vals[0] = 1.0; true },
            None,
            None,
        );

        assert!(success, "Linear inequality should be restored");
        assert!(x_new[0] >= 2.0 - 1e-6,
            "Should satisfy x >= 2, got {}", x_new[0]);
    }

    #[test]
    fn test_restoration_with_bounds() {
        // g(x) = x0 + x1 = 5, from x = (0, 0), with bounds 0 <= xi <= 3
        let mut phase = RestorationPhase::new(100);
        let x = vec![0.5, 0.5];
        let x_l = vec![0.0, 0.0];
        let x_u = vec![3.0, 3.0];
        let g_l = vec![5.0];
        let g_u = vec![5.0];
        let jac_rows = vec![0, 0];
        let jac_cols = vec![0, 1];
        let opts = default_opts();

        let (x_new, _success) = phase.restore(
            &x, &x_l, &x_u, &g_l, &g_u,
            &jac_rows, &jac_cols, 2, 1, &opts,
            &|_theta, _phi| true,
            &|x, g| { g[0] = x[0] + x[1]; true },
            &|_x, vals| { vals[0] = 1.0; vals[1] = 1.0; true },
            None,
            None,
        );

        // Verify bounds are respected
        for i in 0..2 {
            assert!(x_new[i] >= x_l[i], "x[{}] = {} below lower bound", i, x_new[i]);
            assert!(x_new[i] <= x_u[i], "x[{}] = {} above upper bound", i, x_new[i]);
        }
    }

    #[test]
    fn test_restoration_active_flag() {
        let mut phase = RestorationPhase::new(50);
        assert!(!phase.is_active(), "Should not be active initially");

        let x = vec![1.0];
        let x_l = vec![f64::NEG_INFINITY];
        let x_u = vec![f64::INFINITY];
        let opts = default_opts();

        phase.restore(
            &x, &x_l, &x_u, &[], &[],
            &[], &[], 1, 0, &opts,
            &|_theta, _phi| true,
            &|_x, _g| true,
            &|_x, _vals| true,
            None,
            None,
        );

        assert!(!phase.is_active(), "Should not be active after restore completes");
    }
}
