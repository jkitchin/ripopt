use crate::options::SolverOptions;

/// Check if constraint `i` is an equality constraint (g_l ≈ g_u).
#[inline]
pub fn is_equality_constraint(g_l: f64, g_u: f64) -> bool {
    g_l.is_finite() && g_u.is_finite() && (g_l - g_u).abs() < 1e-15
}

/// Result of a convergence check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Not converged.
    NotConverged,
    /// Converged within desired tolerance.
    Converged,
    /// Converged within acceptable tolerance.
    Acceptable,
    /// Diverging (objective growing unboundedly).
    Diverging,
}

/// Information needed to check convergence.
pub struct ConvergenceInfo {
    /// User-facing primal infeasibility: max raw bound-violation on g.
    /// Equality rows: `|g − g_l|`. Inequality rows: `max(g_l−g, 0) +
    /// max(g−g_u, 0)`. This is what the unscaled (top-level) convergence
    /// gate and the restoration trigger see.
    pub primal_inf: f64,
    /// Slack-coupling primal infeasibility for the scaled (barrier-level)
    /// gate: equality rows `|g − g_l|`, inequality rows `|g − s|` where `s`
    /// is the explicit slack iterate. Mirrors Ipopt's
    /// `IpIpoptCalculatedQuantities::curr_primal_infeasibility`
    /// (`||c||_∞ ∪ ||d − s||_∞`). When the caller has not yet plumbed `s`
    /// through, set this equal to `primal_inf` (degrades to the user-side
    /// residual, never tighter than Ipopt's).
    pub primal_inf_internal: f64,
    /// Dual infeasibility: ||grad_f + J^T y - z_l + z_u||_inf using iterative z.
    pub dual_inf: f64,
    /// Dual infeasibility using iterative z with component-wise scaling (for unscaled gate).
    pub dual_inf_unscaled: f64,
    /// Complementarity error using iterative z.
    pub compl_inf: f64,
    /// Current barrier parameter.
    pub mu: f64,
    /// Current objective value.
    pub objective: f64,
    /// Sum of absolute values of all multipliers (y, z_l, z_u).
    /// Used for Ipopt-style dual residual scaling (s_d).
    pub multiplier_sum: f64,
    /// Total number of multiplier components (m + 2n) — denominator for s_d.
    pub multiplier_count: usize,
    /// Sum of absolute values of bound multipliers (z_l, z_u) only.
    /// Used for Ipopt-style complementarity scaling (s_c). See
    /// `IpIpoptCalculatedQuantities.cpp:3677-3687` — Ipopt scales the
    /// complementarity error by the average bound-multiplier magnitude
    /// only, not the average over y as well.
    pub bound_multiplier_sum: f64,
    /// Total number of bound-multiplier components (2n) — denominator for s_c.
    pub bound_multiplier_count: usize,
    /// `‖x‖_∞` of the current iterate, used by the divergence gate
    /// against `options.diverging_iterates_tol` (Ipopt
    /// `IpOptErrorConvCheck.cpp:255`). Default `0.0` when an upstream
    /// caller cannot supply x; that value never triggers divergence.
    pub x_max_abs: f64,
}

/// Test whether the iterate meets Ipopt's acceptable-level thresholds,
/// including the relative objective-change gate
/// `|f_k − f_{k-1}| / max(1, |f_k|) ≤ acceptable_obj_change_tol`
/// (`IpOptErrorConvCheck.cpp:115, 322-330`). When `last_obj` is `None`
/// (iteration 0, no previous objective) the obj-change gate is skipped.
pub fn meets_acceptable_thresholds(
    info: &ConvergenceInfo,
    options: &SolverOptions,
    s_d: f64,
    s_c: f64,
    last_obj: Option<f64>,
) -> bool {
    const ACCEPTABLE_TOL: f64 = 1e-6;
    const ACCEPTABLE_DUAL_INF_TOL: f64 = 1e10;
    const ACCEPTABLE_CONSTR_VIOL_TOL: f64 = 1e-2;
    const ACCEPTABLE_COMPL_INF_TOL: f64 = 1e-2;

    let scaled_ok = info.primal_inf_internal <= ACCEPTABLE_TOL
        && info.dual_inf <= ACCEPTABLE_TOL * s_d
        && info.compl_inf <= ACCEPTABLE_TOL * s_c;
    let unscaled_ok = info.primal_inf <= ACCEPTABLE_CONSTR_VIOL_TOL
        && info.dual_inf_unscaled <= ACCEPTABLE_DUAL_INF_TOL
        && info.compl_inf <= ACCEPTABLE_COMPL_INF_TOL;

    let obj_change_ok = match last_obj {
        Some(prev) => {
            let denom = info.objective.abs().max(1.0);
            (info.objective - prev).abs() / denom <= options.acceptable_obj_change_tol
        }
        None => true,
    };

    scaled_ok && unscaled_ok && obj_change_ok
}

/// Acceptable-level thresholds matching Ipopt defaults
/// (`IpOptErrorConvCheck.cpp:70-121`):
///   acceptable_tol = 1e-6, acceptable_dual_inf_tol = 1e10,
///   acceptable_constr_viol_tol = 1e-2, acceptable_compl_inf_tol = 1e-2.
pub const ACCEPTABLE_TOL: f64 = 1e-6;
pub const ACCEPTABLE_DUAL_INF_TOL: f64 = 1e10;
pub const ACCEPTABLE_CONSTR_VIOL_TOL: f64 = 1e-2;
pub const ACCEPTABLE_COMPL_INF_TOL: f64 = 1e-2;
/// Number of consecutive acceptable iterations required for the
/// `Acceptable` exit (Ipopt's `acceptable_iter`).
pub const NEAR_TOL_ITERS: usize = 15;

/// Check convergence of the IPM algorithm.
///
/// Returns the convergence status based on current optimality measures.
pub fn check_convergence(
    info: &ConvergenceInfo,
    options: &SolverOptions,
    consecutive_acceptable: usize,
) -> ConvergenceStatus {
    check_convergence_with_last_obj(info, options, consecutive_acceptable, None)
}

/// Variant of `check_convergence` that threads the previous iterate's
/// objective for the acceptable-level relative-change gate. Iteration 0
/// callers pass `None`; subsequent iterations pass the prior `f`.
pub fn check_convergence_with_last_obj(
    info: &ConvergenceInfo,
    options: &SolverOptions,
    consecutive_acceptable: usize,
    last_obj: Option<f64>,
) -> ConvergenceStatus {
    // Ipopt-style scaling factors (IpIpoptCalculatedQuantities.cpp:3663-3700):
    //   s_d = max(s_max, sum|y, z_l, z_u| / (m+2n)) / s_max  — for dual residual
    //   s_c = max(s_max, sum|z_l, z_u|    / (2n))    / s_max  — for complementarity
    // Both clamp from below to s_max (so scaling never amplifies residuals)
    // but have no upper cap; trusting the multiplier magnitudes is
    // intentional and matches Ipopt.
    let s_max: f64 = 100.0;
    let s_d = if info.multiplier_count > 0 {
        s_max.max(info.multiplier_sum / info.multiplier_count as f64) / s_max
    } else {
        1.0
    };
    let s_c = if info.bound_multiplier_count > 0 {
        s_max.max(info.bound_multiplier_sum / info.bound_multiplier_count as f64) / s_max
    } else {
        1.0
    };

    let primal_tol = options.tol;
    let dual_tol = options.tol * s_d;
    let compl_tol = options.tol * s_c;

    // Strict convergence: BOTH scaled AND unscaled must pass.
    // Scaled (barrier-level) uses Ipopt's `||c||_∞ ∪ ||d − s||_∞` slack-
    // coupling residual; unscaled (top-level/restoration) uses the user-
    // facing raw bound violation on g.
    let scaled_ok = info.primal_inf_internal <= primal_tol
        && info.dual_inf <= dual_tol
        && info.compl_inf <= compl_tol;
    let unscaled_ok = info.primal_inf <= options.constr_viol_tol
        && info.dual_inf_unscaled <= options.dual_inf_tol
        && info.compl_inf <= options.compl_inf_tol;
    if scaled_ok && unscaled_ok {
        return ConvergenceStatus::Converged;
    }

    // `acceptable_iter == 0` disables the acceptable exit entirely
    // (Ipopt 3.14 `IpOptErrorConvCheck.cpp:241`).
    if options.acceptable_iter > 0
        && meets_acceptable_thresholds(info, options, s_d, s_c, last_obj)
        && consecutive_acceptable >= options.acceptable_iter
    {
        return ConvergenceStatus::Acceptable;
    }

    // Divergence gate: ‖x‖_∞ > diverging_iterates_tol (Ipopt 3.14
    // IpOptErrorConvCheck.cpp:255). Earlier ripopt tested |f| > 1e50,
    // which fired on legitimate large-objective constrained problems.
    if info.x_max_abs > options.diverging_iterates_tol {
        return ConvergenceStatus::Diverging;
    }

    ConvergenceStatus::NotConverged
}

/// Compute primal infeasibility (constraint violation) using 1-norm.
///
/// Returns the sum of absolute constraint violations, matching Ipopt's default
/// `constraint_violation_norm_type = "1-norm"`. The 1-norm is critical for the
/// filter line search: with max-norm, theta values are much smaller for problems
/// with many constraints, making the filter thresholds (theta_min, theta_max,
/// gamma_phi * theta) too tight and causing step rejection.
pub fn primal_infeasibility(g: &[f64], g_l: &[f64], g_u: &[f64]) -> f64 {
    let mut sum_viol = 0.0f64;
    for i in 0..g.len() {
        if g[i] < g_l[i] {
            sum_viol += g_l[i] - g[i];
        }
        if g[i] > g_u[i] {
            sum_viol += g[i] - g_u[i];
        }
    }
    sum_viol
}

/// Compute primal infeasibility using max-norm (infinity norm).
///
/// Returns the maximum absolute constraint violation across all constraints.
/// Used for convergence testing where per-constraint satisfaction matters,
/// while the 1-norm variant is used for filter line search decisions.
pub fn primal_infeasibility_max(g: &[f64], g_l: &[f64], g_u: &[f64]) -> f64 {
    let mut max_viol = 0.0f64;
    for i in 0..g.len() {
        if g[i] < g_l[i] {
            max_viol = max_viol.max(g_l[i] - g[i]);
        }
        if g[i] > g_u[i] {
            max_viol = max_viol.max(g[i] - g_u[i]);
        }
    }
    max_viol
}

/// Slack-coupling primal infeasibility (Ipopt's barrier-level
/// `inf_pr` = `||c(x)||_∞ ∪ ||d(x) − s||_∞`).
///
/// Equality rows (`g_l == g_u`) contribute `|g[i] − g_l[i]|`.
/// Inequality rows contribute `|g[i] − s[i]|`, since the IPM iterates an
/// explicit slack `s` with `g_l ≤ s ≤ g_u` and the equation
/// `g(x) − s = 0` is the actual constraint the Newton system enforces
/// (`IpIpoptCalculatedQuantities.cpp::curr_primal_infeasibility`,
/// 1-norm overload). The 1-norm flavor is used inside the filter line
/// search; the max-norm flavor (see `primal_infeasibility_internal_max`)
/// is used for the barrier-level convergence test.
pub fn primal_infeasibility_internal(
    g: &[f64],
    s: &[f64],
    g_l: &[f64],
    g_u: &[f64],
) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..g.len() {
        if is_equality_constraint(g_l[i], g_u[i]) {
            sum += (g[i] - g_l[i]).abs();
        } else {
            sum += (g[i] - s[i]).abs();
        }
    }
    sum
}

/// Max-norm variant of [`primal_infeasibility_internal`]; used for the
/// barrier-level convergence test (Ipopt's E_mu primal residual).
pub fn primal_infeasibility_internal_max(
    g: &[f64],
    s: &[f64],
    g_l: &[f64],
    g_u: &[f64],
) -> f64 {
    let mut m = 0.0f64;
    for i in 0..g.len() {
        let r = if is_equality_constraint(g_l[i], g_u[i]) {
            (g[i] - g_l[i]).abs()
        } else {
            (g[i] - s[i]).abs()
        };
        if r > m {
            m = r;
        }
    }
    m
}

/// Compute dual infeasibility: ||grad_f + J^T * lambda - z_l + z_u + kappa_d damping||_inf.
///
/// `grad_f`: gradient of objective
/// `jac_rows`, `jac_cols`, `jac_vals`: Jacobian in COO format
/// `lambda`: constraint multipliers
/// `z_l`, `z_u`: bound multipliers
/// `n`: number of variables
/// `kappa_d`, `mu`, `x_l`, `x_u`: T3.9 — Ipopt's `curr_grad_lag_x` adds
///   `+ kappa_d * mu` for one-sided lower-bound vars and
///   `- kappa_d * mu` for one-sided upper-bound vars
///   (`IpIpoptCalculatedQuantities.cpp:888-899`, default `kappa_d = 1e-5`).
#[allow(clippy::too_many_arguments)]
pub fn dual_infeasibility(
    grad_f: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    lambda: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    n: usize,
    kappa_d: f64,
    mu: f64,
    x_l: &[f64],
    x_u: &[f64],
) -> f64 {
    let mut residual = vec![0.0; n];

    // Start with gradient of objective
    residual[..n].copy_from_slice(&grad_f[..n]);

    // Add J^T * lambda (Ipopt convention: L = f + y^T g)
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        residual[col] += jac_vals[idx] * lambda[row];
    }

    // Subtract z_l and add z_u (bound multipliers)
    for i in 0..n {
        residual[i] -= z_l[i];
        residual[i] += z_u[i];
    }

    // T3.9: kappa_d damping for one-sided-bound vars (matches Ipopt's
    // curr_grad_lag_x).
    if kappa_d > 0.0 {
        for i in 0..n {
            let l_fin = x_l[i].is_finite();
            let u_fin = x_u[i].is_finite();
            if l_fin && !u_fin {
                residual[i] += kappa_d * mu;
            } else if !l_fin && u_fin {
                residual[i] -= kappa_d * mu;
            }
        }
    }

    residual.iter().map(|r| r.abs()).fold(0.0f64, f64::max)
}

/// Compute component-wise scaled dual infeasibility.
///
/// Uses `|r_i| / (1 + |grad_f_i|)` per component, which makes the metric
/// insensitive to gradient magnitude across variables. This prevents
/// poorly-scaled problems from having artificially large unscaled dual
/// infeasibility even when the scaled version is small.
#[allow(clippy::too_many_arguments)]
pub fn dual_infeasibility_scaled(
    grad_f: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    lambda: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    n: usize,
    kappa_d: f64,
    mu: f64,
    x_l: &[f64],
    x_u: &[f64],
) -> f64 {
    // T3.1: dropped the ripopt-specific per-component `(1 + |grad_f_i|)`
    // divisor. Ipopt's `IpIpoptCalculatedQuantities::curr_dual_infeasibility`
    // computes the raw max-norm of `grad_f + J^T y - z_L + z_U` and only
    // applies the global `s_d` scaling at the convergence-test boundary
    // (`IpOptErrorConvCheck.cpp:208`). The per-component normalisation
    // could declare false-Optimal on problems with one large gradient
    // component shadowing uniformly small residuals.
    //
    // T3.9: kappa_d damping for one-sided-bound vars
    // (IpIpoptCalculatedQuantities.cpp:888-899).
    let mut residual = vec![0.0; n];
    residual[..n].copy_from_slice(&grad_f[..n]);
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        residual[col] += jac_vals[idx] * lambda[row];
    }
    for i in 0..n {
        residual[i] -= z_l[i];
        residual[i] += z_u[i];
    }
    if kappa_d > 0.0 {
        for i in 0..n {
            let l_fin = x_l[i].is_finite();
            let u_fin = x_u[i].is_finite();
            if l_fin && !u_fin {
                residual[i] += kappa_d * mu;
            } else if !l_fin && u_fin {
                residual[i] -= kappa_d * mu;
            }
        }
    }
    residual.iter().map(|r| r.abs()).fold(0.0f64, f64::max)
}

/// Compute complementarity error for bound constraints.
/// compl = max_i |x_i * z_l_i| where x_i is near lower bound,
///         max_i |s_u_i * z_u_i| where x_i is near upper bound.
///
/// For the barrier method: complementarity = max(|(x-x_l)*z_l - mu|, |(x_u-x)*z_u - mu|).
pub fn complementarity_error(
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    mu: f64,
) -> f64 {
    let mut max_err = 0.0f64;
    let n = x.len();
    for i in 0..n {
        if x_l[i].is_finite() {
            let slack = x[i] - x_l[i];
            max_err = max_err.max((slack * z_l[i] - mu).abs());
        }
        if x_u[i].is_finite() {
            let slack = x_u[i] - x[i];
            max_err = max_err.max((slack * z_u[i] - mu).abs());
        }
    }
    max_err
}

/// Compute full complementarity error including constraint slack complementarity.
///
/// In addition to variable bound complementarity (x-x_l)*z_l and (x_u-x)*z_u,
/// this also checks constraint slack complementarity for inequality constraints
/// using the dedicated slack-bound multipliers `v_l`, `v_u` (Ipopt's `v_L`,
/// `v_U`):
/// - Lower-bounded: (g(x) - g_l) * v_l\[i\]
/// - Upper-bounded: (g_u - g(x)) * v_u\[i\]
/// - Equality constraints are skipped.
///
/// Mirrors Ipopt's `IpIpoptCalculatedQuantities::curr_complementarity` which
/// sums Asum() over four projection blocks: z_L, z_U, v_L, v_U
/// (`IpIpoptCalculatedQuantities.cpp:2467-2497`). The earlier `max(y,0)` /
/// `max(-y,0)` substitute (T0.3) was an approximation that broke when `y`
/// drifted away from the central path even though `v_l*slack = mu` was
/// preserved by `reset_slack_multipliers` — see the HS32 stuck-compl regression.
#[allow(clippy::too_many_arguments)]
pub fn complementarity_error_full(
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    g: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    v_l: &[f64],
    v_u: &[f64],
    mu: f64,
) -> f64 {
    // Start with variable bound complementarity
    let mut max_err = complementarity_error(x, x_l, x_u, z_l, z_u, mu);

    // Add constraint slack complementarity for inequality constraints.
    // Ipopt iterates an explicit slack s ≥ 0 (kept interior by the line
    // search) so v·s − μ is a meaningful central-path residual. ripopt's
    // implicit-slack formulation lets g(x) drift infeasible during the
    // line search, where s := g − g_l can go negative; v_l (set to
    // μ_ks/max(s,1e-20)) is then huge and the unclamped product is
    // nonsense. Clamp the effective slack at 0 to mirror Ipopt's interior
    // s (any constraint infeasibility is already counted by primal_inf).
    let m = g.len();
    for i in 0..m {
        if is_equality_constraint(g_l[i], g_u[i]) {
            continue;
        }
        if g_l[i].is_finite() {
            let slack = (g[i] - g_l[i]).max(0.0);
            max_err = max_err.max((slack * v_l[i] - mu).abs());
        }
        if g_u[i].is_finite() {
            let slack = (g_u[i] - g[i]).max(0.0);
            max_err = max_err.max((slack * v_u[i] - mu).abs());
        }
    }
    max_err
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primal_infeasibility_feasible() {
        let g = vec![1.5, 3.0];
        let g_l = vec![1.0, 2.0];
        let g_u = vec![2.0, 4.0];
        assert_eq!(primal_infeasibility(&g, &g_l, &g_u), 0.0);
    }

    #[test]
    fn test_primal_infeasibility_violated() {
        let g = vec![0.5, 5.0];
        let g_l = vec![1.0, 2.0];
        let g_u = vec![2.0, 4.0];
        // 1-norm: |1.0 - 0.5| + |5.0 - 4.0| = 0.5 + 1.0 = 1.5
        assert_eq!(primal_infeasibility(&g, &g_l, &g_u), 1.5);
    }

    #[test]
    fn test_convergence_optimal() {
        let info = ConvergenceInfo {
            primal_inf: 1e-10,
            primal_inf_internal: 1e-10,
            dual_inf: 1e-10,
            dual_inf_unscaled: 1e-10,
            compl_inf: 1e-10,
            mu: 1e-11,
            objective: 17.0,
            multiplier_sum: 0.0,
            multiplier_count: 0,
            bound_multiplier_sum: 0.0,
            bound_multiplier_count: 0,
            x_max_abs: 0.0,
        };
        let opts = SolverOptions::default();
        assert_eq!(
            check_convergence(&info, &opts, 0),
            ConvergenceStatus::Converged
        );
    }

    #[test]
    fn test_convergence_not_converged() {
        let info = ConvergenceInfo {
            primal_inf: 1e-3,
            primal_inf_internal: 1e-3,
            dual_inf: 1e-3,
            dual_inf_unscaled: 1e-3,
            compl_inf: 1e-3,
            mu: 0.01,
            objective: 17.0,
            multiplier_sum: 0.0,
            multiplier_count: 0,
            bound_multiplier_sum: 0.0,
            bound_multiplier_count: 0,
            x_max_abs: 0.0,
        };
        let opts = SolverOptions::default();
        assert_eq!(
            check_convergence(&info, &opts, 0),
            ConvergenceStatus::NotConverged
        );
    }

    #[test]
    fn test_convergence_diverging() {
        // T0.6: divergence is gated by ‖x‖_∞ > diverging_iterates_tol,
        // not |f| (Ipopt IpOptErrorConvCheck.cpp:255).
        let info = ConvergenceInfo {
            primal_inf: 1e-3,
            primal_inf_internal: 1e-3,
            dual_inf: 1e-3,
            dual_inf_unscaled: 1e-3,
            compl_inf: 1e-3,
            mu: 1e-11,
            objective: 1.0,
            multiplier_sum: 0.0,
            multiplier_count: 0,
            bound_multiplier_sum: 0.0,
            bound_multiplier_count: 0,
            x_max_abs: 1e25,
        };
        let opts = SolverOptions::default();
        assert_eq!(
            check_convergence(&info, &opts, 0),
            ConvergenceStatus::Diverging
        );
    }

    #[test]
    fn test_convergence_diverging_uses_x_not_obj() {
        // T0.6: |f| huge but ‖x‖_∞ small ⇒ NOT diverging.
        let info = ConvergenceInfo {
            primal_inf: 1e-3, primal_inf_internal: 1e-3,
            dual_inf: 1e-3, dual_inf_unscaled: 1e-3,
            compl_inf: 1e-3, mu: 1e-11, objective: 1e60,
            multiplier_sum: 0.0, multiplier_count: 0,
            bound_multiplier_sum: 0.0, bound_multiplier_count: 0,
            x_max_abs: 1e15,
        };
        let opts = SolverOptions::default();
        assert_ne!(check_convergence(&info, &opts, 0), ConvergenceStatus::Diverging);
    }

    #[test]
    fn test_meets_acceptable_thresholds_obj_change_passes() {
        // T0.5: |Δf| / max(1, |f|) = |10.001 - 10.0| / 10.001 ≈ 1e-4.
        // With acceptable_obj_change_tol = 1e-2, gate passes.
        let info = ConvergenceInfo {
            primal_inf: 1e-7, primal_inf_internal: 1e-7,
            dual_inf: 1e-7, dual_inf_unscaled: 1e-7,
            compl_inf: 1e-7, mu: 0.0, objective: 10.001,
            multiplier_sum: 0.0, multiplier_count: 0,
            bound_multiplier_sum: 0.0, bound_multiplier_count: 0,
            x_max_abs: 0.0,
        };
        let mut opts = SolverOptions::default();
        opts.acceptable_obj_change_tol = 1e-2;
        assert!(meets_acceptable_thresholds(&info, &opts, 1.0, 1.0, Some(10.0)),
            "|Δf|/|f| ≈ 1e-4 ≤ 1e-2 should pass");
    }

    #[test]
    fn test_meets_acceptable_thresholds_obj_change_blocks() {
        // T0.5: |Δf| / max(1, |f|) = |11.0 - 10.0| / 11.0 ≈ 0.091.
        // With acceptable_obj_change_tol = 1e-2, gate blocks.
        let info = ConvergenceInfo {
            primal_inf: 1e-7, primal_inf_internal: 1e-7,
            dual_inf: 1e-7, dual_inf_unscaled: 1e-7,
            compl_inf: 1e-7, mu: 0.0, objective: 11.0,
            multiplier_sum: 0.0, multiplier_count: 0,
            bound_multiplier_sum: 0.0, bound_multiplier_count: 0,
            x_max_abs: 0.0,
        };
        let mut opts = SolverOptions::default();
        opts.acceptable_obj_change_tol = 1e-2;
        assert!(!meets_acceptable_thresholds(&info, &opts, 1.0, 1.0, Some(10.0)),
            "|Δf|/|f| ≈ 0.09 > 1e-2 should block");
    }

    #[test]
    fn test_meets_acceptable_thresholds_no_prev_obj_skips_gate() {
        // T0.5: last_obj = None (iter 0) skips the gate.
        let info = ConvergenceInfo {
            primal_inf: 1e-7, primal_inf_internal: 1e-7,
            dual_inf: 1e-7, dual_inf_unscaled: 1e-7,
            compl_inf: 1e-7, mu: 0.0, objective: 11.0,
            multiplier_sum: 0.0, multiplier_count: 0,
            bound_multiplier_sum: 0.0, bound_multiplier_count: 0,
            x_max_abs: 0.0,
        };
        let mut opts = SolverOptions::default();
        opts.acceptable_obj_change_tol = 1e-12; // would block any change
        assert!(meets_acceptable_thresholds(&info, &opts, 1.0, 1.0, None),
            "iter-0 None should bypass the gate");
    }

    #[test]
    fn test_convergence_acceptable() {
        let info = ConvergenceInfo {
            primal_inf: 1e-7,
            primal_inf_internal: 1e-7,
            dual_inf: 1e-7,
            dual_inf_unscaled: 1e-7,
            compl_inf: 1e-7,
            mu: 1e-8,
            objective: 5.0,
            multiplier_sum: 0.0,
            multiplier_count: 0,
            bound_multiplier_sum: 0.0,
            bound_multiplier_count: 0,
            x_max_abs: 0.0,
        };
        let opts = SolverOptions::default();
        // Need enough consecutive near-tolerance iterations (hardcoded NEAR_TOL_ITERS=15).
        assert_eq!(
            check_convergence(&info, &opts, 15),
            ConvergenceStatus::Acceptable
        );
    }

    #[test]
    fn test_convergence_acceptable_insufficient_count() {
        let info = ConvergenceInfo {
            primal_inf: 1e-7,
            primal_inf_internal: 1e-7,
            dual_inf: 1e-7,
            dual_inf_unscaled: 1e-7,
            compl_inf: 1e-7,
            mu: 1e-8,
            objective: 5.0,
            multiplier_sum: 0.0,
            multiplier_count: 0,
            bound_multiplier_sum: 0.0,
            bound_multiplier_count: 0,
            x_max_abs: 0.0,
        };
        let opts = SolverOptions::default();
        // Not enough consecutive iterations (14 < 15)
        assert_eq!(
            check_convergence(&info, &opts, 14),
            ConvergenceStatus::NotConverged
        );
    }

    #[test]
    fn test_convergence_dual_scaling() {
        // Large multipliers should scale the dual tolerance
        let info = ConvergenceInfo {
            primal_inf: 1e-10,
            primal_inf_internal: 1e-10,
            dual_inf: 5e-5, // Would fail without scaling
            dual_inf_unscaled: 5e-5,
            compl_inf: 1e-10,
            mu: 1e-11,
            objective: 1.0,
            multiplier_sum: 1e6, // Large multipliers
            multiplier_count: 10,
            // Pretend all 10 multipliers are bound multipliers for this test
            // so s_d and s_c both equal 1000 — keeps the existing dual_inf
            // arithmetic check below valid.
            bound_multiplier_sum: 1e6,
            bound_multiplier_count: 10,
            x_max_abs: 0.0,
        };
        let opts = SolverOptions::default();
        // s_d = max(100, 1e6/10)/100 = 1e5/100 = 1000
        // dual_tol = 1e-8 * 1000 = 1e-5
        // 5e-5 > 1e-5, so not converged
        assert_eq!(
            check_convergence(&info, &opts, 0),
            ConvergenceStatus::NotConverged
        );

        // With slightly smaller dual_inf it should pass
        let info2 = ConvergenceInfo {
            dual_inf: 5e-6,
            dual_inf_unscaled: 5e-6,
            ..info
        };
        assert_eq!(
            check_convergence(&info2, &opts, 0),
            ConvergenceStatus::Converged
        );
    }

    #[test]
    fn test_convergence_unscaled_gate_blocks_false_convergence() {
        // Scaled dual_inf small, but iterative dual_inf_unscaled large.
        let info = ConvergenceInfo {
            primal_inf: 1e-10,
            primal_inf_internal: 1e-10,
            dual_inf: 1e-10,
            dual_inf_unscaled: 1.5, // > dual_inf_tol=1.0
            compl_inf: 1e-10,
            mu: 1e-11,
            objective: 1.0,
            multiplier_sum: 0.0,
            multiplier_count: 0,
            bound_multiplier_sum: 0.0,
            bound_multiplier_count: 0,
            x_max_abs: 0.0,
        };
        let opts = SolverOptions::default();
        assert_eq!(
            check_convergence(&info, &opts, 0),
            ConvergenceStatus::NotConverged
        );

        let info2 = ConvergenceInfo {
            dual_inf_unscaled: 0.5,
            ..info
        };
        assert_eq!(
            check_convergence(&info2, &opts, 0),
            ConvergenceStatus::Converged
        );
    }

    #[test]
    fn test_complementarity_error_no_bounds() {
        let x = vec![1.0, 2.0];
        let x_l = vec![f64::NEG_INFINITY; 2];
        let x_u = vec![f64::INFINITY; 2];
        let z_l = vec![0.0; 2];
        let z_u = vec![0.0; 2];
        let err = complementarity_error(&x, &x_l, &x_u, &z_l, &z_u, 0.0);
        assert!((err).abs() < 1e-15);
    }

    #[test]
    fn test_complementarity_error_at_optimality() {
        // At optimality: (x - x_l) * z_l = mu
        let mu = 0.01;
        let x = vec![1.1]; // slack = 0.1
        let x_l = vec![1.0];
        let x_u = vec![f64::INFINITY];
        let z_l = vec![mu / 0.1]; // z_l = mu/slack = 0.1
        let z_u = vec![0.0];
        let err = complementarity_error(&x, &x_l, &x_u, &z_l, &z_u, mu);
        assert!(err < 1e-12, "At optimality, complementarity error should be ~0, got {}", err);
    }

    #[test]
    fn test_complementarity_error_away_from_optimality() {
        let mu = 0.01;
        let x = vec![1.5]; // slack = 0.5
        let x_l = vec![1.0];
        let x_u = vec![f64::INFINITY];
        let z_l = vec![1.0]; // z_l * slack = 0.5 >> mu
        let z_u = vec![0.0];
        let err = complementarity_error(&x, &x_l, &x_u, &z_l, &z_u, mu);
        // err = |0.5 * 1.0 - 0.01| = 0.49
        assert!((err - 0.49).abs() < 1e-12);
    }

    #[test]
    fn test_dual_infeasibility_stationarity() {
        // Exact stationarity: grad_f + J^T * lambda - z_l + z_u = 0
        let n = 2;
        let grad_f = vec![1.0, 2.0];
        let jac_rows = vec![0, 0];
        let jac_cols = vec![0, 1];
        let jac_vals = vec![1.0, 1.0]; // J = [1, 1]
        let lambda = vec![-0.5]; // J^T * lambda = [-0.5, -0.5]
        // residual = [1.0 + (-0.5), 2.0 + (-0.5)] - z_l + z_u
        // Need z_l, z_u such that residual = 0
        let z_l = vec![0.5, 1.5];
        let z_u = vec![0.0, 0.0];
        let x_l = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        let x_u = vec![f64::INFINITY, f64::INFINITY];
        let di = dual_infeasibility(&grad_f, &jac_rows, &jac_cols, &jac_vals, &lambda, &z_l, &z_u, n, 0.0, 0.0, &x_l, &x_u);
        assert!(di < 1e-12, "Exact stationarity should give 0, got {}", di);

        // Nonzero case
        let z_l2 = vec![0.0, 0.0];
        let di2 = dual_infeasibility(&grad_f, &jac_rows, &jac_cols, &jac_vals, &lambda, &z_l2, &z_u, n, 0.0, 0.0, &x_l, &x_u);
        assert!(di2 > 0.1, "Non-stationary should give positive dual_inf");
    }

    // T3.1 (2026-04-27): removed
    // `test_dual_infeasibility_scaled_insensitive_to_gradient_magnitude`.
    // Its premise — that the dual-residual max-norm should be divided
    // component-wise by `1 + |grad_f_i|` — was a ripopt-specific
    // heuristic with no Ipopt analog. `dual_infeasibility_scaled` now
    // matches `IpIpoptCalculatedQuantities::curr_dual_infeasibility`'s
    // raw L_inf formula; the global `s_d` scaling is applied at the
    // convergence-test boundary instead.

    #[test]
    fn test_dual_infeasibility_scaled_stationarity() {
        let n = 2;
        let grad_f = vec![1.0, 2.0];
        let jac_rows = vec![0, 0];
        let jac_cols = vec![0, 1];
        let jac_vals = vec![1.0, 1.0];
        let lambda = vec![-0.5];
        let z_l = vec![0.5, 1.5];
        let z_u = vec![0.0, 0.0];
        let x_l = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        let x_u = vec![f64::INFINITY, f64::INFINITY];
        // At stationarity, both scaled and unscaled should be ~0
        let di_s = dual_infeasibility_scaled(&grad_f, &jac_rows, &jac_cols, &jac_vals, &lambda, &z_l, &z_u, n, 0.0, 0.0, &x_l, &x_u);
        assert!(di_s < 1e-12, "Scaled stationarity should give 0, got {}", di_s);
    }

    /// T3.9: For a one-sided lower-bounded variable, the kappa_d damping
    /// should add `+ kappa_d * mu` to grad_lag, exactly canceling a small
    /// negative raw residual driven by an over-estimated z_l.
    /// Mirrors `IpIpoptCalculatedQuantities.cpp:888-899`.
    #[test]
    fn test_dual_infeasibility_kappa_d_one_sided_lower_cancels() {
        let n = 1;
        let grad_f = vec![1.0];
        let jac_rows: Vec<usize> = vec![];
        let jac_cols: Vec<usize> = vec![];
        let jac_vals: Vec<f64> = vec![];
        let lambda: Vec<f64> = vec![];
        // raw grad_lag = grad_f - z_l = 1.0 - (1.0 + 1e-6) = -1e-6
        let z_l = vec![1.0 + 1e-6];
        let z_u = vec![0.0];
        let x_l = vec![0.0];
        let x_u = vec![f64::INFINITY];
        let kappa_d = 1e-5;
        let mu = 0.1;
        // damped = -1e-6 + kappa_d*mu = -1e-6 + 1e-6 = 0
        let di = dual_infeasibility(
            &grad_f, &jac_rows, &jac_cols, &jac_vals, &lambda, &z_l, &z_u, n,
            kappa_d, mu, &x_l, &x_u,
        );
        assert!(di < 1e-15, "damped one-sided-lower should be ~0, got {}", di);

        // Without damping, the residual should be visible at 1e-6.
        let di_undamped = dual_infeasibility(
            &grad_f, &jac_rows, &jac_cols, &jac_vals, &lambda, &z_l, &z_u, n,
            0.0, 0.0, &x_l, &x_u,
        );
        assert!(
            (di_undamped - 1e-6).abs() < 1e-15,
            "undamped should expose 1e-6 residual, got {}",
            di_undamped,
        );
    }

    /// T3.9: For a one-sided upper-bounded variable, the kappa_d damping
    /// should subtract `kappa_d * mu` from grad_lag, canceling a small
    /// positive raw residual.
    #[test]
    fn test_dual_infeasibility_kappa_d_one_sided_upper_cancels() {
        let n = 1;
        let grad_f = vec![-1.0];
        let jac_rows: Vec<usize> = vec![];
        let jac_cols: Vec<usize> = vec![];
        let jac_vals: Vec<f64> = vec![];
        let lambda: Vec<f64> = vec![];
        // raw grad_lag = grad_f + z_u = -1.0 + (1.0 + 1e-6) = +1e-6
        let z_l = vec![0.0];
        let z_u = vec![1.0 + 1e-6];
        let x_l = vec![f64::NEG_INFINITY];
        let x_u = vec![0.0];
        let kappa_d = 1e-5;
        let mu = 0.1;
        // damped = +1e-6 - kappa_d*mu = +1e-6 - 1e-6 = 0
        let di = dual_infeasibility(
            &grad_f, &jac_rows, &jac_cols, &jac_vals, &lambda, &z_l, &z_u, n,
            kappa_d, mu, &x_l, &x_u,
        );
        assert!(di < 1e-15, "damped one-sided-upper should be ~0, got {}", di);
    }

    /// T3.9: A free (no-bounds) variable receives no damping; the
    /// residual is exactly |grad_f|.
    #[test]
    fn test_dual_infeasibility_kappa_d_free_var_undamped() {
        let n = 1;
        let grad_f = vec![0.5];
        let jac_rows: Vec<usize> = vec![];
        let jac_cols: Vec<usize> = vec![];
        let jac_vals: Vec<f64> = vec![];
        let lambda: Vec<f64> = vec![];
        let z_l = vec![0.0];
        let z_u = vec![0.0];
        let x_l = vec![f64::NEG_INFINITY];
        let x_u = vec![f64::INFINITY];
        let di = dual_infeasibility(
            &grad_f, &jac_rows, &jac_cols, &jac_vals, &lambda, &z_l, &z_u, n,
            1e-5, 0.1, &x_l, &x_u,
        );
        assert!(
            (di - 0.5).abs() < 1e-15,
            "free var must not be damped, got {}",
            di,
        );
    }

    /// T3.9: A two-sided-bounded variable should see the +kappa_d*mu
    /// and -kappa_d*mu terms cancel, leaving the raw residual unchanged.
    /// Mirrors Ipopt's `Px_L*1 - Px_U*1` projection algebra.
    #[test]
    fn test_dual_infeasibility_kappa_d_two_sided_no_net_damping() {
        let n = 1;
        let grad_f = vec![0.25];
        let jac_rows: Vec<usize> = vec![];
        let jac_cols: Vec<usize> = vec![];
        let jac_vals: Vec<f64> = vec![];
        let lambda: Vec<f64> = vec![];
        let z_l = vec![0.0];
        let z_u = vec![0.0];
        // Two-sided: both bounds finite — net damping must be zero.
        let x_l = vec![-1.0];
        let x_u = vec![1.0];
        let di = dual_infeasibility(
            &grad_f, &jac_rows, &jac_cols, &jac_vals, &lambda, &z_l, &z_u, n,
            1e-5, 0.1, &x_l, &x_u,
        );
        assert!(
            (di - 0.25).abs() < 1e-15,
            "two-sided var must see no net damping, got {}",
            di,
        );
    }
}
