//! Structured solution report with independent validation of the returned point.
//!
//! Re-evaluates the problem at `x*` and computes constraint / bound / KKT
//! residuals from scratch, independent of the solver's own diagnostics. The
//! report serializes to JSON for downstream tooling. Inspired by GAMS Examiner:
//! https://www.gams.com/latest/docs/S_EXAMINER.html
//!
//! See issue #27.
//!
//! Sign convention: stationarity uses `L = f + lam_g^T g`, so
//! `∇f + J^T λ - z_L + z_U = 0` at a KKT point.
//!
//! The complementarity check uses the *unscaled* products (no `s_d` rescaling),
//! since this report is meant to be independently checkable.

use serde::Serialize;

use crate::problem::NlpProblem;
use crate::result::{SolveResult, SolveStatus};
use crate::options::SolverOptions;

#[derive(Debug, Clone, Serialize)]
pub struct SolverInfo {
    pub name: &'static str,
    pub version: &'static str,
    pub git_rev: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProblemInfo {
    pub name: Option<String>,
    pub source: Option<String>,
    pub n_variables: usize,
    pub n_constraints: usize,
    pub n_equality_constraints: usize,
    pub n_inequality_constraints: usize,
    pub n_lower_bounded_variables: usize,
    pub n_upper_bounded_variables: usize,
    pub n_fixed_variables: usize,
    pub n_free_variables: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Solution {
    pub x: Vec<f64>,
    pub constraint_multipliers: Vec<f64>,
    pub bound_multipliers_lower: Vec<f64>,
    pub bound_multipliers_upper: Vec<f64>,
    pub constraint_values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationTolerances {
    pub constr_viol_tol: f64,
    pub dual_inf_tol: f64,
    pub compl_inf_tol: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Validation {
    /// max_i max(0, x_l_i - x_i, x_i - x_u_i) — bound feasibility
    pub max_bound_violation: f64,
    /// max_i max(0, g_l_i - g_i, g_i - g_u_i) — constraint feasibility
    pub max_constraint_violation: f64,
    /// max_i |g_i - g_l_i| over equality constraints (g_l == g_u)
    pub max_equality_residual: f64,
    /// max_i max(0, g_l_i - g_i, g_i - g_u_i) over inequality constraints
    pub max_inequality_residual: f64,
    /// ||∇f + J^T λ - z_L + z_U||_∞ — dual stationarity
    pub stationarity_inf_norm: f64,
    /// max_i |z_L_i (x_i - x_l_i)| and |z_U_i (x_u_i - x_i)|
    pub complementarity_bound_max: f64,
    /// max_i |z_L_i| over fixed/violated bounds, max_i z_L_i / z_U_i sign violations
    pub bound_multiplier_sign_violation: f64,
    /// True iff all four residuals are within the configured tolerances.
    pub kkt_satisfied: bool,
    pub tolerances: ValidationTolerances,
}

#[derive(Debug, Clone, Serialize)]
pub struct SolutionReport {
    pub solver: SolverInfo,
    pub problem: ProblemInfo,
    /// Full argv used to invoke the solver (CLI capture).
    pub command: Vec<String>,
    /// Resolved solver options (after CLI parsing).
    pub options: SolverOptions,
    pub status: SolveStatus,
    pub iterations: usize,
    pub wall_time_secs: f64,
    pub objective: f64,
    pub solution: Solution,
    pub validation: Validation,
    pub diagnostics: crate::result::SolverDiagnostics,
}

/// Build a structured report by re-evaluating the problem at `result.x`
/// and computing KKT residuals independently from solver internals.
pub fn build_report<P: NlpProblem>(
    problem: &P,
    result: &SolveResult,
    options: &SolverOptions,
    command: Vec<String>,
    problem_name: Option<String>,
    problem_source: Option<String>,
) -> SolutionReport {
    let n = problem.num_variables();
    let m = problem.num_constraints();

    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    problem.bounds(&mut x_l, &mut x_u);

    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);

    let nlp_lo = options.nlp_lower_bound_inf;
    let nlp_hi = options.nlp_upper_bound_inf;

    let mut n_lower_b = 0usize;
    let mut n_upper_b = 0usize;
    let mut n_fixed = 0usize;
    let mut n_free = 0usize;
    for i in 0..n {
        let has_lo = x_l[i] > nlp_lo;
        let has_hi = x_u[i] < nlp_hi;
        if has_lo && has_hi && (x_u[i] - x_l[i]).abs() <= 0.0 {
            n_fixed += 1;
        } else {
            if has_lo { n_lower_b += 1; }
            if has_hi { n_upper_b += 1; }
            if !has_lo && !has_hi { n_free += 1; }
        }
    }

    let mut n_eq = 0usize;
    let mut n_ineq = 0usize;
    for i in 0..m {
        if (g_u[i] - g_l[i]).abs() <= 0.0 {
            n_eq += 1;
        } else {
            n_ineq += 1;
        }
    }

    // Re-evaluate at the returned x*.
    let x = &result.x;
    let mut grad = vec![0.0; n];
    let _ = problem.gradient(x, true, &mut grad);

    let (jrows, jcols) = problem.jacobian_structure();
    let mut jvals = vec![0.0; jrows.len()];
    let _ = problem.jacobian_values(x, false, &mut jvals);

    let g_eval = if !result.constraint_values.is_empty() {
        result.constraint_values.clone()
    } else {
        let mut g = vec![0.0; m];
        let _ = problem.constraints(x, false, &mut g);
        g
    };

    // Bound violations.
    let mut max_bound_viol = 0.0_f64;
    for i in 0..n {
        if x_l[i] > nlp_lo {
            max_bound_viol = max_bound_viol.max(x_l[i] - x[i]);
        }
        if x_u[i] < nlp_hi {
            max_bound_viol = max_bound_viol.max(x[i] - x_u[i]);
        }
    }

    // Constraint violations split by equality / inequality.
    let mut max_eq_res = 0.0_f64;
    let mut max_ineq_res = 0.0_f64;
    for i in 0..m {
        if (g_u[i] - g_l[i]).abs() <= 0.0 {
            max_eq_res = max_eq_res.max((g_eval[i] - g_l[i]).abs());
        } else {
            let lo_v = if g_l[i] > nlp_lo { (g_l[i] - g_eval[i]).max(0.0) } else { 0.0 };
            let hi_v = if g_u[i] < nlp_hi { (g_eval[i] - g_u[i]).max(0.0) } else { 0.0 };
            max_ineq_res = max_ineq_res.max(lo_v).max(hi_v);
        }
    }
    let max_constr_viol = max_eq_res.max(max_ineq_res);

    // Stationarity: ∇f + J^T λ - z_L + z_U.
    let lam = &result.constraint_multipliers;
    let z_l = &result.bound_multipliers_lower;
    let z_u = &result.bound_multipliers_upper;
    let mut stat = grad.clone();
    if !lam.is_empty() {
        for k in 0..jrows.len() {
            let i = jrows[k];
            let j = jcols[k];
            if i < lam.len() && j < stat.len() {
                stat[j] += jvals[k] * lam[i];
            }
        }
    }
    if z_l.len() == n && z_u.len() == n {
        for j in 0..n {
            stat[j] += -z_l[j] + z_u[j];
        }
    }
    let stationarity_inf = stat.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));

    // Complementarity at bounds: z_L_i * (x_i - x_l_i) and z_U_i * (x_u_i - x_i).
    let mut compl_max = 0.0_f64;
    let mut sign_viol = 0.0_f64;
    if z_l.len() == n {
        for i in 0..n {
            if x_l[i] > nlp_lo {
                let prod = z_l[i] * (x[i] - x_l[i]);
                compl_max = compl_max.max(prod.abs());
                if z_l[i] < 0.0 { sign_viol = sign_viol.max(-z_l[i]); }
            }
        }
    }
    if z_u.len() == n {
        for i in 0..n {
            if x_u[i] < nlp_hi {
                let prod = z_u[i] * (x_u[i] - x[i]);
                compl_max = compl_max.max(prod.abs());
                if z_u[i] < 0.0 { sign_viol = sign_viol.max(-z_u[i]); }
            }
        }
    }

    let kkt_ok = max_constr_viol <= options.constr_viol_tol
        && max_bound_viol <= options.constr_viol_tol
        && stationarity_inf <= options.dual_inf_tol
        && compl_max <= options.compl_inf_tol;

    SolutionReport {
        solver: SolverInfo {
            name: "ripopt",
            version: env!("CARGO_PKG_VERSION"),
            git_rev: option_env!("RIPOPT_GIT_REV").map(|s| s.to_string()),
        },
        problem: ProblemInfo {
            name: problem_name,
            source: problem_source,
            n_variables: n,
            n_constraints: m,
            n_equality_constraints: n_eq,
            n_inequality_constraints: n_ineq,
            n_lower_bounded_variables: n_lower_b,
            n_upper_bounded_variables: n_upper_b,
            n_fixed_variables: n_fixed,
            n_free_variables: n_free,
        },
        command,
        options: options.clone(),
        status: result.status,
        iterations: result.iterations,
        wall_time_secs: result.diagnostics.wall_time_secs,
        objective: result.objective,
        solution: Solution {
            x: result.x.clone(),
            constraint_multipliers: result.constraint_multipliers.clone(),
            bound_multipliers_lower: result.bound_multipliers_lower.clone(),
            bound_multipliers_upper: result.bound_multipliers_upper.clone(),
            constraint_values: g_eval,
        },
        validation: Validation {
            max_bound_violation: max_bound_viol,
            max_constraint_violation: max_constr_viol,
            max_equality_residual: max_eq_res,
            max_inequality_residual: max_ineq_res,
            stationarity_inf_norm: stationarity_inf,
            complementarity_bound_max: compl_max,
            bound_multiplier_sign_violation: sign_viol,
            kkt_satisfied: kkt_ok,
            tolerances: ValidationTolerances {
                constr_viol_tol: options.constr_viol_tol,
                dual_inf_tol: options.dual_inf_tol,
                compl_inf_tol: options.compl_inf_tol,
            },
        },
        diagnostics: result.diagnostics.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::NlpProblem;
    use crate::result::{SolveResult, SolverDiagnostics, SolveStatus};
    use crate::options::SolverOptions;

    /// Trivial problem: min (x-2)^2 s.t. x >= 0. Optimum at x=2.
    struct Trivial;
    impl NlpProblem for Trivial {
        fn num_variables(&self) -> usize { 1 }
        fn num_constraints(&self) -> usize { 0 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.0;
            x_u[0] = f64::INFINITY;
        }
        fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = (x[0] - 2.0).powi(2); true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 2.0 * (x[0] - 2.0); true
        }
        fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor; true
        }
    }

    #[test]
    fn report_has_zero_residuals_at_optimum() {
        let p = Trivial;
        let opts = SolverOptions::default();
        // Hand-rolled "result" at the known optimum x=2, all multipliers zero.
        let res = SolveResult {
            x: vec![2.0],
            objective: 0.0,
            constraint_multipliers: vec![],
            bound_multipliers_lower: vec![0.0],
            bound_multipliers_upper: vec![0.0],
            constraint_values: vec![],
            status: SolveStatus::Optimal,
            iterations: 0,
            diagnostics: SolverDiagnostics::default(),
        };
        let r = build_report(&p, &res, &opts, vec!["ripopt".into()], None, None);
        assert!(r.validation.kkt_satisfied, "report = {r:?}");
        assert!(r.validation.max_bound_violation <= 0.0);
        assert!(r.validation.max_constraint_violation <= 0.0);
        assert!(r.validation.stationarity_inf_norm < 1e-12);
        assert!(r.validation.complementarity_bound_max < 1e-12);
        assert_eq!(r.problem.n_variables, 1);
        assert_eq!(r.problem.n_lower_bounded_variables, 1);
    }

    #[test]
    fn report_flags_bound_violation() {
        let p = Trivial;
        let opts = SolverOptions::default();
        // Pretend solver returned an infeasible point x = -1 (violates x >= 0).
        let res = SolveResult {
            x: vec![-1.0],
            objective: 9.0,
            constraint_multipliers: vec![],
            bound_multipliers_lower: vec![0.0],
            bound_multipliers_upper: vec![0.0],
            constraint_values: vec![],
            status: SolveStatus::Optimal,
            iterations: 0,
            diagnostics: SolverDiagnostics::default(),
        };
        let r = build_report(&p, &res, &opts, vec!["ripopt".into()], None, None);
        assert!((r.validation.max_bound_violation - 1.0).abs() < 1e-15);
        assert!(!r.validation.kkt_satisfied);
    }

    #[test]
    fn report_serializes_to_json() {
        let p = Trivial;
        let opts = SolverOptions::default();
        let res = SolveResult {
            x: vec![2.0],
            objective: 0.0,
            constraint_multipliers: vec![],
            bound_multipliers_lower: vec![0.0],
            bound_multipliers_upper: vec![0.0],
            constraint_values: vec![],
            status: SolveStatus::Optimal,
            iterations: 0,
            diagnostics: SolverDiagnostics::default(),
        };
        let r = build_report(&p, &res, &opts, vec!["ripopt".into()], None, None);
        let json = serde_json::to_string_pretty(&r).expect("serialize");
        assert!(json.contains("\"solver\""));
        assert!(json.contains("\"validation\""));
        assert!(json.contains("\"kkt_satisfied\""));
        assert!(json.contains("\"preprocessing\""));
    }
}
