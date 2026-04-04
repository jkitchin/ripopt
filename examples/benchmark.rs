/// Benchmark example: solves 5 standard NLP test problems, times each over
/// multiple runs, and prints a JSON report to stdout.
///
/// Problems: Rosenbrock, HS071, SimpleQP, HS035 (BoundConstrainedQuadratic),
///           PureBoundConstrained.
///
/// Run with:
///   cargo run --example benchmark
///   cargo run --release --example benchmark
use std::time::Instant;

use ripopt::{NlpProblem, SolveResult, SolveStatus, SolverOptions};
use serde::Serialize;

const NUM_RUNS: usize = 20;

// ---- JSON output types -----

#[derive(Serialize)]
struct BenchmarkResult {
    problem: String,
    n_vars: usize,
    n_constraints: usize,
    x_opt: Vec<f64>,
    obj_opt: f64,
    constraint_multipliers: Vec<f64>,
    bound_multipliers_z_l: Vec<f64>,
    bound_multipliers_z_u: Vec<f64>,
    iterations: usize,
    solve_time_avg_s: f64,
    solve_time_std_s: f64,
    constraint_violation: f64,
    status: String,
}

// ---- Problem definitions (copied from tests/correctness.rs) ----

// 1. Rosenbrock (unconstrained)
struct Rosenbrock;

impl NlpProblem for Rosenbrock {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -1.2;
        x0[1] = 1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let x1 = x[0];
        let x2 = x[1];
        grad[0] = -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 * x1);
        grad[1] = 200.0 * (x2 - x1 * x1);
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        let x1 = x[0];
        let x2 = x[1];
        vals[0] = obj_factor * (2.0 - 400.0 * x2 + 1200.0 * x1 * x1);
        vals[1] = obj_factor * (-400.0 * x1);
        vals[2] = obj_factor * 200.0;
    }
}

// 2. HS071
struct HS071;

impl NlpProblem for HS071 {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 2 }

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

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        grad[1] = x[0] * x[3];
        grad[2] = x[0] * x[3] + 1.0;
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (
            vec![0, 0, 0, 0, 1, 1, 1, 1],
            vec![0, 1, 2, 3, 0, 1, 2, 3],
        )
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = x[1] * x[2] * x[3];
        vals[1] = x[0] * x[2] * x[3];
        vals[2] = x[0] * x[1] * x[3];
        vals[3] = x[0] * x[1] * x[2];
        vals[4] = 2.0 * x[0];
        vals[5] = 2.0 * x[1];
        vals[6] = 2.0 * x[2];
        vals[7] = 2.0 * x[3];
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (
            vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
        )
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * 2.0 * x[3];
        vals[1] = obj_factor * x[3];
        vals[2] = 0.0;
        vals[3] = obj_factor * x[3];
        vals[4] = 0.0;
        vals[5] = 0.0;
        vals[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]);
        vals[7] = obj_factor * x[0];
        vals[8] = obj_factor * x[0];
        vals[9] = 0.0;

        // Constraint 1: g1 = x1*x2*x3*x4
        vals[1] += lambda[0] * x[2] * x[3];
        vals[3] += lambda[0] * x[1] * x[3];
        vals[4] += lambda[0] * x[0] * x[3];
        vals[6] += lambda[0] * x[1] * x[2];
        vals[7] += lambda[0] * x[0] * x[2];
        vals[8] += lambda[0] * x[0] * x[1];

        // Constraint 2: g2 = sum xi^2
        vals[0] += lambda[1] * 2.0;
        vals[2] += lambda[1] * 2.0;
        vals[5] += lambda[1] * 2.0;
        vals[9] += lambda[1] * 2.0;
    }
}

// 3. SimpleQP
struct SimpleQP;

impl NlpProblem for SimpleQP {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
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

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        0.5 * (x[0] * x[0] + x[1] * x[1])
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = x[0];
        grad[1] = x[1];
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] + x[1];
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;
        vals[1] = 1.0;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * 1.0;
        vals[1] = obj_factor * 1.0;
    }
}

// 4. BoundConstrainedQuadratic (HS035-like)
struct BoundConstrainedQuadratic;

impl NlpProblem for BoundConstrainedQuadratic {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..3 {
            x_l[i] = 0.0;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = f64::NEG_INFINITY;
        g_u[0] = 3.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.5;
        x0[1] = 0.5;
        x0[2] = 0.5;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        9.0 - 8.0 * x[0] - 6.0 * x[1] - 4.0 * x[2]
            + 2.0 * x[0] * x[0]
            + 2.0 * x[1] * x[1]
            + x[2] * x[2]
            + 2.0 * x[0] * x[1]
            + 2.0 * x[0] * x[2]
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = -8.0 + 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2];
        grad[1] = -6.0 + 2.0 * x[0] + 4.0 * x[1];
        grad[2] = -4.0 + 2.0 * x[0] + 2.0 * x[2];
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] + x[1] + 2.0 * x[2];
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0], vec![0, 1, 2])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;
        vals[1] = 1.0;
        vals[2] = 2.0;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2, 2], vec![0, 0, 1, 0, 2])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * 4.0;
        vals[1] = obj_factor * 2.0;
        vals[2] = obj_factor * 4.0;
        vals[3] = obj_factor * 2.0;
        vals[4] = obj_factor * 2.0;
    }
}

// 5. PureBoundConstrained
struct PureBoundConstrained;

impl NlpProblem for PureBoundConstrained {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 {
            x_l[i] = 0.0;
            x_u[i] = 3.0;
        }
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
        x0[2] = 0.0;
        x0[3] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        (x[0] - 1.0).powi(2)
            + (x[1] - 2.0).powi(2)
            + (x[2] - 3.0).powi(2)
            + (x[3] - 4.0).powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * (x[0] - 1.0);
        grad[1] = 2.0 * (x[1] - 2.0);
        grad[2] = 2.0 * (x[2] - 3.0);
        grad[3] = 2.0 * (x[3] - 4.0);
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2, 3], vec![0, 1, 2, 3])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        for v in vals.iter_mut() {
            *v = obj_factor * 2.0;
        }
    }
}

// ---- Helpers ----

fn status_string(s: SolveStatus) -> &'static str {
    match s {
        SolveStatus::Optimal => "Optimal",
        SolveStatus::Infeasible => "Infeasible",
        SolveStatus::MaxIterations => "MaxIterations",
        SolveStatus::NumericalError => "NumericalError",
        SolveStatus::Unbounded => "Unbounded",
        SolveStatus::RestorationFailed => "RestorationFailed",
        SolveStatus::InternalError => "InternalError",
        SolveStatus::LocalInfeasibility => "LocalInfeasibility",
    }
}

/// Compute the maximum constraint violation.
/// For each constraint i, the violation is max(g_l[i] - g[i], g[i] - g_u[i], 0).
fn max_constraint_violation(
    constraint_values: &[f64],
    g_l: &[f64],
    g_u: &[f64],
) -> f64 {
    let mut max_viol = 0.0_f64;
    for i in 0..constraint_values.len() {
        let g = constraint_values[i];
        // violation below lower bound
        let viol_lo = if g_l[i].is_finite() { (g_l[i] - g).max(0.0) } else { 0.0 };
        // violation above upper bound
        let viol_hi = if g_u[i].is_finite() { (g - g_u[i]).max(0.0) } else { 0.0 };
        max_viol = max_viol.max(viol_lo).max(viol_hi);
    }
    max_viol
}

fn benchmark_problem<P: NlpProblem>(
    name: &str,
    problem: &P,
    options: &SolverOptions,
) -> BenchmarkResult {
    let n_vars = problem.num_variables();
    let n_constraints = problem.num_constraints();

    // Gather constraint bounds for violation computation.
    let mut g_l = vec![0.0; n_constraints];
    let mut g_u = vec![0.0; n_constraints];
    if n_constraints > 0 {
        problem.constraint_bounds(&mut g_l, &mut g_u);
    }

    let mut times = Vec::with_capacity(NUM_RUNS);
    let mut last_result: Option<SolveResult> = None;

    for _ in 0..NUM_RUNS {
        let start = Instant::now();
        let result = ripopt::solve(problem, options);
        let elapsed = start.elapsed().as_secs_f64();
        times.push(elapsed);
        last_result = Some(result);
    }

    let result = last_result.unwrap();

    // Compute timing statistics.
    let n = times.len() as f64;
    let avg = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let constraint_violation = max_constraint_violation(&result.constraint_values, &g_l, &g_u);

    BenchmarkResult {
        problem: name.to_string(),
        n_vars,
        n_constraints,
        x_opt: result.x.clone(),
        obj_opt: result.objective,
        constraint_multipliers: result.constraint_multipliers.clone(),
        bound_multipliers_z_l: result.bound_multipliers_lower.clone(),
        bound_multipliers_z_u: result.bound_multipliers_upper.clone(),
        iterations: result.iterations,
        solve_time_avg_s: avg,
        solve_time_std_s: std_dev,
        constraint_violation,
        status: status_string(result.status).to_string(),
    }
}

fn main() {
    let options = SolverOptions {
        print_level: 0,
        ..SolverOptions::default()
    };

    let mut results = Vec::new();

    results.push(benchmark_problem("Rosenbrock", &Rosenbrock, &options));
    results.push(benchmark_problem("HS071", &HS071, &options));
    results.push(benchmark_problem("SimpleQP", &SimpleQP, &options));
    results.push(benchmark_problem("HS035", &BoundConstrainedQuadratic, &options));
    results.push(benchmark_problem("PureBoundConstrained", &PureBoundConstrained, &options));

    let json = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    println!("{}", json);
}
