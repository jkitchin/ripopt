//! Issue #6 reproducer: incorrect solution on bounded Rosenbrock
//!
//! min  (1-x1)^2 + 100*(x2 - x1^2)^2
//! s.t. -5 <= x1 <= 0.5, -5 <= x2 <= 5
//!
//! Known optimal: f* = 0.25 at (0.5, 0.25)

use ripopt::{NlpProblem, SolveStatus, SolverOptions};

struct BoundedRosenbrock;

impl NlpProblem for BoundedRosenbrock {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = -5.0; x_u[0] = 0.5;
        x_l[1] = -5.0; x_u[1] = 5.0;
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0; x0[1] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2));
        grad[1] = 200.0 * (x[1] - x[0].powi(2));
        true
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * (2.0 - 400.0 * (x[1] - x[0].powi(2)) + 800.0 * x[0].powi(2));
        vals[1] = obj_factor * (-400.0 * x[0]);
        vals[2] = obj_factor * 200.0;
        true
    }
}

fn main() {
    let known_opt = 0.25;

    println!("=== Default options ===");
    let opts = SolverOptions {
        print_level: 5,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&BoundedRosenbrock, &opts);
    let rel_err = (result.objective - known_opt).abs() / known_opt;
    println!("Status: {:?}", result.status);
    println!("Objective: {:.10e} (known: {:.10e})", result.objective, known_opt);
    println!("Solution: ({:.6}, {:.6})", result.x[0], result.x[1]);
    println!("Relative error: {:.2e}", rel_err);
    println!("Iterations: {}", result.iterations);
    if result.status == SolveStatus::Optimal && rel_err > 1e-4 {
        println!(">>> BUG: Optimal status but wrong answer (rel_err={:.2e})", rel_err);
    }

    println!("\n=== mu_init=1e-4 (workaround) ===");
    let opts2 = SolverOptions {
        print_level: 5,
        mu_init: 1e-4,
        ..SolverOptions::default()
    };
    let result2 = ripopt::solve(&BoundedRosenbrock, &opts2);
    let rel_err2 = (result2.objective - known_opt).abs() / known_opt;
    println!("Status: {:?}", result2.status);
    println!("Objective: {:.10e} (known: {:.10e})", result2.objective, known_opt);
    println!("Solution: ({:.6}, {:.6})", result2.x[0], result2.x[1]);
    println!("Relative error: {:.2e}", rel_err2);
    println!("Iterations: {}", result2.iterations);
}
