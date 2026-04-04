//! Rosenbrock function minimization (unconstrained).
//!
//! min f(x) = (1 - x1)^2 + 100*(x2 - x1^2)^2
//!
//! Starting point: (-1.2, 1.0)
//! Known solution:  (1.0, 1.0) with f* = 0.0

use ripopt::{NlpProblem, SolveStatus, SolverOptions};

struct Rosenbrock;

impl NlpProblem for Rosenbrock {
    fn num_variables(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        0
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        // Unbounded
        x_l[0] = f64::NEG_INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {
        // No constraints
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -1.2;
        x0[1] = 1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        (1.0 - x1).powi(2) + 100.0 * (x2 - x1 * x1).powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let (x1, x2) = (x[0], x[1]);
        grad[0] = -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 * x1);
        grad[1] = 200.0 * (x2 - x1 * x1);
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {
        // No constraints
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // No constraints, empty Jacobian
        (vec![], vec![])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {
        // No constraints
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Lower triangle of 2x2 Hessian: (0,0), (1,0), (1,1)
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        let (x1, x2) = (x[0], x[1]);
        // H[0,0] = 2 - 400*x2 + 1200*x1^2
        vals[0] = obj_factor * (2.0 - 400.0 * x2 + 1200.0 * x1 * x1);
        // H[1,0] = -400*x1
        vals[1] = obj_factor * (-400.0 * x1);
        // H[1,1] = 200
        vals[2] = obj_factor * 200.0;
    }
}

fn main() {
    env_logger::init();

    let problem = Rosenbrock;
    let options = SolverOptions::default();

    println!("Solving Rosenbrock function...");
    println!("  Start: (-1.2, 1.0)");
    println!();

    let result = ripopt::solve(&problem, &options);

    println!("Status:     {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Objective:  {:.10e}", result.objective);
    println!("Solution:   x1 = {:.10}, x2 = {:.10}", result.x[0], result.x[1]);

    assert_eq!(result.status, SolveStatus::Optimal);
}
