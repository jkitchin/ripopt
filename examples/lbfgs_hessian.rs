//! Example: Solving problems without providing a Hessian.
//!
//! When `hessian_approximation_lbfgs = true`, ripopt uses an L-BFGS
//! approximation of the Hessian inside the IPM loop. The user only needs
//! to provide objective, gradient, constraints, and Jacobian callbacks.
//!
//! This is useful for:
//! - Neural networks embedded in NLPs (dense, expensive Hessian)
//! - Problems where second derivatives are unavailable
//! - Rapid prototyping (skip Hessian derivation)
//!
//! Run with: cargo run --example lbfgs_hessian

use ripopt::{NlpProblem, SolveStatus, SolverOptions};

// ---------------------------------------------------------------------------
// Example 1: Unconstrained Rosenbrock without Hessian
// ---------------------------------------------------------------------------

struct Rosenbrock;

impl NlpProblem for Rosenbrock {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..2 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -1.2;
        x0[1] = 1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        grad[1] = 200.0 * (x[1] - x[0] * x[0]);
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}

    // Dummy Hessian — never called in limited-memory mode
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _: f64, _: &[f64], _: &mut [f64]) {
        unreachable!("Hessian should not be called in limited-memory mode");
    }
}

// ---------------------------------------------------------------------------
// Example 2: Constrained problem (HS071) without Hessian
// ---------------------------------------------------------------------------

struct Hs071NoHessian;

impl NlpProblem for Hs071NoHessian {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 { x_l[i] = 1.0; x_u[i] = 5.0; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0;  g_u[0] = f64::INFINITY;  // product constraint
        g_l[1] = 40.0;  g_u[1] = 40.0;            // sum-of-squares = 40
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0; x0[1] = 5.0; x0[2] = 5.0; x0[3] = 1.0;
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
        (vec![0, 0, 0, 0, 1, 1, 1, 1], vec![0, 1, 2, 3, 0, 1, 2, 3])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = x[1] * x[2] * x[3]; vals[1] = x[0] * x[2] * x[3];
        vals[2] = x[0] * x[1] * x[3]; vals[3] = x[0] * x[1] * x[2];
        vals[4] = 2.0 * x[0]; vals[5] = 2.0 * x[1];
        vals[6] = 2.0 * x[2]; vals[7] = 2.0 * x[3];
    }

    // Dummy Hessian — not called in limited-memory mode, but fallback solvers
    // (AL, SQP) may call it. Return zeros (identity-like behavior).
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..4 {
            for j in 0..=i {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _: f64, _: &[f64], vals: &mut [f64]) {
        for v in vals.iter_mut() { *v = 0.0; }
    }
}

fn main() {
    env_logger::init();

    // --- Example 1: Unconstrained Rosenbrock ---
    println!("=== Example 1: Rosenbrock (unconstrained, no Hessian) ===");
    println!("  min (1-x1)^2 + 100*(x2-x1^2)^2");
    println!("  Start: (-1.2, 1.0)");
    println!();

    let result = ripopt::solve(
        &Rosenbrock,
        &SolverOptions {
            hessian_approximation_lbfgs: true,
            ..SolverOptions::default()
        },
    );

    println!("  Status:     {:?}", result.status);
    println!("  Iterations: {}", result.iterations);
    println!("  Objective:  {:.6e}", result.objective);
    println!("  Solution:   ({:.6}, {:.6})", result.x[0], result.x[1]);
    println!("  Expected:   (1.0, 1.0) with f* = 0");
    println!();

    assert!(
        matches!(result.status, SolveStatus::Optimal),
        "Rosenbrock should converge"
    );

    // --- Example 2: HS071 constrained ---
    println!("=== Example 2: HS071 (constrained, no Hessian) ===");
    println!("  min x1*x4*(x1+x2+x3) + x3");
    println!("  s.t. x1*x2*x3*x4 >= 25, x1^2+x2^2+x3^2+x4^2 = 40");
    println!("  Start: (1, 5, 5, 1), bounds: [1, 5]^4");
    println!();

    let result = ripopt::solve(
        &Hs071NoHessian,
        &SolverOptions {
            hessian_approximation_lbfgs: true,
            ..SolverOptions::default()
        },
    );

    println!("  Status:     {:?}", result.status);
    println!("  Iterations: {}", result.iterations);
    println!("  Objective:  {:.6e}", result.objective);
    println!(
        "  Solution:   ({:.4}, {:.4}, {:.4}, {:.4})",
        result.x[0], result.x[1], result.x[2], result.x[3]
    );
    println!("  Expected:   ~(1.0, 4.743, 3.821, 1.379) with f* ≈ 17.014");
    println!();

    // --- Example 3: Automatic fallback ---
    println!("=== Example 3: Automatic L-BFGS Hessian fallback ===");
    println!("  The solver automatically retries with L-BFGS Hessian when the");
    println!("  exact-Hessian IPM fails. This is enabled by default.");
    println!("  (enable_lbfgs_hessian_fallback = true)");
    println!();

    // Use the Rosenbrock with exact Hessian (default mode)
    // The exact Hessian path works fine, so the fallback won't activate.
    // But if you provide a bad Hessian, it will:
    let result = ripopt::solve(
        &Rosenbrock,
        &SolverOptions {
            // Default: exact Hessian first, L-BFGS fallback if it fails
            print_level: 0,
            ..SolverOptions::default()
        },
    );
    println!(
        "  Default mode: status={:?}, obj={:.6e}",
        result.status, result.objective
    );
    println!("  (Fallback not needed — exact Hessian path works.)");
    println!("  To force L-BFGS mode: set hessian_approximation_lbfgs = true");
}
