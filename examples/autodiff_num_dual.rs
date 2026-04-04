//! Automatic differentiation with num-dual: zero hand-derived derivatives.
//!
//! This example solves HS071 using the `num-dual` crate to automatically compute
//! gradients, Jacobians, and Hessians. Compare with `examples/hs071.rs` where all
//! derivatives are hand-derived (146 lines vs ~120 lines here, and no derivative algebra).
//!
//! The pattern shown here is the same one used by the `ipopt-ad` crate
//! (https://github.com/prehner/ipopt-ad), which provides a polished wrapper
//! including automatic sparsity detection and caching.
//!
//! Run with: cargo run --example autodiff_num_dual --features num-dual

use nalgebra::SVector;
use num_dual::{gradient, hessian, jacobian, DualNum};
use ripopt::{NlpProblem, SolveStatus, SolverOptions};

// ---------------------------------------------------------------------------
// Problem definition: users only write the math once, generic over D.
// ---------------------------------------------------------------------------

fn hs071_objective<D: DualNum<f64> + Copy>(x: SVector<D, 4>) -> D {
    // f(x) = x0 * x3 * (x0 + x1 + x2) + x2
    x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
}

fn hs071_constraints<D: DualNum<f64> + Copy>(x: SVector<D, 4>) -> SVector<D, 2> {
    // g0 = x0 * x1 * x2 * x3  (>= 25)
    // g1 = x0^2 + x1^2 + x2^2 + x3^2  (= 40)
    SVector::<D, 2>::from([
        x[0] * x[1] * x[2] * x[3],
        x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3],
    ])
}

// ---------------------------------------------------------------------------
// AD-powered NlpProblem wrapper: derivatives are computed automatically.
// ---------------------------------------------------------------------------

struct Hs071AD;

impl NlpProblem for Hs071AD {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.fill(1.0);
        x_u.fill(5.0);
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0;  g_u[0] = f64::INFINITY;
        g_l[1] = 40.0;  g_u[1] = 40.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        hs071_objective(SVector::from_column_slice(x))
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let xv = SVector::from_column_slice(x);
        let (_f, g) = gradient(hs071_objective, &xv);
        grad.copy_from_slice(g.as_slice());
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let xv = SVector::from_column_slice(x);
        let cv: SVector<f64, 2> = hs071_constraints(xv);
        g.copy_from_slice(cv.as_slice());
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // HS071: both constraints depend on all 4 variables (dense 2x4)
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..2 {
            for j in 0..4 {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let xv = SVector::from_column_slice(x);
        let (_g, jac) = jacobian(hs071_constraints, &xv);
        // jac is 2x4; iterate in row-major order matching jacobian_structure
        let mut idx = 0;
        for i in 0..2 {
            for j in 0..4 {
                vals[idx] = jac[(i, j)];
                idx += 1;
            }
        }
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Dense lower triangle of 4x4
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

    fn hessian_values(
        &self,
        x: &[f64],
        _new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) {
        let xv = SVector::from_column_slice(x);

        // Hessian of the Lagrangian: obj_factor * nabla^2 f + sum_i lambda_i * nabla^2 g_i
        // Compute via AD on the combined Lagrangian function.
        let lam0 = lambda[0];
        let lam1 = lambda[1];
        let lagrangian = |x: SVector<_, 4>| {
            let obj = hs071_objective(x);
            let con = hs071_constraints(x);
            obj * obj_factor + con[0] * lam0 + con[1] * lam1
        };

        let (_f, _g, h) = hessian(lagrangian, &xv);

        // Extract lower triangle
        let mut idx = 0;
        for i in 0..4 {
            for j in 0..=i {
                vals[idx] = h[(i, j)];
                idx += 1;
            }
        }
    }
}

fn main() {
    let options = SolverOptions {
        print_level: 5,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&Hs071AD, &options);

    println!();
    println!("HS071 with automatic differentiation (num-dual)");
    println!("================================================");
    println!("Status:     {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Objective:  {:.10}", result.objective);
    println!(
        "Solution:   ({:.6}, {:.6}, {:.6}, {:.6})",
        result.x[0], result.x[1], result.x[2], result.x[3]
    );
    println!();
    println!("No hand-derived derivatives needed! The num-dual crate computes");
    println!("exact gradients, Jacobians, and Hessians automatically.");

    assert_eq!(result.status, SolveStatus::Optimal);
    assert!(
        (result.objective - 17.014).abs() < 0.01,
        "Unexpected objective: {}",
        result.objective
    );
}
