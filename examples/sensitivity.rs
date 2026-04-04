//! Parametric sensitivity analysis example.
//!
//! Demonstrates how to compute how the optimal solution changes when
//! problem parameters are perturbed, without re-solving the NLP.
//!
//! Problem (HS071 with parametric equality constraint):
//!   min  x1*x4*(x1+x2+x3) + x3
//!   s.t. x1*x2*x3*x4 >= 25
//!        x1^2 + x2^2 + x3^2 + x4^2 = p   (p = 40 nominally)
//!        1 <= xi <= 5
//!
//! We solve at p=40, then use sensitivity analysis to predict the solution
//! at p=40.1 and verify against a full re-solve.

use ripopt::{NlpProblem, ParametricNlpProblem, SolveStatus, SolverOptions};

struct Hs071Parametric {
    p: f64,
}

impl NlpProblem for Hs071Parametric {
    fn num_variables(&self) -> usize {
        4
    }
    fn num_constraints(&self) -> usize {
        2
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 {
            x_l[i] = 1.0;
            x_u[i] = 5.0;
        }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0;
        g_u[0] = f64::INFINITY;
        g_l[1] = self.p;
        g_u[1] = self.p;
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
        vals[0] = obj_factor * 2.0 * x[3] + lambda[1] * 2.0;
        vals[1] = obj_factor * x[3] + lambda[0] * x[2] * x[3];
        vals[2] = lambda[1] * 2.0;
        vals[3] = obj_factor * x[3] + lambda[0] * x[1] * x[3];
        vals[4] = lambda[0] * x[0] * x[3];
        vals[5] = lambda[1] * 2.0;
        vals[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]) + lambda[0] * x[1] * x[2];
        vals[7] = obj_factor * x[0] + lambda[0] * x[0] * x[2];
        vals[8] = obj_factor * x[0] + lambda[0] * x[0] * x[1];
        vals[9] = lambda[1] * 2.0;
    }
}

impl ParametricNlpProblem for Hs071Parametric {
    fn num_parameters(&self) -> usize {
        1
    }
    fn jacobian_p_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Constraint 1 has bound = p. Modeling as g(x) - p = 0, so dg/dp = -1.
        (vec![1], vec![0])
    }
    fn jacobian_p_values(&self, _x: &[f64], vals: &mut [f64]) {
        vals[0] = -1.0;
    }
    fn hessian_xp_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![]) // no cross-Hessian terms
    }
    fn hessian_xp_values(
        &self,
        _x: &[f64],
        _obj_factor: f64,
        _lambda: &[f64],
        _vals: &mut [f64],
    ) {
    }
}

fn main() {
    let p0 = 40.0;
    let delta = 0.1;

    let problem = Hs071Parametric { p: p0 };
    let options = SolverOptions {
        print_level: 0,
        ..SolverOptions::default()
    };

    // Step 1: Solve with sensitivity
    println!("=== Parametric Sensitivity Analysis ===\n");
    println!("Solving HS071 at p = {:.1}...", p0);

    let mut ctx = ripopt::solve_with_sensitivity(&problem, &options);
    assert_eq!(ctx.result.status, SolveStatus::Optimal);

    println!("  Status:    {:?}", ctx.result.status);
    println!("  Objective: {:.6}", ctx.result.objective);
    println!(
        "  Solution:  ({:.6}, {:.6}, {:.6}, {:.6})",
        ctx.result.x[0], ctx.result.x[1], ctx.result.x[2], ctx.result.x[3]
    );

    // Step 2: Compute sensitivity for Δp = 0.1
    println!("\nComputing sensitivity for Δp = {:.1}...", delta);

    let dp = [delta];
    let sens = ctx
        .compute_sensitivity(&problem, &[&dp])
        .expect("sensitivity computation failed");

    println!("  dx/dp * Δp:");
    for i in 0..4 {
        println!("    dx[{}]/dp = {:+.6}", i, sens.dx_dp[0][i] / delta);
    }

    // Step 3: Predict solution at p = p0 + delta
    let x_predicted: Vec<f64> = ctx
        .result
        .x
        .iter()
        .zip(sens.dx_dp[0].iter())
        .map(|(x, dx)| x + dx)
        .collect();

    println!(
        "\n  Predicted x(p={:.1}): ({:.6}, {:.6}, {:.6}, {:.6})",
        p0 + delta,
        x_predicted[0],
        x_predicted[1],
        x_predicted[2],
        x_predicted[3]
    );

    // Step 4: Verify by re-solving at the perturbed parameter
    let problem2 = Hs071Parametric { p: p0 + delta };
    let result2 = ripopt::solve(&problem2, &options);

    println!(
        "  Actual    x(p={:.1}): ({:.6}, {:.6}, {:.6}, {:.6})",
        p0 + delta, result2.x[0], result2.x[1], result2.x[2], result2.x[3]
    );

    println!("\n  Prediction errors:");
    for i in 0..4 {
        let err = (x_predicted[i] - result2.x[i]).abs();
        println!("    |Δx[{}]| = {:.2e}", i, err);
    }

    // Step 5: Reduced Hessian (covariance proxy)
    println!("\n=== Reduced Hessian (M⁻¹ upper-left block) ===\n");
    let rh = ctx.reduced_hessian().expect("reduced hessian failed");
    for i in 0..4 {
        print!("  [");
        for j in 0..4 {
            print!("{:+9.4}", rh[i][j]);
        }
        println!(" ]");
    }
}
