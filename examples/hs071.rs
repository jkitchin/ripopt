//! Hock-Schittkowski problem 71 (constrained NLP).
//!
//! min  x1*x4*(x1+x2+x3) + x3
//! s.t. x1*x2*x3*x4 >= 25       (g1)
//!      x1^2+x2^2+x3^2+x4^2 = 40  (g2)
//!      1 <= xi <= 5, i=1..4
//!
//! Starting point: (1.0, 5.0, 5.0, 1.0)
//! Known solution:  x* = (1.0, 4.743, 3.821, 1.379), f* ~ 17.014

use ripopt::{NlpProblem, SolveStatus, SolverOptions};

struct Hs071;

impl NlpProblem for Hs071 {
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
        // g1: x1*x2*x3*x4 >= 25  =>  25 <= g1 <= +inf
        g_l[0] = 25.0;
        g_u[0] = f64::INFINITY;
        // g2: x1^2+x2^2+x3^2+x4^2 = 40  =>  40 <= g2 <= 40
        g_l[1] = 40.0;
        g_u[1] = 40.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0;
        x0[1] = 5.0;
        x0[2] = 5.0;
        x0[3] = 1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        // df/dx1 = x4*(x1+x2+x3) + x1*x4 = x4*(2*x1+x2+x3)
        grad[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        // df/dx2 = x1*x4
        grad[1] = x[0] * x[3];
        // df/dx3 = x1*x4 + 1
        grad[2] = x[0] * x[3] + 1.0;
        // df/dx4 = x1*(x1+x2+x3)
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // g1 depends on all 4 vars, g2 depends on all 4 vars => 8 entries
        (
            vec![0, 0, 0, 0, 1, 1, 1, 1],
            vec![0, 1, 2, 3, 0, 1, 2, 3],
        )
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        // dg1/dx1 = x2*x3*x4
        vals[0] = x[1] * x[2] * x[3];
        // dg1/dx2 = x1*x3*x4
        vals[1] = x[0] * x[2] * x[3];
        // dg1/dx3 = x1*x2*x4
        vals[2] = x[0] * x[1] * x[3];
        // dg1/dx4 = x1*x2*x3
        vals[3] = x[0] * x[1] * x[2];
        // dg2/dx1 = 2*x1
        vals[4] = 2.0 * x[0];
        // dg2/dx2 = 2*x2
        vals[5] = 2.0 * x[1];
        // dg2/dx3 = 2*x3
        vals[6] = 2.0 * x[2];
        // dg2/dx4 = 2*x4
        vals[7] = 2.0 * x[3];
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Lower triangle entries of the 4x4 Hessian of the Lagrangian.
        // We enumerate all lower-triangle positions that can be nonzero:
        //   (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)
        (
            vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
        )
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        // Hessian of objective:
        //   d2f/dx1dx1 = 2*x4
        //   d2f/dx1dx2 = x4       (sym)
        //   d2f/dx1dx3 = x4       (sym)
        //   d2f/dx1dx4 = 2*x1+x2+x3  (sym)
        //   d2f/dx2dx4 = x1       (sym)
        //   d2f/dx3dx4 = x1       (sym)
        //   all others = 0

        // Hessian of g1 = x1*x2*x3*x4:
        //   d2g1/dx1dx2 = x3*x4
        //   d2g1/dx1dx3 = x2*x4
        //   d2g1/dx1dx4 = x2*x3
        //   d2g1/dx2dx3 = x1*x4
        //   d2g1/dx2dx4 = x1*x3
        //   d2g1/dx3dx4 = x1*x2
        //   diagonal = 0

        // Hessian of g2 = x1^2+x2^2+x3^2+x4^2:
        //   d2g2/dxi dxi = 2, off-diagonal = 0

        // Index 0: (0,0) = obj_factor*2*x4 + lambda[1]*2
        vals[0] = obj_factor * 2.0 * x[3] + lambda[1] * 2.0;
        // Index 1: (1,0) = obj_factor*x4 + lambda[0]*x3*x4
        vals[1] = obj_factor * x[3] + lambda[0] * x[2] * x[3];
        // Index 2: (1,1) = lambda[1]*2
        vals[2] = lambda[1] * 2.0;
        // Index 3: (2,0) = obj_factor*x4 + lambda[0]*x2*x4
        vals[3] = obj_factor * x[3] + lambda[0] * x[1] * x[3];
        // Index 4: (2,1) = lambda[0]*x1*x4
        vals[4] = lambda[0] * x[0] * x[3];
        // Index 5: (2,2) = lambda[1]*2
        vals[5] = lambda[1] * 2.0;
        // Index 6: (3,0) = obj_factor*(2*x1+x2+x3) + lambda[0]*x2*x3
        vals[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]) + lambda[0] * x[1] * x[2];
        // Index 7: (3,1) = obj_factor*x1 + lambda[0]*x1*x3
        vals[7] = obj_factor * x[0] + lambda[0] * x[0] * x[2];
        // Index 8: (3,2) = obj_factor*x1 + lambda[0]*x1*x2
        vals[8] = obj_factor * x[0] + lambda[0] * x[0] * x[1];
        // Index 9: (3,3) = lambda[1]*2
        vals[9] = lambda[1] * 2.0;
        true
    }
}

fn main() {
    env_logger::init();

    let problem = Hs071;
    let options = SolverOptions {
        max_iter: 500,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    println!("Solving HS071...");
    println!("  Start: (1.0, 5.0, 5.0, 1.0)");
    println!();

    let result = ripopt::solve(&problem, &options);

    println!("Status:     {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Objective:  {:.10}", result.objective);
    println!(
        "Solution:   x = ({:.6}, {:.6}, {:.6}, {:.6})",
        result.x[0], result.x[1], result.x[2], result.x[3]
    );
    println!(
        "Constraint multipliers: ({:.6}, {:.6})",
        result.constraint_multipliers[0], result.constraint_multipliers[1]
    );
    println!(
        "Constraint values:      g1 = {:.6}, g2 = {:.6}",
        result.constraint_values[0], result.constraint_values[1]
    );

    assert_eq!(result.status, SolveStatus::Optimal);
}
