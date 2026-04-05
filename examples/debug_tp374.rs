use ripopt::{NlpProblem, SolverOptions};
use std::f64::consts::PI;

/// TP374 from the Hock-Schittkowski test suite.
///
/// Minimize x[9]
/// subject to 35 nonlinear inequality constraints g_i(x) >= 0
///
/// The constraints involve trigonometric sums:
///   A(z, x) = sum_{k=1}^{9} x[k-1] * cos(k*z)
///   B(z, x) = sum_{k=1}^{9} x[k-1] * sin(k*z)
///   G(z, x) = A(z, x)^2 + B(z, x)^2
///
/// Constraints 0..9   (i=0..9):   G(z_i, x) - (1 - x[9])^2 >= 0,  z_i = pi/4 * (i * 0.1)
/// Constraints 10..19 (i=10..19): (1 + x[9])^2 - G(z_i, x) >= 0,  z_i = pi/4 * ((i-10) * 0.1)
/// Constraints 20..34 (i=20..34): x[9]^2 - G(z_i, x) >= 0,         z_i = pi/4 * (1.2 + (i-20) * 0.2)
///
/// n=10, m=35, all 35 constraints are nonlinear inequalities (ninl=35).
/// Known optimal: f* = 0.233264
///
/// NOTE: The generated code in hs_problems.rs has EMPTY constraint/jacobian bodies
/// because the code generator failed to translate the Fortran trigonometric
/// constraint functions. This file provides a correct hand-translated implementation
/// from the original Fortran source (PROB.FOR subroutine TP374).

struct TP374;

// Helper functions matching the Fortran TP374A, TP374B, TP374G
fn tp374_a(z: f64, x: &[f64]) -> f64 {
    let mut val = 0.0;
    for k in 1..=9 {
        val += x[k - 1] * (k as f64 * z).cos();
    }
    val
}

fn tp374_b(z: f64, x: &[f64]) -> f64 {
    let mut val = 0.0;
    for k in 1..=9 {
        val += x[k - 1] * (k as f64 * z).sin();
    }
    val
}

fn tp374_g(z: f64, x: &[f64]) -> f64 {
    let a = tp374_a(z, x);
    let b = tp374_b(z, x);
    a * a + b * b
}

impl NlpProblem for TP374 {
    fn num_variables(&self) -> usize { 10 }
    fn num_constraints(&self) -> usize { 35 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..10 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..35 {
            g_l[i] = 0.0;
            g_u[i] = f64::INFINITY;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for i in 0..10 {
            x0[i] = 0.1;
        }
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[9];
        true
    }

    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        for i in 0..9 { grad[i] = 0.0; }
        grad[9] = 1.0;
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        // Constraints 0..9: G(z_i, x) - (1 - x[9])^2 >= 0
        for i in 0..10 {
            let z = PI / 4.0 * (i as f64 * 0.1);
            g[i] = tp374_g(z, x) - (1.0 - x[9]).powi(2);
        }
        // Constraints 10..19: (1 + x[9])^2 - G(z_i, x) >= 0
        for i in 10..20 {
            let z = PI / 4.0 * ((i - 10) as f64 * 0.1);
            g[i] = (1.0 + x[9]).powi(2) - tp374_g(z, x);
        }
        // Constraints 20..34: x[9]^2 - G(z_i, x) >= 0
        for i in 20..35 {
            let z = PI / 4.0 * (1.2 + (i - 20) as f64 * 0.2);
            g[i] = x[9].powi(2) - tp374_g(z, x);
        }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Each of the 35 constraints depends on all 10 variables
        let mut rows = Vec::with_capacity(35 * 10);
        let mut cols = Vec::with_capacity(35 * 10);
        for i in 0..35 {
            for j in 0..10 {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        let mut idx = 0;

        // Constraints 0..9: g_i = G(z, x) - (1 - x[9])^2
        // dg/dx[k-1] = 2*(A*cos(k*z) + B*sin(k*z))  for k=1..9
        // dg/dx[9]   = 2*(1 - x[9])
        for i in 0..10 {
            let z = PI / 4.0 * (i as f64 * 0.1);
            let a = tp374_a(z, x);
            let b = tp374_b(z, x);
            for k in 1..=9 {
                vals[idx] = 2.0 * (a * (k as f64 * z).cos() + b * (k as f64 * z).sin());
                idx += 1;
            }
            vals[idx] = 2.0 * (1.0 - x[9]);
            idx += 1;
        }

        // Constraints 10..19: g_i = (1 + x[9])^2 - G(z, x)
        // dg/dx[k-1] = -2*(A*cos(k*z) + B*sin(k*z))  for k=1..9
        // dg/dx[9]   = 2*(1 + x[9])
        for i in 10..20 {
            let z = PI / 4.0 * ((i - 10) as f64 * 0.1);
            let a = tp374_a(z, x);
            let b = tp374_b(z, x);
            for k in 1..=9 {
                vals[idx] = -2.0 * (a * (k as f64 * z).cos() + b * (k as f64 * z).sin());
                idx += 1;
            }
            vals[idx] = 2.0 * (1.0 + x[9]);
            idx += 1;
        }

        // Constraints 20..34: g_i = x[9]^2 - G(z, x)
        // dg/dx[k-1] = -2*(A*cos(k*z) + B*sin(k*z))  for k=1..9
        // dg/dx[9]   = 2*x[9]
        for i in 20..35 {
            let z = PI / 4.0 * (1.2 + (i - 20) as f64 * 0.2);
            let a = tp374_a(z, x);
            let b = tp374_b(z, x);
            for k in 1..=9 {
                vals[idx] = -2.0 * (a * (k as f64 * z).cos() + b * (k as f64 * z).sin());
                idx += 1;
            }
            vals[idx] = 2.0 * x[9];
            idx += 1;
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // 10x10 lower triangle: all pairs (i, j) with i >= j
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..10 {
            for j in 0..=i {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        // Objective Hessian is zero (f = x[9] is linear).
        // Constraint Hessians:
        // For constraints 0..9 (group 1): g = A^2 + B^2 - (1-x9)^2
        //   d2g/dx[j]dx[k] = 2*(cos(j*z)*cos(k*z) + sin(j*z)*sin(k*z))
        //                   = 2*cos((j-k)*z)       for j,k in 1..9
        //   d2g/dx9^2 = 2
        //   cross terms dx[k]*dx9 = 0 for k<9
        //
        // For constraints 10..19 (group 2): g = (1+x9)^2 - A^2 - B^2
        //   d2g/dx[j]dx[k] = -2*cos((j-k)*z)      for j,k in 1..9
        //   d2g/dx9^2 = 2
        //   cross terms dx[k]*dx9 = 0 for k<9
        //
        // For constraints 20..34 (group 3): g = x9^2 - A^2 - B^2
        //   d2g/dx[j]dx[k] = -2*cos((j-k)*z)      for j,k in 1..9
        //   d2g/dx9^2 = 2
        //   cross terms dx[k]*dx9 = 0 for k<9

        // Initialize to zero
        for v in vals.iter_mut() { *v = 0.0; }

        // Helper: map lower-triangle (i, j) with i >= j to flat index
        // Index of (i,j) = i*(i+1)/2 + j
        let lt = |i: usize, j: usize| -> usize { i * (i + 1) / 2 + j };

        // Group 1: constraints 0..9
        for ci in 0..10 {
            let lam = lambda[ci];
            if lam == 0.0 { continue; }
            let z = PI / 4.0 * (ci as f64 * 0.1);
            // x[j-1], x[k-1] block (j,k = 1..9, so indices 0..8)
            for j in 1..=9usize {
                for k in 1..=j {
                    let h = 2.0 * ((j as f64 - k as f64) * z).cos();
                    vals[lt(j - 1, k - 1)] += lam * h;
                }
            }
            // x[9]^2 term: d2g/dx9^2 = 2
            vals[lt(9, 9)] += lam * 2.0;
        }

        // Group 2: constraints 10..19
        for ci in 10..20 {
            let lam = lambda[ci];
            if lam == 0.0 { continue; }
            let z = PI / 4.0 * ((ci - 10) as f64 * 0.1);
            for j in 1..=9usize {
                for k in 1..=j {
                    let h = -2.0 * ((j as f64 - k as f64) * z).cos();
                    vals[lt(j - 1, k - 1)] += lam * h;
                }
            }
            vals[lt(9, 9)] += lam * 2.0;
        }

        // Group 3: constraints 20..34
        for ci in 20..35 {
            let lam = lambda[ci];
            if lam == 0.0 { continue; }
            let z = PI / 4.0 * (1.2 + (ci - 20) as f64 * 0.2);
            for j in 1..=9usize {
                for k in 1..=j {
                    let h = -2.0 * ((j as f64 - k as f64) * z).cos();
                    vals[lt(j - 1, k - 1)] += lam * h;
                }
            }
            vals[lt(9, 9)] += lam * 2.0;
        }
        true
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .init();

    let problem = TP374;

    // Verify constraint evaluation at initial point
    let mut x0 = vec![0.0; 10];
    problem.initial_point(&mut x0);
    let mut g0 = vec![0.0; 35];
    problem.constraints(&x0, true, &mut g0);
    println!("Initial point: {:?}", x0);
    println!("f(x0) = {}", { let mut _obj_val = 0.0; problem.objective(&x0, true, &mut _obj_val); _obj_val });
    println!("Constraints at x0 (first 10): {:?}", &g0[..10]);
    println!("Constraints at x0 (10..20):   {:?}", &g0[10..20]);
    println!("Constraints at x0 (20..35):   {:?}", &g0[20..35]);
    println!();

    let options = SolverOptions {
        print_level: 10,
        max_iter: 3000,
        mu_strategy_adaptive: true,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    println!("\n=== RESULT ===");
    println!("Status: {:?}", result.status);
    println!("Objective: {:.10}", result.objective);
    println!("x: {:?}", result.x);
    println!("Iterations: {}", result.iterations);
    println!("Known optimal: 0.233264");

    if !result.x.is_empty() {
        let mut g_final = vec![0.0; 35];
        problem.constraints(&result.x, true, &mut g_final);
        println!("\nConstraint values at solution:");
        for (i, gv) in g_final.iter().enumerate() {
            let status = if *gv < -1e-6 { " ** VIOLATED **" } else { "" };
            println!("  g[{:2}] = {:+.8e}{}", i, gv, status);
        }
    }
}
