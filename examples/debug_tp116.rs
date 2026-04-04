use ripopt::{NlpProblem, SolverOptions};

/// TP116 from the Hock-Schittkowski test suite
/// n=13 variables, m=15 constraints (5 linear inequality, 10 nonlinear inequality)
/// Objective: x[10] + x[11] + x[12] (linear)
/// Known optimal: 97.5884089805
///
/// BUG IN GENERATED CODE: Variables x[6..=8] and x[10..=12] have x_l > x_u = 0.0
/// in hs_problems.rs. The Fortran source actually specifies:
///   x_u[6,7,8] = 1000.0  (from XU(I+6)=1.D+3, I=1..3)
///   x_u[10,11,12] = 150.0 (from XU(I+10)=150.D0, I=1..3)
/// The code generator failed to translate these, defaulting x_u to 0.0.
/// cyipopt detects invalid bounds and returns status -11 (Invalid_Problem_Definition).
/// ripopt does not check bounds upfront, so it tries to solve and hits RestorationFailed.
///
/// This debug version uses the CORRECT bounds from the Fortran source.
struct TP116;

impl NlpProblem for TP116 {
    fn num_variables(&self) -> usize { 13 }
    fn num_constraints(&self) -> usize { 15 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        // Variables x[0]..x[5]: bounded intervals
        x_l[0] = 0.1;  x_u[0] = 1.0;
        x_l[1] = 0.1;  x_u[1] = 1.0;
        x_l[2] = 0.1;  x_u[2] = 1.0;
        x_l[3] = 0.0001; x_u[3] = 0.1;
        x_l[4] = 0.1;  x_u[4] = 0.9;
        x_l[5] = 0.1;  x_u[5] = 0.9;
        // Variables x[6]..x[8]: Fortran XU(I+6)=1.D+3 for I=1..3 => x_u = 1000.0
        // Generated code INCORRECTLY has x_u=0.0 (bug: x_l > x_u)
        x_l[6] = 0.1;   x_u[6] = 1000.0;
        x_l[7] = 0.1;   x_u[7] = 1000.0;
        x_l[8] = 500.0;  x_u[8] = 1000.0;
        // Variable x[9]: Fortran XU(10)=500.D0
        x_l[9] = 0.1;   x_u[9] = 500.0;
        // Variables x[10]..x[12]: Fortran XU(I+10)=150.D0 for I=1..3 => x_u = 150.0
        // Generated code INCORRECTLY has x_u=0.0 (bug: x_l > x_u)
        x_l[10] = 1.0;    x_u[10] = 150.0;
        x_l[11] = 0.0001; x_u[11] = 150.0;
        x_l[12] = 0.0001; x_u[12] = 150.0;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        // All 15 constraints are >= 0 (inequality: g(x) >= 0)
        for i in 0..15 {
            g_l[i] = 0.0;
            g_u[i] = f64::INFINITY;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.5;
        x0[1] = 0.8;
        x0[2] = 0.9;
        x0[3] = 0.1;
        x0[4] = 0.14;
        x0[5] = 0.5;
        x0[6] = 489.0;
        x0[7] = 80.0;
        x0[8] = 650.0;
        x0[9] = 450.0;
        x0[10] = 150.0;
        x0[11] = 150.0;
        x0[12] = 150.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[10] + x[11] + x[12]
    }

    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) {
        for g in grad.iter_mut() { *g = 0.0; }
        grad[10] = 1.0;
        grad[11] = 1.0;
        grad[12] = 1.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -x[1] + x[2];
        g[1] = -x[0] + x[1];
        g[2] = -0.002*x[6] + 0.002*x[7] + 1.0;
        g[3] = x[10] + x[11] + x[12] - 50.0;
        g[4] = -x[10] - x[11] - x[12] + 250.0;
        g[5] = x[12] + 1.231059*x[2]*x[9] - 1.262626*x[9];
        g[6] = 0.00975*x[1].powi(2) - 0.975*x[1]*x[4] - 0.03475*x[1] + x[4];
        g[7] = 0.00975*x[2].powi(2) - 0.975*x[2]*x[5] - 0.03475*x[2] + x[5];
        g[8] = -x[0]*x[7] - x[3]*x[6] + x[3]*x[7] + x[4]*x[6];
        g[9] = -x[4] - x[5] + 0.002*x[0]*x[7] - 0.002*x[1]*x[8] - 0.002*x[4]*x[7] + 0.002*x[5]*x[8] + 1.0;
        g[10] = x[1]*x[8] + x[1]*x[9] - 500.0*x[1] - x[2]*x[9] - x[5]*x[8] + 500.0*x[5];
        g[11] = x[1] - 0.002*x[1]*x[9] + 0.002*x[2]*x[9] - 0.9;
        g[12] = 0.00975*x[0].powi(2) - 0.975*x[0]*x[3] - 0.03475*x[0] + x[3];
        g[13] = 1.231059*x[0]*x[7] + x[10] - 1.262626*x[7];
        g[14] = 1.231059*x[1]*x[8] + x[11] - 1.262626*x[8];
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14],
         vec![1, 2, 0, 1, 6, 7, 10, 11, 12, 10, 11, 12, 2, 9, 12, 1, 4, 2, 5, 0, 3, 4, 6, 7, 0, 1, 4, 5, 7, 8, 1, 2, 5, 8, 9, 1, 2, 9, 0, 3, 0, 7, 10, 1, 8, 11])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = -1.0;
        vals[1] = 1.0;
        vals[2] = -1.0;
        vals[3] = 1.0;
        vals[4] = -0.002;
        vals[5] = 0.002;
        vals[6] = 1.0;
        vals[7] = 1.0;
        vals[8] = 1.0;
        vals[9] = -1.0;
        vals[10] = -1.0;
        vals[11] = -1.0;
        vals[12] = 1.231059*x[9];
        vals[13] = 1.231059*x[2] - 1.262626;
        vals[14] = 1.0;
        vals[15] = 0.0195*x[1] - 0.975*x[4] - 0.03475;
        vals[16] = 1.0 - 0.975*x[1];
        vals[17] = 0.0195*x[2] - 0.975*x[5] - 0.03475;
        vals[18] = 1.0 - 0.975*x[2];
        vals[19] = -x[7];
        vals[20] = -x[6] + x[7];
        vals[21] = x[6];
        vals[22] = -x[3] + x[4];
        vals[23] = -x[0] + x[3];
        vals[24] = 0.002*x[7];
        vals[25] = -0.002*x[8];
        vals[26] = -0.002*x[7] - 1.0;
        vals[27] = 0.002*x[8] - 1.0;
        vals[28] = 0.002*x[0] - 0.002*x[4];
        vals[29] = -0.002*x[1] + 0.002*x[5];
        vals[30] = x[8] + x[9] - 500.0;
        vals[31] = -x[9];
        vals[32] = -x[8] + 500.0;
        vals[33] = x[1] - x[5];
        vals[34] = x[1] - x[2];
        vals[35] = 1.0 - 0.002*x[9];
        vals[36] = 0.002*x[9];
        vals[37] = -0.002*x[1] + 0.002*x[2];
        vals[38] = 0.0195*x[0] - 0.975*x[3] - 0.03475;
        vals[39] = 1.0 - 0.975*x[0];
        vals[40] = 1.231059*x[7];
        vals[41] = 1.231059*x[0] - 1.262626;
        vals[42] = 1.0;
        vals[43] = 1.231059*x[8];
        vals[44] = 1.231059*x[1] - 1.262626;
        vals[45] = 1.0;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9],
         vec![0, 1, 2, 0, 1, 2, 3, 4, 0, 3, 4, 1, 5, 1, 2])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = lambda[12] * 0.0195;
        vals[1] = lambda[6] * 0.0195;
        vals[2] = lambda[7] * 0.0195;
        vals[3] = lambda[12] * (-0.975);
        vals[4] = lambda[6] * (-0.975);
        vals[5] = lambda[7] * (-0.975);
        vals[6] = lambda[8] * (-1.0);
        vals[7] = lambda[8] * 1.0;
        vals[8] = lambda[8] * (-1.0) + lambda[9] * 0.002 + lambda[13] * 1.231059;
        vals[9] = lambda[8] * 1.0;
        vals[10] = lambda[9] * (-0.002);
        vals[11] = lambda[9] * (-0.002) + lambda[10] * 1.0 + lambda[14] * 1.231059;
        vals[12] = lambda[9] * 0.002 + lambda[10] * (-1.0);
        vals[13] = lambda[10] * 1.0 + lambda[11] * (-0.002);
        vals[14] = lambda[5] * 1.231059 + lambda[10] * (-1.0) + lambda[11] * 0.002;
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    println!("=== TP116 Debug ===");
    println!("n=13 variables, m=15 constraints (5 linear ineq, 10 nonlinear ineq)");
    println!("Objective: x[10] + x[11] + x[12] (linear)");
    println!("Known optimal: 97.5884089805");
    println!("NOTE: cyipopt fails with status -11 (invalid bounds in generated code)");
    println!("NOTE: Generated code had x_l > x_u for vars 6,7,8,10,11,12 (x_u=0.0 instead of 1000/150)");
    println!("This debug version uses CORRECT bounds from Fortran source:\n  x_u[6,7,8]=1000, x_u[10,11,12]=150\n");

    // Evaluate constraints at initial point to check feasibility
    let x0 = [0.5, 0.8, 0.9, 0.1, 0.14, 0.5, 489.0, 80.0, 650.0, 450.0, 150.0, 150.0, 150.0];
    let mut g = vec![0.0; 15];
    TP116.constraints(&x0, true, &mut g);
    println!("Constraint values at initial point:");
    for (i, gi) in g.iter().enumerate() {
        let status = if *gi >= 0.0 { "OK (>=0)" } else { "VIOLATED (<0)" };
        println!("  g[{:2}] = {:12.6}  {}", i, gi, status);
    }
    println!();

    let options = SolverOptions {
        tol: 1e-8,
        max_iter: 200,
        mu_strategy_adaptive: true,
        print_level: 10,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&TP116, &options);
    println!("\nStatus: {:?}", result.status);
    println!("Objective: {:.10}", result.objective);
    println!("Iterations: {}", result.iterations);
    println!("x: {:?}", result.x);
    println!("Constraint multipliers: {:?}", result.constraint_multipliers);
    println!("Known optimal: 97.5884089805");
}
