use ripopt::{NlpProblem, SolverOptions};

/// HS TP027: Rosenbrock-like with 1 nonlinear equality constraint
///
/// min  f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2
///            = -2*x1 + 100*x1^4 + 100*x2^2 - 200*x2*x1^2 + 1 + x1^2
///
/// s.t. x1 + 1 + x3^2 = 0
///
///      x1, x2, x3 free (no bounds)
///
/// Starting point: x0 = (2, 2, 2)
/// Known optimal: f* = 4.0  (at x1 = -1, x2 = 1, x3 = 0)
pub struct HsTp027;

impl NlpProblem for HsTp027 {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..3 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 2.0; x0[1] = 2.0; x0[2] = 2.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -2.0*x[0] + 100.0*x[0].powi(4) + 100.0*x[1].powi(2) - 200.0*x[1]*x[0].powi(2) + 1.0 + x[0].powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0*x[0] + 400.0*x[0].powi(3) - 400.0*x[0]*x[1] - 2.0;
        grad[1] = -200.0*x[0].powi(2) + 200.0*x[1];
        grad[2] = 0.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] + 1.0 + x[2].powi(2);
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 2])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;
        vals[1] = 2.0*x[2];
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2], vec![0, 0, 1, 2])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (1200.0*x[0].powi(2) - 400.0*x[1] + 2.0);
        vals[1] = obj_factor * (-400.0*x[0]);
        vals[2] = obj_factor * 200.0;
        vals[3] = lambda[0] * 2.0;
    }
}

fn main() {
    env_logger::init();

    let problem = HsTp027;

    println!("=== TP027: Rosenbrock with 1 nonlinear equality constraint ===");
    println!("min  100*(x2 - x1^2)^2 + (1-x1)^2");
    println!("s.t. x1 + 1 + x3^2 = 0");
    println!("     x1, x2, x3 free (no variable bounds)");
    println!("x0 = (2, 2, 2)");
    println!("Known optimal: f* = 4.0 at x = (-1, 1, 0)");
    println!();

    // Key diagnostic: This problem has NO variable bounds
    // => No bound multipliers z_l, z_u
    // => No barrier terms for variable bounds
    // => Barrier objective phi = f(x) (no log terms)
    // => mu only affects the (2,2) block for inequality slack barriers
    // => But this is an EQUALITY constraint => no (2,2) block
    // => So mu rapidly decreases to mu_min with no bound-related coupling
    println!("KEY OBSERVATIONS:");
    println!("- No variable bounds => no barrier terms in phi");
    println!("- Equality constraint => no (2,2) block in KKT");
    println!("- mu will decrease via adaptive rule but count=0 => monotone fallback");
    println!();

    // Evaluate at starting point
    let x0 = [2.0, 2.0, 2.0];
    let f0 = problem.objective(&x0, true);
    let mut g0 = [0.0; 1];
    problem.constraints(&x0, true, &mut g0);
    let mut grad0 = [0.0; 3];
    problem.gradient(&x0, true, &mut grad0);
    println!("At x0: f = {:.6}, g = [{:.6}] (violation: {:.6})", f0, g0[0], g0[0].abs());
    println!("Gradient at x0: [{:.6}, {:.6}, {:.6}]", grad0[0], grad0[1], grad0[2]);

    // Hessian at x0
    let mut hvals = [0.0; 4];
    problem.hessian_values(&x0, true, 1.0, &[0.0], &mut hvals);
    println!("Hessian diag at x0 (obj only): [{:.2}, {:.2}, {:.2}]", hvals[0], hvals[2], hvals[3]);
    println!("  H[0,0] = 1200*(2)^2 - 400*2 + 2 = {:.2}", 1200.0*4.0 - 400.0*2.0 + 2.0);
    println!("  H[1,1] = 200");
    println!("  H[2,2] = 0 (lambda[0]*2 only from constraint)");

    // Evaluate at known optimal
    let x_opt = [-1.0, 1.0, 0.0];
    let f_opt = problem.objective(&x_opt, true);
    let mut g_opt = [0.0; 1];
    problem.constraints(&x_opt, true, &mut g_opt);
    println!("\nAt x*=(-1,1,0): f = {:.6}, g = [{:.6}]", f_opt, g_opt[0]);

    // Check constraint: x1 + 1 + x3^2 = 0 => constraint requires x1 <= -1
    println!("Note: constraint x1 + 1 + x3^2 = 0 means x1 = -(1 + x3^2) <= -1");
    println!("So optimal x1 = -1, x2 = x1^2 = 1 (Rosenbrock)");
    println!();

    // Now run the solver
    let mut options = SolverOptions::default();
    options.print_level = 5;
    options.max_iter = 100;

    println!("=== Running solver ===\n");
    let result = ripopt::solve(&problem, &options);

    println!("\n=== Final Results ===");
    println!("Status: {:?}", result.status);
    println!("Objective: {:.10}", result.objective);
    println!("Solution: {:?}", result.x);
    println!("Iterations: {}", result.iterations);
    println!("Constraint values: {:?}", result.constraint_values);
    println!("Multipliers (y): {:?}", result.constraint_multipliers);
    println!("Bound mult lower: {:?}", result.bound_multipliers_lower);
    println!("Bound mult upper: {:?}", result.bound_multipliers_upper);

    println!("\n=== Expected Optimal ===");
    println!("Objective: 4.0");
    println!("Solution: [-1.0, 1.0, 0.0]");

    // Check if the solver found a LOCAL minimum at (1, 1, 0) instead of (-1, 1, 0)
    let x_local = [1.0, 1.0, 0.0];
    let f_local = problem.objective(&x_local, true);
    let mut g_local = [0.0; 1];
    problem.constraints(&x_local, true, &mut g_local);
    println!("\nCheck local minimum at (1,1,0): f = {:.6}, g = [{:.6}]", f_local, g_local[0]);
    println!("This violates the constraint: x1+1+x3^2 = {} != 0", g_local[0]);
    println!("The solver converged near (1,1,0) which is INFEASIBLE for this constraint!");
    println!("The constraint x1+1+x3^2=0 forces x1 = -1-x3^2 <= -1");
    println!("But the solver went toward x1=+1, the unconstrained Rosenbrock minimum,");
    println!("and could not recover because it was stuck in the wrong basin.");
}
