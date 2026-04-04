use ripopt::{NlpProblem, SolverOptions};

/// HS TP020: Rosenbrock-like with 3 nonlinear inequality constraints
///
/// min  f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2
///            = -2*x1 + 100*x1^4 + 100*x2^2 - 200*x2*x1^2 + 1 + x1^2
///
/// s.t. g1: x1 + x2^2 >= 0
///      g2: x1^2 + x2 >= 0
///      g3: x1^2 + x2^2 >= 1
///
///      -0.5 <= x1 <= 0.5
///      x2 free
///
/// Starting point: x0 = (0.1, 1.0)
pub struct HsTp020;

impl NlpProblem for HsTp020 {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 3 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = -0.5; x_u[0] = 0.5;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = f64::INFINITY;
        g_l[1] = 0.0; g_u[1] = f64::INFINITY;
        g_l[2] = 0.0; g_u[2] = f64::INFINITY;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.1; x0[1] = 1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -2.0*x[0] + 100.0*x[0].powi(4) + 100.0*x[1].powi(2) - 200.0*x[1]*x[0].powi(2) + 1.0 + x[0].powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0*x[0] + 400.0*x[0].powi(3) - 400.0*x[0]*x[1] - 2.0;
        grad[1] = -200.0*x[0].powi(2) + 200.0*x[1];
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] + x[1].powi(2);
        g[1] = x[0].powi(2) + x[1];
        g[2] = -1.0 + x[0].powi(2) + x[1].powi(2);
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1, 2, 2], vec![0, 1, 0, 1, 0, 1])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;       vals[1] = 2.0*x[1];
        vals[2] = 2.0*x[0];  vals[3] = 1.0;
        vals[4] = 2.0*x[0];  vals[5] = 2.0*x[1];
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (1200.0*x[0].powi(2) - 400.0*x[1] + 2.0) + lambda[1]*2.0 + lambda[2]*2.0;
        vals[1] = obj_factor * (-400.0*x[0]);
        vals[2] = obj_factor * 200.0 + lambda[0]*2.0 + lambda[2]*2.0;
    }
}

fn main() {
    env_logger::init();

    let problem = HsTp020;

    println!("=== TP020: Rosenbrock with 3 nonlinear inequality constraints ===");
    println!("min  100*(x2 - x1^2)^2 + (1-x1)^2");
    println!("s.t. g1: x1 + x2^2 >= 0");
    println!("     g2: x1^2 + x2 >= 0");
    println!("     g3: x1^2 + x2^2 >= 1");
    println!("     -0.5 <= x1 <= 0.5");
    println!("x0 = (0.1, 1.0)");
    println!();

    // Evaluate at starting point
    let x0 = [0.1, 1.0];
    let f0 = problem.objective(&x0, true);
    let mut g0 = [0.0; 3];
    problem.constraints(&x0, true, &mut g0);
    let mut grad0 = [0.0; 2];
    problem.gradient(&x0, true, &mut grad0);
    println!("At x0: f = {:.6}, g = [{:.6}, {:.6}, {:.6}]", f0, g0[0], g0[1], g0[2]);
    println!("Gradient at x0: [{:.6}, {:.6}]", grad0[0], grad0[1]);
    println!("Constraint satisfaction: g1={} g2={} g3={}",
             if g0[0] >= 0.0 { "OK" } else { "VIOLATED" },
             if g0[1] >= 0.0 { "OK" } else { "VIOLATED" },
             if g0[2] >= 0.0 { "OK" } else { "VIOLATED" });

    // Evaluate the Hessian eigenvalues at x0 to see if it's indefinite
    let mut hvals = [0.0; 3];
    problem.hessian_values(&x0, true, 1.0, &[0.0, 0.0, 0.0], &mut hvals);
    // H = [[h00, h10], [h10, h11]] = [[hvals[0], hvals[1]], [hvals[1], hvals[2]]]
    let h00 = hvals[0];
    let h10 = hvals[1];
    let h11 = hvals[2];
    let trace = h00 + h11;
    let det = h00 * h11 - h10 * h10;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let eig1 = (trace + disc) / 2.0;
    let eig2 = (trace - disc) / 2.0;
    println!("\nHessian at x0 (obj only): H = [[{:.2}, {:.2}], [{:.2}, {:.2}]]", h00, h10, h10, h11);
    println!("Hessian eigenvalues: {:.4}, {:.4}", eig1, eig2);
    println!("Hessian is {}", if eig2 >= 0.0 { "PSD" } else { "INDEFINITE" });

    // Check what happens at the solution x* = (0.5, ~0.25)
    // The known optimal for TP020 is approximately x* = (0.5, 0.25), f* = 0.25 (bound active)
    let x_star = [0.5, 0.25];
    let f_star = problem.objective(&x_star, true);
    let mut g_star = [0.0; 3];
    problem.constraints(&x_star, true, &mut g_star);
    let mut grad_star = [0.0; 2];
    problem.gradient(&x_star, true, &mut grad_star);
    println!("\nAt x*=(0.5, 0.25): f = {:.6}, g = [{:.6}, {:.6}, {:.6}]", f_star, g_star[0], g_star[1], g_star[2]);
    println!("Gradient at x*: [{:.6}, {:.6}]", grad_star[0], grad_star[1]);
    println!("Constraint satisfaction at x*: g1={} g2={} g3={}",
             if g_star[0] >= 0.0 { "OK" } else { "VIOLATED" },
             if g_star[1] >= 0.0 { "OK" } else { "VIOLATED" },
             if g_star[2] >= -1e-10 { "OK" } else { "VIOLATED" });
    let mut hvals_star = [0.0; 3];
    problem.hessian_values(&x_star, true, 1.0, &[0.0, 0.0, 0.0], &mut hvals_star);
    let h00s = hvals_star[0];
    let h10s = hvals_star[1];
    let h11s = hvals_star[2];
    let traces = h00s + h11s;
    let dets = h00s * h11s - h10s * h10s;
    let discs = (traces * traces - 4.0 * dets).max(0.0).sqrt();
    println!("Hessian eigenvalues at x*: {:.4}, {:.4}", (traces + discs)/2.0, (traces - discs)/2.0);
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
}
