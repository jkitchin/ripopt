use ripopt::{NlpProblem, SolverOptions};

struct TP023;

impl NlpProblem for TP023 {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 5 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = -50.0; x_u[0] = 50.0;
        x_l[1] = -50.0; x_u[1] = 50.0;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..5 { g_l[i] = 0.0; g_u[i] = f64::INFINITY; }
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 3.0; x0[1] = 1.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0].powi(2) + x[1].powi(2);
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1] - 1.0;
        g[1] = x[0].powi(2) + x[1].powi(2) - 1.0;
        g[2] = 9.0 * x[0].powi(2) + x[1].powi(2) - 9.0;
        g[3] = x[0].powi(2) - x[1];
        g[4] = x[1].powi(2) - x[0];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
         vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;
        vals[1] = 1.0;
        vals[2] = 2.0 * x[0];
        vals[3] = 2.0 * x[1];
        vals[4] = 18.0 * x[0];
        vals[5] = 2.0 * x[1];
        vals[6] = 2.0 * x[0];
        vals[7] = -1.0;
        vals[8] = -1.0;
        vals[9] = 2.0 * x[1];
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * 2.0 + lambda[1] * 2.0 + lambda[2] * 18.0 + lambda[3] * 2.0;
        vals[1] = obj_factor * 2.0 + lambda[1] * 2.0 + lambda[2] * 2.0 + lambda[4] * 2.0;
        true
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .init();
    let problem = TP023;
    let options = SolverOptions {
        print_level: 10,
        max_iter: 20,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    println!("\nStatus: {:?}", result.status);
    println!("Objective: {:.6}", result.objective);
    println!("x: {:?}", result.x);
    println!("y: {:?}", result.constraint_multipliers);
    println!("Iterations: {}", result.iterations);

    // Check constraints
    let mut g = vec![0.0; 5];
    problem.constraints(&result.x, true, &mut g);
    println!("g: {:?}", g);
    println!("g_l: [0, 0, 0, 0, 0]");
}
