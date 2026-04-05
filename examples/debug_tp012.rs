use ripopt::{NlpProblem, SolverOptions};

struct TP012;

impl NlpProblem for TP012 {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = f64::INFINITY;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0; x0[1] = 0.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.5 * x[0].powi(2) - x[0] * x[1] - 7.0 * x[0] + x[1].powi(2) - 7.0 * x[1];
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[0] - x[1] - 7.0;
        grad[1] = -x[0] + 2.0 * x[1] - 7.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = -4.0 * x[0].powi(2) - x[1].powi(2) + 25.0;
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = -8.0 * x[0];
        vals[1] = -2.0 * x[1];
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * 1.0 + lambda[0] * (-8.0);
        vals[1] = obj_factor * (-1.0);
        vals[2] = obj_factor * 2.0 + lambda[0] * (-2.0);
        true
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .init();
    let problem = TP012;
    let options = SolverOptions {
        print_level: 10,
        max_iter: 50,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    println!("\nStatus: {:?}", result.status);
    println!("Objective: {:.6}", result.objective);
    println!("x: {:?}", result.x);
    println!("y: {:?}", result.constraint_multipliers);
    println!("Iterations: {}", result.iterations);
}
