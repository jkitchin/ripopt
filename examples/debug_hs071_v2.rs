use ripopt::{NlpProblem, SolverOptions};

struct HS071;

impl NlpProblem for HS071 {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 { x_l[i] = 1.0; x_u[i] = 5.0; }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0; g_u[0] = f64::INFINITY;
        g_l[1] = 40.0; g_u[1] = 40.0;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0; x0[1] = 5.0; x0[2] = 5.0; x0[3] = 1.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        grad[1] = x[0] * x[3];
        grad[2] = x[0] * x[3] + 1.0;
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 0, 1, 1, 1, 1], vec![0, 1, 2, 3, 0, 1, 2, 3])
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = x[1] * x[2] * x[3]; vals[1] = x[0] * x[2] * x[3];
        vals[2] = x[0] * x[1] * x[3]; vals[3] = x[0] * x[1] * x[2];
        vals[4] = 2.0 * x[0]; vals[5] = 2.0 * x[1];
        vals[6] = 2.0 * x[2]; vals[7] = 2.0 * x[3];
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3], vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * 2.0 * x[3];
        vals[1] = obj_factor * x[3];
        vals[2] = 0.0;
        vals[3] = obj_factor * x[3];
        vals[4] = 0.0;
        vals[5] = 0.0;
        vals[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]);
        vals[7] = obj_factor * x[0];
        vals[8] = obj_factor * x[0];
        vals[9] = 0.0;
        vals[1] += lambda[0] * x[2] * x[3];
        vals[3] += lambda[0] * x[1] * x[3];
        vals[4] += lambda[0] * x[0] * x[3];
        vals[6] += lambda[0] * x[1] * x[2];
        vals[7] += lambda[0] * x[0] * x[2];
        vals[8] += lambda[0] * x[0] * x[1];
        vals[0] += lambda[1] * 2.0;
        vals[2] += lambda[1] * 2.0;
        vals[5] += lambda[1] * 2.0;
        vals[9] += lambda[1] * 2.0;
        true
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .init();
    let problem = HS071;
    let options = SolverOptions {
        print_level: 10,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    println!("Status: {:?}", result.status);
    println!("Objective: {:.6}", result.objective);
    println!("x: {:?}", result.x);
    println!("Iterations: {}", result.iterations);
}
