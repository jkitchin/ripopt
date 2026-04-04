use ripopt::{NlpProblem, SolverOptions};

pub struct HsTp045;

impl NlpProblem for HsTp045 {
    fn num_variables(&self) -> usize {
        5
    }

    fn num_constraints(&self) -> usize {
        0
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0;
        x_u[0] = 1.0;
        x_l[1] = 0.0;
        x_u[1] = 2.0;
        x_l[2] = 0.0;
        x_u[2] = 3.0;
        x_l[3] = 0.0;
        x_u[3] = 4.0;
        x_l[4] = 0.0;
        x_u[4] = 5.0;
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 2.0;
        x0[1] = 2.0;
        x0[2] = 2.0;
        x0[3] = 2.0;
        x0[4] = 2.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -0.00833333333333333*x[0]*x[1]*x[2]*x[3]*x[4] + 2.0
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = -0.00833333333333333*x[1]*x[2]*x[3]*x[4];
        grad[1] = -0.00833333333333333*x[0]*x[2]*x[3]*x[4];
        grad[2] = -0.00833333333333333*x[0]*x[1]*x[3]*x[4];
        grad[3] = -0.00833333333333333*x[0]*x[1]*x[2]*x[4];
        grad[4] = -0.00833333333333333*x[0]*x[1]*x[2]*x[3];
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4], vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (-0.00833333333333333*x[2]*x[3]*x[4]);
        vals[1] = obj_factor * (-0.00833333333333333*x[1]*x[3]*x[4]);
        vals[2] = obj_factor * (-0.00833333333333333*x[0]*x[3]*x[4]);
        vals[3] = obj_factor * (-0.00833333333333333*x[1]*x[2]*x[4]);
        vals[4] = obj_factor * (-0.00833333333333333*x[0]*x[2]*x[4]);
        vals[5] = obj_factor * (-0.00833333333333333*x[0]*x[1]*x[4]);
        vals[6] = obj_factor * (-0.00833333333333333*x[1]*x[2]*x[3]);
        vals[7] = obj_factor * (-0.00833333333333333*x[0]*x[2]*x[3]);
        vals[8] = obj_factor * (-0.00833333333333333*x[0]*x[1]*x[3]);
        vals[9] = obj_factor * (-0.00833333333333333*x[0]*x[1]*x[2]);
    }
}

fn main() {
    env_logger::init();

    let problem = HsTp045;

    let mut options = SolverOptions::default();
    options.print_level = 5;
    options.max_iter = 3000;

    let result = ripopt::solve(&problem, &options);

    println!("\n=== Final Results ===");
    println!("Status: {:?}", result.status);
    println!("Objective: {:.10}", result.objective);
    println!("Solution: {:?}", result.x);
    println!("Iterations: {}", result.iterations);

    println!("\n=== Expected Optimal ===");
    println!("Objective: 1.0");
    println!("Solution: [1.0, 2.0, 3.0, 4.0, 5.0]");
    println!("Product: {}", 1.0 * 2.0 * 3.0 * 4.0 * 5.0);
    println!("Obj formula: -0.00833333333333333 * 120 + 2.0 = {}", -0.00833333333333333 * 120.0 + 2.0);
}
