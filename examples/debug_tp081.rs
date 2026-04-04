use ripopt::{NlpProblem, SolverOptions};

struct TP081;

impl NlpProblem for TP081 {
    fn num_variables(&self) -> usize { 5 }
    fn num_constraints(&self) -> usize { 3 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = -2.3; x_u[0] = 2.3;
        x_l[1] = -2.3; x_u[1] = 2.3;
        x_l[2] = -3.2; x_u[2] = 3.2;
        x_l[3] = -3.2; x_u[3] = 3.2;
        x_l[4] = -3.2; x_u[4] = 3.2;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..3 { g_l[i] = 0.0; g_u[i] = 0.0; }
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -2.0; x0[1] = 2.0; x0[2] = 2.0; x0[3] = -1.0; x0[4] = -1.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -1.0*x[0].powi(3) - 0.5*x[0].powi(6) - 1.0*x[1].powi(3) - 0.5*x[1].powi(6) - 1.0*x[0].powi(3)*x[1].powi(3) + (x[0]*x[1]*x[2]*x[3]*x[4]).exp() - 0.5
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = -3.0*x[0].powi(5) - 3.0*x[0].powi(2)*x[1].powi(3) - 3.0*x[0].powi(2) + x[1]*x[2]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp();
        grad[1] = -3.0*x[0].powi(3)*x[1].powi(2) + x[0]*x[2]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() - 3.0*x[1].powi(5) - 3.0*x[1].powi(2);
        grad[2] = x[0]*x[1]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp();
        grad[3] = x[0]*x[1]*x[2]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp();
        grad[4] = x[0]*x[1]*x[2]*x[3]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp();
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -10.0 + x[0].powi(2) + x[1].powi(2) + x[2].powi(2) + x[3].powi(2) + x[4].powi(2);
        g[1] = x[1]*x[2] - 5.0*x[3]*x[4];
        g[2] = 1.0 + x[0].powi(3) + x[1].powi(3);
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2], vec![0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 1])
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 2.0*x[0]; vals[1] = 2.0*x[1]; vals[2] = 2.0*x[2]; vals[3] = 2.0*x[3]; vals[4] = 2.0*x[4];
        vals[5] = x[2]; vals[6] = x[1]; vals[7] = -5.0*x[4]; vals[8] = -5.0*x[3];
        vals[9] = 3.0*x[0].powi(2); vals[10] = 3.0*x[1].powi(2);
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4], vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4])
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (-15.0*x[0].powi(4) - 6.0*x[0]*x[1].powi(3) - 6.0*x[0] + x[1].powi(2)*x[2].powi(2)*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[0] * 2.0 + lambda[2] * 6.0*x[0];
        vals[1] = obj_factor * (-9.0*x[0].powi(2)*x[1].powi(2) + x[0]*x[1]*x[2].powi(2)*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[2]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[2] = obj_factor * (-6.0*x[1]*x[0].powi(3) + x[0].powi(2)*x[2].powi(2)*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() - 15.0*x[1].powi(4) - 6.0*x[1]) + lambda[0] * 2.0 + lambda[2] * 6.0*x[1];
        vals[3] = obj_factor * (x[0]*x[1].powi(2)*x[2]*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[1]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[4] = obj_factor * (x[0].powi(2)*x[1]*x[2]*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[1] * 1.0;
        vals[5] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[0] * 2.0;
        vals[6] = obj_factor * (x[0]*x[1].powi(2)*x[2].powi(2)*x[3]*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[1]*x[2]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[7] = obj_factor * (x[0].powi(2)*x[1]*x[2].powi(2)*x[3]*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[2]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[8] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2]*x[3]*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[1]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[9] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[0] * 2.0;
        vals[10] = obj_factor * (x[0]*x[1].powi(2)*x[2].powi(2)*x[3].powi(2)*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[1]*x[2]*x[3]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[11] = obj_factor * (x[0].powi(2)*x[1]*x[2].powi(2)*x[3].powi(2)*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[2]*x[3]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[12] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2]*x[3].powi(2)*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[1]*x[3]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[13] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2].powi(2)*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[1]*x[2]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[1] * (-5.0);
        vals[14] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2].powi(2)*x[3].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[0] * 2.0;
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .init();
    let problem = TP081;
    let options = SolverOptions {
        print_level: 10,
        max_iter: 200,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    println!("\nStatus: {:?}", result.status);
    println!("Objective: {:.10}", result.objective);
    println!("x: {:?}", result.x);
    println!("y: {:?}", result.constraint_multipliers);
    println!("Iterations: {}", result.iterations);
    println!("Known optimal: 0.0539498477749");

    let mut g = vec![0.0; 3];
    problem.constraints(&result.x, true, &mut g);
    println!("g: {:?}", g);
}
