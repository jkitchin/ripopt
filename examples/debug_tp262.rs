use ripopt::{NlpProblem, SolverOptions};

struct TP262;
impl NlpProblem for TP262 {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 4 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 { x_l[i] = 0.0; x_u[i] = f64::INFINITY; }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..3 { g_l[i] = 0.0; g_u[i] = f64::INFINITY; }
        g_l[3] = 0.0; g_u[3] = 0.0;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0; x0[1] = 1.0; x0[2] = 1.0; x0[3] = 1.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = -0.5*x[0] - x[1] - 0.5*x[2] - x[3];
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = -0.5; grad[1] = -1.0; grad[2] = -0.5; grad[3] = -1.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = -x[0]-x[1]-x[2]-x[3]+10.0;
        g[1] = -0.2*x[0]-0.5*x[1]-x[2]-2.0*x[3]+10.0;
        g[2] = -2.0*x[0]-x[1]-0.5*x[2]-0.2*x[3]+10.0;
        g[3] = x[0]+x[1]+x[2]-2.0*x[3]-6.0;
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3], vec![0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0]=-1.0; vals[1]=-1.0; vals[2]=-1.0; vals[3]=-1.0;
        vals[4]=-0.2; vals[5]=-0.5; vals[6]=-1.0; vals[7]=-2.0;
        vals[8]=-2.0; vals[9]=-1.0; vals[10]=-0.5; vals[11]=-0.2;
        vals[12]=1.0; vals[13]=1.0; vals[14]=1.0; vals[15]=-2.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _o: f64, _l: &[f64], _v: &mut [f64]) -> bool { true }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    let options = SolverOptions { tol: 1e-8, max_iter: 200, mu_strategy_adaptive: true, print_level: 10, ..SolverOptions::default() };
    let result = ripopt::solve(&TP262, &options);
    println!("\nStatus: {:?}", result.status);
    println!("Objective: {:.10}", result.objective);
    println!("x: {:?}", result.x);
    println!("Known optimal: -10.0");
}
