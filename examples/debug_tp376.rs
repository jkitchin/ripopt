use ripopt::{NlpProblem, SolverOptions};

struct TP376;

impl NlpProblem for TP376 {
    fn num_variables(&self) -> usize { 10 }
    fn num_constraints(&self) -> usize { 15 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0; x_u[0] = 10.0;
        x_l[1] = 0.0; x_u[1] = 0.1;
        x_l[2] = 5e-05; x_u[2] = 0.0081;
        x_l[3] = 10.0; x_u[3] = 1000.0;
        x_l[4] = 0.001; x_u[4] = 0.0017;
        x_l[5] = 0.001; x_u[5] = 0.0013;
        x_l[6] = 0.001; x_u[6] = 0.0027;
        x_l[7] = 0.001; x_u[7] = 0.002;
        x_l[8] = 0.001; x_u[8] = 1.0;
        x_l[9] = 0.001; x_u[9] = 1.0;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..14 { g_l[i] = 0.0; g_u[i] = f64::INFINITY; }
        g_l[14] = 0.0; g_u[14] = 0.0;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0; x0[1] = 0.005; x0[2] = 0.0081; x0[3] = 100.0;
        x0[4] = 0.0017; x0[5] = 0.0013; x0[6] = 0.0027; x0[7] = 0.002;
        x0[8] = 0.15; x0[9] = 0.105;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        3000.0*x[0]*(x[0] + 60.0*x[1] + 0.002).recip() + 280000.0*x[1]*(x[0] + 60.0*x[1] + 0.002).recip() - 1200.0*(x[0] + 60.0*x[1] + 0.002).recip()
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let denom2 = 0.004*x[0] + 0.24*x[1] + 3600.0*x[1].powi(2) + 120.0*x[0]*x[1] + 4.0e-6 + x[0].powi(2);
        grad[0] = -100000.0*x[1]/denom2 + 1206.0/denom2;
        grad[1] = 100000.0*x[0]/denom2 + 72560.0/denom2;
        for i in 2..10 { grad[i] = 0.0; }
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] - 0.75*x[2].recip()*x[3].recip();
        g[1] = x[0] - x[8]/(x[3]*x[4]);
        g[2] = x[0] - 10.0*x[3].recip() - x[9]/(x[3]*x[5]);
        g[3] = x[0] - 10.0*x[3].recip() - 0.19*x[3].recip()*x[6].recip();
        g[4] = x[0] - 0.125*x[3].recip()*x[7].recip();
        g[5] = 10000.0*x[1] - 0.00131*x[8]*x[3].powf(1.5)*x[4].powf(0.666);
        g[6] = 10000.0*x[1] - 0.001038*x[9]*x[3].powi(3)*x[5].powf(1.6);
        g[7] = 10000.0*x[1] - 0.000223*x[3].powf(1.5)*x[6].powf(0.666);
        g[8] = 10000.0*x[1] - 7.6e-5*x[3].powf(5.66)*x[7].powf(3.55);
        g[9] = 10000.0*x[1] - 0.000698*x[2].powf(1.2)*x[3].powi(2);
        g[10] = 10000.0*x[1] - 5.0e-5*x[2].powf(1.6)*x[3].powf(3.0);
        g[11] = 10000.0*x[1] - 6.54e-6*x[2].powf(2.42)*x[3].powf(4.17);
        g[12] = 10000.0*x[1] - 0.000257*x[2].powf(0.666)*x[3].powf(1.5);
        g[13] = -2.0*x[3]*x[2].powf(0.803) - 2.003*x[3]*x[4] - 1.885*x[3]*x[5] - 0.184*x[3]*x[7] + 30.0;
        g[14] = x[8] + x[9] - 0.255;
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14],
         vec![0, 2, 3, 0, 3, 4, 8, 0, 3, 5, 9, 0, 3, 6, 0, 3, 7, 1, 3, 4, 8, 1, 3, 5, 9, 1, 3, 6, 1, 3, 7, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 4, 5, 7, 8, 9])
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;
        vals[1] = 0.75*x[2].powi(-2)*x[3].recip();
        vals[2] = 0.75*x[2].recip()*x[3].powi(-2);
        vals[3] = 1.0;
        vals[4] = x[8]/(x[3].powi(2)*x[4]);
        vals[5] = x[8]/(x[3]*x[4].powi(2));
        vals[6] = -1.0/(x[3]*x[4]);
        vals[7] = 1.0;
        vals[8] = 10.0*x[3].powi(-2) + x[9]/(x[3].powi(2)*x[5]);
        vals[9] = x[9]/(x[3]*x[5].powi(2));
        vals[10] = -1.0/(x[3]*x[5]);
        vals[11] = 1.0;
        vals[12] = 10.0*x[3].powi(-2) + 0.19*x[3].powi(-2)*x[6].recip();
        vals[13] = 0.19*x[3].recip()*x[6].powi(-2);
        vals[14] = 1.0;
        vals[15] = 0.125*x[3].powi(-2)*x[7].recip();
        vals[16] = 0.125*x[3].recip()*x[7].powi(-2);
        vals[17] = 10000.0;
        vals[18] = -0.001965*x[8]*x[3].sqrt()*x[4].powf(0.666);
        vals[19] = -0.00087246*x[8]*x[3].powf(1.5)*x[4].powf(-0.334);
        vals[20] = -0.00131*x[3].powf(1.5)*x[4].powf(0.666);
        vals[21] = 10000.0;
        vals[22] = -0.003114*x[9]*x[3].powi(2)*x[5].powf(1.6);
        vals[23] = -0.0016608*x[9]*x[3].powi(3)*x[5].powf(0.6);
        vals[24] = -0.001038*x[3].powi(3)*x[5].powf(1.6);
        vals[25] = 10000.0;
        vals[26] = -0.0003345*x[3].sqrt()*x[6].powf(0.666);
        vals[27] = -0.000148518*x[3].powf(1.5)*x[6].powf(-0.334);
        vals[28] = 10000.0;
        vals[29] = -0.00043016*x[3].powf(4.66)*x[7].powf(3.55);
        vals[30] = -0.0002698*x[3].powf(5.66)*x[7].powf(2.55);
        vals[31] = 10000.0;
        vals[32] = -0.0008376*x[2].powf(0.2)*x[3].powi(2);
        vals[33] = -0.001396*x[3]*x[2].powf(1.2);
        vals[34] = 10000.0;
        vals[35] = -8.0e-5*x[2].powf(0.6)*x[3].powf(3.0);
        vals[36] = -0.00015*x[2].powf(1.6)*x[3].powf(2.0);
        vals[37] = 10000.0;
        vals[38] = -1.58268e-5*x[2].powf(1.42)*x[3].powf(4.17);
        vals[39] = -2.72718e-5*x[2].powf(2.42)*x[3].powf(3.17);
        vals[40] = 10000.0;
        vals[41] = -0.000171162*x[2].powf(-0.334)*x[3].powf(1.5);
        vals[42] = -0.0003855*x[2].powf(0.666)*x[3].sqrt();
        vals[43] = -1.606*x[3]*x[2].powf(-0.197);
        vals[44] = -2.0*x[2].powf(0.803) - 2.003*x[4] - 1.885*x[5] - 0.184*x[7];
        vals[45] = -2.003*x[3];
        vals[46] = -1.885*x[3];
        vals[47] = -0.184*x[3];
        vals[48] = 1.0;
        vals[49] = 1.0;
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9], vec![0, 0, 1, 2, 2, 3, 3, 4, 3, 5, 3, 6, 3, 7, 3, 4, 3, 5])
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        let d = 0.004*x[0] + 0.24*x[1] + 3600.0*x[1].powi(2) + 120.0*x[0]*x[1] + 4.0e-6 + x[0].powi(2);
        let d2 = 1.2e-5*x[0] + 0.006*x[0].powi(2) + 0.00072*x[1] + 21.6*x[1].powi(2) + 216000.0*x[1].powi(3) + 0.72*x[0]*x[1] + 10800.0*x[0]*x[1].powi(2) + 180.0*x[1]*x[0].powi(2) + 8.0e-9 + x[0].powi(3);
        let _ = d;
        vals[0] = obj_factor * (200000.0*x[1]/d2 - 2412.0/d2);
        vals[1] = obj_factor * (-100000.0*x[0]/d2 + 6000000.0*x[1]/d2 - 144920.0/d2);
        vals[2] = obj_factor * (-12000000.0*x[0]/d2 - 8707200.0/d2);
        vals[3] = lambda[0] * (-1.5*x[2].powi(-3)*x[3].recip()) + lambda[9] * (-0.00016752*x[2].powf(-0.8)*x[3].powi(2)) + lambda[10] * (-4.8e-5*x[2].powf(-0.4)*x[3].powf(3.0)) + lambda[11] * (-2.2474056e-5*x[2].powf(0.42)*x[3].powf(4.17)) + lambda[12] * (5.7168108e-5*x[2].powf(-1.334)*x[3].powf(1.5)) + lambda[13] * (0.316382*x[3]*x[2].powf(-1.197));
        vals[4] = lambda[0] * (-0.75*x[2].powi(-2)*x[3].powi(-2)) + lambda[9] * (-0.0016752*x[3]*x[2].powf(0.2)) + lambda[10] * (-0.00024*x[2].powf(0.6)*x[3].powf(2.0)) + lambda[11] * (-6.5997756e-5*x[2].powf(1.42)*x[3].powf(3.17)) + lambda[12] * (-0.000256743*x[2].powf(-0.334)*x[3].sqrt()) + lambda[13] * (-1.606*x[2].powf(-0.197));
        vals[5] = lambda[0] * (-1.5*x[2].recip()*x[3].powi(-3)) + lambda[1] * (-2.0*x[8]/(x[3].powi(3)*x[4])) + lambda[2] * (-20.0*x[3].powi(-3) - 2.0*x[9]/(x[3].powi(3)*x[5])) + lambda[3] * (-20.0*x[3].powi(-3) - 0.38*x[3].powi(-3)*x[6].recip()) + lambda[4] * (-0.25*x[3].powi(-3)*x[7].recip()) + lambda[5] * (-0.0009825*x[8]*x[3].sqrt().recip()*x[4].powf(0.666)) + lambda[6] * (-0.006228*x[3]*x[9]*x[5].powf(1.6)) + lambda[7] * (-0.00016725*x[3].sqrt().recip()*x[6].powf(0.666)) + lambda[8] * (-0.0020045456*x[3].powf(3.66)*x[7].powf(3.55)) + lambda[9] * (-0.001396*x[2].powf(1.2)) + lambda[10] * (-0.0003*x[2].powf(1.6)*x[3].powf(1.0)) + lambda[11] * (-8.6451606e-5*x[2].powf(2.42)*x[3].powf(2.17)) + lambda[12] * (-0.00019275*x[2].powf(0.666)*x[3].sqrt().recip());
        vals[6] = lambda[1] * (-x[8]/(x[3].powi(2)*x[4].powi(2))) + lambda[5] * (-0.00130869*x[8]*x[3].sqrt()*x[4].powf(-0.334)) + lambda[13] * (-2.003);
        vals[7] = lambda[1] * (-2.0*x[8]/(x[3]*x[4].powi(3))) + lambda[5] * (0.00029140164*x[8]*x[3].powf(1.5)*x[4].powf(-1.334));
        vals[8] = lambda[2] * (-x[9]/(x[3].powi(2)*x[5].powi(2))) + lambda[6] * (-0.0049824*x[9]*x[3].powi(2)*x[5].powf(0.6)) + lambda[13] * (-1.885);
        vals[9] = lambda[2] * (-2.0*x[9]/(x[3]*x[5].powi(3))) + lambda[6] * (-0.00099648*x[9]*x[3].powi(3)*x[5].powf(-0.4));
        vals[10] = lambda[3] * (-0.19*x[3].powi(-2)*x[6].powi(-2)) + lambda[7] * (-0.000222777*x[3].sqrt()*x[6].powf(-0.334));
        vals[11] = lambda[3] * (-0.38*x[3].recip()*x[6].powi(-3)) + lambda[7] * (4.9605012e-5*x[3].powf(1.5)*x[6].powf(-1.334));
        vals[12] = lambda[4] * (-0.125*x[3].powi(-2)*x[7].powi(-2)) + lambda[8] * (-0.001527068*x[3].powf(4.66)*x[7].powf(2.55)) + lambda[13] * (-0.184);
        vals[13] = lambda[4] * (-0.25*x[3].recip()*x[7].powi(-3)) + lambda[8] * (-0.00068799*x[3].powf(5.66)*x[7].powf(1.55));
        vals[14] = lambda[1] * (1.0/(x[3].powi(2)*x[4])) + lambda[5] * (-0.001965*x[3].sqrt()*x[4].powf(0.666));
        vals[15] = lambda[1] * (1.0/(x[3]*x[4].powi(2))) + lambda[5] * (-0.00087246*x[3].powf(1.5)*x[4].powf(-0.334));
        vals[16] = lambda[2] * (1.0/(x[3].powi(2)*x[5])) + lambda[6] * (-0.003114*x[3].powi(2)*x[5].powf(1.6));
        vals[17] = lambda[2] * (1.0/(x[3]*x[5].powi(2))) + lambda[6] * (-0.0016608*x[3].powi(3)*x[5].powf(0.6));
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .init();
    let problem = TP376;
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
    println!("Known optimal: -4430.0879");

    let mut g = vec![0.0; 15];
    problem.constraints(&result.x, true, &mut g);
    println!("g: {:?}", g);
}
