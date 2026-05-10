// L-BFGS scaling harness for issue #30 — measures time-per-iteration on
// scalable unconstrained problems with `hessian_approximation_lbfgs`. The
// point is to check whether dropping the dense lower-triangle Hessian
// pattern (task 4) actually pays off at large `n` or whether the dense
// fill was already cheap in practice.
//
// Usage:
//   cargo run --release --bin lbfgs_scaling -- ARWHEAD 500
//   cargo run --release --bin lbfgs_scaling -- GENROSE 2000 --max-iter 200
//   cargo run --release --bin lbfgs_scaling -- ARWHEAD 1000 --json
//
// The JSON output is a single line of key=value-style fields suitable for
// piping into a results aggregator.

use ripopt::{NlpProblem, SolverOptions, solve};

// ARWHEAD (Conn, Gould, Toint). Unconstrained, separable, sparse Hessian.
//   f(x) = sum_{i=1}^{n-1} ((x_i^2 + x_n^2)^2 - 4 x_i + 3)
// Minimum at x = (1, ..., 1, 0) with f* = 0.
struct Arwhead {
    n: usize,
}

impl NlpProblem for Arwhead {
    fn num_variables(&self) -> usize { self.n }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) {
        for i in 0..self.n { x0[i] = 1.0; }
    }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let xn = x[self.n - 1];
        let xn2 = xn * xn;
        let mut acc = 0.0;
        for i in 0..self.n - 1 {
            let xi = x[i];
            let s = xi * xi + xn2;
            acc += s * s - 4.0 * xi + 3.0;
        }
        *obj = acc;
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        let n = self.n;
        let xn = x[n - 1];
        let xn2 = xn * xn;
        let mut gn = 0.0;
        for i in 0..n - 1 {
            let xi = x[i];
            let s = xi * xi + xn2;
            grad[i] = 4.0 * xi * s - 4.0;
            gn += 4.0 * xn * s;
        }
        grad[n - 1] = gn;
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool {
        panic!("hessian_values should not be called in limited-memory mode");
    }
}

// GENROSE (Toint, generalized Rosenbrock). Unconstrained, sparse Hessian.
//   f(x) = 1 + sum_{i=2}^{n} ( 100 (x_i - x_{i-1}^2)^2 + (1 - x_i)^2 )
// Minimum at x = (1, ..., 1) with f* = 1.
struct Genrose {
    n: usize,
}

impl NlpProblem for Genrose {
    fn num_variables(&self) -> usize { self.n }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) {
        // Standard Toint start: x_i = i / (n+1)
        for i in 0..self.n { x0[i] = (i as f64 + 1.0) / (self.n as f64 + 1.0); }
    }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let mut acc = 1.0;
        for i in 1..self.n {
            let a = x[i] - x[i - 1] * x[i - 1];
            let b = 1.0 - x[i];
            acc += 100.0 * a * a + b * b;
        }
        *obj = acc;
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        for g in grad.iter_mut() { *g = 0.0; }
        for i in 1..self.n {
            let a = x[i] - x[i - 1] * x[i - 1];
            let b = 1.0 - x[i];
            grad[i]     += 200.0 * a - 2.0 * b;
            grad[i - 1] += -400.0 * x[i - 1] * a;
        }
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool {
        panic!("hessian_values should not be called in limited-memory mode");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <ARWHEAD|GENROSE> <N> [--max-iter K] [--json]", args[0]);
        std::process::exit(2);
    }
    let problem_name = args[1].to_uppercase();
    let n: usize = args[2].parse().expect("N must be a positive integer");
    let mut max_iter: usize = 500;
    let mut json = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--max-iter" => { max_iter = args[i + 1].parse().expect("--max-iter K"); i += 2; }
            "--json"     => { json = true; i += 1; }
            other        => { eprintln!("unknown flag: {}", other); std::process::exit(2); }
        }
    }

    let opts = SolverOptions {
        print_level: 0,
        hessian_approximation_lbfgs: true,
        max_iter,
        ..SolverOptions::default()
    };

    let wall_start = std::time::Instant::now();
    let result = match problem_name.as_str() {
        "ARWHEAD" => solve(&Arwhead { n }, &opts),
        "GENROSE" => solve(&Genrose { n }, &opts),
        other => { eprintln!("unknown problem: {}", other); std::process::exit(2); }
    };
    let wall = wall_start.elapsed().as_secs_f64();
    let iters = result.iterations.max(1);
    let time_per_iter = wall / iters as f64;

    if json {
        println!(
            "{{\"problem\":\"{}\",\"n\":{},\"status\":\"{:?}\",\"iters\":{},\"wall_secs\":{:.6},\"time_per_iter_secs\":{:.6},\"objective\":{:.6e}}}",
            problem_name, n, result.status, result.iterations, wall, time_per_iter, result.objective
        );
    } else {
        println!(
            "{:8} n={:>6} status={:?} iters={:>4} wall={:.3}s t/iter={:.4}s obj={:.4e}",
            problem_name, n, result.status, result.iterations, wall, time_per_iter, result.objective
        );
    }
}
