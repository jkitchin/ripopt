//! Large-scale benchmark: ripopt vs Ipopt on synthetic problems.
//!
//! Run with: cargo test --release --features ipopt-native -- --ignored large_scale_vs_ipopt --nocapture
//!
//! Both solvers receive the exact same NlpProblem struct via the same Rust trait,
//! ensuring a fair comparison.

// Include problem definitions from the main large-scale test file
#[path = "common/large_scale_problems.rs"]
mod problems;

use problems::*;
use ripopt::{NlpProblem, SolveStatus, SolverOptions};
use std::time::Instant;

// ---- Ipopt C API FFI (same as hs_suite/run_ipopt_native.rs) ----

#[cfg(feature = "ipopt-native")]
mod ipopt_ffi {
    use ripopt::NlpProblem;
    use std::ffi::CString;
    use std::os::raw::c_void;

    type IpoptProblem = *mut c_void;
    type EvalFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
    type EvalGradFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
    type EvalGCB = extern "C" fn(i32, *const f64, bool, i32, *mut f64, *mut c_void) -> bool;
    type EvalJacGCB = extern "C" fn(i32, *const f64, bool, i32, i32, *mut i32, *mut i32, *mut f64, *mut c_void) -> bool;
    type EvalHCB = extern "C" fn(i32, *const f64, bool, f64, i32, *const f64, bool, i32, *mut i32, *mut i32, *mut f64, *mut c_void) -> bool;
    type IntermediateCB = extern "C" fn(i32, i32, f64, f64, f64, f64, f64, f64, f64, f64, i32, *mut c_void) -> bool;

    extern "C" {
        fn CreateIpoptProblem(n: i32, x_l: *mut f64, x_u: *mut f64, m: i32, g_l: *mut f64, g_u: *mut f64, nele_jac: i32, nele_hess: i32, index_style: i32, eval_f: EvalFCB, eval_g: EvalGCB, eval_grad_f: EvalGradFCB, eval_jac_g: EvalJacGCB, eval_h: EvalHCB) -> IpoptProblem;
        fn FreeIpoptProblem(problem: IpoptProblem);
        fn AddIpoptNumOption(problem: IpoptProblem, keyword: *const i8, val: f64) -> bool;
        fn AddIpoptIntOption(problem: IpoptProblem, keyword: *const i8, val: i32) -> bool;
        fn SetIntermediateCallback(problem: IpoptProblem, cb: IntermediateCB) -> bool;
        fn IpoptSolve(problem: IpoptProblem, x: *mut f64, g: *mut f64, obj_val: *mut f64, mult_g: *mut f64, mult_x_l: *mut f64, mult_x_u: *mut f64, user_data: *mut c_void) -> i32;
    }

    struct Wrapper<'a> { problem: &'a dyn NlpProblem, jac_rows: Vec<i32>, jac_cols: Vec<i32>, hess_rows: Vec<i32>, hess_cols: Vec<i32>, iterations: i32 }

    extern "C" fn eval_f(n: i32, x: *const f64, _: bool, obj: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const Wrapper); w.problem.objective(std::slice::from_raw_parts(x, n as usize), true, &mut *obj); true } }
    extern "C" fn eval_grad_f(n: i32, x: *const f64, _: bool, g: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const Wrapper); w.problem.gradient(std::slice::from_raw_parts(x, n as usize), true, std::slice::from_raw_parts_mut(g, n as usize)); true } }
    extern "C" fn eval_g(n: i32, x: *const f64, _: bool, _m: i32, g: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const Wrapper); let m = w.problem.num_constraints(); if m > 0 { w.problem.constraints(std::slice::from_raw_parts(x, n as usize), true, std::slice::from_raw_parts_mut(g, m)); } true } }
    extern "C" fn eval_jac(n: i32, x: *const f64, _: bool, _m: i32, _nj: i32, ir: *mut i32, jc: *mut i32, v: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const Wrapper); if v.is_null() { let nele = w.jac_rows.len(); let rows = std::slice::from_raw_parts_mut(ir, nele); let cols = std::slice::from_raw_parts_mut(jc, nele); for k in 0..nele { rows[k] = w.jac_rows[k]; cols[k] = w.jac_cols[k]; } } else { let nele = w.jac_rows.len(); w.problem.jacobian_values(std::slice::from_raw_parts(x, n as usize), true, std::slice::from_raw_parts_mut(v, nele)); } true } }
    extern "C" fn eval_h(n: i32, x: *const f64, _: bool, of: f64, _m: i32, lam: *const f64, _: bool, _nh: i32, ir: *mut i32, jc: *mut i32, v: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const Wrapper); if v.is_null() { let nele = w.hess_rows.len(); let rows = std::slice::from_raw_parts_mut(ir, nele); let cols = std::slice::from_raw_parts_mut(jc, nele); for k in 0..nele { rows[k] = w.hess_rows[k]; cols[k] = w.hess_cols[k]; } } else { let m = w.problem.num_constraints(); let ls = if m > 0 { std::slice::from_raw_parts(lam, m) } else { &[] }; let nele = w.hess_rows.len(); w.problem.hessian_values(std::slice::from_raw_parts(x, n as usize), true, of, ls, std::slice::from_raw_parts_mut(v, nele)); } true } }
    extern "C" fn intermediate(_: i32, iter: i32, _: f64, _: f64, _: f64, _: f64, _: f64, _: f64, _: f64, _: f64, _: i32, ud: *mut c_void) -> bool { unsafe { (*(ud as *mut Wrapper)).iterations = iter; true } }

    pub struct IpoptResult { pub status: i32, pub objective: f64, pub iterations: i32 }

    pub fn solve_with_ipopt(problem: &dyn NlpProblem) -> IpoptResult {
        let n = problem.num_variables();
        let m = problem.num_constraints();
        let mut x_l = vec![0.0; n]; let mut x_u = vec![0.0; n];
        problem.bounds(&mut x_l, &mut x_u);
        let mut g_l = vec![0.0; m.max(1)]; let mut g_u = vec![0.0; m.max(1)];
        if m > 0 { problem.constraint_bounds(&mut g_l, &mut g_u); }
        let (jr, jc) = problem.jacobian_structure();
        let (hr, hc) = problem.hessian_structure();
        let nele_jac = jr.len(); let nele_hess = hr.len();
        let jac_rows: Vec<i32> = jr.iter().map(|&r| r as i32).collect();
        let jac_cols: Vec<i32> = jc.iter().map(|&c| c as i32).collect();
        let hess_rows: Vec<i32> = hr.iter().map(|&r| r as i32).collect();
        let hess_cols: Vec<i32> = hc.iter().map(|&c| c as i32).collect();
        let mut wrapper = Wrapper { problem, jac_rows, jac_cols, hess_rows, hess_cols, iterations: 0 };
        unsafe {
            let ip = CreateIpoptProblem(n as i32, x_l.as_mut_ptr(), x_u.as_mut_ptr(), m as i32, g_l.as_mut_ptr(), g_u.as_mut_ptr(), nele_jac as i32, nele_hess as i32, 0, eval_f, eval_g, eval_grad_f, eval_jac, eval_h);
            if ip.is_null() { return IpoptResult { status: -199, objective: f64::NAN, iterations: 0 }; }
            AddIpoptNumOption(ip, CString::new("tol").unwrap().as_ptr(), 1e-8);
            AddIpoptIntOption(ip, CString::new("max_iter").unwrap().as_ptr(), 3000);
            AddIpoptIntOption(ip, CString::new("print_level").unwrap().as_ptr(), 0);
            SetIntermediateCallback(ip, intermediate);
            let mut x = vec![0.0; n]; problem.initial_point(&mut x);
            let mut obj = 0.0;
            let mut g = vec![0.0; m.max(1)];
            let mut ml = vec![0.0; m.max(1)]; let mut mxl = vec![0.0; n]; let mut mxu = vec![0.0; n];
            let status = IpoptSolve(ip, x.as_mut_ptr(), g.as_mut_ptr(), &mut obj, ml.as_mut_ptr(), mxl.as_mut_ptr(), mxu.as_mut_ptr(), &mut wrapper as *mut _ as *mut c_void);
            FreeIpoptProblem(ip);
            IpoptResult { status, objective: obj, iterations: wrapper.iterations }
        }
    }
}

// ---- Benchmark macro ----

macro_rules! bench_problem {
    ($name:expr, $problem:expr) => {{
        let problem = $problem;
        let n = problem.num_variables();
        let m = problem.num_constraints();

        // ripopt
        let opts = SolverOptions { print_level: 0, tol: 1e-8, max_iter: 3000, ..SolverOptions::default() };
        let t0 = Instant::now();
        let rr = ripopt::solve(&problem, &opts);
        let rt = t0.elapsed().as_secs_f64();

        // Ipopt
        #[cfg(feature = "ipopt-native")]
        let (is, io, ii, it) = {
            let t0 = Instant::now();
            let ir = ipopt_ffi::solve_with_ipopt(&problem);
            let it = t0.elapsed().as_secs_f64();
            (ir.status, ir.objective, ir.iterations, it)
        };
        #[cfg(not(feature = "ipopt-native"))]
        let (is, io, ii, it) = (-999i32, f64::NAN, 0i32, 0.0f64);

        let rs = format!("{:?}", rr.status);
        let ist = match is { 0 => "Optimal", _ => "Failed" };
        let speedup = if it > 0.0 { it / rt } else { 0.0 };

        eprintln!(
            "BENCH: name={}, n={}, m={}, ripopt_status={}, ripopt_obj={:.6e}, ripopt_iters={}, ripopt_time={:.4}, ipopt_status={}, ipopt_obj={:.6e}, ipopt_iters={}, ipopt_time={:.4}, speedup={:.2}x",
            $name, n, m, rs, rr.objective, rr.iterations, rt, ist, io, ii, it, speedup
        );
    }};
}

#[test]
#[ignore]
fn large_scale_vs_ipopt() {
    eprintln!("\n{:>25} {:>6} {:>6} | {:>10} {:>5} {:>8} | {:>10} {:>5} {:>8} | {:>8}",
        "Problem", "n", "m", "ripopt", "iters", "time", "ipopt", "iters", "time", "speedup");
    eprintln!("{}", "-".repeat(105));

    bench_problem!("Rosenbrock 500", ChainedRosenbrock { n: 500 });
    bench_problem!("Bratu 1K", BratuProblem::new(1000));
    bench_problem!("SparseQP 1K", SparseQP { n: 500 });
    bench_problem!("OptControl 2.5K", OptimalControl::new(1249));
    bench_problem!("Poisson 2.5K", PoissonControl::new(50));
    bench_problem!("Rosenbrock 5K", ChainedRosenbrock { n: 5000 });
    bench_problem!("Bratu 10K", BratuProblem::new(10000));
    bench_problem!("OptControl 20K", OptimalControl::new(9999));
    bench_problem!("Poisson 50K", PoissonControl::new(158));
    bench_problem!("SparseQP 100K", SparseQP { n: 50000 });
}
