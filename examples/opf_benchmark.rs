// Benchmark comparing ripopt vs ipopt on AC Optimal Power Flow problems.
//
// Run with: cargo run --release --features ipopt-native --example opf_benchmark

use ripopt::{NlpProblem, SolverOptions};
use std::time::Instant;

#[path = "../tests/common/opf_problems.rs"]
mod problems;

// =========================================================================
// Ipopt C API FFI (same pattern as benchmark_solvers.rs)
// =========================================================================

#[cfg(feature = "ipopt-native")]
mod ipopt_ffi {
    use std::ffi::CString;
    use std::os::raw::c_void;
    use std::time::Instant;
    use ripopt::NlpProblem;

    type IpoptProblemPtr = *mut c_void;
    extern "C" {
        fn CreateIpoptProblem(n: i32, x_l: *mut f64, x_u: *mut f64, m: i32, g_l: *mut f64, g_u: *mut f64, nele_jac: i32, nele_hess: i32, index_style: i32, eval_f: EvalFCB, eval_g: EvalGCB, eval_grad_f: EvalGradFCB, eval_jac_g: EvalJacGCB, eval_h: EvalHCB) -> IpoptProblemPtr;
        fn FreeIpoptProblem(problem: IpoptProblemPtr);
        fn AddIpoptStrOption(problem: IpoptProblemPtr, keyword: *const i8, val: *const i8) -> bool;
        fn AddIpoptNumOption(problem: IpoptProblemPtr, keyword: *const i8, val: f64) -> bool;
        fn AddIpoptIntOption(problem: IpoptProblemPtr, keyword: *const i8, val: i32) -> bool;
        fn SetIntermediateCallback(problem: IpoptProblemPtr, cb: IntermediateCB) -> bool;
        fn IpoptSolve(problem: IpoptProblemPtr, x: *mut f64, g: *mut f64, obj_val: *mut f64, mult_g: *mut f64, mult_x_l: *mut f64, mult_x_u: *mut f64, user_data: *mut c_void) -> i32;
    }
    type EvalFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
    type EvalGradFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
    type EvalGCB = extern "C" fn(i32, *const f64, bool, i32, *mut f64, *mut c_void) -> bool;
    type EvalJacGCB = extern "C" fn(i32, *const f64, bool, i32, i32, *mut i32, *mut i32, *mut f64, *mut c_void) -> bool;
    type EvalHCB = extern "C" fn(i32, *const f64, bool, f64, i32, *const f64, bool, i32, *mut i32, *mut i32, *mut f64, *mut c_void) -> bool;
    type IntermediateCB = extern "C" fn(i32, i32, f64, f64, f64, f64, f64, f64, f64, f64, i32, *mut c_void) -> bool;

    struct IpoptWrapper<'a> { problem: &'a dyn NlpProblem, jac_rows: Vec<i32>, jac_cols: Vec<i32>, hess_rows: Vec<i32>, hess_cols: Vec<i32>, iterations: i32 }

    extern "C" fn eval_f_cb(n: i32, x: *const f64, _: bool, obj: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const IpoptWrapper); w.problem.objective(std::slice::from_raw_parts(x, n as usize), true, &mut *obj); true } }
    extern "C" fn eval_grad_f_cb(n: i32, x: *const f64, _: bool, g: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const IpoptWrapper); w.problem.gradient(std::slice::from_raw_parts(x, n as usize), true, std::slice::from_raw_parts_mut(g, n as usize)); true } }
    extern "C" fn eval_g_cb(n: i32, x: *const f64, _: bool, _m: i32, g: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const IpoptWrapper); let m = w.problem.num_constraints(); if m > 0 { w.problem.constraints(std::slice::from_raw_parts(x, n as usize), true, std::slice::from_raw_parts_mut(g, m)); } true } }
    extern "C" fn eval_jac_g_cb(n: i32, x: *const f64, _: bool, _m: i32, _nj: i32, ir: *mut i32, jc: *mut i32, vals: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const IpoptWrapper); if vals.is_null() { let nele = w.jac_rows.len(); std::slice::from_raw_parts_mut(ir, nele).copy_from_slice(&w.jac_rows); std::slice::from_raw_parts_mut(jc, nele).copy_from_slice(&w.jac_cols); } else { w.problem.jacobian_values(std::slice::from_raw_parts(x, n as usize), true, std::slice::from_raw_parts_mut(vals, w.jac_rows.len())); } true } }
    extern "C" fn eval_h_cb(n: i32, x: *const f64, _: bool, obj_factor: f64, _m: i32, lambda: *const f64, _: bool, _nh: i32, ir: *mut i32, jc: *mut i32, vals: *mut f64, ud: *mut c_void) -> bool { unsafe { let w = &*(ud as *const IpoptWrapper); if vals.is_null() { let nele = w.hess_rows.len(); std::slice::from_raw_parts_mut(ir, nele).copy_from_slice(&w.hess_rows); std::slice::from_raw_parts_mut(jc, nele).copy_from_slice(&w.hess_cols); } else { let xs = std::slice::from_raw_parts(x, n as usize); let m = w.problem.num_constraints(); let ls = if m > 0 { std::slice::from_raw_parts(lambda, m) } else { &[] }; w.problem.hessian_values(xs, true, obj_factor, ls, std::slice::from_raw_parts_mut(vals, w.hess_rows.len())); } true } }
    extern "C" fn intermediate_cb(_: i32, iter: i32, _: f64, _: f64, _: f64, _: f64, _: f64, _: f64, _: f64, _: f64, _: i32, ud: *mut c_void) -> bool { unsafe { (*(ud as *mut IpoptWrapper)).iterations = iter; } true }

    fn set_str(p: IpoptProblemPtr, k: &str, v: &str) { let ks = CString::new(k).unwrap(); let vs = CString::new(v).unwrap(); unsafe { AddIpoptStrOption(p, ks.as_ptr(), vs.as_ptr()); } }
    fn set_num(p: IpoptProblemPtr, k: &str, v: f64) { let ks = CString::new(k).unwrap(); unsafe { AddIpoptNumOption(p, ks.as_ptr(), v); } }
    fn set_int(p: IpoptProblemPtr, k: &str, v: i32) { let ks = CString::new(k).unwrap(); unsafe { AddIpoptIntOption(p, ks.as_ptr(), v); } }

    pub fn ipopt_status_str(s: i32) -> String { match s { 0 => "Optimal".into(), 1 => "Acceptable".into(), 2 => "Infeasible".into(), -1 => "MaxIter".into(), -2 => "RestorationFailed".into(), other => format!("Status({})", other) } }

    pub fn solve_ipopt(problem: &dyn NlpProblem, tol: f64, max_iter: i32) -> super::SolveResult {
        let n = problem.num_variables(); let m = problem.num_constraints();
        let mut x_l = vec![0.0; n]; let mut x_u = vec![0.0; n]; problem.bounds(&mut x_l, &mut x_u);
        let mut g_l = vec![0.0; m.max(1)]; let mut g_u = vec![0.0; m.max(1)]; if m > 0 { problem.constraint_bounds(&mut g_l, &mut g_u); }
        let (jr, jc) = problem.jacobian_structure(); let (hr, hc) = problem.hessian_structure();
        let mut wrapper = IpoptWrapper { problem, jac_rows: jr.iter().map(|&r| r as i32).collect(), jac_cols: jc.iter().map(|&c| c as i32).collect(), hess_rows: hr.iter().map(|&r| r as i32).collect(), hess_cols: hc.iter().map(|&c| c as i32).collect(), iterations: 0 };
        unsafe {
            let ip = CreateIpoptProblem(n as i32, x_l.as_mut_ptr(), x_u.as_mut_ptr(), m as i32, g_l.as_mut_ptr(), g_u.as_mut_ptr(), wrapper.jac_rows.len() as i32, wrapper.hess_rows.len() as i32, 0, eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb, eval_h_cb);
            set_str(ip, "sb", "yes"); set_str(ip, "mu_strategy", "adaptive"); set_num(ip, "tol", tol); set_int(ip, "max_iter", max_iter); set_int(ip, "print_level", 0);
            SetIntermediateCallback(ip, intermediate_cb);
            let mut x = vec![0.0; n]; problem.initial_point(&mut x);
            let mut g = vec![0.0; m.max(1)]; let mut obj = 0.0; let mut mg = vec![0.0; m.max(1)]; let mut ml = vec![0.0; n]; let mut mu = vec![0.0; n];
            let ud = &mut wrapper as *mut IpoptWrapper as *mut c_void;
            let t0 = Instant::now();
            let status = IpoptSolve(ip, x.as_mut_ptr(), g.as_mut_ptr(), &mut obj, mg.as_mut_ptr(), ml.as_mut_ptr(), mu.as_mut_ptr(), ud);
            let elapsed = t0.elapsed().as_secs_f64(); let iters = wrapper.iterations; FreeIpoptProblem(ip);
            super::SolveResult { status: ipopt_status_str(status), objective: obj, iterations: iters, time_s: elapsed }
        }
    }
}

// =========================================================================
// Result type and ripopt wrapper
// =========================================================================

struct SolveResult { status: String, objective: f64, iterations: i32, time_s: f64 }

#[derive(serde::Serialize)]
struct BenchEntry {
    solver: String,
    name: String,
    n: usize,
    m: usize,
    status: String,
    objective: f64,
    iterations: i32,
    solve_time: f64,
}

fn solve_ripopt<P: NlpProblem>(problem: &P, tol: f64, max_iter: usize) -> SolveResult {
    let options = SolverOptions { tol, max_iter, max_wall_time: 60.0, print_level: 0, ..SolverOptions::default() };
    let t0 = Instant::now();
    let result = ripopt::solve(problem, &options);
    let elapsed = t0.elapsed().as_secs_f64();
    SolveResult { status: format!("{:?}", result.status), objective: result.objective, iterations: result.iterations as i32, time_s: elapsed }
}

// =========================================================================
// Main
// =========================================================================

fn main() {
    let tol = 1e-6;
    let max_iter: usize = 3000;
    let have_ipopt = cfg!(feature = "ipopt-native");

    println!();
    println!("AC Optimal Power Flow Benchmark: ripopt vs ipopt");
    println!("================================================");
    println!();

    let mut header = format!(
        "{:<20} {:>4} {:>4} {:>4} | {:>12} {:>5} {:>8}",
        "Problem", "n", "m", "nnz", "ripopt obj", "iter", "time(s)"
    );
    if have_ipopt {
        header += &format!(" | {:>12} {:>5} {:>8}", "ipopt obj", "iter", "time(s)");
    }
    println!("{}", header);
    let width = header.len();
    println!("{}", "-".repeat(width));

    let mut results: Vec<BenchEntry> = Vec::new();

    macro_rules! bench {
        ($name:expr, $problem:expr, $known_opt:expr) => {{
            let p = $problem;
            let n = p.num_variables();
            let m = p.num_constraints();
            let (jr, _) = p.jacobian_structure();
            let nnz = jr.len();
            let rp = solve_ripopt(&p, tol, max_iter);
            #[allow(unused_mut)]
            let mut line = format!(
                "{:<20} {:>4} {:>4} {:>4} | {:>12.2} {:>5} {:>8.4}",
                $name, n, m, nnz, rp.objective, rp.iterations, rp.time_s
            );
            #[allow(unused_mut)]
            let mut notes = Vec::new();
            if rp.status != "Optimal" {
                notes.push(format!("ripopt={}", rp.status));
            }
            results.push(BenchEntry {
                solver: "ripopt".into(), name: $name.into(), n, m,
                status: rp.status.clone(), objective: rp.objective,
                iterations: rp.iterations, solve_time: rp.time_s,
            });
            let rp_gap = ((rp.objective - $known_opt) / $known_opt * 100.0).abs();

            #[cfg(feature = "ipopt-native")]
            {
                let ip = ipopt_ffi::solve_ipopt(&p, tol, max_iter as i32);
                line += &format!(" | {:>12.2} {:>5} {:>8.4}", ip.objective, ip.iterations, ip.time_s);
                if ip.status != "Optimal" { notes.push(format!("ipopt={}", ip.status)); }
                results.push(BenchEntry {
                    solver: "ipopt".into(), name: $name.into(), n, m,
                    status: ip.status.clone(), objective: ip.objective,
                    iterations: ip.iterations, solve_time: ip.time_s,
                });
            }

            println!("{}", line);
            if !notes.is_empty() {
                println!("  status: {}", notes.join(", "));
            }
            if rp_gap > 1.0 {
                println!("  gap from known optimal: {:.2}%", rp_gap);
            }
        }};
    }

    use problems::*;

    bench!("case3_lmbd", case3_lmbd(), 5812.64);
    bench!("case5_pjm", case5_pjm(), 17551.89);
    bench!("case14_ieee", case14_ieee(), 2178.08);
    bench!("case30_ieee", case30_ieee(), 8081.52);

    println!("{}", "-".repeat(width));

    // Write JSON results
    let results_path = std::env::var("RESULTS_FILE")
        .unwrap_or_else(|_| "opf_results.json".to_string());
    let json = serde_json::to_string_pretty(&results).unwrap();
    std::fs::write(&results_path, json).unwrap();
    eprintln!("Results written to {}", results_path);
}
