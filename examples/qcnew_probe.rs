// Side-by-side iteration trace of ripopt vs Ipopt on a CUTEst problem.
// Used to localize where ripopt and Ipopt diverge in early iterations
// (issue #31, QCNEW).
//
// Usage:
//   cargo run --release --features cutest,ipopt-native --example qcnew_probe -- QCNEW

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_void;

use ripopt::NlpProblem;

#[path = "../benchmarks/cutest/cutest_problem.rs"]
mod cutest_problem;
#[path = "../benchmarks/cutest/cutest_ffi.rs"]
mod cutest_ffi;

use cutest_problem::CutestProblem;

// ---- Ipopt FFI (mirrored from benchmarks/cutest/run_cutest.rs) ----

type IpoptProblem = *mut c_void;

#[allow(non_camel_case_types)]
type EvalFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGradFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGCB = extern "C" fn(i32, *const f64, bool, i32, *mut f64, *mut c_void) -> bool;
type EvalJacGCB =
    extern "C" fn(i32, *const f64, bool, i32, i32, *mut i32, *mut i32, *mut f64, *mut c_void) -> bool;
type EvalHCB = extern "C" fn(
    i32, *const f64, bool, f64, i32, *const f64, bool, i32, *mut i32, *mut i32, *mut f64, *mut c_void,
) -> bool;
type IntermediateCB =
    extern "C" fn(i32, i32, f64, f64, f64, f64, f64, f64, f64, f64, i32, *mut c_void) -> bool;

extern "C" {
    fn CreateIpoptProblem(
        n: i32, x_l: *mut f64, x_u: *mut f64,
        m: i32, g_l: *mut f64, g_u: *mut f64,
        nele_jac: i32, nele_hess: i32, index_style: i32,
        eval_f: EvalFCB, eval_g: EvalGCB, eval_grad_f: EvalGradFCB,
        eval_jac_g: EvalJacGCB, eval_h: EvalHCB,
    ) -> IpoptProblem;
    fn FreeIpoptProblem(problem: IpoptProblem);
    fn AddIpoptStrOption(problem: IpoptProblem, keyword: *const i8, val: *const i8) -> bool;
    fn AddIpoptNumOption(problem: IpoptProblem, keyword: *const i8, val: f64) -> bool;
    fn AddIpoptIntOption(problem: IpoptProblem, keyword: *const i8, val: i32) -> bool;
    fn SetIntermediateCallback(problem: IpoptProblem, cb: IntermediateCB) -> bool;
    fn IpoptSolve(
        problem: IpoptProblem,
        x: *mut f64, g: *mut f64, obj_val: *mut f64,
        mult_g: *mut f64, mult_x_l: *mut f64, mult_x_u: *mut f64,
        user_data: *mut c_void,
    ) -> i32;
    fn GetIpoptCurrentIterate(
        problem: IpoptProblem, scaled: bool, n: i32,
        x: *mut f64, z_L: *mut f64, z_U: *mut f64,
        m: i32, g: *mut f64, lambda: *mut f64,
    ) -> bool;
}

struct Wrap<'a> {
    problem: &'a dyn NlpProblem,
    jac_rows: Vec<i32>,
    jac_cols: Vec<i32>,
    hess_rows: Vec<i32>,
    hess_cols: Vec<i32>,
}

extern "C" fn eval_f(n: i32, x: *const f64, new_x: bool, obj: *mut f64, ud: *mut c_void) -> bool {
    unsafe {
        let w = &*(ud as *const Wrap);
        w.problem.objective(std::slice::from_raw_parts(x, n as usize), new_x, &mut *obj)
    }
}
extern "C" fn eval_grad_f(n: i32, x: *const f64, new_x: bool, g: *mut f64, ud: *mut c_void) -> bool {
    unsafe {
        let w = &*(ud as *const Wrap);
        w.problem.gradient(
            std::slice::from_raw_parts(x, n as usize), new_x,
            std::slice::from_raw_parts_mut(g, n as usize),
        )
    }
}
extern "C" fn eval_g(n: i32, x: *const f64, new_x: bool, _m: i32, g: *mut f64, ud: *mut c_void) -> bool {
    unsafe {
        let w = &*(ud as *const Wrap);
        let m = w.problem.num_constraints();
        if m == 0 { return true; }
        w.problem.constraints(
            std::slice::from_raw_parts(x, n as usize), new_x,
            std::slice::from_raw_parts_mut(g, m),
        )
    }
}
extern "C" fn eval_jac_g(
    n: i32, x: *const f64, new_x: bool, _m: i32, _nele: i32,
    irow: *mut i32, jcol: *mut i32, vals: *mut f64, ud: *mut c_void,
) -> bool {
    unsafe {
        let w = &*(ud as *const Wrap);
        let nele = w.jac_rows.len();
        if vals.is_null() {
            let r = std::slice::from_raw_parts_mut(irow, nele);
            let c = std::slice::from_raw_parts_mut(jcol, nele);
            r.copy_from_slice(&w.jac_rows);
            c.copy_from_slice(&w.jac_cols);
            true
        } else {
            w.problem.jacobian_values(
                std::slice::from_raw_parts(x, n as usize), new_x,
                std::slice::from_raw_parts_mut(vals, nele),
            )
        }
    }
}
extern "C" fn eval_h(
    n: i32, x: *const f64, new_x: bool, obj_factor: f64, _m: i32, lambda: *const f64,
    _new_l: bool, _nele: i32, irow: *mut i32, jcol: *mut i32, vals: *mut f64, ud: *mut c_void,
) -> bool {
    unsafe {
        let w = &*(ud as *const Wrap);
        let nele = w.hess_rows.len();
        if vals.is_null() {
            let r = std::slice::from_raw_parts_mut(irow, nele);
            let c = std::slice::from_raw_parts_mut(jcol, nele);
            r.copy_from_slice(&w.hess_rows);
            c.copy_from_slice(&w.hess_cols);
            true
        } else {
            let m = w.problem.num_constraints();
            let lam = if m > 0 { std::slice::from_raw_parts(lambda, m) } else { &[][..] };
            w.problem.hessian_values(
                std::slice::from_raw_parts(x, n as usize), new_x, obj_factor, lam,
                std::slice::from_raw_parts_mut(vals, nele),
            )
        }
    }
}

#[derive(Clone, Debug)]
struct IpoptIter {
    iter: i32, obj: f64, inf_pr: f64, inf_du: f64, mu: f64,
    d_norm: f64, reg: f64, alpha_du: f64, alpha_pr: f64, ls: i32,
    x: Vec<f64>, z_l: Vec<f64>, z_u: Vec<f64>, lam: Vec<f64>,
}

thread_local! {
    static IPOPT_TRACE: RefCell<Vec<IpoptIter>> = const { RefCell::new(Vec::new()) };
    static IPOPT_PROBLEM: RefCell<usize> = const { RefCell::new(0) };
    static IPOPT_DIMS: RefCell<(i32, i32)> = const { RefCell::new((0, 0)) };
}

extern "C" fn intermediate(
    _alg: i32, iter: i32, obj: f64, inf_pr: f64, inf_du: f64, mu: f64,
    d_norm: f64, reg: f64, alpha_du: f64, alpha_pr: f64, ls: i32, _ud: *mut c_void,
) -> bool {
    let (n, m) = IPOPT_DIMS.with(|d| *d.borrow());
    let p = IPOPT_PROBLEM.with(|p| *p.borrow()) as IpoptProblem;
    let mut x = vec![0.0; n as usize];
    let mut zl = vec![0.0; n as usize];
    let mut zu = vec![0.0; n as usize];
    let mut lam = vec![0.0; m.max(1) as usize];
    // Use SCALED iterates so that we can compare to ripopt's internal values.
    unsafe {
        GetIpoptCurrentIterate(
            p, true, n,
            x.as_mut_ptr(), zl.as_mut_ptr(), zu.as_mut_ptr(),
            m, std::ptr::null_mut(),
            if m > 0 { lam.as_mut_ptr() } else { std::ptr::null_mut() },
        );
    }
    if m == 0 { lam.clear(); }
    IPOPT_TRACE.with(|t| t.borrow_mut().push(IpoptIter {
        iter, obj, inf_pr, inf_du, mu, d_norm, reg, alpha_du, alpha_pr, ls,
        x, z_l: zl, z_u: zu, lam,
    }));
    true
}

fn set_str(p: IpoptProblem, k: &str, v: &str) {
    let kk = CString::new(k).unwrap();
    let vv = CString::new(v).unwrap();
    unsafe { AddIpoptStrOption(p, kk.as_ptr(), vv.as_ptr()); }
}
fn set_num(p: IpoptProblem, k: &str, v: f64) {
    let kk = CString::new(k).unwrap();
    unsafe { AddIpoptNumOption(p, kk.as_ptr(), v); }
}
fn set_int(p: IpoptProblem, k: &str, v: i32) {
    let kk = CString::new(k).unwrap();
    unsafe { AddIpoptIntOption(p, kk.as_ptr(), v); }
}

fn run_ipopt_with(problem: &dyn NlpProblem, mu_strategy: &str) -> Vec<f64> {
    let n = problem.num_variables();
    let m = problem.num_constraints();
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    problem.bounds(&mut x_l, &mut x_u);
    let mut g_l = vec![0.0; m.max(1)];
    let mut g_u = vec![0.0; m.max(1)];
    if m > 0 { problem.constraint_bounds(&mut g_l, &mut g_u); }

    let (jr, jc) = problem.jacobian_structure();
    let (hr, hc) = problem.hessian_structure();
    let mut wrap = Wrap {
        problem,
        jac_rows: jr.iter().map(|&v| v as i32).collect(),
        jac_cols: jc.iter().map(|&v| v as i32).collect(),
        hess_rows: hr.iter().map(|&v| v as i32).collect(),
        hess_cols: hc.iter().map(|&v| v as i32).collect(),
    };

    unsafe {
        let p = CreateIpoptProblem(
            n as i32, x_l.as_mut_ptr(), x_u.as_mut_ptr(),
            m as i32, g_l.as_mut_ptr(), g_u.as_mut_ptr(),
            wrap.jac_rows.len() as i32, wrap.hess_rows.len() as i32, 0,
            eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h,
        );
        set_str(p, "sb", "yes");
        set_str(p, "mu_strategy", mu_strategy);
        set_num(p, "tol", 1e-8);
        set_int(p, "max_iter", 3000);
        // print_level 5 prints the iteration summary; we also capture in callback.
        set_int(p, "print_level", if std::env::var("IPOPT_VERBOSE").is_ok() { 12 } else { 5 });
        // Force Ipopt to use the same default we want to verify.
        if let Ok(v) = std::env::var("IPOPT_BMIV") {
            if let Ok(f) = v.parse::<f64>() {
                set_num(p, "bound_mult_init_val", f);
            }
        }
        if std::env::var("IPOPT_NO_SCALING").is_ok() {
            set_str(p, "nlp_scaling_method", "none");
        }
        SetIntermediateCallback(p, intermediate);
        IPOPT_PROBLEM.with(|pp| *pp.borrow_mut() = p as usize);
        IPOPT_DIMS.with(|d| *d.borrow_mut() = (n as i32, m as i32));

        let mut x = vec![0.0; n];
        problem.initial_point(&mut x);
        let mut g = vec![0.0; m.max(1)];
        let mut obj = 0.0;
        let mut mg = vec![0.0; m.max(1)];
        let mut mxl = vec![0.0; n];
        let mut mxu = vec![0.0; n];

        IpoptSolve(
            p, x.as_mut_ptr(), g.as_mut_ptr(), &mut obj,
            mg.as_mut_ptr(), mxl.as_mut_ptr(), mxu.as_mut_ptr(),
            &mut wrap as *mut Wrap as *mut c_void,
        );
        FreeIpoptProblem(p);
        x
    }
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "QCNEW".to_string());
    let suite_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks").join("cutest");
    let problems_dir = suite_dir.join("problems");
    let lib = problems_dir.join(format!("lib{}.{}", name, std::env::consts::DLL_EXTENSION));
    let outsdif = problems_dir.join(format!("{}_OUTSDIF.d", name));
    let problem = CutestProblem::load(
        &name, lib.to_str().unwrap(), outsdif.to_str().unwrap(),
    ).expect("load");

    println!("=== {} (n={}, m={}) ===", name, problem.num_variables(), problem.num_constraints());

    for strat in ["adaptive", "monotone"] {
        IPOPT_TRACE.with(|t| t.borrow_mut().clear());
        println!("\n--- Ipopt ({}, print_level=5) ---", strat);
        let x_ipopt = run_ipopt_with(&problem, strat);
        let trace = IPOPT_TRACE.with(|t| t.borrow().clone());
        println!("\n[ipopt-{}-trace] iter | obj | inf_pr | inf_du | mu | d_norm | reg | a_du | a_pr | ls", strat);
        for r in &trace {
            println!(
                "  {:>3} {:+.6e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {}",
                r.iter, r.obj, r.inf_pr, r.inf_du, r.mu, r.d_norm, r.reg, r.alpha_du, r.alpha_pr, r.ls,
            );
        }
        println!("[ipopt-{}-final-x] {:?}", strat, x_ipopt);
    }
    let trace = IPOPT_TRACE.with(|t| t.borrow().clone());

    // Dump iter 0 and iter 1 x and multipliers explicitly for direct compare with ripopt.
    for k in [0usize, 1, 2, 3] {
        if let Some(r) = trace.get(k) {
            println!(
                "[ipopt-iter{}] x={:?}\n  z_L={:?}\n  z_U={:?}\n  lam={:?}",
                r.iter, r.x, r.z_l, r.z_u, r.lam,
            );
        }
    }

    use ripopt::{solve, SolverOptions};

    // First: run ripopt with max_iter=1, 2, 3 to capture early-iter x trajectories
    // for direct numerical comparison against Ipopt's iter 1, 2, 3 above.
    for k in [1u32, 2, 3] {
        let opts = SolverOptions {
            tol: 1e-30, max_iter: k as usize, print_level: 0,
            mu_strategy_adaptive: true, max_wall_time: 30.0,
            ..Default::default()
        };
        let r = solve(&problem, &opts);
        println!(
            "[ripopt-iter{}] status={:?} obj={:.10e} pr={:.2e} du={:.2e} compl={:.2e} mu={:.2e}\n  x={:?}",
            k, r.status, r.objective,
            r.diagnostics.final_primal_inf, r.diagnostics.final_dual_inf,
            r.diagnostics.final_compl, r.diagnostics.final_mu, r.x,
        );
    }

    // Also run with print_level=6 max_iter=2 to get the iter-0 internal probe
    // (that prints |y|, |z|, |dz|, ftb_du etc) for direct comparison.
    println!("\n--- ripopt internal iter-0 probe (print_level=6) ---");
    let opts_pl = SolverOptions {
        tol: 1e-30, max_iter: 2, print_level: 6,
        mu_strategy_adaptive: true, max_wall_time: 30.0,
        ..Default::default()
    };
    let _ = solve(&problem, &opts_pl);

    let configs: &[(&str, fn(&mut SolverOptions))] = &[
        ("default-adaptive", |o| {
            o.mu_strategy_adaptive = true;
        }),
        ("noLSy", |o| {
            o.mu_strategy_adaptive = true;
            o.least_squares_mult_init = false;
        }),
        ("noLSy-monotone", |o| {
            o.mu_strategy_adaptive = false;
            o.least_squares_mult_init = false;
        }),
        ("z_init=0.01", |o| {
            o.mu_strategy_adaptive = true;
            o.bound_mult_init_val = 0.01;
        }),
        ("z_mu_based", |o| {
            o.mu_strategy_adaptive = true;
            o.bound_mult_init_method = ripopt::BoundMultInitMethod::MuBased;
        }),
        ("no-nlp-scaling", |o| {
            o.mu_strategy_adaptive = true;
            o.nlp_scaling_method = ripopt::NlpScalingMethod::None;
        }),
        ("monotone", |o| {
            o.mu_strategy_adaptive = false;
        }),
        ("monotone-no-nlp-scaling", |o| {
            o.mu_strategy_adaptive = false;
            o.nlp_scaling_method = ripopt::NlpScalingMethod::None;
        }),
        ("no-linear-detect", |o| {
            o.mu_strategy_adaptive = true;
            o.detect_linear_constraints = false;
        }),
        ("monotone-no-linear-detect", |o| {
            o.mu_strategy_adaptive = false;
            o.detect_linear_constraints = false;
        }),
    ];

    for (label, tweak) in configs {
        println!("\n--- ripopt ({}) ---", label);
        let mut opts = SolverOptions {
            tol: 1e-8,
            max_iter: 30,
            print_level: 0,
            max_wall_time: 30.0,
            ..Default::default()
        };
        tweak(&mut opts);
        let r = solve(&problem, &opts);
        println!("[ripopt-{}] status={:?} obj={:.10e} iter={} pr={:.2e} du={:.2e} compl={:.2e} mu={:.2e}",
            label, r.status, r.objective, r.iterations,
            r.diagnostics.final_primal_inf, r.diagnostics.final_dual_inf,
            r.diagnostics.final_compl, r.diagnostics.final_mu);
        println!("[ripopt-{}-x] {:?}", label, r.x);
    }
}
