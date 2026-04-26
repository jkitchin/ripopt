// Benchmark binary: solve all HS problems with native Ipopt (C API via FFI)
// and output JSON results for comparison with ripopt.

#[path = "generated/hs_problems.rs"]
mod hs_problems;

use hs_problems::{HS_PROBLEMS, HsSolveResult};
use ripopt::NlpProblem;
use std::ffi::CString;
use std::os::raw::c_void;
use std::time::Instant;

// ---- Ipopt C API FFI declarations ----

type IpoptProblem = *mut c_void;

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
}

// Callback function types
type EvalFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGradFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGCB = extern "C" fn(i32, *const f64, bool, i32, *mut f64, *mut c_void) -> bool;
type EvalJacGCB = extern "C" fn(i32, *const f64, bool, i32, i32, *mut i32, *mut i32, *mut f64, *mut c_void) -> bool;
type EvalHCB = extern "C" fn(i32, *const f64, bool, f64, i32, *const f64, bool, i32, *mut i32, *mut i32, *mut f64, *mut c_void) -> bool;
type IntermediateCB = extern "C" fn(i32, i32, f64, f64, f64, f64, f64, f64, f64, f64, i32, *mut c_void) -> bool;

// ---- Problem wrapper for passing through void* ----

struct ProblemWrapper<'a> {
    problem: &'a dyn NlpProblem,
    jac_rows: Vec<i32>,
    jac_cols: Vec<i32>,
    hess_rows: Vec<i32>,
    hess_cols: Vec<i32>,
    iterations: i32,
}

// ---- Callback implementations ----

extern "C" fn eval_f_cb(
    n: i32, x: *const f64, new_x: bool,
    obj_value: *mut f64, user_data: *mut c_void,
) -> bool {
    unsafe {
        let wrapper = &*(user_data as *const ProblemWrapper);
        let x_slice = std::slice::from_raw_parts(x, n as usize);
        wrapper.problem.objective(x_slice, new_x, &mut *obj_value)
    }
}

extern "C" fn eval_grad_f_cb(
    n: i32, x: *const f64, new_x: bool,
    grad_f: *mut f64, user_data: *mut c_void,
) -> bool {
    unsafe {
        let wrapper = &*(user_data as *const ProblemWrapper);
        let x_slice = std::slice::from_raw_parts(x, n as usize);
        let grad_slice = std::slice::from_raw_parts_mut(grad_f, n as usize);
        wrapper.problem.gradient(x_slice, new_x, grad_slice)
    }
}

extern "C" fn eval_g_cb(
    n: i32, x: *const f64, new_x: bool,
    _m: i32, g: *mut f64, user_data: *mut c_void,
) -> bool {
    unsafe {
        let wrapper = &*(user_data as *const ProblemWrapper);
        let x_slice = std::slice::from_raw_parts(x, n as usize);
        let m = wrapper.problem.num_constraints();
        if m > 0 {
            let g_slice = std::slice::from_raw_parts_mut(g, m);
            return wrapper.problem.constraints(x_slice, new_x, g_slice);
        }
        true
    }
}

extern "C" fn eval_jac_g_cb(
    n: i32, x: *const f64, new_x: bool,
    _m: i32, _nele_jac: i32,
    i_row: *mut i32, j_col: *mut i32, values: *mut f64,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        let wrapper = &*(user_data as *const ProblemWrapper);
        if values.is_null() {
            // Return structure
            let nele = wrapper.jac_rows.len();
            let rows = std::slice::from_raw_parts_mut(i_row, nele);
            let cols = std::slice::from_raw_parts_mut(j_col, nele);
            for k in 0..nele {
                rows[k] = wrapper.jac_rows[k];
                cols[k] = wrapper.jac_cols[k];
            }
            true
        } else {
            // Return values
            let x_slice = std::slice::from_raw_parts(x, n as usize);
            let nele = wrapper.jac_rows.len();
            let vals = std::slice::from_raw_parts_mut(values, nele);
            wrapper.problem.jacobian_values(x_slice, new_x, vals)
        }
    }
}

extern "C" fn eval_h_cb(
    n: i32, x: *const f64, new_x: bool,
    obj_factor: f64, _m: i32, lambda: *const f64, _new_lambda: bool,
    _nele_hess: i32,
    i_row: *mut i32, j_col: *mut i32, values: *mut f64,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        let wrapper = &*(user_data as *const ProblemWrapper);
        if values.is_null() {
            // Return structure
            let nele = wrapper.hess_rows.len();
            let rows = std::slice::from_raw_parts_mut(i_row, nele);
            let cols = std::slice::from_raw_parts_mut(j_col, nele);
            for k in 0..nele {
                rows[k] = wrapper.hess_rows[k];
                cols[k] = wrapper.hess_cols[k];
            }
            true
        } else {
            // Return values
            let x_slice = std::slice::from_raw_parts(x, n as usize);
            let m = wrapper.problem.num_constraints();
            let lambda_slice = if m > 0 {
                std::slice::from_raw_parts(lambda, m)
            } else {
                &[]
            };
            let nele = wrapper.hess_rows.len();
            let vals = std::slice::from_raw_parts_mut(values, nele);
            wrapper.problem.hessian_values(x_slice, new_x, obj_factor, lambda_slice, vals)
        }
    }
}

extern "C" fn intermediate_cb(
    _alg_mod: i32, _iter_count: i32, _obj_value: f64,
    _inf_pr: f64, _inf_du: f64, _mu: f64,
    _d_norm: f64, _regularization_size: f64,
    _alpha_du: f64, _alpha_pr: f64, _ls_trials: i32,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        let wrapper = &mut *(user_data as *mut ProblemWrapper);
        wrapper.iterations = _iter_count;
        true
    }
}

// ---- Helper to set a string option ----
fn set_str_option(problem: IpoptProblem, key: &str, val: &str) {
    let k = CString::new(key).unwrap();
    let v = CString::new(val).unwrap();
    unsafe { AddIpoptStrOption(problem, k.as_ptr(), v.as_ptr()); }
}

fn set_num_option(problem: IpoptProblem, key: &str, val: f64) {
    let k = CString::new(key).unwrap();
    unsafe { AddIpoptNumOption(problem, k.as_ptr(), val); }
}

fn set_int_option(problem: IpoptProblem, key: &str, val: i32) {
    let k = CString::new(key).unwrap();
    unsafe { AddIpoptIntOption(problem, k.as_ptr(), val); }
}

// ---- Solve a single problem with native Ipopt ----

struct IpoptResult {
    status: i32,
    objective: f64,
    x: Vec<f64>,
    mult_g: Vec<f64>,
    mult_x_l: Vec<f64>,
    mult_x_u: Vec<f64>,
    iterations: i32,
}

fn solve_with_ipopt(problem: &dyn NlpProblem) -> IpoptResult {
    let n = problem.num_variables();
    let m = problem.num_constraints();

    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    problem.bounds(&mut x_l, &mut x_u);

    let mut g_l = vec![0.0; m.max(1)];
    let mut g_u = vec![0.0; m.max(1)];
    if m > 0 {
        problem.constraint_bounds(&mut g_l, &mut g_u);
    }

    let (jac_rows_usize, jac_cols_usize) = problem.jacobian_structure();
    let nele_jac = jac_rows_usize.len();
    let jac_rows: Vec<i32> = jac_rows_usize.iter().map(|&r| r as i32).collect();
    let jac_cols: Vec<i32> = jac_cols_usize.iter().map(|&c| c as i32).collect();

    let (hess_rows_usize, hess_cols_usize) = problem.hessian_structure();
    let nele_hess = hess_rows_usize.len();
    let hess_rows: Vec<i32> = hess_rows_usize.iter().map(|&r| r as i32).collect();
    let hess_cols: Vec<i32> = hess_cols_usize.iter().map(|&c| c as i32).collect();

    let mut wrapper = ProblemWrapper {
        problem,
        jac_rows,
        jac_cols,
        hess_rows,
        hess_cols,
        iterations: 0,
    };

    unsafe {
        let ipopt_problem = CreateIpoptProblem(
            n as i32, x_l.as_mut_ptr(), x_u.as_mut_ptr(),
            m as i32, g_l.as_mut_ptr(), g_u.as_mut_ptr(),
            nele_jac as i32, nele_hess as i32,
            0, // C-style indexing
            eval_f_cb, eval_g_cb, eval_grad_f_cb,
            eval_jac_g_cb, eval_h_cb,
        );

        if ipopt_problem.is_null() {
            return IpoptResult {
                status: -199, objective: f64::NAN, x: vec![],
                mult_g: vec![], mult_x_l: vec![], mult_x_u: vec![],
                iterations: 0,
            };
        }

        // Set options to match ripopt defaults
        set_str_option(ipopt_problem, "sb", "yes"); // suppress banner
        set_str_option(ipopt_problem, "mu_strategy", "adaptive");
        set_num_option(ipopt_problem, "tol", 1e-8);
        set_int_option(ipopt_problem, "max_iter", 3000);
        set_int_option(ipopt_problem, "print_level", 0);

        SetIntermediateCallback(ipopt_problem, intermediate_cb);

        let mut x = vec![0.0; n];
        problem.initial_point(&mut x);
        let mut g = vec![0.0; m.max(1)];
        let mut obj_val = 0.0;
        let mut mult_g = vec![0.0; m.max(1)];
        let mut mult_x_l = vec![0.0; n];
        let mut mult_x_u = vec![0.0; n];

        let user_data = &mut wrapper as *mut ProblemWrapper as *mut c_void;

        let status = IpoptSolve(
            ipopt_problem,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj_val,
            mult_g.as_mut_ptr(),
            mult_x_l.as_mut_ptr(),
            mult_x_u.as_mut_ptr(),
            user_data,
        );

        let iterations = wrapper.iterations;
        FreeIpoptProblem(ipopt_problem);

        mult_g.truncate(m);

        IpoptResult {
            status,
            objective: obj_val,
            x,
            mult_g,
            mult_x_l,
            mult_x_u,
            iterations,
        }
    }
}

fn status_to_string(status: i32) -> String {
    match status {
        0 => "Optimal".to_string(),
        1 => "Acceptable".to_string(),
        2 => "Infeasible".to_string(),
        -1 => "MaxIterations".to_string(),
        -2 => "RestorationFailed".to_string(),
        -3 => "ErrorInStepComputation".to_string(),
        -13 => "InvalidNumberDetected".to_string(),
        other => format!("IpoptStatus({})", other),
    }
}

// Macro to generate solve calls for each problem (reusing the generated structs)
macro_rules! solve_problem {
    ($prob_struct:expr, $number:expr, $idx:expr, $results:expr) => {{
        let problem = $prob_struct;
        let t0 = Instant::now();
        let result = solve_with_ipopt(&problem);
        let elapsed = t0.elapsed().as_secs_f64();
        $results.push(HsSolveResult {
            number: $number,
            status: status_to_string(result.status),
            objective: result.objective,
            x: result.x,
            constraint_multipliers: result.mult_g,
            bound_multipliers_lower: result.mult_x_l,
            bound_multipliers_upper: result.mult_x_u,
            constraint_values: vec![],
            iterations: result.iterations as usize,
            solve_time: elapsed,
            known_fopt: HS_PROBLEMS[$idx].known_fopt,
            n: HS_PROBLEMS[$idx].n,
            m: HS_PROBLEMS[$idx].m,
            final_primal_inf: 0.0,
            final_dual_inf: 0.0,
            final_compl: 0.0,
            final_mu: 0.0,
            final_s_d: 0.0,
        });
    }};
}

fn solve_all_ipopt() -> Vec<HsSolveResult> {
    use hs_problems::*;
    let mut results = Vec::new();

    solve_problem!(HsTp001, 1, 0, results);
    solve_problem!(HsTp002, 2, 1, results);
    solve_problem!(HsTp003, 3, 2, results);
    solve_problem!(HsTp004, 4, 3, results);
    solve_problem!(HsTp005, 5, 4, results);
    solve_problem!(HsTp006, 6, 5, results);
    solve_problem!(HsTp007, 7, 6, results);
    solve_problem!(HsTp009, 9, 7, results);
    solve_problem!(HsTp010, 10, 8, results);
    solve_problem!(HsTp011, 11, 9, results);
    solve_problem!(HsTp012, 12, 10, results);
    solve_problem!(HsTp013, 13, 11, results);
    solve_problem!(HsTp014, 14, 12, results);
    solve_problem!(HsTp015, 15, 13, results);
    solve_problem!(HsTp016, 16, 14, results);
    solve_problem!(HsTp017, 17, 15, results);
    solve_problem!(HsTp018, 18, 16, results);
    solve_problem!(HsTp019, 19, 17, results);
    solve_problem!(HsTp020, 20, 18, results);
    solve_problem!(HsTp021, 21, 19, results);
    solve_problem!(HsTp022, 22, 20, results);
    solve_problem!(HsTp023, 23, 21, results);
    solve_problem!(HsTp024, 24, 22, results);
    solve_problem!(HsTp026, 26, 23, results);
    solve_problem!(HsTp027, 27, 24, results);
    solve_problem!(HsTp028, 28, 25, results);
    solve_problem!(HsTp029, 29, 26, results);
    solve_problem!(HsTp030, 30, 27, results);
    solve_problem!(HsTp031, 31, 28, results);
    solve_problem!(HsTp032, 32, 29, results);
    solve_problem!(HsTp033, 33, 30, results);
    solve_problem!(HsTp034, 34, 31, results);
    solve_problem!(HsTp035, 35, 32, results);
    solve_problem!(HsTp036, 36, 33, results);
    solve_problem!(HsTp037, 37, 34, results);
    solve_problem!(HsTp038, 38, 35, results);
    solve_problem!(HsTp039, 39, 36, results);
    solve_problem!(HsTp040, 40, 37, results);
    solve_problem!(HsTp041, 41, 38, results);
    solve_problem!(HsTp042, 42, 39, results);
    solve_problem!(HsTp043, 43, 40, results);
    solve_problem!(HsTp044, 44, 41, results);
    solve_problem!(HsTp045, 45, 42, results);
    solve_problem!(HsTp046, 46, 43, results);
    solve_problem!(HsTp047, 47, 44, results);
    solve_problem!(HsTp048, 48, 45, results);
    solve_problem!(HsTp049, 49, 46, results);
    solve_problem!(HsTp050, 50, 47, results);
    solve_problem!(HsTp051, 51, 48, results);
    solve_problem!(HsTp052, 52, 49, results);
    solve_problem!(HsTp053, 53, 50, results);
    solve_problem!(HsTp056, 56, 51, results);
    solve_problem!(HsTp058, 58, 52, results);
    solve_problem!(HsTp060, 60, 53, results);
    solve_problem!(HsTp061, 61, 54, results);
    solve_problem!(HsTp063, 63, 55, results);
    solve_problem!(HsTp064, 64, 56, results);
    solve_problem!(HsTp065, 65, 57, results);
    solve_problem!(HsTp066, 66, 58, results);
    solve_problem!(HsTp071, 71, 59, results);
    solve_problem!(HsTp072, 72, 60, results);
    solve_problem!(HsTp076, 76, 61, results);
    solve_problem!(HsTp077, 77, 62, results);
    solve_problem!(HsTp078, 78, 63, results);
    solve_problem!(HsTp079, 79, 64, results);
    solve_problem!(HsTp080, 80, 65, results);
    solve_problem!(HsTp081, 81, 66, results);
    solve_problem!(HsTp106, 106, 67, results);
    solve_problem!(HsTp108, 108, 68, results);
    solve_problem!(HsTp113, 113, 69, results);
    solve_problem!(HsTp114, 114, 70, results);
    solve_problem!(HsTp116, 116, 71, results);
    solve_problem!(HsTp201, 201, 72, results);
    solve_problem!(HsTp206, 206, 73, results);
    solve_problem!(HsTp211, 211, 74, results);
    solve_problem!(HsTp212, 212, 75, results);
    solve_problem!(HsTp213, 213, 76, results);
    solve_problem!(HsTp214, 214, 77, results);
    solve_problem!(HsTp215, 215, 78, results);
    solve_problem!(HsTp216, 216, 79, results);
    solve_problem!(HsTp217, 217, 80, results);
    solve_problem!(HsTp218, 218, 81, results);
    solve_problem!(HsTp219, 219, 82, results);
    solve_problem!(HsTp220, 220, 83, results);
    solve_problem!(HsTp221, 221, 84, results);
    solve_problem!(HsTp223, 223, 85, results);
    solve_problem!(HsTp224, 224, 86, results);
    solve_problem!(HsTp225, 225, 87, results);
    solve_problem!(HsTp226, 226, 88, results);
    solve_problem!(HsTp227, 227, 89, results);
    solve_problem!(HsTp228, 228, 90, results);
    solve_problem!(HsTp229, 229, 91, results);
    solve_problem!(HsTp230, 230, 92, results);
    solve_problem!(HsTp232, 232, 93, results);
    solve_problem!(HsTp234, 234, 94, results);
    solve_problem!(HsTp235, 235, 95, results);
    solve_problem!(HsTp240, 240, 96, results);
    solve_problem!(HsTp248, 248, 97, results);
    solve_problem!(HsTp249, 249, 98, results);
    solve_problem!(HsTp250, 250, 99, results);
    solve_problem!(HsTp251, 251, 100, results);
    solve_problem!(HsTp252, 252, 101, results);
    solve_problem!(HsTp254, 254, 102, results);
    solve_problem!(HsTp255, 255, 103, results);
    solve_problem!(HsTp256, 256, 104, results);
    solve_problem!(HsTp257, 257, 105, results);
    solve_problem!(HsTp258, 258, 106, results);
    solve_problem!(HsTp259, 259, 107, results);
    solve_problem!(HsTp262, 262, 108, results);
    solve_problem!(HsTp263, 263, 109, results);
    solve_problem!(HsTp264, 264, 110, results);
    solve_problem!(HsTp270, 270, 111, results);
    solve_problem!(HsTp325, 325, 112, results);
    solve_problem!(HsTp335, 335, 113, results);
    solve_problem!(HsTp338, 338, 114, results);
    solve_problem!(HsTp339, 339, 115, results);
    solve_problem!(HsTp344, 344, 116, results);
    solve_problem!(HsTp354, 354, 117, results);
    solve_problem!(HsTp374, 374, 118, results);
    solve_problem!(HsTp376, 376, 119, results);

    results
}

fn main() {
    let n_timing_runs: usize = std::env::var("RIPOPT_TIMING_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    eprintln!("Solving all HS problems with native Ipopt ({} timing runs)...", n_timing_runs);

    // First run: get correctness results
    let mut results = solve_all_ipopt();

    // Additional timing runs: keep minimum solve_time per problem
    for run in 1..n_timing_runs {
        eprintln!("  Timing run {}/{}...", run + 1, n_timing_runs);
        let timing_results = solve_all_ipopt();
        for (r, t) in results.iter_mut().zip(timing_results.iter()) {
            if t.solve_time < r.solve_time {
                r.solve_time = t.solve_time;
            }
        }
    }

    // Summary to stderr
    let total = results.len();
    let optimal = results.iter().filter(|r| r.status == "Optimal").count();
    let acceptable = results.iter().filter(|r| r.status == "Acceptable").count();
    let solved = optimal + acceptable;
    eprintln!("Solved {}/{} ({} optimal, {} acceptable)", solved, total, optimal, acceptable);

    // JSON to stdout
    let json = serde_json::to_string_pretty(&results).unwrap();
    println!("{}", json);

    // Optional JSONL companion (parity with run_ripopt.rs).
    let jsonl_path: Option<std::path::PathBuf> = std::env::var("RESULTS_JSONL")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HS_RESULTS_FILE")
                .ok()
                .map(|p| std::path::PathBuf::from(p).with_extension("jsonl"))
        });
    if let Some(path) = jsonl_path {
        use std::io::Write;
        match std::fs::File::create(&path) {
            Ok(f) => {
                let mut w = std::io::BufWriter::new(f);
                for r in &results {
                    if let Ok(line) = serde_json::to_string(r) {
                        let _ = writeln!(w, "{}", line);
                    }
                }
                let _ = w.flush();
                eprintln!("JSONL stream: {}", path.display());
            }
            Err(e) => eprintln!("WARNING: cannot open {}: {}", path.display(), e),
        }
    }
}
