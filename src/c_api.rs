//! C API for ripopt — mirrors the Ipopt C interface.
//!
//! Compile with `crate-type = ["cdylib", "rlib"]` to produce a shared library.
//! Include `ripopt.h` from C/C++ code.

use std::ffi::{CStr, CString};
use std::io::Write;
use std::os::raw::{c_char, c_double, c_int, c_void};

use crate::options::SolverOptions;
use crate::problem::NlpProblem;
use crate::result::{SolveResult, SolveStatus};

// ---------------------------------------------------------------------------
// Callback function-pointer types (identical to Ipopt C API)
// ---------------------------------------------------------------------------

/// Evaluate objective f(x). Return 1 (true) on success, 0 on failure.
pub type EvalFCb = unsafe extern "C" fn(
    n: c_int,
    x: *const c_double,
    new_x: c_int,
    obj_value: *mut c_double,
    user_data: *mut std::ffi::c_void,
) -> c_int;

/// Evaluate gradient of f(x).
pub type EvalGradFCb = unsafe extern "C" fn(
    n: c_int,
    x: *const c_double,
    new_x: c_int,
    grad_f: *mut c_double,
    user_data: *mut std::ffi::c_void,
) -> c_int;

/// Evaluate constraints g(x).
pub type EvalGCb = unsafe extern "C" fn(
    n: c_int,
    x: *const c_double,
    new_x: c_int,
    m: c_int,
    g: *mut c_double,
    user_data: *mut std::ffi::c_void,
) -> c_int;

/// Evaluate Jacobian of constraints.
/// When `values` is NULL, fill `iRow`/`jCol` with sparsity pattern.
/// When `values` is non-NULL, fill values in the same order.
pub type EvalJacGCb = unsafe extern "C" fn(
    n: c_int,
    x: *const c_double,
    new_x: c_int,
    m: c_int,
    nele_jac: c_int,
    i_row: *mut c_int,
    j_col: *mut c_int,
    values: *mut c_double,
    user_data: *mut std::ffi::c_void,
) -> c_int;

/// Evaluate Hessian of the Lagrangian.
/// When `values` is NULL, fill `iRow`/`jCol` with sparsity pattern.
/// When `values` is non-NULL, fill values in the same order (lower triangle).
pub type EvalHCb = unsafe extern "C" fn(
    n: c_int,
    x: *const c_double,
    new_x: c_int,
    obj_factor: c_double,
    m: c_int,
    lambda: *const c_double,
    new_lambda: c_int,
    nele_hess: c_int,
    i_row: *mut c_int,
    j_col: *mut c_int,
    values: *mut c_double,
    user_data: *mut std::ffi::c_void,
) -> c_int;

/// Intermediate callback, called once per iteration.
/// Return 1 to continue, 0 to request early termination.
pub type IntermediateCb = unsafe extern "C" fn(
    iter: c_int,
    obj_value: c_double,
    inf_pr: c_double,
    inf_du: c_double,
    mu: c_double,
    alpha_pr: c_double,
    alpha_du: c_double,
    ls_trials: c_int,
    user_data: *mut c_void,
) -> c_int;

// ---------------------------------------------------------------------------
// Opaque problem struct
// ---------------------------------------------------------------------------

pub struct CApiProblem {
    n: usize,
    m: usize,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    nele_jac: usize,
    nele_hess: usize,
    eval_f: EvalFCb,
    eval_grad_f: EvalGradFCb,
    eval_g: EvalGCb,
    eval_jac_g: EvalJacGCb,
    eval_h: EvalHCb,
    options: SolverOptions,
    /// Initial point supplied by the caller via `ripopt_solve`.
    initial_x: Vec<f64>,
    user_data: *mut std::ffi::c_void,
    /// Optional log callback (set via `ripopt_set_log_callback`).
    log_cb: Option<crate::logging::LogCb>,
    log_cb_user_data: *mut std::ffi::c_void,
    /// Optional file handle for log output (set via `ripopt_open_output_file`).
    log_file: Option<Box<std::fs::File>>,
    /// Optional intermediate callback (set via `ripopt_set_intermediate_callback`).
    intermediate_cb: Option<IntermediateCb>,
    intermediate_cb_user_data: *mut c_void,
    /// Initial multipliers for warm starting (set from ripopt_solve when warm_start=true).
    initial_y: Vec<f64>,
    initial_z_l: Vec<f64>,
    initial_z_u: Vec<f64>,
    /// Diagnostics from the most recent solve.
    last_iterations: usize,
    last_obj: f64,
    last_primal_inf: f64,
    last_dual_inf: f64,
    last_compl: f64,
    last_wall_time: f64,
}

// SAFETY: The user is responsible for ensuring `user_data` is valid across
// threads.  ripopt is single-threaded internally, so this is safe in practice.
unsafe impl Send for CApiProblem {}
unsafe impl Sync for CApiProblem {}

// ---------------------------------------------------------------------------
// NlpProblem impl — delegates to C callbacks
// ---------------------------------------------------------------------------

impl NlpProblem for CApiProblem {
    fn num_variables(&self) -> usize {
        self.n
    }

    fn num_constraints(&self) -> usize {
        self.m
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.copy_from_slice(&self.x_l);
        x_u.copy_from_slice(&self.x_u);
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l.copy_from_slice(&self.g_l);
        g_u.copy_from_slice(&self.g_u);
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0.copy_from_slice(&self.initial_x);
    }

    fn objective(&self, x: &[f64], new_x: bool, obj: &mut f64) -> bool {
        let ok = unsafe {
            (self.eval_f)(
                self.n as c_int,
                x.as_ptr(),
                new_x as c_int,
                obj,
                self.user_data,
            )
        };
        ok != 0
    }

    fn gradient(&self, x: &[f64], new_x: bool, grad: &mut [f64]) -> bool {
        let ok = unsafe {
            (self.eval_grad_f)(
                self.n as c_int,
                x.as_ptr(),
                new_x as c_int,
                grad.as_mut_ptr(),
                self.user_data,
            )
        };
        ok != 0
    }

    fn constraints(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool {
        let ok = unsafe {
            (self.eval_g)(
                self.n as c_int,
                x.as_ptr(),
                new_x as c_int,
                self.m as c_int,
                g.as_mut_ptr(),
                self.user_data,
            )
        };
        ok != 0
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let nnz = self.nele_jac;
        let mut i_row = vec![0_i32; nnz];
        let mut j_col = vec![0_i32; nnz];
        // Call with values = NULL to get sparsity pattern
        unsafe {
            (self.eval_jac_g)(
                self.n as c_int,
                std::ptr::null(),
                1,
                self.m as c_int,
                nnz as c_int,
                i_row.as_mut_ptr(),
                j_col.as_mut_ptr(),
                std::ptr::null_mut(),
                self.user_data,
            );
        }
        (
            i_row.into_iter().map(|v| v as usize).collect(),
            j_col.into_iter().map(|v| v as usize).collect(),
        )
    }

    fn jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool {
        let nnz = self.nele_jac;
        let ok = unsafe {
            (self.eval_jac_g)(
                self.n as c_int,
                x.as_ptr(),
                new_x as c_int,
                self.m as c_int,
                nnz as c_int,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                vals.as_mut_ptr(),
                self.user_data,
            )
        };
        ok != 0
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let nnz = self.nele_hess;
        let mut i_row = vec![0_i32; nnz];
        let mut j_col = vec![0_i32; nnz];
        // Call with values = NULL to get sparsity pattern
        unsafe {
            (self.eval_h)(
                self.n as c_int,
                std::ptr::null(),
                1,
                1.0,
                self.m as c_int,
                std::ptr::null(),
                1,
                nnz as c_int,
                i_row.as_mut_ptr(),
                j_col.as_mut_ptr(),
                std::ptr::null_mut(),
                self.user_data,
            );
        }
        (
            i_row.into_iter().map(|v| v as usize).collect(),
            j_col.into_iter().map(|v| v as usize).collect(),
        )
    }

    fn hessian_values(&self, x: &[f64], new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        let nnz = self.nele_hess;
        let ok = unsafe {
            (self.eval_h)(
                self.n as c_int,
                x.as_ptr(),
                new_x as c_int,
                obj_factor,
                self.m as c_int,
                lambda.as_ptr(),
                1,
                nnz as c_int,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                vals.as_mut_ptr(),
                self.user_data,
            )
        };
        ok != 0
    }
}

// ---------------------------------------------------------------------------
// Return status (matches C header enum)
// ---------------------------------------------------------------------------

#[repr(C)]
pub enum RipoptReturnStatus {
    SolveSucceeded = 0,
    InfeasibleProblem = 2,
    MaxIterExceeded = 5,
    RestorationFailed = 6,
    ErrorInStepComputation = 7,
    UserRequestedStop = 3,
    NotEnoughDegreesOfFreedom = 10,
    InvalidProblemDefinition = 11,
    InternalError = -1,
}

fn map_status(s: SolveStatus) -> RipoptReturnStatus {
    match s {
        SolveStatus::Optimal => RipoptReturnStatus::SolveSucceeded,
        SolveStatus::Infeasible | SolveStatus::LocalInfeasibility => {
            RipoptReturnStatus::InfeasibleProblem
        }
        SolveStatus::MaxIterations => RipoptReturnStatus::MaxIterExceeded,
        SolveStatus::RestorationFailed => RipoptReturnStatus::RestorationFailed,
        SolveStatus::NumericalError | SolveStatus::EvaluationError => {
            RipoptReturnStatus::ErrorInStepComputation
        }
        SolveStatus::UserRequestedStop => RipoptReturnStatus::UserRequestedStop,
        SolveStatus::Unbounded => RipoptReturnStatus::InvalidProblemDefinition,
        SolveStatus::InternalError => RipoptReturnStatus::InternalError,
    }
}

// ---------------------------------------------------------------------------
// Exported C functions
// ---------------------------------------------------------------------------

/// Create a new ripopt problem handle.
///
/// # Safety
/// All pointer arguments must be valid for the indicated lengths.
#[no_mangle]
pub unsafe extern "C" fn ripopt_create(
    n: c_int,
    x_l: *const c_double,
    x_u: *const c_double,
    m: c_int,
    g_l: *const c_double,
    g_u: *const c_double,
    nele_jac: c_int,
    nele_hess: c_int,
    eval_f: EvalFCb,
    eval_grad_f: EvalGradFCb,
    eval_g: EvalGCb,
    eval_jac_g: EvalJacGCb,
    eval_h: EvalHCb,
) -> *mut CApiProblem {
    let n = n as usize;
    let m = m as usize;
    let (g_l_vec, g_u_vec) = if m > 0 {
        (
            std::slice::from_raw_parts(g_l, m).to_vec(),
            std::slice::from_raw_parts(g_u, m).to_vec(),
        )
    } else {
        (vec![], vec![])
    };
    let problem = Box::new(CApiProblem {
        n,
        m,
        x_l: std::slice::from_raw_parts(x_l, n).to_vec(),
        x_u: std::slice::from_raw_parts(x_u, n).to_vec(),
        g_l: g_l_vec,
        g_u: g_u_vec,
        nele_jac: nele_jac as usize,
        nele_hess: nele_hess as usize,
        eval_f,
        eval_grad_f,
        eval_g,
        eval_jac_g,
        eval_h,
        options: SolverOptions::default(),
        initial_x: vec![0.0; n],
        user_data: std::ptr::null_mut(),
        log_cb: None,
        log_cb_user_data: std::ptr::null_mut(),
        log_file: None,
        intermediate_cb: None,
        intermediate_cb_user_data: std::ptr::null_mut(),
        initial_y: vec![0.0; m],
        initial_z_l: vec![0.0; n],
        initial_z_u: vec![0.0; n],
        last_iterations: 0,
        last_obj: 0.0,
        last_primal_inf: 0.0,
        last_dual_inf: 0.0,
        last_compl: 0.0,
        last_wall_time: 0.0,
    });
    Box::into_raw(problem)
}

/// Free a ripopt problem handle.
///
/// # Safety
/// `problem` must be a valid pointer returned by `ripopt_create`.
#[no_mangle]
pub unsafe extern "C" fn ripopt_free(problem: *mut CApiProblem) {
    if !problem.is_null() {
        drop(Box::from_raw(problem));
    }
}

/// Set a numeric (double) option.
///
/// Returns 1 on success, 0 if the keyword is unknown.
///
/// # Safety
/// `problem` and `keyword` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ripopt_add_num_option(
    problem: *mut CApiProblem,
    keyword: *const c_char,
    val: c_double,
) -> c_int {
    let p = &mut *problem;
    let key = match CStr::from_ptr(keyword).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    match key {
        "tol" => p.options.tol = val,
        "mu_init" => p.options.mu_init = val,
        "mu_min" => p.options.mu_min = val,
        "tau_min" => p.options.tau_min = val,
        "mu_linear_decrease_factor" => p.options.mu_linear_decrease_factor = val,
        "mu_superlinear_decrease_power" => p.options.mu_superlinear_decrease_power = val,
        "bound_push" => p.options.bound_push = val,
        "bound_frac" => p.options.bound_frac = val,
        "slack_bound_push" => p.options.slack_bound_push = val,
        "slack_bound_frac" => p.options.slack_bound_frac = val,
        "constr_viol_tol" => p.options.constr_viol_tol = val,
        "dual_inf_tol" => p.options.dual_inf_tol = val,
        "compl_inf_tol" => p.options.compl_inf_tol = val,
        "warm_start_bound_push" => p.options.warm_start_bound_push = val,
        "warm_start_bound_frac" => p.options.warm_start_bound_frac = val,
        "warm_start_mult_bound_push" => p.options.warm_start_mult_bound_push = val,
        "nlp_lower_bound_inf" => p.options.nlp_lower_bound_inf = val,
        "nlp_upper_bound_inf" => p.options.nlp_upper_bound_inf = val,
        "kappa" => p.options.kappa = val,
        "constr_mult_init_max" => p.options.constr_mult_init_max = val,
        "max_wall_time" => p.options.max_wall_time = val,
        "barrier_tol_factor" => p.options.barrier_tol_factor = val,
        "adaptive_mu_monotone_init_factor" => p.options.adaptive_mu_monotone_init_factor = val,
        _ => return 0,
    }
    1
}

/// Set an integer option.
///
/// Returns 1 on success, 0 if the keyword is unknown.
///
/// # Safety
/// `problem` and `keyword` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ripopt_add_int_option(
    problem: *mut CApiProblem,
    keyword: *const c_char,
    val: c_int,
) -> c_int {
    let p = &mut *problem;
    let key = match CStr::from_ptr(keyword).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    match key {
        "max_iter" => p.options.max_iter = val as usize,
        "print_level" => p.options.print_level = val.clamp(0, 12) as u8,
        "max_soc" => p.options.max_soc = val as usize,
        "watchdog_shortened_iter_trigger" => {
            p.options.watchdog_shortened_iter_trigger = val as usize
        }
        "watchdog_trial_iter_max" => p.options.watchdog_trial_iter_max = val as usize,
        "sparse_threshold" => p.options.sparse_threshold = val as usize,
        "restoration_max_iter" => p.options.restoration_max_iter = val as usize,
        "gondzio_mcc_max" => p.options.gondzio_mcc_max = val as usize,
        _ => return 0,
    }
    1
}

/// Set a string option.
///
/// Returns 1 on success, 0 if the keyword or value is unknown.
///
/// # Safety
/// `problem`, `keyword`, and `val` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn ripopt_add_str_option(
    problem: *mut CApiProblem,
    keyword: *const c_char,
    val: *const c_char,
) -> c_int {
    let p = &mut *problem;
    let key = match CStr::from_ptr(keyword).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let value = match CStr::from_ptr(val).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    match key {
        "mu_strategy" => {
            p.options.mu_strategy_adaptive = value == "adaptive";
        }
        "warm_start_init_point" => {
            p.options.warm_start = value == "yes";
        }
        "mu_allow_increase" => {
            p.options.mu_allow_increase = value == "yes";
        }
        "least_squares_mult_init" => {
            p.options.least_squares_mult_init = value == "yes";
        }
        "constraint_slack_barrier" => {
            p.options.constraint_slack_barrier = value == "yes";
        }
        "disable_nlp_restoration" => {
            p.options.disable_nlp_restoration = value == "yes";
        }
        "enable_slack_fallback" => {
            p.options.enable_slack_fallback = value == "yes";
        }
        "enable_lbfgs_fallback" => {
            p.options.enable_lbfgs_fallback = value == "yes";
        }
        "enable_al_fallback" => {
            p.options.enable_al_fallback = value == "yes";
        }
        "enable_preprocessing" => {
            p.options.enable_preprocessing = value == "yes";
        }
        "detect_linear_constraints" => {
            p.options.detect_linear_constraints = value == "yes";
        }
        "enable_sqp_fallback" => {
            p.options.enable_sqp_fallback = value == "yes";
        }
        "hessian_approximation" => {
            match value {
                "limited-memory" => p.options.hessian_approximation_lbfgs = true,
                "exact" => p.options.hessian_approximation_lbfgs = false,
                _ => return 0,
            }
        }
        "enable_lbfgs_hessian_fallback" => {
            p.options.enable_lbfgs_hessian_fallback = value == "yes";
        }
        "mehrotra_pc" => {
            p.options.mehrotra_pc = value == "yes";
        }
        "proactive_infeasibility_detection" => {
            p.options.proactive_infeasibility_detection = value == "yes";
        }
        _ => return 0,
    }
    1
}

/// Solve the NLP.
///
/// `x` is in/out: on entry the initial point, on exit the solution.
/// `g`, `mult_g`, `mult_x_l`, `mult_x_u` may be NULL if not needed.
///
/// # Safety
/// All non-null pointer arguments must be valid for the indicated lengths.
#[no_mangle]
pub unsafe extern "C" fn ripopt_solve(
    problem: *mut CApiProblem,
    x: *mut c_double,
    g: *mut c_double,
    obj_val: *mut c_double,
    mult_g: *mut c_double,
    mult_x_l: *mut c_double,
    mult_x_u: *mut c_double,
    user_data: *mut std::ffi::c_void,
) -> c_int {
    let p = &mut *problem;

    // Store user_data and initial point
    p.user_data = user_data;
    p.initial_x
        .copy_from_slice(std::slice::from_raw_parts(x, p.n));

    // When warm_start is enabled, read multipliers from input pointers (IPOPT convention)
    if p.options.warm_start {
        if !mult_g.is_null() && p.m > 0 {
            p.initial_y.copy_from_slice(std::slice::from_raw_parts(mult_g, p.m));
        }
        if !mult_x_l.is_null() {
            p.initial_z_l.copy_from_slice(std::slice::from_raw_parts(mult_x_l, p.n));
        }
        if !mult_x_u.is_null() {
            p.initial_z_u.copy_from_slice(std::slice::from_raw_parts(mult_x_u, p.n));
        }
    }

    // Install log callback for this thread (if one was registered)
    if let Some(cb) = p.log_cb {
        crate::logging::set_log_callback(Some((cb, p.log_cb_user_data)));
    }

    // Install intermediate callback for this thread (if one was registered)
    if let Some(cb) = p.intermediate_cb {
        crate::intermediate::set_intermediate_callback(Some((cb, p.intermediate_cb_user_data)));
    }

    // Copy initial multipliers into options for the solver
    let mut opts = p.options.clone();
    if opts.warm_start {
        if p.initial_y.iter().any(|&v| v != 0.0) {
            opts.warm_start_y = Some(p.initial_y.clone());
        }
        if p.initial_z_l.iter().any(|&v| v != 0.0) {
            opts.warm_start_z_l = Some(p.initial_z_l.clone());
        }
        if p.initial_z_u.iter().any(|&v| v != 0.0) {
            opts.warm_start_z_u = Some(p.initial_z_u.clone());
        }
    }

    // Solve
    let result: SolveResult = crate::solve(p, &opts);

    // Clear callbacks
    crate::logging::set_log_callback(None);
    crate::intermediate::set_intermediate_callback(None);

    // Store diagnostics for getter functions
    p.last_iterations = result.iterations;
    p.last_obj = result.objective;
    p.last_primal_inf = result.diagnostics.final_primal_inf;
    p.last_dual_inf = result.diagnostics.final_dual_inf;
    p.last_compl = result.diagnostics.final_compl;
    p.last_wall_time = result.diagnostics.wall_time_secs;

    // Copy primal solution back
    std::slice::from_raw_parts_mut(x, p.n).copy_from_slice(&result.x);

    // Objective
    if !obj_val.is_null() {
        *obj_val = result.objective;
    }

    // Constraint values
    if !g.is_null() {
        std::slice::from_raw_parts_mut(g, p.m).copy_from_slice(&result.constraint_values);
    }

    // Multipliers
    if !mult_g.is_null() {
        std::slice::from_raw_parts_mut(mult_g, p.m)
            .copy_from_slice(&result.constraint_multipliers);
    }
    if !mult_x_l.is_null() {
        std::slice::from_raw_parts_mut(mult_x_l, p.n)
            .copy_from_slice(&result.bound_multipliers_lower);
    }
    if !mult_x_u.is_null() {
        std::slice::from_raw_parts_mut(mult_x_u, p.n)
            .copy_from_slice(&result.bound_multipliers_upper);
    }

    map_status(result.status) as c_int
}

/// Register a log callback for this problem.
///
/// When set, all solver output (iteration table, diagnostics, warnings) is
/// forwarded to `callback(msg, user_data)` instead of being written to stderr.
/// Call with `callback = NULL` to revert to stderr.
///
/// The callback is thread-local: it only applies to the thread that calls
/// `ripopt_solve`, and is cleared automatically after each solve.
///
/// # Safety
/// `problem` must be valid. `callback` and `user_data` must remain valid
/// for the duration of the next `ripopt_solve` call.
#[no_mangle]
pub unsafe extern "C" fn ripopt_set_log_callback(
    problem: *mut CApiProblem,
    callback: Option<crate::logging::LogCb>,
    user_data: *mut std::ffi::c_void,
) {
    let p = &mut *problem;
    p.log_cb = callback;
    p.log_cb_user_data = user_data;
}

/// Return the number of iterations from the most recent solve.
///
/// # Safety
/// `problem` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ripopt_get_iter_count(problem: *const CApiProblem) -> c_int {
    (*problem).last_iterations as c_int
}

/// Return the wall-clock solve time (seconds) from the most recent solve.
///
/// # Safety
/// `problem` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ripopt_get_solve_time(problem: *const CApiProblem) -> c_double {
    (*problem).last_wall_time
}

/// Return the final primal infeasibility from the most recent solve.
///
/// # Safety
/// `problem` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ripopt_get_primal_inf(problem: *const CApiProblem) -> c_double {
    (*problem).last_primal_inf
}

/// Return the final dual infeasibility from the most recent solve.
///
/// # Safety
/// `problem` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ripopt_get_dual_inf(problem: *const CApiProblem) -> c_double {
    (*problem).last_dual_inf
}

/// Return the final complementarity error from the most recent solve.
///
/// # Safety
/// `problem` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ripopt_get_compl_inf(problem: *const CApiProblem) -> c_double {
    (*problem).last_compl
}

// ---------------------------------------------------------------------------
// File logging
// ---------------------------------------------------------------------------

/// File-logging callback: writes each message to the stored file handle.
unsafe extern "C" fn file_log_callback(msg: *const c_char, user_data: *mut c_void) {
    let file = &mut *(user_data as *mut std::fs::File);
    if let Ok(s) = CStr::from_ptr(msg).to_str() {
        let _ = writeln!(file, "{}", s);
    }
}

/// Open a log file for solver output.
///
/// All solver output (iteration table, diagnostics, warnings) is written to
/// the specified file. This overrides any previously set log callback.
/// Pass `print_level` to control verbosity (0 = silent, 5 = verbose).
///
/// Returns 1 on success, 0 if the file cannot be opened.
///
/// # Safety
/// `problem` and `filename` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ripopt_open_output_file(
    problem: *mut CApiProblem,
    filename: *const c_char,
    print_level: c_int,
) -> c_int {
    let p = &mut *problem;
    let path = match CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    match std::fs::File::create(path) {
        Ok(file) => {
            let boxed = Box::new(file);
            let ptr = &*boxed as *const std::fs::File as *mut c_void;
            p.log_file = Some(boxed);
            p.log_cb = Some(file_log_callback);
            p.log_cb_user_data = ptr;
            p.options.print_level = print_level.clamp(0, 12) as u8;
            1
        }
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Intermediate callback
// ---------------------------------------------------------------------------

/// Register an intermediate callback invoked once per solver iteration.
///
/// The callback receives current iteration metrics and returns 1 to continue
/// or 0 to request early termination (solver returns `UserRequestedStop`).
///
/// # Safety
/// `problem` must be valid. The callback and `user_data` must remain valid
/// for the duration of the next `ripopt_solve` call.
#[no_mangle]
pub unsafe extern "C" fn ripopt_set_intermediate_callback(
    problem: *mut CApiProblem,
    callback: Option<IntermediateCb>,
    user_data: *mut c_void,
) {
    let p = &mut *problem;
    p.intermediate_cb = callback;
    p.intermediate_cb_user_data = user_data;
}

// ---------------------------------------------------------------------------
// Problem scaling
// ---------------------------------------------------------------------------

/// Set user-provided problem scaling.
///
/// `obj_scaling` scales the objective function. `g_scaling` (length m) scales
/// each constraint. Pass NULL for `g_scaling` to scale only the objective.
///
/// After calling this function, the solver uses user-provided scaling instead
/// of automatic gradient-based scaling.
///
/// # Safety
/// `problem` must be valid. `g_scaling` (if non-null) must have length m.
#[no_mangle]
pub unsafe extern "C" fn ripopt_set_scaling(
    problem: *mut CApiProblem,
    obj_scaling: c_double,
    g_scaling: *const c_double,
) {
    let p = &mut *problem;
    p.options.user_obj_scaling = Some(obj_scaling);
    if !g_scaling.is_null() && p.m > 0 {
        p.options.user_g_scaling = Some(
            std::slice::from_raw_parts(g_scaling, p.m).to_vec(),
        );
    }
}
