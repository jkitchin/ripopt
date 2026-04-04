// Fair comparison of ripopt vs ipopt (C API) on large-scale benchmark problems.
// Both solvers use the same Rust problem structs, same starting points, same tolerances.
//
// Run with: cargo run --release --features ipopt-native --example compare_large_scale

use ripopt::{NlpProblem, SolverOptions};
use std::f64::consts::PI;
use std::ffi::CString;
use std::os::raw::c_void;
use std::time::Instant;

// =========================================================================
// Ipopt C API FFI (same as run_ipopt_native.rs)
// =========================================================================

type IpoptProblemPtr = *mut c_void;

extern "C" {
    fn CreateIpoptProblem(
        n: i32, x_l: *mut f64, x_u: *mut f64,
        m: i32, g_l: *mut f64, g_u: *mut f64,
        nele_jac: i32, nele_hess: i32, index_style: i32,
        eval_f: EvalFCB, eval_g: EvalGCB, eval_grad_f: EvalGradFCB,
        eval_jac_g: EvalJacGCB, eval_h: EvalHCB,
    ) -> IpoptProblemPtr;

    fn FreeIpoptProblem(problem: IpoptProblemPtr);
    fn AddIpoptStrOption(problem: IpoptProblemPtr, keyword: *const i8, val: *const i8) -> bool;
    fn AddIpoptNumOption(problem: IpoptProblemPtr, keyword: *const i8, val: f64) -> bool;
    fn AddIpoptIntOption(problem: IpoptProblemPtr, keyword: *const i8, val: i32) -> bool;
    fn SetIntermediateCallback(problem: IpoptProblemPtr, cb: IntermediateCB) -> bool;
    fn IpoptSolve(
        problem: IpoptProblemPtr,
        x: *mut f64, g: *mut f64, obj_val: *mut f64,
        mult_g: *mut f64, mult_x_l: *mut f64, mult_x_u: *mut f64,
        user_data: *mut c_void,
    ) -> i32;
}

type EvalFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGradFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGCB = extern "C" fn(i32, *const f64, bool, i32, *mut f64, *mut c_void) -> bool;
type EvalJacGCB = extern "C" fn(i32, *const f64, bool, i32, i32, *mut i32, *mut i32, *mut f64, *mut c_void) -> bool;
type EvalHCB = extern "C" fn(i32, *const f64, bool, f64, i32, *const f64, bool, i32, *mut i32, *mut i32, *mut f64, *mut c_void) -> bool;
type IntermediateCB = extern "C" fn(i32, i32, f64, f64, f64, f64, f64, f64, f64, f64, i32, *mut c_void) -> bool;

struct IpoptWrapper<'a> {
    problem: &'a dyn NlpProblem,
    jac_rows: Vec<i32>,
    jac_cols: Vec<i32>,
    hess_rows: Vec<i32>,
    hess_cols: Vec<i32>,
    iterations: i32,
}

extern "C" fn eval_f_cb(n: i32, x: *const f64, _: bool, obj: *mut f64, ud: *mut c_void) -> bool {
    unsafe {
        let w = &*(ud as *const IpoptWrapper);
        *obj = w.problem.objective(std::slice::from_raw_parts(x, n as usize));
        true
    }
}

extern "C" fn eval_grad_f_cb(n: i32, x: *const f64, _: bool, g: *mut f64, ud: *mut c_void) -> bool {
    unsafe {
        let w = &*(ud as *const IpoptWrapper);
        let xs = std::slice::from_raw_parts(x, n as usize);
        let gs = std::slice::from_raw_parts_mut(g, n as usize);
        w.problem.gradient(xs, gs);
        true
    }
}

extern "C" fn eval_g_cb(n: i32, x: *const f64, _: bool, _m: i32, g: *mut f64, ud: *mut c_void) -> bool {
    unsafe {
        let w = &*(ud as *const IpoptWrapper);
        let m = w.problem.num_constraints();
        if m > 0 {
            let xs = std::slice::from_raw_parts(x, n as usize);
            let gs = std::slice::from_raw_parts_mut(g, m);
            w.problem.constraints(xs, gs);
        }
        true
    }
}

extern "C" fn eval_jac_g_cb(
    n: i32, x: *const f64, _: bool, _m: i32, _nj: i32,
    ir: *mut i32, jc: *mut i32, vals: *mut f64, ud: *mut c_void,
) -> bool {
    unsafe {
        let w = &*(ud as *const IpoptWrapper);
        if vals.is_null() {
            let nele = w.jac_rows.len();
            let rows = std::slice::from_raw_parts_mut(ir, nele);
            let cols = std::slice::from_raw_parts_mut(jc, nele);
            rows.copy_from_slice(&w.jac_rows);
            cols.copy_from_slice(&w.jac_cols);
        } else {
            let xs = std::slice::from_raw_parts(x, n as usize);
            let vs = std::slice::from_raw_parts_mut(vals, w.jac_rows.len());
            w.problem.jacobian_values(xs, vs);
        }
        true
    }
}

extern "C" fn eval_h_cb(
    n: i32, x: *const f64, _: bool, obj_factor: f64,
    _m: i32, lambda: *const f64, _: bool, _nh: i32,
    ir: *mut i32, jc: *mut i32, vals: *mut f64, ud: *mut c_void,
) -> bool {
    unsafe {
        let w = &*(ud as *const IpoptWrapper);
        if vals.is_null() {
            let nele = w.hess_rows.len();
            let rows = std::slice::from_raw_parts_mut(ir, nele);
            let cols = std::slice::from_raw_parts_mut(jc, nele);
            rows.copy_from_slice(&w.hess_rows);
            cols.copy_from_slice(&w.hess_cols);
        } else {
            let xs = std::slice::from_raw_parts(x, n as usize);
            let m = w.problem.num_constraints();
            let ls = if m > 0 { std::slice::from_raw_parts(lambda, m) } else { &[] };
            let vs = std::slice::from_raw_parts_mut(vals, w.hess_rows.len());
            w.problem.hessian_values(xs, obj_factor, ls, vs);
        }
        true
    }
}

extern "C" fn intermediate_cb(
    _: i32, iter: i32, _: f64, _: f64, _: f64, _: f64,
    _: f64, _: f64, _: f64, _: f64, _: i32, ud: *mut c_void,
) -> bool {
    unsafe { (*(ud as *mut IpoptWrapper)).iterations = iter; }
    true
}

fn set_str(p: IpoptProblemPtr, k: &str, v: &str) {
    let ks = CString::new(k).unwrap();
    let vs = CString::new(v).unwrap();
    unsafe { AddIpoptStrOption(p, ks.as_ptr(), vs.as_ptr()); }
}

fn set_num(p: IpoptProblemPtr, k: &str, v: f64) {
    let ks = CString::new(k).unwrap();
    unsafe { AddIpoptNumOption(p, ks.as_ptr(), v); }
}

fn set_int(p: IpoptProblemPtr, k: &str, v: i32) {
    let ks = CString::new(k).unwrap();
    unsafe { AddIpoptIntOption(p, ks.as_ptr(), v); }
}

struct SolveResult {
    status: String,
    objective: f64,
    iterations: i32,
    time_s: f64,
}

fn ipopt_status_str(s: i32) -> String {
    match s {
        0 => "Optimal".into(),
        1 => "Acceptable".into(),
        2 => "Infeasible".into(),
        -1 => "MaxIter".into(),
        -2 => "RestorationFailed".into(),
        other => format!("Status({})", other),
    }
}

fn solve_ipopt(problem: &dyn NlpProblem, tol: f64, max_iter: i32) -> SolveResult {
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

    let mut wrapper = IpoptWrapper {
        problem,
        jac_rows: jr.iter().map(|&r| r as i32).collect(),
        jac_cols: jc.iter().map(|&c| c as i32).collect(),
        hess_rows: hr.iter().map(|&r| r as i32).collect(),
        hess_cols: hc.iter().map(|&c| c as i32).collect(),
        iterations: 0,
    };

    unsafe {
        let ip = CreateIpoptProblem(
            n as i32, x_l.as_mut_ptr(), x_u.as_mut_ptr(),
            m as i32, g_l.as_mut_ptr(), g_u.as_mut_ptr(),
            wrapper.jac_rows.len() as i32, wrapper.hess_rows.len() as i32,
            0, eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb, eval_h_cb,
        );

        set_str(ip, "sb", "yes");
        set_str(ip, "mu_strategy", "adaptive");
        set_num(ip, "tol", tol);
        set_int(ip, "max_iter", max_iter);
        set_int(ip, "print_level", 0);
        SetIntermediateCallback(ip, intermediate_cb);

        let mut x = vec![0.0; n];
        problem.initial_point(&mut x);
        let mut g = vec![0.0; m.max(1)];
        let mut obj = 0.0;
        let mut mg = vec![0.0; m.max(1)];
        let mut ml = vec![0.0; n];
        let mut mu = vec![0.0; n];
        let ud = &mut wrapper as *mut IpoptWrapper as *mut c_void;

        let t0 = Instant::now();
        let status = IpoptSolve(ip, x.as_mut_ptr(), g.as_mut_ptr(), &mut obj,
            mg.as_mut_ptr(), ml.as_mut_ptr(), mu.as_mut_ptr(), ud);
        let elapsed = t0.elapsed().as_secs_f64();
        let iters = wrapper.iterations;
        FreeIpoptProblem(ip);

        SolveResult { status: ipopt_status_str(status), objective: obj, iterations: iters, time_s: elapsed }
    }
}

fn solve_ripopt<P: NlpProblem>(problem: &P, tol: f64, max_iter: usize) -> SolveResult {
    let options = SolverOptions {
        tol,
        max_iter,
        max_wall_time: 300.0,
        print_level: 0,
        ..SolverOptions::default()
    };
    let t0 = Instant::now();
    let result = ripopt::solve(problem, &options);
    let elapsed = t0.elapsed().as_secs_f64();
    SolveResult {
        status: format!("{:?}", result.status),
        objective: result.objective,
        iterations: result.iterations as i32,
        time_s: elapsed,
    }
}

// =========================================================================
// Problem definitions (same as large_scale.rs)
// =========================================================================

struct ChainedRosenbrock { n: usize }
impl NlpProblem for ChainedRosenbrock {
    fn num_variables(&self) -> usize { self.n }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }
    fn constraint_bounds(&self, _: &mut [f64], _: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { for v in x0.iter_mut() { *v = -1.2; } }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        (0..self.n-1).map(|i| { let a = 1.0 - x[i]; let b = x[i+1] - x[i]*x[i]; a*a + 100.0*b*b }).sum()
    }
    fn gradient(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        for v in g.iter_mut() { *v = 0.0; }
        for i in 0..self.n-1 {
            let r = x[i+1] - x[i]*x[i];
            g[i] += -2.0*(1.0 - x[i]) - 400.0*r*x[i];
            g[i+1] += 200.0*r;
        }
    }
    fn constraints(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n;
        let mut rows = Vec::with_capacity(2*n-1);
        let mut cols = Vec::with_capacity(2*n-1);
        rows.push(0); cols.push(0);
        for i in 1..n { rows.push(i); cols.push(i-1); rows.push(i); cols.push(i); }
        (rows, cols)
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, of: f64, _: &[f64], v: &mut [f64]) {
        for val in v.iter_mut() { *val = 0.0; }
        for i in 0..self.n-1 {
            let di = if i == 0 { 0 } else { 2*i };
            v[di] += of * (2.0 + 1200.0*x[i]*x[i] - 400.0*x[i+1]);
            v[2*(i+1)-1] += of * (-400.0*x[i]);
            v[2*(i+1)] += of * 200.0;
        }
    }
}

struct BratuProblem { n: usize, lambda_bratu: f64, h: f64 }
impl BratuProblem {
    fn new(n: usize) -> Self { let h = 1.0/(n as f64+1.0); Self { n, lambda_bratu: 1.0, h } }
}
impl NlpProblem for BratuProblem {
    fn num_variables(&self) -> usize { self.n }
    fn num_constraints(&self) -> usize { self.n - 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
        x_l[0] = 0.0; x_u[0] = 0.0;
        x_l[self.n-1] = 0.0; x_u[self.n-1] = 0.0;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for j in 0..self.n-2 { g_l[j] = 0.0; g_u[j] = 0.0; }
    }
    fn initial_point(&self, x0: &mut [f64]) { for v in x0.iter_mut() { *v = 0.0; } }
    fn objective(&self, _: &[f64], _new_x: bool) -> f64 { 0.0 }
    fn gradient(&self, _: &[f64], _new_x: bool, g: &mut [f64]) { for v in g.iter_mut() { *v = 0.0; } }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let h2 = self.h * self.h;
        for j in 0..self.n-2 {
            let i = j + 1;
            g[j] = (-x[i-1] + 2.0*x[i] - x[i+1])/h2 - self.lambda_bratu*x[i].exp();
        }
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let m = self.n - 2;
        let mut rows = Vec::with_capacity(3*m);
        let mut cols = Vec::with_capacity(3*m);
        for j in 0..m { let i = j+1; rows.push(j); cols.push(i-1); rows.push(j); cols.push(i); rows.push(j); cols.push(i+1); }
        (rows, cols)
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let h2 = self.h * self.h;
        for j in 0..self.n-2 {
            let i = j + 1;
            let b = 3*j;
            vals[b] = -1.0/h2;
            vals[b+1] = 2.0/h2 - self.lambda_bratu*x[i].exp();
            vals[b+2] = -1.0/h2;
        }
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::with_capacity(self.n);
        let mut cols = Vec::with_capacity(self.n);
        for k in 0..self.n { rows.push(k); cols.push(k); }
        (rows, cols)
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, _of: f64, lambda: &[f64], v: &mut [f64]) {
        for val in v.iter_mut() { *val = 0.0; }
        for j in 0..self.n-2 {
            let k = j + 1;
            v[k] += lambda[j] * (-self.lambda_bratu * x[k].exp());
        }
    }
}

struct OptimalControl { t: usize, h: f64, alpha: f64 }
impl OptimalControl {
    fn new(t: usize) -> Self { Self { t, h: 1.0/t as f64, alpha: 0.01 } }
    fn n(&self) -> usize { 2*self.t + 1 }
}
impl NlpProblem for OptimalControl {
    fn num_variables(&self) -> usize { self.n() }
    fn num_constraints(&self) -> usize { self.t + 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n() { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for j in 0..self.t+1 { g_l[j] = 0.0; g_u[j] = 0.0; }
    }
    fn initial_point(&self, x0: &mut [f64]) { for v in x0.iter_mut() { *v = 0.0; } }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let (h, t) = (self.h, self.t);
        let mut f = 0.0;
        for i in 0..=t { let dy = x[i] - 1.0; f += h*dy*dy; }
        for i in 0..t { f += self.alpha*h*x[t+1+i]*x[t+1+i]; }
        f
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let (h, t) = (self.h, self.t);
        for i in 0..=t { grad[i] = 2.0*h*(x[i] - 1.0); }
        for i in 0..t { grad[t+1+i] = 2.0*self.alpha*h*x[t+1+i]; }
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let (h, t) = (self.h, self.t);
        g[0] = x[0];
        for i in 0..t { g[i+1] = x[i+1] - (1.0-h)*x[i] - h*x[t+1+i]; }
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let t = self.t;
        let mut rows = Vec::with_capacity(1 + 3*t);
        let mut cols = Vec::with_capacity(1 + 3*t);
        rows.push(0); cols.push(0);
        for i in 0..t { rows.push(i+1); cols.push(i); rows.push(i+1); cols.push(i+1); rows.push(i+1); cols.push(t+1+i); }
        (rows, cols)
    }
    fn jacobian_values(&self, _: &[f64], _new_x: bool, vals: &mut [f64]) {
        let (h, t) = (self.h, self.t);
        vals[0] = 1.0;
        for i in 0..t { let b = 1+3*i; vals[b] = -(1.0-h); vals[b+1] = 1.0; vals[b+2] = -h; }
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n();
        let mut rows = Vec::with_capacity(n);
        let mut cols = Vec::with_capacity(n);
        for k in 0..n { rows.push(k); cols.push(k); }
        (rows, cols)
    }
    fn hessian_values(&self, _: &[f64], _new_x: bool, of: f64, _: &[f64], v: &mut [f64]) {
        let (h, t) = (self.h, self.t);
        for i in 0..=t { v[i] = of*2.0*h; }
        for i in 0..t { v[t+1+i] = of*2.0*self.alpha*h; }
    }
}

struct PoissonControl { k: usize, h: f64, alpha: f64 }
impl PoissonControl {
    fn new(k: usize) -> Self { Self { k, h: 1.0/(k as f64+1.0), alpha: 0.01 } }
    fn idx_u(&self, i: usize, j: usize) -> usize { i + j*self.k }
    fn idx_f(&self, i: usize, j: usize) -> usize { self.k*self.k + i + j*self.k }
    fn u_desired(&self, i: usize, j: usize) -> f64 {
        let x = (i as f64+1.0)*self.h;
        let y = (j as f64+1.0)*self.h;
        (PI*x).sin() * (PI*y).sin()
    }
}
impl NlpProblem for PoissonControl {
    fn num_variables(&self) -> usize { 2*self.k*self.k }
    fn num_constraints(&self) -> usize { self.k*self.k }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.num_variables() { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for j in 0..self.num_constraints() { g_l[j] = 0.0; g_u[j] = 0.0; }
    }
    fn initial_point(&self, x0: &mut [f64]) { for v in x0.iter_mut() { *v = 0.0; } }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let (k, h2) = (self.k, self.h*self.h);
        let mut f = 0.0;
        for j in 0..k { for i in 0..k {
            let du = x[self.idx_u(i,j)] - self.u_desired(i,j);
            f += 0.5*h2*du*du;
            let fi = x[self.idx_f(i,j)];
            f += 0.5*self.alpha*h2*fi*fi;
        }}
        f
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let (k, h2) = (self.k, self.h*self.h);
        for v in grad.iter_mut() { *v = 0.0; }
        for j in 0..k { for i in 0..k {
            grad[self.idx_u(i,j)] = h2*(x[self.idx_u(i,j)] - self.u_desired(i,j));
            grad[self.idx_f(i,j)] = self.alpha*h2*x[self.idx_f(i,j)];
        }}
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let (k, h2) = (self.k, self.h*self.h);
        for j in 0..k { for i in 0..k {
            let c = j*k + i;
            let mut lap = 4.0*x[self.idx_u(i,j)];
            if i > 0 { lap -= x[self.idx_u(i-1,j)]; }
            if i < k-1 { lap -= x[self.idx_u(i+1,j)]; }
            if j > 0 { lap -= x[self.idx_u(i,j-1)]; }
            if j < k-1 { lap -= x[self.idx_u(i,j+1)]; }
            g[c] = lap/h2 - x[self.idx_f(i,j)];
        }}
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let k = self.k;
        let mut rows = Vec::with_capacity(6*k*k);
        let mut cols = Vec::with_capacity(6*k*k);
        for j in 0..k { for i in 0..k {
            let c = j*k + i;
            rows.push(c); cols.push(self.idx_u(i,j));
            if i > 0 { rows.push(c); cols.push(self.idx_u(i-1,j)); }
            if i < k-1 { rows.push(c); cols.push(self.idx_u(i+1,j)); }
            if j > 0 { rows.push(c); cols.push(self.idx_u(i,j-1)); }
            if j < k-1 { rows.push(c); cols.push(self.idx_u(i,j+1)); }
            rows.push(c); cols.push(self.idx_f(i,j));
        }}
        (rows, cols)
    }
    fn jacobian_values(&self, _: &[f64], _new_x: bool, vals: &mut [f64]) {
        let (k, h2) = (self.k, self.h*self.h);
        let mut idx = 0;
        for j in 0..k { for i in 0..k {
            vals[idx] = 4.0/h2; idx += 1;
            if i > 0 { vals[idx] = -1.0/h2; idx += 1; }
            if i < k-1 { vals[idx] = -1.0/h2; idx += 1; }
            if j > 0 { vals[idx] = -1.0/h2; idx += 1; }
            if j < k-1 { vals[idx] = -1.0/h2; idx += 1; }
            vals[idx] = -1.0; idx += 1;
        }}
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.num_variables();
        let mut rows = Vec::with_capacity(n);
        let mut cols = Vec::with_capacity(n);
        for k in 0..n { rows.push(k); cols.push(k); }
        (rows, cols)
    }
    fn hessian_values(&self, _: &[f64], _new_x: bool, of: f64, _: &[f64], v: &mut [f64]) {
        let (k, h2) = (self.k, self.h*self.h);
        for j in 0..k { for i in 0..k {
            v[self.idx_u(i,j)] = of*h2;
            v[self.idx_f(i,j)] = of*self.alpha*h2;
        }}
    }
}

struct SparseQP { n: usize }
impl NlpProblem for SparseQP {
    fn num_variables(&self) -> usize { self.n }
    fn num_constraints(&self) -> usize { self.n }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n { x_l[i] = 0.0; x_u[i] = 10.0; }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for j in 0..self.n { g_l[j] = f64::NEG_INFINITY; g_u[j] = 2.5; }
    }
    fn initial_point(&self, x0: &mut [f64]) { for v in x0.iter_mut() { *v = 0.5; } }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let n = self.n;
        let mut f = 0.0;
        for i in 0..n { f += 2.0*x[i]*x[i]; if i < n-1 { f -= x[i]*x[i+1]; } f -= x[i]; }
        f
    }
    fn gradient(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            g[i] = 4.0*x[i] - 1.0;
            if i > 0 { g[i] -= x[i-1]; }
            if i < n-1 { g[i] -= x[i+1]; }
        }
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let n = self.n;
        for j in 0..n { g[j] = x[j] + x[(j+1)%n] + x[(j+2)%n]; }
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n;
        let mut rows = Vec::with_capacity(3*n);
        let mut cols = Vec::with_capacity(3*n);
        for j in 0..n { rows.push(j); cols.push(j); rows.push(j); cols.push((j+1)%n); rows.push(j); cols.push((j+2)%n); }
        (rows, cols)
    }
    fn jacobian_values(&self, _: &[f64], _new_x: bool, vals: &mut [f64]) {
        for j in 0..self.n { let b = 3*j; vals[b] = 1.0; vals[b+1] = 1.0; vals[b+2] = 1.0; }
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n;
        let mut rows = Vec::with_capacity(2*n-1);
        let mut cols = Vec::with_capacity(2*n-1);
        rows.push(0); cols.push(0);
        for i in 1..n { rows.push(i); cols.push(i-1); rows.push(i); cols.push(i); }
        (rows, cols)
    }
    fn hessian_values(&self, _: &[f64], _new_x: bool, of: f64, _: &[f64], v: &mut [f64]) {
        v[0] = of*4.0;
        for i in 1..self.n { v[2*i-1] = of*(-1.0); v[2*i] = of*4.0; }
    }
}

// =========================================================================
// Main: run each problem with both solvers, print comparison table
// =========================================================================

fn main() {
    let tol = 1e-6;
    let max_iter = 3000;

    println!("{:<20} {:>5} {:>5} | {:>10} {:>6} {:>8} | {:>10} {:>6} {:>8}",
        "Problem", "n", "m", "ipopt obj", "iters", "time(s)",
        "ripopt obj", "iters", "time(s)");
    println!("{}", "-".repeat(95));

    macro_rules! bench {
        ($name:expr, $n:expr, $m:expr, $problem:expr) => {{
            let p = $problem;
            let ip = solve_ipopt(&p, tol, max_iter as i32);
            let rp = solve_ripopt(&p, tol, max_iter);
            println!("{:<20} {:>5} {:>5} | {:>10.4e} {:>6} {:>8.3} | {:>10.4e} {:>6} {:>8.3}",
                $name, $n, $m,
                ip.objective, ip.iterations, ip.time_s,
                rp.objective, rp.iterations, rp.time_s);
            if ip.status != "Optimal" || rp.status != "Optimal" {
                println!("  status: ipopt={}, ripopt={}", ip.status, rp.status);
            }
        }};
    }

    bench!("Rosenbrock 500", 500, 0, ChainedRosenbrock { n: 500 });
    bench!("Bratu 1K", 1000, 998, BratuProblem::new(1000));
    bench!("OptControl 2.5K", 2499, 1250, OptimalControl::new(1249));
    bench!("Poisson 2.5K", 2450, 1225, PoissonControl::new(35));
    bench!("SparseQP 1K", 500, 500, SparseQP { n: 500 });
    println!("{}", "-".repeat(95));
    bench!("Rosenbrock 5K", 5000, 0, ChainedRosenbrock { n: 5000 });
    bench!("Bratu 10K", 10000, 9998, BratuProblem::new(10000));
    bench!("OptControl 20K", 19999, 10000, OptimalControl::new(9999));
    bench!("Poisson 50K", 49928, 24964, PoissonControl::new(158));
    bench!("SparseQP 100K", 50000, 50000, SparseQP { n: 50000 });
}
