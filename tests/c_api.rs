//! Integration tests for the ripopt C API.
//!
//! These tests call the C API functions via the Rust module path, exercising
//! the same code paths that a C/C++ caller would use.

use std::ffi::CString;
use std::os::raw::{c_double, c_int, c_void};
use std::ptr;

use ripopt::c_api::*;

fn set_silent(nlp: *mut CApiProblem) {
    let key = CString::new("print_level").unwrap();
    unsafe { ripopt_add_int_option(nlp, key.as_ptr(), 0); }
}

// ============================================================================
// HS071: 4 vars, 2 constraints (inequality + equality), variable bounds
// ============================================================================

unsafe extern "C" fn hs071_eval_f(_n: c_int, x: *const c_double, _new_x: c_int, obj: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 4);
    *obj = x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2];
    1
}

unsafe extern "C" fn hs071_eval_grad_f(_n: c_int, x: *const c_double, _new_x: c_int, grad: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 4);
    let g = std::slice::from_raw_parts_mut(grad, 4);
    g[0] = x[3]*(x[0]+x[1]+x[2]) + x[0]*x[3];
    g[1] = x[0]*x[3];
    g[2] = x[0]*x[3] + 1.0;
    g[3] = x[0]*(x[0]+x[1]+x[2]);
    1
}

unsafe extern "C" fn hs071_eval_g(_n: c_int, x: *const c_double, _new_x: c_int, _m: c_int, g: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 4);
    let gv = std::slice::from_raw_parts_mut(g, 2);
    gv[0] = x[0]*x[1]*x[2]*x[3];
    gv[1] = x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3];
    1
}

unsafe extern "C" fn hs071_eval_jac_g(_n: c_int, x: *const c_double, _new_x: c_int, _m: c_int, _nele: c_int, ir: *mut c_int, jc: *mut c_int, vals: *mut c_double, _ud: *mut c_void) -> c_int {
    if vals.is_null() {
        let ir = std::slice::from_raw_parts_mut(ir, 8);
        let jc = std::slice::from_raw_parts_mut(jc, 8);
        ir[0]=0; jc[0]=0; ir[1]=0; jc[1]=1; ir[2]=0; jc[2]=2; ir[3]=0; jc[3]=3;
        ir[4]=1; jc[4]=0; ir[5]=1; jc[5]=1; ir[6]=1; jc[6]=2; ir[7]=1; jc[7]=3;
    } else {
        let x = std::slice::from_raw_parts(x, 4);
        let v = std::slice::from_raw_parts_mut(vals, 8);
        v[0]=x[1]*x[2]*x[3]; v[1]=x[0]*x[2]*x[3]; v[2]=x[0]*x[1]*x[3]; v[3]=x[0]*x[1]*x[2];
        v[4]=2.0*x[0]; v[5]=2.0*x[1]; v[6]=2.0*x[2]; v[7]=2.0*x[3];
    }
    1
}

unsafe extern "C" fn hs071_eval_h(_n: c_int, x: *const c_double, _new_x: c_int, obj_f: c_double, _m: c_int, lam: *const c_double, _new_lam: c_int, _nele: c_int, ir: *mut c_int, jc: *mut c_int, vals: *mut c_double, _ud: *mut c_void) -> c_int {
    if vals.is_null() {
        let ir = std::slice::from_raw_parts_mut(ir, 10);
        let jc = std::slice::from_raw_parts_mut(jc, 10);
        ir[0]=0; jc[0]=0;
        ir[1]=1; jc[1]=0; ir[2]=1; jc[2]=1;
        ir[3]=2; jc[3]=0; ir[4]=2; jc[4]=1; ir[5]=2; jc[5]=2;
        ir[6]=3; jc[6]=0; ir[7]=3; jc[7]=1; ir[8]=3; jc[8]=2; ir[9]=3; jc[9]=3;
    } else {
        let x = std::slice::from_raw_parts(x, 4);
        let lam = std::slice::from_raw_parts(lam, 2);
        let v = std::slice::from_raw_parts_mut(vals, 10);
        v[0] = obj_f * 2.0*x[3];
        v[1] = obj_f * x[3];
        v[2] = 0.0;
        v[3] = obj_f * x[3];
        v[4] = 0.0;
        v[5] = 0.0;
        v[6] = obj_f * (2.0*x[0]+x[1]+x[2]);
        v[7] = obj_f * x[0];
        v[8] = obj_f * x[0];
        v[9] = 0.0;
        v[1] += lam[0]*x[2]*x[3];
        v[3] += lam[0]*x[1]*x[3];
        v[4] += lam[0]*x[0]*x[3];
        v[6] += lam[0]*x[1]*x[2];
        v[7] += lam[0]*x[0]*x[2];
        v[8] += lam[0]*x[0]*x[1];
        v[0] += lam[1]*2.0;
        v[2] += lam[1]*2.0;
        v[5] += lam[1]*2.0;
        v[9] += lam[1]*2.0;
    }
    1
}

#[test]
fn c_api_hs071_basic() {
    let x_l = [1.0, 1.0, 1.0, 1.0];
    let x_u = [5.0, 5.0, 5.0, 5.0];
    let g_l = [25.0, 40.0];
    let g_u = [1e30, 40.0];

    unsafe {
        let nlp = ripopt_create(
            4, x_l.as_ptr(), x_u.as_ptr(),
            2, g_l.as_ptr(), g_u.as_ptr(),
            8, 10, 0,
            hs071_eval_f, hs071_eval_grad_f, hs071_eval_g, hs071_eval_jac_g, hs071_eval_h,
        );
        assert!(!nlp.is_null());
        set_silent(nlp);

        let mut x = [1.0, 5.0, 5.0, 1.0];
        let mut obj = 0.0;
        let status = ripopt_solve(nlp, x.as_mut_ptr(), ptr::null_mut(), &mut obj,
                                   ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        assert_eq!(status, 0, "Expected RIPOPT_SOLVE_SUCCEEDED");
        assert!((obj - 17.0140173).abs() < 1e-4, "Objective {obj} != 17.014");

        ripopt_free(nlp);
    }
}

#[test]
fn c_api_hs071_multiplier_extraction() {
    let x_l = [1.0, 1.0, 1.0, 1.0];
    let x_u = [5.0, 5.0, 5.0, 5.0];
    let g_l = [25.0, 40.0];
    let g_u = [1e30, 40.0];

    unsafe {
        let nlp = ripopt_create(
            4, x_l.as_ptr(), x_u.as_ptr(),
            2, g_l.as_ptr(), g_u.as_ptr(),
            8, 10, 0,
            hs071_eval_f, hs071_eval_grad_f, hs071_eval_g, hs071_eval_jac_g, hs071_eval_h,
        );
        set_silent(nlp);

        let mut x = [1.0, 5.0, 5.0, 1.0];
        let mut obj = 0.0;
        let mut g = [0.0; 2];
        let mut mult_g = [0.0; 2];
        let mut mult_xl = [0.0; 4];
        let mut mult_xu = [0.0; 4];

        let status = ripopt_solve(nlp, x.as_mut_ptr(), g.as_mut_ptr(), &mut obj,
                                   mult_g.as_mut_ptr(), mult_xl.as_mut_ptr(), mult_xu.as_mut_ptr(),
                                   ptr::null_mut());
        assert_eq!(status, 0);

        // Equality constraint g[1] = sum(xi^2) == 40
        assert!((g[1] - 40.0).abs() < 1e-4, "Equality constraint g[1]={} != 40", g[1]);

        // Multipliers should be nonzero for active constraints
        assert!(mult_g[1].abs() > 1e-10, "Equality multiplier should be nonzero: {}", mult_g[1]);

        // x[0] is at lower bound (1.0), so mult_xl[0] should be positive
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0] should be at lower bound");
        assert!(mult_xl[0] > 1e-6, "Lower bound multiplier for x[0] should be positive: {}", mult_xl[0]);

        ripopt_free(nlp);
    }
}

#[test]
fn c_api_null_output_params() {
    let x_l = [1.0, 1.0, 1.0, 1.0];
    let x_u = [5.0, 5.0, 5.0, 5.0];
    let g_l = [25.0, 40.0];
    let g_u = [1e30, 40.0];

    unsafe {
        let nlp = ripopt_create(
            4, x_l.as_ptr(), x_u.as_ptr(),
            2, g_l.as_ptr(), g_u.as_ptr(),
            8, 10, 0,
            hs071_eval_f, hs071_eval_grad_f, hs071_eval_g, hs071_eval_jac_g, hs071_eval_h,
        );
        set_silent(nlp);

        let mut x = [1.0, 5.0, 5.0, 1.0];
        let status = ripopt_solve(nlp, x.as_mut_ptr(),
                                   ptr::null_mut(), ptr::null_mut(),
                                   ptr::null_mut(), ptr::null_mut(), ptr::null_mut(),
                                   ptr::null_mut());
        assert_eq!(status, 0);
        assert!((x[0] - 1.0).abs() < 0.1);

        ripopt_free(nlp);
    }
}

// ============================================================================
// Rosenbrock: unconstrained, 2 variables
// ============================================================================

unsafe extern "C" fn rosen_eval_f(_n: c_int, x: *const c_double, _new_x: c_int, obj: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 2);
    *obj = 100.0*(x[1] - x[0]*x[0]).powi(2) + (1.0 - x[0]).powi(2);
    1
}

unsafe extern "C" fn rosen_eval_grad_f(_n: c_int, x: *const c_double, _new_x: c_int, grad: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 2);
    let g = std::slice::from_raw_parts_mut(grad, 2);
    g[0] = -400.0*x[0]*(x[1] - x[0]*x[0]) - 2.0*(1.0 - x[0]);
    g[1] = 200.0*(x[1] - x[0]*x[0]);
    1
}

unsafe extern "C" fn empty_eval_g(_n: c_int, _x: *const c_double, _new_x: c_int, _m: c_int, _g: *mut c_double, _ud: *mut c_void) -> c_int { 1 }
unsafe extern "C" fn empty_eval_jac_g(_n: c_int, _x: *const c_double, _new_x: c_int, _m: c_int, _nele: c_int, _ir: *mut c_int, _jc: *mut c_int, _v: *mut c_double, _ud: *mut c_void) -> c_int { 1 }

unsafe extern "C" fn rosen_eval_h(_n: c_int, x: *const c_double, _new_x: c_int, obj_f: c_double, _m: c_int, _lam: *const c_double, _new_lam: c_int, _nele: c_int, ir: *mut c_int, jc: *mut c_int, vals: *mut c_double, _ud: *mut c_void) -> c_int {
    if vals.is_null() {
        let ir = std::slice::from_raw_parts_mut(ir, 3);
        let jc = std::slice::from_raw_parts_mut(jc, 3);
        ir[0]=0; jc[0]=0;
        ir[1]=1; jc[1]=0;
        ir[2]=1; jc[2]=1;
    } else {
        let x = std::slice::from_raw_parts(x, 2);
        let v = std::slice::from_raw_parts_mut(vals, 3);
        v[0] = obj_f * (-400.0*(x[1] - 3.0*x[0]*x[0]) + 2.0);
        v[1] = obj_f * (-400.0*x[0]);
        v[2] = obj_f * 200.0;
    }
    1
}

#[test]
fn c_api_rosenbrock_unconstrained() {
    let x_l = [f64::NEG_INFINITY; 2];
    let x_u = [f64::INFINITY; 2];

    unsafe {
        let nlp = ripopt_create(
            2, x_l.as_ptr(), x_u.as_ptr(),
            0, ptr::null(), ptr::null(),
            0, 3, 0,
            rosen_eval_f, rosen_eval_grad_f, empty_eval_g, empty_eval_jac_g, rosen_eval_h,
        );
        assert!(!nlp.is_null());
        set_silent(nlp);

        let mut x = [-1.0, 1.0];
        let mut obj = 0.0;
        let status = ripopt_solve(nlp, x.as_mut_ptr(), ptr::null_mut(), &mut obj,
                                   ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        assert!(status == 0 || status == 1, "Status {status} not optimal/acceptable");
        assert!(obj < 1e-6, "Objective {obj} should be ~0");
        assert!((x[0] - 1.0).abs() < 1e-3, "x[0]={} should be ~1", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-3, "x[1]={} should be ~1", x[1]);

        ripopt_free(nlp);
    }
}

// ============================================================================
// Bound-constrained quadratic: min (x-2)^2 + (y-3)^2 s.t. 0 <= x <= 1.5
// Solution: x=1.5, y=3.0, obj=0.25
// ============================================================================

unsafe extern "C" fn bqp_eval_f(_n: c_int, x: *const c_double, _new_x: c_int, obj: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 2);
    *obj = (x[0]-2.0).powi(2) + (x[1]-3.0).powi(2);
    1
}

unsafe extern "C" fn bqp_eval_grad_f(_n: c_int, x: *const c_double, _new_x: c_int, grad: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 2);
    let g = std::slice::from_raw_parts_mut(grad, 2);
    g[0] = 2.0*(x[0]-2.0);
    g[1] = 2.0*(x[1]-3.0);
    1
}

unsafe extern "C" fn bqp_eval_h(_n: c_int, _x: *const c_double, _new_x: c_int, obj_f: c_double, _m: c_int, _lam: *const c_double, _new_lam: c_int, _nele: c_int, ir: *mut c_int, jc: *mut c_int, vals: *mut c_double, _ud: *mut c_void) -> c_int {
    if vals.is_null() {
        let ir = std::slice::from_raw_parts_mut(ir, 2);
        let jc = std::slice::from_raw_parts_mut(jc, 2);
        ir[0]=0; jc[0]=0;
        ir[1]=1; jc[1]=1;
    } else {
        let v = std::slice::from_raw_parts_mut(vals, 2);
        v[0] = obj_f * 2.0;
        v[1] = obj_f * 2.0;
    }
    1
}

#[test]
fn c_api_bound_constrained_qp() {
    let x_l = [0.0, f64::NEG_INFINITY];
    let x_u = [1.5, f64::INFINITY];

    unsafe {
        let nlp = ripopt_create(
            2, x_l.as_ptr(), x_u.as_ptr(),
            0, ptr::null(), ptr::null(),
            0, 2, 0,
            bqp_eval_f, bqp_eval_grad_f, empty_eval_g, empty_eval_jac_g, bqp_eval_h,
        );
        set_silent(nlp);

        let mut x = [0.5, 0.5];
        let mut obj = 0.0;
        let status = ripopt_solve(nlp, x.as_mut_ptr(), ptr::null_mut(), &mut obj,
                                   ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        assert!(status == 0 || status == 1, "Status {status}");
        assert!((x[0] - 1.5).abs() < 1e-3, "x[0]={} should be 1.5", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-3, "x[1]={} should be 3.0", x[1]);
        assert!((obj - 0.25).abs() < 1e-4, "obj={} should be 0.25", obj);

        ripopt_free(nlp);
    }
}

// ============================================================================
// Equality-only: min x^2 + y^2 s.t. x + y = 1
// Solution: x=0.5, y=0.5, obj=0.5
// ============================================================================

unsafe extern "C" fn eq_eval_f(_n: c_int, x: *const c_double, _new_x: c_int, obj: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 2);
    *obj = x[0]*x[0] + x[1]*x[1];
    1
}

unsafe extern "C" fn eq_eval_grad_f(_n: c_int, x: *const c_double, _new_x: c_int, grad: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 2);
    let g = std::slice::from_raw_parts_mut(grad, 2);
    g[0] = 2.0*x[0];
    g[1] = 2.0*x[1];
    1
}

unsafe extern "C" fn eq_eval_g(_n: c_int, x: *const c_double, _new_x: c_int, _m: c_int, g: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 2);
    *g = x[0] + x[1];
    1
}

unsafe extern "C" fn eq_eval_jac_g(_n: c_int, _x: *const c_double, _new_x: c_int, _m: c_int, _nele: c_int, ir: *mut c_int, jc: *mut c_int, vals: *mut c_double, _ud: *mut c_void) -> c_int {
    if vals.is_null() {
        let ir = std::slice::from_raw_parts_mut(ir, 2);
        let jc = std::slice::from_raw_parts_mut(jc, 2);
        ir[0]=0; jc[0]=0;
        ir[1]=0; jc[1]=1;
    } else {
        let v = std::slice::from_raw_parts_mut(vals, 2);
        v[0] = 1.0;
        v[1] = 1.0;
    }
    1
}

unsafe extern "C" fn eq_eval_h(_n: c_int, _x: *const c_double, _new_x: c_int, obj_f: c_double, _m: c_int, _lam: *const c_double, _new_lam: c_int, _nele: c_int, ir: *mut c_int, jc: *mut c_int, vals: *mut c_double, _ud: *mut c_void) -> c_int {
    if vals.is_null() {
        let ir = std::slice::from_raw_parts_mut(ir, 2);
        let jc = std::slice::from_raw_parts_mut(jc, 2);
        ir[0]=0; jc[0]=0;
        ir[1]=1; jc[1]=1;
    } else {
        let v = std::slice::from_raw_parts_mut(vals, 2);
        v[0] = obj_f * 2.0;
        v[1] = obj_f * 2.0;
    }
    1
}

#[test]
fn c_api_equality_constrained() {
    let x_l = [-1e30; 2];
    let x_u = [1e30; 2];
    let g_l = [1.0];
    let g_u = [1.0];

    unsafe {
        let nlp = ripopt_create(
            2, x_l.as_ptr(), x_u.as_ptr(),
            1, g_l.as_ptr(), g_u.as_ptr(),
            2, 2, 0,
            eq_eval_f, eq_eval_grad_f, eq_eval_g, eq_eval_jac_g, eq_eval_h,
        );
        set_silent(nlp);

        let mut x = [0.0, 0.0];
        let mut obj = 0.0;
        let mut g = [0.0];
        let mut mult_g = [0.0];
        let status = ripopt_solve(nlp, x.as_mut_ptr(), g.as_mut_ptr(), &mut obj,
                                   mult_g.as_mut_ptr(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        assert!(status == 0 || status == 1, "Status {status}");
        assert!((x[0] - 0.5).abs() < 1e-3, "x[0]={} should be 0.5", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-3, "x[1]={} should be 0.5", x[1]);
        assert!((obj - 0.5).abs() < 1e-4, "obj={} should be 0.5", obj);
        assert!((g[0] - 1.0).abs() < 1e-6, "constraint g={} should be 1", g[0]);

        ripopt_free(nlp);
    }
}

// ============================================================================
// Option setting: verify known/unknown options return correct codes
// ============================================================================

#[test]
fn c_api_option_known_returns_1() {
    let x_l = [0.0];
    let x_u = [1.0];

    unsafe {
        let nlp = ripopt_create(
            1, x_l.as_ptr(), x_u.as_ptr(),
            0, ptr::null(), ptr::null(),
            0, 1, 0,
            rosen_eval_f, rosen_eval_grad_f, empty_eval_g, empty_eval_jac_g, rosen_eval_h,
        );

        let key = CString::new("tol").unwrap();
        assert_eq!(ripopt_add_num_option(nlp, key.as_ptr(), 1e-6), 1);

        let key = CString::new("max_wall_time").unwrap();
        assert_eq!(ripopt_add_num_option(nlp, key.as_ptr(), 30.0), 1);

        let key = CString::new("max_iter").unwrap();
        assert_eq!(ripopt_add_int_option(nlp, key.as_ptr(), 500), 1);

        let key = CString::new("print_level").unwrap();
        assert_eq!(ripopt_add_int_option(nlp, key.as_ptr(), 0), 1);

        let key = CString::new("mu_strategy").unwrap();
        let val = CString::new("adaptive").unwrap();
        assert_eq!(ripopt_add_str_option(nlp, key.as_ptr(), val.as_ptr()), 1);

        let key = CString::new("warm_start_init_point").unwrap();
        let val = CString::new("yes").unwrap();
        assert_eq!(ripopt_add_str_option(nlp, key.as_ptr(), val.as_ptr()), 1);

        ripopt_free(nlp);
    }
}

#[test]
fn c_api_option_unknown_returns_0() {
    let x_l = [0.0];
    let x_u = [1.0];

    unsafe {
        let nlp = ripopt_create(
            1, x_l.as_ptr(), x_u.as_ptr(),
            0, ptr::null(), ptr::null(),
            0, 1, 0,
            rosen_eval_f, rosen_eval_grad_f, empty_eval_g, empty_eval_jac_g, rosen_eval_h,
        );

        let key = CString::new("nonexistent_option").unwrap();
        assert_eq!(ripopt_add_num_option(nlp, key.as_ptr(), 1.0), 0);
        assert_eq!(ripopt_add_int_option(nlp, key.as_ptr(), 1), 0);

        let val = CString::new("whatever").unwrap();
        assert_eq!(ripopt_add_str_option(nlp, key.as_ptr(), val.as_ptr()), 0);

        ripopt_free(nlp);
    }
}

// ============================================================================
// user_data passthrough: verify the pointer reaches callbacks
// ============================================================================

unsafe extern "C" fn ud_eval_f(_n: c_int, x: *const c_double, _new_x: c_int, obj: *mut c_double, ud: *mut c_void) -> c_int {
    let offset = *(ud as *const f64);
    let x = std::slice::from_raw_parts(x, 2);
    *obj = (x[0] - offset).powi(2) + (x[1] - offset).powi(2);
    1
}

unsafe extern "C" fn ud_eval_grad_f(_n: c_int, x: *const c_double, _new_x: c_int, grad: *mut c_double, ud: *mut c_void) -> c_int {
    let offset = *(ud as *const f64);
    let x = std::slice::from_raw_parts(x, 2);
    let g = std::slice::from_raw_parts_mut(grad, 2);
    g[0] = 2.0*(x[0] - offset);
    g[1] = 2.0*(x[1] - offset);
    1
}

unsafe extern "C" fn ud_eval_h(_n: c_int, _x: *const c_double, _new_x: c_int, obj_f: c_double, _m: c_int, _lam: *const c_double, _new_lam: c_int, _nele: c_int, ir: *mut c_int, jc: *mut c_int, vals: *mut c_double, _ud: *mut c_void) -> c_int {
    if vals.is_null() {
        let ir = std::slice::from_raw_parts_mut(ir, 2);
        let jc = std::slice::from_raw_parts_mut(jc, 2);
        ir[0]=0; jc[0]=0; ir[1]=1; jc[1]=1;
    } else {
        let v = std::slice::from_raw_parts_mut(vals, 2);
        v[0] = obj_f * 2.0;
        v[1] = obj_f * 2.0;
    }
    1
}

#[test]
fn c_api_user_data_passthrough() {
    let x_l = [-1e30; 2];
    let x_u = [1e30; 2];
    let mut target = 5.0_f64;

    unsafe {
        let nlp = ripopt_create(
            2, x_l.as_ptr(), x_u.as_ptr(),
            0, ptr::null(), ptr::null(),
            0, 2, 0,
            ud_eval_f, ud_eval_grad_f, empty_eval_g, empty_eval_jac_g, ud_eval_h,
        );
        set_silent(nlp);

        let mut x = [0.0, 0.0];
        let mut obj = 0.0;
        let status = ripopt_solve(nlp, x.as_mut_ptr(), ptr::null_mut(), &mut obj,
                                   ptr::null_mut(), ptr::null_mut(), ptr::null_mut(),
                                   &mut target as *mut f64 as *mut c_void);
        assert!(status == 0 || status == 1);
        assert!((x[0] - 5.0).abs() < 1e-2, "x[0]={} should be 5.0 (target from user_data)", x[0]);
        assert!((x[1] - 5.0).abs() < 1e-2, "x[1]={} should be 5.0", x[1]);

        ripopt_free(nlp);
    }
}

// ============================================================================
// HS035: bound-constrained QP with one inequality constraint
// min 9 - 8x1 - 6x2 - 4x3 + 2x1^2 + 2x2^2 + x3^2 + 2x1x2 + 2x1x3
// s.t. x1 + x2 + 2x3 <= 3,  x1,x2,x3 >= 0
// Solution: x*=(4/3, 7/9, 4/9), f*=1/9
// ============================================================================

unsafe extern "C" fn hs035_eval_f(_n: c_int, x: *const c_double, _new_x: c_int, obj: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 3);
    *obj = 9.0 - 8.0*x[0] - 6.0*x[1] - 4.0*x[2]
        + 2.0*x[0]*x[0] + 2.0*x[1]*x[1] + x[2]*x[2]
        + 2.0*x[0]*x[1] + 2.0*x[0]*x[2];
    1
}

unsafe extern "C" fn hs035_eval_grad_f(_n: c_int, x: *const c_double, _new_x: c_int, grad: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 3);
    let g = std::slice::from_raw_parts_mut(grad, 3);
    g[0] = -8.0 + 4.0*x[0] + 2.0*x[1] + 2.0*x[2];
    g[1] = -6.0 + 2.0*x[0] + 4.0*x[1];
    g[2] = -4.0 + 2.0*x[0] + 2.0*x[2];
    1
}

unsafe extern "C" fn hs035_eval_g(_n: c_int, x: *const c_double, _new_x: c_int, _m: c_int, g: *mut c_double, _ud: *mut c_void) -> c_int {
    let x = std::slice::from_raw_parts(x, 3);
    *g = x[0] + x[1] + 2.0*x[2];
    1
}

unsafe extern "C" fn hs035_eval_jac_g(_n: c_int, _x: *const c_double, _new_x: c_int, _m: c_int, _nele: c_int, ir: *mut c_int, jc: *mut c_int, vals: *mut c_double, _ud: *mut c_void) -> c_int {
    if vals.is_null() {
        let ir = std::slice::from_raw_parts_mut(ir, 3);
        let jc = std::slice::from_raw_parts_mut(jc, 3);
        ir[0]=0; jc[0]=0; ir[1]=0; jc[1]=1; ir[2]=0; jc[2]=2;
    } else {
        let v = std::slice::from_raw_parts_mut(vals, 3);
        v[0]=1.0; v[1]=1.0; v[2]=2.0;
    }
    1
}

unsafe extern "C" fn hs035_eval_h(_n: c_int, _x: *const c_double, _new_x: c_int, obj_f: c_double, _m: c_int, _lam: *const c_double, _new_lam: c_int, _nele: c_int, ir: *mut c_int, jc: *mut c_int, vals: *mut c_double, _ud: *mut c_void) -> c_int {
    if vals.is_null() {
        // H = [[4,2,2],[2,4,0],[2,0,2]], lower triangle: (0,0),(1,0),(1,1),(2,0),(2,2)
        let ir = std::slice::from_raw_parts_mut(ir, 5);
        let jc = std::slice::from_raw_parts_mut(jc, 5);
        ir[0]=0; jc[0]=0;
        ir[1]=1; jc[1]=0;
        ir[2]=1; jc[2]=1;
        ir[3]=2; jc[3]=0;
        ir[4]=2; jc[4]=2;
    } else {
        let v = std::slice::from_raw_parts_mut(vals, 5);
        v[0] = obj_f * 4.0;  // (0,0)
        v[1] = obj_f * 2.0;  // (1,0)
        v[2] = obj_f * 4.0;  // (1,1)
        v[3] = obj_f * 2.0;  // (2,0)
        v[4] = obj_f * 2.0;  // (2,2)
    }
    1
}

#[test]
fn c_api_hs035_inequality() {
    let x_l = [0.0, 0.0, 0.0];
    let x_u = [f64::INFINITY, f64::INFINITY, f64::INFINITY];
    let g_l = [f64::NEG_INFINITY];
    let g_u = [3.0];

    unsafe {
        let nlp = ripopt_create(
            3, x_l.as_ptr(), x_u.as_ptr(),
            1, g_l.as_ptr(), g_u.as_ptr(),
            3, 5, 0,
            hs035_eval_f, hs035_eval_grad_f, hs035_eval_g, hs035_eval_jac_g, hs035_eval_h,
        );
        set_silent(nlp);

        let mut x = [0.5, 0.5, 0.5];
        let mut obj = 0.0;
        let status = ripopt_solve(nlp, x.as_mut_ptr(), ptr::null_mut(), &mut obj,
                                   ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        assert!(status == 0 || status == 1, "Status {status}");
        let expected_obj = 1.0/9.0;
        assert!((obj - expected_obj).abs() < 1e-3, "obj={obj} should be {expected_obj}");

        ripopt_free(nlp);
    }
}

// ============================================================================
// Test: options actually affect solver behavior
// ============================================================================

#[test]
fn c_api_max_iter_option() {
    let x_l = [f64::NEG_INFINITY; 2];
    let x_u = [f64::INFINITY; 2];

    unsafe {
        let nlp = ripopt_create(
            2, x_l.as_ptr(), x_u.as_ptr(),
            0, ptr::null(), ptr::null(),
            0, 3, 0,
            rosen_eval_f, rosen_eval_grad_f, empty_eval_g, empty_eval_jac_g, rosen_eval_h,
        );
        set_silent(nlp);

        let key = CString::new("max_iter").unwrap();
        ripopt_add_int_option(nlp, key.as_ptr(), 2);

        let key = CString::new("enable_lbfgs_fallback").unwrap();
        let val = CString::new("no").unwrap();
        ripopt_add_str_option(nlp, key.as_ptr(), val.as_ptr());

        let key = CString::new("enable_lbfgs_hessian_fallback").unwrap();
        let val = CString::new("no").unwrap();
        ripopt_add_str_option(nlp, key.as_ptr(), val.as_ptr());

        let mut x = [-1.0, 1.0];
        let status = ripopt_solve(nlp, x.as_mut_ptr(), ptr::null_mut(), ptr::null_mut(),
                                   ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        assert_eq!(status, -1, "Expected RIPOPT_MAXITER_EXCEEDED (-1), got {status}");

        ripopt_free(nlp);
    }
}

// ============================================================================
// Test: create/free lifecycle — NULL safety
// ============================================================================

#[test]
fn c_api_free_null_is_safe() {
    unsafe {
        ripopt_free(ptr::null_mut());
    }
}
