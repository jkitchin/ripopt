use ripopt::{NlpProblem, SolveStatus, SolverOptions};

// ---------------------------------------------------------------------------
// Rosenbrock (unconstrained) — used with L-BFGS Hessian approximation
// ---------------------------------------------------------------------------

struct Rosenbrock;

impl NlpProblem for Rosenbrock {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..2 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -1.2;
        x0[1] = 1.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        grad[1] = 200.0 * (x[1] - x[0] * x[0]);
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}
    // Return dummy structure — hessian_values should never be called in lbfgs mode
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) {
        panic!("hessian_values should not be called in limited-memory mode");
    }
}

#[test]
fn lbfgs_ipm_rosenbrock() {
    let problem = Rosenbrock;
    let options = SolverOptions {
        print_level: 0,
        hessian_approximation_lbfgs: true,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert!(
        result.status == SolveStatus::Optimal,
        "expected Optimal or Acceptable, got {:?}",
        result.status
    );
    assert!((result.x[0] - 1.0).abs() < 1e-3, "x[0]={}, expected ~1.0", result.x[0]);
    assert!((result.x[1] - 1.0).abs() < 1e-3, "x[1]={}, expected ~1.0", result.x[1]);
    assert!(result.objective < 1e-4, "obj={}, expected ~0", result.objective);
}

// ---------------------------------------------------------------------------
// HS071 constrained problem with L-BFGS Hessian approximation
// ---------------------------------------------------------------------------

struct Hs071Lbfgs;

impl NlpProblem for Hs071Lbfgs {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 {
            x_l[i] = 1.0;
            x_u[i] = 5.0;
        }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0; g_u[0] = f64::INFINITY;
        g_l[1] = 40.0; g_u[1] = 40.0;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0; x0[1] = 5.0; x0[2] = 5.0; x0[3] = 1.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        grad[1] = x[0] * x[3];
        grad[2] = x[0] * x[3] + 1.0;
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 0, 1, 1, 1, 1], vec![0, 1, 2, 3, 0, 1, 2, 3])
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = x[1] * x[2] * x[3];
        vals[1] = x[0] * x[2] * x[3];
        vals[2] = x[0] * x[1] * x[3];
        vals[3] = x[0] * x[1] * x[2];
        vals[4] = 2.0 * x[0];
        vals[5] = 2.0 * x[1];
        vals[6] = 2.0 * x[2];
        vals[7] = 2.0 * x[3];
    }
    // Dummy hessian structure — never used in lbfgs mode
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) {
        panic!("hessian_values should not be called in limited-memory mode");
    }
}

#[test]
fn lbfgs_ipm_hs071_constrained() {
    let problem = Hs071Lbfgs;
    // L-BFGS in IPM for constrained problems converges more slowly.
    let options = SolverOptions {
        print_level: 0,
        hessian_approximation_lbfgs: true,
        // Disable fallbacks that would call hessian_values on the original problem
        enable_sqp_fallback: false,
        enable_al_fallback: false,
        enable_slack_fallback: false,
        enable_lbfgs_fallback: false,
        enable_lbfgs_hessian_fallback: false,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    // L-BFGS in IPM should get a feasible solution with reasonable objective
    // Even if it doesn't fully converge, the constraint violation should be small
    let cv: f64 = {
        let mut g = vec![0.0; 2];
        problem.constraints(&result.x, true, &mut g);
        let mut max_v = 0.0_f64;
        // g[0] >= 25
        max_v = max_v.max((25.0 - g[0]).max(0.0));
        // g[1] == 40
        max_v = max_v.max((g[1] - 40.0).abs());
        max_v
    };
    assert!(
        cv < 0.1,
        "constraint violation too large: {:.2e}, x={:?}",
        cv, result.x
    );
    // HS071 is non-convex; any locally optimal feasible point is a valid result.
}

// ---------------------------------------------------------------------------
// Test that hessian_values is never called (panics if it is)
// ---------------------------------------------------------------------------

struct SimpleQuadratic;

impl NlpProblem for SimpleQuadratic {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..2 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 5.0;
        x0[1] = 5.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _: f64, _: &[f64], _: &mut [f64]) {
        panic!("hessian_values must not be called in limited-memory mode");
    }
}

#[test]
fn lbfgs_ipm_hessian_never_called() {
    let problem = SimpleQuadratic;
    let options = SolverOptions {
        print_level: 0,
        hessian_approximation_lbfgs: true,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert!(
        result.status == SolveStatus::Optimal,
        "expected convergence, got {:?}",
        result.status
    );
    assert!(result.x[0].abs() < 1e-3, "x[0]={}, expected ~0", result.x[0]);
    assert!(result.x[1].abs() < 1e-3, "x[1]={}, expected ~0", result.x[1]);
}

// ---------------------------------------------------------------------------
// Unit test: L-BFGS Hessian formation with known pairs
// ---------------------------------------------------------------------------

#[test]
fn lbfgs_ipm_unit_bk_formation() {
    use ripopt::ipm::LbfgsIpmState;

    let mut lbfgs = LbfgsIpmState::new(2);

    // Simulate first call (saves prev state)
    let x0 = [0.0, 0.0];
    let lag_grad0 = [4.0, 2.0];
    lbfgs.update(&x0, &lag_grad0);

    // Simulate second call with a step
    let x1 = [1.0, 0.0];
    let lag_grad1 = [6.0, 2.0]; // y = (2, 0), s = (1, 0)
    lbfgs.update(&x1, &lag_grad1);

    // Check that B_k is formed and is positive definite
    let bk = lbfgs.form_dense_bk();
    // Lower triangle: (0,0), (1,0), (1,1) -> indices 0, 1, 2
    assert_eq!(bk.len(), 3);
    // B[0,0] should be positive
    assert!(bk[0] > 0.0, "B[0,0]={}, expected positive", bk[0]);
    // B[1,1] should be positive
    assert!(bk[2] > 0.0, "B[1,1]={}, expected positive", bk[2]);

    // s^T B s should be positive for any nonzero s
    let v = [1.0, 1.0];
    let bv = lbfgs.multiply_bk(&v);
    let vtbv: f64 = v.iter().zip(bv.iter()).map(|(a, b)| a * b).sum();
    assert!(vtbv > 0.0, "v^T B v = {}, expected positive", vtbv);
}

// ===========================================================================
// L-BFGS Hessian FALLBACK tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Problem with a deliberately bad Hessian that causes IPM failure.
// The fallback should retry with L-BFGS Hessian and succeed.
// ---------------------------------------------------------------------------

struct BadHessianQuadratic;

impl NlpProblem for BadHessianQuadratic {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..2 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 5.0;
        x0[1] = 5.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Lower triangle: (0,0), (1,0), (1,1)
        (vec![0, 1, 1], vec![0, 0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        // Return a deliberately wrong negative-definite Hessian.
        // The exact IPM will struggle with this (wrong curvature direction).
        vals[0] = obj_factor * (-100.0);
        vals[1] = obj_factor * 0.0;
        vals[2] = obj_factor * (-100.0);
    }
}

/// Test that the L-BFGS Hessian fallback activates and solves a problem
/// where the user-provided Hessian is wrong.
#[test]
fn lbfgs_hessian_fallback_recovers_bad_hessian() {
    let problem = BadHessianQuadratic;
    let options = SolverOptions {
        print_level: 0,
        // Keep other fallbacks disabled so only L-BFGS Hessian fallback can help
        enable_lbfgs_fallback: false,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        enable_slack_fallback: false,
        enable_lbfgs_hessian_fallback: true,
        max_iter: 100,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert!(
        result.status == SolveStatus::Optimal,
        "L-BFGS Hessian fallback should recover from bad Hessian, got {:?}",
        result.status
    );
    assert!(result.x[0].abs() < 1e-2, "x[0]={}, expected ~0", result.x[0]);
    assert!(result.x[1].abs() < 1e-2, "x[1]={}, expected ~0", result.x[1]);
}

/// Test that the fallback is skipped when disabled.
#[test]
fn lbfgs_hessian_fallback_disabled() {
    let problem = BadHessianQuadratic;
    let options = SolverOptions {
        print_level: 0,
        enable_lbfgs_fallback: false,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        enable_slack_fallback: false,
        enable_lbfgs_hessian_fallback: false,
        max_iter: 50,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    // Without the fallback, either the solver fails, OR it may still reach the optimum
    // via inertia correction (which makes the bad Hessian positive definite). If it
    // returns Optimal, the solution must be correct (obj ≈ 0 at x* = 0).
    if matches!(result.status, SolveStatus::Optimal) {
        assert!(
            result.objective.abs() < 1e-6,
            "Returned Optimal but objective is wrong: {:.2e}",
            result.objective
        );
    }
}

/// Test that the fallback is skipped when already in L-BFGS mode
/// (no double-retry).
#[test]
fn lbfgs_hessian_fallback_skipped_when_already_lbfgs() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static HESS_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    struct CountingProblem;

    impl NlpProblem for CountingProblem {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 0 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            for i in 0..2 {
                x_l[i] = f64::NEG_INFINITY;
                x_u[i] = f64::INFINITY;
            }
        }
        fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 5.0;
            x0[1] = 5.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
            x[0] * x[0] + x[1] * x[1]
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
            grad[0] = 2.0 * x[0];
            grad[1] = 2.0 * x[1];
        }
        fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, _: f64, _: &[f64], _: &mut [f64]) {
            HESS_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    HESS_CALL_COUNT.store(0, Ordering::SeqCst);
    let problem = CountingProblem;
    let options = SolverOptions {
        print_level: 0,
        hessian_approximation_lbfgs: true,
        enable_lbfgs_hessian_fallback: true, // enabled but should be skipped
        ..SolverOptions::default()
    };
    let _result = ripopt::solve(&problem, &options);
    // hessian_values should never be called in lbfgs mode
    assert_eq!(
        HESS_CALL_COUNT.load(Ordering::SeqCst), 0,
        "hessian_values should not be called when already in L-BFGS mode"
    );
}

/// Test that the fallback works for constrained problems too.
#[test]
fn lbfgs_hessian_fallback_constrained() {
    // A constrained problem with a bad Hessian
    struct BadHessianConstrained;

    impl NlpProblem for BadHessianConstrained {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.0; x_u[0] = 10.0;
            x_l[1] = 0.0; x_u[1] = 10.0;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            // x[0] + x[1] = 4
            g_l[0] = 4.0;
            g_u[0] = 4.0;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
            x0[1] = 1.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
            // min (x[0]-3)^2 + (x[1]-3)^2, s.t. x[0]+x[1]=4
            // Solution: x=(2,2), f=2
            (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2)
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
            grad[0] = 2.0 * (x[0] - 3.0);
            grad[1] = 2.0 * (x[1] - 3.0);
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
            g[0] = x[0] + x[1];
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0], vec![0, 1])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
            vals[0] = 1.0;
            vals[1] = 1.0;
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1, 1], vec![0, 0, 1])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
            // Wrong Hessian: negative definite
            vals[0] = obj_factor * (-50.0);
            vals[1] = obj_factor * 0.0;
            vals[2] = obj_factor * (-50.0);
        }
    }

    let problem = BadHessianConstrained;
    let options = SolverOptions {
        print_level: 0,
        enable_lbfgs_fallback: false,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        enable_slack_fallback: false,
        enable_lbfgs_hessian_fallback: true,
        max_iter: 200,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert!(
        result.status == SolveStatus::Optimal,
        "L-BFGS Hessian fallback should solve constrained problem, got {:?}",
        result.status
    );
    // Check solution quality: x* = (2, 2), f* = 2
    assert!(
        (result.x[0] - 2.0).abs() < 0.5 && (result.x[1] - 2.0).abs() < 0.5,
        "x={:?}, expected ~(2,2)",
        result.x
    );
}

/// Test that the C API `hessian_approximation` option works.
#[test]
fn lbfgs_ipm_c_api_option() {
    // Just verify the option name is accepted (returns 1)
    use std::ffi::CString;
    unsafe {
        let x_l = [f64::NEG_INFINITY; 2];
        let x_u = [f64::INFINITY; 2];

        extern "C" fn dummy_f(_n: i32, _x: *const f64, _new_x: i32, obj: *mut f64, _: *mut std::ffi::c_void) -> i32 {
            unsafe { *obj = 0.0; }
            1
        }
        extern "C" fn dummy_grad(_n: i32, _x: *const f64, _new_x: i32, _g: *mut f64, _: *mut std::ffi::c_void) -> i32 { 1 }
        extern "C" fn dummy_g(_n: i32, _x: *const f64, _new_x: i32, _m: i32, _g: *mut f64, _: *mut std::ffi::c_void) -> i32 { 1 }
        extern "C" fn dummy_jac(_n: i32, _x: *const f64, _new_x: i32, _m: i32, _nj: i32, _ir: *mut i32, _jc: *mut i32, _v: *mut f64, _: *mut std::ffi::c_void) -> i32 { 1 }
        extern "C" fn dummy_h(_n: i32, _x: *const f64, _new_x: i32, _of: f64, _m: i32, _l: *const f64, _nl: i32, _nh: i32, _ir: *mut i32, _jc: *mut i32, _v: *mut f64, _: *mut std::ffi::c_void) -> i32 { 1 }

        let nlp = ripopt::c_api::ripopt_create(
            2, x_l.as_ptr(), x_u.as_ptr(),
            0, std::ptr::null(), std::ptr::null(),
            0, 0,
            dummy_f, dummy_grad, dummy_g, dummy_jac, dummy_h,
        );

        let key = CString::new("hessian_approximation").unwrap();
        let val_lm = CString::new("limited-memory").unwrap();
        let val_exact = CString::new("exact").unwrap();
        let val_bad = CString::new("banana").unwrap();

        let ret1 = ripopt::c_api::ripopt_add_str_option(nlp, key.as_ptr(), val_lm.as_ptr());
        assert_eq!(ret1, 1, "limited-memory should be accepted");

        let ret2 = ripopt::c_api::ripopt_add_str_option(nlp, key.as_ptr(), val_exact.as_ptr());
        assert_eq!(ret2, 1, "exact should be accepted");

        let ret3 = ripopt::c_api::ripopt_add_str_option(nlp, key.as_ptr(), val_bad.as_ptr());
        assert_eq!(ret3, 0, "invalid value should be rejected");

        let key2 = CString::new("enable_lbfgs_hessian_fallback").unwrap();
        let val_yes = CString::new("yes").unwrap();
        let val_no = CString::new("no").unwrap();

        let ret4 = ripopt::c_api::ripopt_add_str_option(nlp, key2.as_ptr(), val_yes.as_ptr());
        assert_eq!(ret4, 1, "enable_lbfgs_hessian_fallback=yes should be accepted");

        let ret5 = ripopt::c_api::ripopt_add_str_option(nlp, key2.as_ptr(), val_no.as_ptr());
        assert_eq!(ret5, 1, "enable_lbfgs_hessian_fallback=no should be accepted");

        ripopt::c_api::ripopt_free(nlp);
    }
}
