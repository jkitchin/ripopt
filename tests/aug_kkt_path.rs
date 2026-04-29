//! A7 smoke tests: end-to-end solves through the augmented-KKT path
//! (the only path after A7.6). Mehrotra and SOC are wired in. Gondzio
//! MCC is retired (not in Ipopt 3.14).
//! The tests are deliberately small so they isolate the wiring path rather
//! than stress the algorithm.

use ripopt::{NlpProblem, SolveStatus, SolverOptions};

// Convex quadratic: min 0.5*((x1-1)^2 + (x2-2)^2). Optimum at (1, 2), f* = 0.
struct UnconstrainedQuadratic;
impl NlpProblem for UnconstrainedQuadratic {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.fill(f64::NEG_INFINITY);
        x_u.fill(f64::INFINITY);
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; x0[1] = 0.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.5 * ((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2));
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[0] - 1.0;
        grad[1] = x[1] - 2.0;
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 1]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor;
        vals[1] = obj_factor;
        true
    }
}

#[test]
fn aug_path_unconstrained_quadratic_finds_minimum() {
    let opts = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let r = ripopt::solve(&UnconstrainedQuadratic, &opts);
    assert_eq!(r.status, SolveStatus::Optimal, "status={:?}", r.status);
    assert!((r.x[0] - 1.0).abs() < 1e-6, "x1={}", r.x[0]);
    assert!((r.x[1] - 2.0).abs() < 1e-6, "x2={}", r.x[1]);
    assert!(r.objective.abs() < 1e-10);
}

// Bound-constrained quadratic: min 0.5*((x-3)^2) s.t. x in [0, 1].
// Optimum at x = 1 (upper bound active), f* = 2.
struct BoundConstrainedScalar;
impl NlpProblem for BoundConstrainedScalar {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) { x_l[0] = 0.0; x_u[0] = 1.0; }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.5; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.5 * (x[0] - 3.0).powi(2);
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[0] - 3.0;
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor;
        true
    }
}

#[test]
fn aug_path_bound_constrained_scalar_hits_upper_bound() {
    let opts = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let r = ripopt::solve(&BoundConstrainedScalar, &opts);
    assert_eq!(r.status, SolveStatus::Optimal, "status={:?}", r.status);
    assert!((r.x[0] - 1.0).abs() < 1e-6, "x={}", r.x[0]);
}

// Equality-constrained quadratic: min 0.5*(x1^2 + x2^2) s.t. x1 + x2 = 2.
// Lagrangian: x1 = x2 = 1, λ = 1, f* = 1.
struct EqualityConstrained;
impl NlpProblem for EqualityConstrained {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.fill(f64::NEG_INFINITY);
        x_u.fill(f64::INFINITY);
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) { g_l[0] = 2.0; g_u[0] = 2.0; }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; x0[1] = 0.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.5 * (x[0].powi(2) + x[1].powi(2));
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[0]; grad[1] = x[1];
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 0], vec![0, 1]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0; vals[1] = 1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 1]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor;
        vals[1] = obj_factor;
        true
    }
}

#[test]
fn aug_path_equality_constrained_quadratic() {
    let opts = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let r = ripopt::solve(&EqualityConstrained, &opts);
    assert_eq!(r.status, SolveStatus::Optimal, "status={:?}", r.status);
    assert!((r.x[0] - 1.0).abs() < 1e-6, "x1={}", r.x[0]);
    assert!((r.x[1] - 1.0).abs() < 1e-6, "x2={}", r.x[1]);
    assert!((r.objective - 1.0).abs() < 1e-10);
}

// Inequality-constrained quadratic: min 0.5*(x1^2 + x2^2) s.t. x1 + x2 >= 2.
// Optimum at x1 = x2 = 1, λ = 1, f* = 1.
struct InequalityConstrained;
impl NlpProblem for InequalityConstrained {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.fill(f64::NEG_INFINITY);
        x_u.fill(f64::INFINITY);
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 2.0; g_u[0] = f64::INFINITY;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 2.0; x0[1] = 2.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.5 * (x[0].powi(2) + x[1].powi(2));
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[0]; grad[1] = x[1];
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 0], vec![0, 1]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0; vals[1] = 1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 1]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor;
        vals[1] = obj_factor;
        true
    }
}

#[test]
fn aug_path_inequality_constrained_quadratic() {
    let opts = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let r = ripopt::solve(&InequalityConstrained, &opts);
    assert_eq!(r.status, SolveStatus::Optimal, "status={:?}", r.status);
    assert!((r.x[0] - 1.0).abs() < 1e-6, "x1={}", r.x[0]);
    assert!((r.x[1] - 1.0).abs() < 1e-6, "x2={}", r.x[1]);
    assert!((r.objective - 1.0).abs() < 1e-6);
}
